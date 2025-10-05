# app.py

# các thư viện cần thiết
import os
import sys
import cv2
import torch
import numpy as np
import time
import pickle
import json
from collections import defaultdict
from flask import Flask, render_template, jsonify, request, Response, session, send_from_directory, make_response
from flask_socketio import SocketIO, emit
from PIL import Image, ImageOps
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity
import base64
from io import BytesIO
from threading import Thread, Lock
import queue
import uuid
import traceback
import timm

# Khởi tạo Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'mat-khau-bi-mat-cua-ban'
# Khởi tạo SocketIO để giao tiếp real-time với frontend
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Class để quản lý toàn bộ cấu hình của ứng dụng
class Config:
    def __init__(self):
        # Tải cấu hình từ file json khi khởi tạo
        self.load_from_file()
    
    def load_from_file(self):
        # Tải cấu hình từ file config.json
        if os.path.exists('config.json'):
            try:
                with open('config.json', 'r') as f:
                    config_data = json.load(f)
                
                # Cài đặt video
                video_config = config_data.get('video', {})
                self.VIDEO_PATH = video_config.get('default_path', 'query.mp4')
                
                # Cài đặt model
                model_config = config_data.get('models', {})
                self.MODEL_PATH = model_config.get('yolo_path', 'best 3.pt')
                self.RECOG_WEIGHTS = model_config.get('recognition_path', 'arcface_logo_model_best.pth')
                
                # Cài đặt nhận diện
                detect_config = config_data.get('detection', {})
                self.CONF_THRESHOLD = detect_config.get('confidence_threshold', 0.7)
                self.RECOGNITION_THRESHOLD = detect_config.get('recognition_threshold', 0.65)
                self.BATCH_SIZE = detect_config.get('batch_size', 20)
                
                # Cài đặt thư mục
                dir_config = config_data.get('directories', {})
                self.SUPPORT_DIR = dir_config.get('support', 'support')
                self.MASK_DIR = dir_config.get('mask', 'mask')
                self.OUTPUT_DIR = dir_config.get('output', 'output_yolo')
                self.DETECTED_DIR = dir_config.get('detected', 'detected_frames')
                
                # Đường dẫn database
                self.EMBED_DB_PATH = 'embedding_db.pkl'
                
                # Thiết bị chạy (cuda hoặc cpu)
                device_setting = model_config.get('device', 'auto')
                if device_setting == 'auto':
                    self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                else:
                    self.DEVICE = torch.device(device_setting)
                
                print("Đã tải cấu hình từ config.json")
                
            except Exception as e:
                print(f"Lỗi tải config.json: {e}")
                self.set_defaults() # Nếu lỗi thì dùng mặc định
        else:
            print("Không tìm thấy config.json, dùng cài đặt mặc định")
            self.set_defaults()
    
    def set_defaults(self):
        # Các giá trị mặc định nếu không có file config
        self.VIDEO_PATH = 'query.mp4'
        self.MODEL_PATH = 'best 3.pt'
        self.RECOG_WEIGHTS = 'arcface_logo_model_best.pth'
        self.SUPPORT_DIR = 'support'
        self.MASK_DIR = 'mask'
        self.EMBED_DB_PATH = 'embedding_db.pkl'
        self.OUTPUT_DIR = 'output_yolo'
        self.DETECTED_DIR = 'detected_frames'
        self.BATCH_SIZE = 20
        self.CONF_THRESHOLD = 0.7
        self.RECOGNITION_THRESHOLD = 0.65
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Khởi tạo một đối tượng cấu hình toàn cục
Config = Config()

# Các biến toàn cục
video_processor = None # Đối tượng xử lý video
detection_queue = queue.Queue() # Hàng đợi (ít dùng trong phiên bản này)
frame_lock = Lock() # Khóa để tránh xung đột thread (ít dùng trong phiên bản này)

# Hàm tải database embedding
def load_embeddings(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return []

# Hàm lưu database embedding
def save_embeddings(embeddings, path):
    with open(path, 'wb') as f:
        pickle.dump(embeddings, f)

# Model nhận diện logo (ConvNeXt), giống hệt trong file train và main
class LogoEncoder(nn.Module):
    def __init__(self, embedding_dim=512, dropout_rate=0.5):
        super().__init__()
        model_name = 'convnextv2_base.fcmae_ft_in22k_in1k_384'
        checkpoint_path = 'weights/convnext_base_384/model.safetensors'
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            checkpoint_path=checkpoint_path
        )
        in_features = self.backbone.head.fc.in_features
        # Phần MLP để tạo ra embedding vector
        self.embedding = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, embedding_dim * 2),
            nn.BatchNorm1d(embedding_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

    def forward(self, x, mask=None):
        features = self.backbone.forward_features(x)
        if mask is not None:
            # Pooling có trọng số dựa trên mask và attention
            mask = F.interpolate(mask, size=features.shape[2:], mode='nearest')
            mask = mask.clamp(0, 1)
            attention = torch.sigmoid(features.mean(dim=1, keepdim=True))
            weight = attention * mask
            weight_sum = weight.sum(dim=[2,3], keepdim=True) + 1e-6
            pooled = (features * weight).sum(dim=[2,3], keepdim=True) / weight_sum
            pooled = pooled.squeeze(-1).squeeze(-1)
        else:
            # Pooling trung bình nếu không có mask
            pooled = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
        embeddings = self.embedding(pooled)
        normalized_embeddings = F.normalize(embeddings, dim=1)
        return normalized_embeddings

# Class chính để xử lý toàn bộ logic video
class VideoProcessor:
    def __init__(self):
        self.yolo_model = None # Model phát hiện
        self.recog_model = None # Model nhận diện
        self.db_embeddings_tensor = None # Database trên GPU
        self.db_labels = [] # Nhãn tương ứng
        self.cap = None # Đối tượng video capture
        self.is_processing = False # Cờ trạng thái
        self.frame_count = 0
        self.fps = 30
        self.batch_start_frame = 1 # Dùng để quản lý các batch detection
        self.batch_start_time = time.time() # Thời gian bắt đầu batch
        
    def load_models(self):
        # Tải model YOLO và model nhận diện
        try:
            print("Đang tải model YOLO...")
            self.yolo_model = YOLO(Config.MODEL_PATH)
            
            print("Đang tải model nhận diện...")
            self.recog_model = LogoEncoder().to(Config.DEVICE)
            if os.path.exists(Config.RECOG_WEIGHTS):
                checkpoint = torch.load(Config.RECOG_WEIGHTS, map_location=Config.DEVICE)
                self.recog_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            self.recog_model.eval()
            
            self.load_database() # Tải luôn database
            return True
            
        except Exception as e:
            print(f"Lỗi tải model: {e}")
            traceback.print_exc()
            return False
    
    def load_database(self):
        # Tải database embedding lên RAM và GPU
        db = load_embeddings(Config.EMBED_DB_PATH)
        if db:
            db_embeddings_np = np.array([e for e, _ in db])
            self.db_labels = [c for _, c in db]
            self.db_embeddings_tensor = torch.tensor(db_embeddings_np, dtype=torch.float32).to(Config.DEVICE)
            self.db_embeddings_tensor = torch.nn.functional.normalize(self.db_embeddings_tensor, dim=1)
            print(f"Đã tải {len(db)} embeddings từ database")
        else:
            print("Không tìm thấy embedding trong database")
    
    def open_video(self, video_path):
        # Mở file video
        try:
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                return False
            
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            print(f"Mở video thành công. FPS: {self.fps}")
            return True
            
        except Exception as e:
            print(f"Lỗi mở video: {e}")
            return False

    def process_frame(self, frame):
        # Xử lý một frame duy nhất
        start_time = time.time()
        self.frame_count += 1
        output_frame = frame.copy()
        # Chuẩn bị dữ liệu để gửi về frontend
        results_data = {
            'frame_id': self.frame_count,
            'detections': [],
            'frame_base64': '',
            'processing_time': 0
        }
        try:
            # Chạy YOLO
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.yolo_model(rgb, verbose=False)[0]
            detection_count = 0
            crops, masks, bboxes = [], [], []
            if results.boxes is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                confs = results.boxes.conf.cpu().numpy()
                masks_data = results.masks.data.cpu().numpy() if results.masks else []
                # Lọc các detection có confidence thấp
                for i, (box, conf) in enumerate(zip(boxes, confs)):
                    if conf < Config.CONF_THRESHOLD:
                        continue
                    x1, y1, x2, y2 = map(int, box)
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0: continue
                    # Lấy mask tương ứng
                    mask_crop = None
                    if i < len(masks_data):
                        mask_full = masks_data[i]
                        if mask_full.shape[0] >= y2 and mask_full.shape[1] >= x2:
                            mask_crop = mask_full[y1:y2, x1:x2]
                    crops.append(crop)
                    masks.append(mask_crop)
                    bboxes.append((x1, y1, x2, y2))
                
                # Nhận diện theo batch để tăng tốc
                batch_labels_scores = self.recognize_logo_batch(crops, masks)
                
                # Xử lý kết quả và vẽ lên frame
                for idx, ((x1, y1, x2, y2), mask_crop) in enumerate(zip(bboxes, masks)):
                    label, score = batch_labels_scores[idx] if idx < len(batch_labels_scores) else ("Unknown", 0.0)
                    detection_id = str(uuid.uuid4()) # Tạo ID duy nhất cho mỗi detection
                    # Thêm thông tin detection vào kết quả trả về
                    results_data['detections'].append({
                        'id': detection_id, 'bbox': [x1, y1, x2, y2], 'confidence': float(confs[idx]),
                        'label': label, 'score': float(score), 'has_mask': mask_crop is not None,
                        'crop_path': f"/detected/{detection_id}_crop.jpg",
                        'mask_path': f"/detected/{detection_id}_mask.png" if mask_crop is not None else None
                    })
                    # Lưu lại ảnh crop và mask
                    self.save_detection(crops[idx], mask_crop, detection_id, self.frame_count, detection_count)
                    detection_count += 1
                    # Vẽ bounding box và label
                    color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(output_frame, label, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.putText(output_frame, f"{score:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    # Vẽ mask
                    if mask_crop is not None and (x2 - x1 > 0) and (y2 - y1 > 0):
                        try:
                            mask_vis = cv2.resize((mask_crop > 0.5).astype(np.uint8), (x2 - x1, y2 - y1))
                            mask_color = np.zeros((y2 - y1, x2 - x1, 3), dtype=np.uint8)
                            mask_color[:, :, 1] = mask_vis * 255
                            output_frame[y1:y2, x1:x2] = cv2.addWeighted(output_frame[y1:y2, x1:x2], 1.0, mask_color, 0.4, 0)
                        except Exception as me:
                            print(f"Lỗi vẽ mask: {me}")
            # Chuyển frame thành base64 để gửi qua socket
            results_data['frame_base64'] = frame_to_base64(output_frame)
            processing_time = int((time.time() - start_time) * 1000)
            results_data['processing_time'] = processing_time
            # Cứ 300 frame thì reset batch và dọn dẹp file cũ
            if self.frame_count % 300 == 1:
                self.batch_start_frame = self.frame_count
                self.batch_start_time = time.time()
                self.cleanup_old_detections()
        except Exception as e:
            print(f"Lỗi xử lý frame {self.frame_count}: {e}")
            traceback.print_exc()
        results_data['batch_start_frame'] = self.batch_start_frame
        results_data['batch_start_time'] = self.batch_start_time
        return results_data
    
    def save_detection(self, crop, mask, detection_id, frame_id, obj_id):
        # Lưu ảnh crop và mask vào thư mục detected
        try:
            os.makedirs(Config.DETECTED_DIR, exist_ok=True)
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)
            padded_img = resize_with_padding(pil_img, target_size=384)
            crop_path = os.path.join(Config.DETECTED_DIR, f"{detection_id}_crop.jpg")
            padded_img.save(crop_path, quality=90, optimize=True)
            if mask is not None:
                mask_bin = (mask > 0.5).astype(np.uint8) * 255
                mask_pil = Image.fromarray(mask_bin, mode='L')
                padded_mask = resize_with_padding(mask_pil, target_size=384, fill=0)
                mask_path = os.path.join(Config.DETECTED_DIR, f"{detection_id}_mask.png")
                padded_mask.save(mask_path, optimize=True)
        except Exception as e:
            print(f"Lỗi lưu detection {detection_id}: {e}")
            traceback.print_exc()
    
    def recognize_logo(self, crop, mask_crop=None):
        # Nhận diện một logo đơn lẻ (ít dùng hơn bản batch)
        try:
            if self.db_embeddings_tensor is None or len(self.db_labels) == 0:
                return 'Unknown', 0.0
            # Chuẩn bị ảnh
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            ])
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)
            padded_img = resize_with_padding(pil_img, target_size=384)
            img_tensor = transform(padded_img).unsqueeze(0).to(Config.DEVICE)
            # Chuẩn bị mask
            if mask_crop is not None:
                mask_bin = (mask_crop > 0.5).astype(np.uint8) * 255
                mask_pil = Image.fromarray(mask_bin, mode='L')
                padded_mask = resize_with_padding(mask_pil, target_size=384, fill=0)
                mask_tensor = transforms.ToTensor()(padded_mask).unsqueeze(0).to(Config.DEVICE)
            else:
                mask_tensor = torch.ones(1,1,384,384).to(Config.DEVICE)
            # Chạy model và so sánh
            with torch.no_grad():
                embedding = self.recog_model(img_tensor, mask=mask_tensor)
                embedding_np = embedding.cpu().numpy().squeeze()
                db_embeddings_np = self.db_embeddings_tensor.cpu().numpy()
                sims = cosine_similarity([embedding_np], db_embeddings_np)[0]
                max_idx = np.argmax(sims)
                max_sim = sims[max_idx]
                # So sánh với ngưỡng
                if max_sim > Config.RECOGNITION_THRESHOLD:
                    return self.db_labels[max_idx], max_sim
                return 'Unknown', max_sim
        except Exception as e:
            print(f"Lỗi nhận diện logo: {e}")
            return 'Unknown', 0.0

    def recognize_logo_batch(self, crops, masks):
        # Hàm nhận diện theo batch, hiệu quả hơn nhiều
        try:
            if self.db_embeddings_tensor is None or len(self.db_labels) == 0 or not crops:
                return [("Unknown", 0.0)] * len(crops)
            self.recog_model.eval()
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            ])
            batch_tensors, batch_mask_tensors = [], []
            # Chuẩn bị dữ liệu cho cả batch
            for crop, mask_crop in zip(crops, masks):
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(crop_rgb)
                padded_img = resize_with_padding(pil_img, target_size=384)
                img_tensor = transform(padded_img)
                batch_tensors.append(img_tensor)
                if mask_crop is not None:
                    mask_bin = (mask_crop > 0.5).astype(np.uint8) * 255
                    mask_pil = Image.fromarray(mask_bin, mode='L')
                    padded_mask = resize_with_padding(mask_pil, target_size=384, fill=0)
                    mask_tensor = transforms.ToTensor()(padded_mask)
                else:
                    mask_tensor = torch.ones(1, 384, 384)
                batch_mask_tensors.append(mask_tensor)
            batch_tensor = torch.stack(batch_tensors).to(Config.DEVICE)
            batch_mask_tensor = torch.stack(batch_mask_tensors).to(Config.DEVICE)
            # Chạy model và so sánh
            with torch.no_grad():
                batch_embeddings = self.recog_model(batch_tensor, mask=batch_mask_tensor)
                # Dùng phép nhân ma trận để tính cosine similarity cho cả batch, rất nhanh
                similarities = torch.mm(batch_embeddings, self.db_embeddings_tensor.T)
                max_sims, max_indices = torch.max(similarities, dim=1)
                max_sims_cpu = max_sims.cpu().numpy()
                max_indices_cpu = max_indices.cpu().numpy()
                results = []
                for sim, idx in zip(max_sims_cpu, max_indices_cpu):
                    if sim > Config.RECOGNITION_THRESHOLD:
                        results.append((self.db_labels[idx], float(sim)))
                    else:
                        results.append(("Unknown", float(sim)))
                return results
        except Exception as e:
            print(f"Lỗi nhận diện batch: {e}")
            return [("Unknown", 0.0)] * len(crops)

    def cleanup_old_detections(self):
        # Dọn dẹp thư mục detected để tránh đầy ổ cứng
        try:
            detected_dir = Config.DETECTED_DIR
            if not os.path.exists(detected_dir):
                return
            for f in os.listdir(detected_dir):
                if f.endswith('_crop.jpg') or f.endswith('_mask.png'):
                    os.remove(os.path.join(detected_dir, f))
            print(f"Đã xóa toàn bộ detection cũ (batch mới từ frame {self.batch_start_frame})")
        except Exception as e:
            print(f"Lỗi dọn dẹp: {e}")

def resize_with_padding(pil_img, target_size=384, fill=(128,128,128)):
    # Resize giữ tỷ lệ và pad
    w, h = pil_img.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    if pil_img.mode == 'L': # Mask
        resized_img = pil_img.resize((new_w, new_h), Image.NEAREST)
    else: # Ảnh
        resized_img = pil_img.resize((new_w, new_h), Image.BICUBIC)
    pad_w = target_size - new_w
    pad_h = target_size - new_h
    fill_color = fill if isinstance(fill, int) else tuple(fill)
    padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
    padded_img = ImageOps.expand(resized_img, padding, fill=fill_color)
    return padded_img

def image_to_thumb_base64(path, max_side=200):
    # Tạo ảnh thumbnail base64 để hiển thị nhanh trên web
    try:
        if not os.path.exists(path):
            print(f"Không tìm thấy file thumbnail: {path}")
            return None
            
        with Image.open(path) as im:
            im = im.convert('RGB')
            w, h = im.size
            scale = max_side / max(w, h)
            if scale < 1:
                im = im.resize((int(w*scale), int(h*scale)), Image.BILINEAR)
            buf = BytesIO()
            im.save(buf, format='JPEG', quality=75)
            return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Lỗi tạo thumbnail {path}: {e}")
        return None

def frame_to_base64(frame):
    # Chuyển frame OpenCV thành chuỗi base64
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    return frame_base64

# Khởi tạo đối tượng xử lý video toàn cục
video_processor = VideoProcessor()

# Các API endpoint của Flask
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/api/load_models', methods=['POST'])
def load_models():
    # API để client yêu cầu server tải model
    try:
        success = video_processor.load_models()
        return jsonify({'success': success, 'message': 'Tải model thành công' if success else 'Tải model thất bại'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/register_support', methods=['POST'])
def register_support():
    # API để đăng ký logo từ thư mục support
    try:
        data = request.get_json()
        support_dir = data.get('support_dir', Config.SUPPORT_DIR)
        mask_dir = data.get('mask_dir', Config.MASK_DIR)
        
        success = register_support_folder(support_dir, mask_dir)
        if success:
            video_processor.load_database()  # Tải lại database
            
        return jsonify({'success': success, 'message': 'Đăng ký support thành công' if success else 'Đăng ký support thất bại'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/register_detection', methods=['POST'])
def register_detection():
    # API để đăng ký một detection mới làm support
    try:
        data = request.get_json()
        detection_id = data.get('detection_id')
        class_name = data.get('class_name')
        
        if not detection_id or not class_name:
            return jsonify({'success': False, 'message': 'Thiếu detection_id hoặc class_name'})
        
        crop_path = os.path.join(Config.DETECTED_DIR, f"{detection_id}_crop.jpg")
        mask_path = os.path.join(Config.DETECTED_DIR, f"{detection_id}_mask.png")
        
        if not os.path.exists(crop_path):
            return jsonify({'success': False, 'message': 'Không tìm thấy ảnh crop'})
        
        success = register_single_logo(crop_path, mask_path if os.path.exists(mask_path) else None, class_name)
        
        if success:
            video_processor.load_database()  # Tải lại database
            
        return jsonify({'success': success, 'message': f'Đã đăng ký detection là {class_name}' if success else 'Đăng ký detection thất bại'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/start_video', methods=['POST'])
def start_video():
    # API để bắt đầu xử lý video
    try:
        data = request.get_json()
        video_path = data.get('video_path', Config.VIDEO_PATH)
        
        if video_processor.is_processing:
            return jsonify({'success': False, 'message': 'Video đang được xử lý'})
        
        success = video_processor.open_video(video_path)
        if success:
            video_processor.is_processing = True
            # Bắt đầu thread xử lý video để không làm treo server
            thread = Thread(target=process_video_thread)
            thread.daemon = True
            thread.start()
            
        return jsonify({'success': success, 'message': 'Bắt đầu video thành công' if success else 'Bắt đầu video thất bại'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/stop_video', methods=['POST'])
def stop_video():
    # API để dừng xử lý video
    try:
        video_processor.is_processing = False
        if video_processor.cap:
            video_processor.cap.release()
        return jsonify({'success': True, 'message': 'Đã dừng video'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/get_database_info', methods=['GET'])
def get_database_info():
    # API để lấy thông tin về database hiện tại
    try:
        db = load_embeddings(Config.EMBED_DB_PATH)
        class_counts = defaultdict(int)
        for _, class_name in db:
            class_counts[class_name] += 1
        
        return jsonify({
            'success': True,
            'total_embeddings': len(db),
            'classes': dict(class_counts)
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.after_request
def add_no_cache_headers(resp):
    # Chống cache trình duyệt cho các ảnh detection để luôn hiển thị ảnh mới nhất
    if request.path.startswith('/detected') or request.path.startswith('/api/get_detections'):
        resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        resp.headers['Pragma'] = 'no-cache'
        resp.headers['Expires'] = '0'
    return resp

@app.route('/detected/<filename>')
def serve_detected_file_alt(filename):
    # API để phục vụ file ảnh/mask đã được detect
    full_path = os.path.join(Config.DETECTED_DIR, filename)
    if not os.path.exists(full_path):
        # Trả về ảnh trong suốt 1x1 thay vì lỗi 404
        transparent_png = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8\x0f\x00\x00\x01\x00\x01\x00\x00\x00\x00\x00\x00\x18\xdd\x8d\xb4\x1c\x00\x00\x00\x00IEND\xaeB`\x82'
        return Response(transparent_png, mimetype='image/png', headers={
            'Cache-Control': 'no-store, no-cache, must-revalidate, max-age=0',
            'Pragma': 'no-cache',
            'Expires': '0'
        })
    resp = make_response(send_from_directory(Config.DETECTED_DIR, filename))
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp

@app.route('/api/get_detections', methods=['GET'])
def get_detections():
    # API để lấy danh sách tất cả các detection đã lưu
    try:
        include_base64 = request.args.get('base64', '1') == '1'
        batch_start_time = request.args.get('batch_start_time', None)
        detections = []
        
        if os.path.exists(Config.DETECTED_DIR):
            try:
                files = os.listdir(Config.DETECTED_DIR)
            except (OSError, PermissionError):
                files = []
                
            for f in files:
                if not f.endswith('_crop.jpg'):
                    continue
                    
                detection_id = f[:-9]
                crop_path = os.path.join(Config.DETECTED_DIR, f)
                
                if not os.path.exists(crop_path) or not os.path.isfile(crop_path):
                    continue
                    
                try:
                    size = os.path.getsize(crop_path)
                    mtime = os.path.getmtime(crop_path)
                except (OSError, PermissionError):
                    continue
                
                # Lọc theo thời gian bắt đầu batch
                if batch_start_time and mtime < float(batch_start_time):
                    continue
                    
                mask_file = f"{detection_id}_mask.png"
                mask_path = os.path.join(Config.DETECTED_DIR, mask_file)
                
                # Tạo thumbnail
                thumb_b64 = None
                if include_base64:
                    try:
                        thumb_b64 = image_to_thumb_base64(crop_path)
                    except Exception as e:
                        print(f"Lỗi thumbnail cho {crop_path}: {e}")
                        thumb_b64 = None
                
                detections.append({
                    'id': detection_id,
                    'crop_path': f"/detected/{f}?t={int(mtime)}", # Thêm timestamp để chống cache
                    'has_mask': os.path.exists(mask_path) and os.path.isfile(mask_path),
                    'size': size,
                    'mtime': mtime,
                    'thumb': thumb_b64
                })
                
        detections.sort(key=lambda x: x['mtime'], reverse=True) # Sắp xếp mới nhất lên đầu
        return jsonify({'success': True, 'detections': detections})
        
    except Exception as e:
        print(f"Lỗi trong get_detections: {e}")
        return jsonify({'success': False, 'message': str(e), 'detections': []})

# Các sự kiện SocketIO
@socketio.on('connect')
def handle_connect():
    print(f"Client đã kết nối: {request.sid}")
    emit('status', {'message': 'Đã kết nối tới server'})

@socketio.on('disconnect')
def handle_disconnect():
    print(f"Client đã ngắt kết nối: {request.sid}")

# Thread xử lý video
def process_video_thread():
    # Chạy vòng lặp xử lý video trong một thread riêng
    try:
        frame_delay = 1.0 / video_processor.fps if video_processor.fps > 0 else 1.0/30
        
        while video_processor.is_processing and video_processor.cap:
            ret, frame = video_processor.cap.read()
            
            if not ret:
                # Hết video
                socketio.emit('video_ended', {'message': 'Xử lý video hoàn tất'})
                break
            
            # Xử lý frame
            results = video_processor.process_frame(frame)
            
            # Gửi kết quả về cho client qua socket
            socketio.emit('frame_processed', results)
            
            # Điều khiển tốc độ xử lý để khớp với FPS của video
            time.sleep(frame_delay)
            
    except Exception as e:
        print(f"Lỗi trong thread xử lý video: {e}")
        traceback.print_exc()
        socketio.emit('error', {'message': f'Lỗi xử lý video: {str(e)}'})
    
    finally:
        # Dọn dẹp khi kết thúc
        video_processor.is_processing = False
        if video_processor.cap:
            video_processor.cap.release()

# Các hàm hỗ trợ đăng ký logo
def register_support_folder(support_dir, mask_dir=None):
    # Đăng ký toàn bộ logo từ thư mục, nhưng không xóa database cũ
    try:
        if not os.path.exists(support_dir):
            print(f"Thư mục support không tồn tại: {support_dir}")
            return False

        # Quét các thư mục con (mỗi thư mục là một class)
        class_dirs = [d for d in os.listdir(support_dir)
                     if os.path.isdir(os.path.join(support_dir, d))]

        if not class_dirs:
            print("Không có thư mục class nào trong support")
            return False

        total_registered = 0

        for class_name in class_dirs:
            class_path = os.path.join(support_dir, class_name)
            mask_class_path = os.path.join(mask_dir, class_name) if mask_dir else None
            image_files = [f for f in os.listdir(class_path)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

            print(f"Class: {class_name} - Tìm thấy {len(image_files)} ảnh")

            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)
                mask_path = None
                if mask_class_path and os.path.exists(mask_class_path):
                    base_name = os.path.splitext(img_file)[0]
                    possible_mask_names = [
                        f"{base_name}.png", f"{base_name}.jpg",
                        f"{base_name}_mask.png", f"{base_name}_mask.jpg"
                    ]
                    for mask_name in possible_mask_names:
                        potential_mask_path = os.path.join(mask_class_path, mask_name)
                        if os.path.exists(potential_mask_path):
                            mask_path = potential_mask_path
                            break
                
                # Luôn thêm support mới, không xóa gì cả
                success = register_single_logo(img_path, mask_path, class_name)
                if success:
                    total_registered += 1

        print(f"Đã đăng ký {total_registered} logo từ {len(class_dirs)} class")
        return total_registered > 0

    except Exception as e:
        print(f"Lỗi đăng ký thư mục support: {e}")
        traceback.print_exc()
        return False

def register_single_logo(image_path, mask_path, class_name):
    # Đăng ký một logo đơn lẻ
    try:
        if not video_processor.recog_model:
            print("Model nhận diện chưa được tải")
            return False

        video_processor.recog_model.eval()

        # Tự động thêm hậu tố nếu class_name đã tồn tại
        db = load_embeddings(Config.EMBED_DB_PATH)
        existing_classes = [c for _, c in db]
        orig_class_name = class_name
        idx = 1
        while class_name in existing_classes:
            class_name = f"{orig_class_name}_{idx}"
            idx += 1

        # Xử lý ảnh
        image = Image.open(image_path).convert('RGB')
        padded_img = resize_with_padding(image, target_size=384)

        # Xử lý mask
        if mask_path and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')
            padded_mask = resize_with_padding(mask, target_size=384, fill=0)
        else:
            padded_mask = Image.new("L", (384, 384), 255)
            print(f"Cảnh báo: Không có mask cho {class_name}, dùng mask trắng")

        # Chuẩn bị tensor
        image_tensor = transforms.ToTensor()(padded_img)
        image_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image_tensor)
        image_tensor = image_tensor.unsqueeze(0).to(Config.DEVICE)
        mask_tensor = transforms.ToTensor()(padded_mask).unsqueeze(0).to(Config.DEVICE)

        # Trích xuất embedding
        with torch.no_grad():
            embedding = video_processor.recog_model(image_tensor, mask=mask_tensor).cpu().numpy().squeeze()

        # Thêm vào database
        db.append((embedding, class_name))
        save_embeddings(db, Config.EMBED_DB_PATH)

        # Lưu lại ảnh/mask đã xử lý để tiện theo dõi
        support_resize_dir = "support_resized_test"
        support_mask_dir = "support_mask_test"
        os.makedirs(os.path.join(support_resize_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(support_mask_dir, class_name), exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        # Xử lý trùng tên file
        resize_save_dir = os.path.join(support_resize_dir, class_name)
        mask_save_dir = os.path.join(support_mask_dir, class_name)
        resize_filename = f"{base_name}_resized.jpg"
        mask_filename = f"{base_name}_mask.png"
        resize_path = os.path.join(resize_save_dir, resize_filename)
        mask_path_out = os.path.join(mask_save_dir, mask_filename)
        file_idx = 1
        while os.path.exists(resize_path) or os.path.exists(mask_path_out):
            resize_filename = f"{base_name}_resized_{file_idx}.jpg"
            mask_filename = f"{base_name}_mask_{file_idx}.png"
            resize_path = os.path.join(resize_save_dir, resize_filename)
            mask_path_out = os.path.join(mask_save_dir, mask_filename)
            file_idx += 1

        padded_img.save(resize_path)
        padded_mask.save(mask_path_out)
        print(f"Đã lưu ảnh support đã resize tới {resize_path}")
        print(f"Đã lưu mask tới {mask_path_out}")

        print(f"Đăng ký thành công logo: {class_name}")
        return True

    except Exception as e:
        print(f"Lỗi đăng ký logo {class_name}: {e}")
        traceback.print_exc()
        return False

# Entry point của ứng dụng
if __name__ == '__main__':
    # Tạo các thư mục cần thiết
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.DETECTED_DIR, exist_ok=True)
    
    print("Bắt đầu ứng dụng Web nhận diện Logo...")
    print(f"Thiết bị: {Config.DEVICE}")
    print("Truy cập ứng dụng tại: http://localhost:5000")
    
    # Chạy server với socketio
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)