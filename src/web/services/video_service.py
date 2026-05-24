import os
import sys
import cv2
import time
import torch
import numpy as np
import traceback
import uuid
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

from config import Config
from utils import load_embeddings, resize_with_padding, frame_to_base64

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from oslr.models.arcface_model import load_arcface_model

class VideoProcessor:
    def __init__(self):
        self.yolo_model = None
        self.recog_model = None
        self.db_embeddings_tensor = None
        self.db_labels = []
        self.cap = None
        self.is_processing = False
        self.frame_count = 0
        self.fps = 30
        self.batch_start_frame = 1
        self.batch_start_time = time.time()
        
    def load_models(self):
        try:
            print("Đang tải model YOLO...")
            self.yolo_model = YOLO(Config.MODEL_PATH)
            
            print("Đang tải model nhận diện...")
            if os.path.exists(Config.RECOG_WEIGHTS):
                self.recog_model = load_arcface_model(Config.RECOG_WEIGHTS, Config.DEVICE)
            else:
                raise FileNotFoundError(f"Không tìm thấy {Config.RECOG_WEIGHTS}")
            
            self.load_database()
            return True
        except Exception as e:
            print(f"Lỗi tải model: {e}")
            traceback.print_exc()
            return False
    
    def load_database(self):
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
        start_time = time.time()
        self.frame_count += 1
        output_frame = frame.copy()
        
        results_data = {
            'frame_id': self.frame_count,
            'detections': [],
            'frame_base64': '',
            'processing_time': 0
        }
        
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.yolo_model(rgb, verbose=False)[0]
            detection_count = 0
            crops, masks, bboxes = [], [], []
            
            if results.boxes is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                confs = results.boxes.conf.cpu().numpy()
                masks_data = results.masks.data.cpu().numpy() if results.masks else []
                
                for i, (box, conf) in enumerate(zip(boxes, confs)):
                    if conf < Config.CONF_THRESHOLD:
                        continue
                    x1, y1, x2, y2 = map(int, box)
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0: continue
                    
                    mask_crop = None
                    if i < len(masks_data):
                        mask_full = masks_data[i]
                        if mask_full.shape[0] >= y2 and mask_full.shape[1] >= x2:
                            mask_crop = mask_full[y1:y2, x1:x2]
                            
                    crops.append(crop)
                    masks.append(mask_crop)
                    bboxes.append((x1, y1, x2, y2))
                
                batch_labels_scores = self.recognize_logo_batch(crops, masks)
                
                for idx, ((x1, y1, x2, y2), mask_crop) in enumerate(zip(bboxes, masks)):
                    label, score = batch_labels_scores[idx] if idx < len(batch_labels_scores) else ("Unknown", 0.0)
                    detection_id = str(uuid.uuid4())
                    
                    results_data['detections'].append({
                        'id': detection_id, 'bbox': [x1, y1, x2, y2], 'confidence': float(confs[idx]),
                        'label': label, 'score': float(score), 'has_mask': mask_crop is not None,
                        'crop_path': f"/detected/{detection_id}_crop.jpg",
                        'mask_path': f"/detected/{detection_id}_mask.png" if mask_crop is not None else None
                    })
                    
                    self.save_detection(crops[idx], mask_crop, detection_id, self.frame_count, detection_count)
                    detection_count += 1
                    
                    color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(output_frame, label, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.putText(output_frame, f"{score:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    if mask_crop is not None and (x2 - x1 > 0) and (y2 - y1 > 0):
                        try:
                            mask_vis = cv2.resize((mask_crop > 0.5).astype(np.uint8), (x2 - x1, y2 - y1))
                            mask_color = np.zeros((y2 - y1, x2 - x1, 3), dtype=np.uint8)
                            mask_color[:, :, 1] = mask_vis * 255
                            output_frame[y1:y2, x1:x2] = cv2.addWeighted(output_frame[y1:y2, x1:x2], 1.0, mask_color, 0.4, 0)
                        except Exception as me:
                            print(f"Lỗi vẽ mask: {me}")
                            
            results_data['frame_base64'] = frame_to_base64(output_frame)
            results_data['processing_time'] = int((time.time() - start_time) * 1000)
            
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
        try:
            os.makedirs(Config.DETECTED_DIR, exist_ok=True)
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)
            padded_img = resize_with_padding(pil_img, target_size=380)
            crop_path = os.path.join(Config.DETECTED_DIR, f"{detection_id}_crop.jpg")
            padded_img.save(crop_path, quality=90, optimize=True)
            if mask is not None:
                mask_bin = (mask > 0.5).astype(np.uint8) * 255
                mask_pil = Image.fromarray(mask_bin, mode='L')
                padded_mask = resize_with_padding(mask_pil, target_size=380, fill=0)
                mask_path = os.path.join(Config.DETECTED_DIR, f"{detection_id}_mask.png")
                padded_mask.save(mask_path, optimize=True)
        except Exception as e:
            print(f"Lỗi lưu detection {detection_id}: {e}")
            traceback.print_exc()

    def recognize_logo_batch(self, crops, masks):
        try:
            if self.db_embeddings_tensor is None or len(self.db_labels) == 0 or not crops:
                return [("Unknown", 0.0)] * len(crops)
            
            self.recog_model.eval()
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            ])
            
            batch_tensors, batch_mask_tensors = [], []
            for crop, mask_crop in zip(crops, masks):
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(crop_rgb)
                padded_img = resize_with_padding(pil_img, target_size=380)
                batch_tensors.append(transform(padded_img))
                
                if mask_crop is not None:
                    mask_bin = (mask_crop > 0.5).astype(np.uint8) * 255
                    mask_pil = Image.fromarray(mask_bin, mode='L')
                    padded_mask = resize_with_padding(mask_pil, target_size=380, fill=0)
                    batch_mask_tensors.append(transforms.ToTensor()(padded_mask))
                else:
                    batch_mask_tensors.append(torch.ones(1, 380, 380))
                    
            batch_tensor = torch.stack(batch_tensors).to(Config.DEVICE)
            batch_mask_tensor = torch.stack(batch_mask_tensors).to(Config.DEVICE)
            
            with torch.no_grad():
                batch_embeddings = self.recog_model(batch_tensor, mask=batch_mask_tensor)
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

# Global instance
video_processor = VideoProcessor()
