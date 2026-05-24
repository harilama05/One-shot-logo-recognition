# main_convnext.py

import os
import cv2
import torch
import numpy as np
import time
import pickle
from PIL import Image, ImageOps
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity
import timm  # Thêm timm cho ConvNeXt

# ----- Cấu hình -----
VIDEO_PATH = '/home/qmask_lamnh45/query.mp4' # Đường dẫn video đầu vào
MODEL_PATH = '/home/qmask_lamnh45/best 2.pt' # Đường dẫn weight YOLO để phát hiện logo
RECOG_WEIGHTS = '/home/qmask_lamnh45/arcface_logo_model_best.pth' # Đường dẫn weight model nhận diện
SUPPORT_DIR = '/home/qmask_lamnh45/support' # Thư mục chứa ảnh support (ảnh mẫu)
MASK_DIR = '/home/qmask_lamnh45/mask' # Thư mục chứa mask cho ảnh support
EMBED_DB_PATH = '/home/qmask_lamnh45/embedding_db.pkl' # File lưu database embedding
OUTPUT_PATH = '/home/qmask_lamnh45/output.mp4' # Đường dẫn video đầu ra
BATCH_SIZE = 20 # Xử lý N logo cùng lúc để tăng tốc
CONF_THRESHOLD = 0.7 # Ngưỡng tin cậy của YOLO
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Định nghĩa model ConvNeXt -----
# Đây là model trích xuất đặc trưng, giống hệt file train
class LogoEncoder(nn.Module):
    def __init__(self, embedding_dim=512, dropout_rate=0.5):
        super().__init__()
        model_name = 'convnextv2_base.fcmae_ft_in22k_in1k_384'
        checkpoint_path = '/home/qmask_lamnh45/weights/convnext_base_384/model.safetensors'
        print(f"Tải model offline: {model_name} từ {checkpoint_path}")
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            checkpoint_path=checkpoint_path
        )
        in_features = self.backbone.head.fc.in_features
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
            mask = F.interpolate(mask, size=features.shape[2:], mode='nearest')
            mask = mask.clamp(0, 1)
            attention = torch.sigmoid(features.mean(dim=1, keepdim=True))
            weight = attention * mask
            weight_sum = weight.sum(dim=[2,3], keepdim=True) + 1e-6
            pooled = (features * weight).sum(dim=[2,3], keepdim=True) / weight_sum
            pooled = pooled.squeeze(-1).squeeze(-1)
        else:
            pooled = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
        embeddings = self.embedding(pooled)
        normalized_embeddings = F.normalize(embeddings, dim=1)
        return normalized_embeddings

# ----- Các hàm tải/lưu database -----
def load_embeddings(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return []

def save_embeddings(embeddings, path):
    with open(path, 'wb') as f:
        pickle.dump(embeddings, f)

# ----- Các hàm tiện ích -----
def resize_with_padding(pil_img, target_size=384, fill=(128,128,128)):
    # Resize ảnh nhưng giữ nguyên tỷ lệ, phần thiếu sẽ được đệm (pad)
    w, h = pil_img.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    # Dùng NEAREST cho mask để giữ cạnh sắc nét, BICUBIC cho ảnh để mịn
    if pil_img.mode == 'L':
        resized_img = pil_img.resize((new_w, new_h), Image.NEAREST)
    else:
        resized_img = pil_img.resize((new_w, new_h), Image.BICUBIC)
    pad_w = target_size - new_w
    pad_h = target_size - new_h
    fill_color = fill if isinstance(fill, int) else tuple(fill)
    padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
    padded_img = ImageOps.expand(resized_img, padding, fill=fill_color)
    return padded_img

def register_logo(image_path, class_name, model, mask_path=None, db_path="embedding_db.pkl", support_resize_dir="support_resized_test", support_mask_dir="support_mask_test"):
    # Đăng ký một logo mới vào database
    model.eval()
    try:
        image = Image.open(image_path).convert('RGB')
        padded_img = resize_with_padding(image, target_size=384)

        # Nếu có mask thì dùng, không thì tạo mask trắng (toàn bộ ảnh)
        if mask_path and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')
            padded_mask = resize_with_padding(mask, target_size=384, fill=0)
        else:
            # Fallback: tạo mask toàn trắng nếu không có mask
            padded_mask = Image.new("L", (384, 384), 255)
            print(f"Cảnh báo: Không có mask cho {class_name}, dùng mask toàn bộ ảnh")

        # Lưu lại ảnh/mask đã xử lý để debug
        os.makedirs(os.path.join(support_resize_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(support_mask_dir, class_name), exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        padded_img.save(os.path.join(support_resize_dir, class_name, f"{base_name}_resized.jpg"))
        padded_mask.save(os.path.join(support_mask_dir, class_name, f"{base_name}_mask.png"))

        # Chuẩn bị tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(padded_img).unsqueeze(0).to(DEVICE)
        mask_tensor = transforms.ToTensor()(padded_mask).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            embedding = model(image_tensor, mask=mask_tensor).cpu().numpy().squeeze()

        db = load_embeddings(db_path)
        db.append((embedding, class_name))
        save_embeddings(db, db_path)
        print(f"Đăng ký thành công logo: {class_name}")
        print(f"Đã lưu ảnh support đã resize tới {os.path.join(support_resize_dir, class_name)}")
        print(f"Đã lưu mask tới {os.path.join(support_mask_dir, class_name)}")

    except Exception as e:
        print(f"Lỗi đăng ký logo {class_name}: {e}")

def identify_logo(image_path, mask_path=None, model=None, threshold=0.5, db_path="embedding_db.pkl", top_k=3, save_query_dir="query_debug_test"):
    # Hàm nhận diện một ảnh đơn lẻ, chủ yếu để test
    import time
    start_time = time.time()
    model.eval()
    try:
        # Xử lý ảnh
        image = Image.open(image_path).convert('RGB')
        padded_img = resize_with_padding(image, target_size=384, fill=(128,128,128))
        img_tensor = transforms.ToTensor()(padded_img)
        img_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_tensor)
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

        # Xử lý mask
        if mask_path is not None and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")
            padded_mask = resize_with_padding(mask, target_size=384, fill=0)
            mask_tensor = transforms.ToTensor()(padded_mask).unsqueeze(0).to(DEVICE)
        else:
            mask_tensor = torch.ones(1, 1, 384, 384).to(DEVICE)

        # Lưu lại để debug
        os.makedirs(save_query_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        padded_img.save(os.path.join(save_query_dir, f"{base_name}_query.jpg"))
        if mask_path is not None and os.path.exists(mask_path):
            padded_mask.save(os.path.join(save_query_dir, f"{base_name}_query_mask.png"))

        with torch.no_grad():
            query_embedding = model(img_tensor, mask=mask_tensor).cpu().numpy().squeeze()

        db = load_embeddings(db_path)
        if not db:
            return "Chưa có logo nào trong database."

        # So sánh với database
        similarities = []
        for embedding, class_name in db:
            sim = cosine_similarity([query_embedding], [embedding])[0][0]
            similarities.append((sim, class_name))
        similarities.sort(reverse=True)
        inference_time = time.time() - start_time
        print(f"\nTop {top_k} kết quả khớp nhất:")
        for i, (score, cls) in enumerate(similarities[:top_k]):
            print(f"  {i+1}. {cls:20s} : {score:.4f}")
        print(f"Thời gian inference: {inference_time:.4f} giây")
        best_score, best_class = similarities[0]
        if best_score > threshold:
            return f"Nhận diện là: {best_class} (độ tin cậy={best_score:.4f})"
        else:
            return f"Không tìm thấy kết quả khớp (kết quả tốt nhất={best_class}: {score:.4f})"
    except Exception as e:
        return f"Lỗi trong quá trình nhận diện: {e}"

def register_support_folder(support_dir, model, db_path, mask_dir=None):
    # Hàm quan trọng: tự động quét thư mục support để tạo database
    print(f"Đang quét thư mục support: {support_dir}")
    if mask_dir:
        print(f"Đang quét thư mục mask: {mask_dir}")

    if not os.path.exists(support_dir):
        print(f"Thư mục support không tồn tại: {support_dir}")
        return

    # Xóa database cũ để tạo mới
    if os.path.exists(db_path):
        os.remove(db_path)
        print("Đã xóa database cũ")

    # Mỗi thư mục con trong support_dir là một class
    class_dirs = [d for d in os.listdir(support_dir)
                  if os.path.isdir(os.path.join(support_dir, d))]

    if not class_dirs:
        print("Không tìm thấy thư mục class nào trong support")
        return

    total_registered = 0

    for class_name in class_dirs:
        class_path = os.path.join(support_dir, class_name)
        mask_class_path = os.path.join(mask_dir, class_name) if mask_dir else None

        image_files = [f for f in os.listdir(class_path)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        print(f"Class: {class_name} - Tìm thấy {len(image_files)} ảnh")

        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)

            # Tìm mask tương ứng với ảnh
            mask_path = None
            if mask_class_path and os.path.exists(mask_class_path):
                base_name = os.path.splitext(img_file)[0]
                # Thử nhiều kiểu tên mask khác nhau
                possible_mask_names = [
                    f"{base_name}.png",
                    f"{base_name}.jpg",
                    f"{base_name}_mask.png",
                    f"{base_name}_mask.jpg"
                ]

                for mask_name in possible_mask_names:
                    potential_mask_path = os.path.join(mask_class_path, mask_name)
                    if os.path.exists(potential_mask_path):
                        mask_path = potential_mask_path
                        break

            # Gọi hàm register_logo cho từng ảnh
            register_logo(img_path, class_name, model, mask_path, db_path)
            total_registered += 1

    print(f"Đã register {total_registered} logo từ {len(class_dirs)} classes")

def identify_logo_batch_from_arrays(crop_arrays, model, db_embeddings_tensor, db_labels,
                                  masks=None, threshold=0.3, batch_size=16):
    # Hàm nhận diện logo chính, xử lý theo batch từ các mảng numpy
    if not crop_arrays:
        return []

    model.eval()
    results = []

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Xử lý từng batch
    for i in range(0, len(crop_arrays), batch_size):
        batch_crops = crop_arrays[i:i+batch_size]
        batch_masks = masks[i:i+batch_size] if masks else [None] * len(batch_crops)

        batch_tensors = []
        batch_mask_tensors = []

        for crop, mask in zip(batch_crops, batch_masks):
            # Chuyển BGR (từ OpenCV) sang RGB và resize
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)
            padded_img = resize_with_padding(pil_img, target_size=384)

            img_tensor = transform(padded_img)
            batch_tensors.append(img_tensor)

            # Xử lý mask
            if mask is not None:
                mask_bin = (mask > 0.5).astype(np.uint8) * 255
                mask_pil = Image.fromarray(mask_bin, mode='L')
                padded_mask = resize_with_padding(mask_pil, target_size=384, fill=0)
                mask_tensor = transforms.ToTensor()(padded_mask)
            else:
                mask_tensor = torch.ones(1, 384, 384) # Mask mặc định

            batch_mask_tensors.append(mask_tensor)

        # Gom thành batch tensor và đưa lên GPU
        batch_tensor = torch.stack(batch_tensors).to(DEVICE)
        batch_mask_tensor = torch.stack(batch_mask_tensors).to(DEVICE)

        with torch.no_grad():
            batch_embeddings = model(batch_tensor, mask=batch_mask_tensor)

            # Tính similarity với toàn bộ database bằng một phép nhân ma trận (rất nhanh)
            similarities = torch.mm(batch_embeddings, db_embeddings_tensor.T)
            max_sims, max_indices = torch.max(similarities, dim=1)

            # Chuyển kết quả về CPU
            max_sims_cpu = max_sims.cpu().numpy()
            max_indices_cpu = max_indices.cpu().numpy()

            for sim, idx in zip(max_sims_cpu, max_indices_cpu):
                if sim > threshold:
                    label = db_labels[idx]
                    score = float(sim)
                else:
                    label = "Unknown"
                    score = float(sim)

                results.append((label, score))

    return results

def save_crop_and_mask(frame, bbox, mask, save_dir, frame_id, obj_id):
    # Hàm tiện ích để lưu lại các patch YOLO phát hiện được để debug
    os.makedirs(save_dir, exist_ok=True)
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]
    # Lưu ảnh crop đã resize
    if crop.size > 0:
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        padded_img = resize_with_padding(pil_img, target_size=384)
        padded_img.save(os.path.join(save_dir, f"frame{frame_id:05d}_obj{obj_id}_crop_resized.jpg"))
    # Lưu mask crop đã resize
    if mask is not None and mask.size > 0:
        mask_bin = (mask > 0.5).astype(np.uint8) * 255
        mask_pil = Image.fromarray(mask_bin, mode='L')
        padded_mask = resize_with_padding(mask_pil, target_size=384, fill=0)
        padded_mask.save(os.path.join(save_dir, f"frame{frame_id:05d}_obj{obj_id}_mask_resized.png"))

# ----- Hàm chạy pipeline chính -----
def run_logo_recognition():
    print("Khởi chạy pipeline nhận diện logo...")
    start_time = time.time()

    print("Đang tải mô hình YOLO...")
    yolo_model = YOLO(MODEL_PATH)

    print("Đang tải mô hình nhận diện logo ArcFace...")
    recog_model = LogoEncoder().to(DEVICE)
    checkpoint = torch.load(RECOG_WEIGHTS, map_location=DEVICE)
    # strict=False để bỏ qua các key không khớp, hữu ích khi cấu trúc lưu khác chút
    recog_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    recog_model.eval()

    # Nếu chưa có database, tạo nó từ thư mục support
    if not os.path.exists(EMBED_DB_PATH):
        print("Chưa có embedding, đang tạo...")
        register_support_folder(SUPPORT_DIR, recog_model, EMBED_DB_PATH, MASK_DIR)

    db = load_embeddings(EMBED_DB_PATH)
    if not db:
        raise RuntimeError("Không có dữ liệu embedding trong database!")

    # Tải database lên GPU để tính toán nhanh
    db_embeddings_np = np.array([e for e, _ in db])
    db_labels = [c for _, c in db]
    db_embeddings_tensor = torch.tensor(db_embeddings_np, dtype=torch.float32).to(DEVICE)
    db_embeddings_tensor = F.normalize(db_embeddings_tensor, dim=1)

    print("Mở video đầu vào...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Không thể mở video: {VIDEO_PATH}")

    # Chuẩn bị để ghi video output
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Chạy YOLO
        results = yolo_model(rgb)[0]

        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        # Lấy mask nếu model YOLO là model segmentation
        masks = results.masks.data.cpu().numpy() if results.masks else []

        crops, bboxes, masks_keep = [], [], []
        for i, (box, conf) in enumerate(zip(boxes, confs)):
            if conf < CONF_THRESHOLD:
                continue
            x1, y1, x2, y2 = map(int, box)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crops.append(crop)
            bboxes.append((x1, y1, x2, y2))
            # Cắt mask tương ứng với bounding box
            if i < len(masks):
                mask_full = masks[i]
                mask_crop = mask_full[y1:y2, x1:x2] if mask_full.shape[0] >= y2 and mask_full.shape[1] >= x2 else None
                masks_keep.append(mask_crop if mask_crop is not None and mask_crop.size > 0 else None)
            else:
                masks_keep.append(None)
            # Lưu lại patch để debug
            save_crop_and_mask(frame, (x1, y1, x2, y2), mask_crop, "output_yolo", frame_id, i)
        if crops:
            start_recog = time.time()
            # Chạy nhận diện batch
            labels = identify_logo_batch_from_arrays(
                crops, recog_model, db_embeddings_tensor, db_labels,
                masks=masks_keep, threshold=0.4, batch_size=BATCH_SIZE
            )
            recog_time = (time.time() - start_recog) * 1000
            print(f"[Frame {frame_id}]  Thời gian nhận diện: {recog_time:.1f} ms")

            # Vẽ kết quả lên frame
            matched = 0
            for (x1, y1, x2, y2), (label, score), mask in zip(bboxes, labels, masks_keep):
                color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
                if label != "Unknown":
                    matched += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label}", (x1, y1 - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(frame, f"{score:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                # Vẽ mask lên frame
                if mask is not None:
                    mask_vis = (mask > 0.5).astype(np.uint8)
                    mask_vis = cv2.resize(mask_vis, (x2-x1, y2-y1))
                    mask_color = np.zeros_like(frame[y1:y2, x1:x2], dtype=np.uint8)
                    mask_color[:, :, 1] = mask_vis * 255 # Tô màu xanh lá
                    frame[y1:y2, x1:x2] = cv2.addWeighted(frame[y1:y2, x1:x2], 1.0, mask_color, 0.5, 0)

            print(f"[Frame {frame_id}] Logo nhận diện: {matched}/{len(labels)}")
        else:
            print(f"[Frame {frame_id}] Không tìm thấy patch hợp lệ.")

        out_writer.write(frame)

    cap.release()
    out_writer.release()

    elapsed = time.time() - start_time
    print(f"Hoàn thành video. Thời gian xử lý: {elapsed:.2f} giây")
    print(f"Video output lưu tại: {OUTPUT_PATH}")

# Chạy chương trình
if __name__ == "__main__":
    run_logo_recognition()