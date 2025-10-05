# convnext.py

# Bỏ qua chứng chỉ SSL, đôi khi cần thiết khi tải weight tự động
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pickle
import cv2
from PIL import Image
from tqdm import tqdm
import shutil
import random
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from collections import Counter
from efficientnet_pytorch import EfficientNet
import warnings
warnings.filterwarnings('ignore')
from PIL import ImageDraw
from PIL import Image, ImageOps
import timm  # Thư viện cực mạnh để dùng các model SOTA như ConvNeXt, Vision Transformer

# Chọn thiết bị, ưu tiên CUDA nếu có
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Sử dụng thiết bị: {device}")

# Định nghĩa Center Loss
# Mục đích: giảm khoảng cách giữa các feature trong cùng một lớp, kéo chúng về gần tâm của lớp đó
class CenterLoss(nn.Module):
    def __init__(self, num_classes, feature_dim, lambda_c=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.lambda_c = lambda_c # Trọng số của center loss
        # Khởi tạo tâm cho mỗi lớp, đây là tham số sẽ được học
        self.centers = nn.Parameter(torch.randn(num_classes, feature_dim))
        nn.init.xavier_uniform_(self.centers)

    def forward(self, features, labels):
        # features: embedding của batch, labels: nhãn tương ứng
        batch_size = features.size(0)
        # Chuẩn hóa features và centers để tính toán trên mặt cầu đơn vị
        features_norm = F.normalize(features, dim=1)
        centers_norm = F.normalize(self.centers, dim=1)

        # Lấy ra tâm tương ứng với từng sample trong batch
        centers_batch = centers_norm[labels]

        # Tính loss bằng MSE giữa feature và tâm của nó
        center_loss = F.mse_loss(features_norm, centers_batch)

        return self.lambda_c * center_loss

# Định nghĩa Orthogonal Regularization
# Mục đích: làm cho các vector trọng số (đại diện cho các class) trở nên trực giao với nhau
# Giúp tăng khoảng cách giữa các lớp, làm mô hình phân biệt tốt hơn
class OrthogonalRegularization(nn.Module):
    def __init__(self, lambda_orth=1e-4):
        super().__init__()
        self.lambda_orth = lambda_orth # Trọng số của loss này

    def forward(self, weight_matrix):
        # weight_matrix: ma trận trọng số của lớp cuối, shape (num_classes, feature_dim)
        W = F.normalize(weight_matrix, dim=1)
        # Tính W * W^T
        WTW = torch.mm(W, W.t())
        # Ma trận đơn vị
        I = torch.eye(WTW.size(0), device=WTW.device)
        # Loss là khoảng cách Frobenius giữa WTW và I. Khi loss = 0, các vector trực giao
        orth_loss = torch.norm(WTW - I, p='fro') ** 2
        return self.lambda_orth * orth_loss

# Cải tiến ArcFace Loss
# Kết hợp ArcFace, Focal Loss, Center Loss và Orthogonal Regularization
class ImprovedArcFaceLoss(nn.Module):
    def __init__(self, feature_dim, num_classes, scale=30.0, margin=0.6,
                 focal_loss=True, alpha=0.25, gamma=2.0):
        super().__init__()
        # Trọng số W của lớp linear cuối cùng, cũng là đại diện cho các class center
        self.weight = nn.Parameter(torch.randn(num_classes, feature_dim))
        nn.init.xavier_uniform_(self.weight)
        # Các tham số của ArcFace
        self.scale = scale # s: bán kính mặt cầu
        self.margin = margin # m: margin góc
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        # Tùy chọn Focal Loss
        self.focal_loss = focal_loss
        self.alpha = alpha
        self.gamma = gamma

        # Khởi tạo các loss phụ trợ
        self.center_loss = CenterLoss(num_classes, feature_dim, lambda_c=0.1)
        self.orth_reg = OrthogonalRegularization(lambda_orth=1e-4)

    def forward(self, input, label):
        # input: feature embedding từ model, output: nhãn
        # Chuẩn hóa feature và weight
        input_norm = F.normalize(input, dim=1)
        weight_norm = F.normalize(self.weight, dim=1)

        # Tính cosine similarity, tương đương output của lớp linear
        cosine = F.linear(input_norm, weight_norm)
        # Công thức của ArcFace: thêm margin góc vào logit của class đúng
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2).clamp(min=1e-12))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm) # Đảm bảo hàm μονότονος

        # Tạo one-hot vector từ label
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)

        # Chỉ áp dụng margin cho logit của class đúng
        logits = one_hot * phi + (1 - one_hot) * cosine
        logits *= self.scale

        # Tính ArcFace loss, có thể kết hợp với Focal Loss
        if self.focal_loss:
            # Focal Loss: giảm trọng số của các sample dễ đoán, tập trung vào sample khó
            ce_loss = F.cross_entropy(logits, label, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_weight = self.alpha * (1 - pt) ** self.gamma
            arcface_loss = (focal_weight * ce_loss).mean()
        else:
            arcface_loss = F.cross_entropy(logits, label)

        # Tính các loss phụ trợ
        center_loss = self.center_loss(input_norm, label)
        orth_loss = self.orth_reg(self.weight)

        # Tổng hợp các loss lại
        total_loss = arcface_loss + center_loss + orth_loss

        return total_loss, {
            'arcface_loss': arcface_loss.item(),
            'center_loss': center_loss.item(),
            'orth_loss': orth_loss.item()
        }

# Lớp Dataset tùy chỉnh, hỗ trợ đọc ảnh và mask tương ứng
class LogoDatasetWithMask(Dataset):
    def __init__(self, root_dir, transform=None, mask_root="logo_output_mask_split"):
        self.root_dir = root_dir
        self.transform = transform
        self.mask_root = mask_root
        self.samples = []
        self.class_to_idx = {}
        self.class_counts = {}

        # Quét thư mục để lấy danh sách class và ảnh
        classes = sorted([d for d in os.listdir(root_dir)
                         if os.path.isdir(os.path.join(root_dir, d))])
        for idx, class_name in enumerate(classes):
            self.class_to_idx[class_name] = idx
            class_dir = os.path.join(root_dir, class_name)

            class_samples = []
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_dir, img_file)
                    class_samples.append((img_path, idx, class_name))

            self.samples.extend(class_samples)
            self.class_counts[idx] = len(class_samples)

        print(f"Dataset đã tải: {len(self.samples)} ảnh, {len(classes)} lớp")

        # Tính trọng số cho các lớp để xử lý mất cân bằng dữ liệu
        # Lớp nào ít ảnh hơn sẽ có trọng số cao hơn
        max_count = max(self.class_counts.values())
        self.class_weights = {idx: max_count / count for idx, count in self.class_counts.items()}

    def __len__(self):
        return len(self.samples)

    def get_sample_weights(self):
        # Trả về trọng số cho từng sample, dùng trong WeightedRandomSampler
        weights = []
        for _, label, _ in self.samples:
            weights.append(self.class_weights[label])
        return weights

    def __getitem__(self, idx):
        img_path, label, class_name = self.samples[idx]

        # Tải ảnh
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Tải mask tương ứng
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        mask_dir = os.path.join(self.mask_root, class_name)
        mask_path = os.path.join(mask_dir, f"{img_name}_mask.png")

        if os.path.exists(mask_path):
            try:
                mask = Image.open(mask_path).convert("L") # Chuyển sang ảnh xám
                mask_tensor = transforms.ToTensor()(mask)
                mask_tensor = (mask_tensor > 0.5).float() # Nhị phân hóa mask
            except:
                # Nếu mask lỗi, tạo mask mặc định
                mask_tensor = torch.ones(1, 384, 384)
        else:
            # Nếu không tìm thấy mask, tạo mask mặc định (toàn bộ ảnh)
            mask_tensor = torch.ones(1, 384, 384)

        return image, mask_tensor, label

# Model chính để trích xuất đặc trưng
class LogoEncoder(nn.Module):
    def __init__(self, embedding_dim=512, dropout_rate=0.4):
        super().__init__()
        # Sử dụng ConvNeXt V2 làm backbone, tải từ file local để không cần internet
        model_name = 'convnextv2_base.fcmae_ft_in22k_in1k_384'
        checkpoint_path = 'weights/convnext_base_384/model.safetensors'

        print(f"Tải model offline: {model_name} từ {checkpoint_path}")
        self.backbone = timm.create_model(
            model_name,
            pretrained=False, # Không dùng weight mặc định của timm
            checkpoint_path=checkpoint_path # Mà dùng weight từ file
        )

        # Thay thế lớp classifier gốc bằng một MLP để tạo embedding
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

    def forward(self, x, mask=None, return_features=False):
        # Lấy feature map từ backbone
        features = self.backbone.forward_features(x)  # Shape (Batch, Channels, H, W)

        # Nếu có mask, áp dụng cơ chế attention có trọng số từ mask
        if mask is not None:
            mask = F.interpolate(mask, size=features.shape[2:], mode='nearest') # Resize mask cho vừa feature map
            mask = mask.clamp(0, 1)
            # Tạo attention map đơn giản từ feature
            attention = torch.sigmoid(features.mean(dim=1, keepdim=True))
            # Kết hợp attention và mask
            weight = attention * mask
            # Pooling có trọng số, chỉ tập trung vào vùng logo
            weight_sum = weight.sum(dim=[2,3], keepdim=True) + 1e-6
            pooled = (features * weight).sum(dim=[2,3], keepdim=True) / weight_sum
            pooled = pooled.squeeze(-1).squeeze(-1)
        else:
            # Nếu không có mask, dùng pooling trung bình thông thường
            pooled = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)

        # Đưa qua MLP để tạo embedding cuối cùng
        embeddings = self.embedding(pooled)
        # Chuẩn hóa embedding về vector đơn vị
        normalized_embeddings = F.normalize(embeddings, dim=1)

        if return_features:
            # Trả về cả embedding chưa chuẩn hóa để tính loss
            return normalized_embeddings, embeddings
        return normalized_embeddings

# Định nghĩa các phép biến đổi ảnh
def get_transforms():
    transform = transforms.Compose([
        transforms.ToTensor(), # Chuyển ảnh PIL thành Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Chuẩn hóa theo ImageNet
    ])
    # Ở đây train và val dùng chung transform, có thể tùy biến thêm augmentation cho train
    return transform, transform

# Các hàm thao tác với database embedding (lưu/tải bằng pickle)
def save_embeddings(embeddings, path):
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(embeddings, f)

def load_embeddings(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return []

# Hàm đánh giá chất lượng ảnh, cụ thể là độ nét
def compute_sharpness(img_path):
    # Dùng phương pháp phương sai của Laplacian
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0
        return cv2.Laplacian(img, cv2.CV_64F).var()
    except:
        return 0

# Chọn ảnh support cho one-shot learning
def select_support_images(test_dir, k_support=1):
    support_paths = set()
    for class_name in sorted(os.listdir(test_dir)):
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        # Chọn các file có đuôi _000_color0.jpg làm ảnh support
        image_files = [f for f in os.listdir(class_dir) if f.endswith('_000_color0.jpg')]
        for img_file in image_files[:k_support]:
            img_path = os.path.join(class_dir, img_file)
            support_paths.add(img_path)
        print(f"Lớp {class_name}: Đã chọn {len(image_files[:k_support])} ảnh support")
        for img_file in image_files[:k_support]:
            print(f"  {img_file}")
    return support_paths

# Cập nhật database embedding từ các ảnh support
def update_embedding_database(model, support_paths, db_path, transform, mask_root="logo_output_mask_split"):
    model.eval() # Chuyển model sang chế độ đánh giá
    db = []
    print(f"Cập nhật database embedding với {len(support_paths)} ảnh support...")

    with torch.no_grad(): # Không cần tính gradient
        for img_path in tqdm(sorted(support_paths)):
            try:
                class_name = os.path.basename(os.path.dirname(img_path))
                image = Image.open(img_path).convert('RGB')

                image_tensor = transform(image).unsqueeze(0).to(device) # Thêm chiều batch

                # Tải mask tương ứng
                img_name = os.path.splitext(os.path.basename(img_path))[0]
                mask_dir = os.path.join(mask_root, class_name)
                mask_path = os.path.join(mask_dir, f"{img_name}_mask.png")

                if os.path.exists(mask_path):
                    mask = Image.open(mask_path).convert("L")
                    mask_tensor = transforms.ToTensor()(mask)
                    mask_tensor = (mask_tensor > 0.5).float()
                else:
                    mask_tensor = torch.ones(1, 384, 384)

                mask_tensor = mask_tensor.unsqueeze(0).to(device)

                # Trích xuất embedding
                embedding = model(image_tensor, mask=mask_tensor).cpu().numpy().squeeze()
                db.append((embedding, class_name))

            except Exception as e:
                print(f"Lỗi khi xử lý {img_path}: {e}")
                continue

    save_embeddings(db, db_path)
    print(f"Database đã cập nhật với {len(db)} embeddings")

# Hàm huấn luyện model
def train_model(train_dir, model, epochs=100, batch_size=32, lr=1e-4,
                patience=15, db_path="embedding_db_val.pkl",
                support_paths=None, transform_train=None, transform_val=None):

    train_dataset = LogoDatasetWithMask(train_dir, transform=transform_train, mask_root="logo_output_mask_split/train")

    # Dùng WeightedRandomSampler để cân bằng lớp trong quá trình training
    sample_weights = train_dataset.get_sample_weights()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            sampler=sampler, num_workers=4, pin_memory=True)

    num_classes = len(train_dataset.class_to_idx)
    print(f"Training trên {num_classes} lớp với {len(train_dataset)} ảnh")
    print(f"Phân bố lớp: {train_dataset.class_counts}")

    # Sử dụng loss function đã định nghĩa
    criterion = ImprovedArcFaceLoss(
        feature_dim=512,
        num_classes=num_classes,
        scale=30.0,
        margin=0.5,
        focal_loss=True,
    ).to(device)

    # Dùng AdamW optimizer và CosineAnnealing scheduler
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()), # Train cả model và loss
        lr=lr, weight_decay=5e-4
    )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6 # Learning rate sẽ biến đổi theo chu kỳ cosine
    )

    best_acc = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        # Pha training
        model.train()
        criterion.train()
        total_loss = 0
        total_arcface_loss = 0
        total_center_loss = 0
        total_orth_loss = 0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (images, masks, labels) in enumerate(progress_bar):
            images, masks, labels = images.to(device), masks.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass, lấy cả embedding đã chuẩn hóa và chưa chuẩn hóa
            embeddings, raw_features = model(images, mask=masks, return_features=True)
            # Tính loss trên embedding chưa chuẩn hóa
            loss, loss_dict = criterion(raw_features, labels)

            # Kiểm tra loss có bị NaN không
            if torch.isnan(loss):
                print(f"Phát hiện loss NaN ở batch {batch_idx}, bỏ qua...")
                continue

            loss.backward()

            # Cắt gradient để tránh bùng nổ gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(criterion.parameters(), max_norm=1.0)

            optimizer.step()

            # Ghi nhận các giá trị loss
            total_loss += loss.item()
            total_arcface_loss += loss_dict['arcface_loss']
            total_center_loss += loss_dict['center_loss']
            total_orth_loss += loss_dict['orth_loss']
            num_batches += 1

            # Cập nhật thanh tiến trình
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'ArcFace': f'{loss_dict["arcface_loss"]:.4f}',
                'Center': f'{loss_dict["center_loss"]:.4f}',
                'Orth': f'{loss_dict["orth_loss"]:.6f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })

        if num_batches == 0:
            print("Không có batch hợp lệ, bỏ qua epoch")
            continue

        avg_loss = total_loss / num_batches
        avg_arcface = total_arcface_loss / num_batches
        avg_center = total_center_loss / num_batches
        avg_orth = total_orth_loss / num_batches

        scheduler.step()

        print(f"Tổng kết Epoch {epoch+1}:")
        print(f"  Total Loss: {avg_loss:.4f}")
        print(f"  ArcFace Loss: {avg_arcface:.4f}")
        print(f"  Center Loss: {avg_center:.4f}")
        print(f"  Orthogonal Loss: {avg_orth:.6f}")
        print(f"  Best Val Accuracy hiện tại: {best_acc:.4f}")

        # Pha validation
        if support_paths:
            print("Đang chạy validation...")
            # Cập nhật database với model hiện tại
            update_embedding_database(model, support_paths, db_path, transform_val, mask_root="logo_output_mask_split/val")
            # Đánh giá độ chính xác
            acc = evaluate_one_shot(
                model, "data_split/val", db_path, support_paths,
                transform_val, return_acc_only=True, mask_root="logo_output_mask_split/val"
            )
            print(f"Validation Accuracy (Epoch {epoch+1}): {acc:.4f}")

            # Lưu lại model tốt nhất
            if acc >= best_acc:
                best_acc = acc
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'criterion_state_dict': criterion.state_dict(),
                    'epoch': epoch,
                    'best_acc': best_acc,
                    'class_to_idx': train_dataset.class_to_idx
                }, "arcface_logo_model_best.pth")
                patience_counter = 0
                print(f"Đã lưu model tốt nhất! Accuracy: {best_acc:.4f}")
            else:
                patience_counter += 1

            # Dừng sớm nếu không cải thiện sau một số epoch
            if patience_counter >= patience:
                print(f"Dừng sớm tại epoch {epoch+1}")
                break

    print("Hoàn tất training!")
    # Lưu model cuối cùng
    torch.save({
        'model_state_dict': model.state_dict(),
        'criterion_state_dict': criterion.state_dict(),
        'class_to_idx': train_dataset.class_to_idx
    }, "arcface_logo_model_last.pth")
    print("Đã lưu model cuối cùng với tên 'arcface_logo_model_last.pth'")
    return model

# Hàm đánh giá one-shot
def evaluate_one_shot(model, test_dir, db_path, support_paths,
                      transform, batch_size=32, return_acc_only=False,
                      threshold=0.7, mask_root="logo_output_mask_split"):
    support_paths_set = set(os.path.abspath(p) for p in support_paths) if support_paths else set()
    model.eval()

    test_dataset = LogoDatasetWithMask(test_dir, transform=transform, mask_root=mask_root)

    # Lọc ra các ảnh query (không phải ảnh support)
    query_samples = []
    for img_path, label, class_name in test_dataset.samples:
        if os.path.abspath(img_path) not in support_paths_set:
            query_samples.append((img_path, label, class_name))

    if not query_samples:
        print("Không tìm thấy ảnh query!")
        return 0.0 if return_acc_only else None

    db = load_embeddings(db_path)
    if not db:
        print("Không tìm thấy embedding support!")
        return 0.0 if return_acc_only else None

    db_embeddings = np.array([emb for emb, _ in db])
    db_labels = [label for _, label in db]

    predictions = []
    true_labels = []

    with torch.no_grad():
        # Xử lý theo từng batch để tiết kiệm bộ nhớ
        for i in range(0, len(query_samples), batch_size):
            batch_samples = query_samples[i:i+batch_size]
            batch_tensors = []
            batch_masks = []
            batch_labels = []

            for img_path, label, class_name in batch_samples:
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = transform(img)

                    # Tải mask
                    img_name = os.path.splitext(os.path.basename(img_path))[0]
                    mask_dir = os.path.join(mask_root, class_name)
                    mask_path = os.path.join(mask_dir, f"{img_name}_mask.png")

                    if os.path.exists(mask_path):
                        mask = Image.open(mask_path).convert("L")
                        mask_tensor = transforms.ToTensor()(mask)
                        mask_tensor = (mask_tensor > 0.5).float()
                    else:
                        mask_tensor = torch.ones(1, 384, 384)

                    batch_tensors.append(img_tensor)
                    batch_masks.append(mask_tensor)
                    batch_labels.append(class_name)

                except Exception as e:
                    print(f"Lỗi tải {img_path}: {e}")
                    continue

            if not batch_tensors:
                continue

            batch_tensor = torch.stack(batch_tensors).to(device)
            batch_mask_tensor = torch.stack(batch_masks).to(device)

            # Lấy embedding cho batch
            query_embeddings = model(batch_tensor, mask=batch_mask_tensor).cpu().numpy()

            # Tính độ tương đồng cosine
            similarities = cosine_similarity(query_embeddings, db_embeddings)

            # Dự đoán
            batch_predictions = []
            for sim in similarities:
                best_idx = np.argmax(sim)
                best_score = sim[best_idx]
                # Nếu score vượt ngưỡng thì nhận, không thì là "Unknown"
                if best_score >= threshold:
                    batch_predictions.append(db_labels[best_idx])
                else:
                    batch_predictions.append("Unknown")

            predictions.extend(batch_predictions)
            true_labels.extend(batch_labels)

    accuracy = accuracy_score(true_labels, predictions)

    if return_acc_only:
        return accuracy

    # In ra báo cáo chi tiết
    print(f"\nKết quả đánh giá One-Shot:")
    print(f"Số ảnh query: {len(true_labels)}")
    print(f"Số lớp support: {len(set(db_labels))}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nBáo cáo chi tiết:")
    print(classification_report(true_labels, predictions, zero_division=0))
    return accuracy

# Các hàm tiện ích
def register_logo(image_path, mask_path, class_name, model, db_path="embedding_db.pkl", support_resize_dir="support_resized", support_mask_dir="support_mask"):
    # Đăng ký một logo mới vào database
    model.eval()
    try:
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        # Resize và padding ảnh/mask về 384x384
        padded_img = resize_with_padding(image, target_size=384, fill=(128,128,128))
        padded_mask = resize_with_padding(mask, target_size=384, fill=0)
        # Lưu lại ảnh/mask đã xử lý để debug
        os.makedirs(os.path.join(support_resize_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(support_mask_dir, class_name), exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        padded_img.save(os.path.join(support_resize_dir, class_name, f"{base_name}_resized.jpg"))
        padded_mask.save(os.path.join(support_mask_dir, class_name, f"{base_name}_mask.png"))
        # Chuẩn bị tensor cho model
        image_tensor = transforms.ToTensor()(padded_img)
        image_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image_tensor)
        image_tensor = image_tensor.unsqueeze(0).to(device)
        mask_tensor = transforms.ToTensor()(padded_mask).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model(image_tensor, mask=mask_tensor).cpu().numpy().squeeze()
        # Thêm embedding mới vào database
        db = load_embeddings(db_path)
        db.append((embedding, class_name))
        save_embeddings(db, db_path)
        print(f"Đăng ký thành công logo: {class_name}")
        print(f"Đã lưu ảnh support đã resize tới {os.path.join(support_resize_dir, class_name)}")
        print(f"Đã lưu mask tới {os.path.join(support_mask_dir, class_name)}")
    except Exception as e:
        print(f"Lỗi đăng ký logo {class_name}: {e}")

def resize_with_padding(pil_img, target_size=384, fill=128):
    # Resize ảnh nhưng giữ nguyên tỷ lệ, phần thiếu sẽ được đệm (pad)
    w, h = pil_img.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    # Dùng thuật toán khác nhau cho ảnh và mask
    if pil_img.mode == 'L':  # mask
        resized_img = pil_img.resize((new_w, new_h), Image.NEAREST) # Giữ cạnh sắc nét
    else:                     # ảnh RGB
        resized_img = pil_img.resize((new_w, new_h), Image.BICUBIC) # Mịn hơn
    pad_w = target_size - new_w
    pad_h = target_size - new_h
    if isinstance(fill, int):
        fill_color = fill
    else:
        fill_color = tuple(fill)
    padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
    padded_img = ImageOps.expand(resized_img, padding, fill=fill_color)
    return padded_img

def identify_logo(image_path, mask_path=None, model=None, threshold=0.5, db_path="embedding_db.pkl", top_k=3, save_query_dir="query_debug"):
    # Nhận diện một logo đơn lẻ
    import time
    start_time = time.time()
    model.eval()
    try:
        # Xử lý ảnh query
        image = Image.open(image_path).convert('RGB')
        padded_img = resize_with_padding(image, target_size=384, fill=(128,128,128))
        img_tensor = transforms.ToTensor()(padded_img)
        img_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_tensor)
        img_tensor = img_tensor.unsqueeze(0).to(device)

        # Xử lý mask query (nếu có)
        if mask_path is not None and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")
            padded_mask = resize_with_padding(mask, target_size=384, fill=0)
            mask_tensor = transforms.ToTensor()(padded_mask).unsqueeze(0).to(device)
        else:
            mask_tensor = torch.ones(1, 1, 384, 384).to(device)  # Mask toàn 1

        # Lưu ảnh/mask đã xử lý để debug
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
            return f"Không tìm thấy kết quả khớp (kết quả tốt nhất={best_class}: {best_score:.4f})"
    except Exception as e:
        return f"Lỗi trong quá trình nhận diện: {e}"


# Hàm chia dataset
# Chia theo class, không phải theo ảnh
# Đảm bảo các lớp trong tập train, val, test là hoàn toàn khác nhau
def split_dataset_by_class(input_dir, output_dir, train_ratio=0.8,
                          val_ratio=0.1, test_ratio=0.1, seed=42,
                          mask_root=None, mask_output_dir=None):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    all_classes = sorted([d for d in os.listdir(input_dir)
                         if os.path.isdir(os.path.join(input_dir, d))])

    random.seed(seed)
    random.shuffle(all_classes)

    num_total = len(all_classes)
    num_train = int(train_ratio * num_total)
    num_val = int(val_ratio * num_total)

    train_classes = all_classes[:num_train]
    val_classes = all_classes[num_train:num_train + num_val]
    test_classes = all_classes[num_train + num_val:]

    for split_name, split_classes in [('train', train_classes),
                                     ('val', val_classes),
                                     ('test', test_classes)]:
        for class_name in split_classes:
            src_dir = os.path.join(input_dir, class_name)
            dst_dir = os.path.join(output_dir, split_name, class_name)
            os.makedirs(dst_dir, exist_ok=True)

            # Sao chép ảnh
            for file_name in os.listdir(src_dir):
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    shutil.copy2(os.path.join(src_dir, file_name),
                               os.path.join(dst_dir, file_name))

            # Sao chép mask nếu có
            if mask_root and mask_output_dir:
                mask_src_dir = os.path.join(mask_root, class_name)
                mask_dst_dir = os.path.join(mask_output_dir, split_name, class_name)
                if os.path.exists(mask_src_dir):
                    os.makedirs(mask_dst_dir, exist_ok=True)
                    for mask_file in os.listdir(mask_src_dir):
                        if mask_file.lower().endswith('_mask.png'):
                            shutil.copy2(os.path.join(mask_src_dir, mask_file),
                                         os.path.join(mask_dst_dir, mask_file))

    print(f"Hoàn tất chia dataset:")
    print(f"  Train: {len(train_classes)} lớp")
    print(f"  Val: {len(val_classes)} lớp")
    print(f"  Test: {len(test_classes)} lớp")

# Hàm thực thi chính
def main():
    print("=== Hệ thống nhận diện Logo One-Shot ===")

    # Khởi tạo model
    model = LogoEncoder(embedding_dim=512, dropout_rate=0.5).to(device)
    transform_train, transform_val = get_transforms()

    # Kiểm tra và chia dataset nếu cần
    if not os.path.exists("data_split"):
        print("Đang chia dataset...")
        split_dataset_by_class(
            "logo_output_image_bgcolor_384",
            "data_split",
            mask_root="logo_output_mask_384",
            mask_output_dir="logo_output_mask_split"
        )

    # Chọn ảnh support cho tập validation
    print("Chọn ảnh support cho validation...")
    support_val_paths = select_support_images("data_split/val", k_support=1)

    # Huấn luyện model
    print("Bắt đầu training...")
    model = train_model(
        train_dir="data_split/train",
        model=model,
        epochs=100,
        batch_size=28,
        lr=5e-5,
        patience=15,
        db_path="embedding_db_val.pkl",
        support_paths=support_val_paths,
        transform_train=transform_train,
        transform_val=transform_val,
    )

    # Tải model tốt nhất đã lưu
    print("Tải model tốt nhất...")
    try:
        checkpoint = torch.load("arcface_logo_model_best.pth", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Đã tải model với accuracy tốt nhất: {checkpoint.get('best_acc', 'Không rõ'):.4f}")
    except FileNotFoundError:
        print("Không tìm thấy model tốt nhất, sử dụng model hiện tại")

    model.eval()

    # Đánh giá cuối cùng trên tập test
    print("Chuẩn bị đánh giá trên tập test...")
    support_test_paths = select_support_images("data_split/test", k_support=1)
    # Cập nhật database cho tập test
    update_embedding_database(
        model,
        support_test_paths,
        "embedding_db_test.pkl",
        transform_val,
        mask_root="logo_output_mask_split/test"
    )

    print("Đánh giá cuối cùng trên tập test:")
    test_accuracy = evaluate_one_shot(
        model,
        "data_split/test",
        "embedding_db_test.pkl",
        support_test_paths,
        transform_val,
        mask_root="logo_output_mask_split/test"
    )

    print(f"\nAccuracy cuối cùng trên tập Test: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()