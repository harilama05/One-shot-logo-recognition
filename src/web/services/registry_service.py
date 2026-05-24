import os
import traceback
import torch
from PIL import Image
from torchvision import transforms

from config import Config
from utils import load_embeddings, save_embeddings, resize_with_padding
from services.video_service import video_processor

class LogoRegistry:
    @staticmethod
    def register_support_folder(support_dir, mask_dir=None):
        try:
            if not os.path.exists(support_dir):
                print(f"Thư mục support không tồn tại: {support_dir}")
                return False

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
                    
                    success = LogoRegistry.register_single_logo(img_path, mask_path, class_name)
                    if success:
                        total_registered += 1

            print(f"Đã đăng ký {total_registered} logo từ {len(class_dirs)} class")
            return total_registered > 0

        except Exception as e:
            print(f"Lỗi đăng ký thư mục support: {e}")
            traceback.print_exc()
            return False

    @staticmethod
    def register_single_logo(image_path, mask_path, class_name):
        try:
            if not video_processor.recog_model:
                print("Model nhận diện chưa được tải")
                return False

            video_processor.recog_model.eval()

            db = load_embeddings(Config.EMBED_DB_PATH)
            existing_classes = [c for _, c in db]
            orig_class_name = class_name
            idx = 1
            while class_name in existing_classes:
                class_name = f"{orig_class_name}_{idx}"
                idx += 1

            image = Image.open(image_path).convert('RGB')
            padded_img = resize_with_padding(image, target_size=380)

            if mask_path and os.path.exists(mask_path):
                mask = Image.open(mask_path).convert('L')
                padded_mask = resize_with_padding(mask, target_size=380, fill=0)
            else:
                padded_mask = Image.new("L", (380, 380), 255)
                print(f"Cảnh báo: Không có mask cho {class_name}, dùng mask trắng")

            image_tensor = transforms.ToTensor()(padded_img)
            image_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image_tensor)
            image_tensor = image_tensor.unsqueeze(0).to(Config.DEVICE)
            mask_tensor = transforms.ToTensor()(padded_mask).unsqueeze(0).to(Config.DEVICE)

            with torch.no_grad():
                embedding = video_processor.recog_model(image_tensor, mask=mask_tensor).cpu().numpy().squeeze()

            db.append((embedding, class_name))
            save_embeddings(db, Config.EMBED_DB_PATH)

            support_resize_dir = "support_resized_test"
            support_mask_dir = "support_mask_test"
            os.makedirs(os.path.join(support_resize_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(support_mask_dir, class_name), exist_ok=True)
            base_name = os.path.splitext(os.path.basename(image_path))[0]

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
            
            print(f"Đăng ký thành công logo: {class_name}")
            return True

        except Exception as e:
            print(f"Lỗi đăng ký logo {class_name}: {e}")
            traceback.print_exc()
            return False
