import os
import cv2
import pickle
import base64
from io import BytesIO
from PIL import Image, ImageOps

def load_embeddings(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return []

def save_embeddings(embeddings, path):
    with open(path, 'wb') as f:
        pickle.dump(embeddings, f)

def resize_with_padding(pil_img, target_size=380, fill=(128,128,128)):
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
    try:
        if not os.path.exists(path):
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
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode('utf-8')
