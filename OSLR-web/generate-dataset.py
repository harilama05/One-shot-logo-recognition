import os
import random
from PIL import Image, ImageEnhance, ImageDraw
import numpy as np
from torchvision import transforms
from tqdm import tqdm
import cv2
from io import BytesIO
import pandas as pd

# ---------- Luminance ----------
def luminance(rgb):
    r, g, b = rgb
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def get_contrasting_colors(min_lum_diff=100):
    while True:
        bright = tuple(np.random.randint(180, 256, size=3))
        dark = tuple(np.random.randint(0, 120, size=3))
        if abs(luminance(bright) - luminance(dark)) >= min_lum_diff:
            return dark, bright

def full_color_variant_logo(img_pil, mask_pil):
    img1 = ImageEnhance.Brightness(img_pil).enhance(random.uniform(1.1, 1.3))
    img1 = ImageEnhance.Color(img1).enhance(random.uniform(1.0, 1.2))
    img_np = np.array(img1.convert('RGB'))
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    hue_shift_degree = random.choice([random.randint(20, 40), -random.randint(20, 40)])
    hsv[..., 0] = (hsv[..., 0].astype(int) + hue_shift_degree) % 180
    shifted = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    mask = np.array(mask_pil)
    shifted[mask == 0] = [255, 255, 255]
    
    # Xử lý logo đen - sửa lại logic
    logo_mask = mask > 0
    logo_rgb = shifted[logo_mask]
    
    # Tìm pixel đen: tổng RGB < 120 (thay vì < 40 cho tất cả kênh)
    dark_pixels = np.sum(logo_rgb, axis=1) < 120
    
    if dark_pixels.any():
        # Tạo màu ngẫu nhiên sáng
        new_color = np.array([
            random.randint(100, 255),
            random.randint(100, 255), 
            random.randint(100, 255)
        ], dtype=np.uint8)
        
        # Áp dụng màu mới cho pixel đen
        logo_rgb[dark_pixels] = new_color
        shifted[logo_mask] = logo_rgb
    
    img_color = Image.fromarray(shifted)
    if random.random() < 0.2:
        gray = img_color.convert("L").convert("RGB")
        return gray
    return img_color

def generate_10_color_variants_logo(img_pil, mask_pil):
    variants = [img_pil]
    for _ in range(8):
        var = full_color_variant_logo(img_pil, mask_pil)
        variants.append(var)
    # Always add one grayscale variant
    gray = img_pil.convert("L").convert("RGB")
    variants.append(gray)
    return variants

def random_perspective_pair(img: Image.Image, mask: Image.Image, direction=None):
    # img = img.resize((384, 384), Image.Resampling.LANCZOS)
    # mask = mask.resize((384, 384), Image.Resampling.NEAREST)
    w, h = img.size
    src_pts = [(0, 0), (w, 0), (w, h), (0, h)]
    shrink = random.uniform(0.3, 0.5)
    height_shrink = random.uniform(0.2, 0.4)
    if direction is None:
        direction = random.choice(['top', 'bottom', 'left', 'right', 'tl', 'tr', 'bl', 'br'])
    if direction == 'top':
        dst_pts = [
            (w * (1 - shrink) / 2, h * height_shrink),
            (w * (1 + shrink) / 2, h * height_shrink),
            (w, h),
            (0, h),
        ]
    elif direction == 'bottom':
        dst_pts = [
            (0, 0),
            (w, 0),
            (w * (1 + shrink) / 2, h * (1 - height_shrink)),
            (w * (1 - shrink) / 2, h * (1 - height_shrink)),
        ]
    elif direction == 'left':
        dst_pts = [
            (w * height_shrink, 0),
            (w, 0),
            (w, h),
            (w * height_shrink, h),
        ]
    elif direction == 'right':
        dst_pts = [
            (0, 0),
            (w * (1 - height_shrink), 0),
            (w * (1 - height_shrink), h),
            (0, h),
        ]
    elif direction == 'tl':
        dst_pts = [
            (w * height_shrink, h * height_shrink),
            (w, 0),
            (w, h),
            (0, h),
        ]
    elif direction == 'tr':
        dst_pts = [
            (0, 0),
            (w * (1 - height_shrink), h * height_shrink),
            (w, h),
            (0, h),
        ]
    elif direction == 'bl':
        dst_pts = [
            (0, 0),
            (w, 0),
            (w, h),
            (w * height_shrink, h * (1 - height_shrink)),
        ]
    elif direction == 'br':
        dst_pts = [
            (0, 0),
            (w, 0),
            (w * (1 - height_shrink), h * (1 - height_shrink)),
            (0, h),
        ]
    else:
        dst_pts = src_pts
    coeffs = find_perspective_coeffs(src_pts, dst_pts)
    img_warp = img.transform((w, h), Image.PERSPECTIVE, coeffs, resample=Image.Resampling.BICUBIC)
    mask_warp = mask.transform((w, h), Image.PERSPECTIVE, coeffs, resample=Image.Resampling.NEAREST)
    # Binarize mask after transform
    mask_warp = mask_warp.point(lambda p: 255 if p >= 128 else 0)
    return img_warp, mask_warp

def random_perspective(img: Image.Image, direction=None) -> Image.Image:
    # For backward compatibility, keep this for single image
    img = img.resize((384, 384), Image.Resampling.LANCZOS)
    w, h = img.size
    src_pts = [(0, 0), (w, 0), (w, h), (0, h)]
    shrink = random.uniform(0.3, 0.5)
    height_shrink = random.uniform(0.2, 0.4)
    if direction is None:
        direction = random.choice(['top', 'bottom', 'left', 'right', 'tl', 'tr', 'bl', 'br'])
    if direction == 'top':
        dst_pts = [
            (w * (1 - shrink) / 2, h * height_shrink),
            (w * (1 + shrink) / 2, h * height_shrink),
            (w, h),
            (0, h),
        ]
    elif direction == 'bottom':
        dst_pts = [
            (0, 0),
            (w, 0),
            (w * (1 + shrink) / 2, h * (1 - height_shrink)),
            (w * (1 - shrink) / 2, h * (1 - height_shrink)),
        ]
    elif direction == 'left':
        dst_pts = [
            (w * height_shrink, 0),
            (w, 0),
            (w, h),
            (w * height_shrink, h),
        ]
    elif direction == 'right':
        dst_pts = [
            (0, 0),
            (w * (1 - height_shrink), 0),
            (w * (1 - height_shrink), h),
            (0, h),
        ]
    elif direction == 'tl':
        dst_pts = [
            (w * height_shrink, h * height_shrink),
            (w, 0),
            (w, h),
            (0, h),
        ]
    elif direction == 'tr':
        dst_pts = [
            (0, 0),
            (w * (1 - height_shrink), h * height_shrink),
            (w, h),
            (0, h),
        ]
    elif direction == 'bl':
        dst_pts = [
            (0, 0),
            (w, 0),
            (w, h),
            (w * height_shrink, h * (1 - height_shrink)),
        ]
    elif direction == 'br':
        dst_pts = [
            (0, 0),
            (w, 0),
            (w * (1 - height_shrink), h * (1 - height_shrink)),
            (0, h),
        ]
    else:
        dst_pts = src_pts
    coeffs = find_perspective_coeffs(src_pts, dst_pts)
    return img.transform((w, h), Image.PERSPECTIVE, coeffs, resample=Image.Resampling.BICUBIC)

def find_perspective_coeffs(src_pts, dst_pts):
    M = []
    for (x,y),(u,v) in zip(src_pts, dst_pts):
        M.append([x,y,1,0,0,0,-u*x,-u*y])
        M.append([0,0,0,x,y,1,-v*x,-v*y])
    M = np.asarray(M, dtype=np.float64)
    b = np.asarray(dst_pts, dtype=np.float64).reshape(8)
    coeffs, *_ = np.linalg.lstsq(M, b, rcond=None)
    return coeffs

def motion_blur(img):
    arr = np.array(img)
    ksize = random.choice([9, 11])
    kernel = np.zeros((ksize, ksize))
    direction = random.choice(['horizontal', 'vertical', 'diagonal'])
    if direction == 'horizontal':
        kernel[ksize // 2, :] = np.ones(ksize)
    elif direction == 'vertical':
        kernel[:, ksize // 2] = np.ones(ksize)
    else:
        np.fill_diagonal(kernel, 1)
    kernel /= kernel.sum()
    blurred = cv2.filter2D(arr, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    return Image.fromarray(blurred)

def gaussian_blur(img):
    ksize = random.choice([5, 7])
    return transforms.GaussianBlur(ksize)(img)

def occlusion(img):
    img = img.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for _ in range(random.randint(1, 3)):
        occ_w = random.randint(int(w * 0.2), int(w * 0.4))
        occ_h = random.randint(int(h * 0.2), int(h * 0.4))
        x1 = random.randint(0, w - occ_w)
        y1 = random.randint(0, h - occ_h)
        color = tuple(random.randint(0, 255) for _ in range(3))
        draw.rectangle([x1, y1, x1 + occ_w, y1 + occ_h], fill=color)
    return img

def occlusion_both(img, mask):
    img = img.copy()
    mask = mask.copy()
    draw_img = ImageDraw.Draw(img)
    draw_mask = ImageDraw.Draw(mask)
    w, h = img.size
    for _ in range(random.randint(1, 3)):
        occ_w = random.randint(int(w * 0.2), int(w * 0.4))
        occ_h = random.randint(int(h * 0.2), int(h * 0.4))
        x1 = random.randint(0, w - occ_w)
        y1 = random.randint(0, h - occ_h)
        color = tuple(random.randint(0, 255) for _ in range(3))
        draw_img.rectangle([x1, y1, x1 + occ_w, y1 + occ_h], fill=color)
        draw_mask.rectangle([x1, y1, x1 + occ_w, y1 + occ_h], fill=0)
    return img, mask

def crop_part(img):
    w, h = img.size
    ratio = random.uniform(0.5, 0.7)
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    left = random.randint(0, w - new_w)
    top = random.randint(0, h - new_h)
    cropped = img.crop((left, top, left + new_w, top + new_h)).resize((w, h))
    arr = np.array(cropped).astype(np.float32)
    noise = np.random.normal(0, 5, arr.shape)
    arr += noise
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def add_noise(img):
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, 10, arr.shape)
    arr += noise
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def jpeg_compress(img):
    buffer = BytesIO()
    img.save(buffer, format='JPEG', quality=random.randint(25, 60))  # Use 25–60 for jpeg augment
    buffer.seek(0)
    return Image.open(buffer).copy()

def rotate_augment(img, mask):
    # Xoay ngẫu nhiên trong khoảng -15 đến +15 độ, giữ nguyên kích thước, nền trắng
    angle = random.uniform(-15, 15)
    img_rot = img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(255,255,255))
    mask_rot = mask.rotate(angle, resample=Image.Resampling.NEAREST, expand=False, fillcolor=0)
    # Binarize lại mask
    mask_rot = mask_rot.point(lambda p: 255 if p >= 128 else 0)
    return img_rot, mask_rot

def radial_blur(img, center=None, strength=1.5):
    """Tạo hiệu ứng blur xoay tròn như khi camera quay"""
    arr = np.array(img)
    h, w = arr.shape[:2]
    if center is None:
        center = (w//2, h//2)
    
    # Tạo grid tọa độ
    y, x = np.ogrid[:h, :w]
    cx, cy = center
    
    # Tính khoảng cách từ tâm
    distances = np.sqrt((x - cx)**2 + (y - cy)**2)
    max_dist = np.sqrt(cx**2 + cy**2)
    
    # Tạo kernel blur theo khoảng cách
    blur_kernel_size = max(3, int(strength * 5))
    if blur_kernel_size % 2 == 0:
        blur_kernel_size += 1
    
    # Blur nhiều lớp với cường độ khác nhau
    result = arr.copy()
    for i in range(3):
        mask = (distances > i * max_dist / 4).astype(np.float32)
        blurred = cv2.GaussianBlur(arr, (blur_kernel_size, blur_kernel_size), 0)
        result = result * (1 - mask[:, :, np.newaxis]) + blurred * mask[:, :, np.newaxis]
    
    return Image.fromarray(result.astype(np.uint8))

def lens_distortion(img, mask, distortion_coeff=0.3):
    """Tạo biến dạng ống kính barrel/pincushion"""
    arr = np.array(img)
    mask_arr = np.array(mask)
    h, w = arr.shape[:2]
    
    # Tạo grid tọa độ chuẩn hóa
    x, y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    r = np.sqrt(x**2 + y**2)
    
    # Áp dụng distortion (barrel nếu coeff > 0, pincushion nếu < 0)
    r_distorted = r * (1 + distortion_coeff * r**2)
    
    # Tính tọa độ mới
    theta = np.arctan2(y, x)
    x_new = r_distorted * np.cos(theta)
    y_new = r_distorted * np.sin(theta)
    
    # Chuyển về pixel coordinates
    x_map = ((x_new + 1) * w / 2).astype(np.float32)
    y_map = ((y_new + 1) * h / 2).astype(np.float32)
    
    # Áp dụng distortion
    distorted_img = cv2.remap(arr, x_map, y_map, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    distorted_mask = cv2.remap(mask_arr, x_map, y_map, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    # Binarize mask
    distorted_mask = np.where(distorted_mask >= 128, 255, 0).astype(np.uint8)
    
    return Image.fromarray(distorted_img), Image.fromarray(distorted_mask)

def severe_rotation_crop(img, mask):
    """Xoay mạnh và crop một phần"""
    angle = random.uniform(-45, 45)
    # Xoay với expand=True để không mất thông tin
    img_rot = img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(255,255,255))
    mask_rot = mask.rotate(angle, resample=Image.Resampling.NEAREST, expand=True, fillcolor=0)
    
    # Crop ngẫu nhiên 60-80% diện tích
    w, h = img_rot.size
    crop_ratio = random.uniform(0.6, 0.8)
    new_w = int(w * crop_ratio)
    new_h = int(h * crop_ratio)
    
    left = random.randint(0, w - new_w)
    top = random.randint(0, h - new_h)
    
    img_cropped = img_rot.crop((left, top, left + new_w, top + new_h))
    mask_cropped = mask_rot.crop((left, top, left + new_w, top + new_h))
    
    # Resize về 384x384
    img_final = img_cropped.resize((384, 384), Image.Resampling.LANCZOS)
    mask_final = mask_cropped.resize((384, 384), Image.Resampling.NEAREST)
    mask_final = mask_final.point(lambda p: 255 if p >= 128 else 0)
    
    return img_final, mask_final

def multi_occlusion_rotation(img, mask):
    """Kết hợp nhiều occlusion + xoay nhẹ"""
    # Xoay nhẹ trước
    angle = random.uniform(-20, 20)
    img_rot = img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(255,255,255))
    mask_rot = mask.rotate(angle, resample=Image.Resampling.NEAREST, expand=False, fillcolor=0)
    
    # Tạo nhiều occlusion với hình dạng khác nhau
    draw_img = ImageDraw.Draw(img_rot)
    draw_mask = ImageDraw.Draw(mask_rot)
    w, h = img_rot.size
    
    for _ in range(random.randint(2, 4)):
        shape_type = random.choice(['rectangle', 'ellipse', 'polygon'])
        
        if shape_type == 'rectangle':
            occ_w = random.randint(int(w * 0.15), int(w * 0.3))
            occ_h = random.randint(int(h * 0.15), int(h * 0.3))
            x1 = random.randint(0, w - occ_w)
            y1 = random.randint(0, h - occ_h)
            color = tuple(random.randint(0, 255) for _ in range(3))
            draw_img.rectangle([x1, y1, x1 + occ_w, y1 + occ_h], fill=color)
            draw_mask.rectangle([x1, y1, x1 + occ_w, y1 + occ_h], fill=0)
            
        elif shape_type == 'ellipse':
            occ_w = random.randint(int(w * 0.1), int(w * 0.25))
            occ_h = random.randint(int(h * 0.1), int(h * 0.25))
            x1 = random.randint(0, w - occ_w)
            y1 = random.randint(0, h - occ_h)
            color = tuple(random.randint(0, 255) for _ in range(3))
            draw_img.ellipse([x1, y1, x1 + occ_w, y1 + occ_h], fill=color)
            draw_mask.ellipse([x1, y1, x1 + occ_w, y1 + occ_h], fill=0)
            
        else:  # polygon
            center_x = random.randint(int(w * 0.2), int(w * 0.8))
            center_y = random.randint(int(h * 0.2), int(h * 0.8))
            radius = random.randint(int(min(w, h) * 0.05), int(min(w, h) * 0.15))
            sides = random.randint(3, 6)
            
            points = []
            for i in range(sides):
                angle_poly = 2 * np.pi * i / sides
                x = center_x + radius * np.cos(angle_poly)
                y = center_y + radius * np.sin(angle_poly)
                points.append((x, y))
            
            color = tuple(random.randint(0, 255) for _ in range(3))
            draw_img.polygon(points, fill=color)
            draw_mask.polygon(points, fill=0)
    
    mask_rot = mask_rot.point(lambda p: 255 if p >= 128 else 0)
    return img_rot, mask_rot

def motion_blur_distortion(img, mask):
    """Kết hợp motion blur + lens distortion"""
    # Motion blur trước
    img_blur = motion_blur(img)
    
    # Sau đó lens distortion
    distortion_coeff = random.uniform(-0.2, 0.3)
    img_dist, mask_dist = lens_distortion(img_blur, mask, distortion_coeff)
    
    return img_dist, mask_dist

def extreme_perspective_occlusion(img, mask):
    """Perspective mạnh + occlusion + nhiễu - đã giảm độ khắc nghiệt"""
    # Perspective vừa phải hơn
    w, h = img.size
    src_pts = [(0, 0), (w, 0), (w, h), (0, h)]
    shrink = random.uniform(0.4, 0.6)  # Giảm từ 0.6-0.8
    height_shrink = random.uniform(0.25, 0.45)  # Giảm từ 0.4-0.6
    
    direction = random.choice(['top', 'bottom', 'left', 'right', 'tl', 'tr', 'bl', 'br'])
    
    if direction == 'top':
        dst_pts = [
            (w * (1 - shrink) / 2, h * height_shrink),
            (w * (1 + shrink) / 2, h * height_shrink),
            (w, h), (0, h)
        ]
    elif direction == 'bottom':
        dst_pts = [
            (0, 0), (w, 0),
            (w * (1 + shrink) / 2, h * (1 - height_shrink)),
            (w * (1 - shrink) / 2, h * (1 - height_shrink))
        ]
    elif direction == 'left':
        dst_pts = [
            (w * height_shrink, 0), (w, 0),
            (w, h), (w * height_shrink, h)
        ]
    elif direction == 'right':
        dst_pts = [
            (0, 0), (w * (1 - height_shrink), 0),
            (w * (1 - height_shrink), h), (0, h)
        ]
    else:  # corner cases
        if direction == 'tl':
            dst_pts = [(w * height_shrink, h * height_shrink), (w, 0), (w, h), (0, h)]
        elif direction == 'tr':
            dst_pts = [(0, 0), (w * (1 - height_shrink), h * height_shrink), (w, h), (0, h)]
        elif direction == 'bl':
            dst_pts = [(0, 0), (w, 0), (w, h), (w * height_shrink, h * (1 - height_shrink))]
        else:  # br
            dst_pts = [(0, 0), (w, 0), (w * (1 - height_shrink), h * (1 - height_shrink)), (0, h)]
    
    coeffs = find_perspective_coeffs(src_pts, dst_pts)
    img_persp = img.transform((w, h), Image.PERSPECTIVE, coeffs, resample=Image.Resampling.BICUBIC)
    mask_persp = mask.transform((w, h), Image.PERSPECTIVE, coeffs, resample=Image.Resampling.NEAREST)
    
    # Thêm occlusion nhẹ hơn - chỉ 1-2 hình thay vì nhiều
    draw_img = ImageDraw.Draw(img_persp)
    draw_mask = ImageDraw.Draw(mask_persp)
    
    for _ in range(random.randint(1, 2)):  # Giảm từ random nhiều
        occ_w = random.randint(int(w * 0.1), int(w * 0.25))  # Giảm kích thước
        occ_h = random.randint(int(h * 0.1), int(h * 0.25))
        x1 = random.randint(0, w - occ_w)
        y1 = random.randint(0, h - occ_h)
        color = tuple(random.randint(0, 255) for _ in range(3))
        draw_img.rectangle([x1, y1, x1 + occ_w, y1 + occ_h], fill=color)
        draw_mask.rectangle([x1, y1, x1 + occ_w, y1 + occ_h], fill=0)
    
    # Giảm noise
    arr = np.array(img_persp).astype(np.float32)
    noise = np.random.normal(0, 3, arr.shape)  # Giảm từ 8 xuống 3
    arr += noise
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img_final = Image.fromarray(arr)
    
    mask_persp = mask_persp.point(lambda p: 255 if p >= 128 else 0)
    return img_final, mask_persp

def apply_augmentation(img, mask, aug_type):
    # Remove unconditional resize here!
    # img = img.resize((384, 384))
    # mask = mask.resize((384, 384))
    if aug_type == "perspective":
        img = img.resize((384, 384), Image.Resampling.LANCZOS)
        mask = mask.resize((384, 384), Image.Resampling.NEAREST)
        direction = random.choice(['top', 'bottom', 'left', 'right', 'tl', 'tr', 'bl', 'br'])
        img_warp, mask_warp = random_perspective_pair(img, mask, direction)
        return img_warp, mask_warp
    elif aug_type == "motion":
        img = img.resize((384, 384), Image.Resampling.LANCZOS)
        mask = mask.resize((384, 384), Image.Resampling.NEAREST)
        return motion_blur(img), mask
    elif aug_type == "blur":
        img = img.resize((384, 384), Image.Resampling.LANCZOS)
        mask = mask.resize((384, 384), Image.Resampling.NEAREST)
        return gaussian_blur(img), mask
    elif aug_type == "occlusion":
        img = img.resize((384, 384), Image.Resampling.LANCZOS)
        mask = mask.resize((384, 384), Image.Resampling.NEAREST)
        return occlusion_both(img, mask)
    elif aug_type == "crop_part":
        img = img.resize((384, 384), Image.Resampling.LANCZOS)
        mask = mask.resize((384, 384), Image.Resampling.NEAREST)
        w, h = img.size
        ratio = random.uniform(0.5, 0.7)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        left = random.randint(0, w - new_w)
        top = random.randint(0, h - new_h)
        cropped_img = img.crop((left, top, left + new_w, top + new_h)).resize((w, h))
        cropped_mask = mask.crop((left, top, left + new_w, top + new_h)).resize((w, h), Image.Resampling.NEAREST)
        arr = np.array(cropped_img).astype(np.float32)
        noise = np.random.normal(0, 5, arr.shape)
        arr += noise
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        cropped_mask = cropped_mask.point(lambda p: 255 if p >= 128 else 0)
        return Image.fromarray(arr), cropped_mask
    elif aug_type == "noise":
        img = img.resize((384, 384), Image.Resampling.LANCZOS)
        mask = mask.resize((384, 384), Image.Resampling.NEAREST)
        return add_noise(img), mask
    elif aug_type == "jpeg":
        img = img.resize((384, 384), Image.Resampling.LANCZOS)
        mask = mask.resize((384, 384), Image.Resampling.NEAREST)
        return jpeg_compress(img), mask
    elif aug_type == "crop_persp":
        img = img.resize((384, 384), Image.Resampling.LANCZOS)
        mask = mask.resize((384, 384), Image.Resampling.NEAREST)
        w, h = img.size
        ratio = random.uniform(0.5, 0.7)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        left = random.randint(0, w - new_w)
        top = random.randint(0, h - new_h)
        cropped_img = img.crop((left, top, left + new_w, top + new_h)).resize((w, h), Image.Resampling.LANCZOS)
        cropped_mask = mask.crop((left, top, left + new_w, top + new_h)).resize((w, h), Image.Resampling.NEAREST)
        cropped_img = ImageEnhance.Color(cropped_img).enhance(0.85)
        cropped_img = ImageEnhance.Sharpness(cropped_img).enhance(0.75)
        direction = random.choice(['top', 'bottom', 'left', 'right', 'tl', 'tr', 'bl', 'br'])
        img_warp, mask_warp = random_perspective_pair(cropped_img, cropped_mask, direction)
        return img_warp, mask_warp
    elif aug_type == "motion_persp":
        img = img.resize((384, 384), Image.Resampling.LANCZOS)
        mask = mask.resize((384, 384), Image.Resampling.NEAREST)
        img_blur = motion_blur(img)
        img_blur = ImageEnhance.Color(img_blur).enhance(0.85)
        img_blur = ImageEnhance.Sharpness(img_blur).enhance(0.75)
        direction = random.choice(['top', 'bottom', 'left', 'right', 'tl', 'tr', 'bl', 'br'])
        img_warp, mask_warp = random_perspective_pair(img_blur, mask, direction)
        return img_warp, mask_warp
    elif aug_type == "occ_persp":
        img = img.resize((384, 384), Image.Resampling.LANCZOS)
        mask = mask.resize((384, 384), Image.Resampling.NEAREST)
        occ_img, occ_mask = occlusion_both(img, mask)
        occ_img = ImageEnhance.Color(occ_img).enhance(0.85)
        occ_img = ImageEnhance.Sharpness(occ_img).enhance(0.75)
        direction = random.choice(['top', 'bottom', 'left', 'right', 'tl', 'tr', 'bl', 'br'])
        img_warp, mask_warp = random_perspective_pair(occ_img, occ_mask, direction)
        return img_warp, mask_warp
    elif aug_type == "rotate":
        img = img.resize((384, 384), Image.Resampling.LANCZOS)
        mask = mask.resize((384, 384), Image.Resampling.NEAREST)
        return rotate_augment(img, mask)
    elif aug_type == "radial_blur":
        img = img.resize((384, 384), Image.Resampling.LANCZOS)
        mask = mask.resize((384, 384), Image.Resampling.NEAREST)
        return radial_blur(img), mask
    elif aug_type == "lens_distort":
        img = img.resize((384, 384), Image.Resampling.LANCZOS)
        mask = mask.resize((384, 384), Image.Resampling.NEAREST)
        distortion_coeff = random.uniform(-0.3, 0.4)
        return lens_distortion(img, mask, distortion_coeff)
    elif aug_type == "severe_rot_crop":
        img = img.resize((384, 384), Image.Resampling.LANCZOS)
        mask = mask.resize((384, 384), Image.Resampling.NEAREST)
        return severe_rotation_crop(img, mask)
    elif aug_type == "multi_occ_rot":
        img = img.resize((384, 384), Image.Resampling.LANCZOS)
        mask = mask.resize((384, 384), Image.Resampling.NEAREST)
        return multi_occlusion_rotation(img, mask)
    elif aug_type == "motion_distort":
        img = img.resize((384, 384), Image.Resampling.LANCZOS)
        mask = mask.resize((384, 384), Image.Resampling.NEAREST)
        return motion_blur_distortion(img, mask)
    elif aug_type == "extreme_persp_occ":
        img = img.resize((384, 384), Image.Resampling.LANCZOS)
        mask = mask.resize((384, 384), Image.Resampling.NEAREST)
        return extreme_perspective_occlusion(img, mask)
    elif aug_type == "radial_persp":
        img = img.resize((384, 384), Image.Resampling.LANCZOS)
        mask = mask.resize((384, 384), Image.Resampling.NEAREST)
        img_radial = radial_blur(img, strength=random.uniform(1.0, 2.0))
        direction = random.choice(['top', 'bottom', 'left', 'right', 'tl', 'tr', 'bl', 'br'])
        return random_perspective_pair(img_radial, mask, direction)
    elif aug_type == "triple_combo":
        img = img.resize((384, 384), Image.Resampling.LANCZOS)
        mask = mask.resize((384, 384), Image.Resampling.NEAREST)
        # Xoay nhẹ -> occlusion -> perspective
        angle = random.uniform(-15, 15)
        img_rot = img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(255,255,255))
        mask_rot = mask.rotate(angle, resample=Image.Resampling.NEAREST, expand=False, fillcolor=0)
        img_occ, mask_occ = occlusion_both(img_rot, mask_rot)
        direction = random.choice(['top', 'bottom', 'left', 'right'])
        img_final, mask_final = random_perspective_pair(img_occ, mask_occ, direction)
        return img_final, mask_final
    else:
        img = img.resize((384, 384), Image.Resampling.LANCZOS)
        mask = mask.resize((384, 384), Image.Resampling.NEAREST)
        return img, mask

def pad_and_resize(img, mask, target_size=(384, 384)):
    orig_w, orig_h = img.size
    target_w, target_h = target_size
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    mask_resized = mask.resize((new_w, new_h), Image.Resampling.NEAREST)
    # Strict binarization: threshold >=128
    mask_resized_np = np.array(mask_resized)
    mask_resized_np = np.where(mask_resized_np >= 128, 255, 0).astype(np.uint8)
    img_padded = Image.new("RGB", (target_w, target_h), color=(255, 255, 255))
    mask_padded_np = np.zeros((target_h, target_w), dtype=np.uint8)
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    img_padded.paste(img_resized, (paste_x, paste_y))
    mask_padded_np[paste_y:paste_y+new_h, paste_x:paste_x+new_w] = mask_resized_np
    mask_padded = Image.fromarray(mask_padded_np)
    return img_padded, mask_padded

def process_one_image(img_path, mask_path, output_dir, mask_dir, augment_types):
    img = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    img, mask = pad_and_resize(img, mask, target_size=(384, 384))
    color_variants = generate_10_color_variants_logo(img, mask)
    for i, c_img in enumerate(color_variants):
        c_name = f"{base_name}_color{i}"
        img_path_out = os.path.join(output_dir, f"{c_name}.jpg")
        mask_path_out = os.path.join(mask_dir, f"{c_name}_mask.png")
        c_img.save(img_path_out, quality=90, subsampling=1)
        if mask.mode != "L":
            mask = mask.convert("L")
        mask.save(mask_path_out)
        # Combo-augment: 20% chance
        combos = [
            ("blur", "jpeg"),
            ("crop_part", "perspective"),
            ("motion", "jpeg"),
            ("crop_part", "motion"),
        ]
        if random.random() < 0.2:
            combo = random.choice(combos)
            aug_img, aug_mask = apply_augmentation(c_img, mask, combo[0])
            aug_img2, aug_mask2 = apply_augmentation(aug_img, aug_mask, combo[1])
            visible_ratio = np.mean(np.array(aug_mask2) > 0)
            if visible_ratio >= 0.25:
                aug_img_path = os.path.join(output_dir, f"{c_name}_{combo[0]}_{combo[1]}.jpg")
                aug_mask_path = os.path.join(mask_dir, f"{c_name}_{combo[0]}_{combo[1]}_mask.png")
                if aug_mask2.mode != "L":
                    aug_mask2 = aug_mask2.convert("L")
                if combo[1] == "jpeg":
                    aug_img2.save(aug_img_path, quality=95, subsampling=1)
                else:
                    aug_img2.save(aug_img_path, quality=90, subsampling=1)

                aug_mask2.save(aug_mask_path)
        selected_augments = random.sample(augment_types, k=9)
        for aug in selected_augments:
            aug_img, aug_mask = apply_augmentation(c_img, mask, aug)
            visible_ratio = np.mean(np.array(aug_mask) > 0)
            if visible_ratio < 0.25:
                continue
            aug_img_path = os.path.join(output_dir, f"{c_name}_{aug}.jpg")
            aug_mask_path = os.path.join(mask_dir, f"{c_name}_{aug}_mask.png")
            if aug_mask.mode != "L":
                aug_mask = aug_mask.convert("L")
            if aug == "jpeg":
                aug_img.save(aug_img_path, quality=95, subsampling=1)  # thay vì mặc định 75    
            else:
                aug_img.save(aug_img_path, quality=90, subsampling=1)
            aug_mask.save(aug_mask_path)

def generate_10_distinct_color_variants_logo(img_pil, mask_pil):
    """
    Sinh ra 10 biến thể màu sắc khác biệt rõ ràng bằng cách shift hue đều trên vòng màu.
    """
    img_np = np.array(img_pil.convert('RGB'))
    mask_np = np.array(mask_pil)
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    variants = []
    for i in range(10):
        # Chia đều hue trên vòng màu (0-180)
        hue_shift = int(i * 180 / 10)
        hsv_variant = hsv.copy()
        hsv_variant[..., 0] = (hsv_variant[..., 0].astype(int) + hue_shift) % 180
        shifted = cv2.cvtColor(hsv_variant, cv2.COLOR_HSV2RGB)
        # Áp dụng mask: giữ nền trắng
        shifted[mask_np == 0] = [255, 255, 255]
        # Xử lý logo đen
        logo_mask = mask_np > 0
        logo_rgb = shifted[logo_mask]
        dark_pixels = np.sum(logo_rgb, axis=1) < 120
        if dark_pixels.any():
            new_color = np.array([
                random.randint(100, 255),
                random.randint(100, 255),
                random.randint(100, 255)
            ], dtype=np.uint8)
            logo_rgb[dark_pixels] = new_color
            shifted[logo_mask] = logo_rgb
        img_color = Image.fromarray(shifted)
        variants.append(img_color)
    return variants

def process_one_image_limit(img_path, mask_path, output_dir, mask_dir, augment_types, num_images=300, aug_count_dict=None):
    img = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    img, mask = pad_and_resize(img, mask, target_size=(384, 384))
    # Generate 10 distinct color variants
    color_variants = generate_10_distinct_color_variants_logo(img, mask)

    # 1. Save 10 color variants (no augment)
    for color_id, c_img in enumerate(color_variants):
        c_name = f"{base_name}_color{color_id:02d}"
        img_path_out  = os.path.join(output_dir, f"{c_name}.jpg")
        mask_path_out = os.path.join(mask_dir,   f"{c_name}_mask.png")
        c_img.save(img_path_out, quality=90, subsampling=1)
        mask.save(mask_path_out)
        if aug_count_dict is not None:
            aug_count_dict["color"] = aug_count_dict.get("color", 0) + 1

    # 2. Generate 290 augmented images from color variants
    generated = 10
    while generated < num_images:
        # Randomly pick one of the 10 color variants
        color_id = random.randint(0, 9)
        c_img = color_variants[color_id]
        # Randomly pick an augmentation type
        aug = random.choice(augment_types)
        aug_img, aug_mask = apply_augmentation(c_img, mask, aug)
        visible_ratio = np.mean(np.array(aug_mask) > 0)
        if visible_ratio < 0.25:
            continue

        uid = f"{generated:03d}"
        a_name = f"{base_name}_{uid}_color{color_id:02d}_{aug}"
        aug_img_path  = os.path.join(output_dir, f"{a_name}.jpg")
        aug_mask_path = os.path.join(mask_dir,   f"{a_name}_mask.png")

        if aug_mask.mode != "L":
            aug_mask = aug_mask.convert("L")
        if aug == "jpeg":
            aug_img.save(aug_img_path, quality=95, subsampling=1)
        else:
            aug_img.save(aug_img_path, quality=90, subsampling=1)
        aug_mask.save(aug_mask_path)

        generated += 1

        if aug_count_dict is not None:
            aug_count_dict[aug] = aug_count_dict.get(aug, 0) + 1

def process_folder(input_dir, mask_dir, output_root, mask_root, num_images_per_class=300):
    augment_types = [
        "perspective", "motion", "blur", "occlusion", "crop_part",
        "crop_persp", "motion_persp", "occ_persp", "noise", "jpeg",
        "rotate", "radial_blur", "lens_distort", "severe_rot_crop",
        "multi_occ_rot", "motion_distort", "extreme_persp_occ",
        "radial_persp", "triple_combo"
    ]
    images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png'))]
    class_aug_counts = {}
    for img_file in tqdm(images, desc="Processing images"):
        try:
            img_name = os.path.splitext(img_file)[0]
            img_path = os.path.join(input_dir, img_file)
            mask_file = img_name + "_mask.png"
            mask_path = os.path.join(mask_dir, mask_file)
            if not os.path.exists(mask_path):
                continue
            output_dir = os.path.join(output_root, img_name)
            mask_dir_out = os.path.join(mask_root, img_name)
            aug_count_dict = {}
            process_one_image_limit(
                img_path, mask_path, output_dir, mask_dir_out,
                augment_types, num_images=num_images_per_class,
                aug_count_dict=aug_count_dict
            )
            class_aug_counts[img_name] = aug_count_dict
        except Exception as e:
            print(f"[WARN] Skip {img_file}: {e}")
            continue

    # Write summary to Excel
    excel_path = os.path.join(output_root, "augment_summary.xlsx")
    # Collect all augment types used
    all_aug_types = set()
    for d in class_aug_counts.values():
        all_aug_types.update(d.keys())
    all_aug_types = sorted(list(all_aug_types))
    rows = []
    for class_name, aug_dict in class_aug_counts.items():
        row = {"class": class_name}
        total = 0
        for aug in all_aug_types:
            count = aug_dict.get(aug, 0)
            row[aug] = count
            total += count
        row["total"] = total
        rows.append(row)
    df = pd.DataFrame(rows)
    df = df[["class"] + all_aug_types + ["total"]]
    df.to_excel(excel_path, index=False)
    print(f"Augment summary written to {excel_path}")

def get_random_bg_color(mask_np, min_lum_diff=100):
    # Lấy màu logo (mean trên vùng mask)
    logo_pixels = mask_np > 0
    if logo_pixels.sum() == 0:
        return (200, 200, 200)  # fallback
    # Random màu nền, đảm bảo tương phản với logo (dùng get_contrasting_colors)
    # Đơn giản: chọn màu sáng nếu logo tối, hoặc ngược lại
    return get_contrasting_colors(min_lum_diff)[1]

def replace_white_bg_folder(img_dir, out_dir):
    for root, dirs, files in os.walk(img_dir):
        rel_root = os.path.relpath(root, img_dir)
        out_root = os.path.join(out_dir, rel_root) if rel_root != '.' else out_dir
        os.makedirs(out_root, exist_ok=True)
        img_files = [f for f in files if f.lower().endswith('.jpg')]
        for img_file in img_files:
            img_path = os.path.join(root, img_file)
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img)
            
            # Tìm pixel gần trắng (tất cả kênh >= 230)
            near_white = (img_np >= 230).all(axis=2)
            
            # Tạo nhiều loại background khác nhau
            bg_type = random.choice(['pastel', 'solid', 'gradient', 'texture'])
            
            if bg_type == 'pastel':
                # Background pastel nhẹ như hiện tại
                pastel = np.array([
                    random.randint(180, 240),
                    random.randint(180, 240),
                    random.randint(180, 240)
                ], dtype=np.uint8)
                img_np[near_white] = pastel
                
            elif bg_type == 'solid':
                # Background màu đơn sáng hoặc trung bình
                solid_color = np.array([
                    random.randint(100, 200),
                    random.randint(100, 200),
                    random.randint(100, 200)
                ], dtype=np.uint8)
                img_np[near_white] = solid_color
                
            elif bg_type == 'gradient':
                # Background gradient đơn giản
                h, w = img_np.shape[:2]
                if random.choice([True, False]):  # horizontal gradient
                    gradient = np.linspace(0, 1, w)
                    gradient = np.tile(gradient, (h, 1))
                else:  # vertical gradient
                    gradient = np.linspace(0, 1, h)
                    gradient = np.tile(gradient[:, np.newaxis], (1, w))
                
                color1 = np.array([random.randint(120, 200) for _ in range(3)])
                color2 = np.array([random.randint(120, 200) for _ in range(3)])
                
                for i in range(3):
                    bg_channel = color1[i] + (color2[i] - color1[i]) * gradient
                    img_np[near_white, i] = bg_channel[near_white]
                    
            else:  # texture
                # Background texture noise nhẹ
                base_color = np.array([
                    random.randint(140, 200),
                    random.randint(140, 200),
                    random.randint(140, 200)
                ], dtype=np.uint8)
                
                # Thêm noise texture
                h, w = img_np.shape[:2]
                noise = np.random.normal(0, 15, (h, w, 3))
                textured_bg = base_color + noise
                textured_bg = np.clip(textured_bg, 0, 255).astype(np.uint8)
                img_np[near_white] = textured_bg[near_white]
            
            out_img = Image.fromarray(img_np)
            out_img.save(os.path.join(out_root, img_file), quality=90, subsampling=1)

def ensure_100_per_class(img_root, mask_root):
    """
    Đảm bảo mỗi class folder chỉ chứa đúng 100 ảnh và mask tương ứng.
    """
    for class_name in os.listdir(img_root):
        class_img_dir = os.path.join(img_root, class_name)
        class_mask_dir = os.path.join(mask_root, class_name)
        if not os.path.isdir(class_img_dir) or not os.path.isdir(class_mask_dir):
            continue
        img_files = sorted([f for f in os.listdir(class_img_dir) if f.lower().endswith('.jpg')])
        mask_files = sorted([f for f in os.listdir(class_mask_dir) if f.lower().endswith('.png')])
        # Lấy 100 ảnh đầu (nếu nhiều hơn)
        keep_imgs = set(img_files[:100])
        keep_masks = set()
        # Tìm mask tương ứng với ảnh giữ lại
        for img_name in keep_imgs:
            mask_name = os.path.splitext(img_name)[0] + "_mask.png"
            keep_masks.add(mask_name)
        # Xóa ảnh thừa
        for f in img_files:
            if f not in keep_imgs:
                os.remove(os.path.join(class_img_dir, f))
        for f in mask_files:
            if f not in keep_masks:
                os.remove(os.path.join(class_mask_dir, f))

if __name__ == "__main__":
    input_dir = r"D:\\Work\\One shot Logo Recognition\\logo_output_cropped"
    mask_dir = r"D:\\Work\\One shot Logo Recognition\\logo_masks"
    output_dir = r"D:\\Work\\One shot Logo Recognition\\logo_output_image_384"
    mask_output_dir = r"D:\\Work\\One shot Logo Recognition\\logo_output_mask_384"
    process_folder(input_dir, mask_dir, output_dir, mask_output_dir, num_images_per_class=300)

    # Tạo folder ảnh đổi nền trắng thành màu tương phản
    bg_out_dir = r"D:\\Work\\One shot Logo Recognition\\logo_output_image_bgcolor"
    replace_white_bg_folder(output_dir, bg_out_dir)
