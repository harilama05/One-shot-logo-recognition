import cv2
import numpy as np
from torchvision import transforms


def _clamp_bbox(bbox, width, height):
    x1, y1, x2, y2 = map(int, bbox)
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    return x1, y1, x2, y2


def resize_with_padding(img, target_size=380, fill=(128, 128, 128), is_mask=False):
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_CUBIC
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    pad_w = target_size - new_w
    pad_h = target_size - new_h
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2

    pad_value = 0 if is_mask else fill
    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=pad_value,
    )

    return padded


def preprocess_image(img_bgr, target_size=380):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    padded_img = resize_with_padding(img_rgb, target_size=target_size)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    tensor_img = transform(padded_img)
    return tensor_img


def crop_bbox(image, bbox):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = _clamp_bbox(bbox, width, height)
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]


def draw_bbox_mask_label(frame, bbox, mask, label, score, threshold):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = _clamp_bbox(bbox, width, height)
    if x2 <= x1 or y2 <= y1:
        return

    color = (0, 255, 0) if score >= threshold else (0, 0, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    if mask is not None:
        mask_vis = (mask > 0.5).astype(np.uint8)
        mask_vis = cv2.resize(mask_vis, (x2 - x1, y2 - y1))
        mask_color = np.zeros_like(frame[y1:y2, x1:x2], dtype=np.uint8)
        mask_color[:, :, 1] = mask_vis * 255
        frame[y1:y2, x1:x2] = cv2.addWeighted(
            frame[y1:y2, x1:x2], 1.0, mask_color, 0.5, 0
        )

    cv2.putText(
        frame,
        f"{label} {score:.2f}",
        (x1, max(0, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
    )
