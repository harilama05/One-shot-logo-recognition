from ultralytics import YOLO


class YoloModel:
    def __init__(self, model_path, device):
        self.model = YOLO(model_path)
        self.device = device

    def predict(self, frame_bgr, conf_threshold=0.7):
        results = self.model(frame_bgr)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        masks = results.masks.data.cpu().numpy() if results.masks else []
        detections = []
        for i, (box, conf) in enumerate(zip(boxes, confs)):
            if conf < conf_threshold:
                continue
            x1, y1, x2, y2 = map(int, box)
            mask = masks[i] if i < len(masks) else None
            detections.append({"bbox": (x1, y1, x2, y2), "mask": mask, "conf": conf})
        return detections
