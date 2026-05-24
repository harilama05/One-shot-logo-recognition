from ..utils.item_classes import DetectedItem, Detection


class DetectWorker:
    """
    Worker nhận diện các bbox và mask logo trong frame bằng YOLO.
    """

    def __init__(self, yolo_model, conf_threshold=0.7):
        self.yolo_model = yolo_model
        self.conf_threshold = conf_threshold

    def process(self, input_item):
        detections = self.yolo_model.predict(input_item.frame, self.conf_threshold)
        det_objs = [Detection(bbox=det["bbox"], mask=det["mask"]) for det in detections]
        return DetectedItem(
            frame_id=input_item.frame_id, frame=input_item.frame, detections=det_objs
        )
