import cv2

from ..utils.item_classes import InputItem


class InputWorker:
    """
    Worker đọc từng frame từ video đầu vào.
    """

    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        self.frame_id = 0

    def get_video_props(self):
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return fps, width, height

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        self.frame_id += 1
        return InputItem(frame_id=self.frame_id, frame=frame)

    def release(self):
        self.cap.release()
