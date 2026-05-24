import cv2


class OutputWorker:
    """
    Worker ghi frame kết quả ra file video.
    """

    def __init__(self, output_path, fps, width, height):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not self.writer.isOpened():
            raise ValueError(f"Cannot open output video: {output_path}")

    def write(self, frame):
        self.writer.write(frame)

    def release(self):
        self.writer.release()
