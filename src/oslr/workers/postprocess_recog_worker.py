from ..utils.image_utils import draw_bbox_mask_label
from ..utils.item_classes import PostprocessedRecognizedItem


class PostprocessRecogWorker:
    """
    Worker vẽ bbox, mask, label lên frame kết quả.
    """

    def process(self, input_item, recognized_item, threshold=0.4):
        frame = input_item.frame.copy()
        for rec in recognized_item.recognitions:
            draw_bbox_mask_label(
                frame, rec.bbox, rec.mask, rec.label, rec.score, threshold
            )
        return PostprocessedRecognizedItem(
            frame_id=input_item.frame_id, frame_output=frame
        )
