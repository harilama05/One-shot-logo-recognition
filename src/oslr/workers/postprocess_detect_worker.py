from ..utils.image_utils import crop_bbox
from ..utils.item_classes import PostprocessedDetectedItem, Postprocession


class PostprocessDetectWorker:
    """
    Worker crop từng bbox và mask từ frame gốc.
    """

    def process(self, detected_item):
        postprocessions = []
        frame = detected_item.frame
        for det in detected_item.detections:
            crop = crop_bbox(frame, det.bbox)
            if crop is None:
                continue
            postprocessions.append(
                Postprocession(bbox=det.bbox, mask=det.mask, crop=crop)
            )
        return PostprocessedDetectedItem(
            frame_id=detected_item.frame_id, postprocessions=postprocessions
        )
