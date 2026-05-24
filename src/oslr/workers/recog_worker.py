import torch

from ..utils.image_utils import preprocess_image, resize_with_padding
from ..utils.item_classes import RecognizedItem, Recognition


class RecogWorker:
    """
    Worker nhận diện logo từ crop và mask bằng ArcFace, trả về label và score.
    """

    def __init__(self, arcface_model, db_embeddings, db_labels, device, threshold=0.4):
        self.model = arcface_model
        self.device = device
        self.threshold = threshold
        if db_embeddings is None or len(db_embeddings) == 0:
            self.db_embeddings = None
            self.db_labels = []
        else:
            self.db_embeddings = torch.tensor(db_embeddings, dtype=torch.float32).to(
                device
            )
            self.db_embeddings = torch.nn.functional.normalize(self.db_embeddings, dim=1)
            self.db_labels = db_labels

    def process(self, postprocessed_item):
        recognitions = []
        if not postprocessed_item.postprocessions:
            return RecognizedItem(
                frame_id=postprocessed_item.frame_id, recognitions=recognitions
            )
        for proc in postprocessed_item.postprocessions:
            img_tensor = preprocess_image(proc.crop).unsqueeze(0).to(self.device)
            if proc.mask is not None:
                mask = resize_with_padding(proc.mask, target_size=380, is_mask=True)
                mask_tensor = (
                    torch.from_numpy(mask)
                    .float()
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to(self.device)
                    / 255.0
                )
            else:
                mask_tensor = torch.ones(1, 1, 380, 380).to(self.device)
            if self.db_embeddings is None or not self.db_labels:
                label = "Unknown"
                score = 0.0
            else:
                with torch.no_grad():
                    emb = self.model(img_tensor, mask=mask_tensor)
                    sims = torch.mm(emb, self.db_embeddings.T)
                    score, idx = torch.max(sims, dim=1)
                    score = float(score.item())
                    label = (
                        self.db_labels[idx.item()]
                        if score >= self.threshold
                        else "Unknown"
                    )
            recognitions.append(
                Recognition(
                    bbox=proc.bbox, mask=proc.mask, label=label, score=score
                )
            )
        return RecognizedItem(
            frame_id=postprocessed_item.frame_id, recognitions=recognitions
        )
