from pathlib import Path
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet


class LogoEncoder(nn.Module):
    """
    Mô hình ArcFace dùng EfficientNet làm backbone để sinh embedding cho logo.
    """

    def __init__(self, embedding_dim=512, dropout_rate=0.6):
        super().__init__()
        base = EfficientNet.from_pretrained("efficientnet-b4")
        in_features = base._fc.in_features
        base._fc = nn.Identity()
        self.backbone = base
        self.embedding = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, embedding_dim * 2),
            nn.BatchNorm1d(embedding_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, x, mask=None):
        features = self.backbone.extract_features(x)
        if mask is not None:
            mask = F.interpolate(mask, size=features.shape[2:], mode="nearest")
            mask = mask.clamp(0, 1)
            features = features * mask
            mask_sum = mask.sum(dim=[2, 3], keepdim=True) + 1e-6
            pooled = (features.sum(dim=[2, 3], keepdim=True) / mask_sum).squeeze(
                -1
            ).squeeze(-1)
        else:
            pooled = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
        embeddings = self.embedding(pooled)
        return F.normalize(embeddings, dim=1)


def load_arcface_model(weight_path, device):
    """
    Load mô hình ArcFace với trọng số đã huấn luyện.
    """
    model = LogoEncoder().to(device)
    checkpoint = torch.load(weight_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_embeddings(path):
    """
    Load embedding database từ file pickle.
    """
    if not path:
        return np.array([]), []
    path = Path(path)
    if not path.exists():
        return np.array([]), []
    with path.open("rb") as f:
        db = pickle.load(f)
    if not db:
        return np.array([]), []
    embeddings = np.array([e for e, _ in db])
    labels = [c for _, c in db]
    return embeddings, labels
