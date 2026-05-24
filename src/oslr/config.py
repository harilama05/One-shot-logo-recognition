from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PipelineConfig:
    video_path: str
    yolo_weights: str
    recog_weights: str
    embed_db_path: str
    output_path: str
    conf_threshold: float = 0.7
    recog_threshold: float = 0.4
    device: Optional[str] = None
