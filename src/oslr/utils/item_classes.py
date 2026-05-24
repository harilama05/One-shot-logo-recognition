from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class InputItem:
    frame_id: int
    frame: np.ndarray


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]
    mask: np.ndarray | None


@dataclass
class DetectedItem:
    frame_id: int
    frame: np.ndarray
    detections: List[Detection]


@dataclass
class Postprocession:
    bbox: Tuple[int, int, int, int]
    mask: np.ndarray | None
    crop: np.ndarray


@dataclass
class PostprocessedDetectedItem:
    frame_id: int
    postprocessions: List[Postprocession]


@dataclass
class Recognition:
    bbox: Tuple[int, int, int, int]
    mask: np.ndarray | None
    label: str
    score: float


@dataclass
class RecognizedItem:
    frame_id: int
    recognitions: List[Recognition]


@dataclass
class PostprocessedRecognizedItem:
    frame_id: int
    frame_output: np.ndarray
