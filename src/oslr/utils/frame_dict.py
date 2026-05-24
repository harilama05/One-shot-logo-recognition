import threading

import numpy as np


class FrameStore:
    def __init__(self):
        self._store: dict[int, np.ndarray] = {}
        self._lock = threading.Lock()

    def set(self, frame_id: int, frame: np.ndarray):
        with self._lock:
            self._store[frame_id] = frame

    def get(self, frame_id: int) -> np.ndarray | None:
        with self._lock:
            return self._store.get(frame_id, None)

    def delete(self, frame_id: int):
        with self._lock:
            if frame_id in self._store:
                del self._store[frame_id]


frame_store = FrameStore()
