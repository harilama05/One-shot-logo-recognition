from collections import deque
from multiprocessing import Lock


class CircularQueue:
    def __init__(self, maxsize):
        self.queue = deque(maxlen=maxsize)
        self.lock = Lock()

    def put(self, item):
        with self.lock:
            if len(self.queue) == self.queue.maxlen:
                self.queue.popleft()
            self.queue.append(item)

    def get(self):
        with self.lock:
            if self.queue:
                return self.queue.popleft()
            return None

    def is_empty(self):
        with self.lock:
            return len(self.queue) == 0

    def is_full(self):
        with self.lock:
            return len(self.queue) == self.queue.maxlen

    def clear(self):
        with self.lock:
            self.queue.clear()
