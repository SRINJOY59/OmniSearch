"""Small shared helpers: growable numpy arrays, top-k selection, timing."""
from __future__ import annotations

import json
import time
from contextlib import contextmanager

import numpy as np


class GrowableArray:
    """Append-only numpy array with amortised O(1) appends (capacity doubling).

    Used for every per-document column in the engine (doc lengths, attribute
    values, vector matrices, tombstones) so that ingest never re-allocates per
    record.
    """

    def __init__(self, dtype, tail=(), capacity=1024, fill=0):
        self.dtype = np.dtype(dtype)
        self.tail = tuple(tail)
        self.fill = fill
        self._buf = np.full((max(capacity, 1), *self.tail), fill, dtype=self.dtype)
        self._n = 0

    def __len__(self):
        return self._n

    @property
    def data(self) -> np.ndarray:
        return self._buf[: self._n]

    def _reserve(self, n):
        if n <= len(self._buf):
            return
        cap = len(self._buf)
        while cap < n:
            cap *= 2
        buf = np.full((cap, *self.tail), self.fill, dtype=self.dtype)
        buf[: self._n] = self._buf[: self._n]
        self._buf = buf

    def append(self, value) -> int:
        self._reserve(self._n + 1)
        self._buf[self._n] = value
        self._n += 1
        return self._n - 1

    def extend(self, values) -> int:
        values = np.asarray(values, dtype=self.dtype)
        start = self._n
        self._reserve(self._n + len(values))
        self._buf[start : start + len(values)] = values
        self._n += len(values)
        return start

    def resize(self, n):
        """Grow to length n, filling new slots with the fill value."""
        if n > self._n:
            self._reserve(n)
            self._buf[self._n : n] = self.fill
            self._n = n

    def __getitem__(self, idx):
        return self.data[idx]

    def __setitem__(self, idx, value):
        self.data[idx] = value

    @classmethod
    def from_array(cls, arr, fill=0):
        arr = np.asarray(arr)
        g = cls(arr.dtype, arr.shape[1:], capacity=max(len(arr), 1), fill=fill)
        g.extend(arr)
        return g


def topk(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest scores, sorted descending (ties by index)."""
    n = len(scores)
    if n == 0 or k <= 0:
        return np.empty(0, dtype=np.int64)
    if k >= n:
        return np.lexsort((np.arange(n), -scores))
    part = np.argpartition(-scores, k - 1)[:k]
    return part[np.lexsort((part, -scores[part]))]


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


class Timer:
    def __init__(self):
        self.ms = 0.0

    @contextmanager
    def measure(self):
        t0 = time.perf_counter()
        try:
            yield self
        finally:
            self.ms += (time.perf_counter() - t0) * 1000.0


def json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def dumps(obj) -> str:
    return json.dumps(obj, default=json_default)
