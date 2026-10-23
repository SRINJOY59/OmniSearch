"""Dense vector storage and exact scoring.

All vector indexes share one store per field. Scores are always "higher is
better": cosine and inner product return the dot product (vectors normalised
for cosine), L2 returns the negative squared distance.
"""
from __future__ import annotations

import numpy as np

from omnisearch.util import GrowableArray, normalize_rows


class VectorStore:
    def __init__(self, dim: int, metric: str = "cosine"):
        self.dim = dim
        self.metric = metric
        self._vecs = GrowableArray(np.float32, (dim,))
        self._sqnorm = GrowableArray(np.float32)

    def __len__(self):
        return len(self._vecs)

    @property
    def vectors(self) -> np.ndarray:
        return self._vecs.data

    def prepare(self, x) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=np.float32))
        if x.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {x.shape[1]}")
        return normalize_rows(x) if self.metric == "cosine" else x

    def add(self, x) -> np.ndarray:
        x = self.prepare(x)
        start = self._vecs.extend(x)
        self._sqnorm.extend((x * x).sum(axis=1))
        return np.arange(start, start + len(x))

    def query(self, q) -> np.ndarray:
        return self.prepare(q)[0]

    def score(self, q: np.ndarray, pos: np.ndarray | None = None) -> np.ndarray:
        """Exact scores of a prepared query against all (or selected) positions."""
        if pos is None:
            dots = self.vectors @ q
            sq = self._sqnorm.data if self.metric == "l2" else None
        else:
            dots = self.vectors[pos] @ q
            sq = self._sqnorm.data[pos] if self.metric == "l2" else None
        if self.metric == "l2":
            return 2.0 * dots - sq - float(q @ q)
        return dots

    def pairwise(self, pos: np.ndarray) -> np.ndarray:
        v = self.vectors[pos]
        if self.metric == "l2":
            sq = self._sqnorm.data[pos]
            return 2.0 * (v @ v.T) - sq[:, None] - sq[None, :]
        return v @ v.T

    def nbytes(self) -> int:
        return len(self) * self.dim * 4
