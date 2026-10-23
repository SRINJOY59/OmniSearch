"""k-means and vector compression.

* ``kmeans`` - Lloyd's algorithm with k-means++ seeding; used by IVF (coarse
  clusters) and by product quantization (per-subspace codebooks).
* ``ScalarQuantizer`` (sq8) - one byte per dimension: 4x smaller than float32.
* ``ProductQuantizer`` (pq<m>) - splits a vector into m sub-vectors and stores
  the id of the nearest of 256 sub-centroids for each: m bytes per vector.
  Queries are scored with asymmetric distance computation (ADC): a 256 x m
  lookup table of query-to-centroid scores, summed per code.
"""
from __future__ import annotations

import numpy as np

from omnisearch.util import GrowableArray


def _assign(x: np.ndarray, centroids: np.ndarray, chunk: int = 65536) -> np.ndarray:
    """Nearest centroid under L2 (argmax of x.c - |c|^2/2), chunked for memory."""
    half = 0.5 * (centroids * centroids).sum(axis=1)
    out = np.empty(len(x), dtype=np.int64)
    for s in range(0, len(x), chunk):
        out[s : s + chunk] = np.argmax(x[s : s + chunk] @ centroids.T - half, axis=1)
    return out


def kmeans(x: np.ndarray, k: int, iters: int = 20, seed: int = 0, sample: int = 50_000) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    rng = np.random.default_rng(seed)
    if len(x) > sample:
        x = x[rng.choice(len(x), sample, replace=False)]
    k = min(k, len(x))
    # k-means++ seeding on a subsample
    seed_x = x if len(x) <= 20_000 else x[rng.choice(len(x), 20_000, replace=False)]
    centroids = np.empty((k, x.shape[1]), dtype=np.float32)
    centroids[0] = seed_x[rng.integers(len(seed_x))]
    d2 = ((seed_x - centroids[0]) ** 2).sum(axis=1)
    for i in range(1, k):
        total = d2.sum()
        idx = rng.integers(len(seed_x)) if total <= 0 else rng.choice(len(seed_x), p=d2 / total)
        centroids[i] = seed_x[idx]
        d2 = np.minimum(d2, ((seed_x - centroids[i]) ** 2).sum(axis=1))
    for _ in range(iters):
        assign = _assign(x, centroids)
        counts = np.bincount(assign, minlength=k)
        sums = np.zeros_like(centroids)
        np.add.at(sums, assign, x)
        empty = counts == 0
        centroids[~empty] = sums[~empty] / counts[~empty, None]
        if empty.any():  # re-seed empty clusters with random points
            centroids[empty] = x[rng.choice(len(x), int(empty.sum()), replace=False)]
    return centroids


class ScalarQuantizer:
    name = "sq8"

    def __init__(self, dim: int):
        self.dim = dim
        self.vmin = self.scale = None
        self.codes = GrowableArray(np.uint8, (dim,))

    @property
    def trained(self):
        return self.vmin is not None

    def train(self, x: np.ndarray):
        self.vmin = x.min(axis=0)
        self.scale = np.maximum((x.max(axis=0) - self.vmin) / 255.0, 1e-12).astype(np.float32)

    def encode(self, x: np.ndarray) -> np.ndarray:
        return np.clip(np.rint((x - self.vmin) / self.scale), 0, 255).astype(np.uint8)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        return self.vmin + codes.astype(np.float32) * self.scale

    def add(self, x: np.ndarray):
        self.codes.extend(self.encode(x))

    def score(self, q: np.ndarray, pos: np.ndarray | None = None) -> np.ndarray:
        """Inner product with reconstructed vectors: q.vmin + (q*scale).codes."""
        codes = self.codes.data if pos is None else self.codes.data[pos]
        return codes.astype(np.float32) @ (q * self.scale) + float(q @ self.vmin)

    def nbytes(self) -> int:
        return len(self.codes) * self.dim


class ProductQuantizer:
    def __init__(self, dim: int, m: int, seed: int = 0):
        if dim % m:
            raise ValueError(f"dim {dim} not divisible by m={m}")
        self.dim, self.m, self.dsub = dim, m, dim // m
        self.seed = seed
        self.codebooks: np.ndarray | None = None  # (m, 256, dsub)
        self.codes = GrowableArray(np.uint8, (m,))

    @property
    def name(self):
        return f"pq{self.m}"

    @property
    def trained(self):
        return self.codebooks is not None

    def train(self, x: np.ndarray, iters: int = 15):
        ks = min(256, len(x))
        books = np.zeros((self.m, 256, self.dsub), dtype=np.float32)
        for j in range(self.m):
            sub = x[:, j * self.dsub : (j + 1) * self.dsub]
            books[j, :ks] = kmeans(sub, ks, iters=iters, seed=self.seed + j)
        self.codebooks = books

    def encode(self, x: np.ndarray) -> np.ndarray:
        codes = np.empty((len(x), self.m), dtype=np.uint8)
        for j in range(self.m):
            codes[:, j] = _assign(np.ascontiguousarray(x[:, j * self.dsub : (j + 1) * self.dsub]), self.codebooks[j])
        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        return np.concatenate([self.codebooks[j][codes[:, j]] for j in range(self.m)], axis=1)

    def add(self, x: np.ndarray):
        self.codes.extend(self.encode(x))

    def score(self, q: np.ndarray, pos: np.ndarray | None = None) -> np.ndarray:
        lut = np.einsum("mkd,md->mk", self.codebooks, q.reshape(self.m, self.dsub))
        codes = self.codes.data if pos is None else self.codes.data[pos]
        out = np.zeros(len(codes), dtype=np.float32)
        for j in range(self.m):
            out += lut[j][codes[:, j]]
        return out

    def nbytes(self) -> int:
        return len(self.codes) * self.m


def make_quantizer(spec: str | None, dim: int):
    if not spec:
        return None
    if spec == "sq8":
        return ScalarQuantizer(dim)
    if spec.startswith("pq"):
        return ProductQuantizer(dim, int(spec[2:]))
    raise ValueError(f"unknown quantization {spec!r}")
