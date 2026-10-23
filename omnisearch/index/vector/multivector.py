"""Multi-vector index: many vectors per document (e.g. one per video keyframe).

A document's score is the maximum similarity over its vectors (MaxSim), and the
best-matching vector's metadata (the keyframe timestamp) is returned so a hit
can point to a moment inside the video. Vectors of one document are stored
contiguously, so per-document maxima are a single ``np.maximum.reduceat``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from omnisearch.index.vector.hnsw import HNSWIndex
from omnisearch.index.vector.store import VectorStore
from omnisearch.util import GrowableArray, topk


class MultiVectorIndex:
    def __init__(self, dim: int, metric: str = "cosine", index: str = "flat", params: dict | None = None):
        params = params or {}
        self.store = VectorStore(dim, metric)
        self.owner = GrowableArray(np.int64)
        self.meta = GrowableArray(np.float32)  # timestamp (seconds) of each vector
        self.start = GrowableArray(np.int64, fill=-1)
        self.end = GrowableArray(np.int64, fill=-1)
        self.hnsw = HNSWIndex(self.store, M=params.get("M", 16), ef_construction=params.get("ef_construction", 100)) \
            if index == "hnsw" else None
        self._seg = None

    def __len__(self):
        return len(self.store)

    def n_docs(self) -> int:
        return int((self.start.data >= 0).sum())

    def add(self, doc_id: int, vecs, metas=None):
        vecs = np.atleast_2d(vecs)
        if len(vecs) == 0:
            return
        if len(self.owner) and doc_id <= self.owner[len(self.owner) - 1]:
            raise ValueError("documents must be added in increasing id order")
        pos = self.store.add(vecs)
        self.owner.extend(np.full(len(pos), doc_id))
        self.meta.extend(np.zeros(len(pos)) if metas is None else metas)
        self.start.resize(doc_id + 1)
        self.end.resize(doc_id + 1)
        self.start[doc_id], self.end[doc_id] = pos[0], pos[-1] + 1
        self._seg = None

    def build(self):
        if self.hnsw is not None:
            self.hnsw.sync()

    def _segments(self):
        if self._seg is None:
            owner = self.owner.data
            starts = np.flatnonzero(np.r_[True, owner[1:] != owner[:-1]]) if len(owner) else np.empty(0, np.int64)
            self._seg = (starts, owner[starts])
        return self._seg

    def search(self, q, k, allow=None, strategy="exact", ef=None, overfetch=8):
        if len(self.store) == 0:
            return np.empty(0, np.int64), np.empty(0, np.float32), {"scanned": 0}
        if strategy == "index" and self.hnsw is not None:
            allow_pos = None if allow is None else allow[self.owner.data]
            kk = k * overfetch
            pos, sims, st = self.hnsw.search(q, kk, ef=max(ef or 0, kk), allow=allow_pos)
            best: dict[int, float] = {}
            for p, s in zip(pos.tolist(), sims.tolist()):
                d = int(self.owner[p])
                if d not in best:
                    best[d] = s
            docs = np.fromiter(best.keys(), np.int64, len(best))
            scores = np.fromiter(best.values(), np.float32, len(best))
            o = topk(scores, k)
            return docs[o], scores[o], st
        starts, docs = self._segments()
        sims = self.store.score(q)
        doc_scores = np.maximum.reduceat(sims, starts)
        if allow is not None:
            keep = allow[docs]
            docs, doc_scores = docs[keep], doc_scores[keep]
        o = topk(doc_scores, k)
        return docs[o], doc_scores[o], {"scanned": len(sims)}

    def score(self, q, doc_ids) -> np.ndarray:
        out = np.full(len(doc_ids), -np.inf, dtype=np.float32)
        for i, d in enumerate(np.asarray(doc_ids).tolist()):
            if d < len(self.start) and self.start[d] >= 0:
                out[i] = self.store.score(q, np.arange(self.start[d], self.end[d])).max()
        return out

    def best_moment(self, q, doc_id: int) -> float | None:
        if doc_id >= len(self.start) or self.start[doc_id] < 0:
            return None
        rng = np.arange(self.start[doc_id], self.end[doc_id])
        return float(self.meta[rng[int(np.argmax(self.store.score(q, rng)))]])

    def save(self, path: Path):
        np.savez(path, vectors=self.store.vectors, owner=self.owner.data, meta=self.meta.data,
                 start=self.start.data, end=self.end.data)
        if self.hnsw is not None:
            self.hnsw.save(Path(str(path) + ".hnsw.npz"))

    def load(self, path: Path):
        z = np.load(path)
        if len(z["vectors"]):
            self.store.add(z["vectors"])
        self.owner = GrowableArray.from_array(z["owner"])
        self.meta = GrowableArray.from_array(z["meta"])
        self.start = GrowableArray.from_array(z["start"], fill=-1)
        self.end = GrowableArray.from_array(z["end"], fill=-1)
        hp = Path(str(path) + ".hnsw.npz")
        if self.hnsw is not None and hp.exists():
            self.hnsw = HNSWIndex.load(hp, self.store)
        self._seg = None
