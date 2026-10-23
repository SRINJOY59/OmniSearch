"""Inverted file index (IVF): k-means partitions + per-partition id lists.

A query scores the ``nlist`` centroids, then only the vectors in the
``nprobe`` closest partitions. Filters are cheap here: candidate ids are
masked by the bitmap before any vector is scored. Combined with a quantizer
this is IVF-SQ8 / IVF-PQ.
"""
from __future__ import annotations

import math

import numpy as np

from omnisearch.index.vector.flat import score_candidates
from omnisearch.index.vector.quantize import _assign, kmeans
from omnisearch.util import GrowableArray, topk


class IVFIndex:
    def __init__(self, store, nlist: int | None = None, nprobe: int = 8, quantizer=None,
                 refine: int = 4, min_train: int = 256, seed: int = 0):
        self.store = store
        self.nlist = nlist
        self.nprobe = nprobe
        self.quantizer = quantizer
        self.refine = refine
        self.min_train = min_train
        self.seed = seed
        self.centroids: np.ndarray | None = None
        self.lists: list[GrowableArray] = []
        self.assigned = 0

    @property
    def trained(self):
        return self.centroids is not None

    def train(self):
        n = len(self.store)
        nlist = self.nlist or max(1, int(round(4 * math.sqrt(n))))
        self.centroids = kmeans(self.store.vectors, nlist, seed=self.seed)
        self.lists = [GrowableArray(np.int64, capacity=16) for _ in range(len(self.centroids))]
        self.assigned = 0
        self.sync()

    def sync(self):
        if not self.trained:
            if len(self.store) >= self.min_train:
                self.train()
            return
        if self.assigned == len(self.store):
            return
        new = np.arange(self.assigned, len(self.store))
        assign = _assign(self.store.vectors[new], self.centroids)
        order = np.argsort(assign, kind="stable")
        bounds = np.searchsorted(assign[order], np.arange(len(self.centroids) + 1))
        for c in range(len(self.centroids)):
            if bounds[c + 1] > bounds[c]:
                self.lists[c].extend(new[order[bounds[c] : bounds[c + 1]]])
        self.assigned = len(self.store)

    def search(self, q, k, nprobe: int | None = None, allow: np.ndarray | None = None):
        nprobe = min(nprobe or self.nprobe, len(self.lists) or 1)
        if not self.trained:
            cand = None if allow is None else np.flatnonzero(allow)
            pos, scores, n = score_candidates(self.store, self.quantizer, self.refine, q, cand, k)
            return pos, scores, {"lists": 0, "scanned": n}
        probe = topk(self.centroids @ q, nprobe)
        cand = np.concatenate([self.lists[c].data for c in probe])
        if allow is not None:
            cand = cand[allow[cand]]
        if len(cand) == 0:
            return np.empty(0, np.int64), np.empty(0, np.float32), {"lists": nprobe, "scanned": 0}
        pos, scores, n = score_candidates(self.store, self.quantizer, self.refine, q, cand, k)
        return pos, scores, {"lists": nprobe, "scanned": n}

    def list_sizes(self) -> np.ndarray:
        return np.array([len(l) for l in self.lists])

    def save_arrays(self) -> dict:
        if not self.trained:
            return {}
        sizes = self.list_sizes()
        return {
            "centroids": self.centroids,
            "list_sizes": sizes,
            "list_ids": np.concatenate([l.data for l in self.lists]) if sizes.sum() else np.empty(0, np.int64),
            "assigned": np.array([self.assigned]),
        }

    def load_arrays(self, z):
        if "centroids" not in z:
            return
        self.centroids = z["centroids"]
        sizes, ids = z["list_sizes"], z["list_ids"]
        bounds = np.concatenate(([0], np.cumsum(sizes)))
        self.lists = [GrowableArray.from_array(ids[bounds[i] : bounds[i + 1]]) for i in range(len(sizes))]
        self.assigned = int(z["assigned"][0])
