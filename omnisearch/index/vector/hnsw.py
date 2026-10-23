"""Hierarchical Navigable Small World graph (Malkov & Yashunin, 2018), from scratch.

Each vector is a node on layers 0..level, with level drawn from a geometric
distribution, so upper layers are sparse "express lanes". Search descends
greedily through the upper layers, then runs a best-first beam search of width
``ef`` on layer 0.

Filtered search ("in-graph filtering"): every node can be traversed for
navigation, but only nodes allowed by the filter enter the result set. With a
very selective filter the beam has to wander far before it collects ``ef``
allowed nodes, which is why the optimizer switches to an exact pre-filtered
scan below some selectivity.
"""
from __future__ import annotations

import heapq
import json
import math
from pathlib import Path

import numpy as np


class HNSWIndex:
    def __init__(self, store, M: int = 16, ef_construction: int = 100, ef_search: int = 64, seed: int = 0):
        self.store = store
        self.M = M
        self.M0 = 2 * M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.ml = 1.0 / math.log(M)
        self.rng = np.random.default_rng(seed)
        self.levels: list[int] = []
        self.links: list[list[list[int]]] = []
        self.entry = -1
        self.max_level = -1

    def __len__(self):
        return len(self.levels)

    # ---------------------------------------------------------------- search
    def _sim(self, q, nodes):
        return self.store.score(q, np.asarray(nodes, dtype=np.int64))

    def _search_layer(self, q, eps, ef, layer, allow=None, stats=None):
        """Beam search on one layer. eps: [(score, node)]. Returns best-first list."""
        visited = {n for _, n in eps}
        cand = [(-s, n) for s, n in eps]
        heapq.heapify(cand)
        res = [(s, n) for s, n in eps if allow is None or allow[n]]
        heapq.heapify(res)
        while len(res) > ef:
            heapq.heappop(res)
        links = self.links
        while cand:
            negs, c = heapq.heappop(cand)
            if len(res) >= ef and -negs < res[0][0]:
                break
            nbrs = [x for x in links[c][layer] if x not in visited]
            if not nbrs:
                continue
            visited.update(nbrs)
            sims = self._sim(q, nbrs).tolist()
            if stats is not None:
                stats["expansions"] += 1
                stats["distances"] += len(nbrs)
            for x, sx in zip(nbrs, sims):
                if len(res) < ef or sx > res[0][0]:
                    heapq.heappush(cand, (-sx, x))
                    if allow is None or allow[x]:
                        heapq.heappush(res, (sx, x))
                        if len(res) > ef:
                            heapq.heappop(res)
        return sorted(res, reverse=True)

    def search(self, q, k: int, ef: int | None = None, allow: np.ndarray | None = None):
        stats = {"expansions": 0, "distances": 0}
        if self.entry < 0:
            return np.empty(0, np.int64), np.empty(0, np.float32), stats
        ef = max(ef or self.ef_search, k)
        ep = [(float(self._sim(q, [self.entry])[0]), self.entry)]
        for layer in range(self.max_level, 0, -1):
            ep = self._search_layer(q, ep, 1, layer, stats=stats)
        res = self._search_layer(q, ep, ef, 0, allow=allow, stats=stats)[:k]
        ids = np.array([n for _, n in res], dtype=np.int64)
        scores = np.array([s for s, _ in res], dtype=np.float32)
        return ids, scores, stats

    # ----------------------------------------------------------------- build
    def _select(self, cands, m):
        """Neighbour-selection heuristic: keep a candidate only if it is closer to
        the base than to every neighbour already kept (spreads edges out)."""
        if len(cands) <= m:
            return [n for _, n in cands]
        ids = np.array([n for _, n in cands], dtype=np.int64)
        pair = self.store.pairwise(ids)
        kept: list[int] = []
        for i, (s, _) in enumerate(cands):
            if not kept or pair[i, kept].max() < s:
                kept.append(i)
                if len(kept) == m:
                    break
        return [int(ids[i]) for i in kept]

    def _insert(self, pos: int):
        level = int(-math.log(1.0 - self.rng.random()) * self.ml)
        self.levels.append(level)
        self.links.append([[] for _ in range(level + 1)])
        if self.entry < 0:
            self.entry, self.max_level = pos, level
            return
        q = self.store.vectors[pos]
        ep = [(float(self._sim(q, [self.entry])[0]), self.entry)]
        for layer in range(self.max_level, level, -1):
            ep = self._search_layer(q, ep, 1, layer)
        for layer in range(min(level, self.max_level), -1, -1):
            w = self._search_layer(q, ep, self.ef_construction, layer)
            mmax = self.M0 if layer == 0 else self.M
            nbrs = self._select(w, self.M)
            self.links[pos][layer] = nbrs
            for nb in nbrs:
                lst = self.links[nb][layer]
                lst.append(pos)
                if len(lst) > mmax:
                    sims = self._sim(self.store.vectors[nb], lst).tolist()
                    self.links[nb][layer] = self._select(sorted(zip(sims, lst), reverse=True), mmax)
            ep = w
        if level > self.max_level:
            self.entry, self.max_level = pos, level

    def sync(self):
        """Insert every store position not yet in the graph."""
        for pos in range(len(self.levels), len(self.store)):
            self._insert(pos)

    # ----------------------------------------------------------- persistence
    def stats(self) -> dict:
        deg0 = [len(l[0]) for l in self.links]
        return {
            "nodes": len(self),
            "max_level": self.max_level,
            "avg_degree_l0": round(float(np.mean(deg0)), 2) if deg0 else 0,
            "edges": int(sum(len(x) for l in self.links for x in l)),
        }

    def save(self, path: Path):
        levels = np.asarray(self.levels, dtype=np.int32)
        arrays = {"levels": levels}
        for layer in range(self.max_level + 1):
            nodes = np.flatnonzero(levels >= layer)
            lists = [self.links[n][layer] for n in nodes]
            arrays[f"indptr{layer}"] = np.cumsum([0] + [len(x) for x in lists]).astype(np.int64)
            arrays[f"indices{layer}"] = np.asarray([i for x in lists for i in x], dtype=np.int64)
        meta = {"M": self.M, "efc": self.ef_construction, "ef": self.ef_search,
                "entry": self.entry, "max_level": self.max_level}
        arrays["meta"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: Path, store) -> "HNSWIndex":
        z = np.load(path)
        meta = json.loads(z["meta"].tobytes())
        idx = cls(store, M=meta["M"], ef_construction=meta["efc"], ef_search=meta["ef"])
        levels = z["levels"]
        idx.levels = levels.tolist()
        idx.links = [[[] for _ in range(l + 1)] for l in idx.levels]
        for layer in range(meta["max_level"] + 1):
            nodes = np.flatnonzero(levels >= layer)
            indptr, indices = z[f"indptr{layer}"], z[f"indices{layer}"]
            for j, n in enumerate(nodes):
                idx.links[n][layer] = indices[indptr[j] : indptr[j + 1]].tolist()
        idx.entry, idx.max_level = meta["entry"], meta["max_level"]
        return idx
