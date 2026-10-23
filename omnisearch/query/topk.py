"""Top-k aggregation over several ranked sources (Fagin, Lotem & Naor, PODS 2001).

Each source supports
  * sorted access - the next best (doc, score) pair, and
  * random access - the score of an arbitrary doc.

Threshold Algorithm (TA): read every source in parallel by sorted access; for
each new doc, random-access its scores in the other sources. The threshold
tau = aggregate(last score seen in each source) bounds every unseen doc, so
stop as soon as the current k-th best >= tau. TA is instance-optimal for
monotone aggregations, like WEIGHTED fusion with max-normalisation.

No-Random-Access (NRA): sorted access only. Keep a lower bound (known scores)
and an upper bound (known + last-seen score for missing sources) per doc and
stop when k docs have lower bounds no other doc can beat.
"""
from __future__ import annotations

import heapq
import math

import numpy as np

from omnisearch.query.fusion import contribution


class Source:
    """Sorted + random access over one ranked clause.

    ``fetch(n)`` returns the (approximate, for ANN) best n candidates; sorted
    access re-fetches with doubled n when the buffer runs out.
    """

    def __init__(self, name, weight, fetch, random_access, norm="max", first_batch=32):
        self.name = name
        self.weight = weight
        self._fetch = fetch
        self._random = random_access
        self.norm = norm
        self._want = first_batch
        self._ids: list[int] = []
        self._scores: list[float] = []
        self._buffered: set[int] = set()
        self.pos = 0
        self.exhausted = False
        self.top = None
        self.last = math.inf
        self.sorted_accesses = 0
        self.random_accesses = 0
        self.fetches = 0

    def _refill(self):
        ids, scores = self._fetch(self._want)
        self.fetches += 1
        added = 0
        for d, s in zip(ids.tolist(), scores.tolist()):
            if d not in self._buffered:
                self._buffered.add(d)
                self._ids.append(d)
                self._scores.append(s)
                added += 1
        if len(ids) < self._want or added == 0:
            self.exhausted = True
        self._want *= 2

    def next(self):
        while self.pos >= len(self._ids):
            if self.exhausted:
                return None
            self._refill()
        d, s = self._ids[self.pos], self._scores[self.pos]
        self.pos += 1
        self.sorted_accesses += 1
        if self.top is None:
            self.top = s
        self.last = min(self.last, s)
        return d, s

    @property
    def done(self) -> bool:
        return self.exhausted and self.pos >= len(self._ids)

    def random(self, ids) -> np.ndarray:
        self.random_accesses += len(ids)
        return self._random(np.asarray(ids, dtype=np.int64))

    def contrib(self, score) -> float:
        return contribution(score, self.top or 0.0, self.weight, self.norm)

    def bound(self) -> float:
        """Max contribution any not-yet-seen doc can get from this source."""
        if self.done:
            return 0.0
        if self.top is None:
            return self.weight * (1.0 if self.norm == "max" else math.inf)
        return self.contrib(self.last)


def _prime(sources):
    for s in sources:  # first sorted access sets the normalisation constant
        if s.top is None and not s.done and s.next() is not None:
            s.pos -= 1
            s.sorted_accesses -= 1


def threshold_algorithm(sources: list[Source], k: int, block: int = 8):
    _prime(sources)
    totals: dict[int, float] = {}
    heap: list[tuple[float, int]] = []
    rounds = 0
    while True:
        rounds += 1
        new: list[int] = []
        first: dict[int, tuple[int, float]] = {}
        for j, src in enumerate(sources):
            for _ in range(block):
                item = src.next()
                if item is None:
                    break
                d, s = item
                if d not in totals and d not in first:
                    first[d] = (j, s)
                    new.append(d)
        if new:
            parts = np.zeros(len(new))
            for j, src in enumerate(sources):
                need = [i for i, d in enumerate(new) if first[d][0] != j]
                if need:
                    raw = src.random([new[i] for i in need])
                    for i, r in zip(need, raw.tolist()):
                        parts[i] += src.contrib(r)
                for i, d in enumerate(new):
                    if first[d][0] == j:
                        parts[i] += src.contrib(first[d][1])
            for d, total in zip(new, parts.tolist()):
                totals[d] = total
                if len(heap) < k:
                    heapq.heappush(heap, (total, -d))
                elif (total, -d) > heap[0]:  # equal scores: the lower doc id wins
                    heapq.heapreplace(heap, (total, -d))
        tau = sum(s.bound() for s in sources)
        if (len(heap) >= k and heap[0][0] >= tau) or all(s.done for s in sources):
            break
    best = sorted(heap, key=lambda x: (-x[0], -x[1]))
    stats = {
        "rounds": rounds,
        "sorted_accesses": sum(s.sorted_accesses for s in sources),
        "random_accesses": sum(s.random_accesses for s in sources),
        "docs_seen": len(totals),
        "depth": max(s.pos for s in sources),
        "threshold": round(tau, 4),
    }
    return [-d for _, d in best], [t for t, _ in best], stats


def nra(sources: list[Source], k: int, block: int = 8):
    _prime(sources)
    m = len(sources)
    index: dict[int, int] = {}
    lower = np.zeros(1024)
    seen = np.zeros((1024, m), dtype=bool)
    rounds = 0
    while True:
        rounds += 1
        for j, src in enumerate(sources):
            for _ in range(block):
                item = src.next()
                if item is None:
                    break
                d, s = item
                i = index.get(d)
                if i is None:
                    i = index[d] = len(index)
                    if i >= len(lower):
                        lower = np.concatenate([lower, np.zeros(len(lower))])
                        seen = np.concatenate([seen, np.zeros_like(seen)])
                if not seen[i, j]:
                    seen[i, j] = True
                    lower[i] += src.contrib(s)
        n = len(index)
        bounds = np.array([s.bound() for s in sources])
        lo = lower[:n]
        upper = lo + (~seen[:n]) @ bounds
        if n >= k:
            order = np.argsort(-lo, kind="stable")
            kth = lo[order[k - 1]]
            rest = order[k:]
            best_other = max(upper[rest].max() if len(rest) else 0.0, bounds.sum())
            if kth >= best_other:
                break
        if all(s.done for s in sources):
            break
    docs = list(index)
    order = np.lexsort((np.array(docs), -lower[: len(index)]))[:k]
    stats = {
        "rounds": rounds,
        "sorted_accesses": sum(s.sorted_accesses for s in sources),
        "random_accesses": 0,
        "docs_seen": len(index),
        "depth": max(s.pos for s in sources),
    }
    return [docs[i] for i in order], [float(lower[i]) for i in order], stats
