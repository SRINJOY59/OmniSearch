"""Exact (brute-force) vector search, optionally over compressed codes.

With a bitmap filter this is the "pre-filter" plan: only the allowed vectors
are scored, so its cost is proportional to the filter's selectivity.
"""
from __future__ import annotations

import numpy as np

from omnisearch.util import topk


def score_candidates(store, quantizer, refine, q, cand, k):
    """Score candidate positions (None = all), using codes + exact re-ranking if available."""
    approx = quantizer is not None and quantizer.trained and len(quantizer.codes) == len(store)
    scores = (quantizer.score if approx else store.score)(q, cand)
    n = len(scores)
    if approx and refine:
        top = topk(scores, k * refine)
        pos = top if cand is None else cand[top]
        exact = store.score(q, pos)
        o = topk(exact, k)
        return pos[o], exact[o], n
    top = topk(scores, k)
    pos = top if cand is None else cand[top]
    return pos, scores[top], n


class FlatIndex:
    def __init__(self, store, quantizer=None, refine: int = 4):
        self.store = store
        self.quantizer = quantizer
        self.refine = refine

    def search(self, q, k, allow: np.ndarray | None = None):
        cand = None if allow is None else np.flatnonzero(allow)
        pos, scores, n = score_candidates(self.store, self.quantizer, self.refine, q, cand, k)
        return pos, scores, {"scanned": n}
