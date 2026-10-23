"""Rank fusion: combine the ranked lists produced by MATCH / SIMILAR clauses.

* RRF (Reciprocal Rank Fusion): score(d) = sum_i w_i / (k + rank_i(d)).
  Uses ranks only, so BM25 and cosine scales never need reconciling.
* WEIGHTED: score(d) = sum_i w_i * norm_i(s_i(d)). With ``norm='max'``
  (divide by the list's top score) the aggregation is monotone in every input,
  which is what lets the Threshold Algorithm stop early.
* AUTO: RRF with query-adaptive weights - specific, code-like keyword queries
  ("SKU 4471-B") lean on BM25; vague descriptive queries lean on vectors.
"""
from __future__ import annotations

import math
import re

import numpy as np


def contribution(score, top, weight, norm):
    """Normalised, weighted, non-negative contribution of one raw score."""
    if not np.isfinite(score):
        return 0.0
    if norm == "max":
        return weight * max(0.0, score / top) if top > 0 else 0.0
    return weight * max(0.0, score)


def rrf(lists: dict, weights: dict, k: int = 60):
    """lists: name -> (ids, scores) best-first. Returns doc -> fused score, doc -> {name: rank}."""
    fused: dict[int, float] = {}
    ranks: dict[int, dict] = {}
    for name, (ids, _) in lists.items():
        w = weights.get(name, 1.0)
        for r, d in enumerate(ids.tolist(), start=1):
            fused[d] = fused.get(d, 0.0) + w / (k + r)
            ranks.setdefault(d, {})[name] = r
    return fused, ranks


def weighted(lists: dict, completed: dict, weights: dict, norm: str = "max"):
    """Score-based fusion over the union of candidates.

    completed: name -> {doc: raw score} for every candidate (filled by random access).
    """
    fused: dict[int, float] = {}
    ranks: dict[int, dict] = {}
    for name, (ids, scores) in lists.items():
        for r, d in enumerate(ids.tolist(), start=1):
            ranks.setdefault(d, {})[name] = r
    docs = list(ranks)
    for name, (ids, scores) in lists.items():
        w = weights.get(name, 1.0)
        raw = completed[name]
        vals = np.array([raw.get(d, -np.inf) for d in docs], dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        if norm == "minmax" and len(finite):
            lo, hi = finite.min(), finite.max()
            normed = np.where(np.isfinite(vals), (vals - lo) / (hi - lo) if hi > lo else 1.0, 0.0)
            contrib = w * normed
        else:
            top = float(scores[0]) if len(scores) else 0.0
            contrib = np.array([contribution(v, top, w, norm) for v in vals])
        for d, c in zip(docs, contrib.tolist()):
            fused[d] = fused.get(d, 0.0) + c
    return fused, ranks


_CODE = re.compile(r"\b(?=\w*\d)\w{2,}\b|\"|-\d")


def auto_weights(clauses, bound, coll) -> dict:
    """Query-adaptive clause weights (multiplied with any user-given weights)."""
    kw = [c for c in clauses if c.kind == "keyword"]
    other = [c for c in clauses if c.kind != "keyword"]
    if not kw or not other:
        return {c.name: c.weight for c in clauses}
    specs = []
    code_like = False
    for c in kw:
        inv = coll.text[c.field]
        terms = bound[c.name]["terms"]
        if terms and inv.n_docs:
            specs.append(np.mean([inv.idf(t) for t, _ in terms]) / math.log(1 + inv.n_docs))
        code_like |= bool(_CODE.search(c.node.text))
    spec = float(np.mean(specs)) if specs else 0.0
    w_kw = min(0.9, max(0.1, 0.2 + 0.6 * spec + (0.2 if code_like else 0.0)))
    out = {}
    for c in kw:
        out[c.name] = c.weight * w_kw / len(kw)
    for c in other:
        out[c.name] = c.weight * (1.0 - w_kw) / len(other)
    return out
