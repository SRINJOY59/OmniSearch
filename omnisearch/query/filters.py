"""Filter evaluation (three-valued logic) and selectivity estimation.

Predicates evaluate to a pair of boolean masks over doc ids: (TRUE, FALSE).
Anything in neither is UNKNOWN (a NULL attribute), so ``NOT (price < 80)``
does not return documents that have no price, matching SQL semantics.

Each numeric range predicate picks its own access path - a vectorised column
scan or a B+-tree index scan - from the estimated selectivity, the classic
"seq scan vs index scan" decision.
"""
from __future__ import annotations

import numpy as np

from omnisearch.query.ast import And, Between, Compare, Contains, InList, IsNull, Not, Or, to_text


def _norm(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


# ----------------------------------------------------------------- estimation
def estimate(node, coll) -> float:
    if isinstance(node, And):
        s = 1.0
        for it in node.items:
            s *= estimate(it, coll)
        return s
    if isinstance(node, Or):
        s = 0.0
        for it in node.items:
            e = estimate(it, coll)
            s = s + e - s * e
        return s
    if isinstance(node, Not):
        return max(0.0, 1.0 - estimate(node.item, coll))
    if isinstance(node, Contains):
        inv = coll.text[node.field]
        toks = list(dict.fromkeys(inv.analyzer(node.text)))
        if not toks or not inv.n_docs:
            return 1.0
        sels = sorted(inv.df(t) / inv.n_docs for t in toks)
        s = 1.0
        for i, x in enumerate(sels):  # exponential backoff for correlated terms
            s *= x ** (1.0 / (2 ** i))
        return s
    idx = coll.attrs[node.field]
    if idx.kind == "category":
        if isinstance(node, Compare):
            s = idx.selectivity([_norm(node.value)])
            return s if node.op == "=" else max(0.0, idx.known(idx.n).mean() - s)
        if isinstance(node, InList):
            s = idx.selectivity([_norm(v) for v in node.values])
            return max(0.0, idx.known(idx.n).mean() - s) if node.negated else s
        if isinstance(node, IsNull):
            nf = 1.0 - (idx.known(idx.n).mean() if idx.n else 1.0)
            return 1.0 - nf if node.negated else nf
        return 1.0
    st = idx.stats
    if isinstance(node, Compare):
        v = node.value
        return {
            "=": lambda: st.eq(v),
            "!=": lambda: max(0.0, 1.0 - st.null_frac - st.eq(v)),
            "<": lambda: st.range(None, v, hi_inclusive=False),
            "<=": lambda: st.range(None, v),
            ">": lambda: st.range(v, None, lo_inclusive=False),
            ">=": lambda: st.range(v, None),
        }[node.op]()
    if isinstance(node, Between):
        s = st.range(node.lo, node.hi)
        return max(0.0, 1.0 - st.null_frac - s) if node.negated else s
    if isinstance(node, InList):
        s = min(1.0, sum(st.eq(v) for v in node.values))
        return max(0.0, 1.0 - st.null_frac - s) if node.negated else s
    if isinstance(node, IsNull):
        return 1.0 - st.null_frac if node.negated else st.null_frac
    return 1.0


# ----------------------------------------------------------------- evaluation
class FilterEvaluator:
    def __init__(self, coll, n: int, access: str = "auto"):
        self.coll = coll
        self.n = n
        self.access = access
        self.decisions: list[dict] = []

    def __call__(self, node) -> np.ndarray:
        t, _ = self.eval(node)
        return t

    def eval(self, node):
        n = self.n
        if isinstance(node, And):
            t, f = np.ones(n, bool), np.zeros(n, bool)
            for it in node.items:
                ti, fi = self.eval(it)
                t &= ti
                f |= fi
            return t, f
        if isinstance(node, Or):
            t, f = np.zeros(n, bool), np.ones(n, bool)
            for it in node.items:
                ti, fi = self.eval(it)
                t |= ti
                f &= fi
            return t, f
        if isinstance(node, Not):
            t, f = self.eval(node.item)
            return f, t
        if isinstance(node, Contains):
            ids = self.coll.text[node.field].contains_all(node.text)
            t = np.zeros(n, bool)
            if ids is None:
                t[:] = True
            else:
                t[ids[ids < n]] = True
            self.decisions.append({"predicate": to_text(node), "access": "posting-list intersection"})
            return t, ~t
        idx = self.coll.attrs[node.field]
        known = idx.known(n)
        if isinstance(node, IsNull):
            t = ~known
            return (~t, t) if node.negated else (t, ~t)
        if idx.kind == "category":
            if isinstance(node, Compare):
                eq = idx.eq(_norm(node.value), n)
                t = eq if node.op == "=" else known & ~eq
            else:
                hit = idx.isin([_norm(v) for v in node.values], n)
                t = known & ~hit if node.negated else hit
            self.decisions.append({"predicate": to_text(node), "access": "bitmap"})
            return t, known & ~t
        t = self._numeric(node, idx, n)
        return t, known & ~t

    def _numeric(self, node, idx, n):
        if isinstance(node, InList):
            t = np.zeros(n, bool)
            for v in node.values:
                t |= idx.scan("=", v, n)
            if node.negated:
                t = idx.known(n) & ~t
            self.decisions.append({"predicate": to_text(node), "access": "column scan"})
            return t
        if isinstance(node, Compare) and node.op in ("=", "!="):
            self.decisions.append({"predicate": to_text(node), "access": "column scan"})
            return idx.scan(node.op, node.value, n)
        if isinstance(node, Between):
            lo, hi, li, hi_inc = node.lo, node.hi, True, True
        else:
            v, op = node.value, node.op
            lo, hi = (v, None) if op in (">", ">=") else (None, v)
            li, hi_inc = op == ">=", op == "<="
        sel = estimate(node, self.coll) if not getattr(node, "negated", False) else 1.0
        cm = self.coll.cost
        scan_cost = n * cm.col_row
        tree_cost = cm.op_overhead + sel * n * cm.btree_row
        use_tree = self.access == "index" or (self.access == "auto" and tree_cost < scan_cost)
        if getattr(node, "negated", False):
            use_tree = False
        if use_tree:
            t = idx.tree_range(lo, hi, li, hi_inc, n)
        elif isinstance(node, Between):
            t = idx.scan_between(lo, hi, n)
        else:
            t = idx.scan(node.op, node.value, n)
        if getattr(node, "negated", False):
            t = idx.known(n) & ~t
        self.decisions.append({
            "predicate": to_text(node),
            "access": "B+-tree index scan" if use_tree else "column scan",
            "est_sel": sel, "cost_scan_ms": scan_cost * 1e3, "cost_index_ms": tree_cost * 1e3,
        })
        return t
