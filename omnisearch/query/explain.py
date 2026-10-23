"""EXPLAIN / EXPLAIN ANALYZE rendering (text tree and JSON)."""
from __future__ import annotations

from omnisearch.query.ast import to_text


def _ms(seconds):
    return f"{seconds * 1e3:.3f} ms"


def _actual_branch(a: dict) -> str:
    if not a:
        return ""
    keys = [("ms", "{:.2f} ms"), ("rows", "{} rows"), ("postings", "{} postings"),
            ("evaluated", "{} docs scored"), ("expansions", "{} node expansions"),
            ("scanned", "{} vectors scanned"), ("lists", "{} lists probed"), ("fetched", "fetched {}"),
            ("sorted", "{} sorted accesses"), ("random", "{} random accesses")]
    parts = [fmt.format(round(a[k], 2) if isinstance(a[k], float) else a[k]) for k, fmt in keys if k in a]
    return "  [actual: " + ", ".join(parts) + "]"


def render(result) -> str:
    plan = result.plan
    lp = plan.logical
    analyze = bool(result.timings.get("search_ms") is not None)
    lines = [f"Search on {lp.collection}  (doc ids: {plan.n_docs}, est. cost {_ms(plan.est_cost)})"]
    lines.append(f"-> Limit {plan.limit}")
    f = plan.fuse
    if f.algorithm == "none":
        lines.append("  -> Filter-only scan (no scoring clauses)")
    else:
        desc = {"rrf": f"RRF(k={f.params.get('k', 60)})", "weighted": f"WEIGHTED(norm={f.params.get('norm', 'max')})",
                "auto": "AUTO (query-adaptive RRF)"}[f.method]
        algo = {"single": "single list", "full": f"materialise depth {f.depth} + merge",
                "ta": "Threshold Algorithm", "nra": "No-Random-Access", "exhaustive": "score every doc"}[f.algorithm]
        act = ""
        if analyze and f.actual:
            extras = {k: v for k, v in f.actual.items() if k != "ms"}
            act = f"  [actual: {f.actual.get('ms', 0):.2f} ms" + (", " + ", ".join(f"{k}={v}" for k, v in extras.items()) if extras else "") + "]"
        lines.append(f"  -> Fuse {desc}  algorithm: {algo}  est {_ms(f.est_cost)}{act}")
        if f.alternatives:
            lines.append("       rejected: " + "; ".join(f"{a.strategy} {_ms(a.cost)}" for a in f.alternatives))
        if len(plan.branches) > 1:
            lines.append("       weights: " + "  ".join(f"{k}={v:.2f}" for k, v in f.weights.items()))
        for b in plan.branches:
            c = b.clause
            detail = ""
            if c.kind == "keyword":
                df = b.params.get("df", {})
                detail = "terms: " + (", ".join(f"{t}(df={df.get(t, '?')})" for t, _ in (b.terms or [])) or "(none known)")
            else:
                detail = ", ".join(f"{k}={v}" for k, v in b.params.items())
            detail = f"  {detail}" if detail else ""
            lines.append(f"     -> [{c.name}] {b.operator}  k={b.k}{detail}  est {_ms(b.est_cost)}"
                         + (_actual_branch(b.actual) if analyze else ""))
            lines.append(f"          {to_text(c.node)}")
            if b.alternatives:
                lines.append("          rejected: " + "; ".join(
                    f"{a.strategy} {_ms(a.cost)}" + (f" ({a.note})" if a.note else "") for a in b.alternatives))
    if plan.filter is not None:
        fp = plan.filter
        act = ""
        if analyze and fp.actual:
            act = f"  [actual: {fp.actual['rows']} rows, sel {fp.actual['sel']:.2%}, {fp.actual.get('ms', 0):.2f} ms]"
        lines.append(f"  -> Filter: {to_text(fp.expr)}  est sel {fp.est_sel:.2%}{act}")
        for d in fp.actual.get("decisions", []):
            extra = ""
            if "cost_scan_ms" in d:
                extra = f"  (scan {d['cost_scan_ms']:.3f} ms vs index {d['cost_index_ms']:.3f} ms, est sel {d['est_sel']:.2%})"
            lines.append(f"       {d['predicate']}: {d['access']}{extra}")
    if analyze:
        t = result.timings
        lines.append("Timing: " + "  ".join(f"{k[:-3]}={v:.2f}ms" for k, v in t.items()))
    return "\n".join(lines)


def plan_to_dict(result) -> dict:
    plan = result.plan
    return {
        "collection": plan.logical.collection,
        "est_cost_ms": plan.est_cost * 1e3,
        "limit": plan.limit,
        "fuse": {"method": plan.fuse.method, "algorithm": plan.fuse.algorithm, "depth": plan.fuse.depth,
                 "weights": plan.fuse.weights, "est_cost_ms": plan.fuse.est_cost * 1e3, "actual": plan.fuse.actual,
                 "rejected": [{"algorithm": a.strategy, "est_cost_ms": a.cost * 1e3} for a in plan.fuse.alternatives]},
        "branches": [{
            "name": b.clause.name, "kind": b.clause.kind, "operator": b.operator, "strategy": b.strategy,
            "params": b.params, "k": b.k, "est_cost_ms": b.est_cost * 1e3, "actual": b.actual,
            "clause": to_text(b.clause.node),
            "terms": [t for t, _ in (b.terms or [])],
            "rejected": [{"strategy": a.strategy, "est_cost_ms": a.cost * 1e3, "note": a.note} for a in b.alternatives],
        } for b in plan.branches],
        "filter": None if plan.filter is None else {
            "expr": to_text(plan.filter.expr), "est_sel": plan.filter.est_sel,
            "actual": {k: v for k, v in plan.filter.actual.items()},
        },
    }
