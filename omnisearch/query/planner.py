"""Semantic analysis: SearchStmt -> LogicalPlan.

Checks every field against the schema, then splits the WHERE clause into
scoring clauses (MATCH / SIMILAR, which must be top-level AND terms) and a
boolean filter (everything else).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from omnisearch.query.ast import (
    RANKED, And, Between, Compare, Contains, FuseSpec, InList, Match, Not, Or, SearchStmt,
)
from omnisearch.schema import AttributeField, MultiVectorField, Schema, TextField, VectorField

OPTIONS = {
    "strategy": {"auto", "exact", "index", "postfilter"},
    "topk": {"auto", "full", "ta", "nra", "exhaustive"},
    "keyword_algo": {"auto", "exhaustive", "wand"},
    "filter_access": {"auto", "scan", "index"},
    "ef": int, "nprobe": int, "depth": int, "overfetch": int,
}


class PlanError(ValueError):
    pass


@dataclass
class RankedClause:
    name: str  # unique label, e.g. "text" or "image_vec#2"
    kind: str  # keyword | vector | multivector
    field: str
    node: object
    weight: float = 1.0
    params: dict = field(default_factory=dict)


@dataclass
class LogicalPlan:
    collection: str
    ranked: list
    filter: object
    fuse: FuseSpec
    limit: int
    options: dict
    explain: bool = False
    analyze: bool = False


def _check_filter(node, schema: Schema):
    if isinstance(node, RANKED):
        raise PlanError("MATCH/SIMILAR must be top-level AND terms; they cannot appear inside OR or NOT")
    if isinstance(node, (And, Or)):
        for it in node.items:
            _check_filter(it, schema)
        return
    if isinstance(node, Not):
        _check_filter(node.item, schema)
        return
    if isinstance(node, Contains):
        if not isinstance(_field(schema, node.field), TextField):
            raise PlanError(f"CONTAINS needs a text field, {node.field!r} is not one")
        return
    f = _field(schema, node.field)
    if not isinstance(f, AttributeField):
        raise PlanError(f"{node.field!r} is not an attribute field")
    numeric = f.type in ("int", "float")
    if isinstance(node, Compare):
        if not numeric and node.op not in ("=", "!="):
            raise PlanError(f"operator {node.op} needs a numeric field, {node.field!r} is {f.type}")
        if numeric and not isinstance(node.value, (int, float)):
            raise PlanError(f"{node.field!r} is numeric; compare it with a number")
    elif isinstance(node, Between) and not numeric:
        raise PlanError(f"BETWEEN needs a numeric field, {node.field!r} is {f.type}")
    elif isinstance(node, InList) and numeric and not all(isinstance(v, (int, float)) for v in node.values):
        raise PlanError(f"{node.field!r} is numeric; IN list must contain numbers")


def _field(schema: Schema, name: str):
    try:
        return schema.get(name)
    except KeyError:
        raise PlanError(f"unknown field {name!r}") from None


def plan(stmt: SearchStmt, schema: Schema) -> LogicalPlan:
    terms = stmt.where.items if isinstance(stmt.where, And) else ([stmt.where] if stmt.where else [])
    ranked, filters = [], []
    for t in terms:
        (ranked if isinstance(t, RANKED) else filters).append(t)

    clauses, names = [], set()
    for node in ranked:
        f = _field(schema, node.field)
        if isinstance(node, Match):
            if not isinstance(f, TextField):
                raise PlanError(f"MATCH needs a text field, {node.field!r} is not one")
            kind = "keyword"
        else:
            if isinstance(f, VectorField):
                kind = "vector"
            elif isinstance(f, MultiVectorField):
                kind = "multivector"
            else:
                raise PlanError(f"SIMILAR needs a vector field, {node.field!r} is not one")
        opts = dict(node.opts)
        weight = float(opts.pop("weight", 1.0))
        name = node.field
        n = 2
        while name in names:
            name, n = f"{node.field}#{n}", n + 1
        names.add(name)
        clauses.append(RankedClause(name, kind, node.field, node, weight, opts))

    flt = None
    if filters:
        flt = filters[0] if len(filters) == 1 else And(filters)
        _check_filter(flt, schema)

    for k, v in stmt.options.items():
        spec = OPTIONS.get(k)
        if spec is None:
            raise PlanError(f"unknown option {k!r}; valid: {sorted(OPTIONS)}")
        if isinstance(spec, set) and v not in spec:
            raise PlanError(f"option {k} must be one of {sorted(spec)}")
        if spec is int and (not isinstance(v, int) or v <= 0):
            raise PlanError(f"option {k} must be a positive integer")

    if stmt.fuse.method == "weighted" and stmt.fuse.params.get("norm", "max") not in ("max", "minmax", "none"):
        raise PlanError("WEIGHTED norm must be 'max', 'minmax' or 'none'")
    if stmt.options.get("topk") in ("ta", "nra") and stmt.fuse.method != "weighted":
        raise PlanError("TA/NRA need score-based fusion: use FUSE WEIGHTED")

    return LogicalPlan(stmt.collection, clauses, flt, stmt.fuse, stmt.limit, dict(stmt.options),
                       stmt.explain, stmt.analyze)
