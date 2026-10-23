"""Syntax tree for OmniQL.

    [EXPLAIN [ANALYZE]] SEARCH <collection>
      [WHERE <condition>]
      [FUSE RRF(k=60) | WEIGHTED(norm='max') | AUTO]
      [LIMIT n]
      [WITH (option=value, ...)]

Scoring clauses ``MATCH`` (keyword) and ``SIMILAR`` (vector) rank documents;
every other condition is a boolean filter.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Compare:
    field: str
    op: str  # = != < <= > >=
    value: object


@dataclass
class InList:
    field: str
    values: list
    negated: bool = False


@dataclass
class Between:
    field: str
    lo: object
    hi: object
    negated: bool = False


@dataclass
class IsNull:
    field: str
    negated: bool = False


@dataclass
class Contains:
    """Hard keyword filter: the field contains every query term."""

    field: str
    text: str


@dataclass
class And:
    items: list


@dataclass
class Or:
    items: list


@dataclass
class Not:
    item: object


@dataclass
class Match:
    field: str
    text: str
    opts: dict = field(default_factory=dict)


@dataclass
class QueryItem:
    kind: str  # text | image | audio | vector | doc
    value: object
    weight: float = 1.0


@dataclass
class Similar:
    field: str
    items: list
    opts: dict = field(default_factory=dict)


@dataclass
class FuseSpec:
    method: str = "rrf"  # rrf | weighted | auto
    params: dict = field(default_factory=dict)


@dataclass
class SearchStmt:
    collection: str
    where: object = None
    fuse: FuseSpec = field(default_factory=FuseSpec)
    limit: int = 10
    options: dict = field(default_factory=dict)
    explain: bool = False
    analyze: bool = False


RANKED = (Match, Similar)


def to_text(node) -> str:
    """Render a condition back to OmniQL (used by EXPLAIN)."""
    def lit(v):
        if isinstance(v, str):
            return "'" + v.replace("'", "''") + "'"
        if isinstance(v, bool):
            return "TRUE" if v else "FALSE"
        return repr(v)

    if isinstance(node, Compare):
        return f"{node.field} {node.op} {lit(node.value)}"
    if isinstance(node, InList):
        return f"{node.field} {'NOT ' if node.negated else ''}IN ({', '.join(lit(v) for v in node.values)})"
    if isinstance(node, Between):
        return f"{node.field} {'NOT ' if node.negated else ''}BETWEEN {lit(node.lo)} AND {lit(node.hi)}"
    if isinstance(node, IsNull):
        return f"{node.field} IS {'NOT ' if node.negated else ''}NULL"
    if isinstance(node, Contains):
        return f"CONTAINS({node.field}, {lit(node.text)})"
    if isinstance(node, And):
        return " AND ".join(f"({to_text(i)})" if isinstance(i, Or) else to_text(i) for i in node.items)
    if isinstance(node, Or):
        return " OR ".join(f"({to_text(i)})" if isinstance(i, And) else to_text(i) for i in node.items)
    if isinstance(node, Not):
        return f"NOT ({to_text(node.item)})"
    if isinstance(node, Match):
        return f"MATCH({node.field}, {lit(node.text)})"
    if isinstance(node, Similar):
        parts = []
        for it in node.items:
            w = "" if it.weight == 1.0 else f"{it.weight:g}*"
            v = lit(it.value) if it.kind == "text" else (
                f"{it.kind.upper()}({lit(it.value)})" if it.kind != "vector" else f"VECTOR[{len(it.value)} dims]")
            parts.append(w + v)
        return f"SIMILAR({node.field}, {' + '.join(parts)})"
    return str(node)
