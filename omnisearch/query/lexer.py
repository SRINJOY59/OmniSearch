"""Tokenizer for OmniQL."""
from __future__ import annotations

import re
from dataclasses import dataclass

KEYWORDS = {
    "SEARCH", "WHERE", "AND", "OR", "NOT", "IN", "BETWEEN", "IS", "NULL", "FUSE",
    "LIMIT", "WITH", "EXPLAIN", "ANALYZE", "TRUE", "FALSE",
}


class QuerySyntaxError(ValueError):
    def __init__(self, msg, pos=None, text=None):
        if pos is not None and text is not None:
            msg = f"{msg} at position {pos}: ...{text[max(0, pos - 15):pos + 15]!r}"
        super().__init__(msg)


@dataclass
class Token:
    type: str  # KW | IDENT | STRING | NUMBER | OP | EOF
    value: object
    pos: int


_SPEC = [
    ("WS", r"\s+"),
    ("COMMENT", r"--[^\n]*"),
    ("NUMBER", r"\d+\.\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?|\d+(?:[eE][+-]?\d+)?"),
    ("STRING", r"'(?:[^']|'')*'"),
    ("IDENT", r"[A-Za-z_][A-Za-z0-9_]*"),
    ("OP", r"<=|>=|!=|<>|[=<>(),*+\-\[\];]"),
]
_RE = re.compile("|".join(f"(?P<{n}>{p})" for n, p in _SPEC))


def tokenize(text: str) -> list[Token]:
    out, pos = [], 0
    while pos < len(text):
        m = _RE.match(text, pos)
        if not m:
            raise QuerySyntaxError("unexpected character", pos, text)
        kind, val = m.lastgroup, m.group()
        if kind == "NUMBER":
            out.append(Token("NUMBER", float(val) if any(c in val for c in ".eE") else int(val), pos))
        elif kind == "STRING":
            out.append(Token("STRING", val[1:-1].replace("''", "'"), pos))
        elif kind == "IDENT":
            up = val.upper()
            out.append(Token("KW", up, pos) if up in KEYWORDS else Token("IDENT", val, pos))
        elif kind == "OP":
            out.append(Token("OP", "!=" if val == "<>" else val, pos))
        pos = m.end()
    out.append(Token("EOF", None, len(text)))
    return out
