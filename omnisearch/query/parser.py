"""Recursive-descent parser for OmniQL.

Grammar (informal)::

    statement  := [EXPLAIN [ANALYZE]] SEARCH ident [WHERE expr] [FUSE fuse]
                  [LIMIT int] [WITH '(' kv {',' kv} ')'] [';']
    expr       := and_expr {OR and_expr}
    and_expr   := not_expr {AND not_expr}
    not_expr   := NOT not_expr | '(' expr ')' | predicate
    predicate  := MATCH '(' ident ',' string {',' kv} ')'
                | SIMILAR '(' ident ',' qexpr {',' kv} ')'
                | CONTAINS '(' ident ',' string ')'
                | ident cmp literal | ident [NOT] IN '(' literals ')'
                | ident [NOT] BETWEEN literal AND literal | ident IS [NOT] NULL
    qexpr      := qterm {('+'|'-') qterm}
    qterm      := [number '*'] (string | IMAGE(string) | AUDIO(string)
                                | DOC(string) | VECTOR '[' numbers ']')
    fuse       := RRF ['(' kvs ')'] | WEIGHTED ['(' kvs ')'] | AUTO
"""
from __future__ import annotations

from omnisearch.query.ast import (
    And, Between, Compare, Contains, FuseSpec, InList, IsNull, Match, Not, Or,
    QueryItem, SearchStmt, Similar,
)
from omnisearch.query.lexer import QuerySyntaxError, Token, tokenize

_CMP = {"=", "!=", "<", "<=", ">", ">="}
_MEDIA_FUNCS = {"IMAGE": "image", "AUDIO": "audio", "DOC": "doc"}


class Parser:
    def __init__(self, text: str):
        self.text = text
        self.toks = tokenize(text)
        self.i = 0

    # ------------------------------------------------------------ helpers
    @property
    def tok(self) -> Token:
        return self.toks[self.i]

    def error(self, msg):
        raise QuerySyntaxError(msg, self.tok.pos, self.text)

    def advance(self) -> Token:
        t = self.tok
        self.i += 1
        return t

    def is_kw(self, *kws):
        return self.tok.type == "KW" and self.tok.value in kws

    def is_op(self, *ops):
        return self.tok.type == "OP" and self.tok.value in ops

    def is_func(self, name):
        return (self.tok.type == "IDENT" and self.tok.value.upper() == name
                and self.toks[self.i + 1].type == "OP" and self.toks[self.i + 1].value in "([")

    def expect_kw(self, kw):
        if not self.is_kw(kw):
            self.error(f"expected {kw}")
        return self.advance()

    def expect_op(self, op):
        if not self.is_op(op):
            self.error(f"expected '{op}'")
        return self.advance()

    def ident(self) -> str:
        if self.tok.type != "IDENT":
            self.error("expected a name")
        return self.advance().value

    def string(self) -> str:
        if self.tok.type != "STRING":
            self.error("expected a quoted string")
        return self.advance().value

    def number(self):
        neg = False
        if self.is_op("-"):
            self.advance()
            neg = True
        if self.tok.type != "NUMBER":
            self.error("expected a number")
        v = self.advance().value
        return -v if neg else v

    def literal(self):
        if self.tok.type == "STRING":
            return self.advance().value
        if self.is_kw("TRUE", "FALSE"):
            return self.advance().value == "TRUE"
        return self.number()

    def kv(self) -> tuple[str, object]:
        key = self.ident().lower()
        self.expect_op("=")
        if self.tok.type == "IDENT":
            return key, self.advance().value
        return key, self.literal()

    # -------------------------------------------------------- statement
    def parse(self) -> SearchStmt:
        explain = analyze = False
        if self.is_kw("EXPLAIN"):
            self.advance()
            explain = True
            if self.is_kw("ANALYZE"):
                self.advance()
                analyze = True
        self.expect_kw("SEARCH")
        stmt = SearchStmt(collection=self.ident(), explain=explain, analyze=analyze)
        if self.is_kw("WHERE"):
            self.advance()
            stmt.where = self.expr()
        if self.is_kw("FUSE"):
            self.advance()
            stmt.fuse = self.fuse()
        if self.is_kw("LIMIT"):
            self.advance()
            n = self.number()
            if not isinstance(n, int) or n <= 0:
                self.error("LIMIT must be a positive integer")
            stmt.limit = n
        if self.is_kw("WITH"):
            self.advance()
            self.expect_op("(")
            stmt.options = dict([self.kv()])
            while self.is_op(","):
                self.advance()
                stmt.options.update([self.kv()])
            self.expect_op(")")
        if self.is_op(";"):
            self.advance()
        if self.tok.type != "EOF":
            self.error("unexpected input")
        return stmt

    def fuse(self) -> FuseSpec:
        name = self.ident().lower()
        if name not in ("rrf", "weighted", "auto"):
            self.error("FUSE must be RRF, WEIGHTED or AUTO")
        params = {}
        if self.is_op("("):
            self.advance()
            if not self.is_op(")"):
                params.update([self.kv()])
                while self.is_op(","):
                    self.advance()
                    params.update([self.kv()])
            self.expect_op(")")
        return FuseSpec(name, params)

    # ------------------------------------------------------- conditions
    def expr(self):
        items = [self.and_expr()]
        while self.is_kw("OR"):
            self.advance()
            items.append(self.and_expr())
        return items[0] if len(items) == 1 else Or(items)

    def and_expr(self):
        items = [self.not_expr()]
        while self.is_kw("AND"):
            self.advance()
            items.append(self.not_expr())
        flat = []
        for it in items:
            flat.extend(it.items if isinstance(it, And) else [it])
        return flat[0] if len(flat) == 1 else And(flat)

    def not_expr(self):
        if self.is_kw("NOT"):
            self.advance()
            return Not(self.not_expr())
        if self.is_op("("):
            self.advance()
            e = self.expr()
            self.expect_op(")")
            return e
        return self.predicate()

    def predicate(self):
        if self.is_func("MATCH"):
            self.advance()
            self.expect_op("(")
            f = self.ident()
            self.expect_op(",")
            text = self.string()
            return Match(f, text, self.trailing_opts())
        if self.is_func("SIMILAR"):
            self.advance()
            self.expect_op("(")
            f = self.ident()
            self.expect_op(",")
            items = self.qexpr()
            return Similar(f, items, self.trailing_opts())
        if self.is_func("CONTAINS"):
            self.advance()
            self.expect_op("(")
            f = self.ident()
            self.expect_op(",")
            text = self.string()
            self.expect_op(")")
            return Contains(f, text)
        f = self.ident()
        negated = False
        if self.is_kw("NOT"):
            self.advance()
            negated = True
            if not self.is_kw("IN", "BETWEEN"):
                self.error("expected IN or BETWEEN after NOT")
        if self.is_kw("IN"):
            self.advance()
            self.expect_op("(")
            vals = [self.literal()]
            while self.is_op(","):
                self.advance()
                vals.append(self.literal())
            self.expect_op(")")
            return InList(f, vals, negated)
        if self.is_kw("BETWEEN"):
            self.advance()
            lo = self.literal()
            self.expect_kw("AND")
            return Between(f, lo, self.literal(), negated)
        if self.is_kw("IS"):
            self.advance()
            neg = False
            if self.is_kw("NOT"):
                self.advance()
                neg = True
            self.expect_kw("NULL")
            return IsNull(f, neg)
        if self.tok.type == "OP" and self.tok.value in _CMP:
            op = self.advance().value
            return Compare(f, op, self.literal())
        self.error("expected a comparison")

    def trailing_opts(self) -> dict:
        opts = {}
        while self.is_op(","):
            self.advance()
            opts.update([self.kv()])
        self.expect_op(")")
        return opts

    def qexpr(self) -> list[QueryItem]:
        items = [self.qterm(1.0)]
        while self.is_op("+", "-"):
            sign = -1.0 if self.advance().value == "-" else 1.0
            items.append(self.qterm(sign))
        return items

    def qterm(self, sign) -> QueryItem:
        weight = 1.0
        if self.tok.type == "NUMBER" and self.toks[self.i + 1].type == "OP" and self.toks[self.i + 1].value == "*":
            weight = float(self.advance().value)
            self.advance()
        if self.tok.type == "STRING":
            return QueryItem("text", self.advance().value, sign * weight)
        for name, kind in _MEDIA_FUNCS.items():
            if self.is_func(name):
                self.advance()
                self.expect_op("(")
                v = self.string()
                self.expect_op(")")
                return QueryItem(kind, v, sign * weight)
        if self.is_func("VECTOR"):
            self.advance()
            self.expect_op("[")
            vals = [float(self.number())]
            while self.is_op(","):
                self.advance()
                vals.append(float(self.number()))
            self.expect_op("]")
            return QueryItem("vector", vals, sign * weight)
        self.error("expected a query: 'text', IMAGE('path'), AUDIO('path'), DOC('id') or VECTOR[...]")


def parse(text: str) -> SearchStmt:
    return Parser(text).parse()
