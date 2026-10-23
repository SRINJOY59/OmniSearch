"""Text analysis: normalisation, tokenisation, stop words and a light stemmer.

The same analyzer runs at index and query time, so "Sneakers" in a query
matches "sneaker" in a document.
"""
from __future__ import annotations

import re
import unicodedata

_TOKEN = re.compile(r"[a-z0-9]+")

# Deliberately small: "no", "not", "off" etc. are kept because they matter in
# OCR'd signs ("no parking") and product text.
STOPWORDS = frozenset(
    """a an and are as at be been but by for from has have had he her his i if in
    into is it its of on or our she so such than that the their them then there
    these they this those to was we were what when where which while who will
    with you your""".split()
)


def stem(tok: str) -> str:
    """Conservative suffix stripping (plurals, -ing, -ed)."""
    if len(tok) <= 3 or tok.isdigit():
        return tok
    if tok.endswith("ies") and len(tok) > 4:
        return tok[:-3] + "y"
    if tok.endswith("sses"):
        return tok[:-2]
    if tok.endswith("es") and tok[-3] in "sxz" and len(tok) > 4:
        return tok[:-2]
    if tok.endswith("s") and not tok.endswith(("ss", "us", "is")):
        tok = tok[:-1]
    if tok.endswith("ing") and len(tok) > 5:
        tok = tok[:-3]
        if len(tok) > 2 and tok[-1] == tok[-2] and tok[-1] not in "lsz":
            tok = tok[:-1]
    elif tok.endswith("ed") and len(tok) > 4 and not tok.endswith("eed"):
        tok = tok[:-2]
        if len(tok) > 2 and tok[-1] == tok[-2] and tok[-1] not in "lsz":
            tok = tok[:-1]
    return tok


class Analyzer:
    def __init__(self, stem: bool = True, stopwords: bool = True):
        self.use_stem = stem
        self.use_stopwords = stopwords

    def __call__(self, text: str) -> list[str]:
        if not text:
            return []
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
        toks = _TOKEN.findall(text)
        if self.use_stopwords:
            toks = [t for t in toks if t not in STOPWORDS]
        if self.use_stem:
            toks = [stem(t) for t in toks]
        return toks
