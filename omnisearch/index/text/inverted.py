"""Inverted index with BM25 scoring.

Three access paths, all used by the query engine:

* ``search(..., algo="exhaustive")`` decodes every query term's posting list and
  accumulates BM25 contributions with numpy (term-at-a-time).
* ``search(..., algo="wand")`` is document-at-a-time WAND: per-term score upper
  bounds let it skip documents that cannot enter the current top-k.
* ``score_docs`` is random access: the BM25 score of given documents, found by
  binary search in the doc-id-sorted posting lists. The Threshold Algorithm
  relies on it.
"""
from __future__ import annotations

import heapq
import json
import math
import struct
from collections import Counter
from pathlib import Path

import numpy as np

from omnisearch.index.text.analyzer import Analyzer
from omnisearch.index.text.postings import PostingList
from omnisearch.util import GrowableArray, topk

_MAGIC = b"OSX1"


class InvertedIndex:
    def __init__(self, analyzer: Analyzer | None = None, k1: float = 1.2, b: float = 0.75):
        self.analyzer = analyzer or Analyzer()
        self.k1 = k1
        self.b = b
        self.postings: dict[str, PostingList] = {}
        self.doc_len = GrowableArray(np.int32)
        self.n_docs = 0
        self.total_len = 0
        self.version = 0
        self._max_score: dict[str, tuple[int, float]] = {}

    # ------------------------------------------------------------------ build
    def add(self, doc_id: int, text: str):
        toks = self.analyzer(text)
        self.doc_len.resize(doc_id + 1)
        if not toks:
            return
        if self.doc_len[doc_id]:
            raise ValueError(f"doc {doc_id} already indexed")
        for term, tf in Counter(toks).items():
            pl = self.postings.get(term)
            if pl is None:
                pl = self.postings[term] = PostingList()
            pl.add(doc_id, tf)
        self.doc_len[doc_id] = len(toks)
        self.n_docs += 1
        self.total_len += len(toks)
        self.version += 1

    def flush(self):
        for pl in self.postings.values():
            pl.flush()

    # ------------------------------------------------------------- statistics
    @property
    def avgdl(self) -> float:
        return self.total_len / self.n_docs if self.n_docs else 1.0

    def df(self, term: str) -> int:
        pl = self.postings.get(term)
        return len(pl) if pl else 0

    def idf(self, term: str) -> float:
        df = self.df(term)
        return math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))

    def query_terms(self, text: str) -> list[tuple[str, int]]:
        """Analyzed query terms with their query frequency, unknown terms dropped."""
        return [(t, c) for t, c in Counter(self.analyzer(text)).items() if t in self.postings]

    def _term_scores(self, term: str, docs: np.ndarray, tfs: np.ndarray) -> np.ndarray:
        dl = self.doc_len.data[docs]
        tf = tfs.astype(np.float32)
        norm = self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)
        return (self.idf(term) * tf * (self.k1 + 1.0) / (tf + norm)).astype(np.float32)

    def max_score(self, term: str) -> float:
        """Upper bound of a term's BM25 contribution (cached per index version)."""
        cached = self._max_score.get(term)
        if cached and cached[0] == self.version:
            return cached[1]
        docs, tfs = self.postings[term].arrays()
        m = float(self._term_scores(term, docs, tfs).max()) if len(docs) else 0.0
        self._max_score[term] = (self.version, m)
        return m

    # ----------------------------------------------------------------- search
    def score_all(self, terms, allow: np.ndarray | None = None):
        """Exhaustive term-at-a-time scoring. Returns (doc ids, scores, postings scored)."""
        n = len(self.doc_len)
        acc = np.zeros(n, dtype=np.float32)
        hit = np.zeros(n, dtype=bool)
        scanned = 0
        for term, qtf in terms:
            docs, tfs = self.postings[term].arrays()
            acc[docs] += qtf * self._term_scores(term, docs, tfs)
            hit[docs] = True
            scanned += len(docs)
        if allow is not None:
            hit &= allow[:n] if len(allow) >= n else np.pad(allow, (0, n - len(allow)))
        ids = np.flatnonzero(hit)
        return ids, acc[ids], scanned

    def search(self, text_or_terms, k: int, allow: np.ndarray | None = None, algo: str = "exhaustive"):
        terms = self.query_terms(text_or_terms) if isinstance(text_or_terms, str) else text_or_terms
        if not terms:
            return np.empty(0, np.int64), np.empty(0, np.float32), {"postings": 0, "evaluated": 0}
        if algo == "wand":
            return self._wand(terms, k, allow)
        ids, scores, scanned = self.score_all(terms, allow)
        order = topk(scores, k)
        return ids[order], scores[order], {"postings": scanned, "evaluated": len(ids)}

    def score_docs(self, terms, doc_ids) -> np.ndarray:
        """Random access: BM25 scores of specific documents via binary search."""
        doc_ids = np.asarray(doc_ids, dtype=np.int64)
        out = np.zeros(len(doc_ids), dtype=np.float32)
        for term, qtf in terms:
            docs, tfs = self.postings[term].arrays()
            if not len(docs):
                continue
            pos = np.searchsorted(docs, doc_ids)
            pos_c = np.minimum(pos, len(docs) - 1)
            found = docs[pos_c] == doc_ids
            if found.any():
                out[found] += qtf * self._term_scores(term, doc_ids[found], tfs[pos_c[found]])
        return out

    def _wand(self, terms, k, allow):
        cursors = []
        for term, qtf in terms:
            docs, tfs = self.postings[term].arrays()
            cursors.append([docs, tfs, 0, term, qtf, qtf * self.max_score(term)])
        heap: list[tuple[float, int]] = []
        theta = 0.0
        evaluated = 0
        postings = 0
        while True:
            cursors = [c for c in cursors if c[2] < len(c[0])]
            if not cursors:
                break
            cursors.sort(key=lambda c: c[0][c[2]])
            acc, pivot = 0.0, -1
            for i, c in enumerate(cursors):
                acc += c[5]
                if acc > theta:
                    pivot = i
                    break
            if pivot < 0:
                break
            pdoc = int(cursors[pivot][0][cursors[pivot][2]])
            if int(cursors[0][0][cursors[0][2]]) == pdoc:
                at = [c for c in cursors if c[0][c[2]] == pdoc]
                if allow is None or (pdoc < len(allow) and allow[pdoc]):
                    evaluated += 1
                    s = 0.0
                    for docs, tfs, p, term, qtf, _ in at:
                        s += qtf * float(self._term_scores(term, docs[p : p + 1], tfs[p : p + 1])[0])
                    if len(heap) < k:
                        heapq.heappush(heap, (s, -pdoc))
                    elif s > heap[0][0]:
                        heapq.heapreplace(heap, (s, -pdoc))
                    if len(heap) == k:
                        theta = heap[0][0]
                for c in at:
                    c[2] += 1
                    postings += 1
            else:
                for c in cursors[:pivot]:
                    skip = int(np.searchsorted(c[0][c[2] :], pdoc))
                    c[2] += skip
                    postings += 1
        best = sorted(heap, key=lambda x: (-x[0], -x[1]))
        ids = np.array([-d for _, d in best], dtype=np.int64)
        scores = np.array([s for s, _ in best], dtype=np.float32)
        return ids, scores, {"postings": postings, "evaluated": evaluated}

    def contains_all(self, text: str) -> np.ndarray | None:
        """Doc ids containing every analyzed term (posting-list intersection)."""
        toks = list(dict.fromkeys(self.analyzer(text)))
        if not toks:
            return None
        lists = []
        for t in toks:
            if t not in self.postings:
                return np.empty(0, dtype=np.int64)
            lists.append(self.postings[t].arrays()[0])
        lists.sort(key=len)  # intersect smallest first
        out = lists[0]
        for docs in lists[1:]:
            out = out[np.isin(out, docs, assume_unique=True)]
        return out

    # ------------------------------------------------------------ persistence
    def stats(self) -> dict:
        n_post = sum(len(p) for p in self.postings.values())
        return {
            "docs": self.n_docs,
            "terms": len(self.postings),
            "postings": n_post,
            "compressed_bytes": sum(p.nbytes() for p in self.postings.values()),
            "uncompressed_bytes": n_post * 8,  # int32 doc id + int32 tf
            "avgdl": round(self.avgdl, 2),
        }

    def save(self, path: Path):
        """Binary format: magic, header length, JSON header, postings blob."""
        self.flush()
        vocab, blobs, offset = [], [], 0
        for term, pl in self.postings.items():
            vocab.append([term, offset, len(pl.doc_bytes), len(pl.tf_bytes), pl.count, pl.last_doc])
            blobs += [pl.doc_bytes, pl.tf_bytes]
            offset += len(pl.doc_bytes) + len(pl.tf_bytes)
        header = json.dumps({
            "k1": self.k1, "b": self.b, "n_docs": self.n_docs, "total_len": self.total_len,
            "stem": self.analyzer.use_stem, "vocab": vocab,
        }).encode()
        path = Path(path)
        with open(path, "wb") as f:
            f.write(_MAGIC + struct.pack("<Q", len(header)) + header)
            for blob in blobs:
                f.write(blob)
        np.save(path.with_suffix(".doclen.npy"), self.doc_len.data)

    @classmethod
    def load(cls, path: Path) -> "InvertedIndex":
        path = Path(path)
        raw = path.read_bytes()
        if raw[:4] != _MAGIC:
            raise ValueError(f"{path} is not an inverted index file")
        (hlen,) = struct.unpack("<Q", raw[4:12])
        h = json.loads(raw[12 : 12 + hlen])
        body = memoryview(raw)[12 + hlen :]
        idx = cls(Analyzer(stem=h["stem"]), k1=h["k1"], b=h["b"])
        for term, off, nd, nt, count, last in h["vocab"]:
            pl = PostingList()
            pl.doc_bytes = bytes(body[off : off + nd])
            pl.tf_bytes = bytes(body[off + nd : off + nd + nt])
            pl.count, pl.last_doc = count, last
            idx.postings[term] = pl
        idx.doc_len = GrowableArray.from_array(np.load(path.with_suffix(".doclen.npy")))
        idx.n_docs, idx.total_len = h["n_docs"], h["total_len"]
        return idx


__all__ = ["InvertedIndex"]
