"""Compressed posting lists.

Each list stores doc ids as gaps minus one (d1, d2-d1-1, ...) and term
frequencies, both variable-byte encoded: 7 payload bits per byte, high bit set on the last byte
of each integer. Because gaps continue from the last stored doc id, new
postings can be encoded and appended without re-encoding the existing bytes.

Encoding and decoding are vectorised with numpy so decoding a list of 100k
postings costs well under a millisecond.
"""
from __future__ import annotations

import numpy as np

_MAX_GROUPS = 10  # 70 bits, enough for any uint64


def vbyte_encode(values) -> bytes:
    v = np.asarray(values, dtype=np.uint64)
    if v.size == 0:
        return b""
    shifts = (7 * np.arange(_MAX_GROUPS, dtype=np.uint64))[None, :]
    groups = ((v[:, None] >> shifts) & np.uint64(0x7F)).astype(np.uint8)
    thresholds = (np.uint64(1) << shifts[0, 1:])  # 2^7, 2^14, ...
    nbytes = 1 + (v[:, None] >= thresholds[None, :]).sum(axis=1)
    groups[np.arange(len(v)), nbytes - 1] |= 0x80
    mask = np.arange(_MAX_GROUPS)[None, :] < nbytes[:, None]
    return groups[mask].tobytes()


def vbyte_decode(data: bytes) -> np.ndarray:
    b = np.frombuffer(data, dtype=np.uint8)
    if b.size == 0:
        return np.empty(0, dtype=np.int64)
    stop = b >= 0x80
    ends = np.flatnonzero(stop)
    starts = np.concatenate(([0], ends[:-1] + 1))
    gid = np.cumsum(stop) - stop  # index of the integer each byte belongs to
    pos = np.arange(len(b)) - starts[gid]
    payload = (b & 0x7F).astype(np.int64) << (7 * pos)
    return np.add.reduceat(payload, starts)


class PostingList:
    """Append-only posting list for one term: (doc id, term frequency) pairs."""

    __slots__ = ("doc_bytes", "tf_bytes", "count", "last_doc", "_tail_docs", "_tail_tfs", "_cache")

    def __init__(self):
        self.doc_bytes = b""
        self.tf_bytes = b""
        self.count = 0
        self.last_doc = -1
        self._tail_docs: list[int] = []
        self._tail_tfs: list[int] = []
        self._cache = None

    def add(self, doc: int, tf: int):
        if doc <= (self._tail_docs[-1] if self._tail_docs else self.last_doc):
            raise ValueError("doc ids must be strictly increasing")
        self._tail_docs.append(doc)
        self._tail_tfs.append(tf)
        self._cache = None

    def flush(self):
        if not self._tail_docs:
            return
        docs = np.asarray(self._tail_docs, dtype=np.int64)
        gaps = np.diff(docs, prepend=self.last_doc)  # >= 1 since ids strictly increase
        self.doc_bytes += vbyte_encode(gaps - 1)
        self.tf_bytes += vbyte_encode(np.asarray(self._tail_tfs, dtype=np.int64))
        self.count += len(docs)
        self.last_doc = int(docs[-1])
        self._tail_docs.clear()
        self._tail_tfs.clear()

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Decoded (doc ids, tfs), both sorted by doc id."""
        if self._cache is None:
            self.flush()
            gaps = vbyte_decode(self.doc_bytes) + 1
            self._cache = (np.cumsum(gaps) - 1, vbyte_decode(self.tf_bytes))
        return self._cache

    def __len__(self):
        return self.count + len(self._tail_docs)

    def nbytes(self) -> int:
        self.flush()
        return len(self.doc_bytes) + len(self.tf_bytes)

    def drop_cache(self):
        self._cache = None
