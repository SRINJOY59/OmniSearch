"""In-memory B+-tree over (value, doc id) keys.

Leaves are linked so a range predicate is one root-to-leaf descent followed by
a sequential leaf scan. Keys are (value, doc id) pairs, which makes duplicate
values unique and keeps each value's doc ids sorted. ``bulk_load`` builds the
tree bottom-up from sorted keys (how databases build an index on existing
data); ``insert`` handles incremental ingest with node splits.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right

_LOW = -1
_HIGH = 1 << 62


class _Leaf:
    __slots__ = ("keys", "next")

    def __init__(self, keys=None):
        self.keys = keys or []
        self.next = None


class _Inner:
    __slots__ = ("keys", "children")

    def __init__(self, keys, children):
        self.keys = keys
        self.children = children


class BPlusTree:
    def __init__(self, order: int = 64):
        if order < 4:
            raise ValueError("order must be >= 4")
        self.order = order
        self.root = _Leaf()
        self.size = 0

    def __len__(self):
        return self.size

    def _find_leaf(self, key):
        node = self.root
        while isinstance(node, _Inner):
            node = node.children[bisect_right(node.keys, key)]
        return node

    def insert(self, value, doc_id: int):
        key = (value, doc_id)
        split = self._insert(self.root, key)
        if split is not None:
            sep, right = split
            self.root = _Inner([sep], [self.root, right])
        self.size += 1

    def _insert(self, node, key):
        if isinstance(node, _Leaf):
            node.keys.insert(bisect_left(node.keys, key), key)
            if len(node.keys) <= self.order:
                return None
            mid = len(node.keys) // 2
            right = _Leaf(node.keys[mid:])
            node.keys = node.keys[:mid]
            right.next, node.next = node.next, right
            return right.keys[0], right
        i = bisect_right(node.keys, key)
        split = self._insert(node.children[i], key)
        if split is None:
            return None
        sep, right = split
        node.keys.insert(i, sep)
        node.children.insert(i + 1, right)
        if len(node.keys) <= self.order:
            return None
        mid = len(node.keys) // 2
        up = node.keys[mid]
        new = _Inner(node.keys[mid + 1 :], node.children[mid + 1 :])
        node.keys, node.children = node.keys[:mid], node.children[: mid + 1]
        return up, new

    def bulk_load(self, pairs, fill: float = 0.9):
        """Build from (value, doc id) pairs. Replaces the current contents."""
        keys = sorted(pairs)
        self.size = len(keys)
        per = max(2, int(self.order * fill))
        leaves = [_Leaf(keys[i : i + per]) for i in range(0, len(keys), per)] or [_Leaf()]
        for a, b in zip(leaves, leaves[1:]):
            a.next = b
        level = leaves
        while len(level) > 1:
            parents = []
            for i in range(0, len(level), per + 1):
                group = level[i : i + per + 1]
                parents.append(_Inner([self._min_key(c) for c in group[1:]], group))
            level = parents
        self.root = level[0]

    @staticmethod
    def _min_key(node):
        while isinstance(node, _Inner):
            node = node.children[0]
        return node.keys[0]

    def range(self, lo=None, hi=None, lo_inclusive=True, hi_inclusive=True):
        """Yield doc ids whose value is within [lo, hi] (None = unbounded)."""
        if lo is None:
            node = self.root
            while isinstance(node, _Inner):
                node = node.children[0]
            i = 0
        else:
            start = (lo, _LOW) if lo_inclusive else (lo, _HIGH)
            node = self._find_leaf(start)
            i = bisect_left(node.keys, start)
        while node is not None:
            keys = node.keys
            while i < len(keys):
                v, d = keys[i]
                if hi is not None and (v > hi or (v == hi and not hi_inclusive)):
                    return
                yield d
                i += 1
            node, i = node.next, 0

    def height(self) -> int:
        h, node = 1, self.root
        while isinstance(node, _Inner):
            node, h = node.children[0], h + 1
        return h
