"""Record storage in SQLite.

Internal document ids are dense integers (0..N-1) assigned at ingest, so every
index can use plain numpy arrays and bitmaps keyed by doc id. SQLite holds the
mapping to external ids and the stored record payload.
"""
from __future__ import annotations

import functools
import json
import sqlite3
import threading
from pathlib import Path

from omnisearch.util import dumps


def _locked(fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return fn(self, *args, **kwargs)
    return wrapper


class DocStore:
    """One connection shared by API worker threads, so every call is serialised."""

    def __init__(self, path: Path | str):
        self.path = str(path)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS docs ("
            " id INTEGER PRIMARY KEY,"
            " ext_id TEXT UNIQUE NOT NULL,"
            " data TEXT NOT NULL,"
            " alive INTEGER NOT NULL DEFAULT 1)"
        )
        self.conn.commit()

    @_locked
    def next_id(self) -> int:
        row = self.conn.execute("SELECT COALESCE(MAX(id) + 1, 0) FROM docs").fetchone()
        return int(row[0])

    @_locked
    def put_many(self, rows: list[tuple[int, str, dict]]):
        self.conn.executemany(
            "INSERT INTO docs (id, ext_id, data) VALUES (?, ?, ?)",
            [(i, ext, dumps(data)) for i, ext, data in rows],
        )
        self.conn.commit()

    @_locked
    def get_many(self, ids) -> dict[int, dict]:
        ids = [int(i) for i in ids]
        out = {}
        for start in range(0, len(ids), 500):
            chunk = ids[start : start + 500]
            marks = ",".join("?" * len(chunk))
            for i, ext, data in self.conn.execute(
                f"SELECT id, ext_id, data FROM docs WHERE id IN ({marks})", chunk
            ):
                out[i] = {"_id": ext, **json.loads(data)}
        return out

    @_locked
    def internal_id(self, ext_id: str) -> int | None:
        row = self.conn.execute("SELECT id FROM docs WHERE ext_id = ? AND alive = 1", (ext_id,)).fetchone()
        return None if row is None else int(row[0])

    @_locked
    def exists(self, ext_ids) -> set[str]:
        ext_ids = list(ext_ids)
        found = set()
        for start in range(0, len(ext_ids), 500):
            chunk = ext_ids[start : start + 500]
            marks = ",".join("?" * len(chunk))
            found.update(r[0] for r in self.conn.execute(f"SELECT ext_id FROM docs WHERE ext_id IN ({marks})", chunk))
        return found

    @_locked
    def mark_deleted(self, ids):
        self.conn.executemany("UPDATE docs SET alive = 0 WHERE id = ?", [(int(i),) for i in ids])
        self.conn.commit()

    @_locked
    def count(self, alive_only=True) -> int:
        q = "SELECT COUNT(*) FROM docs" + (" WHERE alive = 1" if alive_only else "")
        return int(self.conn.execute(q).fetchone()[0])

    @_locked
    def iter_ids(self, limit=None, offset=0):
        q = "SELECT id, ext_id FROM docs WHERE alive = 1 ORDER BY id"
        if limit is not None:
            q += f" LIMIT {int(limit)} OFFSET {int(offset)}"
        return self.conn.execute(q).fetchall()

    @_locked
    def close(self):
        self.conn.close()
