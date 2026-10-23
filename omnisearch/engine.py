"""Collections and the engine facade.

A ``Collection`` owns one schema, its record store, media store and all
indexes. On disk::

    <collection>/
        schema.json   meta.json   cost.json   alive.npy
        docs.sqlite                     records (SQLite)
        media/                          raw files, content-addressed
        text/<field>.osx                inverted index (custom binary format)
        vectors/<field>.npz (+ .hnsw.npz)
        multivectors/<field>.npz
        attrs/<field>.npz

The engine is single-writer: ``save()`` is the durability checkpoint for the
indexes (the SQLite record store commits on every insert).
"""
from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path

import numpy as np

from omnisearch.encoders.base import get_encoder, load_image
from omnisearch.index.attribute import CategoryIndex, NumericIndex
from omnisearch.index.text import Analyzer, InvertedIndex
from omnisearch.index.vector import MultiVectorIndex, VectorFieldIndex
from omnisearch.ingest.pipeline import IngestPipeline
from omnisearch.query import execute, parse, plan
from omnisearch.query.cost import CostModel, calibrate
from omnisearch.schema import Schema
from omnisearch.storage import DocStore, MediaStore
from omnisearch.util import GrowableArray, normalize_rows


class Collection:
    def __init__(self, path: Path | str, schema: Schema):
        self.path = Path(path)
        self.schema = schema
        self.lock = threading.RLock()
        self.docstore = DocStore(self.path / "docs.sqlite")
        self.media = MediaStore(self.path / "media")
        self.pipeline = IngestPipeline(schema)
        self.cost = CostModel.load(self.path / "cost.json")
        self.alive = GrowableArray(bool, fill=False)
        self.text = {f.name: InvertedIndex(Analyzer(stem=f.stem)) for f in schema.text_fields()}
        self.vectors = {f.name: VectorFieldIndex(f) for f in schema.vector_fields()}
        self.multivectors = {f.name: MultiVectorIndex(f.dim, f.metric, f.index, f.params)
                             for f in schema.multivector_fields()}
        self.attrs = {f.name: (NumericIndex() if f.type in ("int", "float")
                               else CategoryIndex(multi=f.type == "tags"))
                      for f in schema.attribute_fields()}
        self._types = {f.name: f.type for f in schema.attribute_fields()}
        # when set (by the API server), IMAGE()/AUDIO() query paths must live under these roots
        self.query_path_roots: list[Path] | None = None

    # ------------------------------------------------------------ lifecycle
    @classmethod
    def create(cls, path, schema: Schema, overwrite: bool = False) -> "Collection":
        path = Path(path)
        if path.exists():
            if not overwrite:
                raise FileExistsError(f"{path} exists")
            shutil.rmtree(path)
        path.mkdir(parents=True)
        schema.save(path / "schema.json")
        coll = cls(path, schema)
        coll.save()
        return coll

    @classmethod
    def open(cls, path) -> "Collection":
        path = Path(path)
        coll = cls(path, Schema.load(path / "schema.json"))
        coll._load()
        return coll

    @property
    def name(self) -> str:
        return self.schema.name

    @property
    def capacity(self) -> int:
        """Number of doc ids ever assigned (deleted ones included)."""
        return len(self.alive)

    @property
    def n_alive(self) -> int:
        return int(self.alive.data.sum())

    @property
    def has_deletions(self) -> bool:
        return self.n_alive < self.capacity

    def alive_mask(self) -> np.ndarray:
        return self.alive.data.copy()

    # --------------------------------------------------------------- ingest
    def add(self, records: list[dict], base_dir=None, batch_size: int = 64, progress=None) -> list[str]:
        ids = []
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            processed = self.pipeline.process(batch, base_dir)
            ids += self.add_processed(processed)
            if progress:
                progress(min(start + batch_size, len(records)), len(records))
        return ids

    def add_processed(self, processed: list[dict]) -> list[str]:
        with self.lock:
            dup = self.docstore.exists(p["ext_id"] for p in processed)
            if dup:
                raise ValueError(f"ids already exist: {sorted(dup)[:5]}")
            first = self.capacity
            rows = []
            for i, p in enumerate(processed):
                d = first + i
                payload = dict(p["payload"])
                for name, src in p["media"].items():
                    payload[name] = self.media.put(src)
                rows.append((d, p["ext_id"], payload))
                for f, text in p["texts"].items():
                    self.text[f].add(d, text)
                for f, v in p["attrs"].items():
                    self.attrs[f].add(d, _coerce(v, self._types[f]))
                for f, (vecs, times) in p["multivectors"].items():
                    self.multivectors[f].add(d, vecs, times)
            for f, idx in self.vectors.items():
                docs = [first + i for i, p in enumerate(processed) if f in p["vectors"]]
                if docs:
                    idx.add(docs, np.stack([processed[d - first]["vectors"][f] for d in docs]))
            self.docstore.put_many(rows)
            self.alive.resize(first + len(processed))
            self.alive.data[first:] = True
            for idx in self.attrs.values():
                idx.pad(self.capacity)
            return [p["ext_id"] for p in processed]

    def delete(self, ext_ids) -> int:
        with self.lock:
            ids = [self.docstore.internal_id(e) for e in ext_ids]
            ids = [i for i in ids if i is not None]
            self.alive.data[ids] = False
            self.docstore.mark_deleted(ids)
            return len(ids)

    def build(self):
        """Bring every index up to date (HNSW inserts, IVF/PQ training, stats)."""
        with self.lock:
            for idx in self.text.values():
                idx.flush()
            for idx in self.vectors.values():
                idx.build()
            for mv in self.multivectors.values():
                mv.build()
            self.analyze()

    def analyze(self):
        for idx in self.attrs.values():
            if isinstance(idx, NumericIndex):
                idx.analyze()

    def calibrate(self) -> CostModel:
        self.build()
        self.cost = calibrate(self)
        self.cost.save(self.path / "cost.json")
        return self.cost

    # ---------------------------------------------------------------- query
    def query(self, text: str, **options):
        stmt = parse(text)
        if stmt.collection != self.name:
            raise ValueError(f"query targets {stmt.collection!r}, this collection is {self.name!r}")
        stmt.options.update(options)
        logical = plan(stmt, self.schema)
        with self.lock:
            return execute(self, logical)

    def explain(self, text: str, analyze: bool = False) -> str:
        stmt = parse(text)
        stmt.explain, stmt.analyze = True, analyze
        with self.lock:
            return execute(self, plan(stmt, self.schema)).explain()

    def encode_query(self, field: str, items) -> np.ndarray:
        f = self.schema.get(field)
        idx = self.vectors.get(field) or self.multivectors.get(field)
        parts = []
        for it in items:
            if it.kind == "vector":
                v = np.asarray(it.value, dtype=np.float32)
            elif it.kind == "doc":
                d = self.docstore.internal_id(str(it.value))
                if d is None:
                    raise ValueError(f"no document with id {it.value!r}")
                if field in self.vectors:
                    v = self.vectors[field].vector(d)
                else:
                    mv = self.multivectors[field]
                    has = d < len(mv.start) and mv.start[d] >= 0
                    v = mv.store.vectors[mv.start[d] : mv.end[d]].mean(axis=0) if has else None
                if v is None:
                    raise ValueError(f"document {it.value!r} has no {field} vector")
            else:
                value = it.value
                if it.kind == "image":
                    value = load_image(self._resolve(value))
                elif it.kind == "audio":
                    value = str(self._resolve(value))
                v = get_encoder(f.encoder).encode(it.kind, [value])[0]
            if len(v) != f.dim:
                raise ValueError(f"query vector has dim {len(v)}, {field} expects {f.dim}")
            parts.append(it.weight * normalize_rows(v[None])[0])
        q = np.sum(parts, axis=0)
        store = idx.store
        return store.query(q)

    def _resolve(self, p) -> Path:
        path = Path(p)
        if self.query_path_roots is not None:
            path = path.resolve()
            if not any(r.resolve() in path.parents for r in self.query_path_roots):
                try:
                    return self.media.resolve(str(p))
                except ValueError:
                    raise PermissionError(f"query media must be uploaded first: {p}") from None
        if path.exists():
            return path
        try:
            return self.media.resolve(str(p))
        except ValueError:
            raise FileNotFoundError(p) from None

    # -------------------------------------------------------------- records
    def get(self, ext_id: str) -> dict | None:
        d = self.docstore.internal_id(ext_id)
        return None if d is None else self.docstore.get_many([d]).get(d)

    def stats(self) -> dict:
        out = {"name": self.name, "docs": self.n_alive, "doc_ids": self.capacity,
               "text": {k: v.stats() for k, v in self.text.items()},
               "vectors": {k: {"count": len(v), "index": v.kind, **v.memory()} for k, v in self.vectors.items()},
               "multivectors": {k: {"vectors": len(v), "docs": v.n_docs()} for k, v in self.multivectors.items()},
               "attributes": {}}
        for k, a in self.attrs.items():
            if isinstance(a, NumericIndex):
                st = a.stats
                out["attributes"][k] = {"type": "numeric", "distinct": st.n_distinct, "null_frac": round(st.null_frac, 4),
                                        "btree_height": a.tree.height()}
            else:
                out["attributes"][k] = {"type": "category", "values": len(a.bitmaps), "bitmap_bytes": a.nbytes()}
        out["media_bytes"] = self.media.size_bytes()
        out["cost_model"] = {"calibrated": self.cost.calibrated}
        return out

    def facets(self, field: str, limit: int = 50) -> list[tuple[str, int]]:
        idx = self.attrs[field]
        if not isinstance(idx, CategoryIndex):
            raise ValueError("facets need a category field")
        alive = self.alive.data
        counts = [(v, int((idx.eq(v, self.capacity) & alive).sum())) for v in idx.values()]
        return sorted([c for c in counts if c[1]], key=lambda c: -c[1])[:limit]

    def numeric_range(self, field: str):
        vals = self.attrs[field].values.data
        vals = vals[~np.isnan(vals)]
        return (float(vals.min()), float(vals.max())) if len(vals) else (None, None)

    # ---------------------------------------------------------- persistence
    def save(self):
        with self.lock:
            self.build()
            for sub in ("text", "vectors", "multivectors", "attrs"):
                (self.path / sub).mkdir(exist_ok=True)
            np.save(self.path / "alive.npy", self.alive.data)
            for f, idx in self.text.items():
                idx.save(self.path / "text" / f"{f}.osx")
            for f, idx in self.vectors.items():
                idx.save(self.path / "vectors" / f"{f}.npz")
            for f, mv in self.multivectors.items():
                mv.save(self.path / "multivectors" / f"{f}.npz")
            for f, a in self.attrs.items():
                if isinstance(a, NumericIndex):
                    np.savez(self.path / "attrs" / f"{f}.npz", values=a.values.data)
                else:
                    np.savez(self.path / "attrs" / f"{f}.npz", **a.to_arrays())
            (self.path / "meta.json").write_text(json.dumps({"capacity": self.capacity, "alive": self.n_alive}), encoding="utf-8")

    def _load(self):
        p = self.path
        if (p / "alive.npy").exists():
            self.alive = GrowableArray.from_array(np.load(p / "alive.npy"), fill=False)
        for f in self.text:
            if (p / "text" / f"{f}.osx").exists():
                self.text[f] = InvertedIndex.load(p / "text" / f"{f}.osx")
        for f, idx in self.vectors.items():
            if (p / "vectors" / f"{f}.npz").exists():
                idx.load(p / "vectors" / f"{f}.npz")
        for f, mv in self.multivectors.items():
            if (p / "multivectors" / f"{f}.npz").exists():
                mv.load(p / "multivectors" / f"{f}.npz")
        for f, a in self.attrs.items():
            fp = p / "attrs" / f"{f}.npz"
            if not fp.exists():
                continue
            z = np.load(fp)
            if isinstance(a, NumericIndex):
                a.values = GrowableArray.from_array(z["values"], fill=np.nan)
                a.rebuild_tree()
            else:
                a.load_arrays(z)

    def close(self):
        self.docstore.close()


def _coerce(v, typ):
    if v is None:
        return None
    if typ in ("int", "float"):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None
    if typ == "bool":
        return "true" if v in (True, "true", "True", 1, "1") else "false"
    if typ == "tags":
        return [str(x) for x in v] if isinstance(v, (list, tuple, set)) else [str(v)]
    return str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)


class Engine:
    """A directory of collections, addressed by name in OmniQL queries."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._open: dict[str, Collection] = {}

    def names(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if (p / "schema.json").exists())

    def create(self, schema: Schema, overwrite: bool = False) -> Collection:
        coll = Collection.create(self.root / schema.name, schema, overwrite=overwrite)
        self._open[schema.name] = coll
        return coll

    def collection(self, name: str) -> Collection:
        if name not in self._open:
            if not (self.root / name / "schema.json").exists():
                raise KeyError(f"no collection {name!r}")
            self._open[name] = Collection.open(self.root / name)
        return self._open[name]

    def query(self, text: str, **options):
        stmt = parse(text)
        return self.collection(stmt.collection).query(text, **options)
