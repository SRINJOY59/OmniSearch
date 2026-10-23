"""Per-field vector index: maps doc ids to vector positions and exposes the
three physical access paths the optimizer chooses between.

* ``exact``      - brute-force scan of the (filtered) vectors. Pre-filtering.
* ``index``      - the field's ANN index with the filter pushed into it
                   (HNSW in-graph filtering, or IVF candidate masking).
* ``postfilter`` - unfiltered ANN search for ``overfetch`` results, then drop
                   the ones the filter rejects.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from omnisearch.index.vector.flat import FlatIndex
from omnisearch.index.vector.hnsw import HNSWIndex
from omnisearch.index.vector.ivf import IVFIndex
from omnisearch.index.vector.quantize import ProductQuantizer, ScalarQuantizer, make_quantizer
from omnisearch.index.vector.store import VectorStore
from omnisearch.schema import VectorField
from omnisearch.util import GrowableArray

STRATEGIES = ("exact", "index", "postfilter")


class VectorFieldIndex:
    def __init__(self, field: VectorField):
        if field.quantization and field.metric == "l2":
            raise ValueError("quantization requires cosine or ip metric")
        self.field = field
        p = field.params
        self.store = VectorStore(field.dim, field.metric)
        self.pos2doc = GrowableArray(np.int64)
        self.doc2pos = GrowableArray(np.int64, fill=-1)
        self.quant = make_quantizer(field.quantization, field.dim)
        self.refine = p.get("refine", 4)
        self.flat = FlatIndex(self.store, self.quant, self.refine)
        if field.index == "hnsw":
            self.ann = HNSWIndex(self.store, M=p.get("M", 16), ef_construction=p.get("ef_construction", 100),
                                 ef_search=p.get("ef_search", 64))
        elif field.index == "ivf":
            self.ann = IVFIndex(self.store, nlist=p.get("nlist"), nprobe=p.get("nprobe", 8),
                                quantizer=self.quant, refine=self.refine)
        else:
            self.ann = None

    def __len__(self):
        return len(self.store)

    @property
    def kind(self) -> str:
        return self.field.index

    def add(self, doc_ids, vecs):
        doc_ids = np.asarray(doc_ids, dtype=np.int64)
        pos = self.store.add(vecs)
        self.pos2doc.extend(doc_ids)
        self.doc2pos.resize(int(doc_ids.max()) + 1)
        self.doc2pos.data[doc_ids] = pos
        if self.quant is not None and self.quant.trained:
            self.quant.add(self.store.vectors[pos])

    def build(self):
        """Train quantizer / IVF on the current data and index pending vectors."""
        if self.quant is not None and not self.quant.trained and len(self.store) >= 256:
            x = self.store.vectors
            sample = x if len(x) <= 20_000 else x[np.random.default_rng(0).choice(len(x), 20_000, replace=False)]
            self.quant.train(sample)
            self.quant.add(x)
        if self.ann is not None:
            self.ann.sync()

    def _allow_pos(self, allow: np.ndarray) -> np.ndarray:
        docs = self.pos2doc.data
        out = np.zeros(len(docs), dtype=bool)
        inside = docs < len(allow)
        out[inside] = allow[docs[inside]]
        return out

    def prepare_query(self, q) -> np.ndarray:
        return self.store.query(q)

    def search(self, q, k, allow=None, strategy="index", ef=None, nprobe=None, overfetch=None):
        """q must be prepared (``prepare_query``). Returns (doc ids, scores, stats)."""
        if len(self.store) == 0:
            return np.empty(0, np.int64), np.empty(0, np.float32), {}
        allow_pos = None if allow is None else self._allow_pos(allow)
        if strategy == "exact" or self.ann is None:
            pos, scores, st = self.flat.search(q, k, allow=allow_pos)
        elif strategy == "postfilter":
            kk = max(k, overfetch or 4 * k)
            pos, scores, st = self._ann(q, kk, ef=max(ef or 0, kk), nprobe=nprobe, allow=None)
            if allow_pos is not None:
                keep = allow_pos[pos]
                pos, scores = pos[keep], scores[keep]
            pos, scores = pos[:k], scores[:k]
            st = {**st, "fetched": kk}
        else:
            pos, scores, st = self._ann(q, k, ef=ef, nprobe=nprobe, allow=allow_pos)
        return self.pos2doc.data[pos], scores, st

    def _ann(self, q, k, ef=None, nprobe=None, allow=None):
        if isinstance(self.ann, HNSWIndex):
            return self.ann.search(q, k, ef=ef, allow=allow)
        return self.ann.search(q, k, nprobe=nprobe, allow=allow)

    def score(self, q, doc_ids) -> np.ndarray:
        """Random access: exact scores for given docs (-inf if the doc has no vector)."""
        doc_ids = np.asarray(doc_ids, dtype=np.int64)
        out = np.full(len(doc_ids), -np.inf, dtype=np.float32)
        inside = doc_ids < len(self.doc2pos)
        pos = np.full(len(doc_ids), -1, dtype=np.int64)
        pos[inside] = self.doc2pos.data[doc_ids[inside]]
        has = pos >= 0
        if has.any():
            out[has] = self.store.score(q, pos[has])
        return out

    def vector(self, doc_id: int) -> np.ndarray | None:
        if doc_id >= len(self.doc2pos) or self.doc2pos[doc_id] < 0:
            return None
        return self.store.vectors[self.doc2pos[doc_id]]

    def memory(self) -> dict:
        out = {"float32_bytes": self.store.nbytes()}
        if self.quant is not None and self.quant.trained:
            out["code_bytes"] = self.quant.nbytes()
        if isinstance(self.ann, HNSWIndex):
            out["graph_edges"] = self.ann.stats()["edges"]
        return out

    # ----------------------------------------------------------- persistence
    def save(self, path: Path):
        path = Path(path)
        arrays = {"vectors": self.store.vectors, "pos2doc": self.pos2doc.data}
        if isinstance(self.quant, ScalarQuantizer) and self.quant.trained:
            arrays.update(sq_vmin=self.quant.vmin, sq_scale=self.quant.scale, codes=self.quant.codes.data)
        if isinstance(self.quant, ProductQuantizer) and self.quant.trained:
            arrays.update(pq_books=self.quant.codebooks, codes=self.quant.codes.data)
        if isinstance(self.ann, IVFIndex):
            arrays.update({f"ivf_{k}": v for k, v in self.ann.save_arrays().items()})
        np.savez(path, **arrays)
        if isinstance(self.ann, HNSWIndex):
            self.ann.save(path.with_suffix(".hnsw.npz"))

    def load(self, path: Path):
        path = Path(path)
        z = np.load(path)
        if len(z["vectors"]):
            self.add(z["pos2doc"], z["vectors"])
        if "sq_vmin" in z:
            self.quant.vmin, self.quant.scale = z["sq_vmin"], z["sq_scale"]
            self.quant.codes = GrowableArray.from_array(z["codes"])
        if "pq_books" in z:
            self.quant.codebooks = z["pq_books"]
            self.quant.codes = GrowableArray.from_array(z["codes"])
        if isinstance(self.ann, IVFIndex):
            self.ann.load_arrays({k[4:]: z[k] for k in z.files if k.startswith("ivf_")})
        hp = path.with_suffix(".hnsw.npz")
        if isinstance(self.ann, HNSWIndex) and hp.exists():
            self.ann = HNSWIndex.load(hp, self.store)
