"""Ingest pipeline: raw records -> everything the indexes need.

For each batch of records:
  1. resolve media paths (image / audio / video fields);
  2. run extractors that turn media into text: captions and OCR for images,
     Whisper transcripts for audio, keyframe OCR + transcripts for video;
  3. assemble each text field from its sources;
  4. embed vector fields (batched per field) and multi-vector fields
     (one vector per video keyframe, with timestamps);
  5. collect attribute values and the payload to store.

Anything already present in a record is reused, so expensive steps can be
precomputed on a GPU (see scripts/precompute.py) and ingested elsewhere.
"""
from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np

from omnisearch.encoders import media
from omnisearch.encoders.base import get_encoder, get_extractor
from omnisearch.schema import Schema
from omnisearch.util import normalize_rows

_PATH = "__path__"
_FRAMES = "__frames__"


class IngestPipeline:
    def __init__(self, schema: Schema):
        self.schema = schema

    def _extractor(self, name):
        return get_extractor(self.schema.models.get(name, name))

    def _needs_frames(self, video_field: str) -> bool:
        return any(m.source == video_field for m in self.schema.multivector_fields()) or any(
            v.source == video_field for v in self.schema.vector_fields())

    def process(self, records: list[dict], base_dir: Path | str | None = None) -> list[dict]:
        recs = [dict(r) for r in records]
        base = Path(base_dir) if base_dir else None
        for mf in self.schema.media_fields():
            for r in recs:
                v = r.get(mf.name)
                if not v:
                    continue
                p = Path(v)
                if not p.is_absolute() and base is not None:
                    p = base / p
                if not p.exists():
                    raise FileNotFoundError(f"{mf.name} file not found: {p}")
                r[_PATH + mf.name] = str(p)
        self._extract(recs)
        out = [self._assemble(r) for r in recs]
        self._embed(recs, out)
        return out

    # ------------------------------------------------------------ extraction
    def _extract(self, recs):
        for mf in self.schema.media_fields():
            have = [r for r in recs if r.get(_PATH + mf.name)]
            if not have:
                continue
            if mf.kind == "video" and (("ocr" in mf.extract) or self._needs_frames(mf.name)):
                for r in have:
                    if _FRAMES + mf.name not in r and not self._frames_precomputed(r, mf.name):
                        r[_FRAMES + mf.name] = media.keyframes(r[_PATH + mf.name], mf.frame_interval)
            for ex in mf.extract:
                key = f"{mf.name}.{ex}"
                todo = [r for r in have if key not in r]
                if not todo:
                    continue
                if ex == "transcript":
                    segs = self._extractor("transcript")([r[_PATH + mf.name] for r in todo])
                    for r, s in zip(todo, segs):
                        r[key] = " ".join(x["text"] for x in s)
                        r[key + ".segments"] = s
                elif mf.kind == "video":  # OCR on keyframes, with timestamps
                    ocr = self._extractor("ocr")
                    for r in todo:
                        frames = r.get(_FRAMES + mf.name, [])
                        texts = ocr([im for _, im in frames]) if frames else []
                        segs = [{"start": t, "end": t + mf.frame_interval, "text": txt}
                                for (t, _), txt in zip(frames, texts) if txt]
                        r[key] = " ".join(dict.fromkeys(s["text"] for s in segs))
                        r[key + ".segments"] = segs
                else:
                    vals = self._extractor(ex)([r[_PATH + mf.name] for r in todo])
                    for r, v in zip(todo, vals):
                        r[key] = v

    def _frames_precomputed(self, r, video_field):
        return all(r.get(m.name) is not None for m in self.schema.multivector_fields() if m.source == video_field)

    # -------------------------------------------------------------- assembly
    def _assemble(self, r) -> dict:
        texts = {}
        for tf in self.schema.text_fields():
            parts = [r.get(k) for k in tf.input_keys()]
            texts[tf.name] = " \n".join(str(p) for p in parts if p)
        attrs = {af.name: r.get(af.name) for af in self.schema.attribute_fields()}
        mediap = {mf.name: r[_PATH + mf.name] for mf in self.schema.media_fields() if r.get(_PATH + mf.name)}
        skip = {f.name for f in self.schema.vector_fields()} | {f.name for f in self.schema.multivector_fields()}
        payload = {k: v for k, v in r.items()
                   if k not in skip and not k.startswith("__") and k not in ("id", "_id") and _jsonable(v)}
        ext = r.get("id", r.get("_id"))
        return {"ext_id": str(ext) if ext is not None else uuid.uuid4().hex, "payload": payload,
                "texts": texts, "attrs": attrs, "media": mediap, "vectors": {}, "multivectors": {}}

    def _text_value(self, r, key):
        try:
            tf = self.schema.get(key)
            if hasattr(tf, "input_keys"):
                return " \n".join(str(r[k]) for k in tf.input_keys() if r.get(k))
        except KeyError:
            pass
        v = r.get(key)
        return str(v) if v else None

    # ------------------------------------------------------------- embedding
    def _embed(self, recs, out):
        for vf in self.schema.vector_fields():
            kind = self.schema.media_kind(vf.source)
            todo, inputs = [], []
            for i, r in enumerate(recs):
                pre = r.get(vf.name)
                if pre is not None:
                    out[i]["vectors"][vf.name] = np.asarray(pre, dtype=np.float32)
                    continue
                if kind is None:
                    val = self._text_value(r, vf.source)
                elif kind == "video":
                    val = r.get(_FRAMES + vf.source) or None
                else:
                    val = r.get(_PATH + vf.source)
                if val:
                    todo.append(i)
                    inputs.append(val)
            if not todo:
                continue
            enc = get_encoder(vf.encoder)
            if kind == "video":  # one vector per video: mean of its keyframe embeddings
                vecs = np.stack([normalize_rows(enc.encode_image([im for _, im in fr]).mean(axis=0, keepdims=True))[0]
                                 for fr in inputs])
            else:
                vecs = enc.encode(kind or "text", inputs)
            if vecs.shape[1] != vf.dim:
                raise ValueError(f"encoder {vf.encoder} returned dim {vecs.shape[1]}, field {vf.name} expects {vf.dim}")
            for i, v in zip(todo, vecs):
                out[i]["vectors"][vf.name] = v
        for mf in self.schema.multivector_fields():
            enc = None
            for i, r in enumerate(recs):
                pre = r.get(mf.name)
                if pre is not None:
                    out[i]["multivectors"][mf.name] = (np.asarray(pre["vectors"], np.float32),
                                                       np.asarray(pre.get("times", np.zeros(len(pre["vectors"]))), np.float32))
                    continue
                frames = r.get(_FRAMES + mf.source)
                if not frames:
                    continue
                enc = enc or get_encoder(mf.encoder)
                vecs = enc.encode_image([im for _, im in frames])
                out[i]["multivectors"][mf.name] = (vecs, np.array([t for t, _ in frames], np.float32))


def _jsonable(v) -> bool:
    if v is None or isinstance(v, (str, int, float, bool)):
        return True
    if isinstance(v, (list, tuple)):
        return len(v) < 10_000 and all(_jsonable(x) for x in v[:50])
    if isinstance(v, dict):
        return all(isinstance(k, str) and _jsonable(x) for k, x in list(v.items())[:50])
    return False
