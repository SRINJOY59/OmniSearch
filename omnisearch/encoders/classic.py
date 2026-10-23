"""Classical, model-free encoders. They need no downloads, so the engine and
its demo run fully offline, and they make honest baselines for the neural
encoders.

* ``hashed-ngram`` (text): feature hashing of words, word bigrams and
  character trigrams into a fixed-size vector, sublinear tf weighting.
  Tolerates typos and morphology ("sneakrs" ~ "sneakers") but knows nothing
  about meaning.
* ``color-layout`` (image): HSV colour histogram plus a coarse 4x4 colour
  layout and an 8x8 edge-energy grid. Supports query-by-example on images;
  not cross-modal.
"""
from __future__ import annotations

import re
import zlib

import numpy as np

from omnisearch.encoders.base import Encoder, load_image, register_encoder
from omnisearch.util import normalize_rows

_WORD = re.compile(r"[a-z0-9]+")


class HashedNgramEncoder(Encoder):
    name = "hashed-ngram"
    modalities = ("text",)

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _features(self, text: str):
        words = _WORD.findall(text.lower())
        feats = list(words)
        feats += [f"{a}_{b}" for a, b in zip(words, words[1:])]
        for w in words:
            padded = f"#{w}#"
            feats += [padded[i : i + 3] for i in range(len(padded) - 2)]
        return feats

    def encode_text(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            counts: dict[int, float] = {}
            for f in self._features(t or ""):
                h = zlib.crc32(f.encode())
                j = h % self.dim
                counts[j] = counts.get(j, 0.0) + (1.0 if (h >> 31) & 1 else -1.0)
            for j, c in counts.items():
                out[i, j] = np.sign(c) * np.log1p(abs(c))
        return normalize_rows(out)


class ColorLayoutEncoder(Encoder):
    name = "color-layout"
    modalities = ("image",)
    dim = 8 * 4 * 4 + 4 * 4 * 3 + 8 * 8  # 128 + 48 + 64 = 240

    def encode_image(self, images):
        out = np.zeros((len(images), self.dim), dtype=np.float32)
        for i, x in enumerate(images):
            im = load_image(x)
            hsv = np.asarray(im.convert("HSV").resize((64, 64)), dtype=np.float32) / 256.0
            h = np.minimum((hsv[..., 0] * 8).astype(int), 7)
            s = np.minimum((hsv[..., 1] * 4).astype(int), 3)
            v = np.minimum((hsv[..., 2] * 4).astype(int), 3)
            hist = np.bincount((h * 16 + s * 4 + v).ravel(), minlength=128).astype(np.float32)
            hist /= hist.sum()
            layout = np.asarray(im.resize((4, 4)), dtype=np.float32).ravel() / 255.0
            gray = np.asarray(im.convert("L").resize((64, 64)), dtype=np.float32) / 255.0
            gy, gx = np.gradient(gray)
            edges = np.sqrt(gx ** 2 + gy ** 2).reshape(8, 8, 8, 8).mean(axis=(1, 3)).ravel()
            parts = [np.sqrt(hist), layout - layout.mean(), edges / (edges.max() + 1e-6)]
            out[i] = np.concatenate([p / (np.linalg.norm(p) + 1e-9) for p in parts])
        return normalize_rows(out)


register_encoder("hashed-ngram", lambda arg: HashedNgramEncoder(int(arg) if arg else 512))
register_encoder("color-layout", lambda arg: ColorLayoutEncoder())


class PrecomputedEncoder(Encoder):
    """Placeholder for fields whose vectors are supplied with each record
    (e.g. embeddings computed on a GPU elsewhere). Query with VECTOR[...] or DOC('id')."""

    name = "precomputed"

    def __init__(self, dim: int = 0):
        self.dim = dim

    def encode(self, kind, values):
        raise ValueError("this field stores precomputed vectors; query it with VECTOR[...] or DOC('id')")


register_encoder("precomputed", lambda arg: PrecomputedEncoder(int(arg) if arg else 0))
