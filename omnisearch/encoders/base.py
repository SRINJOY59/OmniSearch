"""Encoder / extractor interfaces and a registry addressed by spec strings.

Encoders map inputs into a vector space: ``encode_text``, ``encode_image``,
``encode_audio``. Encoders whose text and image methods share one space (CLIP,
SigLIP) enable cross-modal search. Extractors turn media into text that the
keyword index can use (captions, OCR, speech transcripts).

Specs look like ``"clip"``, ``"clip:ViT-B-16-SigLIP/webli"`` or
``"hashed-ngram:512"``: the name selects a factory, the optional argument
after the colon configures it. Instances are cached per spec.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class UnsupportedModality(ValueError):
    pass


class ModelUnavailable(RuntimeError):
    """Raised when an encoder's model package or weights cannot be loaded."""


class Encoder:
    name = "encoder"
    dim = 0
    modalities: tuple = ()

    def encode_text(self, texts: list[str]) -> np.ndarray:
        raise UnsupportedModality(f"{self.name} cannot encode text")

    def encode_image(self, images: list) -> np.ndarray:
        raise UnsupportedModality(f"{self.name} cannot encode images")

    def encode_audio(self, paths: list) -> np.ndarray:
        raise UnsupportedModality(f"{self.name} cannot encode audio")

    def encode(self, kind: str, values: list) -> np.ndarray:
        fn = {"text": self.encode_text, "image": self.encode_image, "audio": self.encode_audio}.get(kind)
        if fn is None:
            raise UnsupportedModality(f"cannot encode {kind}")
        return np.asarray(fn(values), dtype=np.float32)


def load_image(x):
    from PIL import Image

    if isinstance(x, Image.Image):
        return x.convert("RGB")
    with Image.open(Path(x)) as im:
        return im.convert("RGB")


# Modalities per encoder name, known without loading any model (used by the UI
# to decide which SIMILAR clauses a text box or an uploaded image maps to).
MODALITIES = {
    "hashed-ngram": ("text",), "color-layout": ("image",), "clip": ("text", "image"),
    "hf-clip": ("text", "image"), "clap": ("text", "audio"), "precomputed": (),
}


def encoder_modalities(spec: str) -> tuple:
    return MODALITIES.get(spec.partition(":")[0], ())


_ENCODERS: dict = {}
_EXTRACTORS: dict = {}
_CACHE: dict = {}


def register_encoder(name: str, factory, modalities: tuple | None = None):
    _ENCODERS[name] = factory
    if modalities is not None:
        MODALITIES[name] = tuple(modalities)
    _CACHE.pop(("enc", name), None)


def register_extractor(name: str, factory):
    _EXTRACTORS[name] = factory


def _load(kind, registry, spec):
    key = (kind, spec)
    if key not in _CACHE:
        name, _, arg = spec.partition(":")
        if name not in registry:
            _import_builtin()
        if name not in registry:
            raise KeyError(f"unknown {kind} {name!r}; registered: {sorted(registry)}")
        _CACHE[key] = registry[name](arg or None)
    return _CACHE[key]


def get_encoder(spec: str) -> Encoder:
    return _load("enc", _ENCODERS, spec)


def get_extractor(spec: str):
    return _load("ext", _EXTRACTORS, spec)


def clear_cache():
    _CACHE.clear()


def _import_builtin():
    # importing these modules registers their factories
    from omnisearch.encoders import captioner, clap, classic, clip, ocr, speech  # noqa: F401
