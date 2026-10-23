"""Collection data model.

A record is a flat dict. The schema says which keys are media (image / audio /
video files), which text gets keyword-indexed, which inputs get embedded into
which vector fields, and which keys are structured attributes for filtering.

Derived values produced at ingest are addressed as ``"<media>.<extractor>"``,
e.g. ``"image.caption"``, ``"image.ocr"``, ``"video.transcript"``. A text
field can concatenate raw keys and derived keys::

    TextField("text", sources=["title", "description", "image.caption", "image.ocr"])
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

MEDIA_KINDS = ("image", "audio", "video")
ATTRIBUTE_TYPES = ("int", "float", "category", "tags", "bool")
EXTRACTORS = {
    "image": ("caption", "ocr"),
    "audio": ("transcript",),
    "video": ("transcript", "ocr"),
}


@dataclass
class MediaField:
    name: str
    kind: str
    extract: list[str] = field(default_factory=list)
    frame_interval: float = 2.0  # video: seconds between sampled keyframes

    def __post_init__(self):
        if self.kind not in MEDIA_KINDS:
            raise ValueError(f"media kind must be one of {MEDIA_KINDS}")
        bad = set(self.extract) - set(EXTRACTORS[self.kind])
        if bad:
            raise ValueError(f"{self.kind} media cannot extract {sorted(bad)}")


@dataclass
class TextField:
    name: str
    sources: list[str] = field(default_factory=list)
    stem: bool = True

    def input_keys(self) -> list[str]:
        return self.sources or [self.name]


@dataclass
class VectorField:
    name: str
    dim: int
    encoder: str
    source: str
    metric: str = "cosine"
    index: str = "hnsw"  # flat | hnsw | ivf
    quantization: str | None = None  # None | "sq8" | "pq<m>"
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.index not in ("flat", "hnsw", "ivf"):
            raise ValueError("index must be flat, hnsw or ivf")
        if self.metric not in ("cosine", "ip", "l2"):
            raise ValueError("metric must be cosine, ip or l2")
        if self.quantization and self.index == "hnsw":
            raise ValueError("quantization is supported for flat and ivf indexes")


@dataclass
class MultiVectorField:
    """Many vectors per record (video keyframes); a record scores by its best vector."""

    name: str
    dim: int
    encoder: str
    source: str  # a video media field
    metric: str = "cosine"
    index: str = "flat"  # flat | hnsw (over the individual vectors)
    params: dict = field(default_factory=dict)


@dataclass
class AttributeField:
    name: str
    type: str

    def __post_init__(self):
        if self.type not in ATTRIBUTE_TYPES:
            raise ValueError(f"attribute type must be one of {ATTRIBUTE_TYPES}")


_KINDS = {
    "media": MediaField,
    "text": TextField,
    "vector": VectorField,
    "multivector": MultiVectorField,
    "attribute": AttributeField,
}


@dataclass
class Schema:
    name: str
    fields: list = field(default_factory=list)
    models: dict = field(default_factory=dict)  # extractor name -> model spec

    def __post_init__(self):
        seen = set()
        for f in self.fields:
            if f.name in seen:
                raise ValueError(f"duplicate field {f.name!r}")
            seen.add(f.name)
        media = {f.name for f in self.media_fields()}
        for f in self.multivector_fields():
            if f.source not in media:
                raise ValueError(f"multivector field {f.name!r} needs a media source")

    def _of(self, cls):
        return [f for f in self.fields if isinstance(f, cls)]

    def media_fields(self) -> list[MediaField]:
        return self._of(MediaField)

    def text_fields(self) -> list[TextField]:
        return self._of(TextField)

    def vector_fields(self) -> list[VectorField]:
        return self._of(VectorField)

    def multivector_fields(self) -> list[MultiVectorField]:
        return self._of(MultiVectorField)

    def attribute_fields(self) -> list[AttributeField]:
        return self._of(AttributeField)

    def get(self, name):
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(name)

    def kind_of(self, name) -> str:
        f = self.get(name)
        return next(k for k, cls in _KINDS.items() if isinstance(f, cls))

    def media_kind(self, key: str) -> str | None:
        for f in self.media_fields():
            if f.name == key:
                return f.kind
        return None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "models": self.models,
            "fields": [{"field_type": self.kind_of(f.name), **asdict(f)} for f in self.fields],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Schema":
        fields = []
        for fd in d["fields"]:
            fd = dict(fd)
            fields.append(_KINDS[fd.pop("field_type")](**fd))
        return cls(name=d["name"], fields=fields, models=d.get("models", {}))

    def save(self, path: Path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Schema":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
