"""Content-addressed store for raw media files (images, audio, video).

Files are copied to ``media/<sha256[:2]>/<sha256>.<ext>``; identical files are
stored once. Indexes only ever hold the relative path.
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path


class MediaStore:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, path: Path | str) -> str:
        path = Path(path)
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        digest = h.hexdigest()
        rel = Path(digest[:2]) / f"{digest}{path.suffix.lower()}"
        dest = self.root / rel
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
        return rel.as_posix()

    def resolve(self, rel: str) -> Path:
        p = (self.root / rel).resolve()
        if self.root.resolve() not in p.parents:
            raise ValueError("path escapes media store")
        return p

    def size_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
