"""ffmpeg helpers: keyframe sampling, audio decoding, duration probing."""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def _ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError("ffmpeg is required for audio/video ingest")
    return exe


def duration(path) -> float:
    exe = shutil.which("ffprobe")
    if exe is None:
        return 0.0
    out = subprocess.run([exe, "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return float(json.loads(out)["format"].get("duration", 0.0))


def keyframes(path, interval: float = 2.0, max_frames: int = 300, size: int = 336):
    """Sample one frame every ``interval`` seconds. Returns [(timestamp, PIL.Image)]."""
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            [_ffmpeg(), "-v", "error", "-i", str(path), "-vf",
             f"fps=1/{interval},scale={size}:{size}:force_original_aspect_ratio=decrease",
             "-frames:v", str(max_frames), str(Path(tmp) / "f%05d.png")],
            check=True,
        )
        frames = []
        for i, p in enumerate(sorted(Path(tmp).glob("f*.png"))):
            with Image.open(p) as im:
                frames.append((i * interval, im.convert("RGB")))
    return frames


def decode_audio(path, sr: int = 16000) -> np.ndarray:
    """Decode any audio/video file to mono float32 PCM at ``sr`` Hz."""
    raw = subprocess.run(
        [_ffmpeg(), "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def has_audio(path) -> bool:
    exe = shutil.which("ffprobe")
    if exe is None:
        return True
    out = subprocess.run([exe, "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout
    return bool(out.strip())
