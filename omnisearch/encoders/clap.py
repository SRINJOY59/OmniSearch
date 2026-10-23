"""CLAP audio-text encoder (shared space for sounds and descriptions).

``clap`` / ``clap:<repo>``, default laion/clap-htsat-unfused (512-d), via
transformers. Audio is decoded and resampled to 48 kHz with ffmpeg.
"""
from __future__ import annotations

import numpy as np

from omnisearch.encoders import media
from omnisearch.encoders.base import Encoder, ModelUnavailable, register_encoder
from omnisearch.util import normalize_rows


class CLAPEncoder(Encoder):
    modalities = ("text", "audio")

    def __init__(self, repo: str | None = None, batch_size: int = 16, max_seconds: float = 10.0):
        repo = repo or "laion/clap-htsat-unfused"
        try:
            import torch
            from transformers import ClapModel, ClapProcessor
        except ImportError as e:
            raise ModelUnavailable("pip install -r requirements-models.txt to use CLAP") from e
        from omnisearch.encoders.clip import _device

        self.name = f"clap:{repo}"
        self.torch = torch
        self.device = _device()
        try:
            self.model = ClapModel.from_pretrained(repo).eval().to(self.device)
            self.processor = ClapProcessor.from_pretrained(repo)
        except Exception as e:
            raise ModelUnavailable(f"could not load {repo}: {e}") from e
        self.batch_size = batch_size
        self.max_samples = int(48000 * max_seconds)
        self.dim = int(self.model.config.projection_dim)

    def encode_text(self, texts):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                inp = self.processor(text=list(texts[i : i + self.batch_size]), return_tensors="pt", padding=True)
                out.append(self.model.get_text_features(**inp.to(self.device)).float().cpu().numpy())
        return normalize_rows(np.concatenate(out)) if out else np.zeros((0, self.dim), np.float32)

    def encode_audio(self, paths):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(paths), self.batch_size):
                waves = [media.decode_audio(p, sr=48000)[: self.max_samples] for p in paths[i : i + self.batch_size]]
                inp = self.processor(audios=waves, sampling_rate=48000, return_tensors="pt")
                out.append(self.model.get_audio_features(**inp.to(self.device)).float().cpu().numpy())
        return normalize_rows(np.concatenate(out)) if out else np.zeros((0, self.dim), np.float32)


register_encoder("clap", lambda arg: CLAPEncoder(arg))
