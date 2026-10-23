"""CLIP / SigLIP image-text encoders (shared embedding space).

``clip`` uses open_clip; the argument is ``<model>/<pretrained tag>``:
    clip                          -> ViT-B-32 / laion2b_s34b_b79k (512-d)
    clip:ViT-B-16-SigLIP/webli    -> SigLIP (768-d)
``hf-clip:<repo>`` uses Hugging Face transformers, e.g.
    hf-clip:openai/clip-vit-base-patch32

Weights are downloaded on first use (huggingface.co), so this needs network
access or a pre-populated cache. Use a GPU for large ingests.
"""
from __future__ import annotations

import numpy as np

from omnisearch.encoders.base import Encoder, ModelUnavailable, load_image, register_encoder
from omnisearch.util import normalize_rows


def _device():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class OpenCLIPEncoder(Encoder):
    modalities = ("text", "image")

    def __init__(self, spec: str | None = None, batch_size: int = 64):
        model_name, _, pretrained = (spec or "ViT-B-32/laion2b_s34b_b79k").partition("/")
        try:
            import open_clip
            import torch
        except ImportError as e:
            raise ModelUnavailable("pip install -r requirements-models.txt to use CLIP") from e
        self.name = f"clip:{model_name}/{pretrained}"
        self.torch = torch
        self.device = _device()
        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        except Exception as e:  # network or cache failure
            raise ModelUnavailable(f"could not load {self.name}: {e}") from e
        self.model.eval().to(self.device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.batch_size = batch_size
        with torch.no_grad():
            self.dim = int(self.model.encode_text(self.tokenizer(["x"]).to(self.device)).shape[-1])

    def encode_text(self, texts):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                tok = self.tokenizer(list(texts[i : i + self.batch_size])).to(self.device)
                out.append(self.model.encode_text(tok).float().cpu().numpy())
        return normalize_rows(np.concatenate(out)) if out else np.zeros((0, self.dim), np.float32)

    def encode_image(self, images):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(images), self.batch_size):
                batch = self.torch.stack([self.preprocess(load_image(x)) for x in images[i : i + self.batch_size]])
                out.append(self.model.encode_image(batch.to(self.device)).float().cpu().numpy())
        return normalize_rows(np.concatenate(out)) if out else np.zeros((0, self.dim), np.float32)


class HFCLIPEncoder(Encoder):
    modalities = ("text", "image")

    def __init__(self, repo: str | None = None, batch_size: int = 64):
        repo = repo or "openai/clip-vit-base-patch32"
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as e:
            raise ModelUnavailable("pip install -r requirements-models.txt to use CLIP") from e
        self.name = f"hf-clip:{repo}"
        self.torch = torch
        self.device = _device()
        try:
            self.model = AutoModel.from_pretrained(repo).eval().to(self.device)
            self.processor = AutoProcessor.from_pretrained(repo)
        except Exception as e:
            raise ModelUnavailable(f"could not load {repo}: {e}") from e
        self.batch_size = batch_size
        self.dim = int(self.model.config.projection_dim)

    def encode_text(self, texts):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                inp = self.processor(text=list(texts[i : i + self.batch_size]), return_tensors="pt",
                                     padding=True, truncation=True).to(self.device)
                out.append(self.model.get_text_features(**inp).float().cpu().numpy())
        return normalize_rows(np.concatenate(out)) if out else np.zeros((0, self.dim), np.float32)

    def encode_image(self, images):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(images), self.batch_size):
                ims = [load_image(x) for x in images[i : i + self.batch_size]]
                inp = self.processor(images=ims, return_tensors="pt").to(self.device)
                out.append(self.model.get_image_features(**inp).float().cpu().numpy())
        return normalize_rows(np.concatenate(out)) if out else np.zeros((0, self.dim), np.float32)


register_encoder("clip", lambda arg: OpenCLIPEncoder(arg))
register_encoder("hf-clip", lambda arg: HFCLIPEncoder(arg))
