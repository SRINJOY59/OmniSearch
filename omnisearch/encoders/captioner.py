"""Image captioning extractor (image -> text) with BLIP via transformers.

Default model: Salesforce/blip-image-captioning-base. It was fine-tuned on
MS-COCO captions, so captions are optimistic on COCO itself - evaluate on
Flickr30k, or pick a captioner not trained on your evaluation set.
"""
from __future__ import annotations

from omnisearch.encoders.base import ModelUnavailable, load_image, register_extractor


class BLIPCaptioner:
    def __init__(self, repo: str | None = None, batch_size: int = 16, max_new_tokens: int = 30):
        repo = repo or "Salesforce/blip-image-captioning-base"
        try:
            import torch
            from transformers import BlipForConditionalGeneration, BlipProcessor
        except ImportError as e:
            raise ModelUnavailable("pip install -r requirements-models.txt for captioning") from e
        from omnisearch.encoders.clip import _device

        self.name = f"caption:{repo}"
        self.torch = torch
        self.device = _device()
        try:
            self.processor = BlipProcessor.from_pretrained(repo)
            self.model = BlipForConditionalGeneration.from_pretrained(repo).eval().to(self.device)
        except Exception as e:
            raise ModelUnavailable(f"could not load {repo}: {e}") from e
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens

    def __call__(self, images) -> list[str]:
        out = []
        with self.torch.no_grad():
            for i in range(0, len(images), self.batch_size):
                ims = [load_image(x) for x in images[i : i + self.batch_size]]
                inp = self.processor(images=ims, return_tensors="pt").to(self.device)
                ids = self.model.generate(**inp, max_new_tokens=self.max_new_tokens)
                out += [t.strip() for t in self.processor.batch_decode(ids, skip_special_tokens=True)]
        return out


register_extractor("caption", lambda arg: BLIPCaptioner(arg))
