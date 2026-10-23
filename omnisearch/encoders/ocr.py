"""OCR extractor (image -> text) using RapidOCR (PP-OCR models in ONNX).

The models ship inside the ``rapidocr-onnxruntime`` wheel, so OCR works
without any download at run time.
"""
from __future__ import annotations

import numpy as np

from omnisearch.encoders.base import ModelUnavailable, load_image, register_extractor


class RapidOCRExtractor:
    name = "ocr:rapidocr"

    def __init__(self, min_confidence: float = 0.5):
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as e:
            raise ModelUnavailable("pip install rapidocr-onnxruntime for OCR") from e
        self.engine = RapidOCR()
        self.min_confidence = min_confidence

    def read(self, image) -> list[dict]:
        """Text lines with boxes, top-to-bottom."""
        arr = np.asarray(load_image(image))[:, :, ::-1]  # RGB -> BGR
        result, _ = self.engine(arr)
        lines = []
        for box, text, conf in result or []:
            if float(conf) >= self.min_confidence:
                lines.append({"text": text, "confidence": float(conf), "box": [list(map(float, p)) for p in box]})
        lines.sort(key=lambda l: (round(l["box"][0][1] / 10), l["box"][0][0]))
        return lines

    def __call__(self, images) -> list[str]:
        return [" ".join(l["text"] for l in self.read(im)) for im in images]


register_extractor("ocr", lambda arg: RapidOCRExtractor())
