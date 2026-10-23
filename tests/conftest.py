import zlib

import numpy as np
import pytest

from omnisearch.encoders import Encoder, load_image, register_encoder
from omnisearch.util import normalize_rows


class FakeCrossModal(Encoder):
    """Deterministic test encoder with a shared text/image space.

    A text maps to the sum of per-word random vectors; an image maps to the
    vector of the colour word closest to its mean colour. So the text "red"
    and a red image land near each other - enough to test cross-modal paths.
    """

    name = "fake"
    modalities = ("text", "image")
    dim = 32
    COLORS = {"red": (200, 30, 30), "green": (30, 160, 60), "blue": (30, 60, 200)}

    def _word(self, w):
        rng = np.random.default_rng(zlib.crc32(w.encode()))
        return rng.normal(size=self.dim)

    def encode_text(self, texts):
        return normalize_rows(np.stack([sum(self._word(w) for w in t.lower().split()) if t.split()
                                        else np.ones(self.dim) for t in texts]))

    def encode_image(self, images):
        out = []
        for im in images:
            mean = np.asarray(load_image(im), dtype=float).reshape(-1, 3).mean(axis=0)
            name = min(self.COLORS, key=lambda c: np.linalg.norm(mean - np.array(self.COLORS[c])))
            out.append(self._word(name))
        return normalize_rows(np.stack(out))


register_encoder("fake", lambda arg: FakeCrossModal(), modalities=("text", "image"))


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def clustered(rng):
    centers = rng.normal(size=(20, 32))
    x = (centers[rng.integers(0, 20, 3000)] + 0.5 * rng.normal(size=(3000, 32))).astype(np.float32)
    q = (centers[rng.integers(0, 20, 40)] + 0.5 * rng.normal(size=(40, 32))).astype(np.float32)
    return x, q
