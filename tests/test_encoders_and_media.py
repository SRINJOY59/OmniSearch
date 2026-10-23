import shutil
import subprocess

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from omnisearch.encoders import get_encoder
from omnisearch.encoders import media
from omnisearch.ingest.pipeline import IngestPipeline
from omnisearch.schema import MediaField, MultiVectorField, Schema, TextField, VectorField

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def text_image(text, size=(320, 120)):
    im = Image.new("RGB", size, (255, 255, 255))
    try:
        font = ImageFont.truetype(FONT, 40)
    except OSError:
        font = ImageFont.load_default(size=40)
    ImageDraw.Draw(im).text((size[0] // 2, size[1] // 2), text, fill=(0, 0, 0), font=font, anchor="mm")
    return im


def test_hashed_ngram_tolerates_typos():
    enc = get_encoder("hashed-ngram")
    v = enc.encode_text(["running sneakers", "runing sneakrs", "ceramic coffee mug"])
    assert v.shape == (3, 512)
    assert v[0] @ v[1] > 0.5 > v[0] @ v[2]


def test_color_layout_groups_colours():
    enc = get_encoder("color-layout")
    ims = [Image.new("RGB", (32, 32), c) for c in [(200, 20, 20), (210, 30, 25), (20, 40, 200)]]
    v = enc.encode_image(ims)
    assert v.shape == (3, enc.dim)
    assert v[0] @ v[1] > v[0] @ v[2]


def test_precomputed_encoder_refuses_queries():
    with pytest.raises(ValueError):
        get_encoder("precomputed").encode("text", ["x"])


def test_ocr_reads_rendered_text():
    pytest.importorskip("rapidocr_onnxruntime")
    from omnisearch.encoders.base import get_extractor

    out = get_extractor("ocr")([text_image("EXIT 42")])
    assert "EXIT" in out[0].upper()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_video_keyframes_and_pipeline(tmp_path):
    pytest.importorskip("rapidocr_onnxruntime")
    frames = [text_image(t, (320, 180)) for t in ["ALPHA", "BRAVO", "CHARLIE"]]
    raw = b"".join(f.tobytes() * 4 for f in frames)  # 4 frames per slide at 2 fps = 2 s each
    path = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "320x180", "-r", "2",
                    "-i", "-", "-pix_fmt", "yuv420p", str(path)], input=raw, check=True)
    kf = media.keyframes(path, interval=2.0)
    assert [t for t, _ in kf] == [0.0, 2.0, 4.0]
    schema = Schema("v", [
        MediaField("video", "video", extract=["ocr"], frame_interval=2.0),
        TextField("text", sources=["video.ocr"]),
        MultiVectorField("frames", 240, "color-layout", source="video"),
    ])
    out = IngestPipeline(schema).process([{"id": "v1", "video": str(path)}])[0]
    assert "BRAVO" in out["texts"]["text"].upper()
    segs = out["payload"]["video.ocr.segments"]
    assert any("CHARLIE" in s["text"].upper() and s["start"] == 4.0 for s in segs)
    vecs, times = out["multivectors"]["frames"]
    assert vecs.shape == (3, 240) and times.tolist() == [0.0, 2.0, 4.0]


def test_pipeline_uses_precomputed_values(tmp_path):
    schema = Schema("p", [
        MediaField("image", "image", extract=["caption"]),
        TextField("text", sources=["title", "image.caption"]),
        VectorField("img", 4, "clip", source="image"),
    ])
    Image.new("RGB", (4, 4)).save(tmp_path / "a.png")
    rec = {"id": "x", "title": "hello", "image": "a.png", "image.caption": "a black square", "img": [1, 0, 0, 0]}
    out = IngestPipeline(schema).process([rec], base_dir=tmp_path)[0]  # no model is loaded
    assert out["texts"]["text"] == "hello \na black square"
    assert np.array_equal(out["vectors"]["img"], [1, 0, 0, 0])
    assert "img" not in out["payload"]
