"""Generate a small synthetic multimodal catalogue for demos and smoke tests.

Creates under ``data/demo/``:
  images/   product pictures (drawn shapes with brand text) and street signs
  videos/   short lecture-style videos whose slides change every few seconds
  records.jsonl

The data is synthetic: it exercises every pipeline stage (OCR on images and
video keyframes, colour features, filters, video moments) without needing
model downloads. It is not a benchmark - use the real datasets for accuracy.

    python scripts/make_demo_data.py [--out data/demo] [--products 240]
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

COLORS = {
    "red": (205, 40, 40), "blue": (40, 80, 200), "green": (40, 150, 70), "black": (30, 30, 30),
    "white": (235, 235, 235), "yellow": (235, 200, 40), "orange": (240, 130, 30), "purple": (130, 60, 170),
}
BRANDS = ["NOVA", "ZENITH", "ACME", "PULSE", "ORBIT", "KITE"]
CATEGORIES = {
    "sneakers": ("running sneaker", "lightweight sneaker with a cushioned sole", (40, 160)),
    "t-shirts": ("cotton t-shirt", "soft crew-neck t-shirt, relaxed fit", (10, 45)),
    "mugs": ("ceramic mug", "350 ml ceramic mug, dishwasher safe", (6, 25)),
    "backpacks": ("travel backpack", "water-resistant backpack with a laptop sleeve", (35, 140)),
    "caps": ("baseball cap", "adjustable cap with a curved brim", (12, 40)),
}
SIGNS = [
    ("NO PARKING", "red"), ("EXIT", "green"), ("STOP", "red"), ("SALE 50% OFF", "yellow"),
    ("OPEN 24 HOURS", "blue"), ("WET FLOOR", "yellow"), ("PRIVATE PROPERTY", "black"), ("ONE WAY", "black"),
]
LECTURES = [
    ("lecture-btrees", "Lecture 7: B+ Trees", ["Lecture 7: B+ Trees", "Node splits", "Range scans", "Bulk loading"]),
    ("lecture-hashing", "Lecture 8: Hash Indexes", ["Lecture 8: Hash Indexes", "Extendible hashing", "Linear hashing", "Collisions"]),
    ("lecture-optimizer", "Lecture 10: Query Optimization", ["Lecture 10: Query Optimization", "Cost models", "Join ordering", "Histograms"]),
    ("lecture-vectors", "Lecture 12: Vector Search", ["Lecture 12: Vector Search", "HNSW graphs", "Product quantization", "Hybrid search"]),
]
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def font(size):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default(size=size)


def shade(rgb, f):
    return tuple(max(0, min(255, int(c * f))) for c in rgb)


def draw_product(cat, rgb, brand, rnd) -> Image.Image:
    im = Image.new("RGB", (256, 256), (245, 245, 242))
    d = ImageDraw.Draw(im)
    dark = shade(rgb, 0.6)
    text_rgb = (255, 255, 255) if sum(rgb) < 450 else (20, 20, 20)
    jitter = rnd.randint(-8, 8)
    if cat == "sneakers":
        d.polygon([(30, 170 + jitter), (60, 110), (120, 100), (160, 130), (225, 150), (230, 185), (30, 190)], fill=rgb)
        d.rectangle([28, 185, 232, 205], fill=(250, 250, 250), outline=dark, width=3)
        for x in range(80, 130, 12):
            d.line([(x, 112), (x + 8, 135)], fill=(250, 250, 250), width=3)
        d.text((120, 150), brand, fill=text_rgb, font=font(22), anchor="mm")
    elif cat == "t-shirts":
        d.polygon([(70, 50), (110, 40), (146, 40), (186, 50), (226, 90), (196, 115), (180, 100), (180, 220),
                   (76, 220), (76, 100), (60, 115), (30, 90)], fill=rgb, outline=dark)
        d.text((128, 140), brand, fill=text_rgb, font=font(30), anchor="mm")
    elif cat == "mugs":
        d.rounded_rectangle([60, 60, 170, 210], radius=12, fill=rgb, outline=dark, width=3)
        d.ellipse([155, 95, 215, 170], outline=rgb, width=14)
        d.text((115, 135), brand, fill=text_rgb, font=font(24), anchor="mm")
    elif cat == "backpacks":
        d.rounded_rectangle([60, 50, 196, 225], radius=30, fill=rgb, outline=dark, width=3)
        d.rounded_rectangle([82, 140, 174, 210], radius=12, fill=dark)
        d.arc([95, 20, 161, 80], 180, 360, fill=dark, width=8)
        d.text((128, 100), brand, fill=text_rgb, font=font(26), anchor="mm")
    else:  # caps
        d.chord([50, 70, 206, 210], 180, 360, fill=rgb, outline=dark, width=3)
        d.ellipse([120, 130, 240, 165], fill=dark)
        d.text((128, 115), brand, fill=text_rgb, font=font(24), anchor="mm")
    return im


def draw_sign(text, color, rnd) -> Image.Image:
    rgb = COLORS[color]
    bg = (rnd.randint(150, 200), rnd.randint(170, 210), rnd.randint(190, 230))
    im = Image.new("RGB", (320, 240), bg)
    d = ImageDraw.Draw(im)
    d.rectangle([150, 150, 170, 240], fill=(90, 90, 90))
    d.rounded_rectangle([20, 40, 300, 160], radius=14, fill=rgb, outline=(255, 255, 255), width=6)
    size = 44 if len(text) <= 8 else 26
    fg = (20, 20, 20) if color == "yellow" else (255, 255, 255)
    d.text((160, 100), text, fill=fg, font=font(size), anchor="mm")
    return im


def make_video(path: Path, slides, seconds_per_slide=4, fps=4, size=(480, 270)):
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise SystemExit("ffmpeg is required to build the demo videos")
    palette = [(32, 54, 92), (40, 90, 70), (110, 50, 40), (70, 50, 110)]
    frames = []
    for i, text in enumerate(slides):
        im = Image.new("RGB", size, palette[i % len(palette)])
        d = ImageDraw.Draw(im)
        d.text((size[0] // 2, size[1] // 2), text, fill=(255, 255, 255), font=font(30), anchor="mm")
        d.text((size[0] - 12, size[1] - 12), f"slide {i + 1}", fill=(200, 200, 200), font=font(14), anchor="rd")
        frames += [im.tobytes()] * (seconds_per_slide * fps)
    cmd = [exe, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}",
           "-r", str(fps), "-i", "-", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, input=b"".join(frames), check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/demo")
    ap.add_argument("--products", type=int, default=240)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rnd = random.Random(args.seed)
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "videos").mkdir(parents=True, exist_ok=True)
    records = []
    cats = list(CATEGORIES)
    for i in range(args.products):
        cat = cats[i % len(cats)]
        color = rnd.choice(list(COLORS))
        brand = rnd.choice(BRANDS)
        noun, desc, (lo, hi) = CATEGORIES[cat]
        img = out / "images" / f"p{i:04d}.png"
        draw_product(cat, COLORS[color], brand, rnd).save(img)
        # a third of the descriptions omit the colour: only the image knows it
        title = f"{brand.title()} {noun}" if i % 3 == 0 else f"{brand.title()} {color} {noun}"
        records.append({
            "id": f"p{i:04d}", "title": title, "description": f"{desc}. Model {brand[:2]}-{1000 + i}.",
            "image": f"images/{img.name}", "price": round(rnd.uniform(lo, hi), 2), "year": rnd.randint(2015, 2025),
            "category": cat, "brand": brand, "color": [color],
        })
    for j, (text, color) in enumerate(SIGNS * 3):
        img = out / "images" / f"s{j:03d}.png"
        draw_sign(text, color, rnd).save(img)
        records.append({"id": f"s{j:03d}", "title": "street sign photo", "description": "photo of a sign",
                        "image": f"images/{img.name}", "year": rnd.randint(2015, 2025), "category": "signs",
                        "color": [color]})
    for vid, title, slides in LECTURES:
        path = out / "videos" / f"{vid}.mp4"
        make_video(path, slides)
        records.append({"id": vid, "title": title, "description": "recorded database systems lecture",
                        "video": f"videos/{path.name}", "year": 2024, "category": "lectures"})
    with open(out / "records.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(records)} records to {out}")


if __name__ == "__main__":
    main()
