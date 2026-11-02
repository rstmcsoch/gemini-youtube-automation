"""Living backgrounds for DYPOL blocks: never a flat (or white) frame.

Two looks, both a pure function of time (chunked renders stay identical):
- "field": a slowly drifting slate gradient with soft lime/cyan/violet glows and a
  subtle grid baked into an oversized canvas that pans + breathes, plus drifting particles.
- "photo": a topic still (Pixabay / FLUX), blurred and darkened, with a slow Ken Burns
  move and the same particles.

The canvas is clamped to a dark luma band so a bright stock photo can never read as white.
"""
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageStat

from src import style

OVERSCAN = 1.16
MAX_MEAN_LUMA = 70.0          # backdrops sit dark so text and panels read on top
MIN_MEAN_LUMA = 14.0


def mean_luma(img):
    return ImageStat.Stat(img.convert("L")).mean[0]


def _clamp_luma(img, lo=MIN_MEAN_LUMA, hi=MAX_MEAN_LUMA):
    m = mean_luma(img)
    if m > hi:
        img = Image.eval(img, lambda v, k=hi / m: int(v * k))
    elif m < lo:
        lift = Image.new("RGB", img.size, style.rgb("panel"))
        img = Image.blend(img, lift, 0.5)
    return img


def _field(size, seed):
    rng = random.Random(seed)
    w, h = size
    sw, sh = max(8, w // 12), max(8, h // 12)
    base = style.rgb("background")
    small = Image.new("RGB", (sw, sh), base)
    glow = Image.new("RGB", (sw, sh), (0, 0, 0))
    gd = ImageDraw.Draw(glow)
    tints = [style.rgb("cyan"), style.rgb("lime"), (124, 92, 255), style.rgb("cyan")]
    rng.shuffle(tints)
    for k in range(3):
        cx, cy = rng.uniform(0.05, 0.95) * sw, rng.uniform(0.05, 0.95) * sh
        rad = rng.uniform(0.28, 0.5) * max(sw, sh)
        c = tints[k]
        strength = rng.uniform(0.16, 0.26)
        gd.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=tuple(int(v * strength) for v in c))
    glow = glow.filter(ImageFilter.GaussianBlur(max(sw, sh) / 7))
    from PIL import ImageChops
    small = ImageChops.add(small, glow)
    canvas = small.resize(size, Image.Resampling.BICUBIC)
    # subtle grid baked in (it moves with the pan, so it reads as a slowly travelling grid)
    grid = style.look()["grid"]
    sp = max(24, int(grid["spacing_px"] * min(w, h) / 1080))
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    a = int(255 * max(0.05, float(grid["opacity"]) * 1.5))
    for gx in range(0, w, sp):
        d.line([(gx, 0), (gx, h)], fill=style.rgb("body") + (a,), width=1)
    for gy in range(0, h, sp):
        d.line([(0, gy), (w, gy)], fill=style.rgb("body") + (a,), width=1)
    canvas = Image.alpha_composite(canvas.convert("RGBA"), layer).convert("RGB")
    return canvas


def _photo(path, size):
    w, h = size
    with Image.open(path) as im:
        img = im.convert("RGB")
    scale = max(w / img.width, h / img.height)
    img = img.resize((max(w, math.ceil(img.width * scale)), max(h, math.ceil(img.height * scale))),
                     Image.Resampling.BILINEAR)
    left, top = (img.width - w) // 2, (img.height - h) // 2
    img = img.crop((left, top, left + w, top + h))
    img = img.filter(ImageFilter.GaussianBlur(max(6, min(w, h) // 70)))
    slate = Image.new("RGB", size, style.rgb("background"))
    img = Image.blend(img, slate, 0.58)           # darken towards the DYPOL slate
    return _clamp_luma(img, hi=60.0)


class Backdrop:
    """frame(t) -> RGBA backdrop for time t (seconds)."""

    def __init__(self, size, seed=0, image=None, particles=36):
        self.size = tuple(size)
        w, h = self.size
        big = (int(w * OVERSCAN), int(h * OVERSCAN))
        self.mode = "field"
        canvas = None
        if image and Path(str(image)).exists():
            try:
                canvas = _photo(image, big)
                self.mode = "photo"
            except Exception:  # noqa: BLE001 - a bad image falls back to the field
                canvas = None
        if canvas is None:
            canvas = _clamp_luma(_field(big, seed))
        self.canvas = canvas
        rng = random.Random(seed * 7 + 3)
        self.phase = rng.uniform(0, 2 * math.pi)
        self.period = rng.uniform(16.0, 24.0)
        s = min(w, h) / 1080.0
        palette = [style.rgb("lime"), style.rgb("cyan"), style.rgb("body")]
        self.particles = []
        for _ in range(int(particles)):
            self.particles.append((rng.uniform(0, w), rng.uniform(0, h), rng.uniform(-14, 14) * s,
                                   rng.uniform(-30, -8) * s, rng.uniform(1.5, 4.0) * s,
                                   rng.uniform(0.18, 0.5), rng.choice(palette)))

    def frame(self, t):
        w, h = self.size
        sw, sh = self.canvas.size
        # slow breathing zoom + Lissajous pan (one full loop every ~20 s)
        a = 2 * math.pi * t / self.period + self.phase
        z = 1.0 + 0.035 * (1 + math.sin(a * 0.5))
        cw, ch = sw / OVERSCAN / z * 1.0, sh / OVERSCAN / z * 1.0
        cw, ch = min(cw, sw), min(ch, sh)
        fx = 0.5 + 0.42 * math.sin(a)
        fy = 0.5 + 0.42 * math.cos(a * 0.8)
        x0, y0 = (sw - cw) * fx, (sh - ch) * fy
        img = self.canvas.resize(self.size, Image.Resampling.BILINEAR, box=(x0, y0, x0 + cw, y0 + ch))
        d = ImageDraw.Draw(img, "RGBA")          # blends each particle onto the frame
        for px, py, vx, vy, r, alpha, color in self.particles:
            x = (px + vx * t) % w
            y = (py + vy * t) % h
            flick = 0.65 + 0.35 * math.sin(t * 1.7 + px)
            d.ellipse([x - r, y - r, x + r, y + r], fill=color + (int(255 * alpha * flick),))
        return img.convert("RGBA")


def seed_for(*parts):
    import zlib
    return zlib.crc32("|".join(str(p) for p in parts).encode("utf-8")) & 0xFFFF
