"""Simple vector icons drawn with Pillow primitives (DYPOL look: outline strokes, no fonts, no files).

`draw_icon(draw, name, box, color, width)` draws one icon inside box=(x0, y0, x1, y1).
`pick_icon(text, index)` maps a label to an icon by keyword; unknown labels rotate through a default set.
"""
import math
import re

NAMES = ("chip", "brain", "doc", "code", "chat", "search", "database", "cloud", "gear", "globe", "check",
         "warn", "bolt", "eye", "image", "sun", "lock", "chart", "layers", "spark", "arrow", "user")

_KEYWORDS = [
    ("code", r"code|program|python|function|script|syntax|compil|debug|bug|developer|api|github"),
    ("chat", r"prompt|chat|message|ask|question|answer|reply|convers|language|translat|speech|text|word"),
    ("brain", r"model|neural|brain|learn|train|think|reason|llm|ai\b|intellig|transformer|attention"),
    ("chip", r"gpu|chip|cpu|compute|hardware|processor|silicon|tpu"),
    ("database", r"data|dataset|database|memory|store|storage|record|history|token"),
    ("search", r"search|find|retriev|lookup|query|match|scan|detect"),
    ("doc", r"document|file|page|article|book|paper|read|corpus|dictionar"),
    ("cloud", r"cloud|server|internet|online|weather|rain|storm|forecast"),
    ("sun", r"sun|temperature|heat|climate|season|day"),
    ("gear", r"process|engine|pipeline|step|system|work|tool|automat|run"),
    ("globe", r"world|global|earth|countr|map|planet"),
    ("warn", r"catch|risk|limit|fail|wrong|error|halluc|danger|bias|problem|caveat|but\b"),
    ("check", r"test|verify|correct|valid|pass|result|output|done|answer"),
    ("bolt", r"fast|speed|instant|millisecond|second|quick|real.?time"),
    ("eye", r"see|vision|watch|look|view|recommend"),
    ("image", r"image|photo|picture|video|pixel|visual"),
    ("lock", r"secur|privacy|safe|password|encrypt|lock"),
    ("chart", r"predict|probab|chance|score|percent|statist|number|graph|trend|accuracy"),
    ("layers", r"layer|stack|vector|embedding|dimension|space"),
    ("user", r"user|you\b|people|person|human|viewer"),
]
_DEFAULT = ("spark", "layers", "gear", "chart", "brain", "arrow")


def pick_icon(text, index=0):
    t = str(text or "").lower()
    for name, pattern in _KEYWORDS:
        if re.search(pattern, t):
            return name
    return _DEFAULT[index % len(_DEFAULT)]


def draw_icon(draw, name, box, color, width=None):
    x0, y0, x1, y1 = [float(v) for v in box]
    s = min(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w = int(width or max(2, round(s / 14)))
    fn = _ICONS.get(name) or _ICONS["spark"]
    fn(draw, cx, cy, s / 2, color, w)


def _rr(d, box, r, color, w):
    d.rounded_rectangle([int(v) for v in box], radius=int(r), outline=color, width=w)


def _chip(d, cx, cy, r, c, w):
    k = r * 0.55
    _rr(d, (cx - k, cy - k, cx + k, cy + k), r * 0.12, c, w)
    d.rectangle([cx - k * 0.45, cy - k * 0.45, cx + k * 0.45, cy + k * 0.45], fill=c)
    for i in (-0.5, 0, 0.5):
        for a, b in ((cx + i * k, cy - k), (cx + i * k, cy + k)):
            d.line([(a, b), (a, b + (-1 if b < cy else 1) * r * 0.3)], fill=c, width=w)
        for a, b in ((cx - k, cy + i * k), (cx + k, cy + i * k)):
            d.line([(a, b), (a + (-1 if a < cx else 1) * r * 0.3, b)], fill=c, width=w)


def _brain(d, cx, cy, r, c, w):
    for dx in (-0.35, 0.35):
        d.ellipse([cx + dx * r - r * 0.5, cy - r * 0.75, cx + dx * r + r * 0.5, cy + r * 0.75], outline=c, width=w)
    d.line([(cx, cy - r * 0.7), (cx, cy + r * 0.7)], fill=c, width=w)
    for k in (-0.3, 0.2):
        d.arc([cx - r * 0.75, cy + k * r - r * 0.2, cx - r * 0.15, cy + k * r + r * 0.2], 200, 340, fill=c, width=w)
        d.arc([cx + r * 0.15, cy + k * r - r * 0.2, cx + r * 0.75, cy + k * r + r * 0.2], 200, 340, fill=c, width=w)


def _doc(d, cx, cy, r, c, w):
    x0, y0, x1, y1 = cx - r * 0.6, cy - r * 0.8, cx + r * 0.6, cy + r * 0.8
    f = r * 0.35
    d.line([(x0, y0), (x1 - f, y0), (x1, y0 + f), (x1, y1), (x0, y1), (x0, y0)], fill=c, width=w, joint="curve")
    d.line([(x1 - f, y0), (x1 - f, y0 + f), (x1, y0 + f)], fill=c, width=w)
    for k in range(3):
        y = y0 + r * (0.65 + 0.3 * k)
        d.line([(x0 + r * 0.25, y), (x1 - r * 0.25 - (r * 0.25 if k == 2 else 0), y)], fill=c, width=w)


def _code(d, cx, cy, r, c, w):
    d.line([(cx - r * 0.35, cy - r * 0.5), (cx - r * 0.8, cy), (cx - r * 0.35, cy + r * 0.5)], fill=c, width=w,
           joint="curve")
    d.line([(cx + r * 0.35, cy - r * 0.5), (cx + r * 0.8, cy), (cx + r * 0.35, cy + r * 0.5)], fill=c, width=w,
           joint="curve")
    d.line([(cx + r * 0.15, cy - r * 0.7), (cx - r * 0.15, cy + r * 0.7)], fill=c, width=w)


def _chat(d, cx, cy, r, c, w):
    _rr(d, (cx - r * 0.85, cy - r * 0.65, cx + r * 0.85, cy + r * 0.4), r * 0.25, c, w)
    d.line([(cx - r * 0.4, cy + r * 0.4), (cx - r * 0.55, cy + r * 0.8), (cx - r * 0.05, cy + r * 0.4)], fill=c,
           width=w, joint="curve")
    for k in (-0.4, 0, 0.4):
        rr = max(2, w * 0.8)
        d.ellipse([cx + k * r - rr, cy - r * 0.12 - rr, cx + k * r + rr, cy - r * 0.12 + rr], fill=c)


def _search(d, cx, cy, r, c, w):
    d.ellipse([cx - r * 0.7, cy - r * 0.7, cx + r * 0.25, cy + r * 0.25], outline=c, width=w)
    d.line([(cx + r * 0.15, cy + r * 0.15), (cx + r * 0.75, cy + r * 0.75)], fill=c, width=int(w * 1.5))


def _database(d, cx, cy, r, c, w):
    x0, x1 = cx - r * 0.65, cx + r * 0.65
    for k, y in enumerate((cy - r * 0.6, cy - r * 0.05, cy + r * 0.5)):
        d.arc([x0, y - r * 0.2, x1, y + r * 0.2], 0, 360 if k == 0 else 180, fill=c, width=w)
    d.line([(x0, cy - r * 0.6), (x0, cy + r * 0.5)], fill=c, width=w)
    d.line([(x1, cy - r * 0.6), (x1, cy + r * 0.5)], fill=c, width=w)


def _cloud(d, cx, cy, r, c, w):
    d.arc([cx - r * 0.85, cy - r * 0.15, cx - r * 0.15, cy + r * 0.55], 90, 270, fill=c, width=w)
    d.arc([cx - r * 0.5, cy - r * 0.6, cx + r * 0.3, cy + r * 0.2], 180, 330, fill=c, width=w)
    d.arc([cx + r * 0.0, cy - r * 0.35, cx + r * 0.85, cy + r * 0.55], 250, 90, fill=c, width=w)
    d.line([(cx - r * 0.5, cy + r * 0.55), (cx + r * 0.42, cy + r * 0.55)], fill=c, width=w)


def _gear(d, cx, cy, r, c, w):
    for k in range(8):
        a = k * math.pi / 4
        d.line([(cx + math.cos(a) * r * 0.55, cy + math.sin(a) * r * 0.55),
                (cx + math.cos(a) * r * 0.85, cy + math.sin(a) * r * 0.85)], fill=c, width=int(w * 1.6))
    d.ellipse([cx - r * 0.58, cy - r * 0.58, cx + r * 0.58, cy + r * 0.58], outline=c, width=w)
    d.ellipse([cx - r * 0.2, cy - r * 0.2, cx + r * 0.2, cy + r * 0.2], outline=c, width=w)


def _globe(d, cx, cy, r, c, w):
    k = r * 0.8
    d.ellipse([cx - k, cy - k, cx + k, cy + k], outline=c, width=w)
    d.ellipse([cx - k * 0.42, cy - k, cx + k * 0.42, cy + k], outline=c, width=w)
    d.line([(cx - k, cy), (cx + k, cy)], fill=c, width=w)
    d.arc([cx - k * 0.9, cy - k * 0.95, cx + k * 0.9, cy - k * 0.15], 20, 160, fill=c, width=w)


def _check(d, cx, cy, r, c, w):
    d.ellipse([cx - r * 0.8, cy - r * 0.8, cx + r * 0.8, cy + r * 0.8], outline=c, width=w)
    d.line([(cx - r * 0.4, cy), (cx - r * 0.1, cy + r * 0.32), (cx + r * 0.42, cy - r * 0.3)], fill=c,
           width=int(w * 1.3), joint="curve")


def _warn(d, cx, cy, r, c, w):
    d.line([(cx, cy - r * 0.8), (cx + r * 0.85, cy + r * 0.7), (cx - r * 0.85, cy + r * 0.7), (cx, cy - r * 0.8)],
           fill=c, width=w, joint="curve")
    d.line([(cx, cy - r * 0.3), (cx, cy + r * 0.2)], fill=c, width=int(w * 1.3))
    rr = w
    d.ellipse([cx - rr, cy + r * 0.42 - rr, cx + rr, cy + r * 0.42 + rr], fill=c)


def _bolt(d, cx, cy, r, c, w):
    pts = [(cx + r * 0.15, cy - r * 0.85), (cx - r * 0.5, cy + r * 0.1), (cx - r * 0.02, cy + r * 0.1),
           (cx - r * 0.15, cy + r * 0.85), (cx + r * 0.5, cy - r * 0.15), (cx + r * 0.02, cy - r * 0.15)]
    d.line(pts + pts[:1], fill=c, width=w, joint="curve")


def _eye(d, cx, cy, r, c, w):
    d.arc([cx - r * 0.9, cy - r * 0.55, cx + r * 0.9, cy + r * 0.85], 200, 340, fill=c, width=w)
    d.arc([cx - r * 0.9, cy - r * 0.85, cx + r * 0.9, cy + r * 0.55], 20, 160, fill=c, width=w)
    d.ellipse([cx - r * 0.25, cy - r * 0.25, cx + r * 0.25, cy + r * 0.25], fill=c)


def _image(d, cx, cy, r, c, w):
    _rr(d, (cx - r * 0.85, cy - r * 0.65, cx + r * 0.85, cy + r * 0.65), r * 0.12, c, w)
    d.line([(cx - r * 0.7, cy + r * 0.5), (cx - r * 0.2, cy - r * 0.05), (cx + r * 0.15, cy + r * 0.3),
            (cx + r * 0.35, cy + r * 0.1), (cx + r * 0.7, cy + r * 0.5)], fill=c, width=w, joint="curve")
    d.ellipse([cx + r * 0.3, cy - r * 0.45, cx + r * 0.55, cy - r * 0.2], outline=c, width=w)


def _sun(d, cx, cy, r, c, w):
    d.ellipse([cx - r * 0.38, cy - r * 0.38, cx + r * 0.38, cy + r * 0.38], outline=c, width=w)
    for k in range(8):
        a = k * math.pi / 4
        d.line([(cx + math.cos(a) * r * 0.56, cy + math.sin(a) * r * 0.56),
                (cx + math.cos(a) * r * 0.85, cy + math.sin(a) * r * 0.85)], fill=c, width=w)


def _lock(d, cx, cy, r, c, w):
    _rr(d, (cx - r * 0.6, cy - r * 0.1, cx + r * 0.6, cy + r * 0.75), r * 0.12, c, w)
    d.arc([cx - r * 0.4, cy - r * 0.75, cx + r * 0.4, cy + r * 0.1], 180, 360, fill=c, width=w)
    d.line([(cx - r * 0.4, cy - r * 0.33), (cx - r * 0.4, cy - r * 0.1)], fill=c, width=w)
    d.line([(cx + r * 0.4, cy - r * 0.33), (cx + r * 0.4, cy - r * 0.1)], fill=c, width=w)


def _chart(d, cx, cy, r, c, w):
    d.line([(cx - r * 0.8, cy - r * 0.8), (cx - r * 0.8, cy + r * 0.75), (cx + r * 0.85, cy + r * 0.75)], fill=c,
           width=w)
    for k, hgt in enumerate((0.45, 0.85, 0.65, 1.2)):
        x = cx - r * 0.5 + k * r * 0.38
        d.rectangle([x, cy + r * 0.75 - r * hgt, x + r * 0.22, cy + r * 0.75], fill=c)


def _layers(d, cx, cy, r, c, w):
    for k in (0.35, 0.0, -0.35):
        y = cy + k * r
        d.line([(cx, y - r * 0.35), (cx + r * 0.85, y), (cx, y + r * 0.35), (cx - r * 0.85, y), (cx, y - r * 0.35)],
               fill=c, width=w, joint="curve")


def _spark(d, cx, cy, r, c, w):
    pts = []
    for k in range(8):
        a = -math.pi / 2 + k * math.pi / 4
        rad = r * (0.85 if k % 2 == 0 else 0.25)
        pts.append((cx + math.cos(a) * rad, cy + math.sin(a) * rad))
    d.line(pts + pts[:1], fill=c, width=w, joint="curve")


def _arrow(d, cx, cy, r, c, w):
    d.line([(cx - r * 0.8, cy), (cx + r * 0.7, cy)], fill=c, width=int(w * 1.3))
    d.line([(cx + r * 0.25, cy - r * 0.45), (cx + r * 0.75, cy), (cx + r * 0.25, cy + r * 0.45)], fill=c,
           width=int(w * 1.3), joint="curve")


def _user(d, cx, cy, r, c, w):
    d.ellipse([cx - r * 0.32, cy - r * 0.8, cx + r * 0.32, cy - r * 0.16], outline=c, width=w)
    d.arc([cx - r * 0.7, cy + r * 0.0, cx + r * 0.7, cy + r * 1.3], 180, 360, fill=c, width=w)
    d.line([(cx - r * 0.7, cy + r * 0.65), (cx + r * 0.7, cy + r * 0.65)], fill=c, width=w)


_ICONS = {"chip": _chip, "brain": _brain, "doc": _doc, "code": _code, "chat": _chat, "search": _search,
          "database": _database, "cloud": _cloud, "gear": _gear, "globe": _globe, "check": _check, "warn": _warn,
          "bolt": _bolt, "eye": _eye, "image": _image, "sun": _sun, "lock": _lock, "chart": _chart,
          "layers": _layers, "spark": _spark, "arrow": _arrow, "user": _user}
