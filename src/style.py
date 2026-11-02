"""The one place the DYPOL style guide (style/style.guide.yaml) is loaded.

Everything else asks this module for rules, colours and fonts so the numbers
live in a single YAML file.
"""
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
GUIDE_PATH = ROOT / "style" / "style.guide.yaml"


@lru_cache(maxsize=1)
def guide():
    with open(GUIDE_PATH, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict) or "long" not in data or "look" not in data or "blocks" not in data:
        raise ValueError(f"style guide {GUIDE_PATH} is missing long/look/blocks sections")
    return data


def long_rules():
    return guide()["long"]


def block_kinds():
    return tuple(guide()["blocks"]["kinds"])


def rendered_kinds():
    return tuple(guide()["blocks"]["rendered"])


def look():
    return guide()["look"]


def rgb(name):
    """Palette colour as an (r, g, b) tuple, e.g. rgb('lime')."""
    value = look()["palette"][name].lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


_FONTS = {}


def font(role, size, weight=400):
    """Space Grotesk ('headline') or JetBrains Mono ('mono') at a given weight (variable fonts)."""
    from PIL import ImageFont
    key = (role, int(size), int(weight))
    if key not in _FONTS:
        spec = look()["fonts"][role]
        try:
            fnt = ImageFont.truetype(str(ROOT / spec["file"]), int(size))
            try:
                axes = fnt.get_variation_axes()
                lo, hi = axes[0]["minimum"], axes[0]["maximum"]
                fnt.set_variation_by_axes([max(lo, min(hi, weight))])
            except (OSError, AttributeError, IndexError, KeyError):
                pass  # FreeType without variation support: default instance
        except OSError:
            fallback = "/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf"
            try:
                fnt = ImageFont.truetype(fallback % ("Mono" if role == "mono" else ("-Bold" if weight >= 600 else "")),
                                         int(size))
            except OSError:
                fnt = ImageFont.load_default()
        _FONTS[key] = fnt
    return _FONTS[key]
