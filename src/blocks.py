"""CPU block renderers for Engine A (DYPOL look): Pillow frames, eased motion.

Each source follows the explainer source protocol: `frame(local_t) -> RGB Image`
(a pure function of local time, so chunked parallel renders stay identical to a
single pass) and `close()`. Built by `make_source` from an asset dict:

    {"kind": "block", "block": "<kind>", "spec": {...shot fields...}}
"""
import math
import re

from PIL import Image, ImageDraw

from src import backdrop as bd
from src import icons
from src import style

PROVIDER = "dypol_block"


# ------------------------------------------------------------------ helpers

def ease_in_out(p):
    p = max(0.0, min(1.0, p))
    return 4 * p * p * p if p < 0.5 else 1 - (-2 * p + 2) ** 3 / 2


def ease_out(p):
    p = max(0.0, min(1.0, p))
    return 1 - (1 - p) ** 3


def _alpha(img, a):
    if a >= 0.999:
        return img
    out = img.copy()
    out.putalpha(out.getchannel("A").point(lambda v: int(v * max(0.0, a))))
    return out


def _rgba(name, a=255):
    return style.rgb(name) + (a,)


class _Base:
    tag = ""

    def __init__(self, spec, size, span):
        self.spec = dict(spec or {})
        self.size = tuple(size)
        self.span = max(0.5, float(span))
        self.w, self.h = self.size
        # Portrait (Shorts, 1080x1920): scale by width, keep content in the upper safe zone
        # (clear of the captions at ~63 % and the Shorts UI at the bottom / right).
        self.portrait = self.h > self.w
        self.s = (self.w / 1080.0) if self.portrait else (self.h / 1080.0)
        self.margin = int(self.w * (0.08 if self.portrait else style.long_rules()["text"]["safe_margin"]))
        self.cy = int(self.h * 0.36) if self.portrait else self.h // 2
        self.bg = Image.new("RGBA", self.size, _rgba("background"))
        self.backdrop = bd.Backdrop(self.size, seed=bd.seed_for(type(self).__name__, self._seed_text()),
                                    image=self.spec.get("backdrop_image"))

    def _seed_text(self):
        return "|".join(str(self.spec.get(k, "")) for k in ("kind", "text", "title", "value", "nodes", "code"))

    def bg_frame(self, t):
        """Living background (drifting gradient + grid + particles, or a blurred topic still): RGBA."""
        return self.backdrop.frame(t)

    def px(self, v):
        return max(1, int(round(v * self.s)))

    def font(self, role, size, weight=400):
        return style.font(role, self.px(size), weight)

    def draw_tag(self, img, text, y=None, color="lime"):
        """Left-edge lime bar + mono uppercase label (DYPOL label system)."""
        d = ImageDraw.Draw(img)
        x = self.margin
        if y is None:
            y = int(self.h * 0.12) if self.portrait else self.px(72)
        f = self.font("mono", 30, 600)
        bar = style.look()["label_bar_px"]
        th = self.px(36)
        d.rectangle([x, y, x + self.px(bar) - 1, y + th], fill=_rgba(color))
        d.text((x + self.px(bar) + self.px(16), y + th / 2), str(text).upper()[:28], font=f,
               fill=_rgba("muted"), anchor="lm")

    def frame(self, local_t):
        raise NotImplementedError

    def close(self):
        pass


def _fit_font(role, text, max_w, start, minimum, weight, scale):
    size = start
    while size > minimum:
        f = style.font(role, int(size * scale), weight)
        if f.getlength(text) <= max_w:
            return f
        size -= 4
    return style.font(role, int(minimum * scale), weight)


# ------------------------------------------------------------------ code_panel

_TERMINAL = {"bash", "sh", "shell", "console", "zsh", "terminal", "powershell", "cmd"}


def _token_color(ttype):
    from pygments import token as T
    if ttype in T.Comment:
        return "muted"
    if ttype in T.String or ttype in T.Number or ttype in T.Literal:
        return "amber"
    if ttype in T.Keyword or ttype in T.Name.Builtin or ttype in T.Operator.Word:
        return "cyan"
    if ttype in T.Name.Function or ttype in T.Name.Class or ttype in T.Name.Decorator:
        return "lime"
    return "body"


def _colored_lines(code, language):
    """[[(text, colour_name), ...] per line] via Pygments; plain text if the lexer is unknown."""
    try:
        from pygments.lexers import get_lexer_by_name
        from pygments.util import ClassNotFound
        try:
            lexer = get_lexer_by_name(language or "text", stripnl=False, ensurenl=False)
        except ClassNotFound:
            lexer = get_lexer_by_name("text", stripnl=False, ensurenl=False)
        tokens = list(lexer.get_tokens(code))
    except ImportError:
        tokens = None
    lines = [[]]
    if tokens is None:
        return [[(line, "body")] for line in code.split("\n")]
    for ttype, value in tokens:
        parts = value.split("\n")
        for k, part in enumerate(parts):
            if k:
                lines.append([])
            if part:
                lines[-1].append((part, _token_color(ttype)))
    while lines and not lines[-1]:
        lines.pop()
    return lines or [[("", "body")]]


class CodePanelSource(_Base):
    """Editor or terminal window; code types in with syntax colours and a cursor, then the key line lights up."""

    def __init__(self, spec, size, span):
        super().__init__(spec, size, span)
        code = str(self.spec.get("code") or "print('hello')").replace("\t", "    ").rstrip("\n")
        lang = str(self.spec.get("language") or "python").lower()
        self.terminal = lang in _TERMINAL
        self.lines = _colored_lines(code, "bash" if self.terminal else lang)
        self.font_ = self.font("mono", 34, 400)
        self.lh = self.px(56)
        # panel geometry
        n = len(self.lines)
        ph = self.px(64) + self.px(48) * 2 + self.lh * n
        ph = min(ph, (int(self.h * 0.42) if self.portrait else self.h - self.px(260)))
        pw = int(self.w * (0.86 if self.portrait else 0.80))
        top = self.cy - ph // 2 + self.px(30)
        self.box = ((self.w - pw) // 2, top, (self.w + pw) // 2, top + ph)
        self.gutter = 0 if self.terminal else self.px(80)
        self.prompt = "$ " if self.terminal else ""
        self.text_x = self.box[0] + self.px(44) + self.gutter
        self.text_y = self.box[1] + self.px(64) + self.px(40)
        self.base = self._base()
        self.line_imgs = [self._line_img(i, toks) for i, toks in enumerate(self.lines)]
        self.line_lens = [sum(len(t) for t, _ in toks) for toks in self.lines]
        total_chars = max(1, sum(self.line_lens))
        cps = float(style.guide()["blocks"]["code_panel"]["chars_per_s"])
        self.t0 = 0.45
        budget = max(0.6, self.span * 0.68 - self.t0)
        self.cps = max(cps, total_chars / budget)
        self.t_done = self.t0 + total_chars / self.cps + 0.12 * n
        hl = self.spec.get("highlight_line")
        if not isinstance(hl, int) or not 1 <= hl <= n:
            hl = max((i for i, ln in enumerate(self.line_lens) if ln), default=0) + 1
        self.hl = hl - 1

    def _base(self):
        img = Image.new("RGBA", self.size, (0, 0, 0, 0))     # drawn over the living backdrop each frame
        self.draw_tag(img, "Terminal" if self.terminal else "Code")
        d = ImageDraw.Draw(img)
        x0, y0, x1, y1 = self.box
        r = self.px(style.look()["panel_radius_px"])
        d.rounded_rectangle(self.box, radius=r, fill=_rgba("panel"), outline=_rgba("border"), width=max(1, self.px(1)))
        d.line([(x0, y0 + self.px(64)), (x1, y0 + self.px(64))], fill=_rgba("border"), width=max(1, self.px(1)))
        for k in range(3):
            cx, cy, rr = x0 + self.px(36) + k * self.px(26), y0 + self.px(32), self.px(7)
            d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=_rgba("border"))
        title = self.spec.get("title") or ("bash" if self.terminal else f"main.{self._ext()}")
        d.text(((x0 + x1) / 2, y0 + self.px(32)), str(title)[:40], font=self.font("mono", 24, 400),
               fill=_rgba("muted"), anchor="mm")
        if self.gutter:
            for i in range(len(self.lines)):
                d.text((x0 + self.px(44) + self.gutter - self.px(28), self.text_y + i * self.lh + self.lh / 2),
                       str(i + 1), font=self.font("mono", 28, 400), fill=_rgba("border"), anchor="rm")
        return img

    def _ext(self):
        return {"python": "py", "javascript": "js", "typescript": "ts", "json": "json", "yaml": "yaml",
                "rust": "rs", "go": "go"}.get(str(self.spec.get("language", "python")).lower(), "txt")

    def _line_img(self, i, toks):
        width = self.box[2] - self.text_x - self.px(30)
        img = Image.new("RGBA", (width, self.lh), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        x = 0
        if self.prompt:
            d.text((0, self.lh / 2), self.prompt, font=self.font_, fill=_rgba("lime"), anchor="lm")
            x = self.font_.getlength(self.prompt)
        self.prompt_w = x
        for text, color in toks:
            d.text((x, self.lh / 2), text, font=self.font_, fill=_rgba(color), anchor="lm")
            x += self.font_.getlength(text)
        return img

    def _x_for(self, i, chars):
        text = "".join(t for t, _ in self.lines[i])[:chars]
        return self.prompt_w + self.font_.getlength(text)

    def frame(self, local_t):
        t = local_t
        img = self.bg_frame(t)
        a = ease_out(t / 0.35)
        # the editor window slides up into place while it fades in
        img.alpha_composite(_alpha(self.base, a), (0, int(self.px(40) * (1 - a))))
        # characters typed so far (each line also costs a short "enter" pause)
        budget = max(0.0, t - self.t0)
        cur_line, cur_chars = 0, 0
        for i, n in enumerate(self.line_lens):
            need = n / self.cps + 0.12
            if budget >= need:
                budget -= need
                cur_line, cur_chars = i + 1, 0
                continue
            cur_line, cur_chars = i, min(n, int(budget * self.cps))
            break
        done = cur_line >= len(self.lines)
        d = ImageDraw.Draw(img)
        # highlight bar sweep behind the key line, after typing
        if done:
            p = ease_in_out((t - self.t_done - 0.15) / 0.35)
            if p > 0:
                y = self.text_y + self.hl * self.lh
                x0 = self.box[0] + self.px(14)
                x1 = x0 + (self.box[2] - self.px(14) - x0) * p
                hl = Image.new("RGBA", self.size, (0, 0, 0, 0))
                ImageDraw.Draw(hl).rectangle([x0, y + self.px(3), x1, y + self.lh - self.px(3)],
                                             fill=_rgba("lime", 46))
                ImageDraw.Draw(hl).rectangle([x0, y + self.px(3), x0 + self.px(5), y + self.lh - self.px(3)],
                                             fill=_rgba("lime", 255))
                img = Image.alpha_composite(img, hl)
        for i, line in enumerate(self.line_imgs):
            y = self.text_y + i * self.lh
            if i < cur_line:
                img.alpha_composite(line, (self.text_x, y))
            elif i == cur_line and not done:
                cut = int(math.ceil(self._x_for(i, cur_chars)))
                if cut > 0:
                    img.alpha_composite(line.crop((0, 0, min(cut, line.width), line.height)), (self.text_x, y))
        # cursor: solid while typing, 1 Hz blink after
        if done:
            ci, cx = len(self.lines) - 1, self._x_for(len(self.lines) - 1, self.line_lens[-1])
            visible = (t - self.t_done) % 1.0 < 0.55
        else:
            ci, cx, visible = cur_line, self._x_for(cur_line, cur_chars), t >= self.t0 - 0.2
        if visible and a > 0.5:
            x, y = self.text_x + cx + self.px(3), self.text_y + ci * self.lh
            ImageDraw.Draw(img).rectangle([x, y + self.px(10), x + self.px(16), y + self.lh - self.px(10)],
                                          fill=_rgba("lime"))
        del d
        return img.convert("RGB")


# ------------------------------------------------------------------ diagram

class DiagramSource(_Base):
    """2-5 nodes (icon + label) connected by arrows that draw step by step; then data packets
    travel the chain continuously and the node they reach lights up. Horizontal in 16:9,
    vertical in 9:16."""

    def __init__(self, spec, size, span):
        super().__init__(spec, size, span)
        nodes = [str(n) for n in (self.spec.get("nodes") or []) if str(n).strip()]
        cfg = style.guide()["blocks"]["diagram"]
        nodes = nodes[: cfg["max_nodes"]]
        while len(nodes) < cfg["min_nodes"]:
            nodes.append("Result" if nodes else "Input")
        self.nodes = nodes
        given = self.spec.get("icons") or []
        self.icon_names = [given[k] if k < len(given) and given[k] in icons.NAMES else icons.pick_icon(lbl, k)
                           for k, lbl in enumerate(nodes)]
        n = len(nodes)
        if self.portrait:
            gap = self.px(70)
            top, bottom = int(self.h * 0.17), int(self.h * 0.56)
            bh = min(self.px(210), (bottom - top - gap * (n - 1)) // n)
            bw = int(self.w * 0.70)
            total = bh * n + gap * (n - 1)
            x = (self.w - bw) // 2
            y = top + (bottom - top - total) // 2
            self.boxes = [(x, y + k * (bh + gap), x + bw, y + k * (bh + gap) + bh) for k in range(n)]
        else:
            gap = self.px(110)
            avail = self.w - 2 * self.margin - gap * (n - 1)
            bw = min(self.px(380), avail // n)
            bh = self.px(230)
            total = bw * n + gap * (n - 1)
            x = (self.w - total) // 2
            y = (self.h - bh) // 2 + self.px(30)
            self.boxes = [(x + k * (bw + gap), y, x + k * (bw + gap) + bw, y + bh) for k in range(n)]
        self.base = self._base()
        self.node_imgs = [(self._node(k, False), self._node(k, True)) for k in range(n)]
        steps = 2 * n - 1
        self.build = min(max(1.2, self.span * 0.62), 0.55 * steps)
        self.step = self.build / steps
        self.t0 = 0.25

    def _base(self):
        img = Image.new("RGBA", self.size, (0, 0, 0, 0))
        self.draw_tag(img, self.spec.get("title") or "How it works", color="cyan")
        return img

    def _node(self, k, active):
        x0, y0, x1, y1 = self.boxes[k]
        w, h = x1 - x0, y1 - y0
        pad = self.px(12)
        img = Image.new("RGBA", (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        r = self.px(style.look()["panel_radius_px"])
        if active:
            d.rounded_rectangle([0, 0, w + 2 * pad - 1, h + 2 * pad - 1], radius=r + pad, fill=_rgba("lime", 34))
        d.rounded_rectangle([pad, pad, pad + w - 1, pad + h - 1], radius=r, fill=_rgba("panel", 236),
                            outline=_rgba("lime" if active else "border"), width=self.px(3) if active else self.px(2))
        d.text((pad + self.px(20), pad + self.px(22)), f"{k + 1:02d}", font=self.font("mono", 24, 500),
               fill=_rgba("lime" if active else "cyan"), anchor="lt")
        icon_c = _rgba("lime" if active else "cyan")
        if self.portrait:
            isz = min(h - self.px(60), self.px(110))
            ib = (pad + self.px(70), pad + (h - isz) / 2, pad + self.px(70) + isz, pad + (h + isz) / 2)
            text_x0, text_w = ib[2] + self.px(30), w - (ib[2] - pad) - self.px(50)
            text_cy = pad + h / 2
        else:
            isz = self.px(78)
            ib = (pad + (w - isz) / 2, pad + self.px(30), pad + (w + isz) / 2, pad + self.px(30) + isz)
            text_x0, text_w = pad + self.px(18), w - self.px(36)
            text_cy = pad + self.px(30) + isz + (h - self.px(30) - isz) / 2
        icons.draw_icon(d, self.icon_names[k], ib, icon_c, width=max(2, self.px(5)))
        label = self.nodes[k]
        f = _fit_font("headline", label, text_w, 46 if not self.portrait else 58, 26, 600, self.s)
        words, lines = label.split(), [""]
        for word in words:
            trial = (lines[-1] + " " + word).strip()
            if f.getlength(trial) <= text_w or not lines[-1]:
                lines[-1] = trial
            else:
                lines.append(word)
        lines = lines[:2]
        lh = f.size * 1.15
        cy = text_cy - lh * (len(lines) - 1) / 2
        for i, line in enumerate(lines):
            if self.portrait:
                d.text((text_x0, cy + i * lh), line, font=f, fill=_rgba("body"), anchor="lm")
            else:
                d.text((text_x0 + text_w / 2, cy + i * lh), line, font=f, fill=_rgba("body"), anchor="mm")
        return img

    def _ends(self, k):
        a, b = self.boxes[k], self.boxes[k + 1]
        if self.portrait:
            x = (a[0] + a[2]) / 2
            return (x, a[3] + self.px(12)), (x, b[1] - self.px(12))
        y = (a[1] + a[3]) / 2
        return (a[2] + self.px(14), y), (b[0] - self.px(14), y)

    def _arrow(self, img, k, p, color="border"):
        if p <= 0:
            return
        d = ImageDraw.Draw(img)
        (x0, y0), (x1, y1) = self._ends(k)
        xe, ye = x0 + (x1 - x0) * p, y0 + (y1 - y0) * p
        d.line([(x0, y0), (xe, ye)], fill=_rgba(color), width=self.px(4))
        if p > 0.6:
            hs = self.px(14)
            if self.portrait:
                d.polygon([(xe, ye), (xe - hs, ye - hs * 1.3), (xe + hs, ye - hs * 1.3)], fill=_rgba(color))
            else:
                d.polygon([(xe, ye), (xe - hs * 1.3, ye - hs), (xe - hs * 1.3, ye + hs)], fill=_rgba(color))

    def _packets(self, img, t, built_arrows):
        """Glowing data packets travel each drawn arrow continuously (the 'flow')."""
        d = ImageDraw.Draw(img, "RGBA")
        for k in range(built_arrows):
            (x0, y0), (x1, y1) = self._ends(k)
            for j in range(2):
                q = ((t * 0.9) + j * 0.5 + k * 0.17) % 1.0
                x, y = x0 + (x1 - x0) * q, y0 + (y1 - y0) * q
                for rad, a in ((self.px(16), 60), (self.px(8), 255)):
                    d.ellipse([x - rad, y - rad, x + rad, y + rad], fill=_rgba("lime", a))

    def frame(self, local_t):
        t = local_t - self.t0
        img = self.bg_frame(local_t)
        img.alpha_composite(self.base)
        n = len(self.nodes)
        built = t / self.step if self.step else 99
        active = min(n - 1, max(0, int(built // 2)))
        complete = t >= self.build
        drawn = 0
        for k in range(n - 1):
            p = ease_in_out((t - (2 * k + 1) * self.step) / (self.step * 0.9))
            self._arrow(img, k, p, color="cyan" if complete or k < active else "muted")
            drawn += p >= 1
        for k in range(n):
            p = ease_out((t - 2 * k * self.step) / 0.35)
            if p <= 0:
                continue
            is_active = (not complete and k == active)
            if complete:
                # after the build, a lime pulse walks the chain (keeps the frame moving)
                cycle = max(0.9, 0.6 * n)
                is_active = int(((t - self.build) % cycle) / cycle * n) == k
            node = self.node_imgs[k][1 if is_active else 0]
            pad = self.px(12)
            bob = int(self.px(5) * math.sin(local_t * 1.6 + k * 1.3)) if p >= 1 else 0
            if self.portrait:
                x, y = self.boxes[k][0] - pad + int(self.px(60) * (1 - p)), self.boxes[k][1] - pad
                x += bob
            else:
                x, y = self.boxes[k][0] - pad, self.boxes[k][1] - pad + int(self.px(24) * (1 - p)) + bob
            img.alpha_composite(_alpha(node, p), (x, y))
        self._packets(img, local_t, drawn)
        return img.convert("RGB")


# ------------------------------------------------------------------ chapter_card

class ChapterCardSource(_Base):
    """Chapter number + title animate in; a progress spine shows where we are."""

    def __init__(self, spec, size, span):
        super().__init__(spec, size, span)
        try:
            self.number = int(self.spec.get("number") or 1)
        except (TypeError, ValueError):
            self.number = 1
        try:
            self.total = max(self.number, int(self.spec.get("total") or self.number))
        except (TypeError, ValueError):
            self.total = self.number
        self.title = str(self.spec.get("title") or "Next up").strip()
        self.num_font = self.font("headline", 300, 700)
        self.title_font = _fit_font("headline", self.title, self.w - self.margin * 2 - self.px(560), 104, 60, 700,
                                    self.s)
        self.base = self.bg.copy()
        self.num_img = self._text_img(f"{self.number:02d}", self.num_font, "lime")
        self.words = [self._text_img(wd, self.title_font, "body") for wd in self.title.split()]
        self.tag_img = Image.new("RGBA", self.size, (0, 0, 0, 0))
        self.draw_tag(self.tag_img, f"Chapter {self.number:02d}", y=self.h // 2 - self.px(130))

    @staticmethod
    def _text_img(text, fnt, color):
        """Tight horizontally, but a shared ascent/descent box so words in a line keep one baseline."""
        l, _, r, _ = fnt.getbbox(text, anchor="ls")
        ascent, descent = fnt.getmetrics()
        img = Image.new("RGBA", (r - l + 4, ascent + descent + 4), (0, 0, 0, 0))
        ImageDraw.Draw(img).text((-l + 2, ascent + 2), text, font=fnt, fill=_rgba(color), anchor="ls")
        return img

    def frame(self, local_t):
        t = local_t
        img = self.bg_frame(t)
        cy = self.h // 2
        # number slides in from the left
        p = ease_out(t / 0.45)
        nx = self.margin + int(-self.px(80) * (1 - p))
        img.alpha_composite(_alpha(self.num_img, p), (nx, cy - self.num_img.height // 2))
        # divider grows
        dx = self.margin + self.num_img.width + self.px(56)
        dp = ease_in_out((t - 0.15) / 0.4)
        if dp > 0:
            hh = int(self.px(120) * dp)
            ImageDraw.Draw(img).rectangle([dx, cy - hh, dx + self.px(4), cy + hh], fill=_rgba("border"))
        # tag + title words rise in, staggered
        tx = dx + self.px(56)
        tp = ease_out((t - 0.2) / 0.35)
        if tp > 0:
            img.alpha_composite(_alpha(self.tag_img.crop((self.margin, 0, self.w, self.h)), tp), (tx, 0))
        x = tx
        space = self.title_font.getlength(" ")
        line_y = cy - self.px(40)
        for i, wimg in enumerate(self.words):
            wp = ease_out((t - 0.3 - 0.08 * i) / 0.35)
            if x + wimg.width > self.w - self.margin:
                x, line_y = tx, line_y + int(self.title_font.size * 1.15)
            if wp > 0:
                img.alpha_composite(_alpha(wimg, wp), (int(x), int(line_y + self.px(30) * (1 - wp))))
            x += wimg.width + space
        # progress spine (one segment per chapter), current one fills lime
        d = ImageDraw.Draw(img)
        seg_gap = self.px(10)
        sx0, sx1 = self.margin, self.w - self.margin
        sy = self.h - self.px(110)
        seg = (sx1 - sx0 - seg_gap * (self.total - 1)) / self.total
        fill = ease_in_out((t - 0.5) / 0.6)
        for k in range(self.total):
            a = sx0 + k * (seg + seg_gap)
            d.rectangle([a, sy, a + seg, sy + self.px(6)], fill=_rgba("border"))
            if k < self.number - 1:
                d.rectangle([a, sy, a + seg, sy + self.px(6)], fill=_rgba("muted"))
            elif k == self.number - 1 and fill > 0:
                d.rectangle([a, sy, a + seg * fill, sy + self.px(6)], fill=_rgba("lime"))
        # slow underline under the title keeps motion going during the hold
        up = ease_in_out((t - 0.9) / max(0.5, self.span - 1.2))
        if up > 0:
            ly = line_y + int(self.title_font.size * 1.45)
            d.rectangle([tx, ly, tx + (self.w - self.margin - tx) * 0.35 * up, ly + self.px(4)], fill=_rgba("lime"))
        return img.convert("RGB")


# ------------------------------------------------------------------ kinetic_text / count_up (simple versions)

class KineticTextSource(_Base):
    """1-6 words slam in (scale 110 -> 100 %, fade); a lime highlighter block sweeps behind the key word."""

    def __init__(self, spec, size, span):
        super().__init__(spec, size, span)
        text = str(self.spec.get("text") or " ".join(self.spec.get("keywords") or []) or "Here's the point")
        self.words = text.split()[:8]
        self.f = _fit_font("headline", text, self.w - 2 * self.margin, 132, 72, 700, self.s)
        key = next((i for i, wd in enumerate(self.words) if re.search(r"\d", wd)), len(self.words) - 1)
        self.key = key
        # layout: up to 2 centred lines
        maxw = self.w - 2 * self.margin
        lines, cur = [[]], 0.0
        sp = self.f.getlength(" ")
        for i, wd in enumerate(self.words):
            wl = self.f.getlength(wd)
            if lines[-1] and cur + sp + wl > maxw:
                lines.append([])
                cur = 0.0
            cur = cur + (sp if lines[-1] else 0) + wl
            lines[-1].append(i)
        lh = int(self.f.size * 1.2)
        top = self.cy - lh * len(lines) / 2
        self.text_bottom = top + lh * len(lines)
        self.pos = {}
        for li, idxs in enumerate(lines):
            width = sum(self.f.getlength(self.words[i]) for i in idxs) + sp * (len(idxs) - 1)
            x = (self.w - width) / 2
            for i in idxs:
                self.pos[i] = (x, top + li * lh, self.f.getlength(self.words[i]), lh)
                x += self.f.getlength(self.words[i]) + sp
        self.imgs = [ChapterCardSource._text_img(wd, self.f, "body") for wd in self.words]
        self.tag = self.spec.get("label") or ""
        self.accent = "amber" if str(self.tag).lower().startswith(("the catch", "catch", "but")) else "lime"
        # a small icon chip slides in and floats; a comet travels a rail under the words (flow)
        self.icon_name = self.spec.get("icon") if self.spec.get("icon") in icons.NAMES else \
            icons.pick_icon(f"{self.tag} {text}")
        isz = self.px(150)
        self.chip = Image.new("RGBA", (isz, isz), (0, 0, 0, 0))
        cd = ImageDraw.Draw(self.chip)
        cd.rounded_rectangle([0, 0, isz - 1, isz - 1], radius=self.px(28), fill=_rgba("panel", 230),
                             outline=_rgba(self.accent), width=self.px(3))
        icons.draw_icon(cd, self.icon_name, (isz * 0.22, isz * 0.22, isz * 0.78, isz * 0.78), _rgba(self.accent),
                        width=max(2, self.px(6)))

    def _flow(self, img, t, drift):
        d = ImageDraw.Draw(img, "RGBA")
        # rail under the text draws in, then a comet travels it continuously
        y = self.text_bottom + self.px(40) - drift
        x0, x1 = self.margin + self.px(40), self.w - self.margin - self.px(40)
        rp = ease_in_out((t - 0.2) / 0.6)
        if rp > 0:
            d.line([(x0, y), (x0 + (x1 - x0) * rp, y)], fill=_rgba("border", 255), width=self.px(3))
        if rp >= 1:
            q = ((t - 0.8) * 0.45) % 1.0
            cx = x0 + (x1 - x0) * q
            for k in range(8):
                tx = cx - k * self.px(18)
                if tx < x0:
                    break
                d.line([(tx, y), (tx - self.px(18), y)], fill=_rgba(self.accent, int(230 * (1 - k / 8))),
                       width=self.px(5))
            rr = self.px(9)
            d.ellipse([cx - rr, y - rr, cx + rr, y + rr], fill=_rgba(self.accent))
        # icon chip slides in from the right edge and floats
        cp = ease_out((t - 0.35) / 0.5)
        if cp > 0:
            if self.portrait:
                bx = self.w - self.margin - self.chip.width
                by = int(self.h * 0.12) + self.px(70)
            else:
                bx = self.w - self.margin - self.chip.width
                by = self.px(140)
            bx += int(self.px(220) * (1 - cp))
            by += int(self.px(10) * math.sin(t * 1.5))
            img.alpha_composite(_alpha(self.chip, cp), (bx, by))

    def frame(self, local_t):
        t = local_t
        img = self.bg_frame(t)
        if self.tag:
            self.draw_tag(img, self.tag, color=self.accent)
        drift = int(self.px(14) * ease_in_out(t / self.span))
        self._flow(img, t, drift)
        all_in = 0.1 + 0.16 * len(self.words) + 0.2
        hp = ease_in_out((t - all_in) / 0.3)
        if hp > 0:
            x, y, wl, lh = self.pos[self.key]
            ImageDraw.Draw(img).rectangle([x - self.px(14), y + lh * 0.12 - drift,
                                           x - self.px(14) + (wl + self.px(28)) * hp, y + lh * 0.95 - drift],
                                          fill=_rgba(self.accent, 255))
        for i, wimg in enumerate(self.imgs):
            p = ease_out((t - 0.1 - 0.16 * i) / 0.25)
            if p <= 0:
                continue
            sc = 1.1 - 0.1 * p
            im = wimg if sc <= 1.001 else wimg.resize((int(wimg.width * sc), int(wimg.height * sc)),
                                                       Image.Resampling.BILINEAR)
            x, y, wl, lh = self.pos[i]
            cx, cy = x + wl / 2, y + lh / 2 - drift
            if i == self.key and hp > 0.5:
                im = ChapterCardSource._text_img(self.words[i], self.f, "background")
            img.alpha_composite(_alpha(im, p), (int(cx - im.width / 2), int(cy - im.height / 2)))
        return img.convert("RGB")


def format_number(value, decimals):
    text = f"{value:,.{decimals}f}"
    return text


class CountUpSource(_Base):
    """One big number counts up (ease-out), with a label and an optional source line."""

    def __init__(self, spec, size, span):
        super().__init__(spec, size, span)
        from src.shotlist import parse_number
        parsed = parse_number(self.spec.get("value")) or (100.0, 0)
        self.value, self.dec = parsed
        self.prefix = str(self.spec.get("prefix") or "")
        self.suffix = str(self.spec.get("suffix") or "")
        final = f"{self.prefix}{format_number(self.value, self.dec)}{self.suffix}"
        self.f = _fit_font("headline", final, self.w - 2 * self.margin, 240, 130, 700, self.s)
        self.final_w = self.f.getlength(final)
        self.label = str(self.spec.get("label") or "")
        self.lf = self.font("headline", 56, 500)
        self.source = str(self.spec.get("source") or "")
        self.sf = self.font("mono", 26, 400)

    def _meter(self, img, t, p, y):
        """A segmented meter fills with the count (share of 100 for a %); a glint keeps sweeping it."""
        frac = min(1.0, self.value / 100.0) if self.suffix == "%" else 1.0
        d = ImageDraw.Draw(img, "RGBA")
        x0, x1 = self.margin + self.px(60), self.w - self.margin - self.px(60)
        my = y - self.f.size - self.px(50)
        segs = 24
        gap = self.px(6)
        sw = (x1 - x0 - gap * (segs - 1)) / segs
        lit = frac * p * segs
        for k in range(segs):
            a = x0 + k * (sw + gap)
            on = k < lit
            d.rounded_rectangle([a, my, a + sw, my + self.px(14)], radius=self.px(4),
                                fill=_rgba("lime" if on else "border", 255 if on else 200))
        if p > 0.95:
            q = ((t * 0.5) % 1.0)
            gx = x0 + (x1 - x0) * frac * q
            d.ellipse([gx - self.px(14), my - self.px(7), gx + self.px(14), my + self.px(21)],
                      fill=_rgba("body", 150))

    def frame(self, local_t):
        t = local_t
        img = self.bg_frame(t)
        self.draw_tag(img, self.spec.get("title") or "By the numbers")
        d = ImageDraw.Draw(img)
        drift = int(self.px(12) * ease_in_out(t / self.span))
        p = ease_out((t - 0.15) / 0.9)
        cur = self.value * p
        num = format_number(cur, self.dec)
        x = (self.w - self.final_w) / 2
        y = self.cy - self.px(70) - drift
        self._meter(img, t, p, y)
        d = ImageDraw.Draw(img)
        d.text((x, y), self.prefix, font=self.f, fill=_rgba("body"), anchor="ls")
        x += self.f.getlength(self.prefix)
        d.text((x, y), num, font=self.f, fill=_rgba("lime"), anchor="ls")
        x += self.f.getlength(num)
        d.text((x, y), self.suffix, font=self.f, fill=_rgba("body"), anchor="ls")
        if self.label:
            lp = ease_out((t - 0.5) / 0.35)
            ly = self.cy + self.px(60) - drift + int(self.px(20) * (1 - lp))
            lw = self.lf.getlength(self.label)
            hp = ease_in_out((t - 1.15) / 0.3)
            if hp > 0:
                d.rectangle([(self.w - lw) / 2 - self.px(14), ly - self.px(6),
                             (self.w - lw) / 2 - self.px(14) + (lw + self.px(28)) * hp, ly + self.px(70)],
                            fill=_rgba("lime", 255))
            layer = Image.new("RGBA", self.size, (0, 0, 0, 0))
            ImageDraw.Draw(layer).text((self.w / 2, ly), self.label, font=self.lf,
                                       fill=_rgba("background" if hp > 0.5 else "body"), anchor="mt")
            img.alpha_composite(_alpha(layer, lp))
        if self.source:
            sp = ease_out((t - 0.8) / 0.4)
            layer = Image.new("RGBA", self.size, (0, 0, 0, 0))
            sy = int(self.h * 0.56) if self.portrait else self.h - self.px(80)
            ImageDraw.Draw(layer).text((self.margin, sy), f"Source: {self.source}", font=self.sf,
                                       fill=_rgba("muted"), anchor="ls")
            img.alpha_composite(_alpha(layer, sp))
        return img.convert("RGB")


# ------------------------------------------------------------------ icon_flow

def _bezier(p0, p1, p2, p3, n=48):
    pts = []
    for i in range(n + 1):
        u = i / n
        a, b, c, e = (1 - u) ** 3, 3 * u * (1 - u) ** 2, 3 * u * u * (1 - u), u ** 3
        pts.append((a * p0[0] + b * p1[0] + c * p2[0] + e * p3[0], a * p0[1] + b * p1[1] + c * p2[1] + e * p3[1]))
    return pts


class IconFlowSource(_Base):
    """2-4 icon cards slide in along a curved path; the path draws itself, then a glowing token
    travels card to card continuously and each card pops as it arrives (a visual, not a text card)."""

    def __init__(self, spec, size, span):
        super().__init__(spec, size, span)
        labels = [str(x).strip() for x in (self.spec.get("nodes") or []) if str(x).strip()][:4]
        if len(labels) < 2:
            words = [wd for wd in str(self.spec.get("text") or "").split() if len(wd) > 2][:3]
            labels = (labels + words + ["Input", "Output"])[:max(2, len(labels) + len(words))][:4]
            if len(labels) < 2:
                labels = ["Input", "Output"]
        self.labels = labels
        given = self.spec.get("icons") or []
        self.icon_names = [given[k] if k < len(given) and given[k] in icons.NAMES else icons.pick_icon(lbl, k + 1)
                           for k, lbl in enumerate(labels)]
        n = len(labels)
        self.card = self.px(250 if self.portrait else 220)
        if self.portrait:
            top, bottom = int(self.h * 0.25), int(self.h * 0.48)
            ys = [top + (bottom - top) * (k / max(1, n - 1)) for k in range(n)] if n > 1 else [self.cy]
            xs = [self.w * (0.30 if k % 2 == 0 else 0.70) for k in range(n)]
        else:
            left, right = self.margin + self.px(180), self.w - self.margin - self.px(180)
            xs = [left + (right - left) * (k / max(1, n - 1)) for k in range(n)]
            ys = [self.cy + self.px(30) + (self.px(-90) if k % 2 else self.px(90)) for k in range(n)]
        self.centers = list(zip(xs, ys))
        self.path = []
        for k in range(n - 1):
            (ax, ay), (bx, by) = self.centers[k], self.centers[k + 1]
            if self.portrait:
                seg = _bezier((ax, ay), (ax, ay + (by - ay) * 0.6), (bx, by - (by - ay) * 0.6), (bx, by))
            else:
                seg = _bezier((ax, ay), (ax + (bx - ax) * 0.55, ay), (bx - (bx - ax) * 0.55, by), (bx, by))
            self.path.extend(seg if not self.path else seg[1:])
        self.cum = [0.0]
        for (x0, y0), (x1, y1) in zip(self.path, self.path[1:]):
            self.cum.append(self.cum[-1] + math.hypot(x1 - x0, y1 - y0))
        self.length = max(1.0, self.cum[-1])
        self.card_at = []
        for c in self.centers:
            best = min(range(len(self.path)), key=lambda i: math.hypot(self.path[i][0] - c[0], self.path[i][1] - c[1]))
            self.card_at.append(self.cum[best] / self.length)
        self.cards = [(self._card(k, False), self._card(k, True)) for k in range(n)]
        self.lf = _fit_font("headline", max(labels, key=len), self.card + self.px(120), 44, 26, 600, self.s)
        self.title = str(self.spec.get("title") or "")
        self.t_in = 0.25
        self.step = min(0.45, max(0.25, self.span * 0.5 / n))

    def _card(self, k, hot):
        c = self.card
        pad = self.px(14)
        img = Image.new("RGBA", (c + 2 * pad, c + 2 * pad), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        r = self.px(30)
        if hot:
            d.rounded_rectangle([0, 0, c + 2 * pad - 1, c + 2 * pad - 1], radius=r + pad, fill=_rgba("lime", 46))
        d.rounded_rectangle([pad, pad, pad + c - 1, pad + c - 1], radius=r, fill=_rgba("panel", 238),
                            outline=_rgba("lime" if hot else "border"), width=self.px(3))
        icons.draw_icon(d, self.icon_names[k], (pad + c * 0.24, pad + c * 0.24, pad + c * 0.76, pad + c * 0.76),
                        _rgba("lime" if hot else "cyan"), width=max(2, self.px(7)))
        return img

    def _point(self, q):
        target = q * self.length
        lo, hi = 0, len(self.cum) - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if self.cum[mid] <= target:
                lo = mid
            else:
                hi = mid
        seg = max(1e-6, self.cum[hi] - self.cum[lo])
        u = (target - self.cum[lo]) / seg
        (x0, y0), (x1, y1) = self.path[lo], self.path[hi]
        return x0 + (x1 - x0) * u, y0 + (y1 - y0) * u

    def frame(self, local_t):
        t = local_t
        img = self.bg_frame(t)
        self.draw_tag(img, self.title or "Flow", color="cyan")
        d = ImageDraw.Draw(img, "RGBA")
        n = len(self.labels)
        # the path draws itself (dashed), card by card
        drawn = ease_in_out((t - self.t_in) / max(0.3, self.step * n))
        upto = drawn * self.length
        dash, gapl = self.px(22), self.px(14)
        pos = 0.0
        while pos < upto:
            a, b = self._point(pos / self.length), self._point(min(upto, pos + dash) / self.length)
            d.line([a, b], fill=_rgba("muted", 220), width=self.px(5))
            pos += dash + gapl
        # token travels continuously once the path is down; trailing glow
        token_q = None
        if drawn >= 1:
            token_q = ((t - self.t_in - self.step * n) * 0.32) % 1.0
            for k in range(10, -1, -1):
                q = token_q - k * 0.012
                if q < 0:
                    continue
                x, y = self._point(q)
                rad = self.px(16) * (1 - k / 13)
                d.ellipse([x - rad, y - rad, x + rad, y + rad], fill=_rgba("lime", int(255 * (1 - k / 11))))
        for k in range(n):
            p = ease_out((t - self.t_in - k * self.step) / 0.4)
            if p <= 0:
                continue
            hot = token_q is not None and abs(token_q - self.card_at[k]) < 0.07
            card = self.cards[k][1 if hot else 0]
            cx, cy = self.centers[k]
            cy += self.px(8) * math.sin(t * 1.4 + k)
            if self.portrait:
                cx += (-1 if k % 2 == 0 else 1) * self.px(200) * (1 - p)
            else:
                cy += self.px(120) * (1 - p)
            img.alpha_composite(_alpha(card, p), (int(cx - card.width / 2), int(cy - card.height / 2)))
            lp = ease_out((t - self.t_in - k * self.step - 0.2) / 0.35)
            if lp > 0:
                ImageDraw.Draw(img, "RGBA").text((cx, cy + card.height / 2 + self.px(14)), self.labels[k],
                                                 font=self.lf, fill=_rgba("body", int(255 * lp)), anchor="mt")
        return img.convert("RGB")


# ------------------------------------------------------------------ factory

SOURCES = {
    "code_panel": CodePanelSource,
    "diagram": DiagramSource,
    "chapter_card": ChapterCardSource,
    "kinetic_text": KineticTextSource,
    "count_up": CountUpSource,
    "icon_flow": IconFlowSource,
}

TEXT_DOMINANT = ("kinetic_text", "chapter_card")


def is_block(asset):
    return bool(asset) and asset.get("kind") == "block"


def make_source(asset, size, span):
    return SOURCES[asset["block"]](asset.get("spec") or {}, size, span)


def block_asset(shot, preview_path=None, size=(1920, 1080), span=4.6):
    """Asset dict for a rendered block shot; optionally writes a preview PNG (manifest / review)."""
    kind = shot["kind"]
    asset = {"kind": "block", "block": kind, "spec": dict(shot), "provider": PROVIDER, "license": "own (DYPOL)",
             "reusable": False, "source_url": "", "asset_id": f"{kind}"}
    if preview_path is not None:
        src = make_source(asset, size, span)
        src.frame(span * 0.85).save(preview_path)
        asset["path"] = str(preview_path)
    return asset


def render_clip(kind, spec, out_path, seconds=2.0, size=None, fps=None):
    """Render one block to an H.264 mp4 (tests / previews): Pillow frames piped into ffmpeg."""
    import subprocess
    size = tuple(size or style.look()["size"])
    fps = int(fps or style.look()["fps"])
    src = SOURCES[kind](dict(spec, kind=kind), size, seconds)
    w, h = size
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
           "-pix_fmt", "yuv420p", str(out_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for f in range(int(round(seconds * fps))):
            proc.stdin.write(src.frame(f / fps).tobytes())
    finally:
        proc.stdin.close()
        code = proc.wait()
        src.close()
    if code != 0:
        raise RuntimeError(f"ffmpeg failed ({code}) rendering {kind}")
    return out_path
