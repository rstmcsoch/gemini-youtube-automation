"""CPU block renderers for Engine A (DYPOL look): Pillow frames, eased motion.

Each source follows the explainer source protocol: `frame(local_t) -> RGB Image`
(a pure function of local time, so chunked parallel renders stay identical to a
single pass) and `close()`. Built by `make_source` from an asset dict:

    {"kind": "block", "block": "<kind>", "spec": {...shot fields...}}
"""
import math
import re

from PIL import Image, ImageDraw

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
        self.s = self.h / 1080.0
        self.margin = int(self.w * style.long_rules()["text"]["safe_margin"])
        self.bg = Image.new("RGBA", self.size, _rgba("background"))

    def px(self, v):
        return max(1, int(round(v * self.s)))

    def font(self, role, size, weight=400):
        return style.font(role, self.px(size), weight)

    def draw_tag(self, img, text, y=None, color="lime"):
        """Left-edge lime bar + mono uppercase label (DYPOL label system)."""
        d = ImageDraw.Draw(img)
        x, y = self.margin, self.px(72) if y is None else y
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
        ph = min(ph, self.h - self.px(260))
        pw = int(self.w * 0.80)
        self.box = ((self.w - pw) // 2, (self.h - ph) // 2 + self.px(30), (self.w + pw) // 2,
                    (self.h + ph) // 2 + self.px(30))
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
        img = self.bg.copy()
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
        img = self.base.copy()
        a = ease_out(t / 0.35)
        if a < 1:
            fade = Image.new("RGBA", self.size, _rgba("background", int(255 * (1 - a))))
            img = Image.alpha_composite(img, fade)
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
    """2-5 nodes connected by arrows that build step by step; a pulse then travels the chain."""

    def __init__(self, spec, size, span):
        super().__init__(spec, size, span)
        nodes = [str(n) for n in (self.spec.get("nodes") or []) if str(n).strip()]
        cfg = style.guide()["blocks"]["diagram"]
        nodes = nodes[: cfg["max_nodes"]]
        while len(nodes) < cfg["min_nodes"]:
            nodes.append("Result" if nodes else "Input")
        self.nodes = nodes
        n = len(nodes)
        gap = self.px(110)
        avail = self.w - 2 * self.margin - gap * (n - 1)
        bw = min(self.px(380), avail // n)
        bh = self.px(170)
        total = bw * n + gap * (n - 1)
        x = (self.w - total) // 2
        y = (self.h - bh) // 2 + self.px(40)
        self.boxes = [(x + k * (bw + gap), y, x + k * (bw + gap) + bw, y + bh) for k in range(n)]
        self.base = self._base()
        self.node_imgs = [(self._node(k, False), self._node(k, True)) for k in range(n)]
        steps = 2 * n - 1
        self.build = min(max(1.2, self.span * 0.62), 0.55 * steps)
        self.step = self.build / steps
        self.t0 = 0.25

    def _base(self):
        img = self.bg.copy()
        grid = style.look()["grid"]
        layer = Image.new("RGBA", self.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        a = int(255 * grid["opacity"])
        sp = self.px(grid["spacing_px"])
        for gx in range(0, self.w, sp):
            d.line([(gx, 0), (gx, self.h)], fill=_rgba("body", a), width=1)
        for gy in range(0, self.h, sp):
            d.line([(0, gy), (self.w, gy)], fill=_rgba("body", a), width=1)
        img = Image.alpha_composite(img, layer)
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
        d.rounded_rectangle([pad, pad, pad + w - 1, pad + h - 1], radius=r, fill=_rgba("panel"),
                            outline=_rgba("lime" if active else "border"), width=self.px(3) if active else self.px(2))
        d.text((pad + self.px(20), pad + self.px(22)), f"{k + 1:02d}", font=self.font("mono", 24, 500),
               fill=_rgba("lime" if active else "cyan"), anchor="lt")
        label = self.nodes[k]
        f = _fit_font("headline", label, w - self.px(36), 46, 26, 600, self.s)
        words, lines = label.split(), [""]
        for word in words:
            trial = (lines[-1] + " " + word).strip()
            if f.getlength(trial) <= w - self.px(36) or not lines[-1]:
                lines[-1] = trial
            else:
                lines.append(word)
        lines = lines[:2]
        lh = f.size * 1.15
        cy = pad + h / 2 + self.px(14) - lh * (len(lines) - 1) / 2
        for i, line in enumerate(lines):
            d.text((pad + w / 2, cy + i * lh), line, font=f, fill=_rgba("body"), anchor="mm")
        return img

    def _arrow(self, img, k, p, color="border"):
        if p <= 0:
            return
        d = ImageDraw.Draw(img)
        x0 = self.boxes[k][2] + self.px(14)
        x1 = self.boxes[k + 1][0] - self.px(14)
        y = (self.boxes[k][1] + self.boxes[k][3]) / 2
        xe = x0 + (x1 - x0) * p
        d.line([(x0, y), (xe, y)], fill=_rgba(color), width=self.px(4))
        if p > 0.6:
            hs = self.px(14)
            d.polygon([(xe, y), (xe - hs * 1.3, y - hs), (xe - hs * 1.3, y + hs)], fill=_rgba(color))

    def frame(self, local_t):
        t = local_t - self.t0
        img = self.base.copy()
        n = len(self.nodes)
        built = t / self.step if self.step else 99
        active = min(n - 1, max(0, int(built // 2)))
        complete = t >= self.build
        for k in range(n - 1):
            self._arrow(img, k, ease_in_out((t - (2 * k + 1) * self.step) / (self.step * 0.9)),
                        color="cyan" if complete or k < active else "muted")
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
            x, y = self.boxes[k][0] - pad, self.boxes[k][1] - pad + int(self.px(24) * (1 - p))
            img.alpha_composite(_alpha(node, p), (x, y))
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
        img = self.base.copy()
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
        top = self.h / 2 - lh * len(lines) / 2
        self.pos = {}
        for li, idxs in enumerate(lines):
            width = sum(self.f.getlength(self.words[i]) for i in idxs) + sp * (len(idxs) - 1)
            x = (self.w - width) / 2
            for i in idxs:
                self.pos[i] = (x, top + li * lh, self.f.getlength(self.words[i]), lh)
                x += self.f.getlength(self.words[i]) + sp
        self.imgs = [ChapterCardSource._text_img(wd, self.f, "body") for wd in self.words]
        self.tag = self.spec.get("label") or ""

    def frame(self, local_t):
        t = local_t
        img = self.bg.copy()
        if self.tag:
            self.draw_tag(img, self.tag)
        drift = int(self.px(14) * ease_in_out(t / self.span))
        all_in = 0.1 + 0.16 * len(self.words) + 0.2
        hp = ease_in_out((t - all_in) / 0.3)
        if hp > 0:
            x, y, wl, lh = self.pos[self.key]
            ImageDraw.Draw(img).rectangle([x - self.px(14), y + lh * 0.12 - drift,
                                           x - self.px(14) + (wl + self.px(28)) * hp, y + lh * 0.95 - drift],
                                          fill=_rgba("lime", 255))
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

    def frame(self, local_t):
        t = local_t
        img = self.bg.copy()
        self.draw_tag(img, self.spec.get("title") or "By the numbers")
        d = ImageDraw.Draw(img)
        drift = int(self.px(12) * ease_in_out(t / self.span))
        p = ease_out((t - 0.15) / 0.9)
        cur = self.value * p
        num = format_number(cur, self.dec)
        x = (self.w - self.final_w) / 2
        y = self.h / 2 - self.px(70) - drift
        d.text((x, y), self.prefix, font=self.f, fill=_rgba("body"), anchor="ls")
        x += self.f.getlength(self.prefix)
        d.text((x, y), num, font=self.f, fill=_rgba("lime"), anchor="ls")
        x += self.f.getlength(num)
        d.text((x, y), self.suffix, font=self.f, fill=_rgba("body"), anchor="ls")
        if self.label:
            lp = ease_out((t - 0.5) / 0.35)
            ly = self.h / 2 + self.px(60) - drift + int(self.px(20) * (1 - lp))
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
            ImageDraw.Draw(layer).text((self.margin, self.h - self.px(80)), f"Source: {self.source}", font=self.sf,
                                       fill=_rgba("muted"), anchor="ls")
            img.alpha_composite(_alpha(layer, sp))
        return img.convert("RGB")


# ------------------------------------------------------------------ factory

SOURCES = {
    "code_panel": CodePanelSource,
    "diagram": DiagramSource,
    "chapter_card": ChapterCardSource,
    "kinetic_text": KineticTextSource,
    "count_up": CountUpSource,
}


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
