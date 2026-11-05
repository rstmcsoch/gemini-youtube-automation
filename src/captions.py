"""Caption files (SRT) for YouTube, built from the narration word timings the
pipeline already estimates for burned-in captions and bullet reveals."""

from pathlib import Path


def _ts(seconds):
    ms = max(0, int(round(float(seconds) * 1000)))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def cue_groups(word_times, max_words=10, max_chars=42, max_span=5.0):
    """Group (word, start, end) into readable subtitle cues; break after sentence ends."""
    cues, current = [], []
    for item in word_times or []:
        current.append(item)
        text = " ".join(w for w, _, _ in current)
        word = str(item[0])
        if (len(current) >= max_words or len(text) >= max_chars
                or word.endswith((".", "!", "?")) or item[2] - current[0][1] >= max_span):
            cues.append(current)
            current = []
    if current:
        cues.append(current)
    return cues


def words_to_srt(word_times, offset=0.0):
    """SRT text for word timings; ``offset`` shifts every cue (seconds)."""
    blocks = []
    cues = cue_groups([(str(w), float(a) + offset, float(b) + offset) for w, a, b in (word_times or [])
                       if str(w).strip()])
    for n, cue in enumerate(cues, 1):
        start, end = cue[0][1], max(cue[-1][2], cue[0][1] + 0.3)
        if n < len(cues):
            end = min(end, cues[n][0][1])  # never overlap the next cue
        end = max(end, start + 0.1)
        text = " ".join(w for w, _, _ in cue).replace("#", "").strip()
        blocks.append(f"{n}\n{_ts(start)} --> {_ts(end)}\n{text}\n")
    return "\n".join(blocks)


def segments_to_words(segments):
    """Flatten [(slide_start_seconds, [(word, start, end), ...]), ...] onto one timeline."""
    out = []
    for start, words in segments:
        out.extend((w, float(a) + float(start), float(b) + float(start)) for w, a, b in words or [])
    return out


def write_srt(word_times, path):
    """Write an SRT file; returns its path, or None when there is nothing to caption."""
    text = words_to_srt(word_times)
    if not text.strip():
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# Burned-in word-level karaoke captions (ASS, rendered by libass in the encoder)
# ---------------------------------------------------------------------------

_PUNCT_BREAK = (".", "!", "?", ",", ";", ":")


def _ass_ts(seconds):
    cs = max(0, int(round(float(seconds) * 100)))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def _ass_color(rgb, alpha=0):
    r, g, b = rgb
    return f"&H{alpha:02X}{b:02X}{g:02X}{r:02X}"


def _clean_word(word):
    return str(word).replace("\\", "").replace("{", "").replace("}", "").replace("#", "").strip()


def karaoke_groups(word_times, max_words=6, max_chars=40, min_words=1, max_gap=0.8):
    """Group words into caption lines: break after punctuation, at max_words/max_chars or a long pause.

    A group never holds fewer than ``min_words`` unless the text runs out (Shorts: 2-4 words)."""
    items = [(_clean_word(w), float(a), float(b)) for w, a, b in (word_times or []) if _clean_word(w)]
    groups, cur = [], []
    for item in items:
        if cur and (item[1] - cur[-1][2] > max_gap
                    or len(" ".join(w for w, _, _ in cur + [item])) > max_chars):
            groups.append(cur)
            cur = []
        cur.append(item)
        if len(cur) >= max_words or (len(cur) >= min_words and item[0].endswith(_PUNCT_BREAK)):
            groups.append(cur)
            cur = []
    if cur:
        if groups and len(cur) < min_words and len(groups[-1]) + len(cur) <= max_words:
            groups[-1].extend(cur)
        else:
            groups.append(cur)
    # No orphan word: a lone word after a full line that did not end a phrase takes the
    # previous line's last two words with it ("... matters for" / "you." -> "... matters" / "for you.").
    for k in range(1, len(groups)):
        prev, this = groups[k - 1], groups[k]
        if (len(this) == 1 and len(prev) >= max(3, min_words + 2) and not prev[-1][0].endswith(_PUNCT_BREAK)
                and this[0][1] - prev[-1][2] <= max_gap):
            groups[k - 1], groups[k] = prev[:-2], prev[-2:] + this
    return groups


# Caption placement (fraction of frame height, line centre).
#  * long 16:9: 0.78 - the lower-third panel is lifted above the caption band (panel_bottom_frac),
#    so the two never overlap.
#  * Short 9:16: 0.75 - inside the YouTube Shorts safe zone, clear of the bottom ~20 % (title, buttons).
LONG_CENTER_Y = 0.78
SHORT_CENTER_Y = 0.75
SHORT_UI_BOTTOM = 0.80          # nothing of ours below this on a Short
LINE_HEIGHT = 1.25              # caption line box / font size (incl. outline)


def caption_settings(mode):
    from src import style
    cfg = style.look().get("captions") or {}
    base = {"long": {"max_words": 6, "max_chars": 40, "size_px": 56, "center_y": LONG_CENTER_Y, "min_words": 1},
            "short": {"max_words": 4, "max_chars": 22, "size_px": 92, "center_y": SHORT_CENTER_Y, "min_words": 2}}[mode]
    base.update(cfg.get(mode) or {})
    base["family"] = cfg.get("family", "Space Grotesk")
    return base


def caption_band(mode, size):
    """(top, bottom) pixel rows the caption line occupies."""
    w, h = size
    cfg = caption_settings(mode)
    scale = (h / 1080.0) if mode == "long" else (w / 1080.0)
    half = max(12, int(round(cfg["size_px"] * scale))) * LINE_HEIGHT / 2
    cy = h * float(cfg["center_y"])
    return int(cy - half), int(cy + half)


def panel_bottom_frac(size, gap=0.02):
    """`bottom` for explainer.render_lower_third when captions are burned in: the panel's lower edge
    sits `gap` (fraction of height) above the long caption line."""
    top, _ = caption_band("long", size)
    return round(1.0 - top / float(size[1]) + gap, 4)


# ---------------------------------------------------------------- key-word highlights

_SENT_END = (".", "!", "?")
_NUM_SUFFIX = {"percent", "%", "x", "times", "million", "billion", "thousand", "hundred", "k"}
_STOP = set("""a an the and or but if then than that this these those there their they them we you your our
it its is are was were be been being have has had do does did of to in on at by for from with as into onto
about over under after before between through during without within also just very really more most much
many some any every each other such what which who whom whose when where why how not no can could would
should will shall may might must here so too again still even only own same both few one""".split())


def _bare(word):
    return str(word).strip("(\"'“”‘’[]").rstrip(".,;:!?)\"'”’").strip()


def _is_numeric(word):
    return any(ch.isdigit() for ch in str(word))


def number_flags(words):
    """[bool] per word: numbers, percentages and years, digits or spelled out (+ a unit word after)."""
    from src import numwords
    words = [str(w) for w in words]
    flags = [False] * len(words)
    try:
        norm = numwords.normalize(words)
    except Exception:  # noqa: BLE001 - highlighting is cosmetic
        norm = list(words)
    i = 0
    while i < len(words):
        if _is_numeric(words[i]):
            flags[i] = True
            j = i + 1
        elif _is_numeric(norm[i]):
            j = i + 1
            while j < len(words) and norm[j] == "":
                j += 1
            nxt = _bare(words[j]).lower() if j < len(words) else ""
            # A lone spelled "one"/"zero" is usually not a figure ("one of the ...").
            if _bare(norm[i]) not in ("0", "1") or nxt in _NUM_SUFFIX:
                for k in range(i, j):
                    flags[k] = True
        else:
            i += 1
            continue
        if j < len(words) and _bare(words[j]).lower() in _NUM_SUFFIX:
            flags[j] = True
            j += 1
        i = j
    return flags


def key_word_flags(words):
    """[bool] per word: highlight numbers / percentages / years (digits or spelled out) and at most
    ONE key term per sentence (acronym > mid-sentence proper noun > long content word)."""
    words = [str(w) for w in words]
    flags = number_flags(words)
    # One key term per sentence.
    start = 0
    for end in range(len(words)):
        if words[end].rstrip(")\"'”’").endswith(_SENT_END) or end == len(words) - 1:
            best, best_score = None, 0.0
            for k in range(start, end + 1):
                if flags[k]:
                    continue
                bare = _bare(words[k])
                low = bare.lower()
                if not bare or low in _STOP or not any(c.isalpha() for c in bare):
                    continue
                letters = [c for c in bare if c.isalpha()]
                if len(letters) >= 2 and all(c.isupper() for c in letters) and len(bare) <= 8:
                    score = 3.0
                elif bare[0].isupper() and k != start:
                    score = 2.0
                elif len(bare) >= 7:
                    score = 1.0 + len(bare) / 100.0
                else:
                    continue
                if score > best_score:
                    best, best_score = k, score
            if best is not None:
                flags[best] = True
            start = end + 1
    return flags


def words_to_ass(word_times, size, mode="long"):
    """ASS script: one event per caption line. Words are plain white with the outline; only the
    key words (numbers / percentages / years, and at most one key term per sentence) are lime.

    Long video: one line of up to 6 words at ~78 % height, above the lower-third band.
    Short: 2-4 words at ~75 % height (Shorts safe zone)."""
    from src import style
    w, h = size
    cfg = caption_settings(mode)
    scale = (h / 1080.0) if mode == "long" else (w / 1080.0)
    fs = max(12, int(round(cfg["size_px"] * scale)))
    white = _ass_color((255, 255, 255))
    lime = _ass_color(style.rgb("lime"))
    outline = _ass_color(style.rgb("background"))
    shadow = _ass_color(style.rgb("background"), 0x60)
    out_w = max(2, int(round(fs * 0.075)))
    margin = int(w * 0.06)
    head = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {w}", f"PlayResY: {h}", "WrapStyle: 0",
        "ScaledBorderAndShadow: yes", "", "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, "
        "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding",
        f"Style: Cap,{cfg['family']},{fs},{white},{lime},{outline},{shadow},-1,0,0,0,100,100,0,0,1,{out_w},"
        f"{max(1, out_w // 2)},5,{margin},{margin},0,1",
        "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    pos = f"{{\\an5\\pos({w // 2},{int(h * float(cfg['center_y']))})}}"
    groups = karaoke_groups(word_times, cfg["max_words"], cfg["max_chars"], cfg.get("min_words", 1))
    flags = key_word_flags([wd for g in groups for wd, _, _ in g])
    events, n = [], 0
    for g, group in enumerate(groups):
        end_g = group[-1][2] + 0.35
        if g + 1 < len(groups):
            nxt = groups[g + 1][0][1]
            end_g = nxt if nxt - group[-1][2] < 0.6 else min(end_g, nxt)
        parts = []
        for word, _, _ in group:
            parts.append(f"{{\\c{lime}}}{word}{{\\c{white}}}" if flags[n] else word)
            n += 1
        start = group[0][1]
        if end_g - start < 0.04:
            continue
        events.append(f"Dialogue: 0,{_ass_ts(start)},{_ass_ts(end_g)},Cap,,0,0,0,,{pos}{' '.join(parts)}")
    return "\n".join(head + events) + "\n"


def write_ass(word_times, path, size, mode="long"):
    """Write the burned-in caption script; None when there is nothing to caption."""
    if not any(_clean_word(w) for w, _, _ in (word_times or [])):
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(words_to_ass(word_times, size, mode), encoding="utf-8")
    return str(path)
