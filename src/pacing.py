"""Variable shot timing for Engine A (no more fixed ~4.6 s shots).

Cuts land on narration clause boundaries (sentence ends first, then commas /
semicolons / dashes, then any word gap) so a long beat is split where the
speaker breathes. Every shot is screen_change_min_s..screen_change_max_s long
(2-4 s); a slide that opens on a chapter_card gets a short 1.0-1.5 s first shot.
Pure functions: no rendering, fully unit-testable.
"""
from src import style

_SENTENCE = (".", "!", "?")
_CLAUSE = (",", ";", ":", "—", "–", ")")


def rules():
    p = style.long_rules()["pacing"]
    card = p.get("chapter_card_s") or [1.0, 1.5]
    return {"lo": float(p["screen_change_min_s"]), "hi": float(p["screen_change_max_s"]),
            "target": float(p.get("target_shot_s", 3.0)), "max_static": float(p["max_static_s"]),
            "card": (float(card[0]), float(card[1]))}


CUT_LEAD = 0.04   # s: a cut may lead the next word's onset by up to one frame-ish


def cut_candidates(words, duration):
    """[(time, strength)] between words: 2 = sentence end, 1 = clause break, 0 = plain word gap.

    The cut lands on the next word's start (CUT_LEAD earlier when the gap allows), so a new
    picture arrives with the new phrase; with Kokoro word timings that start is exact."""
    out = []
    words = [(str(w), float(a), float(b)) for w, a, b in (words or [])]
    for (w, a, b), (_, na, _) in zip(words, words[1:]):
        t = max(min(b, na), na - CUT_LEAD)
        if not 0.0 < t < duration:
            continue
        strength = 2 if w.endswith(_SENTENCE) else (1 if w.endswith(_CLAUSE) or w in ("-", "—") else 0)
        out.append((round(t, 4), strength))
    return out


def _pick(cands, lo_t, hi_t, want):
    """Best candidate in [lo_t, hi_t]: strongest boundary, then closest to `want`."""
    best = None
    for t, strength in cands:
        if lo_t - 1e-9 <= t <= hi_t + 1e-9:
            key = (-strength, abs(t - want))
            if best is None or key < best[0]:
                best = (key, t)
    return best[1] if best else None


def plan_variable_shots(duration, words=(), first=None, lo=None, hi=None, target=None):
    """Local (start, end) shots for one slide.

    first: (min, max) length of a short opening shot (chapter card), or None.
    Every other shot is lo..hi seconds (a slide shorter than lo is one shot)."""
    r = rules()
    lo = r["lo"] if lo is None else lo
    hi = r["hi"] if hi is None else hi
    target = r["target"] if target is None else target
    duration = max(0.1, float(duration))
    cands = cut_candidates(words, duration)
    cuts, t = [], 0.0
    if first and duration > first[1] + lo * 0.5:
        c = _pick(cands, first[0], first[1], (first[0] + first[1]) / 2)
        t = c if c is not None else (first[0] + first[1]) / 2
        cuts.append(t)
    while duration - t > hi + 1e-9:
        lo_t, hi_t = t + lo, min(t + hi, duration - lo)
        want = min(t + target, hi_t)
        # A sentence end anywhere in the window beats a comma near the target; a comma beats a word gap.
        c = _pick(cands, lo_t, hi_t, want)
        if c is None:
            c = want
        cuts.append(c)
        t = c
    edges = [0.0] + cuts + [duration]
    return [(round(a, 4), round(b, 4)) for a, b in zip(edges, edges[1:]) if b - a > 1e-6]


def build_timeline(slide_durations, slide_words=None, chapter_flags=None, lo=None, hi=None, target=None):
    """Global timeline (same shape as storyboard.build_timeline) with variable, clause-aligned shots.

    slide_words[i]: (word, start, end) relative to slide i's start; chapter_flags[i]: the slide
    opens on a chapter_card (its first shot is a 1.0-1.5 s stinger)."""
    card = rules()["card"]
    shots, slides, t = [], [], 0.0
    for index, duration in enumerate(slide_durations):
        words = (slide_words or [None] * len(slide_durations))[index] or []
        flag = bool((chapter_flags or [False] * len(slide_durations))[index])
        start = t
        local = plan_variable_shots(duration, words, first=card if flag else None, lo=lo, hi=hi, target=target)
        ids = []
        for j, (a, b) in enumerate(local):
            shots.append({"slide": index, "shot": j, "start": round(start + a, 4), "end": round(start + b, 4)})
            ids.append(len(shots) - 1)
        if shots:
            shots[-1]["end"] = round(start + float(duration), 4)
        slides.append({"index": index, "start": round(start, 4), "end": round(start + float(duration), 4),
                       "shots": ids})
        t = start + float(duration)
    return {"total": round(t, 4), "slides": slides, "shots": shots}


def stats(timeline):
    lengths = [s["end"] - s["start"] for s in timeline.get("shots", [])]
    if not lengths:
        return {"shots": 0}
    return {"shots": len(lengths), "min_s": round(min(lengths), 3), "max_s": round(max(lengths), 3),
            "mean_s": round(sum(lengths) / len(lengths), 3)}
