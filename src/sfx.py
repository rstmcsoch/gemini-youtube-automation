"""Sound design for Engine A: a small CC0 Kenney set (sfx/, see sfx/LICENSES.md).

* A whoosh or pop on block changes, varied, ~4-8 per minute (never closer than 60/8 s).
* Soft keyboard clicks while a code_panel types.
* Everything is placed on one track whose level sits ~12 dB under the narration
  (clicks lower still); the mixer then ducks the music and loudnorms to -14 LUFS.
Planning is pure (testable); only `build_track` touches audio.
"""
import random
from pathlib import Path

from src import style

SFX_DIR = style.ROOT / "sfx"

# peak_s: where the clip's energy peaks, so the peak lands on the cut.
CLIPS = {
    "whoosh_01": {"category": "whoosh", "peak_s": 0.52},
    "whoosh_02": {"category": "whoosh", "peak_s": 0.0},
    "whoosh_03": {"category": "whoosh", "peak_s": 0.52},
    "whoosh_04": {"category": "whoosh", "peak_s": 0.05},
    "pop_01": {"category": "pop", "peak_s": 0.0},
    "pop_02": {"category": "pop", "peak_s": 0.0},
    "pop_03": {"category": "pop", "peak_s": 0.0},
    "click_01": {"category": "click", "peak_s": 0.0},
    "click_02": {"category": "click", "peak_s": 0.0},
    "click_03": {"category": "click", "peak_s": 0.0},
    "click_04": {"category": "click", "peak_s": 0.0},
}

WHOOSH_KINDS = {"chapter_card", "code_panel", "diagram", "stock_or_ai_still"}
POP_KINDS = {"kinetic_text", "count_up"}


def audio_rules():
    a = style.look().get("audio") or {}
    rate = a.get("sfx_per_min") or [4, 8]
    return {"under_db": float(a.get("sfx_under_voice_db", 12)), "clicks_under_db": float(a.get("clicks_under_voice_db", 22)),
            "min_per_min": float(rate[0]), "max_per_min": float(rate[1])}


def clip_path(clip_id):
    return SFX_DIR / f"{clip_id}.ogg"


def available():
    return all(clip_path(c).exists() for c in CLIPS)


def _by_category(cat):
    return [c for c, v in CLIPS.items() if v["category"] == cat]


def block_changes(timeline, kinds, fixed):
    """[(cut_time, new_kind)] where a new visual unit starts (a different block or kind)."""
    out, prev = [], None
    for g, shot in enumerate(timeline["shots"]):
        unit = (fixed.get(g, {}).get("block_id") or f"shot{g}", kinds[g] if g < len(kinds) else None)
        if g and unit[0] != prev[0]:
            out.append((shot["start"], unit[1]))
        prev = unit
    return out


def plan_transitions(changes, total, seed=0):
    """Pick cuts for whoosh/pop hits: ~4-8 per minute, varied clips, never the same clip twice running."""
    r = audio_rules()
    min_gap = 60.0 / r["max_per_min"]
    target_gap = 60.0 / ((r["min_per_min"] + r["max_per_min"]) / 2)
    rng = random.Random(seed)
    events, last, last_clip = [], -1e9, None
    pools = {"whoosh": _by_category("whoosh"), "pop": _by_category("pop")}
    turn = {"whoosh": rng.randrange(4), "pop": rng.randrange(3)}
    for k, (t, kind) in enumerate(changes):
        if t < 0.3 or t > total - 0.3 or t - last < min_gap:
            continue
        priority = kind == "chapter_card"
        nxt = next((t2 for t2, _ in changes[k + 1:] if t2 > t), float("inf"))
        # Hit this cut if it is a chapter, the target spacing has passed, or waiting would drop below 4/min.
        if not (priority or t - last >= target_gap or nxt - last > 60.0 / r["min_per_min"]):
            continue
        cat = "pop" if kind in POP_KINDS else "whoosh"
        pool = pools[cat]
        clip = pool[turn[cat] % len(pool)]
        turn[cat] += 1
        if clip == last_clip:
            clip = pool[turn[cat] % len(pool)]
            turn[cat] += 1
        start = max(0.0, t - CLIPS[clip]["peak_s"])
        events.append({"t": round(start, 3), "clip": clip, "cut": round(t, 3), "gain_db": round(rng.uniform(-1.5, 0.5), 2)})
        last, last_clip = t, clip
    return events


def code_typing_windows(timeline, fixed):
    """[(start, end)] absolute typing windows of each code_panel block (from the renderer's own timing)."""
    from src import blocks
    from src import explainer as ex
    windows = ex.shot_windows(timeline)
    seen, out = set(), []
    for g in sorted(fixed):
        asset = fixed[g]
        if asset.get("block") != "code_panel" or asset.get("block_id") in seen:
            continue
        seen.add(asset.get("block_id"))
        span = float(asset.get("block_span") or (windows[g][1] - windows[g][0]))
        start = windows[g][0] - float(asset.get("block_offset") or 0.0)
        try:
            src = blocks.CodePanelSource(asset.get("spec") or {}, (320, 180), span)
            t0, t1 = src.t0, min(src.t_done, span)
        except Exception:  # noqa: BLE001 - no clicks rather than no video
            continue
        if t1 > t0:
            out.append((round(start + t0, 3), round(start + t1, 3)))
    return out


def plan_clicks(windows, seed=0, rate=9.0):
    """Keyboard clicks at ~rate per second with human jitter inside each typing window."""
    rng = random.Random(seed + 7)
    clicks = _by_category("click")
    events = []
    for a, b in windows:
        t = a + rng.uniform(0.0, 0.05)
        while t < b:
            events.append({"t": round(t, 3), "clip": clicks[rng.randrange(len(clicks))], "click": True,
                           "gain_db": round(rng.uniform(-3.0, 0.0), 2)})
            t += rng.uniform(0.6, 1.5) / rate
    return events


def plan(timeline, kinds, fixed, seed=0):
    events = plan_transitions(block_changes(timeline, kinds, fixed), timeline["total"], seed)
    events += plan_clicks(code_typing_windows(timeline, fixed), seed)
    return sorted(events, key=lambda e: e["t"])


def per_minute(events, total):
    hits = [e for e in events if not e.get("click")]
    return round(len(hits) / max(1e-6, total / 60.0), 2)


def build_track(events, total, out_wav, voice_wav):
    """Render the events to one 48 kHz stereo wav, levelled against the narration; None if nothing to add."""
    if not events or not available():
        return None
    from pydub import AudioSegment
    r = audio_rules()
    voice = AudioSegment.from_file(str(voice_wav))
    voice_db = voice.dBFS if voice.dBFS != float("-inf") else -20.0
    track = AudioSegment.silent(duration=int(total * 1000) + 50, frame_rate=48000).set_channels(2)
    cache = {}
    for e in events:
        if e["clip"] not in cache:
            seg = AudioSegment.from_file(str(clip_path(e["clip"]))).set_frame_rate(48000).set_channels(2)
            under = r["clicks_under_db"] if CLIPS[e["clip"]]["category"] == "click" else r["under_db"]
            level = seg.dBFS if seg.dBFS != float("-inf") else -20.0
            cache[e["clip"]] = seg.apply_gain(voice_db - under - level)
        seg = cache[e["clip"]].apply_gain(e.get("gain_db", 0.0))
        track = track.overlay(seg, position=int(e["t"] * 1000))
    track = track[: int(total * 1000)]
    track.export(str(out_wav), format="wav")
    return Path(out_wav)
