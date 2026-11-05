"""Sound design for Engine A: soft, calm transitions generated in code (numpy).

No third-party audio files: every sound is synthesised here, so licensing is clean.

* chime  - soft sine chime (fundamental + quiet fifth/octave), slow attack and long decay.
* whoosh - band-passed noise with a slow swell-and-fade envelope (a breath, not a swish).
* swell  - gentle pad of detuned sines rising over ~1.2 s, for chapter cards.

Hits land on block changes, varied, ~4-8 per minute (never closer than 60/8 s). The track is
levelled so each hit's loudest 150 ms sits >= 18-20 dB under the narration (audio.sfx_under_voice_db),
then the mixer ducks it (and the music) under speech and limits the final file to <= -1 dBTP.
Planning is pure (testable); only `build_track` touches audio.
"""
import random
from pathlib import Path

from src import style

RATE = 48000
MIN_UNDER_VOICE_DB = 18.0

# peak_s: where the clip's energy peaks, so the peak lands on the cut.
CLIPS = {
    "chime_soft_01": {"category": "chime", "peak_s": 0.08, "freq": 523.25, "dur": 1.6},
    "chime_soft_02": {"category": "chime", "peak_s": 0.08, "freq": 587.33, "dur": 1.6},
    "chime_soft_03": {"category": "chime", "peak_s": 0.08, "freq": 659.25, "dur": 1.6},
    "whoosh_soft_01": {"category": "whoosh", "peak_s": 0.45, "band": (300.0, 1800.0), "dur": 0.9},
    "whoosh_soft_02": {"category": "whoosh", "peak_s": 0.45, "band": (200.0, 1200.0), "dur": 0.9},
    "whoosh_soft_03": {"category": "whoosh", "peak_s": 0.45, "band": (400.0, 2400.0), "dur": 0.9},
    "swell_soft_01": {"category": "swell", "peak_s": 1.1, "freq": 261.63, "dur": 1.6},
    "swell_soft_02": {"category": "swell", "peak_s": 1.1, "freq": 293.66, "dur": 1.6},
}

WHOOSH_KINDS = {"code_panel", "diagram", "stock_or_ai_still"}
CHIME_KINDS = {"kinetic_text", "count_up"}
SWELL_KINDS = {"chapter_card"}


def audio_rules():
    a = style.look().get("audio") or {}
    rate = a.get("sfx_per_min") or [4, 8]
    under = max(MIN_UNDER_VOICE_DB, float(a.get("sfx_under_voice_db", 20)))
    return {"under_db": under, "min_per_min": float(rate[0]), "max_per_min": float(rate[1])}


def available():
    """Procedural: always available (needs only numpy)."""
    return True


def _by_category(cat):
    return [c for c, v in CLIPS.items() if v["category"] == cat]


def _category_for(kind):
    if kind in SWELL_KINDS:
        return "swell"
    if kind in CHIME_KINDS:
        return "chime"
    return "whoosh"


# ------------------------------------------------------------------ synthesis

def _envelope(n, attack, release, curve=2.0):
    import numpy as np
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    a = max(1, int(attack * n))
    env = np.ones(n)
    env[:a] = (np.sin(np.linspace(0, np.pi / 2, a)) ** 2)
    rel = max(1, int(release * n))
    env[n - rel:] *= np.linspace(1.0, 0.0, rel) ** curve
    return env, t


def synth(clip_id, rate=RATE):
    """Mono float clip (peak 1.0) for a procedural sound; deterministic per clip id."""
    import numpy as np
    spec = CLIPS[clip_id]
    n = int(spec["dur"] * rate)
    t = np.arange(n) / rate
    cat = spec["category"]
    if cat == "chime":
        f = spec["freq"]
        tone = (np.sin(2 * np.pi * f * t) + 0.25 * np.sin(2 * np.pi * f * 1.5 * t)
                + 0.12 * np.sin(2 * np.pi * f * 2.0 * t))
        attack = np.clip(t / 0.06, 0, 1) ** 2                      # 60 ms soft attack
        decay = np.exp(-np.maximum(0.0, t - 0.06) / 0.45)          # slow exponential decay
        y = tone * attack * decay
    elif cat == "swell":
        f = spec["freq"]
        tone = sum(np.sin(2 * np.pi * f * m * (1 + d) * t) * g
                   for m, d, g in ((1.0, 0.0, 1.0), (1.0, 0.003, 0.7), (1.5, -0.002, 0.35), (2.0, 0.0, 0.2)))
        rise = np.clip(t / spec["peak_s"], 0, 1) ** 2.2             # gentle swell up to the cut
        fall = np.where(t > spec["peak_s"], np.exp(-(t - spec["peak_s"]) / 0.18), 1.0)
        y = tone * rise * fall
    else:  # whoosh: band-passed noise, smooth bell envelope
        rng = np.random.default_rng(sum(map(ord, clip_id)))
        noise = rng.standard_normal(n)
        spectrum = np.fft.rfft(noise)
        freqs = np.fft.rfftfreq(n, 1.0 / rate)
        lo, hi = spec["band"]
        band = np.exp(-0.5 * (np.log(np.maximum(freqs, 1.0) / np.sqrt(lo * hi)) / (np.log(hi / lo) / 2.5)) ** 2)
        y = np.fft.irfft(spectrum * band, n)
        bell = np.exp(-0.5 * ((t - spec["peak_s"]) / (spec["dur"] * 0.2)) ** 2)
        y = y * bell
    # 10 ms fade at both ends: no clicks.
    edge = int(0.01 * rate)
    y[:edge] *= np.linspace(0, 1, edge)
    y[-edge:] *= np.linspace(1, 0, edge)
    peak = float(np.max(np.abs(y))) or 1.0
    return y / peak


def loudest_window_db(samples, rate=RATE, window=0.15):
    """dBFS (RMS, 16-bit full scale = 0 dB) of the loudest `window` seconds of a mono/stereo int-scale array."""
    import numpy as np
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)
    w = max(1, int(window * rate))
    if len(x) <= w:
        rms = np.sqrt(np.mean(x ** 2)) if len(x) else 0.0
    else:
        c = np.concatenate([[0.0], np.cumsum(x ** 2)])
        rms = np.sqrt(np.max(c[w:] - c[:-w]) / w)
    return 20 * np.log10(max(rms, 1e-9) / 32768.0)


# ------------------------------------------------------------------ planning

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
    """Pick cuts for soft hits: ~4-8 per minute, varied clips, never the same clip twice running."""
    r = audio_rules()
    min_gap = 60.0 / r["max_per_min"]
    target_gap = 60.0 / ((r["min_per_min"] + r["max_per_min"]) / 2)
    rng = random.Random(seed)
    events, last, last_clip = [], -1e9, None
    pools = {c: _by_category(c) for c in ("chime", "whoosh", "swell")}
    turn = {c: rng.randrange(len(p)) for c, p in pools.items()}
    for k, (t, kind) in enumerate(changes):
        if t < 0.3 or t > total - 0.3 or t - last < min_gap:
            continue
        priority = kind in SWELL_KINDS
        nxt = next((t2 for t2, _ in changes[k + 1:] if t2 > t), float("inf"))
        # Hit this cut if it is a chapter, the target spacing has passed, or waiting would drop below 4/min.
        if not (priority or t - last >= target_gap or nxt - last > 60.0 / r["min_per_min"]):
            continue
        cat = _category_for(kind)
        pool = pools[cat]
        clip = pool[turn[cat] % len(pool)]
        turn[cat] += 1
        if clip == last_clip:
            clip = pool[turn[cat] % len(pool)]
            turn[cat] += 1
        start = max(0.0, t - CLIPS[clip]["peak_s"])
        events.append({"t": round(start, 3), "clip": clip, "cut": round(t, 3), "gain_db": round(rng.uniform(-1.5, 0.0), 2)})
        last, last_clip = t, clip
    return events


def plan(timeline, kinds, fixed, seed=0):
    """Soft transition hits only (the old keyboard clicks under code panels are gone: calmer mix)."""
    return sorted(plan_transitions(block_changes(timeline, kinds, fixed), timeline["total"], seed),
                  key=lambda e: e["t"])


def per_minute(events, total):
    return round(len(events) / max(1e-6, total / 60.0), 2)


def build_track(events, total, out_wav, voice_wav):
    """Render the events to one 48 kHz stereo wav, levelled against the narration; None if nothing to add.

    Each clip's loudest 150 ms is set `under_db` (>= 18 dB) below the narration's overall level
    (which includes pauses, so the real gap to speech is larger). Mixed in a numpy buffer."""
    if not events:
        return None
    import wave
    import numpy as np
    from pydub import AudioSegment
    r = audio_rules()
    voice = AudioSegment.from_file(str(voice_wav))
    voice_db = voice.dBFS if voice.dBFS != float("-inf") else -20.0
    n = int(total * RATE)
    track = np.zeros((n + RATE // 20, 2), dtype=np.float64)
    cache = {}
    for e in events:
        if e["clip"] not in CLIPS:
            continue
        if e["clip"] not in cache:
            mono = synth(e["clip"]) * 32767.0
            gain = voice_db - r["under_db"] - loudest_window_db(mono)
            mono = mono * (10 ** (gain / 20.0))
            cache[e["clip"]] = np.stack([mono, mono], axis=1)
        clip = cache[e["clip"]] * (10 ** (min(0.0, float(e.get("gain_db", 0.0))) / 20.0))
        at = int(round(float(e["t"]) * RATE))
        if at >= len(track) or at < 0:
            continue
        clip = clip[: len(track) - at]
        track[at: at + len(clip)] += clip
    pcm = np.clip(np.round(track[:n]), -32768, 32767).astype("<i2")
    with wave.open(str(out_wav), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(pcm.tobytes())
    return Path(out_wav)
