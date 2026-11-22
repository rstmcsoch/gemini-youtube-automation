"""Sound design for Engine A: calm, sparse cues generated in code (numpy).

No third-party audio files: every sound is synthesised here, so licensing is clean.

* chime - soft low sine chime (fundamental + quiet fifth/octave, all <= 1.5 kHz), 60 ms attack.
* swell - low pad of detuned sines rising over ~1.1 s, for chapter cards.

No noise sweeps of any kind, nothing for code panels / diagrams / stills, nothing for code typing.
At most one cue per section (a script slide, or a chapter card), never closer than 7.5 s.
Each cue is levelled by integrated loudness (ITU-R BS.1770, K-weighted, gated) to sit
`under_db` (>= 24 dB) below the narration's integrated LUFS; the mixer then ducks it under speech.
DYPOL_SFX=off disables cues entirely. Planning is pure (testable); only `build_track` touches audio.
"""
import os
import random
from pathlib import Path

from src import style

RATE = 48000
MIN_UNDER_VOICE_DB = 24.0
MAX_FREQ_HZ = 1500.0
MIN_ATTACK_S = 0.02

# peak_s: where the clip's energy peaks, so the peak lands on the cut.
CLIPS = {
    "chime_soft_01": {"category": "chime", "peak_s": 0.08, "freq": 523.25, "dur": 1.6},
    "chime_soft_02": {"category": "chime", "peak_s": 0.08, "freq": 587.33, "dur": 1.6},
    "chime_soft_03": {"category": "chime", "peak_s": 0.08, "freq": 659.25, "dur": 1.6},
    "swell_soft_01": {"category": "swell", "peak_s": 1.1, "freq": 261.63, "dur": 1.6},
    "swell_soft_02": {"category": "swell", "peak_s": 1.1, "freq": 293.66, "dur": 1.6},
}

# Only these visual kinds earn a cue; everything else (code panels, diagrams, stills...) stays silent.
CUE_FOR_KIND = {"chapter_card": "swell", "kinetic_text": "chime", "count_up": "chime"}


def enabled():
    return os.environ.get("DYPOL_SFX", "").strip().lower() not in {"off", "0", "false", "no", "none"}


def audio_rules():
    a = style.look().get("audio") or {}
    rate = a.get("sfx_per_min") or [4, 8]
    under = max(MIN_UNDER_VOICE_DB, float(a.get("sfx_under_voice_db", MIN_UNDER_VOICE_DB)))
    return {"under_db": under, "min_per_min": float(rate[0]), "max_per_min": float(rate[1])}


def available():
    """Procedural: always available (needs only numpy) unless switched off with DYPOL_SFX=off."""
    return enabled()


def _by_category(cat):
    return [c for c, v in CLIPS.items() if v["category"] == cat]


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
    else:
        raise ValueError(f"unknown sfx category {cat!r}")
    # 10 ms fade at both ends: no clicks.
    edge = int(0.01 * rate)
    y[:edge] *= np.linspace(0, 1, edge)
    y[-edge:] *= np.linspace(1, 0, edge)
    peak = float(np.max(np.abs(y))) or 1.0
    return y / peak


def _k_weight(x, rate):
    """Apply the BS.1770 K-weighting magnitude response (shelf + RLB high-pass) in the frequency domain."""
    import math
    import numpy as np
    n = x.shape[0]
    size = 1 << max(1, int(np.ceil(np.log2(max(n, 2)))))
    f = np.fft.rfftfreq(size, 1.0 / rate)
    z = np.exp(-2j * np.pi * f / rate)

    def biquad(b, a):
        return (b[0] + b[1] * z + b[2] * z * z) / (a[0] + a[1] * z + a[2] * z * z)

    # BS.1770-4 filter parameters, designed per sample rate (bilinear transform).
    f0, g, q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    k = math.tan(math.pi * f0 / rate)
    vh = 10 ** (g / 20.0)
    vb = vh ** 0.4996667741545416
    a0 = 1.0 + k / q + k * k
    shelf = biquad([(vh + vb * k / q + k * k) / a0, 2.0 * (k * k - vh) / a0, (vh - vb * k / q + k * k) / a0],
                   [1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0])
    f0, q = 38.13547087602444, 0.5003270373238773
    k = math.tan(math.pi * f0 / rate)
    a0 = 1.0 + k / q + k * k
    hp = biquad([1.0, -2.0, 1.0], [1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0])
    h = shelf * hp
    spec = np.fft.rfft(x, size, axis=0) * h[:, None]
    return np.fft.irfft(spec, size, axis=0)[:n]


def integrated_lufs(samples, rate=RATE):
    """Integrated loudness (LUFS, BS.1770: K-weighted, 400 ms blocks / 75 % overlap, -70 LUFS absolute
    and -10 LU relative gates) of a float array scaled to +-1.0, shape (n,) or (n, channels)."""
    import numpy as np
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if not len(x):
        return float("-inf")
    y = _k_weight(x, rate)
    w, hop = int(0.4 * rate), int(0.1 * rate)
    if len(y) < w:
        y = np.concatenate([y, np.zeros((w - len(y), y.shape[1]))])
    c = np.concatenate([np.zeros((1, y.shape[1])), np.cumsum(y ** 2, axis=0)])
    starts = np.arange(0, len(y) - w + 1, hop)
    z = ((c[starts + w] - c[starts]) / w).sum(axis=1)          # channel weights are 1.0 (L/R/C)
    lk = -0.691 + 10 * np.log10(np.maximum(z, 1e-20))
    z = z[lk > -70.0]
    if not len(z):
        return float("-inf")
    rel = -0.691 + 10 * np.log10(z.mean()) - 10.0
    z = z[-0.691 + 10 * np.log10(z) > rel]
    return float(-0.691 + 10 * np.log10(z.mean()))


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


def _sections(timeline, kinds):
    """Section id per shot: the script slide it belongs to; a chapter card always opens a new one."""
    out, sec, prev_slide = [], 0, None
    for g, shot in enumerate(timeline["shots"]):
        slide = shot.get("slide", 0)
        kind = kinds[g] if g < len(kinds) else None
        if g and (slide != prev_slide or kind == "chapter_card"):
            sec += 1
        out.append(sec)
        prev_slide = slide
    return out


def plan_transitions(changes, total, seed=0, sections=None):
    """Pick cuts for soft cues: at most one per section, only on chapter cards (swell) or
    kinetic text / count-ups (chime), >= 60/max_per_min s apart, never the same clip twice running.

    changes: [(cut_time, new_kind)]; sections: parallel list of section ids (None: each change is
    its own section)."""
    if not enabled():
        return []
    r = audio_rules()
    min_gap = 60.0 / r["max_per_min"]
    rng = random.Random(seed)
    events, last, last_clip, used = [], -1e9, None, set()
    pools = {c: _by_category(c) for c in ("chime", "swell")}
    turn = {c: rng.randrange(len(p)) for c, p in pools.items()}
    for k, (t, kind) in enumerate(changes):
        cat = CUE_FOR_KIND.get(kind)
        sec = sections[k] if sections is not None else k
        if cat is None or sec in used or t < 0.3 or t > total - 0.3 or t - last < min_gap:
            continue
        pool = pools[cat]
        clip = pool[turn[cat] % len(pool)]
        turn[cat] += 1
        if clip == last_clip:
            clip = pool[turn[cat] % len(pool)]
            turn[cat] += 1
        start = max(0.0, t - CLIPS[clip]["peak_s"])
        events.append({"t": round(start, 3), "clip": clip, "cut": round(t, 3), "gain_db": round(rng.uniform(-1.5, 0.0), 2)})
        last, last_clip = t, clip
        used.add(sec)
    return events


def plan(timeline, kinds, fixed, seed=0):
    """Calm cues only: no noise sweeps, no keyboard clicks; at most one per section. [] when DYPOL_SFX=off."""
    if not enabled():
        return []
    secs = _sections(timeline, kinds)
    changes, change_secs, prev = [], [], None
    for g, shot in enumerate(timeline["shots"]):
        unit = fixed.get(g, {}).get("block_id") or f"shot{g}"
        if g and unit != prev:
            changes.append((shot["start"], kinds[g] if g < len(kinds) else None))
            change_secs.append(secs[g])
        prev = unit
    return sorted(plan_transitions(changes, timeline["total"], seed, change_secs), key=lambda e: e["t"])


def per_minute(events, total):
    return round(len(events) / max(1e-6, total / 60.0), 2)


def _load_float(path):
    """(samples float +-1.0 shaped (n, ch), rate) via pydub (wav/mp3/...)."""
    import numpy as np
    from pydub import AudioSegment
    seg = AudioSegment.from_file(str(path))
    x = np.array(seg.get_array_of_samples(), dtype=np.float64).reshape(-1, seg.channels)
    return x / float(1 << (8 * seg.sample_width - 1)), seg.frame_rate


def build_track(events, total, out_wav, voice_wav):
    """Render the events to one 48 kHz stereo wav, levelled against the narration; None if nothing to add.

    Each clip is gained so its integrated loudness (as the stereo track carries it) is `under_db`
    (>= 24 dB) below the narration's integrated LUFS. Mixed in a numpy buffer."""
    if not events or not enabled():
        return None
    import wave
    import numpy as np
    r = audio_rules()
    voice, vrate = _load_float(voice_wav)
    voice_lufs = integrated_lufs(voice, vrate)
    if voice_lufs == float("-inf"):
        voice_lufs = -23.0
    target = voice_lufs - r["under_db"]
    n = int(total * RATE)
    track = np.zeros((n + RATE // 20, 2), dtype=np.float64)
    cache = {}
    for e in events:
        if e["clip"] not in CLIPS:
            continue
        if e["clip"] not in cache:
            mono = synth(e["clip"])
            stereo = np.stack([mono, mono], axis=1)
            gain = target - integrated_lufs(stereo)
            cache[e["clip"]] = stereo * (10 ** (gain / 20.0))
        clip = cache[e["clip"]] * (10 ** (min(0.0, float(e.get("gain_db", 0.0))) / 20.0))
        at = int(round(float(e["t"]) * RATE))
        if at >= len(track) or at < 0:
            continue
        clip = clip[: len(track) - at]
        track[at: at + len(clip)] += clip
    pcm = np.clip(np.round(track[:n] * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(out_wav), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(pcm.tobytes())
    return Path(out_wav)
