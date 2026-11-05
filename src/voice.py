"""DYPOL brand narration voice (Engine A).

One male voice shared with Engine B: Kokoro-82M (Apache-2.0, local CPU via kokoro-onnx),
voice ``am_michael`` (picked 3 Nov 2026 by a measured bake-off: lowest word error rate at
2x / 2.5x playback, natural pauses at commas and full stops, 3.6 st pitch movement).

Settings live in style/style.guide.yaml -> ``voice:``; env vars override them:
DYPOL_TTS_PROVIDER (kokoro | gtts), KOKORO_VOICE, KOKORO_SPEED, KOKORO_MODEL_FILE,
KOKORO_MODEL_DIR. DYPOL_VOICE (set by the workflow from the console's `voice` input or the
DYPOL_DEFAULT_VOICE repo variable) wins over KOKORO_VOICE; it is a Kokoro voice id
("am_michael") or a blend ("am_michael:0.6,am_adam:0.4"). Every id is checked against the
voices in the loaded Kokoro voices file; an invalid value logs one line and uses am_michael.

Word timings: the default model is the timestamped Kokoro-82M ONNX export
(onnx-community/Kokoro-82M-v1.0-ONNX-timestamped, pinned revision, Apache-2.0). It renders the
same audio as kokoro-v1.0.onnx and also returns per-phoneme durations. Those are mapped onto the
script's words (``map_word_timings``) by aligning each word's own pronunciation with the spoken
phonemes, so "2024" (four spoken words) still lands on one script word. The result is written
beside the WAV (``<name>.words.json``) and read back by src/production.py, which drives cuts,
block cues, bullet reveals and captions from the word starts.

Pace: the narration targets ``wpm_target`` (150-165 WPM, speech only). After each render the
measured words-per-minute is checked; outside the band, the text is rendered once more at a
corrected speed, and that speed is kept for the rest of the run so every slide sounds alike.
"""
import os
import wave
from pathlib import Path

from src import style

KOKORO_RELEASE_BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
KOKORO_VOICES_FILE = "voices-v1.0.bin"
KOKORO_TS_FILE = "kokoro-v1.0-timestamped.onnx"
KOKORO_TS_REPO = "onnx-community/Kokoro-82M-v1.0-ONNX-timestamped"
KOKORO_TS_REVISION = "dd4401a9add81ac692d20e240d22ec9dda82cc29"
KOKORO_URLS = {
    KOKORO_TS_FILE: f"https://huggingface.co/{KOKORO_TS_REPO}/resolve/{KOKORO_TS_REVISION}/onnx/model.onnx",
}
KOKORO_SHA256 = {
    KOKORO_TS_FILE: "651ea8291843a92276a4a003581a215cb07d15e47dde6fcfb1b768f9a1682054",
    "kokoro-v1.0.onnx": "7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5",
    "kokoro-v1.0.int8.onnx": "6e742170d309016e5891a994e1ce1559c702a2ccd0075e67ef7157974f6406cb",
    "voices-v1.0.bin": "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d",
}
DEFAULTS = {"provider": "kokoro", "voice": "am_michael", "speed": 1.08, "lang": "en-us",
            "model_file": KOKORO_TS_FILE, "wpm_min": 150, "wpm_max": 165, "wpm_target": 157,
            "speed_min": 0.95, "speed_max": 1.4, "fallback": "gtts"}

FALLBACK_VOICE = "am_michael"

_STATE = {"engine": None, "speed": None, "meta": None, "voice": None}


def settings():
    """Voice settings: style guide `voice:` block, then env overrides; speeds clamped."""
    cfg = dict(DEFAULTS)
    try:
        cfg.update({k: v for k, v in (style.guide().get("voice") or {}).items() if v is not None})
    except Exception:  # noqa: BLE001 - a broken guide never silences the narration
        pass
    env = {"provider": "DYPOL_TTS_PROVIDER", "voice": "KOKORO_VOICE", "speed": "KOKORO_SPEED",
           "model_file": "KOKORO_MODEL_FILE"}
    for key, name in env.items():
        value = (os.environ.get(name) or "").strip()
        if value:
            cfg[key] = value
    dypol_voice = (os.environ.get("DYPOL_VOICE") or "").strip()
    if dypol_voice:
        cfg["voice"] = dypol_voice
    try:
        cfg["speed"] = float(cfg["speed"])
    except (TypeError, ValueError):
        cfg["speed"] = DEFAULTS["speed"]
    cfg["speed"] = min(float(cfg["speed_max"]), max(float(cfg["speed_min"]), cfg["speed"]))
    cfg["provider"] = str(cfg["provider"]).strip().lower()
    return cfg


def parse_voice_spec(spec):
    """[(id, weight), ...] for "am_michael" or "am_michael:0.6,am_adam:0.4"; weights sum to 1.
    Raises ValueError on a malformed spec."""
    text = str(spec or "").strip()
    if not text:
        raise ValueError("empty voice")
    parts = []
    for raw in text.split(","):
        item = raw.strip()
        if not item:
            raise ValueError(f"empty part in voice {text!r}")
        if ":" in item:
            vid, _, weight = item.partition(":")
            try:
                w = float(weight)
            except ValueError:
                raise ValueError(f"bad weight in voice {text!r}") from None
        else:
            vid, w = item, 1.0
        vid = vid.strip()
        if not vid or w <= 0 or w != w:
            raise ValueError(f"bad part {item!r} in voice {text!r}")
        parts.append((vid, w))
    if len({vid for vid, _ in parts}) != len(parts):
        raise ValueError(f"repeated id in voice {text!r}")
    total = sum(w for _, w in parts)
    return [(vid, w / total) for vid, w in parts]


def resolve_voice(spec, available):
    """Validated voice components for `spec` against the `available` Kokoro ids.
    Returns (canonical_spec, [(id, weight), ...]); anything invalid logs one line and falls
    back to am_michael."""
    known = set(available or [])
    try:
        parts = parse_voice_spec(spec)
        unknown = [vid for vid, _ in parts if vid not in known]
        if unknown:
            raise ValueError(f"unknown Kokoro voice id(s) {', '.join(unknown)}")
    except ValueError as e:
        print(f"⚠️ VOICE_INVALID {str(spec)[:80]!r}: {e}; using {FALLBACK_VOICE}", flush=True)
        return FALLBACK_VOICE, [(FALLBACK_VOICE, 1.0)]
    if len(parts) == 1:
        return parts[0][0], [(parts[0][0], 1.0)]
    canonical = ",".join(f"{vid}:{round(w, 4):g}" for vid, w in parts)
    return canonical, parts


def _voice_arg(engine, cfg):
    """(voice argument for Kokoro.create, canonical spec). A blend is the weighted sum of the
    voice style vectors. Resolved once per run."""
    if _STATE["voice"] is None:
        try:
            available = list(engine.get_voices())
        except Exception:  # noqa: BLE001 - an engine without a voice list only gets the default
            available = [FALLBACK_VOICE]
        canonical, parts = resolve_voice(cfg.get("voice"), available)
        if len(parts) == 1:
            arg = parts[0][0]
        else:
            arg = sum(engine.get_voice_style(vid) * w for vid, w in parts)
        _STATE["voice"] = (arg, canonical)
        print(f"🎤 narration voice: {canonical}", flush=True)
    return _STATE["voice"]


def model_dir():
    default = Path.home() / ".cache" / "kokoro-onnx"
    return Path(os.environ.get("KOKORO_MODEL_DIR", "") or str(default)).expanduser()


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_files(model_file):
    """(model, voices) paths; downloads the official release files when missing; checks SHA-256."""
    import requests
    folder = model_dir()
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in (model_file, KOKORO_VOICES_FILE):
        path = folder / name
        if not path.exists() or path.stat().st_size == 0:
            print(f"🎤 downloading Kokoro file {name}", flush=True)
            tmp = path.with_suffix(path.suffix + ".part")
            url = KOKORO_URLS.get(name) or f"{KOKORO_RELEASE_BASE}/{name}"
            with requests.get(url, stream=True, timeout=(10, 180)) as r:
                r.raise_for_status()
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
            tmp.replace(path)
        expected = KOKORO_SHA256.get(name)
        if expected and _sha256(path) != expected:
            path.unlink(missing_ok=True)
            raise RuntimeError(f"Kokoro file checksum mismatch: {name}")
        paths.append(path)
    return paths[0], paths[1]


def _engine(cfg):
    if _STATE["engine"] is None:
        from kokoro_onnx import Kokoro
        model, voices = ensure_files(cfg["model_file"])
        _STATE["engine"] = enable_timings(Kokoro(str(model), str(voices)))
    return _STATE["engine"]


def enable_timings(engine):
    """kokoro-onnx looks for an output named 'duration'; the timestamped export names its second
    output 'durations'. Switch create_timed() on for either name."""
    try:
        outputs = engine.sess.get_outputs()
        if not getattr(engine, "has_timings", False) and len(outputs) > 1 and \
                outputs[1].name in ("duration", "durations"):
            engine.has_timings = True
    except Exception:  # noqa: BLE001 - no timings only means the estimate is used
        pass
    return engine


def wpm(words, seconds):
    return 0.0 if seconds <= 0 else words / seconds * 60.0


def corrected_speed(measured_wpm, speed, cfg):
    """Speed that moves `measured_wpm` to the target, or None when already inside the band
    (or when the correction would leave the allowed speed range)."""
    lo, hi = float(cfg["wpm_min"]), float(cfg["wpm_max"])
    if measured_wpm <= 0 or lo <= measured_wpm <= hi:
        return None
    new = round(speed * float(cfg["wpm_target"]) / measured_wpm, 3)
    new = min(float(cfg["speed_max"]), max(float(cfg["speed_min"]), new))
    return None if abs(new - speed) < 0.005 else new


def write_wav(samples, rate, path):
    import numpy as np
    pcm = (np.clip(np.asarray(samples, dtype="float32"), -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(rate))
        wf.writeframes(pcm.tobytes())


def kokoro_speech(text, wav_path, cfg=None):
    """Render `text` with Kokoro to `wav_path` (16-bit mono). Returns a meta dict."""
    cfg = cfg or settings()
    text = str(text or "").strip()
    if not text:
        raise RuntimeError("empty narration text")
    engine = _engine(cfg)
    voice_arg, voice_name = _voice_arg(engine, cfg)
    speed = _STATE["speed"] or cfg["speed"]
    words = len(text.split())
    Path(timings_path(wav_path)).unlink(missing_ok=True)   # never leave a stale sidecar
    samples, rate, phones = _render(engine, text, voice_arg, speed, cfg["lang"])
    seconds = len(samples) / float(rate) if samples is not None else 0.0
    measured = wpm(words, seconds)
    # Short fragments (a hook line) give a noisy rate: only tune the pace on 40+ words.
    new = corrected_speed(measured, speed, cfg) if words >= 40 else None
    if new is not None:
        print(f"🎤 pace {measured:.0f} WPM at speed {speed}; re-rendering at {new} "
              f"(target {cfg['wpm_min']}-{cfg['wpm_max']})", flush=True)
        samples, rate, phones = _render(engine, text, voice_arg, new, cfg["lang"])
        seconds = len(samples) / float(rate)
        measured = wpm(words, seconds)
        speed = new
    if words >= 40:
        _STATE["speed"] = speed   # keep one pace across slides
    if samples is None or seconds < 0.5 or not (1.2 <= words / max(seconds, 0.01) <= 4.5):
        raise RuntimeError(f"Kokoro audio implausible ({seconds:.1f}s for {words} words)")
    write_wav(samples, rate, wav_path)
    timed = None
    if phones:
        try:
            timed = map_word_timings(text.split(), phonemize_words(engine, text.split(), cfg["lang"]),
                                     phones, seconds)
            timed = snap_word_starts(timed, samples, rate) if timed else timed
        except Exception as exc:  # noqa: BLE001 - the estimate covers a failed mapping
            print(f"⚠️ word timing map failed ({str(exc)[:120]})", flush=True)
    if timed:
        save_word_timings(wav_path, text, timed, source="kokoro-durations")
    meta = {"provider": "kokoro", "model": "Kokoro-82M", "model_file": cfg["model_file"],
            "voice": voice_name, "speed": speed, "wpm": round(measured, 1),
            "word_timings": "kokoro-durations" if timed else "none",
            "license": "Apache-2.0 (Kokoro-82M)"}
    _STATE["meta"] = meta
    return meta


def _render(engine, text, voice_arg, speed, lang):
    """(samples, rate, [(phoneme, start, end), ...] or None). Timed when the model returns durations."""
    if getattr(engine, "has_timings", False) and hasattr(engine, "create_timed"):
        samples, rate, spoken = engine.create_timed(text, voice=voice_arg, speed=speed, lang=lang)
        phones = [(str(t.phoneme), float(t.start), float(t.end)) for t in (spoken or [])]
        return samples, rate, phones or None
    samples, rate = engine.create(text, voice=voice_arg, speed=speed, lang=lang)
    return samples, rate, None


# ---------------------------------------------------------------------------
# Word timings: phoneme durations -> script words
# ---------------------------------------------------------------------------

# Not sounds: word gaps, punctuation and stress / length marks (they shift with context).
_NOT_SOUND = set(" \t;:,.!?¡¿—…\"«»“”(){}[]-–'ˈˌː")
MIN_MATCHED = 0.6     # below this share of words found in the audio, the mapping is not trusted


def phonemize_words(engine, words, lang="en-us"):
    """Each script word's own pronunciation (espeak, as Kokoro speaks it). One batched call when
    the phonemizer accepts a list; else word by word through the engine's tokenizer."""
    words = list(words)
    try:
        import phonemizer
        from kokoro_onnx.tokenizer import _espeak_lock
        with _espeak_lock:
            out = phonemizer.phonemize(words, lang, preserve_punctuation=True, with_stress=True,
                                       preserve_empty_lines=True)
        if isinstance(out, list) and len(out) == len(words):
            return [str(x) for x in out]
    except Exception:  # noqa: BLE001 - fall through to word-by-word
        pass
    return [engine.tokenizer.phonemize(w, lang) if w.strip() else "" for w in words]


def map_word_timings(words, word_phonemes, phones, duration=None):
    """[(word, start, end)] for the script `words`, from the spoken `phones` [(phoneme, start, end)].

    The phonemizer splits the spoken text into its own words ("2024" -> "two thousand twenty
    four"), so words are not counted off; each word's pronunciation is aligned with the spoken
    phoneme sequence (difflib on sounds only) and a word starts at its first matched sound
    (stress marks just before it included). Words with no matched sound share the gap between
    their neighbours. None when fewer than MIN_MATCHED of the words are found."""
    import difflib
    words = [str(w) for w in words]
    if not words or not phones or len(word_phonemes) != len(words):
        return None
    spoken, spoken_idx = [], []
    for i, (ph, _, _) in enumerate(phones):
        if ph not in _NOT_SOUND and ph.strip():
            spoken.append(ph)
            spoken_idx.append(i)
    script, owner = [], []
    for k, ph in enumerate(word_phonemes):
        for ch in str(ph or ""):
            if ch not in _NOT_SOUND and ch.strip():
                script.append(ch)
                owner.append(k)
    if not spoken or not script:
        return None
    first, last = [None] * len(words), [None] * len(words)
    sm = difflib.SequenceMatcher(None, script, spoken, autojunk=False)
    for a, b, size in sm.get_matching_blocks():
        for j in range(size):
            k, pi = owner[a + j], spoken_idx[b + j]
            if first[k] is None or pi < first[k]:
                first[k] = pi
            if last[k] is None or pi > last[k]:
                last[k] = pi
    found = sum(1 for f in first if f is not None)
    if found < MIN_MATCHED * len(words):
        return None
    starts, ends = [None] * len(words), [None] * len(words)
    for k in range(len(words)):
        if first[k] is None:
            continue
        i = first[k]
        while i > 0 and phones[i - 1][0] in ("ˈ", "ˌ"):   # the stress mark opens the word
            i -= 1
        starts[k], ends[k] = phones[i][1], phones[last[k]][2]
    total = float(duration) if duration else phones[-1][2]
    # Unmatched runs share the gap between their neighbours.
    k = 0
    while k < len(words):
        if starts[k] is not None:
            k += 1
            continue
        j = k
        while j < len(words) and starts[j] is None:
            j += 1
        lo = ends[k - 1] if k > 0 else 0.0
        hi = starts[j] if j < len(words) else total
        hi = max(hi, lo)
        step = (hi - lo) / (j - k)
        for m in range(k, j):
            starts[m], ends[m] = lo + step * (m - k), lo + step * (m - k + 1)
        k = j
    out, prev = [], 0.0
    for k, w in enumerate(words):
        s = max(prev, float(starts[k]))
        e = max(float(ends[k]), s + 0.02)
        if k + 1 < len(words) and starts[k + 1] is not None and starts[k + 1] > s:
            e = min(e, float(starts[k + 1]))
        out.append((w, round(s, 3), round(e, 3)))
        prev = s
    return out


_PAUSE_PUNCT = (",", ";", ":", ".", "!", "?", "…", "—", "-")


def snap_word_starts(words, samples, rate, before=0.2, after=0.2, inner=(0.12, 0.12),
                     quiet_db=-40.0, min_quiet=0.04, min_sound=0.03):
    """Word starts moved onto the audible onset where a silence precedes the word.

    Kokoro's durations line up with the audio inside a phrase, but at a comma or full stop the
    model often voices the pause a little before the mark's own duration, so the next word's
    duration-based start can trail its real onset by ~0.1-0.15 s; and a word opening on a stop
    ("before") is timed from its silent closure, ~0.1 s ahead of the burst the viewer hears.
    For each word the quiet run (>= min_quiet s, quiet_db below the loudest 10 ms frame, followed
    by >= min_sound s of sound) whose end is nearest the start is looked for within
    [start - before, start + after] after punctuation, else [start - inner[0], start + inner[1]]
    and touching the boundary with the previous word;
    (after punctuation the longest such run: the pause itself); the word then starts where that
    run ends and the previous word ends no later than where it began. A word with no silence before it keeps its duration-based start."""
    import numpy as np
    words = list(words or [])
    x = np.asarray(samples, dtype="float64").ravel() if samples is not None else np.zeros(0)
    frame = max(1, int(0.01 * float(rate)))
    n = len(x) // frame
    if not words or n < 3:
        return words
    rms = np.sqrt((x[: n * frame].reshape(n, frame) ** 2).mean(axis=1))
    quiet = 20 * np.log10(rms + 1e-9) <= 20 * np.log10(rms.max() + 1e-9) + quiet_db
    blip = max(1, int(round(min_sound * rate / frame)))
    i = 0
    while i < n:                         # a click shorter than min_sound is not an onset
        if quiet[i]:
            i += 1
            continue
        j = i
        while j < n and not quiet[j]:
            j += 1
        if j - i < blip and i > 0 and j < n:
            quiet[i:j] = True
        i = j
    runs, i = [], 0                      # (first quiet frame, first loud frame after it)
    while i < n:
        if quiet[i]:
            j = i
            while j < n and quiet[j]:
                j += 1
            if (j - i) * frame / rate >= min_quiet and j < n:
                runs.append((i, j))
            i = j
        else:
            i += 1
    out = [list(w) for w in words]
    for k in range(1, len(out)):
        punct = str(out[k - 1][0]).endswith(_PAUSE_PUNCT)
        lo, hi = (before, after) if punct else inner
        start = float(out[k][1])
        nxt = float(out[k + 1][1]) if k + 1 < len(out) else float("inf")
        best = None
        for a, b in runs:
            onset = b * frame / rate
            if not (start - lo <= onset <= start + hi and float(out[k - 1][1]) + 0.05 < onset < nxt - 0.02):
                continue
            # Mid-phrase, the silence must sit at the word boundary (not a stop closure inside
            # the previous word): it reaches the previous word's end and begins by this start.
            if not punct and (onset < float(out[k - 1][2]) - 0.08 or a * frame / rate > start + 0.05):
                continue
            # After punctuation the pause is the longest silence near the mark; elsewhere the
            # silence nearest the duration-based start.
            key = (-(b - a), abs(onset - start)) if punct else (abs(onset - start),)
            if best is None or key < best[2]:
                best = (a * frame / rate, onset, key)
        if best is None:
            continue
        out[k][1] = round(best[1], 3)
        out[k][2] = round(max(float(out[k][2]), best[1] + 0.02), 3)
        out[k - 1][2] = round(max(float(out[k - 1][1]) + 0.02, min(float(out[k - 1][2]), best[0])), 3)
    return [(w, a, b) for w, a, b in out]


def timings_path(wav_path):
    """Sidecar holding the word timings of a narration WAV."""
    p = Path(str(wav_path))
    return p.with_name(p.stem + ".words.json")


def _norm_text(text):
    return " ".join(str(text or "").split())


def save_word_timings(wav_path, text, words, source="kokoro-durations"):
    import json
    data = {"text": _norm_text(text), "source": source,
            "words": [[str(w), float(a), float(b)] for w, a, b in words]}
    timings_path(wav_path).write_text(json.dumps(data), encoding="utf-8")


def load_word_timings(wav_path, text, duration=None):
    """[(word, start, end)] saved for `wav_path` when they belong to exactly `text`, else None."""
    import json
    path = timings_path(wav_path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    words = [(str(w), float(a), float(b)) for w, a, b in data.get("words") or []]
    if data.get("text") != _norm_text(text) or len(words) != len(_norm_text(text).split()) or not words:
        return None
    if duration is not None:
        dur = float(duration)
        if words[-1][1] > dur + 0.25:
            return None             # timings of some other (longer) audio
        words = [(w, round(min(a, dur), 3), round(min(max(b, a), dur), 3)) for w, a, b in words]
    return words


def carry_word_timings(src_wav, dst_wav, shift=0.0, scale=1.0):
    """Copy src's word timings to dst after an audio edit: t -> (t + shift) / scale, clipped at 0
    (silence trimmed off the front = negative shift; atempo factor = scale)."""
    import json
    try:
        data = json.loads(timings_path(src_wav).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        timings_path(dst_wav).unlink(missing_ok=True)
        return False
    scale = float(scale) or 1.0
    data["words"] = [[w, round(max(0.0, (a + shift) / scale), 3), round(max(0.0, (b + shift) / scale), 3)]
                     for w, a, b in data.get("words") or []]
    timings_path(dst_wav).write_text(json.dumps(data), encoding="utf-8")
    return True


def last_meta():
    return dict(_STATE["meta"] or {})


def reset():
    _STATE.update({"engine": None, "speed": None, "meta": None, "voice": None})
