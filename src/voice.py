"""DYPOL brand narration voice (Engine A).

One male voice shared with Engine B: Kokoro-82M (Apache-2.0, local CPU via kokoro-onnx),
voice ``am_michael`` (picked 3 Nov 2026 by a measured bake-off: lowest word error rate at
2x / 2.5x playback, natural pauses at commas and full stops, 3.6 st pitch movement).

Settings live in style/style.guide.yaml -> ``voice:``; env vars override them:
DYPOL_TTS_PROVIDER (kokoro | gtts), KOKORO_VOICE, KOKORO_SPEED, KOKORO_MODEL_FILE,
KOKORO_MODEL_DIR.

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
KOKORO_SHA256 = {
    "kokoro-v1.0.onnx": "7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5",
    "kokoro-v1.0.int8.onnx": "6e742170d309016e5891a994e1ce1559c702a2ccd0075e67ef7157974f6406cb",
    "voices-v1.0.bin": "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d",
}
DEFAULTS = {"provider": "kokoro", "voice": "am_michael", "speed": 1.08, "lang": "en-us",
            "model_file": "kokoro-v1.0.onnx", "wpm_min": 150, "wpm_max": 165, "wpm_target": 157,
            "speed_min": 0.95, "speed_max": 1.3, "fallback": "gtts"}

_STATE = {"engine": None, "speed": None, "meta": None}


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
    try:
        cfg["speed"] = float(cfg["speed"])
    except (TypeError, ValueError):
        cfg["speed"] = DEFAULTS["speed"]
    cfg["speed"] = min(float(cfg["speed_max"]), max(float(cfg["speed_min"]), cfg["speed"]))
    cfg["provider"] = str(cfg["provider"]).strip().lower()
    return cfg


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
            with requests.get(f"{KOKORO_RELEASE_BASE}/{name}", stream=True, timeout=(10, 180)) as r:
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
        _STATE["engine"] = Kokoro(str(model), str(voices))
    return _STATE["engine"]


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
    speed = _STATE["speed"] or cfg["speed"]
    words = len(text.split())
    samples, rate = engine.create(text, voice=cfg["voice"], speed=speed, lang=cfg["lang"])
    seconds = len(samples) / float(rate) if samples is not None else 0.0
    measured = wpm(words, seconds)
    # Short fragments (a hook line) give a noisy rate: only tune the pace on 40+ words.
    new = corrected_speed(measured, speed, cfg) if words >= 40 else None
    if new is not None:
        print(f"🎤 pace {measured:.0f} WPM at speed {speed}; re-rendering at {new} "
              f"(target {cfg['wpm_min']}-{cfg['wpm_max']})", flush=True)
        samples, rate = engine.create(text, voice=cfg["voice"], speed=new, lang=cfg["lang"])
        seconds = len(samples) / float(rate)
        measured = wpm(words, seconds)
        speed = new
    if words >= 40:
        _STATE["speed"] = speed   # keep one pace across slides
    if samples is None or seconds < 0.5 or not (1.2 <= words / max(seconds, 0.01) <= 4.5):
        raise RuntimeError(f"Kokoro audio implausible ({seconds:.1f}s for {words} words)")
    write_wav(samples, rate, wav_path)
    meta = {"provider": "kokoro", "model": "Kokoro-82M", "model_file": cfg["model_file"],
            "voice": cfg["voice"], "speed": speed, "wpm": round(measured, 1),
            "license": "Apache-2.0 (Kokoro-82M)"}
    _STATE["meta"] = meta
    return meta


def last_meta():
    return dict(_STATE["meta"] or {})


def reset():
    _STATE.update({"engine": None, "speed": None, "meta": None})
