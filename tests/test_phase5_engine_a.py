"""Engine A piece 5: render speed, stock-first visuals, NVIDIA stall breaker, numpy SFX mix,
brand male voice (Kokoro am_michael) and the depth-rewrite tolerance."""
import json
import wave
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src import backdrop, explainer as ex, storyboard as sb, style, visuals, voice


@pytest.fixture(autouse=True)
def _fresh(monkeypatch, tmp_path):
    visuals.reset_breakers()
    voice.reset()
    monkeypatch.setenv("VERTICALS_HOME", str(tmp_path / "vh"))
    for name in ("DYPOL_TTS_PROVIDER", "KOKORO_VOICE", "KOKORO_SPEED", "KOKORO_MODEL_FILE"):
        monkeypatch.delenv(name, raising=False)
    yield
    visuals.reset_breakers()
    voice.reset()


# ------------------------------------------------------------------ render speed

def test_backdrop_base_updates_at_15_fps_but_particles_every_frame():
    bd = backdrop.Backdrop((640, 360), seed=5)
    a, b = bd.frame(1.0), bd.frame(1.0 + 1 / 30)          # same 1/15 s bucket
    assert bd._base(1.0) is bd._base(1.0 + 1 / 30)
    assert a.tobytes() != b.tobytes()                      # particles still moved
    assert bd.frame(0.5).tobytes() != bd.frame(3.0).tobytes()
    fresh = backdrop.Backdrop((640, 360), seed=5)          # pure function of t (chunk-safe)
    assert fresh.frame(1.0 + 1 / 30).tobytes() == b.tobytes()
    assert a.mode == "RGBA"


def test_cached_base_is_not_mutated_by_particles():
    bd = backdrop.Backdrop((320, 180), seed=2)
    base = bd._base(2.0).tobytes()
    bd.frame(2.0)
    bd.frame(2.0 + 1 / 30)
    assert bd._base(2.0).tobytes() == base


def test_grade_vignette_has_no_dither_and_keeps_grain():
    grade = ex.grade_filter()
    assert "vignette=angle=0.38:dither=0" in grade and "noise=" in grade and "eq=" in grade


def test_sfx_track_is_mixed_with_numpy(tmp_path, monkeypatch):
    from src import sfx
    if not sfx.available():
        pytest.skip("sfx clips missing")
    voice_wav = tmp_path / "v.wav"
    rate = 48000
    tone = (np.sin(np.arange(rate * 3) * 2 * np.pi * 220 / rate) * 8000).astype("<i2")
    with wave.open(str(voice_wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(tone.tobytes())
    clip = next(iter(sfx.CLIPS))
    events = [{"t": 0.5, "clip": clip}] + [{"t": 1.0 + k * 0.05, "clip": clip, "click": True}
                                           for k in range(400)]
    out = sfx.build_track(events, 3.0, tmp_path / "sfx.wav", voice_wav)
    with wave.open(str(out)) as wf:
        assert wf.getframerate() == 48000 and wf.getnchannels() == 2
        frames = wf.getnframes()
        data = np.frombuffer(wf.readframes(frames), dtype="<i2").reshape(-1, 2)
    assert abs(frames / 48000 - 3.0) < 0.01
    assert np.abs(data[: int(0.45 * rate)]).max() == 0          # silent before the first hit
    assert np.abs(data[int(0.5 * rate): int(0.7 * rate)]).max() > 0


# ------------------------------------------------------------------ stock first, FLUX last

def _img(path, color):
    Image.new("RGB", (64, 36), color).save(path)
    return path


def _hit(i, tags):
    return {"id": i, "tags": tags, "imageWidth": 4000, "imageHeight": 3000,
            "largeImageURL": f"https://example.test/{i}.jpg", "pageURL": f"https://pixabay.com/{i}"}


def _stub_download(monkeypatch):
    counter = iter(range(1000))

    def fake_download(candidate, path):
        _img(path, (next(counter) * 7 % 255, 60, 90))
        return {"sha256": visuals._sha256(path), "width": 4000, "height": 3000}

    monkeypatch.setattr(visuals, "_download_candidate", fake_download)


def test_stock_comes_before_flux_even_when_flux_is_available(monkeypatch, tmp_path):
    flux_calls = []
    monkeypatch.setattr(visuals, "ai_image_available", lambda: True)
    monkeypatch.setattr(visuals, "flux_generate", lambda *a, **k: flux_calls.append(a) or None)
    monkeypatch.setattr(visuals, "pixabay_video_hits", lambda q: [])
    monkeypatch.setattr(visuals, "pixabay_still_hits", lambda q, o: [
        _hit(10 + k, "weather, radar, satellite") for k in range(6)] if "weather" in q else [])
    _stub_download(monkeypatch)
    spec = {"visual_queries": ["weather radar"], "terms": sb.topic_terms(["weather radar forecast"])}
    assets = visuals.gather_visuals(spec, tmp_path, 4, {"hashes": set(), "ids": set()}, label="w", min_count=0)
    assert [a["provider"] for a in assets] == ["pixabay"] * 4
    assert flux_calls == []


def test_flux_only_when_stock_has_nothing_relevant(monkeypatch, tmp_path):
    order = []
    monkeypatch.setattr(visuals, "ai_image_available", lambda: True)

    def fake_flux(scene, path, orientation="landscape", seed=None):
        order.append("flux")
        _img(path, (len(order) * 30 % 255, 10, 10))
        return {"kind": "image", "provider": "cloudflare_workers_ai_flux", "sha256": visuals._sha256(path),
                "path": str(path), "reusable": True}

    monkeypatch.setattr(visuals, "flux_generate", fake_flux)
    monkeypatch.setattr(visuals, "pixabay_video_hits", lambda q: order.append(f"video:{q}") or [])
    monkeypatch.setattr(visuals, "pixabay_still_hits", lambda q, o: order.append(f"still:{q}") or [])
    spec = {"visual_queries": ["weather radar", "storm satellite"],
            "terms": sb.topic_terms(["weather radar forecast"])}
    assets = visuals.gather_visuals(spec, tmp_path, 1, {"hashes": set(), "ids": set()}, label="w", min_count=0)
    assert [a["provider"] for a in assets] == ["cloudflare_workers_ai_flux"]
    first_flux = order.index("flux")
    assert order[0] == "video:weather radar" and order[1] == "still:weather radar"
    assert any(o.endswith("storm satellite") for o in order[:first_flux])   # topic widening first
    assert not any(o.split(":", 1)[-1] in sb.FALLBACK_TECH_QUERIES for o in order[:first_flux])


def test_stock_videos_per_call_are_bounded(monkeypatch, tmp_path):
    video_tries = []
    monkeypatch.setattr(visuals, "ai_image_available", lambda: False)
    monkeypatch.setattr(visuals, "_stock_video", lambda *a, **k: video_tries.append(1) or {
        "kind": "video", "provider": "pixabay_video", "sha256": str(len(video_tries)), "path": "x"})
    monkeypatch.setattr(visuals, "_stock_still", lambda *a, **k: {
        "kind": "image", "provider": "pixabay", "sha256": "s", "path": "y"})
    spec = {"visual_queries": ["weather"], "terms": {"weather"}}
    assets = visuals.gather_visuals(spec, tmp_path, 4, {"hashes": set(), "ids": set()}, label="w", min_count=0)
    assert [a["kind"] for a in assets].count("video") == visuals.STOCK_VIDEOS_PER_CALL
    assert len(assets) == 4


# ------------------------------------------------------------------ NVIDIA FLUX stalls

def test_nvidia_timeouts_open_the_breaker_after_two(monkeypatch, capsys):
    visuals.trip_breaker("flux", "HTTP 429")
    monkeypatch.setenv("NVIDIA_API_KEY", "k")
    monkeypatch.setattr(visuals, "_nvidia_throttle", lambda: None)
    calls = []

    def slow(*a, **k):
        calls.append(k.get("timeout"))
        raise visuals.requests.Timeout("Read timed out. (read timeout=45)")

    monkeypatch.setattr(visuals.requests, "post", slow)
    for _ in range(6):
        assert visuals.nvidia_flux_request("storm clouds") is None
    assert len(calls) == visuals.NVIDIA_MAX_STALLS == 2
    assert calls[0][1] == visuals.NVIDIA_READ_TIMEOUT <= 45
    assert visuals.breaker_open(visuals.NVIDIA_FLUX_PROVIDER)
    assert "DYPOL_PROVIDER_LIMIT provider=nvidia_flux" in capsys.readouterr().out


def test_nvidia_success_resets_the_stall_count(monkeypatch):
    visuals.trip_breaker("flux", "HTTP 429")
    monkeypatch.setenv("NVIDIA_API_KEY", "k")
    monkeypatch.setattr(visuals, "_nvidia_throttle", lambda: None)
    seq = iter(["timeout", "ok", "timeout", "ok"])

    class R:
        status_code = 200
        text = ""

        def json(self):
            return {"artifacts": [{"base64": ""}]}

    def post(*a, **k):
        if next(seq) == "timeout":
            raise visuals.requests.Timeout("t")
        return R()

    monkeypatch.setattr(visuals.requests, "post", post)
    for _ in range(4):
        visuals.nvidia_flux_request("x")
    assert not visuals.breaker_open(visuals.NVIDIA_FLUX_PROVIDER)


# ------------------------------------------------------------------ brand voice

def test_voice_is_configured_in_the_style_guide():
    v = style.guide()["voice"]
    assert v["provider"] == "kokoro" and v["voice"] == "am_michael"
    assert v["wpm_min"] == 150 and v["wpm_max"] == 165 and 150 <= v["wpm_target"] <= 165
    cfg = voice.settings()
    assert cfg["voice"] == "am_michael" and cfg["lang"] == "en-us"


def test_voice_env_overrides_and_speed_clamp(monkeypatch):
    monkeypatch.setenv("KOKORO_VOICE", "bm_george")
    monkeypatch.setenv("KOKORO_SPEED", "3.0")
    cfg = voice.settings()
    assert cfg["voice"] == "bm_george" and cfg["speed"] == cfg["speed_max"]


def test_pace_correction_targets_150_to_165():
    cfg = voice.settings()
    assert voice.corrected_speed(155, 1.1, cfg) is None
    faster = voice.corrected_speed(135, 1.08, cfg)
    assert faster and 1.2 < faster < 1.3
    slower = voice.corrected_speed(190, 1.2, cfg)
    assert slower and slower < 1.2
    assert voice.corrected_speed(60, 1.2, cfg) == cfg["speed_max"]


class _FakeKokoro:
    def __init__(self):
        self.calls = []

    def create(self, text, voice=None, speed=1.0, lang=None):
        self.calls.append((voice, speed, lang))
        words = len(text.split())
        seconds = words / (135.0 * speed / 1.08) * 60.0      # 135 WPM at 1.08, like the real voice
        return np.zeros(int(seconds * 24000), dtype="float32"), 24000


def test_kokoro_speech_retunes_pace_once_and_keeps_it(monkeypatch, tmp_path):
    fake = _FakeKokoro()
    monkeypatch.setattr(voice, "_engine", lambda cfg: fake)
    monkeypatch.setenv("KOKORO_SPEED", "1.08")
    text = " ".join(["word"] * 120)
    meta = voice.kokoro_speech(text, tmp_path / "a.wav")
    assert len(fake.calls) == 2 and fake.calls[0][0] == "am_michael" and fake.calls[0][2] == "en-us"
    assert 150 <= meta["wpm"] <= 165
    voice.kokoro_speech(text, tmp_path / "b.wav")
    assert len(fake.calls) == 3 and fake.calls[2][1] == meta["speed"]       # kept, no re-render
    with wave.open(str(tmp_path / "b.wav")) as wf:
        assert wf.getframerate() == 24000 and wf.getsampwidth() == 2


def test_text_to_speech_uses_kokoro_and_falls_back_to_gtts(monkeypatch, tmp_path):
    import src.generator as gen
    fake = _FakeKokoro()
    monkeypatch.setattr(voice, "_engine", lambda cfg: fake)
    out = gen.text_to_speech(" ".join(["hello"] * 50), tmp_path / "s.mp3")
    assert Path(out).suffix == ".wav" and Path(out).exists() and fake.calls

    def broken(cfg):
        raise RuntimeError("no model")

    monkeypatch.setattr(voice, "_engine", broken)
    seen = []
    monkeypatch.setattr(gen, "_gtts_speech", lambda text, path: seen.append(text) or Path(path))
    gen.text_to_speech("fallback line", tmp_path / "t.mp3")
    assert seen == ["fallback line"]


def test_requirements_and_workflow_carry_the_voice():
    root = Path(__file__).resolve().parent.parent
    assert "kokoro-onnx" in (root / "requirements.txt").read_text()
    wf = (root / ".github/workflows/main.yml").read_text()
    assert "~/.cache/kokoro-onnx" in wf and "kokoro-onnx-model-files-v2" in wf


# ------------------------------------------------------------------ script depth

class _Resp:
    def __init__(self, text):
        self.text = text


def _content(words_per_slide, slides=7):
    return {"long_form_slides": [{"title": f"S{k}", "content": " ".join(["word"] * words_per_slide)}
                                 for k in range(slides)],
            "short_script": " ".join(["short"] * 140)}


def test_no_rewrite_when_a_few_words_short(monkeypatch):
    import src.generator as gen
    calls = []
    monkeypatch.setattr(gen, "_generate_text_with_fallback", lambda *a, **k: calls.append(1))
    near = _content(int(gen.LONG_MIN_WORDS * 0.9 / 7) + 1)
    gen.ensure_depth(object(), near, "prompt")
    assert calls == []


def test_rewrite_when_clearly_thin(monkeypatch):
    import src.generator as gen
    calls = []

    def fake(client, prompt, task_name="", **_k):
        calls.append(task_name)
        return _Resp(json.dumps(_content(200)))

    monkeypatch.setattr(gen, "_generate_text_with_fallback", fake)
    gen.ensure_depth(object(), _content(120), "prompt")
    assert calls == ["lesson depth rewrite"]
    assert gen.LONG_MIN_WORDS >= 1250 and gen.SHORT_WORDS[0] >= 130
