"""Gemini retry/fallback (llm_unavailable), reviewer feedback, failure categories, parallel render."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

import src.generator as gen
from src import explainer as ex
from src import storyboard as sb
from src.failure import failure_category, report_failure

REPO = Path(__file__).resolve().parents[1]


class _Resp:
    def __init__(self, text="ok"):
        self.text = text


class _ServerError(Exception):
    def __init__(self, code, msg):
        super().__init__(msg)
        self.code = code


class _Client:
    """Fake genai client: `script` maps model -> list of outcomes (exception or text)."""

    def __init__(self, script):
        self.script = {k: list(v) for k, v in script.items()}
        self.calls = []
        self.models = self

    def generate_content(self, model, contents):
        self.calls.append(model)
        outcome = self.script[model].pop(0) if self.script.get(model) else _ServerError(503, "503 UNAVAILABLE")
        if isinstance(outcome, Exception):
            raise outcome
        return _Resp(outcome)


@pytest.fixture
def sleeps(monkeypatch):
    waits = []
    monkeypatch.setattr(gen, "_sleep", waits.append)
    monkeypatch.setenv("GEMINI_TEXT_MODEL", "primary")
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "secondary")
    monkeypatch.delenv("GEMINI_RETRY_DELAYS", raising=False)
    return waits


def test_falls_back_to_secondary_model_on_503(sleeps):
    client = _Client({"primary": [_ServerError(503, "503 UNAVAILABLE")], "secondary": ["hello"]})
    assert gen._generate_text_with_fallback(client, "p").text == "hello"
    assert client.calls == ["primary", "secondary"]
    assert 10 not in sleeps                      # no long backoff when a fallback answers


def test_server_disconnected_retries_rounds_with_10_30_60_backoff(sleeps):
    disconnected = type("RemoteProtocolError", (Exception,), {})("Server disconnected without sending a response.")
    client = _Client({
        "primary": [disconnected, _ServerError(500, "500 INTERNAL"), _ServerError(502, "bad gateway"), "late ok"],
        "secondary": [_ServerError(503, "503"), _ServerError(503, "503"), _ServerError(503, "503")],
    })
    assert gen._generate_text_with_fallback(client, "p").text == "late ok"
    assert [w for w in sleeps if w >= 10] == [10, 30, 60]
    assert client.calls == ["primary", "secondary"] * 3 + ["primary"]


def test_exhausted_retries_raise_llm_unavailable_with_category(sleeps):
    client = _Client({})
    with pytest.raises(gen.LLMUnavailableError) as info:
        gen._generate_text_with_fallback(client, "p", task_name="lesson content generation")
    assert len(client.calls) == 8                 # 4 rounds x 2 models
    assert failure_category(info.value) == "llm_unavailable"
    assert "lesson content generation" in str(info.value)


def test_non_transient_error_is_not_retried(sleeps):
    client = _Client({"primary": [_ServerError(400, "400 INVALID_ARGUMENT: bad prompt")]})
    with pytest.raises(_ServerError):
        gen._generate_text_with_fallback(client, "p")
    assert client.calls == ["primary"]


def test_missing_fallback_model_is_dropped_not_fatal(sleeps):
    client = _Client({"primary": [_ServerError(503, "503"), "ok"],
                      "secondary": [_ServerError(404, "404 NOT_FOUND: models/secondary is not found")]})
    assert gen._generate_text_with_fallback(client, "p").text == "ok"
    assert client.calls == ["primary", "secondary", "primary"]


def test_empty_response_counts_as_transient(sleeps):
    client = _Client({"primary": [""], "secondary": ["filled"]})
    assert gen._generate_text_with_fallback(client, "p").text == "filled"


def test_retry_delays_env_override(sleeps, monkeypatch):
    monkeypatch.setenv("GEMINI_RETRY_DELAYS", "1,2")
    client = _Client({})
    with pytest.raises(gen.LLMUnavailableError):
        gen._generate_text_with_fallback(client, "p")
    assert sleeps.count(2.0) == 1                 # round-3 backoff from the override
    assert len(client.calls) == 6                 # 3 rounds x 2 models


def test_main_prints_failure_category_line(tmp_path, monkeypatch, capsys):
    category = report_failure(gen.LLMUnavailableError("down"), tmp_path)
    assert category == "llm_unavailable"
    assert "DYPOL_FAILURE_CATEGORY=llm_unavailable" in capsys.readouterr().out
    record = json.loads((tmp_path / "dypol_failure.json").read_text())
    assert record["failure_category"] == "llm_unavailable"


@pytest.mark.parametrize("exc,expected", [
    (RuntimeError("ffmpeg encode failed with exit code 1"), "render"),
    (RuntimeError("YouTube upload refused"), "publish"),
    (RuntimeError("Pixabay returned garbage"), "asset"),
    (ValueError("something odd"), "unknown"),
])
def test_failure_category_heuristics(exc, expected):
    assert failure_category(exc) == expected


def test_failure_category_follows_exception_chain():
    try:
        try:
            raise gen.LLMUnavailableError("down")
        except gen.LLMUnavailableError as inner:
            raise RuntimeError("wrapper") from inner
    except RuntimeError as outer:
        assert failure_category(outer) == "llm_unavailable"


# ------------------------------------------------------------------ reviewer feedback

def test_reviewer_feedback_capped_and_injected(monkeypatch):
    captured = {}

    def fake(client, prompt, task_name=""):
        captured["prompt"] = prompt
        return _Resp(json.dumps({"long_form_slides": []}))

    monkeypatch.setenv("GOOGLE_API_KEY", "test")
    monkeypatch.setenv("REVIEWER_FEEDBACK", "Open with the phishing statistic. " + "y" * 4000)
    monkeypatch.setattr(gen, "_generate_text_with_fallback", fake)
    monkeypatch.setattr(gen.genai, "Client", lambda api_key: object())
    gen.generate_lesson_content("Topic")
    prompt = captured["prompt"]
    assert "REVIEWER FEEDBACK" in prompt and "Open with the phishing statistic." in prompt
    assert len(gen.reviewer_feedback_text()) == gen.REVIEWER_FEEDBACK_MAX_CHARS


def test_no_feedback_block_when_unset(monkeypatch):
    captured = {}
    monkeypatch.setenv("GOOGLE_API_KEY", "test")
    monkeypatch.delenv("REVIEWER_FEEDBACK", raising=False)
    monkeypatch.setattr(gen, "_generate_text_with_fallback",
                        lambda c, p, task_name="": captured.setdefault("p", p) and _Resp("{}"))
    monkeypatch.setattr(gen.genai, "Client", lambda api_key: object())
    gen.generate_lesson_content("Topic")
    assert "REVIEWER FEEDBACK" not in captured["p"]


# ------------------------------------------------------------------ parallel render

def test_chunk_bounds_cover_every_frame_once():
    for frames, workers in [(690, 3), (100, 4), (7, 3), (5, 1)]:
        bounds = ex.chunk_bounds(frames, workers)
        assert bounds[0][0] == 0 and bounds[-1][1] == frames
        assert all(a < b for a, b in bounds)
        assert all(bounds[k][1] == bounds[k + 1][0] for k in range(len(bounds) - 1))


def _img(path, color):
    Image.new("RGB", (400, 240), color).save(path)
    return str(path)


def _tone(path, seconds):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"sine=frequency=330:duration={seconds}",
                    str(path)], check=True)
    return path


def _probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_entries",
                          "stream=codec_type,nb_read_frames,width,height,duration", "-of", "json", str(path)],
                         capture_output=True, text=True).stdout
    return {s["codec_type"]: s for s in json.loads(out)["streams"]}


def test_parallel_render_matches_serial_frame_count_and_size(tmp_path):
    size = (320, 180)
    tl = sb.build_timeline([4.5, 4.5], 1.5)
    imgs = [_img(tmp_path / f"i{k}.png", c) for k, c in enumerate([(200, 60, 40), (40, 160, 200), (90, 200, 90)])]
    assets = [{"kind": "image", "path": imgs[k % 3]} for k in range(len(tl["shots"]))]
    audio = _tone(tmp_path / "a.wav", tl["total"])
    serial = ex.render(tl, assets, audio, tmp_path / "s.mp4", size, lambda f, t: f, fps=10, workers=1)
    logs = []
    parallel = ex.render(tl, assets, audio, tmp_path / "p.mp4", size, lambda f, t: f, fps=10, workers=2,
                         progress=logs.append)
    assert any("parallel render: 2 chunks" in line for line in logs)
    assert not any("falling back" in line for line in logs)
    s, p = _probe(serial), _probe(parallel)
    assert p["video"]["nb_read_frames"] == s["video"]["nb_read_frames"] == str(int(tl["total"] * 10))
    assert (p["video"]["width"], p["video"]["height"]) == size
    assert abs(float(p["audio"]["duration"]) - float(p["video"]["duration"])) < 0.15
    assert not list(tmp_path.glob("render_chunks_*"))      # temp chunks cleaned up


def test_parallel_render_failure_falls_back_to_serial(tmp_path, monkeypatch):
    size = (160, 90)
    tl = sb.build_timeline([4.5, 4.5], 1.5)
    img = _img(tmp_path / "i.png", (10, 120, 200))
    assets = [{"kind": "image", "path": img} for _ in tl["shots"]]
    audio = _tone(tmp_path / "a.wav", tl["total"])

    def boom(*a, **k):
        raise RuntimeError("pool died")

    monkeypatch.setattr(ex, "_render_parallel", boom)
    logs = []
    out = ex.render(tl, assets, audio, tmp_path / "o.mp4", size, lambda f, t: f, fps=10, workers=3,
                    progress=logs.append)
    assert out.exists() and any("falling back to single pass" in line for line in logs)


def test_short_timelines_stay_serial(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(ex, "_render_parallel", lambda *a, **k: called.append(1))
    monkeypatch.setattr(ex, "_render_serial", lambda *a, **k: tmp_path / "x.mp4")
    tl = {"total": 3.0, "shots": [], "slides": []}
    ex.render(tl, [], "a.wav", tmp_path / "x.mp4", (16, 16), lambda f, t: f, fps=30, workers=4)
    assert not called


def test_render_workers_env(monkeypatch):
    monkeypatch.setenv("DYPOL_RENDER_WORKERS", "3")
    assert ex.render_workers() == 3
    monkeypatch.setenv("DYPOL_RENDER_WORKERS", "nope")
    assert 1 <= ex.render_workers() <= 4


def test_final_render_keeps_1080p_and_quality_settings():
    from src import production as prod
    assert prod.LONG_SIZE == (1920, 1080) and prod.SHORT_SIZE == (1080, 1920)
    cmd = ex._encoder_cmd((1920, 1080), 30, 21, "o.mp4", "a.wav", 10.0)
    assert cmd[cmd.index("-preset") + 1] == "veryfast" and cmd[cmd.index("-crf") + 1] == "21"
    assert "-r" in cmd and cmd[cmd.index("-r") + 1] == "30"


def test_reviewer_feedback_accepts_console_env_name(monkeypatch):
    monkeypatch.delenv("REVIEWER_FEEDBACK", raising=False)
    monkeypatch.setenv("DYPOL_REVIEWER_FEEDBACK", "Use a calmer tone.")
    assert gen.reviewer_feedback_text() == "Use a calmer tone."
