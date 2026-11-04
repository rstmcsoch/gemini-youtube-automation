"""Strong-model script, topic relevance and hard QC failures (Engine A, 4 Nov 2026).

On 3 Nov the Gemini 3.8/3.6 models returned 503 and flash-lite wrote the depth rewrite (Engine A) and an
off-topic Short script (Engine B). Weak models no longer write scripts, an off-topic script is retried and
then fails as "script model unavailable", and hard QC failures mark the draft "needs fixes".
"""
import json

import pytest

import src.generator as gen
from src import qc
from src import topic_guard as tg
from src.failure import failure_category
from dypol_control import PhasePolicy, execute_phase, record_upload_results

# Engine B draft 6 (3 Nov, flash-lite): a history lesson with no AI mechanism.
EARBUDS_OFF_TOPIC = (
    "Your earbuds generate sound waves to cancel background noise 50 times a second. Active noise control works "
    "by adding a second sound designed specifically to cancel unwanted environmental sound. The concept dates "
    "back to the late 1930s. Later developmental work that began in the 1950s eventually resulted in aviation "
    "headsets. The technology became available in the late 1980s. Modern earbuds use microphones, amplifiers, "
    "and digital signal processing to detect surrounding sound. The newer iteration incorporates the H2 chip "
    "and Bluetooth 5.3 connectivity. Hardware miniaturization will push noise cancellation into every device.")
# Engine B draft 7 (3 Nov, gemini-3.6-flash): explains the mechanism.
PHOTOS_ON_TOPIC = (
    "Your phone automatically analyzes visual features across your entire photo library. First, deep neural "
    "networks detect facial bounding boxes and pinpoint facial landmarks. The model converts those measurements "
    "into vector embeddings. Next, machine learning algorithms cluster matching faces together, face-sorting "
    "your library. But here is the "
    "catch: clustering models still struggle with heavy shadows and profile angles, misgrouping strangers.")
CAR_TEXT = ("Self-driving cars fuse camera, radar and lidar data. A neural network model detects pedestrians and "
            "predicts their path, and the planner steers the car. The AI was trained on millions of miles.")


def _lesson(text, short=None):
    return {"hook": "", "long_form_slides": [{"title": "t", "content": text}] * 3,
            "short_script": short if short is not None else text}


class _Resp:
    def __init__(self, text):
        self.text = text


class _Err(Exception):
    def __init__(self, code=503):
        super().__init__("503 UNAVAILABLE")
        self.code = code


class _Client:
    def __init__(self, outcomes):
        self.outcomes = {k: list(v) for k, v in outcomes.items()}
        self.calls = []
        self.models = self

    def generate_content(self, model, contents):
        self.calls.append(model)
        queue = self.outcomes.get(model)
        outcome = queue.pop(0) if queue else _Err()
        if isinstance(outcome, Exception):
            raise outcome
        return _Resp(outcome)


@pytest.fixture
def models(monkeypatch):
    waits = []
    monkeypatch.setattr(gen, "_sleep", waits.append)
    monkeypatch.setenv("GEMINI_TEXT_MODEL", "gemini-3.8-flash")
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "gemini-3.6-flash,gemini-3.5-flash-lite")
    monkeypatch.delenv("GEMINI_RETRY_DELAYS", raising=False)
    monkeypatch.delenv("DYPOL_TOPIC_CHECK", raising=False)
    monkeypatch.delenv("DYPOL_SCRIPT_BUDGET_S", raising=False)
    return waits


# ------------------------------------------------------------------ topic relevance

def test_relevance_flags_the_ai_free_earbuds_script_and_passes_a_real_explainer():
    assert tg.relevance("How noise-cancelling earbuds use AI", EARBUDS_OFF_TOPIC)["verdict"] == tg.OFF_TOPIC
    assert tg.relevance("How AI sorts your photos by face", PHOTOS_ON_TOPIC)["verdict"] == tg.ON_TOPIC
    off = tg.relevance("How AI powers self-driving cars", "Boil pasta in salted water for nine minutes.")
    assert off["verdict"] == tg.OFF_TOPIC and "driv" in " ".join(off["missing"])


def test_relevance_borderline_is_settled_by_a_model_verdict():
    r = tg.relevance("How AI writes code from a prompt quickly", "The AI model reads your prompt. The model "
                     "predicts each token of code, trained on public code.")
    assert r["verdict"] == tg.BORDERLINE
    assert tg.apply_judge(r, tg.parse_judge('{"on_topic": true, "reason": "explains it"}'))["verdict"] == tg.ON_TOPIC
    assert tg.apply_judge(r, tg.parse_judge('```json\n{"on_topic": false, "reason": "x"}\n```'))["verdict"] \
        == tg.OFF_TOPIC
    assert tg.parse_judge("no json") is None


def test_weak_models_are_recognised():
    assert tg.is_weak_model("gemini-3.5-flash-lite") and not tg.is_weak_model("gemini-3.6-flash")
    assert not tg.is_weak_model("gemini-3.8-flash")
    assert tg.strong_models(["gemini-3.8-flash", "gemini-3.5-flash-lite"]) == ["gemini-3.8-flash"]


# ------------------------------------------------------------------ strong model only

def test_strong_only_never_calls_the_lite_model(models):
    client = _Client({"gemini-3.5-flash-lite": ["weak script"] * 9})
    with pytest.raises(gen.LLMUnavailableError):
        gen._generate_text_with_fallback(client, "p", strong_only=True)
    assert "gemini-3.5-flash-lite" not in client.calls and client.calls
    # without strong_only the lite model still answers other tasks
    assert gen._generate_text_with_fallback(_Client({"gemini-3.5-flash-lite": ["ok"]}), "p").text == "ok"


def test_script_fails_clearly_when_strong_models_stay_down(models, monkeypatch):
    now = {"t": 0.0}
    monkeypatch.setattr(gen, "_clock", lambda: now["t"])
    monkeypatch.setattr(gen, "_sleep", lambda s: now.__setitem__("t", now["t"] + s))
    monkeypatch.setattr(gen, "ensure_depth", lambda client, content, prompt, topic=None: content)
    client = _Client({"gemini-3.5-flash-lite": [json.dumps(_lesson(CAR_TEXT))] * 50})
    with pytest.raises(gen.ScriptModelUnavailableError) as info:
        gen._strong_on_topic_script(client, "PROMPT", "How AI powers self-driving cars")
    assert "script model unavailable" in str(info.value)
    assert failure_category(info.value) == "llm_unavailable"
    assert "gemini-3.5-flash-lite" not in client.calls
    assert 400 <= now["t"] <= 600              # retried with backoff for about 10 minutes, then gave up
    assert client.calls.count("gemini-3.8-flash") >= 8   # several whole attempts, never a weak model


def test_off_topic_script_is_retried_with_a_correction(models, monkeypatch):
    monkeypatch.setattr(gen, "ensure_depth", lambda client, content, prompt, topic=None: content)
    client = _Client({"gemini-3.8-flash": [json.dumps(_lesson(EARBUDS_OFF_TOPIC)),
                                           json.dumps(_lesson(EARBUDS_OFF_TOPIC + " " + CAR_TEXT * 3))]})
    seen = []
    real = gen._generate_text_with_fallback

    def spy(client_, prompt, **kw):
        seen.append(prompt)
        return real(client_, prompt, **kw)

    gen_spy = pytest.MonkeyPatch()
    gen_spy.setattr(gen, "_generate_text_with_fallback", spy)
    try:
        out = gen._strong_on_topic_script(client, "PROMPT", "How noise-cancelling earbuds use AI")
    finally:
        gen_spy.undo()
    assert out["_script_meta"]["model"] == "gemini-3.8-flash" and out["_script_meta"]["attempts"] == 2
    assert out["_script_meta"]["relevance"]["verdict"] != tg.OFF_TOPIC
    assert "TOPIC CORRECTION" not in seen[0] and "TOPIC CORRECTION" in seen[1]


def test_off_topic_three_times_fails_the_draft(models, monkeypatch):
    monkeypatch.setattr(gen, "ensure_depth", lambda client, content, prompt, topic=None: content)
    client = _Client({"gemini-3.8-flash": [json.dumps(_lesson(EARBUDS_OFF_TOPIC))] * 5})
    with pytest.raises(gen.ScriptModelUnavailableError) as info:
        gen._strong_on_topic_script(client, "PROMPT", "How noise-cancelling earbuds use AI")
    assert "off-topic" in str(info.value) and client.calls.count("gemini-3.8-flash") == 3


def test_depth_rewrite_uses_strong_models_only(models, monkeypatch):
    seen = {}

    def fake(client, prompt, task_name="", **kw):
        seen.update(kw)
        raise gen.LLMUnavailableError("down")

    monkeypatch.setattr(gen, "_generate_text_with_fallback", fake)
    thin = _lesson("word " * 20)
    assert gen.ensure_depth(object(), thin, "prompt", topic="x") is thin   # first script kept
    assert seen.get("strong_only") is True


# ------------------------------------------------------------------ hard QC failures

def _report(long=(1920, 1080), short=(1080, 1920), freezes=()):
    return {"videos": {"long": {"width": long[0], "height": long[1], "freezes": list(freezes)},
                       "short": {"width": short[0], "height": short[1], "freezes": []}}}


def test_hard_failures_resolution_static_and_off_topic():
    assert qc.hard_failures(_report()) == []
    assert qc.hard_failures(_report(long=(1280, 720), short=(720, 1280))) == []
    low = qc.hard_failures(_report(long=(854, 480)))
    assert low and "resolution 854x480" in low[0]
    assert qc.hard_failures(_report(long=(3840, 2160)))                     # never above 1080p
    assert qc.hard_failures(_report(short=(1920, 1080)))                    # a landscape Short
    assert qc.hard_failures(_report(freezes=[{"start": 10.0, "end": 13.0, "duration": 3.0}])) == []
    static = qc.hard_failures(_report(freezes=[{"start": 10.0, "end": 14.2, "duration": 4.2}]))
    assert static and "static for 4.2 s" in static[0]
    off = qc.hard_failures(_report(), relevance={"verdict": "off_topic", "reasons": ["long: no AI"]})
    assert off and off[0].startswith("off-topic script")


def test_freeze_window_is_never_longer_than_the_hard_limit():
    assert qc._freeze_min_s() <= qc.HARD_STATIC_S


def test_result_marks_needs_fixes(tmp_path):
    q = {"verdict": "needs_fixes", "hard_failures": ["long: resolution 854x480 is outside 1280x720-1920x1080"]}
    out = record_upload_results([], tmp_path / "r.json", job_id="j", status="draft_only", quality=q)
    assert out["needs_fixes"] is True and out["status"] == "draft_only" and out["quality"] == q
    assert json.loads((tmp_path / "r.json").read_text())["needs_fixes"] is True
    ok = record_upload_results([], tmp_path / "r2.json", job_id="j", status="draft_only")
    assert "needs_fixes" not in ok


def test_full_phase_never_uploads_a_draft_that_needs_fixes(tmp_path, monkeypatch):
    monkeypatch.setenv("DYPOL_TOPIC", "How AI powers self-driving cars")
    policy = PhasePolicy(phase="full", allow_generate=True, allow_upload=True, review_required=False,
                         auto_publish=True, visibility="private", job_id="job-x")
    uploads = []
    candidate = {"lesson_title": "How AI powers self-driving cars", "needs_fixes": True,
                 "quality": {"verdict": "needs_fixes", "hard_failures": ["long: picture static for 5.0 s"]},
                 "assets": [{"kind": "long", "path": "x.mp4", "title": "T", "description": "", "tags": ""}]}
    result = execute_phase(policy, generate_fn=lambda lesson: dict(candidate),
                           upload_fn=lambda asset, p: uploads.append(asset) or {"youtube_id": "y"},
                           plan={"lessons": []}, candidate_root=tmp_path / "c", results_path=tmp_path / "r.json")
    assert uploads == [] and result["status"] == "draft_only" and result["needs_fixes"] is True
