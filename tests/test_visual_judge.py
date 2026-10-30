"""Unit tests for the AI visual judge (HTTP layer mocked; no network)."""
import json
from pathlib import Path

import pytest
from PIL import Image

vj = pytest.importorskip("src.visual_judge")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class Clock:
    def __init__(self):
        self.t = 1000.0
        self.sleeps = []

    def now(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(round(s, 3))
        self.t += s


def ok(verdict, model_text=None):
    text = model_text if model_text is not None else json.dumps(verdict)
    return 200, {}, json.dumps({"choices": [{"message": {"content": text}}]})


GOOD = {"description": "server racks", "subject_match": "yes", "issues": [], "reason": "fits", "score": 8}
MEH = {"description": "circuit board", "subject_match": "partial", "issues": ["generic_filler"],
       "reason": "filler", "score": 5}
BAD_SUBJECT = {"description": "a kitten", "subject_match": "no", "issues": ["wrong_subject"],
               "reason": "cat", "score": 2}
WATERMARK = {"description": "stock photo", "subject_match": "yes", "issues": ["watermark_or_text"],
             "reason": "watermark", "score": 8}
BLURRY = {"description": "blur", "subject_match": "partial", "issues": ["low_quality"], "reason": "blurry",
          "score": 3}


class Transport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.bodies = []

    def __call__(self, url, headers, body, timeout):
        assert url == vj.API_URL
        assert timeout == vj.REQUEST_TIMEOUT
        assert headers["Authorization"].startswith("Bearer ")
        self.bodies.append(json.loads(body))
        r = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(r, Exception):
            raise r
        return r


def make_judge(responses, models="m1,m2,m3", store=None, **env):
    clock = Clock()
    transport = Transport(responses)
    environment = {"NVIDIA_API_KEY": "nvapi-test", "VISUAL_JUDGE_MODELS": models, **env}
    judge = vj.VisualJudge(env=environment, transport=transport, sleep=clock.sleep, clock=clock.now,
                           store=store or vj.JudgeStore(vj.MemoryBackend()))
    judge.rng.seed(7)
    return judge, transport, clock


def image(tmp_path, name="a.png", color=(30, 90, 160), size=(800, 600)):
    path = Path(tmp_path) / name
    Image.new("RGB", size, color).save(path)
    return path


# ---------------------------------------------------------------------------
# decision rule
# ---------------------------------------------------------------------------

def test_decision_rule():
    assert vj.decide(vj.normalize_verdict(GOOD)) == vj.PASS
    assert vj.decide(vj.normalize_verdict(MEH)) == vj.MARGINAL
    assert vj.decide(vj.normalize_verdict(BAD_SUBJECT)) == vj.FAIL
    # hard-fail issue beats a high score, and the score is clamped to <= 3
    w = vj.normalize_verdict(WATERMARK)
    assert w["decision"] == vj.FAIL and w["score"] == 3
    # high score but subject_match == "no" is not a PASS (bias strict)
    assert vj.decide({"score": 9, "subject_match": "no", "issues": []}) == vj.FAIL
    assert vj.decide({"score": 7, "subject_match": "partial", "issues": ["cluttered"]}) == vj.PASS
    assert vj.decide({"score": 5, "subject_match": "no", "issues": ["off_topic"]}) == vj.MARGINAL
    assert vj.decide({"score": 4, "subject_match": "yes", "issues": []}) == vj.FAIL
    assert vj.decide({"score": 6, "subject_match": "yes", "issues": ["sensitive_content"]}) == vj.FAIL
    assert vj.decide({"score": 6, "subject_match": "partial", "issues": ["generic_filler"]}) == vj.PASS
    assert vj.decide({"score": 6, "subject_match": "partial", "issues": []}) == vj.PASS
    assert vj.decide({"score": 6, "subject_match": "no", "issues": []}) == vj.MARGINAL
    assert vj.decide({"score": 6, "subject_match": "partial", "issues": ["off_topic"]}) == vj.MARGINAL
    assert vj.decide({"score": 5, "subject_match": "partial", "issues": ["generic_filler"]}) == vj.MARGINAL


# ---------------------------------------------------------------------------
# JSON hygiene
# ---------------------------------------------------------------------------

def test_parse_fences_prose_and_think_blocks():
    fenced = "```json\n" + json.dumps(GOOD) + "\n```"
    assert vj.extract_json(fenced)["score"] == 8
    prose = 'Sure! Here is my verdict: {"description": "a {weird} brace", "subject_match": "yes", ' \
            '"issues": [], "reason": "ok", "score": 9} Hope that helps.'
    assert vj.extract_json(prose)["score"] == 9
    think = "<think>reasoning {not json}</think>" + json.dumps(MEH)
    assert vj.extract_json(think)["score"] == 5
    assert vj.extract_json("no json here") is None
    assert vj.normalize_verdict({"score": "N/A"}) is None


def test_normalize_filters_unknown_issues_and_clamps():
    v = vj.normalize_verdict({"description": "x", "subject_match": "Partially", "issues": ["bogus", "cluttered"],
                              "reason": "r", "score": 14})
    assert v["issues"] == ["cluttered"] and v["score"] == 10 and v["subject_match"] == "partial"


def test_parse_failure_retried_once_then_gives_up(tmp_path):
    judge, transport, _ = make_judge([ok(None, "I cannot comply"), ok(GOOD)], models="m1")
    v = judge.judge_image(image(tmp_path), "t", "line", "query", "px:1")
    assert v["decision"] == vj.PASS and len(transport.bodies) == 2 and judge.parse_failures == 1
    judge2, transport2, _ = make_judge([ok(None, "garbage")], models="m1")
    assert judge2.judge_image(image(tmp_path, "b.png"), "t", "l", "q", "px:2") is None
    assert len(transport2.bodies) == 2


def test_request_payload_shape(tmp_path):
    judge, transport, _ = make_judge([ok(GOOD)])
    judge.judge_image(image(tmp_path), "Video", "Narration line", "data center", "px:1")
    body = transport.bodies[0]
    assert body["temperature"] == 0 and body["response_format"] == {"type": "json_object"}
    assert body["messages"][0]["role"] == "system"
    parts = body["messages"][1]["content"]
    assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert "NARRATION LINE: \"Narration line\"" in parts[0]["text"]
    assert len([p for p in parts if p["type"] == "image_url"]) == 1  # one image per request


def test_response_format_unsupported_falls_back_silently(tmp_path):
    judge, transport, _ = make_judge([(400, {}, "response_format not supported"), ok(GOOD)])
    v = judge.judge_image(image(tmp_path), "t", "l", "q", "px:1")
    assert v["decision"] == vj.PASS
    assert "response_format" in transport.bodies[0] and "response_format" not in transport.bodies[1]


# ---------------------------------------------------------------------------
# pacing, backoff, Retry-After
# ---------------------------------------------------------------------------

def test_pacing_min_interval(tmp_path):
    judge, transport, clock = make_judge([ok(GOOD)])
    judge.judge_image(image(tmp_path, "a.png", (10, 10, 10)), "t", "l", "q1", "a")
    judge.judge_image(image(tmp_path, "b.png", (200, 10, 10)), "t", "l", "q2", "b")
    assert clock.sleeps == [2.0]


def test_429_honours_retry_after(tmp_path):
    judge, transport, clock = make_judge([(429, {"Retry-After": "7"}, "slow down"), ok(GOOD)])
    v = judge.judge_image(image(tmp_path), "t", "l", "q", "px:1")
    assert v["decision"] == vj.PASS
    assert 7.0 in clock.sleeps


def test_429_backoff_schedule_with_jitter(tmp_path):
    judge, transport, clock = make_judge([(429, {}, "")] * 3 + [ok(GOOD)], models="m1")
    v = judge.judge_image(image(tmp_path), "t", "l", "q", "px:1")
    assert v["decision"] == vj.PASS
    backoffs = [s for s in clock.sleeps if s != 2.0]
    assert len(backoffs) == 3
    for got, base in zip(backoffs, (5, 15, 45)):
        assert base * 0.8 <= got <= base * 1.2


# ---------------------------------------------------------------------------
# circuit breaker + fallback chain
# ---------------------------------------------------------------------------

def test_circuit_opens_after_three_exhausted_requests(tmp_path):
    judge, transport, clock = make_judge([(429, {"Retry-After": "1"}, "")], models="m1,m2,m3")
    assert judge.judge_image(image(tmp_path), "t", "l", "q", "px:1") is None
    assert judge.status == "circuit_open"
    assert "3 consecutive" in judge.circuit_reason
    s = judge.summary()
    assert s["needs_review"] and s["circuit"]["state"] == "open"
    calls = len(transport.bodies)
    assert judge.judge_image(image(tmp_path, "z.png"), "t", "l", "q", "px:2") is None
    assert len(transport.bodies) == calls  # skipped for the rest of the run


def test_any_402_opens_circuit(tmp_path):
    judge, transport, _ = make_judge([(402, {}, "payment required")])
    assert judge.judge_image(image(tmp_path), "t", "l", "q", "px:1") is None
    assert judge.status == "circuit_open" and "402" in judge.circuit_reason
    assert len(transport.bodies) == 1


def test_five_parse_failures_open_circuit(tmp_path):
    judge, transport, _ = make_judge([ok(None, "nope")], models="m1")
    for i in range(3):
        judge.judge_image(image(tmp_path, f"{i}.png", (i * 40, 0, 0)), "t", "l", "q", f"px:{i}")
    assert judge.status == "circuit_open" and "parse" in judge.circuit_reason
    assert judge.parse_failures == 5


def test_404_walks_fallback_chain_and_logs_model(tmp_path):
    judge, transport, _ = make_judge([(404, {}, "model not found"), ok(GOOD)], models="m1,m2")
    v = judge.judge_image(image(tmp_path), "t", "l", "q", "px:1")
    assert v["judge_model"] == "m2"
    assert [b["model"] for b in transport.bodies] == ["m1", "m2"]
    assert "m1" in judge.dead_models
    judge.judge_image(image(tmp_path, "b.png", (0, 200, 0)), "t", "l", "q", "px:2")
    assert transport.bodies[-1]["model"] == "m2"  # dropped model is not retried
    assert judge.summary()["models_used"] == {"m2": 2}


def test_persistent_429_on_primary_moves_to_fallback(tmp_path):
    judge, transport, _ = make_judge([(429, {"Retry-After": "1"}, "")] * 4 + [ok(GOOD)], models="m1,m2")
    v = judge.judge_image(image(tmp_path), "t", "l", "q", "px:1")
    assert v["judge_model"] == "m2" and judge.status == "active"


def test_all_models_gone_opens_circuit(tmp_path):
    judge, _, _ = make_judge([(404, {}, "")], models="m1,m2")
    assert judge.judge_image(image(tmp_path), "t", "l", "q", "px:1") is None
    assert judge.status == "circuit_open"


def test_network_errors_are_retried(tmp_path):
    judge, transport, _ = make_judge([TimeoutError("read timed out"), ok(GOOD)], models="m1")
    assert judge.judge_image(image(tmp_path), "t", "l", "q", "px:1")["decision"] == vj.PASS


# ---------------------------------------------------------------------------
# caps
# ---------------------------------------------------------------------------

def test_call_cap_per_video(tmp_path, monkeypatch):
    monkeypatch.setattr(vj, "MAX_CALLS_PER_VIDEO", 2)
    judge, transport, _ = make_judge([ok(GOOD)])
    for i in range(4):
        judge.judge_image(image(tmp_path, f"{i}.png", (i * 50, 9, 9)), "t", "l", "q", f"px:{i}")
    assert len(transport.bodies) == 2
    s = judge.summary()
    assert s["capped"] and s["needs_review"] and "call cap" in s["needs_review_reasons"][0]
    judge.begin_video("second")  # caps are per video
    assert judge.judge_image(image(tmp_path, "n.png", (1, 2, 250)), "t", "l", "q", "px:n") is not None


def test_wall_time_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(vj, "MAX_WALL_SECONDS_PER_VIDEO", 20.0)
    judge, transport, _ = make_judge([(429, {"Retry-After": "30"}, "")], models="m1")
    assert judge.judge_image(image(tmp_path), "t", "l", "q", "px:1") is None
    assert "wall-time" in judge.summary()["capped"]


# ---------------------------------------------------------------------------
# cache + rejection memory
# ---------------------------------------------------------------------------

def test_score_cache_hits_and_persists(tmp_path):
    store_path = tmp_path / "cache.json"
    judge, transport, _ = make_judge([ok(GOOD)], store=vj.JudgeStore(vj.JsonFileBackend(store_path)))
    img = image(tmp_path)
    judge.judge_image(img, "t", "l", "Data Center", "px:1")
    v = judge.judge_image(img, "t", "l", "center data", "px:1")  # same normalised query
    assert v["cached"] and len(transport.bodies) == 1 and judge.summary()["cache_hits"] == 1
    # persisted: a new judge over the same file is a cache hit with zero calls
    judge2, transport2, _ = make_judge([ok(MEH)], store=vj.JudgeStore(vj.JsonFileBackend(store_path)))
    assert judge2.judge_image(img, "t", "l", "data center", "px:1")["score"] == 8
    assert transport2.bodies == []
    # a different query is a different context -> miss
    judge2.judge_image(img, "t", "l", "kitten", "px:1")
    assert len(transport2.bodies) == 1


def test_cache_key_is_prompt_versioned(tmp_path):
    jpeg = vj.image_jpeg_bytes(image(tmp_path, size=(2000, 1000)))
    from PIL import Image as _I
    import io
    with _I.open(io.BytesIO(jpeg)) as im:
        assert max(im.size) == 512
    assert vj.score_cache_key(jpeg, "v1") != vj.score_cache_key(jpeg, "v2")


def test_global_hard_fail_cached_for_any_query(tmp_path):
    judge, transport, _ = make_judge([ok(WATERMARK)])
    img = image(tmp_path)
    judge.judge_image(img, "t", "l", "laptop", "px:9")
    v = judge.judge_image(img, "t", "l", "anything else", "px:9")
    assert v["cached"] and v["decision"] == vj.FAIL and len(transport.bodies) == 1


def test_rejection_memory_global_vs_per_query(tmp_path):
    judge, transport, _ = make_judge([ok(WATERMARK), ok(BAD_SUBJECT)])
    judge.judge_image(image(tmp_path, "w.png", (1, 1, 1)), "t", "l", "server room", "px:1")
    judge.judge_image(image(tmp_path, "k.png", (250, 250, 250)), "t", "l", "Cyber Security", "px:2")
    assert judge.is_rejected("px:1", "totally different").startswith("global:")
    assert judge.is_rejected("px:2", "security cyber").startswith("query:")
    assert judge.is_rejected("px:2", "kittens") is None
    assert judge.is_rejected("px:3", "server room") is None


def test_marginal_is_not_blacklisted(tmp_path):
    judge, _, _ = make_judge([ok(MEH)])
    judge.judge_image(image(tmp_path), "t", "l", "chip", "px:5")
    assert judge.is_rejected("px:5", "chip") is None


# ---------------------------------------------------------------------------
# disabled path
# ---------------------------------------------------------------------------

def test_disabled_without_key(tmp_path):
    calls = []
    judge = vj.VisualJudge(env={}, transport=lambda *a: calls.append(a))
    assert judge.status == "disabled" and not judge.active
    assert judge.judge_image(image(tmp_path), "t", "l", "q", "px:1") is None
    assert judge.is_rejected("px:1", "q") is None
    assert judge.check_contact_sheet([(str(image(tmp_path)), "x")]) is None
    s = judge.summary()
    assert s["judge_status"] == "disabled" and s["needs_review"] is False and calls == []


def test_explicitly_disabled_with_key():
    judge = vj.VisualJudge(env={"NVIDIA_API_KEY": "nvapi-x", "VISUAL_JUDGE_ENABLED": "0"})
    assert judge.status == "disabled"
    assert vj.VisualJudge(env={"NVIDIA_API_KEY": "nvapi-x"}).status == "active"


def test_env_config_defaults():
    judge = vj.VisualJudge(env={"NVIDIA_API_KEY": "k"})
    assert judge.models[0] == "nvidia/nemotron-3.5-super-vl-120b-a12b"
    assert judge.models[1:] == ["meta/llama-3.2-90b-vision-instruct", "meta/llama-3.2-11b-vision-instruct"]
    assert judge.min_interval == 2.0
    assert vj.VisualJudge(env={"NVIDIA_API_KEY": "k", "VISUAL_JUDGE_MIN_INTERVAL": "3.5"}).min_interval == 3.5


# ---------------------------------------------------------------------------
# clip frame selection
# ---------------------------------------------------------------------------

def _fake_frames(monkeypatch, tmp_path):
    fractions = []

    def fake_extract(clip, fraction, out, duration=None):
        fractions.append(fraction)
        Image.new("RGB", (64, 64), (int(fraction * 200), 10, 10)).save(out, format="JPEG")
        return Path(out)
    monkeypatch.setattr(vj, "extract_clip_frame", fake_extract)
    monkeypatch.setattr(vj, "probe_duration", lambda p: 10.0)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    return fractions, clip


def test_clip_judged_on_50pct_frame_only(monkeypatch, tmp_path):
    fractions, clip = _fake_frames(monkeypatch, tmp_path)
    judge, transport, _ = make_judge([ok(GOOD)])
    v = judge.judge_clip(clip, "t", "l", "q", "pxv:1")
    assert fractions == [0.5] and v["frame"] == "50%" and len(transport.bodies) == 1


def test_clip_low_quality_retries_25pct(monkeypatch, tmp_path):
    fractions, clip = _fake_frames(monkeypatch, tmp_path)
    judge, transport, _ = make_judge([ok(BLURRY), ok(GOOD)])
    v = judge.judge_clip(clip, "t", "l", "q", "pxv:1")
    assert fractions == [0.5, 0.25] and v["decision"] == vj.PASS and v["frame"] == "25%"
    assert judge.is_rejected("pxv:1", "q") is None


def test_clip_wrong_subject_rejects_whole_clip(monkeypatch, tmp_path):
    fractions, clip = _fake_frames(monkeypatch, tmp_path)
    judge, transport, _ = make_judge([ok(BAD_SUBJECT)])
    v = judge.judge_clip(clip, "t", "l", "q", "pxv:1")
    assert fractions == [0.5] and v["decision"] == vj.FAIL
    assert judge.is_rejected("pxv:1", "q").startswith("query:")


# ---------------------------------------------------------------------------
# shot loop + contact sheet + helpers
# ---------------------------------------------------------------------------

def test_shot_judging_keeps_best_marginal():
    shot = vj.ShotJudging(type("J", (), {"active": True})(), limit=5)
    assert shot.consider(vj.normalize_verdict(BAD_SUBJECT), "a") == "reject"
    assert shot.consider(dict(vj.normalize_verdict(MEH), score=5, decision="MARGINAL"), "b") == "keep_marginal"
    assert shot.consider(dict(vj.normalize_verdict(MEH), score=6, decision="MARGINAL"), "c") == "keep_marginal" and shot.displaced == "b"
    assert shot.consider(dict(vj.normalize_verdict(MEH), score=5, decision="MARGINAL"), "d") == "reject"
    assert shot.consider(vj.normalize_verdict(GOOD), "e") == "accept"
    assert shot.consider(None, "f") == "unjudged"
    assert shot.marginal[1] == "c" and shot.judged == 5 and not shot.can_judge_more()


def test_contact_sheet_numbered_grid(tmp_path):
    imgs = [str(image(tmp_path, f"{i}.png", (i * 15, 80, 120))) for i in range(14)]
    out = vj.compose_contact_sheet(imgs[:12], list(range(1, 13)), tmp_path / "sheet.jpg")
    with Image.open(out) as im:
        assert im.size == (4 * 384, 3 * 384)
    judge, transport, _ = make_judge([
        ok(None, json.dumps({"bad_shots": [{"n": 3, "reason": "kitten"}, {"n": 99, "reason": "x"}],
                             "overall": "needs_review"})),
        ok(None, json.dumps({"bad_shots": [], "overall": "ok"}))])
    result = judge.check_contact_sheet([(p, f"line {i}") for i, p in enumerate(imgs)], "Video", tmp_path, "long")
    assert result["sheets"] == 2 and result["bad_shots"] == [{"n": 3, "reason": "kitten"}]
    assert len(transport.bodies) == 2
    assert "1: \"line 0\"" in transport.bodies[0]["messages"][1]["content"][0]["text"]
    assert judge.summary()["needs_review"]


def test_broaden_query():
    assert vj.broaden_query("liquid cooling data center") == "data center"
    assert vj.broaden_query("quantum chip") == "chip"
    assert vj.broaden_query("server room") == "server"
    assert vj.broaden_query("server") is None


def test_verdict_fields_for_manifest():
    v = dict(vj.normalize_verdict(GOOD), judge_model="m1", prompt_version=vj.PROMPT_VERSION)
    rec = vj.verdict_fields(v, True, "best MARGINAL")
    assert set(rec) >= {"judge_model", "score", "subject_match", "issues", "reason", "decision",
                        "prompt_version", "needs_review", "review_reason"}
