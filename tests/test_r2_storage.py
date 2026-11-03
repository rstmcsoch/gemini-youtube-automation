"""R2 draft storage (3 Nov 2026): resolution rule, review copy settings, captions, markers, workflow wiring."""
import importlib.util
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("dypol_r2", ROOT / "scripts" / "dypol_r2.py")
r2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r2)


def test_resolution_window():
    assert r2.check_resolution("long", {"w": 1920, "h": 1080})[0]
    assert r2.check_resolution("long", {"w": 1280, "h": 720})[0]
    assert not r2.check_resolution("long", {"w": 2560, "h": 1440})[0]
    assert not r2.check_resolution("long", {"w": 854, "h": 480})[0]
    assert r2.check_resolution("short", {"w": 1080, "h": 1920})[0]
    assert r2.check_resolution("short", {"w": 720, "h": 1280})[0]
    assert not r2.check_resolution("short", {"w": 1440, "h": 2560})[0]
    assert not r2.check_resolution("short", {"w": 1920, "h": 1080})[0]  # wrong orientation
    assert not r2.check_resolution("short", {"w": 0, "h": 0})[0]


def test_review_copy_never_upscales():
    assert r2.preview_filter("long", {"w": 1920, "h": 1080})[1:] == (960, 540)
    assert r2.preview_filter("short", {"w": 1080, "h": 1920})[1:] == (720, 1280)
    assert r2.preview_filter("long", {"w": 800, "h": 450})[1:] == (800, 450)
    assert r2.preview_filter("short", {"w": 540, "h": 960})[1:] == (540, 960)


def test_review_copy_encoder_settings(monkeypatch):
    seen = {}
    monkeypatch.setattr(r2.subprocess, "run", lambda cmd, **kw: seen.setdefault("cmd", cmd))
    r2.make_preview("in.mp4", "out.mp4", "long", {"w": 1920, "h": 1080, "fps": 30})
    cmd = " ".join(seen["cmd"])
    for part in ("-crf 28", "-preset veryfast", "+faststart", "-g 30", "expr:gte(t,n_forced*1)", "libx264"):
        assert part in cmd


def test_srt_to_vtt():
    vtt = r2.srt_to_vtt("1\n00:00:00,000 --> 00:00:01,500\nHi\n\n2\n00:00:01,500 --> 00:00:03,000\nThere\n")
    assert vtt.startswith("WEBVTT\n\n00:00:00.000 --> 00:00:01.500\nHi")
    assert "\n2\n" not in vtt


def test_qc_markers_both_shapes():
    a = r2.qc_markers({"videos": {"long": {"silences": [{"start": 3, "end": 5}], "freezes": [{"start": 9, "end": 16}]}}})
    assert [m["kind"] for m in a["long"]] == ["silence", "freeze"]
    b = r2.qc_markers({"black": [{"start": 1.0, "end": 1.6}]})
    assert b["short"][0]["t"] == 1.0


def test_chapters_from_storyboard(tmp_path):
    p = tmp_path / "storyboard.json"
    p.write_text(json.dumps({"timeline": {"slides": [{"start": 0}, {"start": 0.4}, {"start": 30}]},
                             "slides": [{"title": "Intro"}, {"title": "x"}, {"title": "Main"}]}))
    assert r2.chapters_from_storyboard(p) == [{"t": 0.0, "title": "Intro"}, {"t": 30.0, "title": "Main"}]


def test_upload_failure_never_fails_the_draft(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise RuntimeError("network down")
    monkeypatch.setattr(r2, "build", boom)
    assert r2.main(["--engine", "gemini", "--job-id", "job-x", "--root", str(tmp_path)]) == 0


def test_check_mode_policy(monkeypatch, tmp_path):
    monkeypatch.setattr(r2, "discover", lambda e, r: {"long": "a.mp4", "short": None})
    monkeypatch.setattr(r2, "probe", lambda p: {"w": 2560, "h": 1440, "duration": 1, "fps": 30})
    monkeypatch.setenv("DYPOL_RESOLUTION_POLICY", "fail")
    assert r2.main(["--engine", "gemini", "--job-id", "j", "--root", str(tmp_path), "--check"]) == 3
    monkeypatch.setenv("DYPOL_RESOLUTION_POLICY", "warn")
    assert r2.main(["--engine", "gemini", "--job-id", "j", "--root", str(tmp_path), "--check"]) == 0


def test_workflow_wires_r2_with_secrets_and_fallback():
    wf = next((ROOT / ".github" / "workflows").glob("main.yml"), None) or ROOT / ".github" / "workflows" / "verticals.yml"
    text = wf.read_text()
    assert "scripts/dypol_r2.py" in text and "--check" in text
    data = yaml.safe_load(text)
    steps = [s for job in data["jobs"].values() for s in job.get("steps", [])]
    store = [s for s in steps if "R2" in str(s.get("name", ""))]
    assert store and store[0].get("continue-on-error") is True
    env = store[0]["env"]
    for k in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET"):
        assert env[k] == "${{ secrets.%s }}" % k
    # The artifact fallback is still there.
    assert any("preview-" in str((s.get("with") or {}).get("name", "")) for s in steps)
