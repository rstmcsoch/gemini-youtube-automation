"""Console voice + design pattern selection (contract 5 Nov 2026)."""
import json

import numpy as np
import pytest

from src import voice

AVAILABLE = ["am_michael", "am_adam", "af_bella"]


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    voice.reset()
    for name in ("DYPOL_VOICE", "KOKORO_VOICE", "DYPOL_TTS_PROVIDER", "KOKORO_SPEED"):
        monkeypatch.delenv(name, raising=False)
    yield
    voice.reset()


def test_single_valid_id_is_kept():
    assert voice.resolve_voice("am_adam", AVAILABLE) == ("am_adam", [("am_adam", 1.0)])


def test_blend_is_parsed_and_normalised():
    name, parts = voice.resolve_voice("am_michael:0.6, am_adam:0.4", AVAILABLE)
    assert name == "am_michael:0.6,am_adam:0.4"
    assert parts == [("am_michael", pytest.approx(0.6)), ("am_adam", pytest.approx(0.4))]
    _, parts = voice.resolve_voice("am_michael:3,am_adam:1", AVAILABLE)
    assert [round(w, 2) for _, w in parts] == [0.75, 0.25]


@pytest.mark.parametrize("bad", ["zz_nobody", "am_michael:0.5,zz_nobody:0.5", "am_michael:x",
                                 "am_michael:0", ",", "am_adam:0.5,am_adam:0.5", ""])
def test_invalid_voice_logs_and_falls_back(bad, capsys):
    assert voice.resolve_voice(bad, AVAILABLE) == ("am_michael", [("am_michael", 1.0)])
    assert "VOICE_INVALID" in capsys.readouterr().out


def test_dypol_voice_env_wins_over_kokoro_voice(monkeypatch):
    monkeypatch.setenv("KOKORO_VOICE", "af_bella")
    monkeypatch.setenv("DYPOL_VOICE", "am_adam")
    assert voice.settings()["voice"] == "am_adam"


class _Fake:
    def __init__(self):
        self.calls = []

    def get_voices(self):
        return AVAILABLE

    def get_voice_style(self, name):
        return np.full((4,), {"am_michael": 1.0, "am_adam": 3.0}.get(name, 0.0), dtype="float32")

    def create(self, text, voice=None, speed=1.0, lang=None):
        self.calls.append(voice)
        seconds = len(text.split()) / 157.0 * 60.0
        return np.zeros(int(seconds * 24000), dtype="float32"), 24000


def test_kokoro_speech_blends_styles_and_records_voice(monkeypatch, tmp_path):
    fake = _Fake()
    monkeypatch.setattr(voice, "_engine", lambda cfg: fake)
    monkeypatch.setenv("DYPOL_VOICE", "am_michael:0.5,am_adam:0.5")
    meta = voice.kokoro_speech(" ".join(["word"] * 20), tmp_path / "a.wav")
    assert meta["voice"] == "am_michael:0.5,am_adam:0.5"
    assert np.allclose(fake.calls[0], 2.0)


def test_kokoro_speech_invalid_env_uses_am_michael(monkeypatch, tmp_path):
    fake = _Fake()
    monkeypatch.setattr(voice, "_engine", lambda cfg: fake)
    monkeypatch.setenv("DYPOL_VOICE", "not_a_voice")
    meta = voice.kokoro_speech(" ".join(["word"] * 20), tmp_path / "a.wav")
    assert meta["voice"] == "am_michael" and fake.calls == ["am_michael"]


def test_design_pattern_registry_and_fallback(capsys):
    import main
    entries = json.loads((main.DESIGN_PATTERNS_DIR / "index.json").read_text())
    assert {e["id"]: e["status"] for e in entries} == {"elements": "ready", "characters": "coming_soon"}
    assert (main.DESIGN_PATTERNS_DIR / "elements.md").exists()
    assert main.resolve_design_pattern("elements") == "elements"
    assert main.resolve_design_pattern("") == "elements"
    assert main.resolve_design_pattern("characters") == "elements"
    assert main.resolve_design_pattern("nope") == "elements"
    assert capsys.readouterr().out.count("DESIGN_PATTERN_FALLBACK") == 2


def test_workflow_carries_voice_pattern_and_privacy_inputs():
    import yaml
    from pathlib import Path
    wf = yaml.safe_load((Path(__file__).resolve().parent.parent / ".github/workflows/main.yml").read_text())
    inputs = wf[True]["workflow_dispatch"]["inputs"]
    for name in ("voice", "design_pattern", "privacy_status", "privacy"):
        assert inputs[name]["required"] is False
    assert len(inputs) <= 25
    env = next(s for s in wf["jobs"]["build-and-deploy"]["steps"] if s.get("id") == "production-run")["env"]
    assert env["DYPOL_VOICE"] == "${{ inputs.voice || vars.DYPOL_DEFAULT_VOICE || 'am_michael' }}"
    assert env["DYPOL_PATTERN"] == "${{ inputs.design_pattern || vars.DYPOL_DEFAULT_PATTERN || 'elements' }}"
    assert env["YOUTUBE_PRIVACY_STATUS"] == "${{ inputs.privacy_status || inputs.privacy || 'unlisted' }}"
