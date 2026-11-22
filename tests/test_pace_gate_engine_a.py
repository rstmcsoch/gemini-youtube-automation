"""Engine A pace fix (22 Nov 2026): a Short shipped at 139 WPM (rule 150-165). The voice no longer
carries a sticky speed between slides, each slide's second correction is a secant step through its
own two measurements (max 2 re-renders), and the finished Short is gated: under 150 WPM -> atempo
towards 157 (cap x1.12), still out of band -> QC hard failure "pace_out_of_band"."""
import json
import subprocess

import numpy as np
import pytest

from src import explainer as ex
from src import production as prod
from src import qc
from src import short_rules as sr
from src import voice


class _NonLinearKokoro:
    """WPM grows sub-linearly with speed (like the real voice), so one proportional step falls short."""

    def __init__(self):
        self.calls = []

    def create(self, text, voice=None, speed=1.0, lang=None):
        self.calls.append(speed)
        rate = 135.0 * (speed / 1.08) ** 0.6
        seconds = len(text.split()) / rate * 60.0
        return np.zeros(int(seconds * 24000), dtype="float32"), 24000


def test_no_sticky_speed_between_slides(monkeypatch, tmp_path):
    fake = _NonLinearKokoro()
    monkeypatch.setattr(voice, "_engine", lambda cfg: fake)
    monkeypatch.setenv("KOKORO_SPEED", "1.08")
    voice.reset()
    text = " ".join(["word"] * 120)
    voice.kokoro_speech(text, tmp_path / "a.wav")
    first = len(fake.calls)
    voice.kokoro_speech(text, tmp_path / "b.wav")
    assert fake.calls[first] == 1.08                      # second slide starts at the configured speed
    assert "speed" not in voice._STATE


def test_secant_correction_reaches_band_within_two_rerenders(monkeypatch, tmp_path):
    fake = _NonLinearKokoro()
    monkeypatch.setattr(voice, "_engine", lambda cfg: fake)
    monkeypatch.setenv("KOKORO_SPEED", "1.08")
    meta = voice.kokoro_speech(" ".join(["word"] * 120), tmp_path / "a.wav")
    assert len(fake.calls) <= 3                           # 1 render + max 2 corrective re-renders
    assert 150 <= meta["wpm"] <= 165
    # the third speed is the secant through the slide's own two measurements, not proportional
    cfg = voice.settings()
    s0, s1 = fake.calls[0], fake.calls[1]
    w = lambda s: 135.0 * (s / 1.08) ** 0.6
    assert fake.calls[2] == voice.secant_speed(s0, w(s0), s1, w(s1), cfg)


def test_secant_speed_hits_target_on_a_line():
    cfg = voice.settings()
    # WPM = 100 * speed + 30 -> 157 at speed 1.27
    assert voice.secant_speed(1.0, 130.0, 1.1, 140.0, cfg) == pytest.approx(1.27, abs=0.002)


def _tone(path, seconds):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    f"sine=frequency=300:duration={seconds}", "-ar", "48000", str(path)], check=True)
    return path


def test_short_gate_lifts_139_wpm_into_band_and_carries_timings(tmp_path):
    words = 139                                           # 139 words in 60 s = 139 WPM
    raw = _tone(tmp_path / "v.wav", 60.0)
    script = " ".join(["word"] * words)
    voice.save_word_timings(raw, script, [("word", 10.0, 10.3)] * words)
    _, out = prod.fit_short_audio(lambda t, p: raw, script, "", tmp_path)
    pace = prod.last_short_pace()
    assert sr.wpm(words, ex.audio_duration(out)) >= 150 and pace["wpm"] >= 150 and pace["in_band"]
    assert pace["tempo"] == pytest.approx(min(1.12, 157 / 139), abs=1e-3)
    data = json.loads(voice.timings_path(out).read_text())
    assert data["words"][0][1] < 10.0                     # timings scaled with the tempo
    assert not qc.hard_failures({"videos": {}, "short_pace": pace})


def test_short_too_slow_for_capped_tempo_is_a_hard_failure(tmp_path):
    words = 120                                           # 120 WPM: x1.12 only reaches ~134 WPM
    raw = _tone(tmp_path / "v.wav", 60.0)
    _, out = prod.fit_short_audio(lambda t, p: raw, " ".join(["word"] * words), "", tmp_path)
    pace = prod.last_short_pace()
    assert pace["tempo"] == pytest.approx(1.12) and pace["in_band"] is False
    fails = qc.hard_failures({"videos": {}, "short_pace": pace})
    assert any(f.startswith("pace_out_of_band") for f in fails)
    # without an explicit short_pace, QC reads the last fit_short_audio when a Short was analysed
    fails = qc.hard_failures({"videos": {"short": {"width": 1080, "height": 1920}}})
    assert any(f.startswith("pace_out_of_band") for f in fails)


def test_out_of_band_sets_needs_fixes(tmp_path, monkeypatch):
    monkeypatch.setattr(qc, "contact_sheet", lambda *a, **k: None)
    report = qc.run(None, None, tmp_path, extra={"short_pace": {"wpm": 139.0, "in_band": False}})
    assert report["verdict"] == "needs_fixes"
    assert any(f.startswith("pace_out_of_band") for f in report["hard_failures"])
