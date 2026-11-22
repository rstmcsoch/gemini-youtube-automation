"""Calm SFX: no whooshes reachable, low soft chimes/swell only, LUFS-levelled >= 22 dB under the
voice (target 24), at most one cue per section, nothing for code panels, DYPOL_SFX=off switch."""
import inspect
import wave
from pathlib import Path

import numpy as np
import pytest

from src import sfx

RATE = 48000


def _speechlike(path, seconds=8.0):
    """Synthetic 'voice': harmonic 140 Hz buzz, syllable-modulated, with pauses."""
    t = np.arange(int(seconds * RATE)) / RATE
    buzz = sum(np.sin(2 * np.pi * 140 * h * t) / h for h in range(1, 12))
    syll = np.clip(np.sin(2 * np.pi * 4.0 * t), 0, None)
    gate = (np.sin(2 * np.pi * 0.35 * t) > -0.6).astype(float)
    x = buzz * syll * gate
    x = 0.3 * x / np.max(np.abs(x))
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes((x * 32767).astype("<i2").tobytes())
    return path


def _read(path):
    with wave.open(str(path)) as wf:
        ch, n = wf.getnchannels(), wf.getnframes()
        return np.frombuffer(wf.readframes(n), dtype="<i2").reshape(-1, ch) / 32768.0


def test_no_whoosh_reachable():
    assert "whoosh" not in inspect.getsource(sfx).lower()
    assert {v["category"] for v in sfx.CLIPS.values()} <= {"chime", "swell"}
    assert set(sfx.CUE_FOR_KIND.values()) <= {"chime", "swell"}
    for kind in ("code_panel", "diagram", "stock_or_ai_still", "icon_flow", None):
        assert kind not in sfx.CUE_FOR_KIND
    changes = [(3.0 + 9.0 * i, k) for i, k in enumerate(["code_panel", "diagram", "stock_or_ai_still"] * 6)]
    assert sfx.plan_transitions(changes, 200.0, seed=1) == []
    for seed in range(5):
        changes = [(3.0 + 8.0 * i, k) for i, k in
                   enumerate(["kinetic_text", "code_panel", "chapter_card", "diagram", "count_up"] * 6)]
        for e in sfx.plan_transitions(changes, 300.0, seed=seed):
            assert e["clip"] in sfx.CLIPS and sfx.CLIPS[e["clip"]]["category"] in {"chime", "swell"}


@pytest.mark.parametrize("clip", list(sfx.CLIPS))
def test_clips_low_and_soft(clip):
    y = sfx.synth(clip)
    spec = np.abs(np.fft.rfft(y)) ** 2
    f = np.fft.rfftfreq(len(y), 1.0 / sfx.RATE)
    assert spec[f > sfx.MAX_FREQ_HZ].sum() / spec.sum() < 1e-3           # nothing above 1.5 kHz
    first = np.max(np.abs(y[: int(sfx.MIN_ATTACK_S * sfx.RATE)]))
    assert first < 0.5 * np.max(np.abs(y))                               # attack >= 20 ms


def test_lufs_gap_at_least_22_db_on_synthetic_voice(tmp_path):
    voice = _speechlike(tmp_path / "v.wav")
    events = [{"t": 1.0, "clip": "chime_soft_02", "gain_db": 0.0}, {"t": 5.0, "clip": "swell_soft_01", "gain_db": 0.0}]
    out = sfx.build_track(events, 8.0, tmp_path / "s.wav", voice)
    gap = sfx.integrated_lufs(_read(voice)) - sfx.integrated_lufs(_read(out))
    assert 22.0 <= gap <= 27.0, gap
    from src import explainer as ex                                    # independent check (ffmpeg loudnorm)
    assert ex.measure_lufs(voice) - ex.measure_lufs(out) >= 22.0


def test_integrated_lufs_matches_reference_tone():
    t = np.arange(RATE * 5) / RATE
    assert sfx.integrated_lufs(0.1 * np.sin(2 * np.pi * 1000 * t)) == pytest.approx(-23.0, abs=0.2)


def test_at_most_one_cue_per_section():
    from src import storyboard as sb
    tl = sb.build_timeline([30.0, 30.0, 30.0], 3.0)
    kinds = ["kinetic_text", "count_up", "chapter_card"] * (len(tl["shots"]) // 3 + 1)
    kinds = kinds[: len(tl["shots"])]
    events = sfx.plan(tl, kinds, {}, seed=2)
    secs = sfx._sections(tl, kinds)
    cut_sec = {s["start"]: secs[g] for g, s in enumerate(tl["shots"])}
    hit = [cut_sec[e["cut"]] for e in events]
    assert events and len(hit) == len(set(hit))


def test_off_switch(tmp_path, monkeypatch):
    monkeypatch.setenv("DYPOL_SFX", "off")
    assert not sfx.enabled() and not sfx.available()
    assert sfx.plan_transitions([(5.0, "kinetic_text"), (20.0, "chapter_card")], 60.0) == []
    voice = _speechlike(tmp_path / "v.wav", 3.0)
    assert sfx.build_track([{"t": 1.0, "clip": "chime_soft_01"}], 3.0, tmp_path / "s.wav", voice) is None
    assert not (tmp_path / "s.wav").exists()
    monkeypatch.setenv("DYPOL_SFX", "on")
    assert sfx.enabled()
