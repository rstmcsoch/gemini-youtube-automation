"""Engine A final fixes (4 Nov 2026): the Short is never slowed below 150 WPM (length comes from one
content rewrite), its hook needs a number or a surprising specific, and the long video's true peak is
held at or under -1 dBTP (enforced on the mp4 and checked by QC)."""
import json
import subprocess
import types

import pytest

from src import explainer as ex
from src import production as prod
from src import qc
from src import short_rules as sr


def _tone(path, seconds, amp=0.3):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    f"sine=frequency=300:duration={seconds}", "-af", f"volume={amp}", "-ar", "48000", str(path)],
                   check=True)
    return path


# ------------------------------------------------------------------ pace floor
def test_slow_factor_never_goes_under_150_wpm():
    # 120 words in 46 s = 156.5 WPM; reaching 50.3 s would need x0.915 (143 WPM) -> capped at 150 WPM
    f = sr.slow_factor(120, 46.0, 50.3)
    assert sr.wpm(120, 46.0 / f) == pytest.approx(150.0, abs=0.01)
    assert f > 0.95
    assert sr.slow_factor(115, 46.0, 50.3) == 1.0        # already at 150 WPM -> not slowed at all
    assert sr.slow_factor(120, 50.5, 50.3) == 1.0        # long enough
    assert sr.slow_factor(160, 49.5, 50.3) == pytest.approx(49.5 / 50.3)   # tiny gap, pace stays > 150


def test_short_too_short_asks_for_more_content_once_and_never_slows_under_150(tmp_path):
    short_voice = _tone(tmp_path / "s.wav", 40.0)
    calls = []

    def expand(script, extra):
        calls.append(extra)
        return script                                       # the model gave nothing longer

    script = " ".join(["word"] * 104)                      # 104 words / 40 s = 156 WPM
    out, voice = prod.fit_short_audio(lambda t, p: short_voice, script, "", tmp_path, expand_fn=expand)
    assert len(calls) == 1 and calls[0] > 0                 # exactly one rewrite
    dur = ex.audio_duration(voice)
    assert sr.wpm(104, dur) >= 149.5                        # never under 150 WPM
    assert dur < 50.0                                       # stays short rather than dragging


def test_rewrite_with_more_content_is_used(tmp_path):
    short_voice, long_voice = _tone(tmp_path / "s.wav", 44.0), _tone(tmp_path / "l.wav", 52.0)
    script = " ".join(["word"] * 115)
    out, voice = prod.fit_short_audio(lambda t, p: long_voice if len(t.split()) > 115 else short_voice,
                                      script, "", tmp_path, expand_fn=lambda s, n: s + " extra" * n)
    assert len(out.split()) > 115 and ex.audio_duration(voice) == pytest.approx(52.0, abs=0.3)


def test_no_x090_slowdown_left():
    src = (prod.Path(prod.__file__)).read_text()
    assert "max(0.9, dur" not in src


# ------------------------------------------------------------------ hook rule
def test_hook_specific_matches_engine_b():
    vague = "Your phone automatically analyzes visual features across your entire photo library."
    assert sr.hook_specific(vague + " Then it groups faces.") == ""
    assert sr.hook_specific(vague + " It checks 128 points.") == ""   # later numbers do not count
    assert sr.hook_specific("Your phone sorts 3 billion faces a day.") == "number"
    assert sr.hook_specific("An AI can spot a broken wrist faster than most doctors.") == "specific"


def test_vague_short_hook_gets_one_rewrite(monkeypatch):
    from src import generator as gen
    prompts = []

    def fake(client, prompt, **kw):
        prompts.append(prompt)
        return types.SimpleNamespace(text=json.dumps({
            "short_hook": "Your phone compares 30,000 face points per photo.",
            "short_script": "Your phone compares 30,000 face points per photo. Then it groups the faces. "
                            "But twins fool it. Check your albums."}))

    monkeypatch.setattr(gen, "_generate_text_with_fallback", fake)
    content = {"short_hook": "Ever wondered how photos get sorted?",
               "short_script": "Ever wondered how photos get sorted? Then it groups the faces. But twins fool it. "
                               "Check your albums.",
               "short": {"hook": "Ever wondered how photos get sorted?"}}
    out = gen.enforce_short_hook(object(), content)
    assert len(prompts) == 1 and "number written in digits" in prompts[0]
    assert out["short_hook"].startswith("Your phone compares 30,000")
    assert out["short"]["hook"].startswith("Your phone compares 30,000")
    assert out["_short_hook"]["rewritten"] is True and out["_short_hook"]["specific"] == "number"


def test_good_hook_is_left_alone_and_failed_rewrite_warns(monkeypatch):
    from src import generator as gen
    calls = []
    monkeypatch.setattr(gen, "_generate_text_with_fallback",
                        lambda *a, **k: calls.append(1) or types.SimpleNamespace(text='{"short_hook": "Nope.", '
                                                                                       '"short_script": "Nope."}'))
    good = {"short_hook": "AI reads X-rays 30% faster.", "short_script": "AI reads X-rays 30% faster. More."}
    assert gen.enforce_short_hook(object(), good)["_short_hook"]["specific"] == "number" and not calls
    vague = {"short_hook": "Photos are smart.", "short_script": "Photos are smart. They group faces."}
    out = gen.enforce_short_hook(object(), vague)
    assert len(calls) == 1 and out["short_script"] == "Photos are smart. They group faces."
    assert "warning" in out["_short_hook"]


def test_prompt_and_main_carry_the_rules():
    root = prod.Path(prod.__file__).resolve().parent.parent
    gen_src = (root / "src/generator.py").read_text()
    assert "MUST\n           contain a concrete number" in gen_src
    main = (root / "main.py").read_text()
    assert "expand_short_script" in main and 'short_hook") or "", 12)' in main


# ------------------------------------------------------------------ true peak
def test_mix_targets_minus_2_dbtp_with_oversampled_limiter():
    assert ex.WAV_TRUE_PEAK_DBTP <= -2.0 and ex.MAX_TRUE_PEAK_DBTP == -1.0
    lim = ex.true_peak_limiter(-2.0)
    assert lim.startswith("aresample=192000,alimiter=") and lim.endswith("aresample=48000")


def test_enforce_true_peak_brings_a_hot_mp4_under_minus_1(tmp_path):
    wav = tmp_path / "hot.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "anoisesrc=d=8:c=pink:a=0.2",
                    "-f", "lavfi", "-i", "aevalsrc='if(lt(mod(t,0.5),0.002),0.99,0)':s=48000:d=8",
                    "-filter_complex", "[0][1]amix=inputs=2:normalize=0,aformat=channel_layouts=stereo",
                    str(wav)], check=True)
    mp4 = tmp_path / "hot.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=gray:s=160x90:d=8",
                    "-i", str(wav), "-af", "volume=3dB", "-c:v", "libx264", "-c:a", "aac", "-b:a", "192k",
                    "-shortest", str(mp4)], check=True)
    assert ex.measure_true_peak(mp4) > -1.0
    result = ex.enforce_true_peak(mp4)
    assert 1 <= result["passes"] <= 2                      # bounded, never loops
    assert result["after"] <= -1.0
    assert subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                           "stream=codec_name", "-of", "csv=p=0", str(mp4)],
                          capture_output=True, text=True).stdout.strip() == "h264"


def test_qc_true_peak_over_minus_1_is_a_hard_failure():
    rep = {"videos": {"long": {"width": 1920, "height": 1080, "loudness": {"true_peak_dbtp": -0.2}, "freezes": []},
                      "short": {"width": 1080, "height": 1920, "loudness": {"true_peak_dbtp": -1.3},
                                "freezes": []}}}
    hard = qc.hard_failures(rep)
    assert hard == ["long: true peak -0.2 dBTP (limit -1 dBTP)"]
