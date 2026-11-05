"""Exact audio-visual sync: Kokoro timestamped model -> word timings -> cuts, cues, captions."""
import json
import wave
from pathlib import Path

import numpy as np
import pytest

from src import captions as caps, explainer as ex, pacing, production as prod, storyboard as sb, voice

STEP = 0.05


def _phones(spoken):
    """[(phoneme, start, end)] at STEP s per character of the spoken phoneme string."""
    return [(ch, round(i * STEP, 3), round((i + 1) * STEP, 3)) for i, ch in enumerate(spoken)]


@pytest.fixture(autouse=True)
def _fresh():
    voice.reset()
    yield
    voice.reset()


def test_mapping_aligns_a_number_spoken_as_several_words():
    words = ["In", "2024,", "cars", "won."]
    per_word = ["ɪn", "tˈuː θˈaʊzənd twˈɛnti fˈɔːɹ,", "kˈɑːɹz", "wˈʌn."]
    spoken = "ɪn tˈuː θˈaʊzənd twˈɛnti fˈɔːɹ, kˈɑːɹz wˈʌn."
    out = voice.map_word_timings(words, per_word, _phones(spoken))
    assert [w for w, _, _ in out] == words
    i_two, i_r = spoken.index("t"), spoken.index("ɹ,")
    assert out[1][1] == pytest.approx(i_two * STEP)                 # "2024" starts at "two"
    assert out[1][2] == pytest.approx((i_r + 1) * STEP)             # ...and ends after "four"
    assert out[2][1] == pytest.approx(spoken.index("k") * STEP)      # next word not shifted by 3 words
    assert out[3][1] == pytest.approx(spoken.index("w", i_r) * STEP)
    assert all(a <= b for (_, a, _), (_, b, _) in zip(out, out[1:]))


def test_mapping_tolerates_context_differences_and_fills_unmatched_words():
    words = ["The", "GPU", "—", "runs", "fast."]
    per_word = ["ðə", "dʒˌiːpˌiːjˈuː", "", "ɹˈʌnz", "fˈæst."]
    spoken = "ðɪ dʒˌiːpˌiːjˈuː ɹˈʌnz fˈæst."           # "the" spoken as ðɪ, the dash silent
    out = voice.map_word_timings(words, per_word, _phones(spoken))
    assert out is not None and len(out) == 5
    assert out[0][1] == 0.0
    assert out[1][2] <= out[2][1] + 1e-6 <= out[3][1]               # dash sits in the gap
    assert out[3][1] == pytest.approx(spoken.index("ɹ") * STEP)


def test_mapping_refuses_when_audio_does_not_match_the_text():
    words = ["completely", "different", "words", "here"]
    per_word = ["kəmplˈiːtli", "dˈɪfɹənt", "wˈɜːdz", "hˈɪɹ"]
    assert voice.map_word_timings(words, per_word, _phones("ʃʊ bæ ɡʊ ɒ")) is None


def _tone(seconds, rate=24000, amp=0.3):
    t = np.arange(int(seconds * rate)) / rate
    return (amp * np.sin(2 * np.pi * 220 * t)).astype("float32")


def test_snap_moves_a_word_after_punctuation_onto_its_audible_onset():
    rate = 24000
    audio = np.concatenate([_tone(1.0), np.zeros(int(0.3 * rate), "float32"), _tone(0.7)])
    words = [("one,", 0.0, 1.12), ("two", 1.45, 2.0)]                # durations put "two" 0.15 s late
    out = voice.snap_word_starts(words, audio, rate)
    assert out[1][1] == pytest.approx(1.3, abs=0.011)
    assert out[0][2] <= 1.0 + 0.011
    # Mid-phrase word with no silence before it: untouched.
    flat = _tone(2.0)
    assert voice.snap_word_starts([("a", 0.0, 0.9), ("b", 0.9, 2.0)], flat, rate) == [("a", 0.0, 0.9), ("b", 0.9, 2.0)]


def test_snap_ignores_a_stop_closure_inside_the_previous_word():
    rate = 24000
    # "books and": closure at 0.40-0.50 inside "books" (ends 0.70); "and" starts 0.70 with no gap.
    audio = np.concatenate([_tone(0.4), np.zeros(int(0.1 * rate), "float32"), _tone(0.9)])
    words = [("books", 0.0, 0.70), ("and", 0.70, 1.4)]
    assert voice.snap_word_starts(words, audio, rate)[1][1] == 0.70


class _Timing:
    def __init__(self, phoneme, start, end):
        self.phoneme, self.start, self.end = phoneme, start, end


class _Tok:
    def phonemize(self, text, lang="en-us"):
        return {"In": "ɪn", "2024,": "tˈuː θˈaʊzənd twˈɛnti fˈɔːɹ,", "cars": "kˈɑːɹz", "won.": "wˈʌn."}[text]


class _TimedKokoro:
    has_timings = True
    tokenizer = _Tok()

    def create_timed(self, text, voice=None, speed=1.0, lang=None):
        spoken = "ɪn tˈuː θˈaʊzənd twˈɛnti fˈɔːɹ, kˈɑːɹz wˈʌn."
        seconds = len(spoken) * STEP
        return (_tone(seconds), 24000, [_Timing(*p) for p in _phones(spoken)])


def test_kokoro_speech_writes_word_timings_beside_the_wav(monkeypatch, tmp_path):
    monkeypatch.setattr(voice, "_engine", lambda cfg: _TimedKokoro())
    monkeypatch.setattr(voice, "phonemize_words", lambda engine, words, lang="en-us":
                        [engine.tokenizer.phonemize(w) for w in words])
    text = "In 2024, cars won."
    meta = voice.kokoro_speech(text, tmp_path / "a.wav")
    assert meta["word_timings"] == "kokoro-durations"
    words = voice.load_word_timings(tmp_path / "a.wav", text, ex.audio_duration(tmp_path / "a.wav"))
    assert [w for w, _, _ in words] == text.split()
    assert words[2][1] == pytest.approx("ɪn tˈuː θˈaʊzənd twˈɛnti fˈɔːɹ, kˈɑːɹz".index("k") * STEP, abs=0.011)
    assert voice.load_word_timings(tmp_path / "a.wav", "Other text here now.") is None   # text must match


def test_untimed_engine_writes_no_sidecar(monkeypatch, tmp_path):
    class Plain:
        def create(self, text, voice=None, speed=1.0, lang=None):
            return _tone(len(text.split()) * 0.4), 24000
    monkeypatch.setattr(voice, "_engine", lambda cfg: Plain())
    voice.timings_path(tmp_path / "b.wav").write_text("{}")          # stale sidecar is removed
    meta = voice.kokoro_speech("one two three four five", tmp_path / "b.wav")
    assert meta["word_timings"] == "none" and not voice.timings_path(tmp_path / "b.wav").exists()


def test_enable_timings_accepts_the_durations_output_name():
    class Out:
        def __init__(self, name):
            self.name = name

    class Sess:
        def get_outputs(self):
            return [Out("waveform"), Out("durations")]

    class Eng:
        has_timings = False
        sess = Sess()

    assert voice.enable_timings(Eng()).has_timings is True


def _write(path, samples, rate=24000):
    voice.write_wav(samples, rate, path)
    return path


def test_timings_follow_silence_trim_and_tempo(tmp_path):
    rate = 24000
    src = _write(tmp_path / "v.wav", np.concatenate([np.zeros(int(0.5 * rate), "float32"), _tone(2.0)]))
    voice.save_word_timings(src, "alpha beta", [("alpha", 0.5, 1.4), ("beta", 1.5, 2.5)])
    trimmed = ex.trim_silence(src, tmp_path / "v_trim.wav")
    words = voice.load_word_timings(trimmed, "alpha beta")
    assert words[0][1] == pytest.approx(0.06, abs=0.02)              # 60 ms kept before speech
    assert words[1][1] == pytest.approx(1.06, abs=0.02)
    fast = ex.change_tempo(trimmed, tmp_path / "v_fast.wav", 1.25)
    words_fast = voice.load_word_timings(fast, "alpha beta")
    assert words_fast[1][1] == pytest.approx(words[1][1] / 1.25, abs=0.005)


def test_production_uses_kokoro_timings_and_falls_back_with_a_log_line(tmp_path, capsys):
    rate = 24000
    clip = _write(tmp_path / "s.wav", np.concatenate([_tone(1.0), np.zeros(int(0.3 * rate), "float32"), _tone(1.0)]))
    exact = [("Hello", 0.0, 1.0), ("again.", 1.3, 2.3)]
    voice.save_word_timings(clip, "Hello again.", exact)
    words, source = prod.narration_word_timings("Hello again.", clip)
    assert source == "kokoro" and words == exact
    other = _write(tmp_path / "g.wav", _tone(2.3))                    # gTTS fallback: no sidecar
    words, source = prod.narration_word_timings("Hello again.", other)
    assert source == "estimate" and len(words) == 2
    prod._log_timing_sources("short", [source])
    assert "WORD_TIMING_FALLBACK" in capsys.readouterr().out


WORDS = [("First,", 0.10, 0.50), ("the", 0.80, 0.95), ("model", 1.00, 1.40), ("loads.", 1.45, 2.00),
         ("Then", 2.60, 2.85), ("retrieval", 2.90, 3.50), ("ranks", 3.55, 3.90), ("passages.", 3.95, 4.80)]


def test_cuts_land_on_word_starts():
    starts = {round(a, 4) for _, a, _ in WORDS}
    for t, _ in pacing.cut_candidates(WORDS, 6.0):
        nxt = min(s for s in starts if s >= t - 1e-9)
        assert 0.0 <= nxt - t <= pacing.CUT_LEAD + 1e-6               # on (or a hair before) a word start
    cands = dict(pacing.cut_candidates(WORDS, 6.0))
    assert cands[round(2.60 - pacing.CUT_LEAD, 4)] == 2              # the sentence cut leads "Then"


def test_bullet_reveals_and_captions_start_on_word_starts():
    times = sb.bullet_reveal_times(["model loads", "retrieval ranks passages"], WORDS, 6.0,
                                   min_first=0.0, min_gap=0.5, tail=0.5)
    assert times == [1.00, 2.90]
    ass = caps.words_to_ass(WORDS, (1920, 1080), "long")
    starts = [line.split(",")[1] for line in ass.splitlines() if line.startswith("Dialogue:")]
    word_starts = {caps._ass_ts(a) for _, a, _ in WORDS}
    assert starts and all(s in word_starts for s in starts)


def test_workflow_pins_the_timestamped_model():
    root = Path(__file__).resolve().parent.parent
    wf = (root / ".github/workflows/main.yml").read_text()
    assert "kokoro-onnx-model-files-v2" in wf
    assert voice.KOKORO_TS_REVISION in wf and voice.KOKORO_SHA256[voice.KOKORO_TS_FILE] in wf
    assert "sha256sum -c" in wf
    assert voice.KOKORO_TS_REVISION in voice.KOKORO_URLS[voice.KOKORO_TS_FILE]
    assert voice.settings()["model_file"] == voice.KOKORO_TS_FILE
