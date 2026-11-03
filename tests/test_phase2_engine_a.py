"""Engine A quality upgrade, piece 2: variable shot timing, shot mix, karaoke captions,
SFX, grade and the QC stage."""
import json
import re
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from src import captions as caps
from src import explainer as ex
from src import pacing
from src import production as prod
from src import qc
from src import sfx
from src import shotlist
from src import storyboard as sb

REPO = Path(__file__).resolve().parents[1]

NARRATION = ("A tokenizer splits your text into small pieces, called tokens. The model reads those tokens, "
             "predicts the next one, and repeats the loop until the answer is done. That is why long prompts "
             "cost 3x more, and why 70% of teams now trim them. Short prompts are faster; they are also cheaper. "
             "Keep the instructions, drop the filler, and measure the result every single week.")


def _tone(path, seconds):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"sine=frequency=300:duration={seconds}",
                    "-ar", "48000", str(path)], check=True)
    return path


# ------------------------------------------------------------------ 1. variable shot timing

def test_variable_shots_are_2_to_4_s_and_cut_on_clauses():
    words = sb.estimate_word_timings(NARRATION, 30.0)
    shots = pacing.plan_variable_shots(30.45, words)
    lengths = [b - a for a, b in shots]
    assert all(2.0 - 1e-6 <= d <= 4.0 + 1e-6 for d in lengths), lengths
    assert len(set(round(d, 1) for d in lengths)) > 2, "shot lengths vary"
    cands = {t: strength for t, strength in pacing.cut_candidates(words, 30.45)}
    cuts = [b for _, b in shots[:-1]]
    assert all(c in cands for c in cuts), "every cut sits between two words"
    assert sum(cands[c] > 0 for c in cuts) >= len(cuts) // 2, "most cuts land on clause/sentence ends"


def test_chapter_card_shot_is_1_to_1_5_s_and_timeline_has_no_gaps():
    words = [sb.estimate_word_timings(NARRATION, 30.0), sb.estimate_word_timings("Short outro line here.", 3.0)]
    tl = pacing.build_timeline([30.45, 3.45], words, [True, False])
    first = tl["shots"][tl["slides"][0]["shots"][0]]
    assert 1.0 <= first["end"] - first["start"] <= 1.5
    assert sb.coverage_gaps(tl) == []
    assert abs(tl["total"] - 33.9) < 1e-3
    for a, b in zip(tl["shots"], tl["shots"][1:]):
        assert abs(a["end"] - b["start"]) < 1e-3
    assert max(s["end"] - s["start"] for s in tl["shots"]) <= pacing.rules()["max_static"]


def test_no_words_still_gives_bounded_shots():
    shots = pacing.plan_variable_shots(17.0, [])
    assert all(2.0 - 1e-6 <= b - a <= 4.0 + 1e-6 for a, b in shots)
    assert shots[0][0] == 0.0 and shots[-1][1] == 17.0


# ------------------------------------------------------------------ 2. shot mix

def _mix_fixture():
    words = [sb.estimate_word_timings(NARRATION, 30.0)]
    slide = sb.normalize_slide({"title": "How tokens work", "content": NARRATION,
                                "bullets": ["Text becomes tokens", "Model predicts next token", "Short prompts cost less"],
                                "shots": [{"kind": "stock_or_ai_still"}, {"kind": "code_panel",
                                                                          "code": "print(len(tok.encode(p)))"},
                                          {"kind": "stock_or_ai_still"}, {"kind": "stock_or_ai_still"}]})
    tl = pacing.build_timeline([30.45], words, [True])
    return tl, [slide], words


def test_mix_enforced_demo_40_and_stills_15(tmp_path):
    tl, slides, words = _mix_fixture()
    fixed, kinds = prod.plan_blocks(tl, slides, tmp_path, (480, 270), "long", words=words)
    report = prod.style_report(kinds, fixed, tl)
    assert report["demo_share"] >= 0.40, report
    assert report["still_share"] <= 0.15, report
    assert report["runs_over_limit"] == []
    first = tl["slides"][0]["shots"][0]
    assert kinds[first] == "chapter_card"
    # every multi-shot block keeps one clock: same span, offsets from its first window
    windows = ex.shot_windows(tl)
    by_id = {}
    for g, a in fixed.items():
        by_id.setdefault(a["block_id"], []).append(g)
    for bid, members in by_id.items():
        members.sort()
        spans = {round(fixed[g]["block_span"], 3) for g in members}
        assert len(spans) == 1
        for g in members:
            assert fixed[g]["block_offset"] == pytest.approx(windows[g][0] - windows[members[0]][0], abs=1e-3)
        assert fixed[members[0]]["block_span"] <= 10.0 + 1e-6


def test_spoken_number_parsing():
    spec = prod.spoken_number("about 70% of teams now trim".split())
    assert spec["value"] == "70" and spec["suffix"] == "%" and spec["label"]
    assert prod.spoken_number("prompts cost 3x more".split())["suffix"] == "x"
    # Spelled-out numbers count since 3 Nov 2026 (run 41646837309 got 1 count_up from a spelled-out script)
    assert prod.spoken_number("forty-two percent".split())["value"] == "42"
    assert prod.spoken_number("back in 2024 we".split()) is None          # years are not count_ups
    assert prod.spoken_number("12 percent faster".split())["suffix"] == "%"


# ------------------------------------------------------------------ 3. captions

def test_karaoke_groups_long_and_short():
    words = sb.estimate_word_timings(NARRATION, 30.0)
    long_groups = caps.karaoke_groups(words, 6, 40, 1)
    assert all(1 <= len(g) <= 6 for g in long_groups)
    assert sum(len(g) for g in long_groups) == len(words)
    short_groups = caps.karaoke_groups(words, 4, 22, 2)
    assert all(len(g) <= 4 for g in short_groups)
    assert sum(len(g) >= 2 for g in short_groups) >= len(short_groups) - 1


def test_no_orphan_word_after_a_full_line():
    words = sb.estimate_word_timings("Here is why that matters for you. Next line.", 4.0)
    groups = caps.karaoke_groups(words, 6, 40, 1)
    assert [w for w, _, _ in groups[1]][-1] == "you."
    assert len(groups[1]) >= 2


def test_ass_has_one_event_per_word_with_lime_active_word():
    words = sb.estimate_word_timings("Seventy percent of developers now use AI tools.", 4.0)
    text = caps.words_to_ass(words, (1920, 1080), "long")
    events = [line for line in text.splitlines() if line.startswith("Dialogue:")]
    assert len(events) == len(words)
    assert "Style: Cap,Space Grotesk," in text and "PlayResY: 1080" in text
    lime = "&H0060F5C8"       # #C8F560 in ASS BGR order
    assert all(line.count(lime) == 1 for line in events)
    assert all("\\pos(960,955)" in line for line in events)          # lower third
    short = caps.words_to_ass(words, (1080, 1920), "short")
    assert "\\pos(540,1209)" in short                                  # ~63 % of the height


def test_srt_upload_file_unchanged_by_karaoke():
    words = [("Hello", 0.0, 0.4), ("world.", 0.4, 0.9)]
    assert caps.words_to_srt(words) == "1\n00:00:00,000 --> 00:00:00,900\nHello world.\n"


def test_vendored_caption_font_exists():
    assert (REPO / "assets/fonts/dypol/SpaceGrotesk-Bold.ttf").exists()
    assert (REPO / "assets/fonts/dypol/OFL-SpaceGrotesk.txt").exists()


# ------------------------------------------------------------------ 4. SFX

def test_sfx_files_vendored_with_licence():
    licence = (REPO / "sfx/LICENSES.md").read_text()
    assert "CC0" in licence
    for clip in sfx.CLIPS:
        assert sfx.clip_path(clip).exists(), clip
        assert f"{clip}.ogg" in licence


def _changes(total, step=3.0):
    kinds = ["diagram", "kinetic_text", "stock_or_ai_still", "count_up", "code_panel"]
    return [(round(t, 2), kinds[k % len(kinds)]) for k, t in enumerate(
        [step * i for i in range(1, int(total / step))])]


def test_transition_hits_4_to_8_per_minute_and_varied():
    total = 240.0
    events = sfx.plan_transitions(_changes(total), total, seed=3)
    rate = len(events) / (total / 60)
    assert 4.0 <= rate <= 8.0, rate
    cuts = [e["cut"] for e in events]
    assert all(b - a >= 7.5 - 1e-6 for a, b in zip(cuts, cuts[1:]))
    clips = [e["clip"] for e in events]
    assert all(a != b for a, b in zip(clips, clips[1:]))
    assert len(set(clips)) >= 4
    whoosh = next(e for e in events if e["clip"] in ("whoosh_01", "whoosh_03"))
    assert whoosh["t"] == pytest.approx(whoosh["cut"] - 0.52, abs=0.01)   # peak lands on the cut


def test_clicks_follow_code_typing(tmp_path):
    tl = sb.build_timeline([8.0], 4.0)
    windows = ex.shot_windows(tl)
    from src import blocks
    span = windows[1][1] - windows[0][0]
    base = dict(blocks.block_asset({"kind": "code_panel", "code": "x = 1\nprint(x)"}), block_id="b", block_span=span)
    fixed = {0: dict(base, block_offset=0.0), 1: dict(base, block_offset=windows[1][0] - windows[0][0])}
    typing = sfx.code_typing_windows(tl, fixed)
    assert len(typing) == 1
    a, b = typing[0]
    clicks = sfx.plan_clicks(typing, seed=1)
    assert clicks and all(a <= c["t"] < b for c in clicks)
    assert all(c["clip"].startswith("click_") for c in clicks)


def test_sfx_track_sits_about_12_db_under_voice(tmp_path):
    from pydub import AudioSegment
    voice = _tone(tmp_path / "v.wav", 6.0)
    events = [{"t": 1.0, "clip": "pop_01", "gain_db": 0.0}, {"t": 3.0, "clip": "whoosh_02", "gain_db": 0.0}]
    out = sfx.build_track(events, 6.0, tmp_path / "s.wav", voice)
    seg, v = AudioSegment.from_file(str(out)), AudioSegment.from_file(str(voice))
    assert abs(len(seg) - 6000) < 30
    hit = seg[1000:1000 + 150]
    assert v.dBFS - 16 <= hit.dBFS <= v.dBFS - 8


def test_mix_with_sfx_and_ducked_music_hits_minus_14(tmp_path):
    voice = _tone(tmp_path / "v.wav", 6.0)
    music = _tone(tmp_path / "m.wav", 3.0)
    track = sfx.build_track([{"t": 2.0, "clip": "pop_02", "gain_db": 0.0}], 6.0, tmp_path / "s.wav", voice)
    out = ex.mix_music_and_normalize(voice, tmp_path / "o.wav", music_path=music, music_db=-26, sfx_path=track)
    assert ex.measure_lufs(out) == pytest.approx(-14.0, abs=1.0)
    graph = ex.mix_graph(6.0, music=True, sfx=True)
    assert "sidechaincompress" in graph and "amix=inputs=3" in graph


# ------------------------------------------------------------------ 5. grade + burn-in in the encoder

def test_encoder_applies_filters_on_absolute_time():
    cmd = ex._encoder_cmd((320, 180), 30, 21, "x.ts", vf="eq=contrast=1.06", offset=12.5)
    vf = cmd[cmd.index("-vf") + 1]
    assert vf.startswith("setpts=PTS+12.500000/TB,") and vf.endswith("setpts=PTS-STARTPTS")
    assert "-vf" not in ex._encoder_cmd((320, 180), 30, 21, "x.ts")
    look = ex.finish_filter(None)
    assert "vignette" in look and "noise" in look and "eq=" in look


@pytest.mark.skipif(not ex.filter_available("subtitles"), reason="ffmpeg without libass")
def test_parallel_render_burns_captions_in_the_right_chunk(tmp_path):
    size = (640, 360)
    tl = sb.build_timeline([4.5, 4.5], 1.5)
    img = tmp_path / "i.png"
    Image.new("RGB", (640, 360), (40, 60, 90)).save(img)
    assets = [{"kind": "image", "path": str(img)} for _ in tl["shots"]]
    audio = _tone(tmp_path / "a.wav", tl["total"])
    ass = caps.write_ass([("LIME", 6.0, 8.5)], tmp_path / "c.ass", size, "long")
    vf = ex.finish_filter(ass, grade=False)
    logs = []
    out = ex.render(tl, assets, audio, tmp_path / "o.mp4", size, lambda f, t: f, fps=10, workers=2,
                    progress=logs.append, vf=vf)
    assert any("parallel render: 2 chunks" in line for line in logs)
    assert not any("failed" in line for line in logs)

    def lime_pixels(t):
        data = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", str(out), "-frames:v", "1",
                               "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True).stdout
        from io import BytesIO
        im = Image.open(BytesIO(data)).convert("RGB")
        return sum(1 for r, g, b in im.getdata() if g > 180 and r > 150 and b < 140)
    assert lime_pixels(7.0) > 20       # second chunk: the caption is at 6-8.5 s of absolute time
    assert lime_pixels(2.0) == 0


def test_render_falls_back_without_filters(tmp_path):
    size = (160, 90)
    tl = sb.build_timeline([2.0], 1.0)
    img = tmp_path / "i.png"
    Image.new("RGB", (320, 180), (10, 120, 200)).save(img)
    audio = _tone(tmp_path / "a.wav", tl["total"])
    logs = []
    out = ex.render(tl, [{"kind": "image", "path": str(img)}] * len(tl["shots"]), audio, tmp_path / "o.mp4",
                    size, lambda f, t: f, fps=10, workers=1, progress=logs.append, vf="nosuchfilter=1")
    assert out.exists() and any("retrying without grade/captions" in line for line in logs)


# ------------------------------------------------------------------ 6. QC

SAMPLE_LOG = """
[blackdetect @ 0x1] black_start:3.2 black_end:4.0 black_duration:0.8
[freezedetect @ 0x2] lavfi.freezedetect.freeze_start: 10.5
[freezedetect @ 0x2] lavfi.freezedetect.freeze_duration: 7.1
[freezedetect @ 0x2] lavfi.freezedetect.freeze_end: 17.6
[silencedetect @ 0x3] silence_start: 20.0
[silencedetect @ 0x3] silence_end: 23.5 | silence_duration: 3.5
[Parsed_ebur128_0 @ 0x4] Summary:

  Integrated loudness:
    I:         -16.2 LUFS
    Threshold: -26.4 LUFS

  Loudness range:
    LRA:         5.1 LU

  True peak:
    Peak:       -0.4 dBFS
"""


def test_qc_parse_and_warnings():
    result = qc.parse_log(SAMPLE_LOG, 60.0)
    result["duration"] = 60.0
    assert result["loudness"] == {"integrated_lufs": -16.2, "lra_lu": 5.1, "true_peak_dbtp": -0.4}
    assert result["black"][0]["duration"] == 0.8
    assert result["freezes"][0] == {"start": 10.5, "end": 17.6, "duration": 7.1}
    assert result["silences"][0]["duration"] == 3.5
    warnings = qc.warnings_for("long", result)
    assert len(warnings) == 5 and all(w.startswith("long: ") for w in warnings)


def test_qc_run_writes_json_and_small_contact_sheet(tmp_path):
    video = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=640x360:d=4:r=15",
                    "-f", "lavfi", "-i", "sine=frequency=300:duration=4", "-shortest", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", str(video)], check=True)
    short = tmp_path / "s.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=360x640:d=3:r=15",
                    "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", "3", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", str(short)], check=True)
    report = qc.run(video, short, tmp_path / "out")
    data = json.loads((tmp_path / "out/qc.json").read_text())       # strict JSON (no -Infinity)
    assert set(data["videos"]) == {"long", "short"}
    assert any("short:" in w and "black" in w for w in report["warnings"])
    sheet = tmp_path / "out/contact.png"
    assert sheet.exists() and sheet.stat().st_size <= qc.MAX_CONTACT_BYTES
    assert Image.open(sheet).width == 1280


def test_main_records_qc_without_touching_review_flag():
    main = (REPO / "main.py").read_text()
    assert "qc.run(long_video_path, short_video_path, OUTPUT_DIR" in main
    assert '"qc": qc_summary' in main
    assert "review_required" not in main[main.index("qc_summary = {"):main.index("hashtags = ")]
