"""CPU block renderers: 1920x1080 @ 30 fps, 2 s samples checked with ffprobe; motion; wiring."""
import json
import subprocess

import pytest

from src import blocks, explainer as ex, production, shotlist, storyboard as sb

SPECS = {
    "code_panel": {"code": "import dypol\nagent = dypol.Agent()\nprint(agent.run('hi'))", "language": "python"},
    "diagram": {"nodes": ["Prompt", "Retriever", "LLM"]},
    "chapter_card": {"number": 2, "total": 6, "title": "How it works"},
    "kinetic_text": {"text": "10x cheaper"},
    "count_up": {"value": "94.5", "suffix": "%", "label": "accuracy"},
}


def _probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                          "stream=width,height,r_frame_rate,nb_read_frames:format=duration", "-of", "json",
                          str(path)], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


@pytest.mark.parametrize("kind", sorted(SPECS))
def test_renderer_2s_sample_ffprobe(tmp_path, kind):
    out = blocks.render_clip(kind, SPECS[kind], tmp_path / f"{kind}.mp4", seconds=2.0)
    info = _probe(out)
    stream = info["streams"][0]
    assert (stream["width"], stream["height"]) == (1920, 1080)
    assert stream["r_frame_rate"] == "30/1"
    assert int(stream["nb_read_frames"]) == 60
    assert abs(float(info["format"]["duration"]) - 2.0) < 0.1


@pytest.mark.parametrize("kind", sorted(SPECS))
def test_renderer_moves_and_is_deterministic(kind):
    src = blocks.SOURCES[kind](SPECS[kind], (1920, 1080), 4.0)
    a, b = src.frame(0.2).tobytes(), src.frame(1.2).tobytes()
    assert a != b                                        # eased motion, not a static card
    assert src.frame(1.2).tobytes() == b                 # pure function of t (chunked render safe)
    assert src.frame(0.0).size == (1920, 1080)


def _timeline_and_slides():
    timeline = sb.build_timeline([30.0, 20.0], sb.LONG_SHOT_TARGET)
    slides = [sb.normalize_slide({"title": "Tokens", "content": "x", "bullets": ["Tokens are word pieces"],
                                  "shots": [{"kind": "diagram", "nodes": ["Text", "Tokens"]},
                                            {"kind": "stock_or_ai_still"},
                                            {"kind": "code_panel", "code": "print(len(tokens))"}]}),
              sb.normalize_slide({"title": "Cost", "content": "y", "bullets": ["Cheaper per token"],
                                  "shots": [{"kind": "count_up", "value": "80", "suffix": "%", "label": "price cut"}]})]
    return timeline, slides


def test_plan_blocks_maps_shots_and_spans_continuously(tmp_path):
    timeline, slides = _timeline_and_slides()
    fixed, kinds = production.plan_blocks(timeline, slides, tmp_path)
    first = timeline["slides"][0]["shots"]
    assert kinds[first[0]] == "chapter_card"
    assert "diagram" in kinds and "code_panel" in kinds and "count_up" in kinds
    spanning = [a for a in fixed.values() if a["block_offset"] > 0]
    assert spanning, "a multi-shot block keeps one clock"
    units = [k for k, _ in production._units(kinds, fixed)]
    assert shotlist.runs_over(units, shotlist.max_run()) == []
    report = production.style_report(kinds, fixed)
    assert 0 <= report["still_share"] <= 1


def test_block_source_wired_into_chunked_renderer(tmp_path):
    timeline = sb.build_timeline([4.0], sb.LONG_SHOT_TARGET)
    spec = {"kind": "diagram", "nodes": ["A", "B", "C"]}
    windows = ex.shot_windows(timeline)
    span = windows[-1][1] - windows[0][0]
    assets = []
    for i in range(len(timeline["shots"])):
        assets.append(dict(blocks.block_asset(spec), block_span=span, block_offset=windows[i][0] - windows[0][0]))
    audio = tmp_path / "a.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
                    "-t", "4", str(audio)], check=True)
    out = ex.render(timeline, assets, audio, tmp_path / "out.mp4", (1920, 1080), lambda f, t: f, workers=1)
    info = _probe(out)
    assert (info["streams"][0]["width"], info["streams"][0]["height"]) == (1920, 1080)
    assert abs(float(info["format"]["duration"]) - 4.0) < 0.15
