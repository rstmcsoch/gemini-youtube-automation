"""Engine A piece 4: living backgrounds (never flat / white), flow motion, the real Short, script depth."""
import json
import subprocess

import pytest
from PIL import Image, ImageStat

from src import backdrop, blocks, explainer as ex, production as prod, storyboard as sb

SPECS = {
    "code_panel": {"code": "import dypol\nprint(dypol.run('hi'))", "language": "python"},
    "diagram": {"nodes": ["Prompt", "Tokens", "Model"]},
    "chapter_card": {"number": 2, "total": 6, "title": "How it works"},
    "kinetic_text": {"text": "Predicts the next token", "label": "Key idea"},
    "count_up": {"value": "55", "suffix": "%", "label": "faster task completion", "source": "GitHub, 2022"},
    "icon_flow": {"nodes": ["Prompt", "Model", "Code"], "title": "The loop"},
}
SIZES = [(1920, 1080), (1080, 1920)]


def _luma(img):
    return ImageStat.Stat(img.convert("L")).mean[0]


def _white_share(img):
    hist = img.convert("L").histogram()
    return sum(hist[236:]) / max(1, sum(hist))


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("kind", sorted(SPECS))
def test_block_backgrounds_are_never_white_or_flat(kind, size):
    src = blocks.SOURCES[kind](dict(SPECS[kind], kind=kind), size, 5.0)
    for t in (0.0, 0.6, 2.5, 4.8):
        frame = src.frame(t)
        assert frame.size == size
        assert 8 < _luma(frame) < 110, (kind, size, t, _luma(frame))
        assert _white_share(frame) < 0.12, (kind, size, t)
    # the background itself is not one flat colour
    assert ImageStat.Stat(src.bg_frame(1.0).convert("L")).stddev[0] > 2.0


@pytest.mark.parametrize("size", SIZES)
def test_backdrop_drifts_and_photo_backdrop_is_darkened(tmp_path, size):
    field = backdrop.Backdrop(size, seed=3)
    assert field.frame(0.0).tobytes() != field.frame(2.0).tobytes()          # living, not static
    assert field.frame(1.3).tobytes() == field.frame(1.3).tobytes()          # pure function of t
    white = tmp_path / "white.png"
    Image.new("RGB", (800, 800), (252, 252, 252)).save(white)
    photo = backdrop.Backdrop(size, seed=1, image=str(white))
    assert photo.mode == "photo"
    assert _luma(photo.frame(0.5)) <= 75


def test_white_still_is_tamed(tmp_path):
    white = tmp_path / "w.png"
    Image.new("RGB", (1200, 800), (250, 250, 250)).save(white)
    still = ex.StillSource(str(white), (960, 540), 3.0)
    assert _luma(still.frame(1.0)) < 130
    normal = tmp_path / "n.png"
    Image.new("RGB", (1200, 800), (90, 100, 110)).save(normal)
    assert abs(_luma(ex.StillSource(str(normal), (960, 540), 3.0).frame(1.0)) - _luma(Image.open(normal))) < 3


@pytest.mark.parametrize("kind", ["diagram", "icon_flow", "kinetic_text"])
def test_flow_blocks_keep_moving_after_the_build(kind):
    src = blocks.SOURCES[kind](dict(SPECS[kind], kind=kind), (1920, 1080), 8.0)
    late = [src.frame(t).tobytes() for t in (5.0, 5.5, 6.0)]
    assert len(set(late)) == 3


def test_portrait_blocks_stay_clear_of_the_caption_zone():
    d = blocks.DiagramSource({"nodes": ["A", "B", "C", "D"]}, (1080, 1920), 5.0)
    assert max(b[3] for b in d.boxes) < 0.6 * 1920
    f = blocks.IconFlowSource({"nodes": ["A", "B", "C"]}, (1080, 1920), 5.0)
    assert max(y for _, y in f.centers) + f.card / 2 + 80 < 0.6 * 1920


def _timeline(n_shots=9):
    timeline = sb.build_timeline([3.0 * n_shots], 3.0)
    return timeline


def test_text_governor_breaks_three_text_blocks_in_a_row(tmp_path):
    timeline = sb.build_timeline([30.0], 3.0)
    n = len(timeline["shots"])
    windows = ex.shot_windows(timeline)
    fixed, kinds = {}, []
    for g in range(n):
        spec = {"kind": "kinetic_text", "text": f"Point number {g}"}
        fixed[g] = dict(blocks.block_asset(spec), block_id=f"k{g}", block_span=windows[g][1] - windows[g][0],
                        block_offset=0.0)
        kinds.append("kinetic_text")
    slides = [{"title": "Tokens", "bullets": ["Split the prompt", "Predict tokens"]}]
    prod._govern_text(timeline, slides, fixed, kinds, None, (1920, 1080), "long")
    assert prod.text_runs(kinds, fixed) <= 2
    assert "icon_flow" in kinds


def test_number_label_reads_alone():
    assert prod.number_label("dimensions for every word in the model".split()) == "Dimensions for every word"
    assert prod.number_label("means complete".split()) == ""
    assert prod.number_label("of the code".split()) == ""
    spec = prod.spoken_number("about 512 dimensions for every word".split())
    assert spec["value"] == "512" and spec["label"] == "Dimensions for every word"


PLAN = {"hook": "Can AI write code from one sentence?",
        "beats": [{"text": "First, your prompt becomes tokens.", "nodes": ["Prompt", "Tokens"], "icon": "chat"},
                  {"text": "Then it predicts the next token, 55 percent faster than typing.", "nodes": ["Predict",
                                                                                                        "Repeat"]}],
        "catch": {"text": "But it can invent functions that do not exist."},
        "takeaway": "Review every line it writes.",
        "number": {"value": "55", "suffix": "%", "label": "faster task completion", "source": "GitHub, 2022"}}


def _script(plan):
    return " ".join([plan["hook"]] + [b["text"] for b in plan["beats"]] + [plan["catch"]["text"], plan["takeaway"]])


def test_short_sections_follow_the_structure_and_fallback():
    roles = [s["role"] for s in prod.short_sections(PLAN, _script(PLAN))]
    assert roles == ["hook", "beat", "beat", "catch", "takeaway"]
    loose = ("Why does this matter? One. Two. Three. Four is next. But it fails on edge cases. Remember this.")
    roles = [s["role"] for s in prod.short_sections(None, loose)]
    assert roles[0] == "hook" and roles[-1] == "takeaway" and "catch" in roles and "beat" in roles


def test_short_number_is_never_invented():
    words = [(w, i * 0.4, i * 0.4 + 0.3) for i, w in enumerate(_script(PLAN).split())]
    sourced = prod.short_number(PLAN["number"], words)
    assert sourced["source"] and sourced["value"] == "55"
    unsourced = prod.short_number(dict(PLAN["number"], source=""), words)
    assert unsourced["value"] == "55" and not unsourced.get("source")          # the spoken number, no source
    plain = [(w, i * 0.4, i * 0.4 + 0.3) for i, w in enumerate("No figures are said here at all.".split())]
    assert prod.short_number(None, plain) is None


def test_plan_short_is_a_motion_short(tmp_path):
    script = " ".join([_script(PLAN)] * 4)
    words = [(w, i * 0.42, i * 0.42 + 0.36) for i, w in enumerate(script.split())]
    total = words[-1][2] + 0.6
    timeline = prod.pacing.build_timeline([total], [words], [False])
    plan = dict(PLAN, hook=PLAN["hook"], beats=PLAN["beats"])
    sections = prod.short_sections(None, script)
    number = {k: v for k, v in prod.short_number(PLAN["number"], words).items() if k != "sourced"}
    fixed, kinds = prod.plan_short(timeline, words, sections, number, photos=[], n_stills=2)
    assert "count_up" in kinds and ("diagram" in kinds or "icon_flow" in kinds)
    assert prod.text_runs(kinds, fixed) <= 2
    assert kinds.count("stock_or_ai_still") / len(kinds) <= 0.25
    assert all(fixed[g]["spec"].get("kind") == kinds[g] for g in fixed)
    del plan


def _tone(path, seconds):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"sine=frequency=300:duration={seconds}",
                    "-ar", "44100", str(path)], check=True)
    return path


def test_short_audio_lands_in_50_to_65_seconds(tmp_path):
    # 4 Nov 2026: too short -> one content rewrite (expand_fn), never slowed below 150 WPM.
    short_voice = _tone(tmp_path / "s.wav", 46.0)
    more_voice = _tone(tmp_path / "m.wav", 53.0)
    script = " ".join(["word"] * 120)                     # 120 words in 46 s = 157 WPM
    out, voice = prod.fit_short_audio(lambda text, path: more_voice if len(text.split()) > 120 else short_voice,
                                      script, "", tmp_path, expand_fn=lambda s, n: s + " more" * n)
    assert ex.audio_duration(voice) >= 50.0 and len(out.split()) > 120
    long_voice = _tone(tmp_path / "l.wav", 72.0)
    _, voice = prod.fit_short_audio(lambda text, path: long_voice, "Hook. Body.", "", tmp_path)
    assert ex.audio_duration(voice) <= prod.SHORT_MAX


def test_main_short_has_no_cta_and_uses_the_plan():
    from pathlib import Path
    main = (Path(prod.__file__).resolve().parent.parent / "main.py").read_text()
    assert 'fit_short_audio(text_to_speech, short_script, "", short_slide_dir,' in main
    assert "expand_fn=lambda s, n: expand_short_script(s, n, lesson['title'])" in main
    assert "plan=short_plan" in main


class _Resp:
    def __init__(self, text):
        self.text = text


def _content(words_per_chapter, short_words=130):
    return {"long_form_slides": [{"title": f"C{i}", "content": " ".join(["word"] * words_per_chapter)}
                                 for i in range(7)],
            "short_script": " ".join(["w"] * short_words)}


def test_depth_rewrite_once_when_thin(monkeypatch):
    import src.generator as gen
    calls = []

    def fake(client, prompt, task_name="", **_k):
        calls.append(task_name)
        return _Resp(json.dumps(_content(200)))

    monkeypatch.setattr(gen, "_generate_text_with_fallback", fake)
    out = gen.ensure_depth(object(), _content(100), "prompt")
    assert calls == ["lesson depth rewrite"]
    assert gen.narration_words(out)[0] == 1400
    calls.clear()
    out = gen.ensure_depth(object(), _content(200), "prompt")
    assert calls == [] and gen.narration_words(out)[0] == 1400


def test_prompt_asks_for_depth_and_a_structured_short(monkeypatch):
    import src.generator as gen
    captured = {}

    def fake(client, prompt, task_name="", **_k):
        captured.setdefault("prompt", prompt)
        return _Resp(json.dumps({"long_form_slides": []}))

    monkeypatch.setenv("GOOGLE_API_KEY", "test")
    monkeypatch.setattr(gen, "_generate_text_with_fallback", fake)
    monkeypatch.setattr(gen.genai, "Client", lambda api_key: object())
    monkeypatch.setenv("DYPOL_TOPIC_CHECK", "0")  # stub scripts have no narration
    gen.generate_lesson_content("Topic")
    p = captured["prompt"]
    assert "worked walkthrough" in p and "the catch" in p and str(gen.LONG_MIN_WORDS) in p
    assert '"beats"' in p and '"number"' in p and "NO call to action" in p and "{chapter_words}" not in p
