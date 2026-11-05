"""Calm captions + sound (5 Nov 2026): caption placement, key-word-only highlights,
procedural SFX >= 18 dB under the voice, no old pop/click/whoosh files referenced."""
import re
from pathlib import Path

import pytest

from src import captions as caps
from src import explainer as ex
from src import sfx
from src import storyboard as sb

REPO = Path(__file__).resolve().parents[1]
LIME = "&H0060F5C8"


def test_caption_center_y_values():
    assert caps.caption_settings("long")["center_y"] == pytest.approx(0.78, abs=0.02)
    assert 0.78 <= caps.caption_settings("long")["center_y"] <= 0.80
    assert caps.caption_settings("short")["center_y"] == pytest.approx(0.75, abs=0.01)


def test_short_caption_inside_shorts_safe_zone():
    size = (1080, 1920)
    top, bottom = caps.caption_band("short", size)
    assert bottom <= 0.80 * size[1]          # above the bottom ~20 % Shorts UI
    assert top >= 0.60 * size[1]


def test_long_caption_clears_lower_third_panel():
    size = (1920, 1080)
    cap_top, cap_bottom = caps.caption_band("long", size)
    slide = {"kicker": "Why it matters", "title": "A long headline that wraps onto a second line here",
             "bullets": ["First bullet point text", "Second bullet point text"]}
    panel, (x0, y0), _ = ex.render_lower_third(slide, size, bottom=caps.panel_bottom_frac(size))
    assert y0 + panel.size[1] < cap_top        # panel's lower edge sits above the caption line
    assert y0 > 0
    assert cap_bottom < size[1]


def _lime_words(line):
    return re.findall(r"\{\\c" + re.escape(LIME) + r"\}([^{]+)\{", line)


def test_at_most_one_key_term_per_sentence_plus_numbers():
    text = ("Seventy percent of developers now use AI tools daily. Back in 2024 only 12% did, says Stack Overflow. "
            "The adoption curve keeps climbing quickly across every engineering organisation. It grew 3x since 1999.")
    words = text.split()
    flags = caps.key_word_flags(words)
    nums = caps.number_flags(words)
    sentences, cur = [], []
    for w, f, n in zip(words, flags, nums):
        cur.append((w, f and not n))
        if w.endswith((".", "!", "?")):
            sentences.append(cur)
            cur = []
    assert len(sentences) == 4
    for sent in sentences:
        assert sum(f for _, f in sent) <= 1, sent             # at most one key term per sentence
    lit = {caps._bare(w) for w, f in zip(words, flags) if f}
    assert {"Seventy", "percent", "2024", "12%", "3x", "1999"} <= lit
    assert all(f for f, n in zip(flags, nums) if n)            # every number is highlighted


def test_ass_plain_white_body_and_few_highlights():
    words = sb.estimate_word_timings("The model now answers questions much faster than before.", 4.0)
    text = caps.words_to_ass(words, (1920, 1080), "long")
    events = [line for line in text.splitlines() if line.startswith("Dialogue:")]
    lit = [w for line in events for w in _lime_words(line)]
    assert len(lit) <= 1                                   # no numbers -> at most one key term
    assert ",&H00FFFFFF," in text.split("Style: Cap,")[1]   # primary colour plain white


def test_sfx_offset_at_least_18_db():
    assert sfx.audio_rules()["under_db"] >= 18.0
    assert sfx.MIN_UNDER_VOICE_DB >= 18.0


def test_no_old_pop_files_referenced():
    for path in [REPO / "src/sfx.py", REPO / "src/production.py", REPO / "src/explainer.py",
                 REPO / "style/style.guide.yaml"]:
        body = path.read_text()
        assert ".ogg" not in body, path
        assert not re.search(r"\b(pop|click|squish)_0\d", body), path
        assert "Kenney" not in body, path
    assert not any(re.match(r"(pop|click|squish|whoosh)_0\d$", c) for c in sfx.CLIPS)


def test_sfx_and_music_ducked_under_speech():
    graph = ex.mix_graph(10.0, music=True, sfx=True)
    assert "[m0][km]sidechaincompress" in graph and "[s0][ks]sidechaincompress" in graph
    only_sfx = ex.mix_graph(10.0, music=False, sfx=True)
    assert "[1:a]" in only_sfx and "[s0][ks]sidechaincompress" in only_sfx and "asplit=2" in only_sfx
