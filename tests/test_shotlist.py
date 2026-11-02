"""Shot-list schema: closed block enum, no 3 same kinds in a row, chapter rules, hook check."""
from src import shotlist, style, storyboard as sb


def test_style_guide_loads_from_one_module():
    g = style.guide()
    assert g["long"]["narration"]["wpm_target"] == 158
    assert g["long"]["hook"]["first_word_by_s"] == 0.5
    assert g["long"]["hook"]["stakes_by_s"] == 3.0 and g["long"]["hook"]["number_by_s"] == 8.0
    assert g["long"]["chapters"]["min_interval_s"] == 60 and g["long"]["chapters"]["max_interval_s"] == 120
    assert g["long"]["pacing"]["max_static_s"] == 6.0
    assert g["long"]["variety"]["max_same_kind_run"] == 2
    assert g["long"]["variety"]["min_demo_share"] == 0.40 and g["long"]["variety"]["max_still_share"] == 0.15
    assert style.rgb("background") == (0x0F, 0x14, 0x1C)
    assert style.rgb("lime") == (0xC8, 0xF5, 0x60)
    assert set(style.block_kinds()) == {"kinetic_text", "count_up", "code_panel", "diagram", "chapter_card",
                                        "stock_or_ai_still"}
    for role in ("headline", "mono"):
        spec = style.look()["fonts"][role]
        assert (style.ROOT / spec["file"]).exists()
        assert "Open Font License" in (style.ROOT / spec["license"]).read_text(encoding="utf-8")


def test_unknown_kind_never_passes_through():
    shot = shotlist.normalize_shot({"kind": "hologram", "fallback": "diagram", "nodes": ["A", "B"]})
    assert shot["kind"] == "diagram"
    shot = shotlist.normalize_shot({"kind": "hologram"})
    assert shot["kind"] in shotlist.KINDS
    assert shotlist.normalize_shot({"kind": "terminal", "code": "ls"})["kind"] == "code_panel"


def test_validate_flags_enum_and_runs():
    bad = [{"kind": "chapter_card", "title": "x", "fallback": "kinetic_text"},
           {"kind": "kinetic_text", "text": "a", "fallback": "stock_or_ai_still"},
           {"kind": "kinetic_text", "text": "b", "fallback": "stock_or_ai_still"},
           {"kind": "kinetic_text", "text": "c", "fallback": "stock_or_ai_still"},
           {"kind": "laser", "fallback": "stock_or_ai_still"}]
    errors = shotlist.validate(bad)
    assert any("run of 3" in e for e in errors)
    assert any("not in enum" in e for e in errors)


def test_enforce_repairs_to_valid():
    raw = [{"kind": "kinetic_text", "text": "one"}, {"kind": "kinetic_text", "text": "two"},
           {"kind": "kinetic_text", "text": "three"}, {"kind": "diagram", "nodes": ["only one"]},
           {"kind": "count_up", "value": "no number"}, {"kind": "code_panel", "code": "print(1)"},
           {"kind": "code_panel", "code": "print(2)"}, {"kind": "code_panel", "code": "print(3)"},
           {"kind": "code_panel", "code": "print(4)"}, {"kind": "made_up"}]
    shots = shotlist.enforce(raw, chapter_title="How tokens work")
    assert shots[0]["kind"] == "chapter_card" and shots[0]["title"] == "How tokens work"
    assert shotlist.validate(shots) == []
    assert all(s["kind"] in shotlist.KINDS and s["fallback"] in shotlist.KINDS for s in shots)


def test_normalize_slide_carries_valid_shots():
    slide = sb.normalize_slide({"title": "Embeddings", "content": "Words become vectors.",
                                "shots": [{"kind": "diagram", "nodes": ["Word", "Vector"], "keywords": ["vectors"]}]})
    assert slide["shots"][0]["kind"] == "chapter_card"
    assert slide["shots"][1]["kind"] == "diagram"
    assert shotlist.validate(slide["shots"]) == []


def test_hook_rule():
    assert shotlist.hook_issues("OpenAI just cut API prices by 80 percent. That changes who can build.") == []
    issues = shotlist.hook_issues("Hello and welcome to the channel, today we talk about many interesting things")
    assert any("greeting" in i for i in issues) and any("number" in i for i in issues)


def test_prompt_has_enum_hook_and_chapter_rules():
    text = shotlist.prompt_rules()
    for kind in shotlist.KINDS:
        assert kind in text
    assert "158" in text and "chapter_card" in text and "Never 3 shots of the same kind" in text
    assert "No greeting" in text and "hook_number" in text and "ask" in text
