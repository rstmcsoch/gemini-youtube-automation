"""Spelled-out numbers in narration still give count_ups, hook numbers and word sync
(draft run 41646837309, 3 Nov 2026: 1 count_up in a 658 s video whose script spelled numbers)."""
from src import production as prod, shotlist, voice
from src.numwords import normalize


def test_normalize_is_index_preserving():
    words = "temperature two hundred eighty-eight point two Kelvin.".split()
    out = normalize(words)
    assert len(out) == len(words) and out[1] == "288.2" and out[2:6] == ["", "", "", ""] and out[6] == "Kelvin."


def test_normalize_cases():
    assert normalize("humidity sixty-five percent, wind".split())[1] == "65"
    assert normalize("in under ten seconds".split())[2] == "10"
    assert normalize("thirty-four point zero five North,".split())[0] == "34.05"
    assert normalize("seven hundred twenty steps".split())[0] == "720"
    assert normalize("three million users".split())[0] == "3000000"
    assert normalize("one thousand and five people".split())[0] == "1005"
    assert normalize("fourteen forty longitude".split())[0] == "1440"
    assert normalize("in twenty twenty-four the".split())[1] == "2024"
    assert normalize("point taken".split()) == ["point", "taken"]
    assert normalize("the model works".split()) == ["the", "model", "works"]
    assert normalize("ends at ten. Then".split())[2] == "10."


def test_spoken_number_reads_spelled_numbers():
    spec = prod.spoken_number("humidity sixty-five percent across the grid".split())
    assert spec["value"] == "65" and spec["suffix"] == "%"
    spec = prod.spoken_number("predicts two hundred eighty-eight point two kelvin tomorrow".split())
    assert spec["value"] == "288.2"
    assert prod.spoken_number("in twenty twenty-four we".split()) is None      # a year, not a count_up
    assert prod.spoken_number("one specific grid index".split()) is None       # 1 is not a stat


def test_find_number_keeps_word_index():
    words = [(w, i * 0.3, i * 0.3 + 0.25) for i, w in enumerate("it runs in under ten seconds flat".split())]
    assert prod._find_number(words, "10") == 4


def test_hook_check_accepts_spelled_number():
    issues = shotlist.hook_issues("Ten seconds. That is how long AI needs to forecast ten days of weather.")
    assert not any("no number" in i for i in issues)


def test_prompt_asks_for_digits():
    assert "as digits" in shotlist.prompt_rules()


def test_short_pace_can_reach_band_past_1_3():
    cfg = voice.settings()
    assert float(cfg["speed_max"]) >= 1.4
    assert voice.corrected_speed(143, 1.3, cfg) > 1.3
