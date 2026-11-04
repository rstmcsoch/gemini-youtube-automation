"""Engine A Short rules (owner, 4 Nov 2026).

* Pace: the voice is never slowed below MIN_WPM (150 words a minute). A Short that is too short for
  the 50 s minimum gets ONE script rewrite with more content instead; only the gap that is left after
  that may be closed by slowing, and never below 150 WPM. Before this, the 50 s rule slowed the
  voice x0.90 (about 125 WPM) and it dragged.
* Hook: the Short's first sentence (the hook) carries a concrete number or a surprising specific,
  the same rule Engine B uses (verticals/beats.py hook_specific).
"""
import math
import re

MIN_WPM = 150.0

HOOK_NUMBER_RE = re.compile(
    r"\d|\b(two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|forty|fifty|sixty|"
    r"seventy|eighty|ninety|hundreds?|thousands?|millions?|billions?|trillions?|dozens?|half|twice|double|triple|"
    r"percent)\b", re.I)
HOOK_SPECIFIC_RE = re.compile(
    r"\b(faster|slower|smaller|bigger|larger|cheaper|better|worse|more|fewer|less|longer|shorter|stronger|"
    r"quieter|louder|sharper|earlier|later)\s+(\w+\s+){0,4}?than\b"
    r"|\b(milli|micro|nano|pico)(seconds?|meters?|metres?|watts?)\b"
    r"|\b(every|per|each|a|an)\s+(second|minute|hour|day|heartbeat|blink)\b"
    r"|\btimes\s+(a|per|faster|more|less|smaller|bigger)\b"
    r"|\b(world's|fastest|slowest|smallest|largest|biggest)\b", re.I)


def first_sentence(text):
    text = " ".join(str(text or "").split())
    parts = [x for x in re.split(r"(?<=[.!?])\s+", text) if x]
    return parts[0] if parts else ""


def hook_specific(text):
    """'number' or 'specific' when the first sentence (the hook) has one, else ''."""
    first = first_sentence(text)
    if HOOK_NUMBER_RE.search(first):
        return "number"
    if HOOK_SPECIFIC_RE.search(first):
        return "specific"
    return ""


def wpm(words, seconds):
    return 60.0 * words / seconds if seconds > 0 else 0.0


def slow_factor(words, seconds, target_s, min_wpm=MIN_WPM):
    """atempo factor (<= 1) that brings `seconds` towards `target_s` without going under `min_wpm`.
    1.0 when the voice is already at or below the floor, or long enough."""
    if seconds <= 0 or seconds >= target_s or words <= 0:
        return 1.0
    floor = min_wpm / wpm(words, seconds)          # slowing by f gives wpm * f
    return min(1.0, max(floor, seconds / target_s))


def words_needed(words, seconds, target_s):
    """Extra words for the voice to reach target_s at its own pace (+5 % margin)."""
    if seconds <= 0 or words <= 0 or seconds >= target_s:
        return 0
    return int(math.ceil(words * (target_s / seconds - 1.0) * 1.05)) + 3
