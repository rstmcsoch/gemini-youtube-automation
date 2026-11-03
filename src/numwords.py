"""Spelled-out English numbers -> digits, index-preserving.

The narration is written for the voice, and the model sometimes spells numbers out
("two hundred eighty-eight point two", "sixty-five percent"). Count_up detection and
the hook check look for digits, so draft run 41646837309 (3 Nov 2026) got 1 count_up
instead of ~20. `normalize(words)` returns a list of the same length: the first word of
each spelled number becomes its digits (with trailing punctuation kept) and the rest of
that span become "" so word indexes and timings still line up.
"""
import re

_ONES = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
         "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
         "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
         "ninety": 90}
_SCALES = {"thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}
_PUNCT = ".,;:!?)\"'”"


def _bare(word):
    return str(word).strip("(\"'“”").rstrip(_PUNCT).lower()


def _parts(word):
    """'eighty-eight' -> ['eighty', 'eight']; None when it is not a number word."""
    bits = _bare(word).split("-")
    if bits and all(b in _ONES or b in _TENS or b in _SCALES or b == "hundred" for b in bits):
        return bits
    return None


def _parse(words, i):
    """(value_string, end_index) for a spelled number starting at words[i], else None."""
    total, current, j, seen = 0, 0, i, False
    while j < len(words):
        bits = _parts(words[j])
        if bits is None:
            if seen and _bare(words[j]) == "and" and j + 1 < len(words) and _parts(words[j + 1]) and current % 100 == 0 \
                    and (total or current):
                j += 1
                continue
            break
        trial_total, trial = total, current
        fits = True
        for b in bits:
            low = trial % 100
            if b in _ONES and _ONES[b] < 10:
                fits = low % 10 == 0 and not 10 <= low <= 19
                trial += _ONES[b]
            elif b in _ONES or b in _TENS:
                fits = low == 0
                trial += _ONES.get(b) or _TENS[b]
            elif b == "hundred":
                fits = 0 < trial < 100 or (trial == 0 and not seen)
                trial = max(trial, 1) * 100
            else:
                fits = trial_total == 0 or trial > 0
                trial_total += max(trial, 1) * _SCALES[b]
                trial = 0
            if not fits:
                break
        if not fits:
            break   # "fourteen forty" is two numbers, not 54
        total, current = trial_total, trial
        seen = True
        end_punct = str(words[j]).rstrip() != str(words[j]).rstrip().rstrip(_PUNCT)
        j += 1
        if end_punct:
            break
    if not seen:
        return None
    value = str(total + current)
    # "point five" / "point zero five" decimals
    if j + 1 < len(words) and _bare(words[j]) == "point" and str(words[j - 1]).rstrip(_PUNCT) == str(words[j - 1]):
        digits, k = "", j + 1
        while k < len(words):
            b = _bare(words[k])
            if b in _ONES and _ONES[b] < 10:
                digits += str(_ONES[b])
                stop = str(words[k]).rstrip(_PUNCT) != str(words[k])
                k += 1
                if stop:
                    break
            else:
                break
        if digits:
            value, j = f"{value}.{digits}", k
    return value, j


def normalize(words):
    words = [str(w) for w in words]
    out = list(words)
    i = 0
    while i < len(words):
        if _parts(words[i]) is None:
            i += 1
            continue
        parsed = _parse(words, i)
        if not parsed:
            i += 1
            continue
        value, end = parsed
        # Spoken years and pairs: "twenty twenty-four" -> 2024, "fourteen forty" -> 1440.
        if value.isdigit() and 10 <= int(value) <= 99 and end < len(words) \
                and str(words[end - 1]).rstrip(_PUNCT) == str(words[end - 1]) and _parts(words[end]):
            nxt = _parse(words, end)
            if nxt and nxt[0].isdigit() and 0 < int(nxt[0]) <= 99:
                value, end = str(int(value) * 100 + int(nxt[0])), nxt[1]
        tail = words[end - 1][len(words[end - 1].rstrip(_PUNCT)):]
        lead = re.match(r"^[(\"'“]*", words[i]).group(0)
        out[i] = f"{lead}{value}{tail}"
        for k in range(i + 1, end):
            out[k] = ""
        i = end
    return out
