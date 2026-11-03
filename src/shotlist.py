"""Shot-list schema for Engine A: every shot has a block kind from a closed enum.

The model proposes shots per chapter; `enforce` repairs them so that
`validate` always passes afterwards (enum only, never 3 same kinds in a row,
each chapter opens on a chapter_card, renderer payloads are usable).
"""
import re

from src import style

KINDS = style.block_kinds()
RENDERED = set(style.rendered_kinds())
STILL = "stock_or_ai_still"
CHAPTER = "chapter_card"

ALIASES = {
    "kinetic_type": "kinetic_text", "text": "kinetic_text", "kinetic": "kinetic_text", "title": "kinetic_text",
    "stat_slam": "count_up", "counter": "count_up", "stat": "count_up", "number": "count_up",
    "terminal": "code_panel", "code": "code_panel", "editor": "code_panel",
    "diagram_build": "diagram", "flow": "diagram", "flowchart": "diagram",
    "chapter": "chapter_card", "section_card": "chapter_card",
    "still": "stock_or_ai_still", "stock": "stock_or_ai_still", "image": "stock_or_ai_still",
    "ai_still": "stock_or_ai_still", "photo": "stock_or_ai_still", "video": "stock_or_ai_still",
}

_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def max_run():
    return int(style.long_rules()["variety"]["max_same_kind_run"])


def _kind(value):
    k = re.sub(r"[^a-z_]", "", str(value or "").strip().lower().replace(" ", "_").replace("-", "_"))
    k = ALIASES.get(k, k)
    return k if k in KINDS else None


def _words(value, limit):
    return " ".join(str(value or "").split()[:limit]).strip()


def _keywords(raw):
    if isinstance(raw, str):
        raw = re.split(r"[,;]", raw)
    out = []
    for item in raw or []:
        item = _words(item, 4)
        if item and item.lower() not in {o.lower() for o in out}:
            out.append(item)
    return out[:4]


def parse_number(value):
    """'94.5%' -> (94.5, 1 decimal); None when no number."""
    m = _NUM.search(str(value if value is not None else ""))
    if not m:
        return None
    text = m.group(0).replace(",", "")
    try:
        return float(text), (len(text.split(".")[1]) if "." in text else 0)
    except ValueError:
        return None


def _payload_ok(shot):
    """True when the renderer for this kind has what it needs."""
    kind = shot["kind"]
    if kind == "kinetic_text":
        return bool(shot.get("text"))
    if kind == "count_up":
        return parse_number(shot.get("value")) is not None
    if kind == "code_panel":
        return bool(str(shot.get("code", "")).strip())
    if kind == "diagram":
        cfg = style.guide()["blocks"]["diagram"]
        return cfg["min_nodes"] <= len(shot.get("nodes") or []) <= cfg["max_nodes"]
    if kind == CHAPTER:
        return bool(shot.get("title"))
    return True


def normalize_shot(raw):
    """Coerce one model shot into the schema. Unknown kinds fall back (never pass through)."""
    raw = dict(raw or {}) if isinstance(raw, dict) else {"kind": raw}
    asked = _kind(raw.get("kind") or raw.get("block") or raw.get("type"))
    fallback = _kind(raw.get("fallback")) or style.guide()["blocks"]["default_fallback"]
    shot = {"kind": asked or fallback, "keywords": _keywords(raw.get("keywords")), "fallback": fallback}
    if shot["fallback"] == shot["kind"]:
        shot["fallback"] = STILL if shot["kind"] != STILL else "kinetic_text"
    text = _words(raw.get("text") or raw.get("on_screen_text") or "", 10)
    if text:
        shot["text"] = text
    if raw.get("value") not in (None, ""):
        shot["value"] = str(raw.get("value"))[:24]
    for key, limit in (("label", 6), ("prefix", 2), ("suffix", 2), ("title", 6), ("source", 8)):
        if raw.get(key):
            shot[key] = _words(raw[key], limit)
    if raw.get("code"):
        lines = str(raw["code"]).replace("\t", "    ").splitlines()
        shot["code"] = "\n".join(line[:64] for line in lines[: style.guide()["blocks"]["code_panel"]["max_lines"]])
        shot["language"] = str(raw.get("language") or "python").lower()[:20]
        if raw.get("highlight_line") not in (None, ""):
            try:
                shot["highlight_line"] = int(raw["highlight_line"])
            except (TypeError, ValueError):
                pass
    if raw.get("nodes"):
        nodes = [_words(n.get("label") if isinstance(n, dict) else n, 3) for n in raw["nodes"]]
        shot["nodes"] = [n for n in nodes if n][: style.guide()["blocks"]["diagram"]["max_nodes"]]
    if raw.get("number") not in (None, ""):
        shot["number"] = raw["number"]
    return shot


def _degrade(shot, keywords_text=""):
    """Swap a shot for its fallback kind, keeping keywords (used for payload gaps and runs)."""
    new = dict(shot, kind=shot["fallback"])
    new["fallback"] = STILL if new["kind"] != STILL else "kinetic_text"
    if new["kind"] == "kinetic_text" and not new.get("text"):
        new["text"] = _words(keywords_text or " ".join(shot.get("keywords") or []), 6)
        if not new["text"]:
            new["kind"] = STILL
    if not _payload_ok(new):
        new["kind"] = STILL
    return new


def _break_runs(shots, limit):
    out = []
    for shot in shots:
        run = 0
        for prev in reversed(out):
            if prev["kind"] != shot["kind"]:
                break
            run += 1
        if run >= limit:
            alt = _degrade(shot)
            if alt["kind"] == shot["kind"]:
                alt = dict(shot, kind=STILL if shot["kind"] != STILL else "kinetic_text")
                if alt["kind"] == "kinetic_text" and not alt.get("text"):
                    alt["text"] = _words(" ".join(shot.get("keywords") or []) or "Here's the point", 6)
            shot = alt
        out.append(shot)
    return out


def enforce(raw_shots, chapter_title="", chapter=True):
    """Repair a chapter's shots: enum, payloads, chapter_card first, no run > max_run()."""
    shots = [normalize_shot(s) for s in (raw_shots or []) if s]
    shots = [s if _payload_ok(s) else _degrade(s) for s in shots]
    if chapter:
        shots = [s for s in shots if s["kind"] != CHAPTER] if shots and shots[0]["kind"] != CHAPTER else \
            shots[:1] + [s for s in shots[1:] if s["kind"] != CHAPTER]
        if not shots or shots[0]["kind"] != CHAPTER:
            shots.insert(0, {"kind": CHAPTER, "title": _words(chapter_title, 6) or "Next up",
                             "keywords": [], "fallback": "kinetic_text", "text": _words(chapter_title, 6)})
        elif not shots[0].get("title"):
            shots[0]["title"] = _words(chapter_title, 6) or "Next up"
    else:
        shots = [s for s in shots if s["kind"] != CHAPTER]
    return _break_runs(shots, max_run())


def validate(shots, chapter=True):
    """List of rule violations (empty == valid)."""
    errors = []
    for i, shot in enumerate(shots or []):
        if shot.get("kind") not in KINDS:
            errors.append(f"shot {i}: kind {shot.get('kind')!r} not in enum")
        elif not _payload_ok(shot):
            errors.append(f"shot {i}: {shot['kind']} payload incomplete")
        if shot.get("fallback") not in KINDS:
            errors.append(f"shot {i}: fallback {shot.get('fallback')!r} not in enum")
    errors += [f"run of {n} x {k} at shot {i}" for i, k, n in runs_over(
        [s.get("kind") for s in shots or []], max_run())]
    if chapter and shots and shots[0].get("kind") != CHAPTER:
        errors.append("chapter does not open on a chapter_card")
    return errors


def runs_over(kinds, limit):
    """[(start_index, kind, length)] for runs of one kind longer than `limit`."""
    out, i = [], 0
    kinds = list(kinds)
    while i < len(kinds):
        j = i
        while j + 1 < len(kinds) and kinds[j + 1] == kinds[i]:
            j += 1
        if j - i + 1 > limit:
            out.append((i, kinds[i], j - i + 1))
        i = j + 1
    return out


def hook_issues(hook_line, wpm=None):
    """Check the spoken cold open against the hook rule: short first sentence, a number by ~8 s."""
    rules = style.long_rules()
    wpm = wpm or rules["narration"]["wpm_target"]
    words = str(hook_line or "").split()
    issues = []
    if not words:
        return ["no hook line"]
    if re.match(r"(?i)^(hello|hi|hey|welcome)\b", words[0]):
        issues.append("hook opens with a greeting (no intro allowed)")
    first = re.split(r"(?<=[.!?])\s", " ".join(words), maxsplit=1)[0].split()
    if len(first) > rules["hook"]["max_first_sentence_words"]:
        issues.append(f"first sentence has {len(first)} words (> {rules['hook']['max_first_sentence_words']})")
    by_number = int(rules["hook"]["number_by_s"] * wpm / 60)
    from src.numwords import normalize
    if not any(_NUM.search(w) for w in normalize(words[:by_number]) if w):
        issues.append(f"no number in the first {by_number} words (~{rules['hook']['number_by_s']:.0f} s)")
    return issues


def shot_kinds_share(kinds):
    kinds = list(kinds)
    n = max(1, len(kinds))
    demo = set(style.long_rules()["variety"]["demo_kinds"])
    return {"demo_share": round(sum(k in demo for k in kinds) / n, 3),
            "still_share": round(sum(k == STILL for k in kinds) / n, 3)}


PROMPT_RULES = """
        SHOT LIST RULES (Engine A long video, DYPOL style guide):
        - Narration pace is ~{wpm} words per minute. Each chapter object is one chapter of {cmin}-{cmax} seconds
          of narration ({wmin}-{wmax} words).
        - "shots": a list of 8 to 14 shot objects per chapter (one every ~6-10 s of narration), in narration order. Every shot has:
            "kind": exactly one of {kinds}
            "keywords": 1-3 words from this chapter's narration that the shot appears on
            "fallback": another kind from the same list, used if this one cannot be drawn
          plus the fields its kind needs:
            kinetic_text: "text" (1-6 words, the key phrase, never the full narration sentence)
            count_up: "value" (the number, e.g. "94.5"), optional "prefix" ("$"), "suffix" ("%", "x", "ms"),
                      "label" (2-5 plain words naming what the number counts, readable on its own,
                      e.g. "dimensions per word", "of code suggestions kept"; never a sentence fragment
                      or a unit code like "512D"), optional "source" (e.g. "OpenAI blog, Oct 2026")
            code_panel: "code" (2-8 short lines of real, runnable code or shell commands), "language"
                      ("python", "bash", "javascript", "json"...), optional "title" (file name), optional
                      "highlight_line" (1-based line to light up)
            diagram: "nodes" (2-5 labels of 1-3 words, in the order they connect), optional "title"
            chapter_card: "title" (2-5 words)
            stock_or_ai_still: only for real-world context (a data centre, a phone in hand), never abstract ideas
        - The FIRST shot of every chapter is a chapter_card.
        - Never 3 shots of the same kind in a row.
        - At least 40% of shots are demo-style (code_panel, diagram, count_up); stock_or_ai_still at most 15%.
        - Every number you say gets a count_up shot.
        - Write every number in the narration as digits ("288.2 Kelvin", "65 percent", "10 seconds"), never spelled
          out ("two hundred eighty-eight"). The voice reads digits naturally and the counters need them.
        - Something new appears on screen every 2-4 seconds: code types, diagram steps build, numbers count.
"""

HOOK_RULES = """
        HOOK AND ENDING RULES:
        - "hook": the cold open, spoken first. No greeting, no channel name, no "welcome", no "today we will".
          The first sentence (at most 14 words) names the subject and what happened. By the third second say why it
          matters (the stakes). Within the first 20 words say one concrete number.
        - "hook_number": that number as {{"value": "...", "prefix": "", "suffix": "", "label": "2-5 plain words naming
          what it counts", "source": "publisher + year if it is a published figure"}}.
        - "takeaway": one sentence, the single thing to remember; "takeaway_short": the same in at most 6 words.
        - "ask": exactly one specific comment question for the viewer, at most 12 words (no "like and subscribe").
"""


def prompt_rules():
    r = style.long_rules()
    wpm = r["narration"]["wpm_target"]
    cmin, cmax = r["chapters"]["min_interval_s"], r["chapters"]["max_interval_s"]
    return (PROMPT_RULES.format(wpm=wpm, cmin=cmin, cmax=cmax, wmin=int(cmin * wpm / 60),
                                wmax=int(cmax * wpm / 60), kinds=", ".join(KINDS))
            + HOOK_RULES.format())
