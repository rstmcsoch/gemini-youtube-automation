"""Topic relevance of a generated script, and which text models are strong enough to write one.

Why: when the strong Gemini models return 503, the fallback chain ends at a "lite" model.
On 3 Nov that model wrote an Engine B script about noise-cancelling earbuds that was a
history lesson with no AI in it, and every later check only warned, so it shipped.

``relevance(topic, text)`` is a cheap keyword-overlap check (no network):
  * coverage: share of the topic's content words (stemmed, prefix-matched) found in the script;
  * for an AI topic, the script must name the AI mechanism (AI / model / neural / trained /
    machine learning / algorithm ...) at least ``max(2, words / 150)`` times.
Verdicts: ``off_topic`` (coverage < 0.5, or an AI topic with too few AI terms),
``on_topic`` (coverage >= 0.75), else ``borderline`` (the caller may ask the best available
strong model to decide; without an answer, borderline passes with a warning).

This file is shared by Engine A (src/topic_guard.py) and Engine B (verticals/topic_guard.py);
keep the two copies identical.
"""
import json
import os
import re

STOP = {
    "a", "an", "the", "of", "to", "in", "on", "by", "or", "and", "is", "are", "was", "were", "be", "how", "what",
    "why", "when", "where", "which", "who", "you", "your", "yours", "it", "its", "do", "does", "did", "for", "with",
    "from", "into", "can", "could", "will", "would", "should", "use", "uses", "used", "using", "makes", "make",
    "this", "that", "these", "those", "about", "really", "actually", "work", "works", "new", "now", "just", "get",
    "gets", "next", "inside", "behind", "explained", "explain", "secret", "secrets", "way", "ways", "every", "all",
}
AI_TOPIC_RE = re.compile(r"\b(ai|a\.i\.|artificial intelligence|machine learning|ml|neural|deep learning|"
                         r"llms?|gpt|algorithms?|computer vision)\b", re.I)
AI_TERM_RE = re.compile(
    r"\b(ai|a\.i\.|artificial intelligence|machine[- ]learning|deep[- ]learning|neural|networks?|models?|"
    r"algorithms?|trained|training|learns?|learned|learning|classif\w*|predict\w*|inference|embeddings?|"
    r"transformers?|llms?|weights?|datasets?|vectors?|tokens?|detectors?|recogni[sz]\w*)\b", re.I)
AI_WORDS = {"ai", "artificial", "intelligence", "machine", "learning", "ml", "neural", "deep", "llm", "llms",
            "algorithm", "algorithms"}
WORD_RE = re.compile(r"[a-z0-9][a-z0-9']*")
WEAK_MODEL_TOKENS = ("lite", "nano", "mini", "tiny", "small", "8b", "7b", "3b", "1b")

OFF_TOPIC, BORDERLINE, ON_TOPIC = "off_topic", "borderline", "on_topic"
FAIL_COVERAGE, PASS_COVERAGE = 0.5, 0.75
SCRIPT_UNAVAILABLE = "script model unavailable"


def is_weak_model(name):
    """True for models too weak to write a script ('*-lite', '*-nano'...); $DYPOL_WEAK_MODELS adds names,
    $DYPOL_STRONG_MODELS overrides (comma-separated, exact names)."""
    name = str(name or "").strip().lower()
    strong = {m.strip().lower() for m in os.environ.get("DYPOL_STRONG_MODELS", "").split(",") if m.strip()}
    if name in strong:
        return False
    weak = {m.strip().lower() for m in os.environ.get("DYPOL_WEAK_MODELS", "").split(",") if m.strip()}
    if name in weak:
        return True
    parts = re.split(r"[-_./:]", name)
    return any(tok in parts for tok in WEAK_MODEL_TOKENS)


def strong_models(models):
    return [m for m in models if not is_weak_model(m)]


def _stem(word):
    w = word.lower().strip("'")
    if w.endswith("'s"):
        w = w[:-2]
    for suf in ("ings", "ing", "ers", "er", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)]
    return w


def _matches(key, stem):
    if key == stem:
        return True
    n = min(len(key), len(stem))
    return n >= 5 and key[:5] == stem[:5] and (key.startswith(stem) or stem.startswith(key) or n >= 6
                                                and key[:6] == stem[:6])


def topic_keywords(topic):
    """Stemmed content words of the topic (no stop words, no AI words: those are checked separately)."""
    out = []
    for w in WORD_RE.findall(str(topic or "").lower().replace("-", " ")):
        if w in STOP or w in AI_WORDS or len(w) < 3 or w.isdigit():
            continue
        s = _stem(w)
        if s not in out:
            out.append(s)
    return out


def relevance(topic, text):
    """Keyword-overlap relevance of `text` (the narration) to `topic`; see the module docstring."""
    text = " ".join(str(text or "").split())
    words = WORD_RE.findall(text.lower().replace("-", " "))
    stems = {_stem(w) for w in words}
    keys = topic_keywords(topic)
    hits = [k for k in keys if any(_matches(k, s) for s in stems)]
    missing = [k for k in keys if k not in hits]
    coverage = (len(hits) / len(keys)) if keys else 1.0
    ai_topic = bool(AI_TOPIC_RE.search(str(topic or "")))
    ai_hits = len(AI_TERM_RE.findall(text)) if ai_topic else 0
    ai_needed = max(2, round(len(words) / 150)) if ai_topic else 0
    reasons = []
    if not words:
        reasons.append("empty script")
    if keys and coverage < FAIL_COVERAGE:
        reasons.append(f"topic words missing from the script: {', '.join(missing)}")
    if ai_topic and ai_hits < ai_needed:
        reasons.append(f"the AI mechanism is barely mentioned ({ai_hits} AI terms, need {ai_needed})")
    if reasons:
        verdict = OFF_TOPIC
    elif coverage >= PASS_COVERAGE:
        verdict = ON_TOPIC
    else:
        verdict = BORDERLINE
        reasons.append(f"only {len(hits)}/{len(keys)} topic words found (missing: {', '.join(missing)})")
    return {"verdict": verdict, "coverage": round(coverage, 2), "topic_words": keys, "missing": missing,
            "ai_topic": ai_topic, "ai_terms": ai_hits, "ai_terms_needed": ai_needed, "words": len(words),
            "reasons": reasons, "method": "keywords"}


def judge_prompt(topic, text):
    return (
        "You check whether a video narration script explains its topic. Topic: \"" + str(topic) + "\".\n"
        "Answer on_topic=true only if the script mainly explains this exact topic (for an AI topic: how the AI "
        "part works). A history lesson, a product ad or a different subject is off topic.\n"
        "Reply with JSON only: {\"on_topic\": true|false, \"reason\": \"<= 20 words\"}\n\nSCRIPT:\n"
        + " ".join(str(text or "").split())[:6000]
    )


def parse_judge(raw):
    """{'on_topic': bool, 'reason': str} from a model reply, or None when it can't be read."""
    s = str(raw or "").strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    m = re.search(r"\{.*\}", s, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except ValueError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("on_topic"), bool):
        return None
    return {"on_topic": data["on_topic"], "reason": str(data.get("reason") or "")[:200]}


def apply_judge(result, judged, model=""):
    """Fold a model verdict into a borderline keyword result."""
    if result.get("verdict") != BORDERLINE or not judged:
        return result
    out = dict(result)
    out["method"] = "keywords+model"
    out["judge_model"] = model
    out["judge_reason"] = judged.get("reason", "")
    if judged["on_topic"]:
        out["verdict"] = ON_TOPIC
    else:
        out["verdict"] = OFF_TOPIC
        out["reasons"] = list(out.get("reasons") or []) + [f"model says off topic: {judged.get('reason', '')}"]
    return out


def correction_note(topic, result):
    return (
        "\n\nTOPIC CORRECTION: the previous script drifted off the topic \"" + str(topic) + "\" ("
        + "; ".join(result.get("reasons") or ["off topic"]) + "). Explain exactly this topic: what goes in, "
        "what the AI/model does with it step by step, what comes out, and its limits. History, product "
        "launches and specs may only support that explanation. Return the complete JSON object again."
    )
