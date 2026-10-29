"""Pure planning helpers for the explainer-style editor (no network, no rendering).

Everything here is deterministic so the timing/visual rules can be unit tested:
on-screen text limits, stock-tag relevance, shot planning, bullet reveal timing
and word-level caption timing.
"""
import math
import re

MAX_BULLETS = 3
MAX_BULLET_WORDS = 8

LONG_SHOT_TARGET = 4.6     # seconds; keeps every cut (incl. crossfade overlap) at <= ~5 s
SHORT_SHOT_TARGET = 3.8    # seconds; Shorts cut every ~3-4 s
XFADE = 0.4                # crossfade between shots and slides (0.3-0.5 s)

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "what", "when", "why", "how", "are", "was", "were",
    "will", "can", "its", "it's", "you", "your", "our", "their", "they", "from", "into", "about",
    "just", "more", "most", "than", "then", "there", "here", "have", "has", "had", "been", "being",
    "today", "means", "mean", "reportedly", "plans", "chapter", "part", "lesson", "think", "look",
    "understand", "simple", "beneath", "headlines", "modern", "new", "now", "all", "one", "also",
    "like", "some", "very", "really", "these", "those", "which", "who", "does", "not", "but", "use",
    "using", "used", "make", "makes", "made", "get", "gets", "need", "needs", "don", "doesn", "isn",
    "every", "each", "only", "even", "still", "way", "ways", "thing", "things", "work", "works",
    "imagine", "explained", "matters", "teams", "team", "everyday", "via", "per", "out", "off",
    "over", "under", "between", "while", "where", "would", "could", "should", "may", "might",
}

# Brand/organisation tokens: never a reason to accept a stock hit, never sent to FLUX.
BRANDS = {
    "google", "deepmind", "gemini", "openai", "chatgpt", "gpt", "microsoft", "copilot", "apple",
    "iphone", "meta", "facebook", "instagram", "whatsapp", "amazon", "aws", "anthropic", "claude",
    "nvidia", "tesla", "samsung", "intel", "amd", "ibm", "oracle", "github", "youtube", "android",
    "windows", "linux", "pixel", "galaxy", "xai", "grok", "mistral", "llama", "huggingface", "x",
    "twitter", "tiktok", "netflix", "adobe", "salesforce", "cloudflare", "dypol",
}

# Words that match stock tags for the wrong reason ("AI model" vs fashion model, "human in
# the loop" vs a rope loop). They may appear in queries but never count as relevance proof.
AMBIGUOUS = {
    "model", "models", "loop", "loops", "field", "shield", "human", "humans", "people", "person",
    "man", "men", "woman", "women", "girl", "boy", "child", "children", "ahead", "road", "core",
    "challenge", "advantage", "playing", "image", "images", "picture", "photo", "background",
    "abstract", "concept", "style", "vintage", "retro", "free", "beautiful", "nature", "art",
    "design", "idea", "ideas", "business", "smile", "portrait", "release", "releases", "launch",
}

ANIMALS = {
    "cat", "cats", "kitten", "kitty", "dog", "dogs", "puppy", "pet", "pets", "animal", "animals",
    "bird", "birds", "horse", "tiger", "lion", "fish", "rabbit", "bunny", "hamster", "parrot",
    "cow", "sheep", "goat", "pig", "monkey", "bear", "fox", "wolf", "owl", "duck", "insect",
    "butterfly", "spider", "snake", "wildlife", "zoo", "feline", "canine", "mammal",
}

# Off-topic or sensitive imagery that keeps sneaking in through ambiguous tags.
OFF_TOPIC = {
    "cosplay", "corset", "princess", "fashion", "wedding", "bride", "lingerie", "bikini", "makeup",
    "depression", "suicide", "hang", "hanging", "noose", "rope", "gun", "weapon", "blood", "death",
    "funeral", "war", "soldier", "warrior", "religion", "church", "politics", "flag", "christmas",
    "halloween", "food", "cake", "recipe", "drink", "alcohol",
}

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9'+-]*")


def _stem(token):
    t = token.casefold().strip("'-+.")
    for suffix in ("ies", "es", "s"):
        if len(t) > 4 and t.endswith(suffix) and not t.endswith("ss"):
            return t[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return t


def tokens(text):
    """Lower-cased, stemmed content tokens of a text."""
    out = []
    for word in _WORD.findall(str(text or "")):
        k = _stem(word)
        if len(k) < 3 and k not in {"ai", "ml"}:
            continue
        if k in STOPWORDS:
            continue
        out.append(k)
    return out


def topic_terms(*texts):
    """Terms that can prove a stock hit is on-topic (no brands, no ambiguous words)."""
    terms = set()
    for text in texts:
        if isinstance(text, (list, tuple, set)):
            text = " ".join(str(t) for t in text)
        for k in tokens(text):
            if k in BRANDS or k in AMBIGUOUS or k in {"ai", "ml"}:
                continue
            terms.add(k)
    return terms


# Stock hits must visibly show technology. Metaphor matches (a seal for "phishing",
# fire for "firewall", chess for "cat-and-mouse game", a heart padlock, a sprinkler
# sign) all passed the topic-overlap check in the first redesigned draft.
TECH_ANCHORS_RAW = {
    "computer", "server", "network", "data", "datacenter", "code", "coding", "programming",
    "programmer", "software", "hardware", "cyber", "cybersecurity", "security", "hacker", "hacking",
    "technology", "tech", "digital", "chip", "processor", "cpu", "gpu", "circuit", "electronics",
    "electronic", "laptop", "keyboard", "monitor", "screen", "internet", "cable", "fiber", "optic",
    "robot", "robotics", "artificial", "intelligence", "algorithm", "smartphone", "binary", "matrix",
    "router", "encryption", "firewall", "virus", "malware", "password", "website", "dashboard",
    "analytics", "automation", "engineer", "semiconductor", "silicon", "microchip", "motherboard",
    "email", "e-mail", "spam", "phishing", "radar", "hacker", "coder", "developer",
}
TECH_ANCHORS = {_stem(w) for w in TECH_ANCHORS_RAW} | TECH_ANCHORS_RAW


def tag_relevance(tags, terms):
    """Return (ok, overlap, reason) for a stock hit's comma-separated tags."""
    tag_tokens = set()
    for tag in str(tags or "").split(","):
        tag_tokens.update(_stem(w) for w in _WORD.findall(tag))
    terms = set(terms or ())
    raw_topic = set(terms)
    animals = (tag_tokens & ANIMALS) - raw_topic
    if animals:
        return False, set(), f"animal tags not in topic: {sorted(animals)[:3]}"
    off = (tag_tokens & OFF_TOPIC) - raw_topic
    if off:
        return False, set(), f"off-topic tags: {sorted(off)[:3]}"
    overlap = (tag_tokens & terms) - BRANDS - AMBIGUOUS
    if not overlap:
        return False, set(), "no tag overlaps topic keywords"
    if not (tag_tokens & TECH_ANCHORS):
        return False, set(), "no technology subject in tags"
    return True, overlap, "ok"


def _clean_line(text):
    text = re.sub(r"[#*_`\"“”]+", "", str(text or ""))
    text = re.sub(r"\s+", " ", text).strip(" -–—•:;,.")
    return text


def clamp_bullet(text, max_words=MAX_BULLET_WORDS):
    words = _clean_line(text).split()
    if len(words) > max_words:
        words = words[:max_words]
        # Do not end on a dangling connector.
        while len(words) > 3 and words[-1].casefold().strip(",;:") in STOPWORDS | {"a", "an", "of", "to", "in", "on", "by", "or"}:
            words.pop()
    return " ".join(words).rstrip(",;:")


_FILLER = re.compile(r"^(imagine|think of it like|think of|picture|in short|basically|so|now|well|today),?\s+", re.I)


def bullets_from_paragraph(paragraph, limit=MAX_BULLETS):
    """Backward-compatible fallback when the model did not return bullets."""
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", str(paragraph or "").strip()) if s.strip()]
    bullets = []
    for sentence in sentences:
        sentence = _FILLER.sub("", sentence.strip())
        clause = re.split(r"[,;:—–]| - ", sentence)[0]
        bullet = clamp_bullet(clause)
        if len(bullet.split()) >= 3 and bullet.casefold() not in {b.casefold() for b in bullets}:
            bullets.append(bullet[:1].upper() + bullet[1:])
        if len(bullets) >= limit:
            break
    return bullets


def normalize_bullets(raw, paragraph=""):
    items = raw if isinstance(raw, list) else ([raw] if isinstance(raw, str) and raw.strip() else [])
    bullets = []
    for item in items:
        b = clamp_bullet(item)
        if b and b.casefold() not in {x.casefold() for x in bullets}:
            bullets.append(b)
        if len(bullets) >= MAX_BULLETS:
            break
    return bullets or bullets_from_paragraph(paragraph)


def strip_brands(text):
    words = []
    for word in str(text or "").split():
        core = re.sub(r"[^A-Za-z0-9]", "", word).casefold()
        if core in BRANDS:
            continue
        # CamelCase product/company names (DeepMind, OpenAI, ChatGPT).
        if re.match(r"^[A-Z][a-z]+[A-Z]", re.sub(r"[^A-Za-z]", "", word)):
            continue
        words.append(word)
    return " ".join(words)


def normalize_queries(raw, fallback_text=""):
    items = raw if isinstance(raw, list) else []
    out = []
    for item in items:
        q = " ".join(strip_brands(_clean_line(item)).split()[:4])
        if q and q.casefold() not in {x.casefold() for x in out}:
            out.append(q)
        if len(out) >= 3:
            break
    if not out:
        terms = [t for t in tokens(strip_brands(fallback_text)) if t not in AMBIGUOUS and t not in BRANDS]
        seen = []
        for t in terms:
            if t not in seen:
                seen.append(t)
        for i in range(0, min(len(seen), 6), 2):
            out.append(" ".join(seen[i:i + 2]))
        out = out[:3] or ["technology"]
    return out


_FLUX_BANNED = re.compile(
    r"\b(logo|logos|text|texts|word|words|letter|letters|typography|caption|captions|headline|label|labels|"
    r"sign|signs|signage|banner|watermark|brand|brands|trademark|website|webpage|screenshot|dashboard|"
    r"newspaper|book cover|poster|code on screen|written|writing)\b", re.I)


def flux_prompt(scene, orientation="landscape"):
    """FLUX prompt without brands, logos, text or screens carrying writing."""
    scene = strip_brands(re.sub(r"[\"“”'‘’][^\"“”'‘’]*[\"“”'‘’]", " ", str(scene or "")))
    scene = _FLUX_BANNED.sub(" ", scene)
    scene = re.sub(r"\s+", " ", scene).strip(" ,.") or "abstract visualisation of digital technology"
    frame = "wide 16:9 cinematic composition" if orientation == "landscape" else "vertical 9:16 composition"
    return (f"Editorial photo-illustration, {frame}, soft cinematic lighting, rich detail: {scene}. "
            "No text, no letters, no numbers, no logos, no brand names, no watermark, "
            "no screens or monitors showing writing.")


def normalize_slide(slide, lesson_title=""):
    """Turn model output (old or new schema) into the editor's slide spec."""
    slide = dict(slide or {})
    title = _clean_line(slide.get("title", "")) or lesson_title
    content = str(slide.get("content", "") or slide.get("narration", "") or "").strip()
    bullets = normalize_bullets(slide.get("bullets"), content)
    queries = normalize_queries(slide.get("visual_queries"), f"{title} {content}")
    scene = slide.get("image_prompt") or f"{', '.join(queries)}"
    return {
        "title": title,
        "content": content,
        "bullets": bullets,
        "visual_queries": queries,
        "image_prompt": str(scene),
        "kicker": slide.get("kicker", ""),
    }


# ---------------------------------------------------------------- timing

def plan_shots(duration, target):
    """Evenly spaced cut points so each shot is <= target seconds (min two shots)."""
    duration = max(0.1, float(duration))
    n = max(2, math.ceil(duration / target))
    step = duration / n
    return [(round(i * step, 4), round((i + 1) * step, 4)) for i in range(n)]


def unique_visual_count(n_shots):
    return max(2, min(4, n_shots))


def build_timeline(slide_durations, target=LONG_SHOT_TARGET):
    """Global shot list. Slides are back-to-back, so video length == narration length."""
    shots, slides, t = [], [], 0.0
    for index, duration in enumerate(slide_durations):
        start, end = t, t + float(duration)
        local = plan_shots(duration, target)
        slide_shots = []
        for j, (a, b) in enumerate(local):
            shots.append({"slide": index, "shot": j, "start": round(start + a, 4), "end": round(start + b, 4)})
            slide_shots.append(len(shots) - 1)
        slides.append({"index": index, "start": round(start, 4), "end": round(end, 4), "shots": slide_shots})
        t = end
    return {"total": round(t, 4), "slides": slides, "shots": shots}


def coverage_gaps(timeline, fps=30):
    """Times (sampled per frame) where no shot is visible: these render black."""
    gaps = []
    total = timeline["total"]
    shots = timeline["shots"]
    frames = int(round(total * fps))
    for f in range(frames):
        t = f / fps
        if not any(s["start"] - XFADE / 2 <= t < s["end"] + XFADE / 2 for s in shots):
            gaps.append(round(t, 3))
    return gaps


def assign_assets(n_shots, assets, seed=0):
    """Map shots to assets: each unique asset once first, extra shots reuse only own/AI assets.

    Third-party stills and videos are never repeated (rights/"used once" rule).
    """
    if not assets:
        return []
    order = list(range(len(assets)))
    reusable = [i for i, a in enumerate(assets) if a.get("reusable")]
    plan = []
    for k in range(n_shots):
        if k < len(order):
            plan.append(order[k])
        elif reusable:
            plan.append(reusable[(k - len(order) + seed) % len(reusable)])
        else:
            plan.append(None)  # caller supplies a generated graphic
    # Avoid the same reusable asset in two consecutive shots when possible.
    for k in range(1, len(plan)):
        if plan[k] is not None and plan[k] == plan[k - 1] and len(reusable) > 1:
            alt = [r for r in reusable if r != plan[k - 1] and (k + 1 >= len(plan) or r != plan[k + 1])]
            if alt:
                plan[k] = alt[0]
    return plan


def estimate_word_timings(text, duration, speech_regions=None, lead=0.05):
    """Approximate per-word (word, start, end) for TTS audio without a forced aligner.

    Words are weighted by length plus punctuation pauses and mapped onto the detected
    speech regions (silences removed), which keeps captions within ~0.2 s on gTTS.
    """
    words = str(text or "").split()
    if not words:
        return []
    duration = max(0.1, float(duration))
    regions = [(max(0.0, a), min(duration, b)) for a, b in (speech_regions or []) if b - a > 0.05]
    if not regions:
        regions = [(min(lead, duration * 0.1), duration)]
    speech_total = sum(b - a for a, b in regions)
    weights = []
    for w in words:
        weight = len(re.sub(r"[^A-Za-z0-9]", "", w)) + 2.0
        if w.endswith((",", ";", ":")):
            weight += 2.0
        elif w.endswith((".", "!", "?")):
            weight += 3.5
        weights.append(weight)
    total_w = sum(weights)

    def to_time(speech_t):
        acc = 0.0
        for a, b in regions:
            if speech_t <= acc + (b - a) + 1e-9:
                return a + (speech_t - acc)
            acc += b - a
        return regions[-1][1]

    out, acc = [], 0.0
    for w, weight in zip(words, weights):
        s = to_time(acc / total_w * speech_total)
        acc += weight
        e = to_time(acc / total_w * speech_total)
        out.append((w, round(s, 3), round(max(e, s + 0.05), 3)))
    return out


def bullet_reveal_times(bullets, word_times, duration, min_first=0.6, min_gap=1.2, tail=1.5):
    """When each bullet should animate in: as the narration reaches its words, else evenly."""
    if not bullets:
        return []
    duration = float(duration)
    stems = [_stem(re.sub(r"[^A-Za-z0-9]", "", w)) for w, _, _ in word_times]
    found, cursor = [], 0
    for bullet in bullets:
        keys = {k for k in tokens(bullet) if len(k) >= 4}
        best, best_i = 0, None
        for i in range(cursor, len(stems)):
            if stems[i] not in keys:
                continue  # windows start on a matching word, so the reveal lands on it
            score = len(keys & set(stems[i:i + 7]))
            if score > best:
                best, best_i = score, i
        if best_i is not None:
            found.append(word_times[best_i][1])
            cursor = best_i + 1
        else:
            found.append(None)
    n = len(bullets)
    even = [duration * (0.12 + 0.6 * k / max(1, n - 1 if n > 1 else 1)) for k in range(n)] if n > 1 else [duration * 0.15]
    times, last = [], -1e9
    latest = max(min_first, duration - tail)
    for k in range(n):
        t = found[k] if found[k] is not None else even[k]
        t = max(t, min_first, last + min_gap)
        t = min(t, latest)
        times.append(round(t, 3))
        last = t
    # If clamping at the tail collapsed the order, spread them back evenly.
    if any(b - a < min_gap * 0.5 for a, b in zip(times, times[1:])):
        span = max(0.1, latest - min_first)
        times = [round(min_first + span * k / max(1, n - 1), 3) if n > 1 else round(min_first, 3) for k in range(n)]
    return times


def caption_chunks(word_times, max_words=3, max_chars=18):
    """Group words into short caption chunks; break after punctuation."""
    chunks, current = [], []
    for item in word_times:
        current.append(item)
        word = item[0]
        text_len = len(" ".join(w for w, _, _ in current))
        if len(current) >= max_words or word.endswith((".", ",", "!", "?", ";", ":")) or text_len >= max_chars:
            chunks.append(current)
            current = []
    if current:
        chunks.append(current)
    return chunks
