"""AI visual-fit judge ("frame checker") for DYPOL engines.

One image per request to an OpenAI-compatible vision model on NVIDIA NIM
(https://integrate.api.nvidia.com/v1/chat/completions). The judge is a net on
top of the mechanical filters (tags / off-topic words / darkness / resolution /
dedupe), which always run first. It never blocks the pipeline: any failure
(no key, HTTP errors, unparseable output, caps, open circuit) returns ``None``
and callers keep today's mechanical behaviour.

This file is shared verbatim between Engine A (src/visual_judge.py) and
Engine B (verticals/visual_judge.py); it depends only on the standard library,
``requests`` and Pillow.

Configuration (environment):
  NVIDIA_API_KEY            nvapi-... key; missing => judge disabled.
  VISUAL_JUDGE_ENABLED      auto (default: on when a key exists) | 1/true | 0/false.
  VISUAL_JUDGE_MODELS       comma list, primary first (fallback chain).
  VISUAL_JUDGE_MIN_INTERVAL seconds between request starts (default 2.0).
  VISUAL_JUDGE_CACHE_PATH   JSON file for the score cache + rejection memory.
"""
import base64
import hashlib
import io
import json
import os
import random
import re
import subprocess
import time
from email.utils import parsedate_to_datetime
from pathlib import Path

PROMPT_VERSION = "fc-v3-2026-10-29"
API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_MODELS = ("nvidia/nemotron-3.5-super-vl-120b-a12b,meta/llama-3.2-90b-vision-instruct,"
                  "meta/llama-3.2-11b-vision-instruct")
REQUEST_TIMEOUT = 30.0
BACKOFF = (5.0, 15.0, 45.0)
MAX_RETRIES = 3
JITTER = 0.20
MAX_RETRY_AFTER = 60.0
MAX_CALLS_PER_VIDEO = 100
MAX_WALL_SECONDS_PER_VIDEO = 240.0
CIRCUIT_EXHAUSTED = 3
CIRCUIT_PARSE_FAILURES = 5
CANDIDATES_PER_SHOT = 5
BROADENED_CANDIDATES = 3
CONTACT_SHEET_COLS, CONTACT_SHEET_ROWS, CONTACT_SHEET_TILE = 4, 3, 384

ISSUES = ("wrong_subject", "generic_filler", "off_topic", "mood_mismatch", "sensitive_content",
          "watermark_or_text", "low_quality", "cluttered")
HARD_FAIL_ISSUES = frozenset({"wrong_subject", "sensitive_content", "watermark_or_text"})
GLOBAL_REJECT_ISSUES = frozenset({"sensitive_content", "watermark_or_text", "low_quality"})
QUERY_REJECT_ISSUES = frozenset({"wrong_subject", "off_topic", "generic_filler"})
PASS, MARGINAL, FAIL = "PASS", "MARGINAL", "FAIL"

SYSTEM_PROMPT = (
    "You are a strict visual QA judge for a short-form tech news channel.\n"
    "Decide whether one image is good enough to show full-screen while a\n"
    "specific narration sentence is spoken. Judge only what is literally\n"
    "visible, never what you assume."
)

USER_TEMPLATE = (
    "VIDEO TOPIC: {video_title}\n"
    "NARRATION LINE: \"{narration_line}\"\n"
    "INTENDED VISUAL (search query used): \"{query}\"\n"
    "\n"
    "Judge fit to the VIDEO TOPIC and NARRATION LINE. The search query is only a hint and is often a "
    "poor visual metaphor: an image that merely matches the query (a vintage lock, chess pieces, a coffee "
    "cup, a highway, a microscope, crystals, a staircase, a radar dish, a warning light, police, animals, "
    "scenery) but shows no technology is off_topic or wrong_subject for a tech story and scores 1-4, "
    "however literally it matches the query.\n"
    "Real technology imagery that a viewer instantly reads as this story's domain counts as on-topic "
    "(7-8) even when it cannot literally show an abstract idea: for security/AI/network stories that "
    "includes server rooms and racks, network cables and switches, people working at computers or "
    "keyboards, code or data on screens, and padlocks or shields shown with computers or circuits. Do not require a stock image to show AI itself, a "
    "brand, or an abstract process (no stock photo can); that alone is not generic_filler and such imagery "
    "should score 7-8. Use generic_filler for imagery outside the story's domain or meaningless decoration "
    "(abstract light streaks, crystals, random objects, decorative patterns). "
    "Video frames may show mild motion blur; only flag low_quality when the subject is unrecognisable.\n"
    "Rubric:\n"
    "9-10 = directly shows what the narration describes; connection is instant.\n"
    "7-8 = clearly on-topic for this line; obvious without explanation.\n"
    "5-6 = right general domain but wrong subject, or generic tech filler (abstract circuits, crystals, "
    "random hardware, pretty but meaningless).\n"
    "3-4 = connection needs explaining; likely to confuse viewers.\n"
    "1-2 = unrelated subject, or inappropriate for a consumer-tech audience.\n"
    "\n"
    "Hard fails (score must be \u22643 regardless of relevance): visible watermark or large burned-in text; "
    "graphic violence, distress, or riot/conflict imagery; heavy motion blur or extreme low resolution; "
    "collage/screenshot dominated by unreadable small text; prominent brand logos unrelated to the story.\n"
    "\n"
    "Return ONLY this JSON, no markdown: {{\"description\": \"<one sentence: what is literally visible>\", "
    "\"subject_match\": \"yes|partial|no\", \"issues\": [...], \"reason\": \"<one sentence>\", "
    "\"score\": <1-10>}}\n"
    "\n"
    "issues must be from: wrong_subject, generic_filler, off_topic, mood_mismatch, sensitive_content, "
    "watermark_or_text, low_quality, cluttered. Use [] if none."
)

CONTACT_SHEET_TEMPLATE = (
    "This grid shows all {n} shots of one video in numbered order. Narration per shot: {lines} "
    "Which numbered shots look off-topic, jarring, or inconsistent with their line? Return ONLY JSON: "
    "{{\"bad_shots\": [{{\"n\": <int>, \"reason\": \"...\"}}], \"overall\": \"ok\"|\"needs_review\"}}"
)


# ---------------------------------------------------------------------------
# Pure helpers: decision rule, JSON hygiene, query normalisation
# ---------------------------------------------------------------------------

def decide(verdict):
    """PASS / MARGINAL / FAIL from a normalised verdict (issues + subject_match beat the score)."""
    if not isinstance(verdict, dict):
        return FAIL
    score = verdict.get("score")
    issues = set(verdict.get("issues") or [])
    if issues & HARD_FAIL_ISSUES:
        return FAIL
    if not isinstance(score, (int, float)):
        return FAIL
    if score >= 7 and verdict.get("subject_match") != "no":
        return PASS
    # Calibrated 2026-10-30 (golden set, prompt v3): a score-6, on-domain image whose only complaint is
    # generic_filler is a usable tech b-roll shot; no historical reject scored 6. Accepts 83.7% -> 90.7%.
    if score >= 6 and verdict.get("subject_match") in ("yes", "partial") and issues <= {"generic_filler"}:
        return PASS
    if 5 <= score < 7:
        return MARGINAL
    return FAIL  # score < 5, or a high score that contradicts subject_match == "no" (bias strict)


def _strip_fences(text):
    text = re.sub(r"<think>.*?</think>", "", str(text or ""), flags=re.S | re.I)
    text = re.sub(r"^\s*```[a-zA-Z0-9_-]*\s*", "", text.strip())
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def _first_object(text):
    """First balanced {...} in text (string-aware), or None."""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        start = text.find("{", start + 1)
    return None


def extract_json(text):
    """Parse model output into a dict: strip fences/think blocks, json.loads, else first {...}."""
    cleaned = _strip_fences(text)
    for attempt in (cleaned, _first_object(cleaned) or ""):
        if not attempt:
            continue
        try:
            value = json.loads(attempt)
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict):
            return value
    return None


def normalize_verdict(raw):
    """Validate/normalise a parsed verdict; None when it is not a usable verdict."""
    if not isinstance(raw, dict):
        return None
    score = raw.get("score")
    if isinstance(score, str):
        m = re.search(r"\d+(?:\.\d+)?", score)
        score = float(m.group(0)) if m else None
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return None
    score = max(1.0, min(10.0, float(score)))
    score = int(score) if float(score).is_integer() else round(score, 1)
    match = str(raw.get("subject_match") or "").strip().casefold()
    if match not in {"yes", "partial", "no"}:
        match = "partial" if match.startswith("part") else ("yes" if match.startswith("y") else "no")
    issues = raw.get("issues") or []
    if isinstance(issues, str):
        issues = re.split(r"[,\s]+", issues)
    issues = [str(i).strip().casefold() for i in issues if str(i).strip().casefold() in ISSUES]
    issues = list(dict.fromkeys(issues))
    verdict = {
        "description": str(raw.get("description") or "")[:300],
        "subject_match": match,
        "issues": issues,
        "reason": str(raw.get("reason") or "")[:300],
        "score": score,
    }
    if set(issues) & HARD_FAIL_ISSUES and score > 3:
        verdict["score"] = 3  # rubric: hard fails score <= 3 regardless of relevance
    verdict["decision"] = decide(verdict)
    return verdict


def normalize_query(query):
    """Order/case/punctuation-insensitive form of a search query for rejection memory."""
    words = re.findall(r"[a-z0-9]+", str(query or "").casefold())
    return " ".join(sorted(dict.fromkeys(words)))


_GENERIC_HEADS = {"room", "center", "centre", "screen", "system", "systems", "technology", "tech", "scene",
                  "view", "shot", "background", "concept", "image", "photo", "device", "devices", "office",
                  "digital", "future", "world", "industry", "business"}


def broaden_query(query):
    """Drop the rarest (leading modifier) keywords: 'liquid cooling data center' -> 'data center'.

    English noun phrases keep their head at the end, so the trailing words are the
    broad subject; when that tail is only generic words ('room', 'system') the
    leading subject word is kept instead ('server room' -> 'server').
    Returns None when the query cannot be broadened.
    """
    words = [w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9+-]*", str(query or ""))]
    if len(words) <= 1:
        return None
    keep = words[-2:] if len(words) >= 3 else words[-1:]
    if all(w.casefold() in _GENERIC_HEADS for w in keep):
        keep = [w for w in words[:-1] if w.casefold() not in _GENERIC_HEADS][-1:] or keep
    broad = " ".join(keep)
    return broad if broad.casefold() != str(query).strip().casefold() else None


def _env_flag(value):
    return str(value or "auto").strip().casefold()


# ---------------------------------------------------------------------------
# Images: 512 px JPEG q75 normalisation (cache key + request payload), clip frames,
# contact sheets
# ---------------------------------------------------------------------------

def image_jpeg_bytes(source, max_side=512, quality=75):
    """Downscale an image (path or bytes) to <=max_side px, JPEG q75 bytes."""
    from PIL import Image
    data = Path(source).read_bytes() if not isinstance(source, (bytes, bytearray)) else bytes(source)
    with Image.open(io.BytesIO(data)) as im:
        im = im.convert("RGB")
        im.thumbnail((max_side, max_side), Image.LANCZOS)
        out = io.BytesIO()
        im.save(out, format="JPEG", quality=quality)
    return out.getvalue()


def score_cache_key(jpeg_bytes, prompt_version=PROMPT_VERSION):
    return hashlib.sha256(jpeg_bytes + str(prompt_version).encode()).hexdigest()


def probe_duration(path):
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                              "-of", "csv=p=0", str(path)], capture_output=True, text=True, timeout=30).stdout
        return float(out.strip() or 0)
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0.0


def extract_clip_frame(clip_path, fraction, out_path, duration=None):
    """One JPEG frame at `fraction` of the clip's duration (ffmpeg -ss t -frames:v 1 -q:v 4)."""
    duration = duration if duration else probe_duration(clip_path)
    if not duration or duration <= 0:
        return None
    t = max(0.0, min(duration - 0.05, duration * float(fraction)))
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", str(clip_path),
                        "-frames:v", "1", "-q:v", "4", str(out_path)],
                       check=True, capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return Path(out_path) if Path(out_path).exists() and Path(out_path).stat().st_size else None


def compose_contact_sheet(images, numbers, out_path, cols=CONTACT_SHEET_COLS, rows=CONTACT_SHEET_ROWS,
                          tile=CONTACT_SHEET_TILE):
    """4x3 grid of ~384 px tiles with each shot number burned into its tile corner."""
    from PIL import Image, ImageDraw, ImageFont
    sheet = Image.new("RGB", (cols * tile, rows * tile), (16, 16, 16))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 44)
    except OSError:
        try:
            font = ImageFont.load_default(size=44)
        except TypeError:
            font = ImageFont.load_default()
    for slot, (path, number) in enumerate(zip(images, numbers)):
        if slot >= cols * rows:
            break
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                im.thumbnail((tile, tile), Image.LANCZOS)
                x0 = (slot % cols) * tile + (tile - im.width) // 2
                y0 = (slot // cols) * tile + (tile - im.height) // 2
                sheet.paste(im, (x0, y0))
        except OSError:
            continue
        bx, by = (slot % cols) * tile + 6, (slot // cols) * tile + 6
        label = str(number)
        draw.rectangle((bx, by, bx + 30 + 26 * len(label), by + 56), fill=(0, 0, 0))
        draw.text((bx + 10, by + 4), label, fill=(255, 230, 0), font=font)
    sheet.save(out_path, format="JPEG", quality=85)
    return Path(out_path)


# ---------------------------------------------------------------------------
# Store: score cache + rejection memory behind a pluggable backend
# ---------------------------------------------------------------------------

class JsonFileBackend:
    """Local JSON file. Survives within one job (or across runs if the file is persisted)."""

    def __init__(self, path):
        self.path = Path(path)

    def load(self):
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def save(self, data):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError:
            pass  # cache is best-effort; never break the run


class MemoryBackend:
    def __init__(self, data=None):
        self.data = data or {}

    def load(self):
        return json.loads(json.dumps(self.data))

    def save(self, data):
        self.data = json.loads(json.dumps(data))


# TODO(R2): add an R2Backend(load/save) that GETs/PUTs one JSON object in the shared
# Cloudflare R2 bucket so both engines share the cache across runs (the plan's
# persistence layer). Select it in default_backend() when R2 credentials exist.
def default_backend(env=None):
    env = env if env is not None else os.environ
    path = env.get("VISUAL_JUDGE_CACHE_PATH", "").strip()
    if not path:
        base = env.get("VERTICALS_HOME") or env.get("DYPOL_WORK_DIR") or os.path.join(os.getcwd(), "output")
        path = os.path.join(base, "visual_judge_cache.json")
    return JsonFileBackend(path)


class JudgeStore:
    """Score cache (key = sha256(512px JPEG q75 + prompt_version)) and rejection memory."""

    def __init__(self, backend):
        self.backend = backend
        data = backend.load()
        self.scores = data.get("scores") if isinstance(data.get("scores"), dict) else {}
        self.global_rejects = data.get("global_rejects") if isinstance(data.get("global_rejects"), dict) else {}
        self.query_rejects = data.get("query_rejects") if isinstance(data.get("query_rejects"), dict) else {}

    def _save(self):
        self.backend.save({"version": 1, "scores": self.scores, "global_rejects": self.global_rejects,
                           "query_rejects": self.query_rejects})

    def get_verdict(self, key, query):
        """Cached verdict for this image: exact normalised-query match, or any context-free hard fail."""
        entry = self.scores.get(key)
        if not isinstance(entry, dict):
            return None
        nq = normalize_query(query)
        if isinstance(entry.get(nq), dict):
            return dict(entry[nq])
        for verdict in entry.values():
            if isinstance(verdict, dict) and set(verdict.get("issues") or []) & GLOBAL_REJECT_ISSUES:
                return dict(verdict)
        return None

    def put_verdict(self, key, query, verdict):
        self.scores.setdefault(key, {})[normalize_query(query)] = dict(verdict)
        self._save()

    def rejection(self, asset_key, query):
        if not asset_key:
            return None
        if asset_key in self.global_rejects:
            return "global:" + ",".join(self.global_rejects[asset_key].get("issues", []))
        q = self.query_rejects.get(asset_key) or {}
        nq = normalize_query(query)
        if nq in q:
            return "query:" + ",".join(q[nq].get("issues", []))
        return None

    def record_rejection(self, asset_key, query, issues):
        if not asset_key:
            return
        issues = set(issues or [])
        changed = False
        if issues & GLOBAL_REJECT_ISSUES:
            self.global_rejects[asset_key] = {"issues": sorted(issues & GLOBAL_REJECT_ISSUES), "at": int(time.time())}
            changed = True
        elif issues & QUERY_REJECT_ISSUES:
            self.query_rejects.setdefault(asset_key, {})[normalize_query(query)] = {
                "issues": sorted(issues & QUERY_REJECT_ISSUES), "at": int(time.time())}
            changed = True
        if changed:
            self._save()


# ---------------------------------------------------------------------------
# HTTP transport (mockable): returns (status, headers, text); raises on network error
# ---------------------------------------------------------------------------

def requests_transport(url, headers, body, timeout):
    import requests
    response = requests.post(url, headers=headers, data=body, timeout=timeout)
    return response.status_code, dict(response.headers or {}), response.text


class _Exhausted(Exception):
    pass


class _ModelGone(Exception):
    pass


class _CircuitTrip(Exception):
    pass


class _Capped(Exception):
    pass


def _retry_after_seconds(headers):
    value = None
    for k, v in (headers or {}).items():
        if str(k).casefold() == "retry-after":
            value = v
            break
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        pass
    try:
        dt = parsedate_to_datetime(str(value))
        return max(0.0, dt.timestamp() - time.time())
    except (TypeError, ValueError, IndexError, OverflowError):
        return None


class VisualJudge:
    def __init__(self, env=None, transport=None, sleep=None, clock=None, rng=None, store=None):
        self.env = dict(env if env is not None else os.environ)
        self.api_key = self.env.get("NVIDIA_API_KEY", "").strip()
        flag = _env_flag(self.env.get("VISUAL_JUDGE_ENABLED"))
        self.enabled = bool(self.api_key) and flag not in {"0", "false", "no", "off", "disabled"}
        self.disabled_reason = None if self.enabled else ("no NVIDIA_API_KEY" if not self.api_key
                                                         else "VISUAL_JUDGE_ENABLED=" + flag)
        models = self.env.get("VISUAL_JUDGE_MODELS", "").strip() or DEFAULT_MODELS
        self.models = [m.strip() for m in models.split(",") if m.strip()]
        try:
            self.min_interval = float(self.env.get("VISUAL_JUDGE_MIN_INTERVAL", "2.0"))
        except ValueError:
            self.min_interval = 2.0
        self.transport = transport or requests_transport
        self.sleep = sleep or time.sleep
        self.clock = clock or time.monotonic
        self.rng = rng or random.Random()
        self._store = store
        self.dead_models = {}
        self.compat = {}          # model -> 0 full | 1 no response_format/template kwargs | 2 no system role
        self.max_tokens = {}      # model -> raised after a reasoning-only, truncated answer
        self.last_request_at = None
        self.circuit_reason = None
        self.consecutive_exhausted = 0
        self.parse_failures = 0
        self.total_calls = 0
        self.total_cache_hits = 0
        self.videos = {}
        self.current = None
        self.begin_video("default")

    # ----- state -------------------------------------------------------------
    @property
    def store(self):
        if self._store is None:
            self._store = JudgeStore(default_backend(self.env))
        return self._store

    @property
    def active(self):
        return self.enabled and self.circuit_reason is None

    @property
    def status(self):
        if not self.enabled:
            return "disabled"
        return "circuit_open" if self.circuit_reason else "active"

    def begin_video(self, label):
        self.current = self.videos.setdefault(str(label), {
            "label": str(label), "calls": 0, "cache_hits": 0, "wall_seconds": 0.0, "capped": None,
            "needs_review_reasons": [], "models_used": {}, "verdicts": [], "contact_sheet": None,
        })
        return self.current

    def flag_review(self, reason, video=None):
        v = self.videos.get(video) if video else self.current
        if v is not None and reason not in v["needs_review_reasons"]:
            v["needs_review_reasons"].append(reason)

    def _open_circuit(self, reason):
        if self.circuit_reason is None:
            self.circuit_reason = reason
            print(f"⚠️ visual judge circuit open ({reason}); finishing with mechanical filters only.", flush=True)
        self.flag_review(f"visual judge circuit open: {reason}")

    def summary(self, label=None):
        v = self.videos.get(label) if label else self.current
        v = v or {"calls": 0, "cache_hits": 0, "wall_seconds": 0.0, "needs_review_reasons": [],
                  "models_used": {}, "verdicts": [], "contact_sheet": None, "capped": None}
        reasons = list(v["needs_review_reasons"])
        if self.enabled and self.circuit_reason and not any("circuit open" in r for r in reasons):
            reasons.append(f"visual judge circuit open: {self.circuit_reason}")
        return {
            "judge_status": self.status, "disabled_reason": self.disabled_reason,
            "prompt_version": PROMPT_VERSION, "models": list(self.models),
            "calls": v["calls"], "cache_hits": v["cache_hits"], "wall_seconds": round(v["wall_seconds"], 2),
            "parse_failures": self.parse_failures,
            "circuit": {"state": "open" if self.circuit_reason else "closed", "reason": self.circuit_reason},
            "capped": v["capped"], "models_used": dict(v["models_used"]),
            "dead_models": dict(self.dead_models),
            "needs_review": bool(reasons) and self.enabled, "needs_review_reasons": reasons if self.enabled else [],
            "contact_sheet": v["contact_sheet"], "verdicts": list(v["verdicts"]),
        }

    def run_summary(self):
        return {
            "judge_status": self.status, "prompt_version": PROMPT_VERSION, "calls": self.total_calls,
            "cache_hits": self.total_cache_hits, "circuit": {"state": "open" if self.circuit_reason else "closed",
                                                             "reason": self.circuit_reason},
            "videos": {k: self.summary(k) for k in self.videos if self.videos[k]["calls"] or
                       self.videos[k]["cache_hits"] or self.videos[k]["verdicts"] or k != "default"},
        }

    # ----- budget ------------------------------------------------------------
    def _check_caps(self):
        v = self.current
        if v["capped"]:
            raise _Capped(v["capped"])
        if v["calls"] >= MAX_CALLS_PER_VIDEO:
            v["capped"] = f"judge call cap ({MAX_CALLS_PER_VIDEO}) reached"
        elif v["wall_seconds"] >= MAX_WALL_SECONDS_PER_VIDEO:
            v["capped"] = f"judge wall-time cap ({int(MAX_WALL_SECONDS_PER_VIDEO)} s) reached"
        if v["capped"]:
            self.flag_review(v["capped"])
            raise _Capped(v["capped"])

    def _sleep(self, seconds):
        remaining = MAX_WALL_SECONDS_PER_VIDEO - self.current["wall_seconds"]
        if seconds > remaining:
            self.current["capped"] = f"judge wall-time cap ({int(MAX_WALL_SECONDS_PER_VIDEO)} s) reached"
            self.flag_review(self.current["capped"])
            raise _Capped(self.current["capped"])
        if seconds > 0:
            self.sleep(seconds)
            self.current["wall_seconds"] += seconds

    def _pace(self):
        if self.last_request_at is not None:
            wait = self.min_interval - (self.clock() - self.last_request_at)
            if wait > 0:
                self._sleep(wait)
        self.last_request_at = self.clock()

    # ----- HTTP --------------------------------------------------------------
    def _payload(self, model, prompt_text, jpeg_bytes, level):
        url = "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode()
        user_text = prompt_text if level < 2 else SYSTEM_PROMPT + "\n\n" + prompt_text
        messages = [] if level >= 2 else [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.append({"role": "user", "content": [{"type": "text", "text": user_text},
                                                     {"type": "image_url", "image_url": {"url": url}}]})
        body = {"model": model, "messages": messages, "temperature": 0, "top_p": 1,
                "max_tokens": self.max_tokens.get(model, 1024), "stream": False}
        if level == 0:
            body["response_format"] = {"type": "json_object"}
            if "nemotron" in model.casefold():
                # Nemotron 3.5 VL reasons by default (content=null, reasoning only); turn it off.
                body["chat_template_kwargs"] = {"enable_thinking": False}
        return body

    def _call_model(self, model, prompt_text, jpeg_bytes):
        """One logical request with 429/5xx/timeout retries. Returns response text."""
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json",
                   "Accept": "application/json"}
        retries = 0
        while True:
            self._check_caps()
            self._pace()
            level = self.compat.get(model, 0)
            body = json.dumps(self._payload(model, prompt_text, jpeg_bytes, level))
            started = self.clock()
            self.current["calls"] += 1
            self.total_calls += 1
            try:
                status, resp_headers, text = self.transport(API_URL, headers, body, REQUEST_TIMEOUT)
            except Exception as exc:  # noqa: BLE001 - timeout / connection error
                status, resp_headers, text = None, {}, str(exc)
            self.current["wall_seconds"] += max(0.0, self.clock() - started)
            if status == 200:
                try:
                    data = json.loads(text)
                    choice = (data.get("choices") or [{}])[0]
                    msg = choice.get("message") or {}
                    content = msg.get("content")
                    if isinstance(content, list):
                        content = " ".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
                    if not content and choice.get("finish_reason") == "length" \
                            and self.max_tokens.get(model, 1024) < 4096:
                        # Reasoning ate the budget (content=null): retry once with a bigger budget.
                        self.max_tokens[model] = 4096
                        continue
                    return str(content or msg.get("reasoning_content") or "")
                except (ValueError, AttributeError, IndexError, TypeError):
                    return str(text or "")
            if status == 402:
                raise _CircuitTrip("HTTP 402 (payment/credits required)")
            if status == 404:
                raise _ModelGone("HTTP 404")
            if status in (400, 422) and level < 2:
                # response_format / system role unsupported on this model: fall back silently.
                self.compat[model] = level + 1
                continue
            if status in (401, 403):
                raise _CircuitTrip(f"HTTP {status} (key rejected)")
            if status == 429 or status is None or (isinstance(status, int) and status >= 500):
                if retries >= MAX_RETRIES:
                    raise _Exhausted(f"HTTP {status or 'timeout'} after {MAX_RETRIES} retries")
                wait = _retry_after_seconds(resp_headers) if status == 429 else None
                if wait is None:
                    wait = BACKOFF[min(retries, len(BACKOFF) - 1)] * (1 + self.rng.uniform(-JITTER, JITTER))
                retries += 1
                self._sleep(min(MAX_RETRY_AFTER, wait))
                continue
            raise _Exhausted(f"HTTP {status}")

    def _request_json(self, prompt_text, jpeg_bytes, validator):
        """Walk the fallback chain; parse with one retry. Returns (parsed, model) or None."""
        for model in self.models:
            if model in self.dead_models:
                continue
            for attempt in range(2):  # first try + one retry on unparseable output
                try:
                    text = self._call_model(model, prompt_text, jpeg_bytes)
                except _ModelGone as exc:
                    self.dead_models[model] = str(exc)
                    break
                except _Exhausted as exc:
                    self.dead_models[model] = str(exc)
                    self.consecutive_exhausted += 1
                    if self.consecutive_exhausted >= CIRCUIT_EXHAUSTED:
                        raise _CircuitTrip(f"{CIRCUIT_EXHAUSTED} consecutive exhausted-retry requests")
                    break
                self.consecutive_exhausted = 0
                parsed = validator(extract_json(text))
                if parsed is not None:
                    self.current["models_used"][model] = self.current["models_used"].get(model, 0) + 1
                    return parsed, model
                self.parse_failures += 1
                if self.parse_failures >= CIRCUIT_PARSE_FAILURES:
                    raise _CircuitTrip(f"{CIRCUIT_PARSE_FAILURES} JSON parse failures")
            else:
                return None  # two unparseable answers from this model; give up on this image
        if all(m in self.dead_models for m in self.models):
            raise _CircuitTrip("no judge model available (" + "; ".join(
                f"{m}: {r}" for m, r in self.dead_models.items()) + ")")
        return None

    # ----- public API --------------------------------------------------------
    def is_rejected(self, asset_key, query):
        """Rejection memory, checked at fetch time before downloading. None when free to use."""
        if not self.enabled:
            return None
        try:
            return self.store.rejection(asset_key, query)
        except Exception:  # noqa: BLE001
            return None

    def judge_image(self, image, video_title="", narration_line="", query="", asset_key="", record=True):
        """Judge one image (path or bytes). Returns a verdict dict with decision, or None if unavailable."""
        if not self.active:
            return None
        try:
            jpeg = image_jpeg_bytes(image)
        except Exception:  # noqa: BLE001 - unreadable image: let mechanical filters decide
            return None
        key = score_cache_key(jpeg)
        try:
            cached = self.store.get_verdict(key, query)
        except Exception:  # noqa: BLE001
            cached = None
        if cached:
            cached["cached"] = True
            self.current["cache_hits"] += 1
            self.total_cache_hits += 1
            verdict = cached
        else:
            prompt_text = USER_TEMPLATE.format(
                video_title=" ".join(str(video_title or "").split())[:200],
                narration_line=" ".join(str(narration_line or "").split()).replace('"', "'")[:500],
                query=" ".join(str(query or "").split()).replace('"', "'")[:150])
            try:
                result = self._request_json(prompt_text, jpeg, normalize_verdict)
            except _CircuitTrip as exc:
                self._open_circuit(str(exc))
                return None
            except _Capped:
                return None
            if result is None:
                return None
            verdict, model = result
            verdict.update({"judge_model": model, "prompt_version": PROMPT_VERSION, "cached": False})
            try:
                self.store.put_verdict(key, query, verdict)
            except Exception:  # noqa: BLE001
                pass
        verdict["prompt_version"] = PROMPT_VERSION
        verdict["decision"] = decide(verdict)
        if record:
            self.record(verdict, asset_key, query)
        return verdict

    def record(self, verdict, asset_key, query):
        self.current["verdicts"].append({
            "asset": asset_key, "query": query, "decision": verdict.get("decision"),
            "score": verdict.get("score"), "subject_match": verdict.get("subject_match"),
            "issues": verdict.get("issues"), "reason": verdict.get("reason"),
            "judge_model": verdict.get("judge_model"), "cached": verdict.get("cached", False),
        })
        if verdict.get("decision") == FAIL:
            try:
                self.store.record_rejection(asset_key, query, verdict.get("issues"))
            except Exception:  # noqa: BLE001
                pass

    def judge_clip(self, clip_path, video_title="", narration_line="", query="", asset_key="", work_dir=None):
        """Judge a video clip on its 50 % frame; retry once at 25 % only when it failed on low_quality."""
        if not self.active:
            return None
        clip_path = Path(clip_path)
        work_dir = Path(work_dir or clip_path.parent)
        duration = probe_duration(clip_path)
        frame = extract_clip_frame(clip_path, 0.50, work_dir / f"{clip_path.stem}_judge50.jpg", duration)
        if frame is None:
            return None
        try:
            verdict = self.judge_image(frame, video_title, narration_line, query, asset_key, record=False)
            if verdict and verdict["decision"] == FAIL and "low_quality" in verdict.get("issues", []) \
                    and not set(verdict.get("issues", [])) & (QUERY_REJECT_ISSUES | {"sensitive_content"}):
                retry_frame = extract_clip_frame(clip_path, 0.25, work_dir / f"{clip_path.stem}_judge25.jpg",
                                                 duration)
                if retry_frame is not None:
                    second = self.judge_image(retry_frame, video_title, narration_line, query, asset_key,
                                              record=False)
                    if second:
                        second["frame"] = "25%"
                        second["retried_from"] = "50% low_quality"
                        verdict = second
                    Path(retry_frame).unlink(missing_ok=True)
            if verdict:
                verdict.setdefault("frame", "50%")
                self.record(verdict, asset_key, query)
            return verdict
        finally:
            Path(frame).unlink(missing_ok=True)

    def check_contact_sheet(self, shots, video_title="", work_dir=".", label=None):
        """Post-render check. shots: [(image_path, narration_line)] in order. Returns result dict or None.

        Shots are numbered 1..N; sheets hold 12 tiles (4x3), one request per sheet.
        """
        if not self.active or not shots:
            return None
        per = CONTACT_SHEET_COLS * CONTACT_SHEET_ROWS
        bad, overall, sheets = [], "ok", 0
        for start in range(0, len(shots), per):
            chunk = shots[start:start + per]
            numbers = list(range(start + 1, start + len(chunk) + 1))
            sheet = Path(work_dir) / f"contact_sheet_{label or 'video'}_{start // per + 1}.jpg"
            try:
                compose_contact_sheet([p for p, _ in chunk], numbers, sheet)
                jpeg = sheet.read_bytes()
            except Exception:  # noqa: BLE001
                continue
            lines = " ".join(f"{n}: \"{' '.join(str(t or '').split())[:140]}\"" for n, (_, t) in zip(numbers, chunk))
            prompt_text = CONTACT_SHEET_TEMPLATE.format(n=len(shots), lines=lines)
            if video_title:
                prompt_text = f"VIDEO TOPIC: {video_title}\n" + prompt_text

            def _valid(raw, lo=numbers[0], hi=numbers[-1]):
                if not isinstance(raw, dict) or "bad_shots" not in raw and "overall" not in raw:
                    return None
                out = []
                for item in raw.get("bad_shots") or []:
                    try:
                        n = int(item.get("n"))
                    except (AttributeError, TypeError, ValueError):
                        continue
                    if lo <= n <= hi:
                        out.append({"n": n, "reason": str(item.get("reason") or "")[:200]})
                ov = str(raw.get("overall") or "ok").casefold()
                return {"bad_shots": out, "overall": "needs_review" if ov.startswith("need") else "ok"}
            try:
                result = self._request_json(prompt_text, jpeg, _valid)
            except _CircuitTrip as exc:
                self._open_circuit(str(exc))
                break
            except _Capped:
                break
            if result:
                parsed, model = result
                sheets += 1
                bad.extend(parsed["bad_shots"])
                if parsed["overall"] == "needs_review":
                    overall = "needs_review"
        if not sheets:
            return None
        if bad:
            overall = "needs_review"
        outcome = {"bad_shots": bad, "overall": overall, "sheets": sheets, "shots": len(shots),
                   "prompt_version": PROMPT_VERSION}
        self.current["contact_sheet"] = outcome
        if overall == "needs_review":
            detail = ", ".join(f"#{b['n']}" for b in bad) or "overall"
            self.flag_review(f"contact sheet flagged shots: {detail}")
        return outcome


# ---------------------------------------------------------------------------
# Shot-level candidate loop (shared by both engines)
# ---------------------------------------------------------------------------

class ShotJudging:
    """Tracks one shot's judged candidates: PASS stops, MARGINAL is kept as last resort.

    Usage per candidate that survived the mechanical filters:
        outcome = shot.consider(verdict)   # 'accept' | 'keep_marginal' | 'reject' | 'unjudged'
    """

    def __init__(self, judge, limit=CANDIDATES_PER_SHOT):
        self.judge = judge
        self.limit = limit
        self.judged = 0
        self.marginal = None   # (verdict, payload)
        self.displaced = None
        self.failures = []

    @property
    def active(self):
        return bool(self.judge and self.judge.active)

    def can_judge_more(self):
        return self.active and self.judged < self.limit

    def extend(self, extra):
        self.limit += extra

    def consider(self, verdict, payload=None):
        """Returns 'accept' | 'keep_marginal' | 'reject' | 'unjudged'.

        After 'keep_marginal', ``self.displaced`` holds the payload of a weaker
        marginal that was replaced (the caller discards it), else None.
        """
        self.displaced = None
        if verdict is None:
            return "unjudged"
        self.judged += 1
        decision = verdict.get("decision")
        if decision == PASS:
            return "accept"
        if decision == MARGINAL:
            if self.marginal is None or (verdict.get("score") or 0) > (self.marginal[0].get("score") or 0):
                self.displaced = self.marginal[1] if self.marginal else None
                self.marginal = (verdict, payload)
                return "keep_marginal"
            return "reject"
        self.failures.append(verdict)
        return "reject"


def verdict_fields(verdict, needs_review=False, review_reason=""):
    """Flat per-asset manifest record of a verdict."""
    if not verdict:
        return None
    out = {k: verdict.get(k) for k in ("judge_model", "score", "subject_match", "issues", "reason",
                                         "decision", "prompt_version")}
    out["cached"] = bool(verdict.get("cached"))
    if verdict.get("frame"):
        out["frame"] = verdict["frame"]
    if needs_review:
        out["needs_review"] = True
        out["review_reason"] = review_reason
    return out


_JUDGE = None


def get_judge():
    """Process-wide judge (one per run so pacing, caps and the circuit are shared)."""
    global _JUDGE
    if _JUDGE is None:
        _JUDGE = VisualJudge()
    return _JUDGE


def reset_judge(judge=None):
    global _JUDGE
    _JUDGE = judge
    return judge
