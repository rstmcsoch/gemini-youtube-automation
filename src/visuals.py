"""Reusable rights-aware still-image providers for DYPOL verticals.

No social/news-site scraping. Provider failures are non-fatal; rights are an
eligibility gate, not a relevance-score component.
"""
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import requests
from PIL import Image

from src import storyboard as sb

ALLOWED_WIKIMEDIA_LICENSES = {"public domain", "pd", "cc0", "cc-zero", "public domain mark", "pd-mark"}
TIMEOUT = (5, 15)

# ---------------------------------------------------------------------------
# Per-run circuit breakers. Rule: when a provider hits a quota / rate limit, skip
# that source for the rest of the run and continue with what is available;
# never stall retrying it and never pad with placeholders because of it.
# ---------------------------------------------------------------------------
_QUOTA_MARKERS = ("daily free allocation", "quota", "rate limit", "rate-limit", "ratelimit",
                  "too many requests", "resource_exhausted", "neurons", "limit exceeded",
                  "exceeded your", "capacity temporarily exceeded")
_BREAKERS = {}


def is_quota_error(status_code, body=""):
    """True for HTTP 429 or any error body that reads as a quota / allocation limit."""
    try:
        if int(status_code) == 429:
            return True
    except (TypeError, ValueError):
        pass
    text = str(body or "").casefold()
    return any(marker in text for marker in _QUOTA_MARKERS)


def breaker_open(name):
    return name in _BREAKERS


def trip_breaker(name, reason=""):
    """Open the breaker for `name` for the rest of this process; logs exactly once."""
    if name in _BREAKERS:
        return False
    _BREAKERS[name] = {"reason": str(reason)[:200], "at": _now()}
    print(f"⚠️ {name} quota/limit hit ({str(reason)[:160]}); skipping {name} for the rest of this run "
          "and continuing with other sources.", flush=True)
    return True


def reset_breakers():
    _BREAKERS.clear()


def breaker_status():
    return {name: dict(info) for name, info in _BREAKERS.items()}


def flux_available():
    """FLUX can be called: credentials present and its breaker is closed."""
    return (not breaker_open("flux") and bool(os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip())
            and bool(os.environ.get("CLOUDFLARE_API_TOKEN", "").strip()))


def _response_text(response):
    try:
        return str(getattr(response, "text", "") or "")
    except Exception:  # noqa: BLE001 - a broken body must never break the run
        return ""


def _now():
    return datetime.now(timezone.utc).isoformat()


def _tokens(value):
    return {t.casefold() for t in re.findall(r"[A-Za-z0-9]{3,}", str(value or ""))}


def relevance_score(candidate, intent, target=(9, 16)):
    """Deterministic relevance/portrait-fit score; duplicates are filtered separately."""
    query = " ".join([*(intent.get("entities") or []), *(intent.get("keywords") or [])])
    terms = _tokens(query)
    title = _tokens(candidate.get("title", ""))
    entity_terms = _tokens(" ".join(intent.get("entities") or []))
    entity_match = len(entity_terms & title) / max(1, len(entity_terms))
    keyword_match = len((_tokens(" ".join(intent.get("keywords") or []))) & title) / max(1, len(_tokens(" ".join(intent.get("keywords") or []))))
    width, height = candidate.get("width", 0), candidate.get("height", 0)
    ratio = (width / height) if width and height else 0
    orientation_fit = 1.0 - min(1.0, abs(ratio - (target[0] / target[1])))
    return round(0.55 * entity_match + 0.30 * keyword_match + 0.15 * orientation_fit, 4)


def _allowed_wikimedia_license(info):
    short = str(info.get("shortname") or "").strip().casefold()
    name = str(info.get("name") or "").strip().casefold()
    url = str(info.get("url") or "").casefold()
    text = " ".join((short, name, url))
    # Explicit allowlist only. Reject attribution/share-alike licenses first.
    if any(term in text for term in ("cc-by", "cc by", "attribution", "share-alike", "share alike")):
        return False
    if short in {"cc0", "cc zero", "pd-mark", "public domain mark", "public domain"}:
        return True
    if "public domain" in name or "public domain" in short:
        return True
    if "creativecommons.org/publicdomain/zero/" in url:
        return True
    if "pd-mark" in url or "publicdomain/mark/" in url:
        return True
    return False


def _download_candidate(candidate, output_path):
    response = requests.get(candidate["download_url"], timeout=TIMEOUT)
    response.raise_for_status()
    output_path.write_bytes(response.content)
    with Image.open(output_path) as im:
        im.verify()
    with Image.open(output_path) as im:
        width, height = im.size
        im.convert("RGB").save(output_path, format="PNG")
    candidate["width"], candidate["height"] = width, height
    candidate["sha256"] = hashlib.sha256(output_path.read_bytes()).hexdigest()
    candidate["retrieved_at"] = _now()
    return candidate


def _wikimedia_candidates(query):
    params = {
        "action": "query", "format": "json", "generator": "search",
        "gsrsearch": f"filetype:bitmap {query}", "gsrnamespace": 6,
        "gsrlimit": 8, "prop": "imageinfo", "iiprop": "url|size|extmetadata",
        "iiurlwidth": 1200,
    }
    response = requests.get("https://commons.wikimedia.org/w/api.php", params=params, timeout=TIMEOUT,
                            headers={"User-Agent": "DYPOLCreatorConsole/1.0 (visual asset lookup)"})
    response.raise_for_status()
    pages = response.json().get("query", {}).get("pages", {})
    out = []
    for page in pages.values():
        info = (page.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata") or {}
        license_info = meta.get("LicenseShortName") or {}
        license_name = license_info.get("value", "")
        license_url = (meta.get("LicenseUrl") or {}).get("value", "")
        if not _allowed_wikimedia_license({"shortname": license_name, "url": license_url,
                                           "name": (meta.get("UsageTerms") or {}).get("value", "")}):
            continue
        # Never publish creator names; only the generic source/rights category is retained.
        out.append({
            "provider": "wikimedia_commons", "source_url": info.get("descriptionurl", ""),
            "asset_id": str(page.get("pageid", "")), "license": license_name or "Public domain/CC0",
            "license_url": license_url, "download_url": info.get("thumburl") or info.get("url"),
            "title": re.sub(r"<[^>]+>", " ", page.get("title", "")),
            "width": info.get("thumbwidth") or info.get("width", 0),
            "height": info.get("thumbheight") or info.get("height", 0),
        })
    return out


def _pixabay_candidates(query):
    key = os.environ.get("PIXABAY_API_KEY", "").strip()
    if not key:
        return []
    cache_dir = Path(os.environ.get("VERTICALS_HOME", str(Path.home() / ".verticals"))) / "cache" / "pixabay"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.sha256(query.casefold().encode()).hexdigest()
    cache_file = cache_dir / f"{cache_key}.json"
    data = None
    # Pixabay API terms require caching API results for 24 hours.
    if cache_file.exists() and time.time() - cache_file.stat().st_mtime < 86400:
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = None
    if data is None:
        if breaker_open("pixabay"):
            return []
        try:
            response = requests.get("https://pixabay.com/api/", params={
                "key": key, "q": query[:100], "image_type": "photo",
                "orientation": "vertical", "safesearch": "true", "per_page": 12,
            }, timeout=TIMEOUT)
            if response.status_code == 429:
                trip_breaker("pixabay", "HTTP 429")
                return []
            response.raise_for_status()
            data = response.json()
            cache_file.write_text(json.dumps(data), encoding="utf-8")
        except (requests.RequestException, ValueError, OSError):
            return []
    candidates = []
    for hit in data.get("hits", []):
        if sb.offtopic_reason(hit.get("tags", "")):
            continue
        candidates.append({
            "provider": "pixabay", "source_url": hit.get("pageURL", ""),
            "asset_id": str(hit.get("id", "")), "license": "Pixabay Content License",
            "license_url": "https://pixabay.com/service/license-summary/",
            "download_url": hit.get("largeImageURL") or hit.get("webformatURL"),
            "title": hit.get("tags", ""), "width": hit.get("imageWidth", 0),
            "height": hit.get("imageHeight", ""),
        })
    return candidates


def _cloudflare_generate(prompt, output_path):
    account = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
    token = os.environ.get("CLOUDFLARE_API_TOKEN", "").strip()
    if not account or not token or breaker_open("flux"):
        return None
    url = f"https://api.cloudflare.com/client/v4/accounts/{quote(account, safe='')}/ai/run/@cf/black-forest-labs/flux-1-schnell"
    try:
        response = requests.post(url, headers={"Authorization": f"Bearer {token}"},
                                 json={"prompt": f"Vertical 9:16 editorial illustration, no text, no watermark: {prompt}"},
                                 timeout=(8, 60))
        if response.status_code != 200 and is_quota_error(response.status_code, _response_text(response)):
            trip_breaker("flux", f"HTTP {response.status_code}")
            return None
        response.raise_for_status()
        data = response.json()
        image_data = data.get("result", {}).get("image")
        if not image_data:
            return None
        import base64
        output_path.write_bytes(base64.b64decode(image_data))
        with Image.open(output_path) as im:
            im.verify()
        return {
            "provider": "cloudflare_workers_ai", "source_url": url, "asset_id": "flux-1-schnell",
            "license": "AI-generated; review applicable model terms",
            "license_url": "https://developers.cloudflare.com/workers-ai/",
            "retrieved_at": _now(), "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        }
    except (requests.RequestException, ValueError, OSError):
        return None


_GEMINI_IMAGE_DISABLED_FOR_QUOTA = False


def _gemini_image_generate(prompt, output_path):
    """Optional Gemini image generation; permanently short-circuit this process on zero-quota 429."""
    global _GEMINI_IMAGE_DISABLED_FOR_QUOTA
    if _GEMINI_IMAGE_DISABLED_FOR_QUOTA:
        return None
    if os.environ.get("DYPOL_GEMINI_IMAGE_ENABLED", "").strip().casefold() not in {"1", "true", "yes"}:
        return None
    key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not key:
        return None
    model = os.environ.get("DYPOL_GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image").strip()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{quote(model, safe='')}:generateContent"
    try:
        response = requests.post(
            url, params={"key": key},
            json={"contents": [{"parts": [{"text": f"Create a clean editorial illustration, no text or watermark: {prompt}"}]}],
                  "generationConfig": {"responseModalities": ["IMAGE"]}},
            timeout=(8, 60),
        )
        if response.status_code == 429:
            body = ""
            try:
                body = json.dumps(response.json()).casefold()
            except (ValueError, TypeError):
                body = str(getattr(response, "text", "")).casefold()
            if any(term in body for term in ("quota", "resource_exhausted", "billing", "limit: 0", "free_tier")):
                _GEMINI_IMAGE_DISABLED_FOR_QUOTA = True
                print("Gemini image generation disabled for this run: zero-quota 429; using fallback providers.")
            return None
        response.raise_for_status()
        payload = response.json()
        for candidate in payload.get("candidates", []):
            for part in (candidate.get("content") or {}).get("parts", []):
                inline = part.get("inlineData") or part.get("inline_data") or {}
                encoded = inline.get("data")
                if encoded:
                    import base64
                    output_path.write_bytes(base64.b64decode(encoded))
                    with Image.open(output_path) as image:
                        image.verify()
                    return {
                        "provider": "gemini_image", "source_url": url.split("?")[0],
                        "asset_id": model, "license": "AI-generated; review applicable Google terms",
                        "license_url": "https://ai.google.dev/gemini-api/terms",
                        "retrieved_at": _now(),
                        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
                    }
        return None
    except (requests.RequestException, ValueError, OSError):
        return None


def select_visual(intent, output_path, used_hashes=None):
    """Resolve one still using the intent-specific order; returns (manifest, review_required)."""
    used_hashes = used_hashes if used_hashes is not None else set()
    kind = intent.get("type", "concept")
    # Stock/archive search works best with a short query: up to 2 entities + 3 keywords.
    ents = list(intent.get("entities") or [])[:2]
    kws = [k for k in (intent.get("keywords") or []) if k not in ents][:3]
    query = " ".join([*ents, *kws]).strip()[:100]
    query = query or str(intent.get("prompt", "technology concept"))
    providers = ["wikimedia", "pixabay"] if kind in {"news_event", "product"} else ["pixabay"]
    candidates = []
    for provider in providers:
        try:
            candidates = _wikimedia_candidates(query) if provider == "wikimedia" else _pixabay_candidates(query)
        except (requests.RequestException, ValueError, KeyError):
            candidates = []
        ranked = sorted(candidates, key=lambda c: relevance_score(c, intent), reverse=True)
        for candidate in ranked:
            if not candidate.get("download_url"):
                continue
            if sb.offtopic_reason(candidate.get("title", ""), _tokens(query)):
                continue
            score = relevance_score(candidate, intent)
            if score < 0.20:
                continue
            if int(candidate.get("width") or 0) < 360 or int(candidate.get("height") or 0) < 640:
                continue
            if kind in {"news_event", "product"} and intent.get("entities"):
                entity_terms = _tokens(" ".join(intent.get("entities") or []))
                title_terms = _tokens(candidate.get("title", ""))
                entity_match = len(entity_terms & title_terms) / max(1, len(entity_terms))
                if entity_match < 0.25:
                    continue
            try:
                metadata = _download_candidate(candidate, output_path)
                digest = metadata["sha256"]
                if digest in used_hashes:
                    output_path.unlink(missing_ok=True)
                    continue
                used_hashes.add(digest)
                metadata.pop("download_url", None)
                metadata["relevance_score"] = relevance_score(candidate, intent)
                return metadata, False
            except (requests.RequestException, OSError, ValueError):
                output_path.unlink(missing_ok=True)
                continue
    ai = _cloudflare_generate(query, output_path)
    if ai:
        digest = ai["sha256"]
        if digest not in used_hashes:
            used_hashes.add(digest)
            return ai, False
        output_path.unlink(missing_ok=True)
    # Gemini is opt-in and runs only after licensed stock and FLUX have failed.
    ai = _gemini_image_generate(query, output_path)
    if ai:
        digest = ai["sha256"]
        if digest not in used_hashes:
            used_hashes.add(digest)
            return ai, False
        output_path.unlink(missing_ok=True)
    output_path.unlink(missing_ok=True)
    return {"provider": "gradient", "source_url": "", "asset_id": "gradient-fallback",
            "license": "Original generated gradient", "license_url": "", "retrieved_at": _now(),
            "sha256": ""}, bool(kind in {"news_event", "product"} and intent.get("real_image_essential"))


# ---------------------------------------------------------------------------
# Multi-visual sourcing for the explainer editor (long video + Short).
# Every returned asset carries flat manifest fields: provider, source_url,
# asset_id, license, license_url, sha256, kind, retrieved_at.
# ---------------------------------------------------------------------------
import subprocess

PIXABAY_LICENSE = "Pixabay Content License"
PIXABAY_LICENSE_URL = "https://pixabay.com/service/license-summary/"
FLUX_SOURCE_URL = "https://developers.cloudflare.com/workers-ai/models/flux-1-schnell/"
FLUX_LICENSE = "AI-generated with FLUX.1 [schnell] (Apache-2.0 model) on Cloudflare Workers AI"
FLUX_LICENSE_URL = "https://huggingface.co/black-forest-labs/FLUX.1-schnell"
MAX_VIDEO_BYTES = 60 * 1024 * 1024


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _pixabay_cached_get(endpoint, params):
    """GET a Pixabay endpoint with the mandatory 24 h result cache. Never raises."""
    key = os.environ.get("PIXABAY_API_KEY", "").strip()
    if not key:
        return {}
    cache_dir = Path(os.environ.get("VERTICALS_HOME", str(Path.home() / ".verticals"))) / "cache" / "pixabay"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ident = json.dumps({"endpoint": endpoint, **params}, sort_keys=True).casefold()
    cache_file = cache_dir / f"{hashlib.sha256(ident.encode()).hexdigest()}.json"
    if cache_file.exists() and time.time() - cache_file.stat().st_mtime < 86400:
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    if breaker_open("pixabay"):
        return {}  # cached results above are still served; no new requests this run
    try:
        response = requests.get(endpoint, params={"key": key, **params}, timeout=TIMEOUT)
        if response.status_code == 429:
            trip_breaker("pixabay", "HTTP 429")
            return {}
        response.raise_for_status()
        data = response.json()
        cache_file.write_text(json.dumps(data), encoding="utf-8")
        return data
    except (requests.RequestException, ValueError, OSError):
        return {}


def pixabay_still_hits(query, orientation="horizontal"):
    data = _pixabay_cached_get("https://pixabay.com/api/", {
        "q": query[:100], "image_type": "photo", "orientation": orientation,
        "safesearch": "true", "per_page": 20, "order": "popular",
    })
    return data.get("hits", []) if isinstance(data, dict) else []


def pixabay_video_hits(query):
    data = _pixabay_cached_get("https://pixabay.com/api/videos/", {
        "q": query[:100], "video_type": "film", "safesearch": "true", "per_page": 15,
    })
    return data.get("hits", []) if isinstance(data, dict) else []


def _probe_duration(path):
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                              "-of", "csv=p=0", str(path)], capture_output=True, text=True, timeout=30).stdout
        return float(out.strip() or 0)
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0.0


def _judge():
    """The run's visual judge, or None when unavailable (judge never blocks the pipeline)."""
    try:
        from src.visual_judge import get_judge
        judge = get_judge()
        return judge if judge.active else None
    except Exception:  # noqa: BLE001
        return None


def _judge_note(asset, verdict, needs_review=False, reason=""):
    from src.visual_judge import verdict_fields
    record = verdict_fields(verdict, needs_review, reason)
    if record:
        asset["judge"] = record
    return asset


def _ranked_still_hits(query, terms, used, orientation):
    min_w, min_h = (1280, 720) if orientation == "horizontal" else (720, 1280)
    ranked = []
    for hit in pixabay_still_hits(query, orientation):
        ok, overlap, _ = sb.tag_relevance(hit.get("tags", ""), terms)
        if not ok or f"pixabay:{hit.get('id')}" in used["ids"]:
            continue
        if int(hit.get("imageWidth") or 0) < min_w or int(hit.get("imageHeight") or 0) < min_h:
            continue
        ranked.append((len(overlap), hit, overlap))
    ranked.sort(key=lambda item: -item[0])
    return [(hit, overlap) for _, hit, overlap in ranked]


def _fetch_still(hit, out_dir, label, used):
    """Download + dedupe (mechanical). Returns (path, meta) without registering it as used."""
    path = Path(out_dir) / f"{label}_px_{hit.get('id')}.png"
    try:
        meta = _download_candidate({"download_url": hit.get("largeImageURL") or hit.get("webformatURL")}, path)
    except (requests.RequestException, OSError, ValueError):
        path.unlink(missing_ok=True)
        return None
    if meta["sha256"] in used["hashes"]:
        path.unlink(missing_ok=True)
        return None
    return path, meta


def _still_asset(hit, meta, path, query, overlap, used):
    used["hashes"].add(meta["sha256"])
    used["ids"].add(f"pixabay:{hit.get('id')}")
    return {
        "kind": "image", "provider": "pixabay", "source_url": hit.get("pageURL", ""),
        "asset_id": str(hit.get("id", "")), "license": PIXABAY_LICENSE, "license_url": PIXABAY_LICENSE_URL,
        "sha256": meta["sha256"], "retrieved_at": _now(), "tags": hit.get("tags", ""),
        "query": query, "matched_terms": sorted(overlap), "width": meta["width"], "height": meta["height"],
        "path": str(path), "reusable": False, "third_party": True,
    }


def _stock_still(query, terms, out_dir, used, orientation, label, ctx=None, judge_limit=None, broaden=True,
                 jstate=None):
    ctx = ctx or {}
    judge = _judge()
    if judge is None:
        # Judge disabled/unavailable: exactly the mechanical behaviour.
        for hit, overlap in _ranked_still_hits(query, terms, used, orientation)[:4]:
            fetched = _fetch_still(hit, out_dir, label, used)
            if fetched:
                return _still_asset(hit, fetched[1], fetched[0], query, overlap, used)
        return None
    from src.visual_judge import ShotJudging, CANDIDATES_PER_SHOT, BROADENED_CANDIDATES, broaden_query
    shot = ShotJudging(judge, judge_limit or CANDIDATES_PER_SHOT)
    tried = set()

    def attempt(q, extra):
        shot.extend(extra)
        for hit, overlap in _ranked_still_hits(q, terms, used, orientation):
            if not shot.can_judge_more() and shot.active:
                return None
            key = f"pixabay:{hit.get('id')}"
            if key in tried:
                continue
            tried.add(key)
            if judge.is_rejected(key, q):
                continue  # rejection memory: skip before downloading
            fetched = _fetch_still(hit, out_dir, label, used)
            if not fetched:
                continue
            path, meta = fetched
            verdict = judge.judge_image(path, ctx.get("video_title", ""), ctx.get("narration", ""), q, key)
            outcome = shot.consider(verdict, (hit, meta, path, overlap, q))
            if outcome in ("accept", "unjudged"):
                return _judge_note(_still_asset(hit, meta, path, q, overlap, used), verdict)
            if outcome == "keep_marginal":
                if shot.displaced:
                    Path(shot.displaced[2]).unlink(missing_ok=True)
                continue
            path.unlink(missing_ok=True)
        return None

    asset = attempt(query, 0)
    broad = broaden_query(query) if broaden else None
    if asset is None and broad and shot.active:
        asset = attempt(broad, BROADENED_CANDIDATES)
    if asset is not None:
        if shot.marginal:
            Path(shot.marginal[1][2]).unlink(missing_ok=True)
        return asset
    if shot.marginal:
        verdict, (hit, meta, path, overlap, q) = shot.marginal
        if meta["sha256"] not in used["hashes"]:
            return _judge_note(_still_asset(hit, meta, path, q, overlap, used), verdict, True,
                               f"best MARGINAL still after {shot.judged} judged candidates")
        Path(path).unlink(missing_ok=True)
    if jstate is not None and shot.judged:
        jstate["rejected"] = jstate.get("rejected", 0) + shot.judged
    return None


def _ranked_video_hits(query, terms, used, min_seconds):
    ranked = []
    for hit in pixabay_video_hits(query):
        ok, overlap, _ = sb.tag_relevance(hit.get("tags", ""), terms)
        if not ok or f"pixabay-video:{hit.get('id')}" in used["ids"]:
            continue
        if float(hit.get("duration") or 0) < min_seconds:
            continue
        videos = hit.get("videos") or {}
        rendition = None
        for name in ("large", "medium"):
            v = videos.get(name) or {}
            if v.get("url") and int(v.get("width") or 0) >= 1280 and int(v.get("size") or 0) <= MAX_VIDEO_BYTES:
                rendition = v
                break
        if rendition:
            ranked.append((len(overlap), hit, rendition, overlap))
    ranked.sort(key=lambda item: -item[0])
    return [(hit, rendition, overlap) for _, hit, rendition, overlap in ranked]


def _fetch_video(hit, rendition, out_dir, label, used):
    """Download + duration/dedupe checks (mechanical). Returns (path, duration, digest) or None."""
    path = Path(out_dir) / f"{label}_pxv_{hit.get('id')}.mp4"
    try:
        response = requests.get(rendition["url"], timeout=(8, 90))
        response.raise_for_status()
        path.write_bytes(response.content)
    except (requests.RequestException, OSError):
        path.unlink(missing_ok=True)
        return None
    duration = _probe_duration(path)
    digest = _sha256(path)
    if duration < 2 or digest in used["hashes"]:
        path.unlink(missing_ok=True)
        return None
    return path, duration, digest


def _video_asset(hit, rendition, overlap, path, duration, digest, query, used):
    used["hashes"].add(digest)
    used["ids"].add(f"pixabay-video:{hit.get('id')}")
    return {
        "kind": "video", "provider": "pixabay_video", "source_url": hit.get("pageURL", ""),
        "asset_id": str(hit.get("id", "")), "license": PIXABAY_LICENSE, "license_url": PIXABAY_LICENSE_URL,
        "sha256": digest, "retrieved_at": _now(), "tags": hit.get("tags", ""), "query": query,
        "matched_terms": sorted(overlap), "width": rendition.get("width"), "height": rendition.get("height"),
        "duration": round(duration, 3), "path": str(path), "reusable": False, "third_party": True,
    }


def _stock_video(query, terms, out_dir, used, label, min_seconds=5, ctx=None, judge_limit=None, broaden=True,
                 jstate=None):
    ctx = ctx or {}
    judge = _judge()
    if judge is None:
        for hit, rendition, overlap in _ranked_video_hits(query, terms, used, min_seconds)[:3]:
            fetched = _fetch_video(hit, rendition, out_dir, label, used)
            if fetched:
                return _video_asset(hit, rendition, overlap, *fetched, query, used)
        return None
    from src.visual_judge import ShotJudging, CANDIDATES_PER_SHOT, BROADENED_CANDIDATES, broaden_query
    shot = ShotJudging(judge, judge_limit or CANDIDATES_PER_SHOT)
    tried = set()

    def attempt(q, extra):
        shot.extend(extra)
        for hit, rendition, overlap in _ranked_video_hits(q, terms, used, min_seconds):
            if not shot.can_judge_more() and shot.active:
                return None
            key = f"pixabay-video:{hit.get('id')}"
            if key in tried:
                continue
            tried.add(key)
            if judge.is_rejected(key, q):
                continue
            fetched = _fetch_video(hit, rendition, out_dir, label, used)
            if not fetched:
                continue
            path = fetched[0]
            # Clips: judged on the 50 % frame; one 25 % retry only for low_quality.
            verdict = judge.judge_clip(path, ctx.get("video_title", ""), ctx.get("narration", ""), q, key)
            outcome = shot.consider(verdict, (hit, rendition, overlap, fetched, q))
            if outcome in ("accept", "unjudged"):
                return _judge_note(_video_asset(hit, rendition, overlap, *fetched, q, used), verdict)
            if outcome == "keep_marginal":
                if shot.displaced:
                    Path(shot.displaced[3][0]).unlink(missing_ok=True)
                continue
            path.unlink(missing_ok=True)
        return None

    asset = attempt(query, 0)
    broad = broaden_query(query) if broaden else None
    if asset is None and broad and shot.active:
        asset = attempt(broad, BROADENED_CANDIDATES)
    if asset is not None:
        if shot.marginal:
            Path(shot.marginal[1][3][0]).unlink(missing_ok=True)
        return asset
    if shot.marginal:
        verdict, (hit, rendition, overlap, fetched, q) = shot.marginal
        if fetched[2] not in used["hashes"]:
            return _judge_note(_video_asset(hit, rendition, overlap, *fetched, q, used), verdict, True,
                               f"best MARGINAL clip after {shot.judged} judged candidates")
        Path(fetched[0]).unlink(missing_ok=True)
    if jstate is not None and shot.judged:
        jstate["rejected"] = jstate.get("rejected", 0) + shot.judged
    return None


def flux_generate(scene, output_path, orientation="landscape", seed=None):
    """Generate one FLUX still; returns a flat manifest dict or None. Never raises."""
    account = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
    token = os.environ.get("CLOUDFLARE_API_TOKEN", "").strip()
    if not account or not token or breaker_open("flux"):
        return None
    prompt = sb.flux_prompt(scene, orientation)
    url = f"https://api.cloudflare.com/client/v4/accounts/{quote(account, safe='')}/ai/run/@cf/black-forest-labs/flux-1-schnell"
    # flux-1-schnell accepts only prompt + steps (a "seed" field is rejected with
    # HTTP 400 "Additional or unevaluated properties '/seed'"), so variety comes
    # from the prompt variants instead.
    body = {"prompt": prompt, "steps": 6}
    try:
        response = requests.post(url, headers={"Authorization": f"Bearer {token}"}, json=body, timeout=(8, 90))
        if response.status_code != 200:
            body = _response_text(response)
            if is_quota_error(response.status_code, body):
                # e.g. 429 "you have used up your daily free allocation of 10,000 neurons"
                trip_breaker("flux", f"HTTP {response.status_code} {body[:120]}")
            else:
                print(f"FLUX request failed: HTTP {response.status_code} {body[:200]}")
            return None
        image_data = (response.json().get("result") or {}).get("image")
        if not image_data:
            return None
        import base64
        output_path = Path(output_path)
        output_path.write_bytes(base64.b64decode(image_data))
        with Image.open(output_path) as im:
            im.verify()
        with Image.open(output_path) as im:
            width, height = im.size
            im.convert("RGB").save(output_path, format="PNG")
    except (requests.RequestException, ValueError, OSError) as exc:
        print(f"FLUX generation failed: {exc}")
        return None
    return {
        "kind": "image", "provider": "cloudflare_workers_ai_flux", "source_url": FLUX_SOURCE_URL,
        "asset_id": "flux-1-schnell", "license": FLUX_LICENSE, "license_url": FLUX_LICENSE_URL,
        "sha256": _sha256(output_path), "retrieved_at": _now(), "prompt": prompt,
        "width": width, "height": height, "path": str(output_path), "reusable": True, "third_party": False,
    }


def generated_graphic(output_path, size=(1920, 1080), seed=0):
    """Last-resort original graphic (never black): soft gradient with glow rings."""
    from PIL import ImageDraw, ImageFilter
    import random
    rng = random.Random(seed)
    width, height = size
    palettes = [((8, 20, 48), (22, 70, 120)), ((12, 16, 40), (70, 30, 110)), ((6, 30, 38), (20, 110, 120)),
                ((20, 14, 36), (140, 60, 30))]
    top, bottom = palettes[seed % len(palettes)]
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)
    for y in range(0, height, 4):
        r = y / max(1, height - 1)
        draw.rectangle((0, y, width, y + 4), fill=tuple(int(top[i] * (1 - r) + bottom[i] * r) for i in range(3)))
    glow = Image.new("RGB", (width, height), (0, 0, 0))
    gd = ImageDraw.Draw(glow)
    for _ in range(6):
        cx, cy = rng.randint(0, width), rng.randint(0, height)
        rad = rng.randint(min(size) // 8, min(size) // 3)
        color = (rng.randint(30, 90), rng.randint(90, 180), rng.randint(160, 255))
        gd.ellipse((cx - rad, cy - rad, cx + rad, cy + rad), fill=color)
    glow = glow.filter(ImageFilter.GaussianBlur(min(size) // 10))
    img = Image.blend(img, glow, 0.35)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    return {
        "kind": "image", "provider": "dypol_generated_graphic", "source_url": "",
        "asset_id": f"graphic-{seed}", "license": "Original generated graphic (no third-party content)",
        "license_url": "", "sha256": _sha256(output_path), "retrieved_at": _now(),
        "width": width, "height": height, "path": str(output_path), "reusable": True, "third_party": False,
    }


_FLUX_VARIANTS = ["wide establishing shot", "close-up detail shot", "conceptual metaphor scene",
                  "overhead perspective"]


def _stock_attempt(kind, query, terms, out_dir, used, px_orientation, label, ctx=None, judge_limit=None,
                   broaden=True, jstate=None):
    if kind == "video":
        return _stock_video(query, terms, out_dir, used, label, ctx=ctx, judge_limit=judge_limit,
                            broaden=broaden, jstate=jstate)
    return _stock_still(query, terms, out_dir, used, px_orientation, label, ctx=ctx, judge_limit=judge_limit,
                        broaden=broaden, jstate=jstate)


WIDENING_JUDGE_LIMIT = 2  # widening queries: judge at most 2 candidates each, no broadened re-fetch


def gather_visuals(spec, out_dir, count, used, orientation="landscape", label="s",
                   allow_video=True, allow_still=True, seed=0, min_count=None):
    """Collect up to `count` distinct visuals for one slide/Short.

    spec: {"visual_queries": [...], "image_prompt": str, "terms": set}
    used: job-wide {"hashes": set(), "ids": set()} so third-party assets are used once.
    Mix: one stock video, one stock still, FLUX for the rest.

    When FLUX is unavailable (no credentials or its quota breaker tripped) the FLUX
    slots are filled from Pixabay video + stills instead, trying every slide query
    and then broad tech-anchor queries. A gradient graphic is only produced while
    fewer than `min_count` assets exist (default: `count`, i.e. never fewer than
    asked); callers that can reuse real assets pass min_count=0 and fill the rest.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    min_count = count if min_count is None else max(0, min(int(min_count), count))
    queries = list(spec.get("visual_queries") or ["technology"])
    terms = set(spec.get("terms") or sb.topic_terms(queries))
    px_orientation = "horizontal" if orientation == "landscape" else "vertical"
    slots = []
    if allow_video:
        slots.append("video")
    if allow_still:
        slots.append("still")
    while len(slots) < count:
        slots.append("flux")
    slots = slots[:count]
    if count >= 2 and "flux" not in slots:
        slots[-1] = "flux"  # at least one reusable own visual per slide
    stock_kinds = [k for k, ok in (("video", allow_video), ("still", allow_still)) if ok]
    broad = sb.fallback_queries(terms, seed=seed)
    ctx = {"video_title": spec.get("video_title", ""), "narration": spec.get("narration", "")}
    assets, flux_n = [], 0
    for index, slot in enumerate(slots):
        query = queries[index % len(queries)]
        asset = None
        jstate = {}
        sub = f"{label}_{index}"
        order = {"video": ["video", "still", "flux"], "still": ["still", "flux"], "flux": ["flux", "still"]}[slot]
        for provider in order:
            if provider in ("video", "still"):
                if (provider == "video" and allow_video) or (provider == "still" and allow_still):
                    asset = _stock_attempt(provider, query, terms, out_dir, used, px_orientation, sub,
                                           ctx=ctx, jstate=jstate)
            elif provider == "flux" and flux_available():
                variant = _FLUX_VARIANTS[flux_n % len(_FLUX_VARIANTS)]
                asset = flux_generate(f"{spec.get('image_prompt') or query}, {variant}",
                                      out_dir / f"{sub}_flux.png", orientation,
                                      seed=seed * 31 + index)
                flux_n += 1
                if asset and asset["sha256"] in used["hashes"]:
                    asset = None
                if asset:
                    used["hashes"].add(asset["sha256"])
            if asset:
                break
        if asset is None and stock_kinds:
            # Stock widening: other slide queries first (strict topic gate), then broad
            # tech-anchor queries whose own words may prove relevance. Each query is one
            # cached Pixabay request; after a Pixabay 429 only cached results are served.
            # At most one extra clip per slide (index 2) keeps downloads bounded; stills otherwise.
            first = "video" if slot == "video" or (slot == "flux" and index == 2) else "still"
            kinds = [first] + [k for k in ("video", "still") if k != first]
            kinds = [k for k in kinds if k in stock_kinds]
            extra = [(q, terms) for q in queries if q != query]
            extra += [(q, terms | sb.topic_terms(q)) for q in broad]
            for q, q_terms in extra:
                for kind in kinds:
                    asset = _stock_attempt(kind, q, q_terms, out_dir, used, px_orientation, sub,
                                           ctx=ctx, judge_limit=WIDENING_JUDGE_LIMIT, broaden=False,
                                           jstate=jstate)
                    if asset:
                        asset["fallback_query"] = True
                        break
                if asset:
                    break
        if asset is None:
            if len(assets) >= min_count:
                continue  # caller reuses real assets instead of a placeholder
            size = (1920, 1080) if orientation == "landscape" else (1080, 1920)
            asset = generated_graphic(out_dir / f"{sub}_graphic.png", size, seed=seed * 7 + index)
        asset["query"] = asset.get("query", query)
        if jstate.get("rejected") and (asset.get("judge") or {}).get("decision") != "PASS" \
                and not (asset.get("judge") or {}).get("needs_review"):
            # The judge rejected every stock candidate for this slot; whatever filled it
            # (FLUX, a widened query, a graphic) was not approved by the judge.
            judge = dict(asset.get("judge") or {"decision": "not_judged"})
            judge.update({"needs_review": True,
                          "review_reason": f"judge rejected all {jstate['rejected']} stock candidates; "
                                           f"used {asset.get('provider')} fallback"})
            asset["judge"] = judge
        assets.append(asset)
    return assets
