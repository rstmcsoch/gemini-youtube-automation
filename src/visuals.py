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

ALLOWED_WIKIMEDIA_LICENSES = {"public domain", "pd", "cc0", "cc-zero", "public domain mark", "pd-mark"}
TIMEOUT = (5, 15)


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
        try:
            response = requests.get("https://pixabay.com/api/", params={
                "key": key, "q": query[:100], "image_type": "photo",
                "orientation": "vertical", "safesearch": "true", "per_page": 12,
            }, timeout=TIMEOUT)
            if response.status_code == 429:
                return []
            response.raise_for_status()
            data = response.json()
            cache_file.write_text(json.dumps(data), encoding="utf-8")
        except (requests.RequestException, ValueError, OSError):
            return []
    candidates = []
    for hit in data.get("hits", []):
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
    if not account or not token:
        return None
    url = f"https://api.cloudflare.com/client/v4/accounts/{quote(account, safe='')}/ai/run/@cf/black-forest-labs/flux-1-schnell"
    try:
        response = requests.post(url, headers={"Authorization": f"Bearer {token}"},
                                 json={"prompt": f"Vertical 9:16 editorial illustration, no text, no watermark: {prompt}"},
                                 timeout=(8, 60))
        if response.status_code == 429:
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


def select_visual(intent, output_path, used_hashes=None):
    """Resolve one still using the intent-specific order; returns (manifest, review_required)."""
    used_hashes = used_hashes if used_hashes is not None else set()
    kind = intent.get("type", "concept")
    query = " ".join([*(intent.get("entities") or []), *(intent.get("keywords") or [])]).strip()
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
    output_path.unlink(missing_ok=True)
    return {"provider": "gradient", "source_url": "", "asset_id": "gradient-fallback",
            "license": "Original generated gradient", "license_url": "", "retrieved_at": _now(),
            "sha256": ""}, bool(kind in {"news_event", "product"} and intent.get("real_image_essential"))
