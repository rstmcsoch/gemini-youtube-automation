"""Quota circuit breakers, FLUX->Pixabay fallback, gradient cap and off-topic stock filters."""
from pathlib import Path

import pytest
from PIL import Image

from src import production as prod
from src import storyboard as sb
from src import visuals


@pytest.fixture(autouse=True)
def _fresh_breakers(monkeypatch, tmp_path):
    visuals.reset_breakers()
    monkeypatch.setenv("VERTICALS_HOME", str(tmp_path / "vh"))
    yield
    visuals.reset_breakers()


class _Resp:
    def __init__(self, status, text="", payload=None):
        self.status_code = status
        self.text = text
        self._payload = payload or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise visuals.requests.HTTPError(str(self.status_code))


FLUX_429 = ('{"errors":[{"message":"AiError: Capacity temporarily exceeded / you have used up your '
            'daily free allocation of 10,000 neurons","code":4006}],"success":false}')


# ------------------------------------------------------------------ circuit breakers

def test_quota_error_detection():
    assert visuals.is_quota_error(429)
    assert visuals.is_quota_error(400, "you have used up your daily free allocation of 10,000 neurons")
    assert visuals.is_quota_error(403, '{"error": "RESOURCE_EXHAUSTED quota"}')
    assert not visuals.is_quota_error(400, "Additional or unevaluated properties '/seed'")
    assert not visuals.is_quota_error(500, "internal error")


def test_flux_breaker_trips_once_on_429_and_stops_calling(monkeypatch, tmp_path, capsys):
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(url)
        return _Resp(429, FLUX_429)

    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "a")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "t")
    monkeypatch.setattr(visuals.requests, "post", fake_post)
    assert visuals.flux_available()
    for k in range(5):
        assert visuals.flux_generate("servers", tmp_path / f"x{k}.png") is None
    assert len(calls) == 1                       # never called again after the first 429
    assert visuals.breaker_open("flux") and not visuals.flux_available()
    out = capsys.readouterr().out
    assert out.count("quota/limit hit") == 1     # logged once
    assert "flux" in visuals.breaker_status()
    # the legacy single-still path honours the same breaker
    assert visuals._cloudflare_generate("x", tmp_path / "y.png") is None
    assert len(calls) == 1


def test_flux_breaker_trips_on_daily_allocation_body_without_429(monkeypatch, tmp_path):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "a")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "t")
    monkeypatch.setattr(visuals.requests, "post", lambda *a, **k: _Resp(400, FLUX_429))
    assert visuals.flux_generate("servers", tmp_path / "x.png") is None
    assert visuals.breaker_open("flux")


def test_flux_non_quota_error_does_not_trip(monkeypatch, tmp_path):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "a")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "t")
    monkeypatch.setattr(visuals.requests, "post", lambda *a, **k: _Resp(400, "bad prompt"))
    assert visuals.flux_generate("servers", tmp_path / "x.png") is None
    assert not visuals.breaker_open("flux")


def test_pixabay_breaker_skips_requests_but_serves_cache(monkeypatch):
    monkeypatch.setenv("PIXABAY_API_KEY", "k")
    calls = []

    def fake_get(url, params=None, timeout=None, **kw):
        calls.append(params.get("q"))
        if params.get("q") == "cached":
            return _Resp(200, payload={"hits": [{"id": 1, "tags": "server, network"}]})
        return _Resp(429, "Too many requests")

    monkeypatch.setattr(visuals.requests, "get", fake_get)
    assert visuals.pixabay_still_hits("cached")[0]["id"] == 1
    assert visuals.pixabay_still_hits("other") == []
    assert visuals.breaker_open("pixabay")
    assert visuals.pixabay_video_hits("third") == []
    assert visuals._pixabay_candidates("fourth") == []
    assert calls == ["cached", "other"]          # no requests after the 429
    assert visuals.pixabay_still_hits("cached")[0]["id"] == 1   # 24 h cache still served


# ------------------------------------------------------------------ FLUX -> Pixabay fallback

def _img(path, color=(10, 20, 30)):
    Image.new("RGB", (64, 36), color).save(path)
    return path


def _still_hit(i, tags):
    return {"id": i, "tags": tags, "imageWidth": 4000, "imageHeight": 3000,
            "largeImageURL": f"https://example.test/{i}.jpg", "pageURL": f"https://pixabay.com/{i}"}


def test_gather_uses_pixabay_instead_of_flux_after_quota(monkeypatch, tmp_path):
    visuals.trip_breaker("flux", "HTTP 429")
    flux_calls = []
    monkeypatch.setattr(visuals, "flux_generate", lambda *a, **k: flux_calls.append(1))
    monkeypatch.setattr(visuals, "pixabay_video_hits", lambda q: [])
    queried = []
    pool = {
        "phishing email": [_still_hit(1, "phishing, email, computer")],
        "email inbox": [_still_hit(2, "email, inbox, laptop"), _still_hit(3, "email, laptop, screen")],
        "cybersecurity": [_still_hit(4, "cybersecurity, hacker, computer")],
        "padlock digital": [_still_hit(5, "padlock, digital, security")],
    }

    def stills(q, o):
        queried.append(q)
        return pool.get(q, [])

    monkeypatch.setattr(visuals, "pixabay_still_hits", stills)
    counter = iter(range(100))

    def fake_download(candidate, path):
        _img(path, (next(counter) * 7 % 255, 50, 90))
        return {"sha256": visuals._sha256(path), "width": 4000, "height": 3000}

    monkeypatch.setattr(visuals, "_download_candidate", fake_download)
    used = {"hashes": set(), "ids": set()}
    spec = {"visual_queries": ["phishing email"], "terms": sb.topic_terms(["phishing email detection"])}
    assets = visuals.gather_visuals(spec, tmp_path, 4, used, label="t", min_count=0)
    assert not flux_calls
    assert [a["provider"] for a in assets] == ["pixabay"] * 4
    assert len({a["asset_id"] for a in assets}) == 4
    assert any(q in queried for q in sb.FALLBACK_TECH_QUERIES)
    assert not any(a["provider"] == sb.GRAPHIC_PROVIDER for a in assets)


def test_gather_skips_placeholder_when_min_count_zero(monkeypatch, tmp_path):
    visuals.trip_breaker("flux", "HTTP 429")
    monkeypatch.setattr(visuals, "pixabay_video_hits", lambda q: [])
    monkeypatch.setattr(visuals, "pixabay_still_hits", lambda q, o: [])
    used = {"hashes": set(), "ids": set()}
    spec = {"visual_queries": ["phishing email"]}
    assert visuals.gather_visuals(spec, tmp_path, 3, used, label="z", min_count=0) == []
    # Default contract (min_count=count) still never returns fewer assets than asked.
    legacy = visuals.gather_visuals(spec, tmp_path, 2, used, label="y")
    assert [a["provider"] for a in legacy] == [sb.GRAPHIC_PROVIDER] * 2


def test_assign_reuses_stock_before_gradient():
    stock = [{"provider": "pixabay", "reusable": False}, {"provider": "pixabay_video", "reusable": False}]
    plan = sb.assign_assets(6, stock, allow_stock_reuse=True)
    assert None not in plan and set(plan) == {0, 1}
    assert all(a != b for a, b in zip(plan, plan[1:]))
    # Own/AI assets are still preferred for reuse when they exist.
    mixed = stock + [{"provider": "cloudflare_workers_ai_flux", "reusable": True}]
    assert all(p == 2 for p in sb.assign_assets(6, mixed, allow_stock_reuse=True)[3:])
    # Real stock is preferred over an existing gradient graphic.
    with_grad = stock + [{"provider": sb.GRAPHIC_PROVIDER, "reusable": True}]
    assert 2 not in sb.assign_assets(7, with_grad, allow_stock_reuse=True)[3:]


def _timeline(slide_shots):
    shots, slides, t = [], [], 0.0
    for s, n in enumerate(slide_shots):
        ids = []
        for j in range(n):
            shots.append({"slide": s, "shot": j, "start": t, "end": t + 4.0})
            ids.append(len(shots) - 1)
            t += 4.0
        slides.append({"index": s, "start": ids and shots[ids[0]]["start"], "end": t, "shots": ids})
    return {"total": t, "slides": slides, "shots": shots}


def _stock_asset(tmp_path, name):
    p = _img(tmp_path / f"{name}.png")
    return {"kind": "image", "provider": "pixabay", "source_url": "https://pixabay.com/x", "asset_id": name,
            "license": "Pixabay Content License", "license_url": "https://pixabay.com/l", "sha256": name,
            "path": str(p), "reusable": False, "third_party": True}


def test_long_assignment_borrows_real_assets_and_stays_under_cap(tmp_path):
    tl = _timeline([4, 4, 3])
    per_slide = [[_stock_asset(tmp_path, "a"), _stock_asset(tmp_path, "b")], [], [_stock_asset(tmp_path, "c")]]
    shot_assets, manifest = prod._assign(tl, per_slide, (64, 36), tmp_path, "long")
    assert not any(a["provider"] == sb.GRAPHIC_PROVIDER for a in shot_assets)
    assert not any(e["review_required"] for e in manifest)
    assert any(e.get("borrowed") for e in manifest)
    assert sorted(s for e in manifest for s in e["shots"]) == list(range(len(tl["shots"])))


def test_gradient_share_over_ten_percent_marks_review_required(tmp_path):
    tl = _timeline([4, 4])
    shot_assets, manifest = prod._assign(tl, [[], []], (64, 36), tmp_path, "long")
    assert sb.graphic_share(shot_assets) == 1.0
    grads = [e for e in manifest if e["provider"] == sb.GRAPHIC_PROVIDER]
    assert grads and all(e["review_required"] for e in grads)


def test_gradient_share_at_or_under_cap_is_not_flagged():
    real = {"provider": "pixabay"}
    grad = {"provider": sb.GRAPHIC_PROVIDER}
    manifest = [{"provider": sb.GRAPHIC_PROVIDER, "review_required": False}]
    assert prod.apply_graphic_cap([real] * 9 + [grad], manifest, "t") == pytest.approx(0.1)
    assert manifest[0]["review_required"] is False
    prod.apply_graphic_cap([real] * 8 + [grad] * 2, manifest, "t")
    assert manifest[0]["review_required"] is True


# ------------------------------------------------------------------ off-topic filters

TERMS = sb.topic_terms("how ai catches phishing emails security network digital")


@pytest.mark.parametrize("tags", [
    "barrier, road, construction, technology",
    "traffic cone, warning, digital",
    "crystal, prism, light, technology",
    "chandelier, crystals, hanging, digital",
    "spiral staircase, stairs, architecture, technology",
    "woman, camera, photographer, digital",
    "camera lens, photography, technology",
    "microscope, science, technology, digital",
    "laboratory, research, data",
    "police, riot, security, network",
    "crowd, people, network",
    "cat, computer, keyboard",
    "city skyline, network, digital",
])
def test_off_topic_stock_tags_rejected(tags):
    ok, _, reason = sb.tag_relevance(tags, TERMS)
    assert not ok, tags
    assert reason


@pytest.mark.parametrize("tags", [
    "email, phishing, computer, laptop",
    "server, data center, network",
    "hacker, phishing, keyboard",
    "padlock, digital, security",
])
def test_on_topic_tech_tags_still_accepted(tags):
    assert sb.tag_relevance(tags, TERMS)[0], tags


def test_laboratory_allowed_with_strong_computing_subject():
    assert sb.offtopic_reason("laboratory, server, computer") == ""
    assert sb.offtopic_reason("laboratory, technology, digital")


def test_off_topic_title_filter_for_wikimedia_candidates(monkeypatch, tmp_path):
    monkeypatch.setattr(visuals, "_wikimedia_candidates", lambda q: [
        {"provider": "wikimedia_commons", "title": "Spiral staircase phishing email",
         "download_url": "https://example.test/s.jpg", "width": 1080, "height": 1920}])
    monkeypatch.setattr(visuals, "_pixabay_candidates", lambda q: [])
    monkeypatch.setattr(visuals, "_cloudflare_generate", lambda p, o: None)
    downloads = []
    monkeypatch.setattr(visuals, "_download_candidate", lambda c, p: downloads.append(c) or c)
    manifest, _ = visuals.select_visual({"type": "news_event", "entities": ["phishing email"],
                                         "keywords": ["phishing"]}, tmp_path / "f.png", set())
    assert not downloads and manifest["provider"] == "gradient"


def test_fallback_queries_prefer_topic_overlap():
    q = sb.fallback_queries(sb.topic_terms("phishing email inbox"))
    assert q[0] == "email inbox"
    assert set(q) == set(sb.FALLBACK_TECH_QUERIES)
