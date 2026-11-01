"""NVIDIA hosted FLUX.1-schnell fallback (only after Cloudflare FLUX hits its daily limit)."""
import base64
import io

import pytest
from PIL import Image

from dypol_control import manifest_has_ai_asset, resolve_contains_synthetic_media
from src import production as prod
from src import visuals

CF_429 = ('{"errors":[{"message":"AiError: you have used up your daily free allocation of 10,000 neurons",'
          '"code":4006}],"success":false}')


def _png_b64(color=(30, 60, 90), size=(64, 36)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


class _Resp:
    def __init__(self, status, payload=None, text=""):
        self.status_code = status
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise visuals.requests.HTTPError(str(self.status_code))


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    visuals.reset_breakers()
    monkeypatch.setenv("VERTICALS_HOME", str(tmp_path / "vh"))
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-token")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    yield
    visuals.reset_breakers()


def _router(cf, nv, calls):
    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append((url, json))
        return (nv if "nvidia" in url else cf)(json)
    return fake_post


def test_cloudflare_success_never_calls_nvidia(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(visuals.requests, "post", _router(
        lambda b: _Resp(200, {"result": {"image": _png_b64()}}), lambda b: _Resp(500), calls))
    asset = visuals.flux_generate("servers in a rack", tmp_path / "a.png")
    assert asset["provider"] == "cloudflare_workers_ai_flux"
    assert all("nvidia" not in url for url, _ in calls)
    assert not visuals.nvidia_flux_available()  # Cloudflare limit not hit -> fallback idle


def test_no_cloudflare_credentials_is_not_a_limit(monkeypatch, tmp_path):
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN")
    calls = []
    monkeypatch.setattr(visuals.requests, "post", _router(lambda b: _Resp(500), lambda b: _Resp(500), calls))
    assert visuals.flux_generate("servers", tmp_path / "a.png") is None
    assert calls == []  # NVIDIA is a fallback for the Cloudflare daily limit only


def test_cloudflare_limit_falls_back_to_nvidia_same_prompt(monkeypatch, tmp_path, capsys):
    calls = []
    monkeypatch.setattr(visuals.requests, "post", _router(
        lambda b: _Resp(429, text=CF_429),
        lambda b: _Resp(200, {"artifacts": [{"base64": _png_b64(), "finishReason": "SUCCESS", "seed": 7}]}),
        calls))
    asset = visuals.flux_generate("servers in a rack", tmp_path / "a.png", orientation="portrait", seed=3)
    out = capsys.readouterr().out
    assert "DYPOL_PROVIDER_LIMIT provider=flux " in out
    assert "provider=nvidia_flux" not in out
    assert asset["provider"] == "nvidia_flux" and asset["fallback_for"] == "cloudflare_workers_ai_flux"
    assert asset["path"].endswith("a.png") and asset["sha256"]
    (cf_url, cf_body), (nv_url, nv_body) = calls
    assert nv_url == visuals.NVIDIA_FLUX_URL
    assert nv_body["prompt"] == cf_body["prompt"]            # same no-text / no-logo prompt
    assert "no text" in nv_body["prompt"].lower()
    assert (nv_body["width"], nv_body["height"]) == (768, 1344) and nv_body["steps"] <= 4
    # Next image: Cloudflare is skipped for the rest of the run, NVIDIA only.
    calls.clear()
    assert visuals.flux_generate("cables", tmp_path / "b.png")["provider"] == "nvidia_flux"
    assert [u for u, _ in calls] == [visuals.NVIDIA_FLUX_URL]


def test_nvidia_limit_logs_once_and_stops(monkeypatch, tmp_path, capsys):
    calls = []
    monkeypatch.setattr(visuals.requests, "post", _router(
        lambda b: _Resp(429, text=CF_429), lambda b: _Resp(429, text='{"detail":"Too Many Requests"}'), calls))
    for k in range(4):
        assert visuals.flux_generate("servers", tmp_path / f"x{k}.png") is None
    out = capsys.readouterr().out
    assert out.count("DYPOL_PROVIDER_LIMIT provider=nvidia_flux") == 1
    assert out.count("DYPOL_PROVIDER_LIMIT provider=flux ") == 1
    assert len(calls) == 2                       # one Cloudflare, one NVIDIA, then nothing
    assert not visuals.ai_image_available()
    assert set(visuals.breaker_status()) == {"flux", "nvidia_flux"}


def test_nvidia_limit_falls_through_to_stock_in_gather(monkeypatch, tmp_path):
    visuals.trip_breaker("flux", "HTTP 429")
    visuals.trip_breaker("nvidia_flux", "HTTP 429")
    monkeypatch.setattr(visuals, "flux_generate", lambda *a, **k: pytest.fail("no AI source may be called"))
    stock = []

    def fake_stock(kind, query, terms, out_dir, used, px, label, **kw):
        stock.append(kind)
        path = tmp_path / f"{label}_{kind}.png"
        Image.new("RGB", (32, 18)).save(path)
        return {"kind": "image", "provider": "pixabay", "path": str(path), "sha256": label + kind}

    monkeypatch.setattr(visuals, "_stock_attempt", fake_stock)
    assets = visuals.gather_visuals({"visual_queries": ["data center"]}, tmp_path, 3, {"hashes": set(), "ids": set()},
                                    allow_video=False)
    assert assets and all(a["provider"] == "pixabay" for a in assets)


def test_gather_uses_nvidia_for_flux_slots_after_cloudflare_limit(monkeypatch, tmp_path):
    visuals.trip_breaker("flux", "HTTP 429")
    monkeypatch.setattr(visuals.requests, "post", _router(
        lambda b: pytest.fail("Cloudflare must not be called after its limit"),
        lambda b: _Resp(200, {"artifacts": [{"base64": _png_b64(color=((b["seed"] * 80) % 255, 40, 200)),
                                             "finishReason": "SUCCESS"}]}), []))
    monkeypatch.setattr(visuals, "_stock_attempt", lambda *a, **k: None)
    assets = visuals.gather_visuals({"visual_queries": ["data center"], "image_prompt": "server hall"},
                                    tmp_path, 2, {"hashes": set(), "ids": set()},
                                    allow_video=False, allow_still=False)
    assert [a["provider"] for a in assets] == ["nvidia_flux", "nvidia_flux"]


def test_square_retry_on_422_and_content_filter(monkeypatch, tmp_path):
    visuals.trip_breaker("flux", "HTTP 429")
    sizes = []

    def nv(body):
        sizes.append((body["width"], body["height"]))
        if (body["width"], body["height"]) != (1024, 1024):
            return _Resp(422, text='{"detail":"width must be 1024"}')
        return _Resp(200, {"artifacts": [{"base64": _png_b64(), "finishReason": "SUCCESS"}]})

    monkeypatch.setattr(visuals.requests, "post", _router(lambda b: _Resp(500), nv, []))
    assert visuals.flux_generate("servers", tmp_path / "a.png")["provider"] == "nvidia_flux"
    assert sizes == [(1344, 768), (1024, 1024)]
    visuals.flux_generate("servers 2", tmp_path / "b.png")
    assert sizes[-1] == (1024, 1024)  # remembered for the rest of the run
    monkeypatch.setattr(visuals.requests, "post", _router(lambda b: _Resp(500), lambda b: _Resp(
        200, {"artifacts": [{"base64": _png_b64(), "finishReason": "CONTENT_FILTERED"}]}), []))
    assert visuals.flux_generate("servers 3", tmp_path / "c.png") is None


def test_throttle_spaces_requests(monkeypatch, tmp_path):
    visuals.trip_breaker("flux", "HTTP 429")
    monkeypatch.setenv("DYPOL_NVIDIA_FLUX_MIN_INTERVAL", "1.5")
    sleeps = []
    monkeypatch.setattr(visuals.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(visuals.requests, "post", _router(lambda b: _Resp(500), lambda b: _Resp(
        200, {"artifacts": [{"base64": _png_b64(), "finishReason": "SUCCESS"}]}), []))
    visuals.flux_generate("a", tmp_path / "a.png")
    visuals.flux_generate("b", tmp_path / "b.png")
    assert sleeps and 1.0 < sleeps[-1] <= 1.5  # <= 40 requests / minute


def test_disabled_flag_and_auth_error(monkeypatch, tmp_path):
    visuals.trip_breaker("flux", "HTTP 429")
    monkeypatch.setenv("DYPOL_NVIDIA_FLUX_ENABLED", "0")
    assert not visuals.nvidia_flux_available()
    monkeypatch.setenv("DYPOL_NVIDIA_FLUX_ENABLED", "1")
    calls = []
    monkeypatch.setattr(visuals.requests, "post", _router(lambda b: _Resp(500), lambda b: _Resp(401), calls))
    assert visuals.flux_generate("a", tmp_path / "a.png") is None
    assert visuals.flux_generate("b", tmp_path / "b.png") is None
    assert len(calls) == 1 and "nvidia_flux" not in visuals.breaker_status()  # auth error is not a quota


def test_manifest_records_provider_and_disclosure(tmp_path):
    asset = {"kind": "image", "provider": "nvidia_flux", "asset_id": "flux.1-schnell",
             "license": visuals.NVIDIA_FLUX_LICENSE, "source_url": visuals.NVIDIA_FLUX_SOURCE_URL,
             "fallback_for": "cloudflare_workers_ai_flux", "path": str(tmp_path / "n.png"), "sha256": "x"}
    timeline = {"shots": [{"start": 0.0, "end": 2.0}]}
    entry = prod.manifest_entry(asset, "long", 1, [0], timeline)
    assert entry["provider"] == "nvidia_flux" and entry["fallback_for"] == "cloudflare_workers_ai_flux"
    assert manifest_has_ai_asset([entry]) is True
    assert resolve_contains_synthetic_media([entry], env={}) is True
