import hashlib

from src.visuals import _allowed_wikimedia_license, relevance_score, select_visual


def test_wikimedia_license_gate_only_allows_no_attribution_categories():
    assert _allowed_wikimedia_license({"shortname": "Public domain", "url": ""})
    assert _allowed_wikimedia_license({"shortname": "CC0", "url": "https://creativecommons.org/publicdomain/zero/1.0/"})
    assert _allowed_wikimedia_license({"shortname": "PD-Mark", "url": ""})
    assert not _allowed_wikimedia_license({"shortname": "CC BY 4.0", "url": "https://creativecommons.org/licenses/by/4.0/"})
    assert not _allowed_wikimedia_license({"shortname": "CC BY-SA 4.0", "url": "https://creativecommons.org/licenses/by-sa/4.0/"})


def test_relevance_score_prefers_entity_and_keyword_match():
    intent = {"entities": ["ExamplePhone"], "keywords": ["foldable", "display"]}
    strong = relevance_score({"title": "ExamplePhone foldable display", "width": 1080, "height": 1920}, intent)
    weak = relevance_score({"title": "forest mountain", "width": 1920, "height": 1080}, intent)
    assert strong > weak


def test_visual_provider_failure_uses_gradient_and_flags_essential_news(monkeypatch, tmp_path):
    import src.visuals as visuals
    monkeypatch.setattr(visuals, "_wikimedia_candidates", lambda query: [])
    monkeypatch.setattr(visuals, "_pixabay_candidates", lambda query: [])
    monkeypatch.setattr(visuals, "_cloudflare_generate", lambda prompt, path: None)
    manifest, review_required = select_visual({
        "type": "news_event", "entities": ["Example Corp"], "keywords": ["launch"],
        "real_image_essential": True,
    }, tmp_path / "frame.png", set())
    assert manifest["provider"] == "gradient"
    assert review_required is True


def test_visual_provider_failure_does_not_flag_concept(monkeypatch, tmp_path):
    import src.visuals as visuals
    monkeypatch.setattr(visuals, "_pixabay_candidates", lambda query: [])
    monkeypatch.setattr(visuals, "_cloudflare_generate", lambda prompt, path: None)
    manifest, review_required = select_visual({
        "type": "concept", "entities": [], "keywords": ["neural network"],
        "real_image_essential": False,
    }, tmp_path / "frame.png", set())
    assert manifest["provider"] == "gradient"
    assert review_required is False


def test_gemini_image_is_opt_in(monkeypatch, tmp_path):
    import src.visuals as visuals
    called = []
    monkeypatch.setattr(visuals.requests, "post", lambda *a, **k: called.append(True))
    monkeypatch.delenv("DYPOL_GEMINI_IMAGE_ENABLED", raising=False)
    assert visuals._gemini_image_generate("concept", tmp_path / "x.png") is None
    assert called == []


def test_gemini_zero_quota_429_short_circuits_later_calls(monkeypatch, tmp_path):
    import src.visuals as visuals
    class Response:
        status_code = 429
        text = '{"error":{"status":"RESOURCE_EXHAUSTED","message":"quota limit: 0"}}'
        def json(self):
            return {"error": {"status": "RESOURCE_EXHAUSTED", "message": "quota limit: 0"}}
    calls = []
    monkeypatch.setattr(visuals.requests, "post", lambda *a, **k: (calls.append(True) or Response()))
    monkeypatch.setenv("DYPOL_GEMINI_IMAGE_ENABLED", "1")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    visuals._GEMINI_IMAGE_DISABLED_FOR_QUOTA = False
    assert visuals._gemini_image_generate("concept", tmp_path / "one.png") is None
    assert visuals._GEMINI_IMAGE_DISABLED_FOR_QUOTA is True
    assert visuals._gemini_image_generate("concept", tmp_path / "two.png") is None
    assert len(calls) == 1
