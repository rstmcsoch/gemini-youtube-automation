"""YouTube altered/synthetic-content disclosure (status.containsSyntheticMedia)."""

import sys
import types

import pytest

from dypol_control import (
    PhasePolicy,
    asset_visual_manifest,
    build_insert_body,
    execute_phase,
    manifest_has_ai_asset,
    resolve_contains_synthetic_media,
    save_candidate,
)

FLUX_ENTRY = {
    "video": "short", "kind": "image", "provider": "cloudflare_workers_ai_flux",
    "asset_id": "flux-1-schnell",
    "license": "AI-generated with FLUX.1 [schnell] (Apache-2.0 model) on Cloudflare Workers AI",
}
STOCK_ENTRY = {"video": "long", "kind": "video", "provider": "pixabay_video", "asset_id": "123", "license": "Pixabay"}
GRAPHIC_ENTRY = {"video": "long", "kind": "image", "provider": "dypol_generated_graphic",
                 "asset_id": "graphic-1", "license": "Original generated graphic (no third-party content)"}


@pytest.mark.parametrize("entry", [
    {"provider": "cloudflare_workers_ai"},
    {"provider": "cloudflare_workers_ai_flux"},
    {"provider": "FLUX"},
    {"provider": "ai-generated"},
    {"provider": "gemini_image", "license": "AI-generated; review applicable Google terms"},
    {"asset": {"provider": "cloudflare_workers_ai", "asset_id": "flux-1-schnell"}},  # legacy nested shape
])
def test_ai_assets_detected(entry):
    assert manifest_has_ai_asset([STOCK_ENTRY, entry]) is True


def test_non_ai_manifest_not_flagged():
    assert manifest_has_ai_asset([STOCK_ENTRY, GRAPHIC_ENTRY, {"provider": "wikimedia_commons"}]) is False
    assert manifest_has_ai_asset([]) is False
    assert manifest_has_ai_asset(None) is False


def test_auto_mode_follows_manifest():
    assert resolve_contains_synthetic_media([FLUX_ENTRY], env={}) is True
    assert resolve_contains_synthetic_media([STOCK_ENTRY], env={}) is False
    assert resolve_contains_synthetic_media([FLUX_ENTRY], env={"YOUTUBE_CONTAINS_SYNTHETIC_MEDIA": "auto"}) is True


def test_env_override_forces_value():
    assert resolve_contains_synthetic_media([STOCK_ENTRY], env={"YOUTUBE_CONTAINS_SYNTHETIC_MEDIA": "true"}) is True
    assert resolve_contains_synthetic_media([FLUX_ENTRY], env={"YOUTUBE_CONTAINS_SYNTHETIC_MEDIA": "false"}) is False
    # Unknown values fall back to auto detection.
    assert resolve_contains_synthetic_media([FLUX_ENTRY], env={"YOUTUBE_CONTAINS_SYNTHETIC_MEDIA": "bogus"}) is True


def test_insert_body_carries_flag_and_keeps_privacy():
    body = build_insert_body("T", "D", "a,b", "unlisted", contains_synthetic_media=True)
    assert body["status"] == {
        "privacyStatus": "unlisted", "selfDeclaredMadeForKids": False, "containsSyntheticMedia": True,
    }
    assert build_insert_body("T", "D", "", "private")["status"]["containsSyntheticMedia"] is False


def test_asset_manifest_is_per_video():
    candidate = {"visual_manifest": [STOCK_ENTRY, FLUX_ENTRY]}
    assert asset_visual_manifest(candidate, {"kind": "long"}) == [STOCK_ENTRY]
    assert asset_visual_manifest(candidate, {"kind": "short"}) == [FLUX_ENTRY]
    # Untagged manifests fall back to the whole candidate manifest.
    assert asset_visual_manifest({"visual_manifest": [{"provider": "flux"}]}, {"kind": "long"}) == [{"provider": "flux"}]


def test_publish_phase_passes_each_videos_manifest_to_uploader(tmp_path):
    long_path, short_path = tmp_path / "long.mp4", tmp_path / "short.mp4"
    long_path.write_bytes(b"x")
    short_path.write_bytes(b"x")
    candidate = {
        "job_id": "job-1", "lesson_title": "Lesson", "review_required": False,
        "visual_manifest": [STOCK_ENTRY, FLUX_ENTRY],
        "assets": [
            {"kind": "long", "path": str(long_path), "title": "Long", "description": "", "tags": ""},
            {"kind": "short", "path": str(short_path), "title": "Short", "description": "", "tags": ""},
        ],
    }
    save_candidate(tmp_path / "candidates", "job-1", candidate)
    policy = PhasePolicy(phase="publish", allow_generate=False, allow_upload=True, review_required=True,
                         auto_publish=False, visibility="private", job_id="job-1")
    seen = {}

    def fake_upload(asset, _policy):
        seen[asset["kind"]] = resolve_contains_synthetic_media(asset.get("visual_manifest"), env={})
        return {"youtube_id": asset["kind"] + "-id", "kind": asset["kind"]}

    result = execute_phase(policy, generate_fn=None, upload_fn=fake_upload, plan={"lessons": []},
                           candidate_root=tmp_path / "candidates", results_path=tmp_path / "r.json")
    assert seen == {"long": False, "short": True}
    assert result["status"] == "uploaded" and result["visibility"] == "private"


def _stub_google_modules(monkeypatch):
    """Let src.uploader import without the Google client libraries installed."""
    try:
        import googleapiclient.http  # noqa: F401
        import google_auth_oauthlib.flow  # noqa: F401
        import google.oauth2.credentials  # noqa: F401
        return
    except ImportError:
        pass
    names = {
        "google": {}, "google.oauth2": {}, "google.oauth2.credentials": {"Credentials": object},
        "google.auth": {}, "google.auth.transport": {}, "google.auth.transport.requests": {"Request": object},
        "google_auth_oauthlib": {}, "google_auth_oauthlib.flow": {"InstalledAppFlow": object},
        "googleapiclient": {}, "googleapiclient.discovery": {"build": None},
        "googleapiclient.http": {"MediaFileUpload": lambda *a, **k: None},
    }
    for name, attrs in names.items():
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, "src.uploader", raising=False)


@pytest.mark.parametrize("manifest,env_value,expected", [
    ([FLUX_ENTRY], None, True),
    ([STOCK_ENTRY], None, False),
    ([STOCK_ENTRY], "true", True),
    ([FLUX_ENTRY], "false", False),
])
def test_uploader_insert_body(monkeypatch, tmp_path, manifest, env_value, expected):
    _stub_google_modules(monkeypatch)
    import src.uploader as uploader

    captured = {}

    class FakeRequest:
        def next_chunk(self):
            return None, {"id": "vid123"}

    class FakeVideos:
        def insert(self, part, body, media_body):
            captured["part"], captured["body"] = part, body
            return FakeRequest()

    class FakeYouTube:
        def videos(self):
            return FakeVideos()

    monkeypatch.setattr(uploader, "get_authenticated_service", lambda: FakeYouTube())
    monkeypatch.setattr(uploader, "MediaFileUpload", lambda *a, **k: None)
    monkeypatch.setenv("YOUTUBE_PRIVACY_STATUS", "unlisted")
    if env_value is None:
        monkeypatch.delenv("YOUTUBE_CONTAINS_SYNTHETIC_MEDIA", raising=False)
    else:
        monkeypatch.setenv("YOUTUBE_CONTAINS_SYNTHETIC_MEDIA", env_value)
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")

    record = uploader.upload_to_youtube(video, "Title", "Desc", "a,b", kind="short", visual_manifest=manifest)

    assert captured["body"]["status"]["containsSyntheticMedia"] is expected
    assert captured["body"]["status"]["privacyStatus"] == "unlisted"
    assert "status" in captured["part"]
    assert record["contains_synthetic_media"] is expected
    assert record["visibility"] == "unlisted"
