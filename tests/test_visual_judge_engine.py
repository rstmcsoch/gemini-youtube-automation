"""Engine A integration of the AI visual judge (mocked HTTP; no network)."""
import json
from pathlib import Path

from PIL import Image

from src import production
from src import storyboard as sb
from src import visual_judge as vj
import src.visuals as visuals

GOOD = {"description": "server racks", "subject_match": "yes", "issues": [], "reason": "fits", "score": 8}
MEH = {"description": "circuit", "subject_match": "partial", "issues": ["generic_filler"], "reason": "x", "score": 5}
CAT = {"description": "kitten", "subject_match": "no", "issues": ["wrong_subject"], "reason": "cat", "score": 1}


def _resp(v):
    return 200, {}, json.dumps({"choices": [{"message": {"content": json.dumps(v)}}]})


def _install(responses):
    queue = list(responses)
    bodies = []

    def transport(url, headers, body, timeout):
        bodies.append(json.loads(body))
        r = queue.pop(0) if len(queue) > 1 else queue[0]
        return r
    judge = vj.VisualJudge(env={"NVIDIA_API_KEY": "nvapi-test", "VISUAL_JUDGE_MODELS": "m1"},
                           transport=transport, sleep=lambda s: None,
                           store=vj.JudgeStore(vj.MemoryBackend()))
    vj.reset_judge(judge)
    return judge, bodies


def _hit(i, tags="server, computer, data center"):
    return {"id": i, "tags": tags, "imageWidth": 4000, "imageHeight": 3000,
            "largeImageURL": f"https://example.test/{i}.jpg", "pageURL": f"https://pixabay.com/{i}"}


def _patch_stock(monkeypatch, hits):
    visuals.trip_breaker("flux", "test")  # keep FLUX out of the slot order
    monkeypatch.setattr(visuals, "pixabay_video_hits", lambda q: [])
    monkeypatch.setattr(visuals, "pixabay_still_hits", lambda q, o: hits if "server" in q else [])
    downloads = []

    def fake_download(candidate, path):
        downloads.append(candidate["download_url"])
        n = len(downloads)
        Image.new("RGB", (64, 36), (n * 37 % 255, n * 91 % 255, 120)).save(path)
        return {"sha256": visuals._sha256(path), "width": 4000, "height": 3000}
    monkeypatch.setattr(visuals, "_download_candidate", fake_download)
    return downloads


SPEC = {"visual_queries": ["server room"], "terms": sb.topic_terms(["server room data center"]),
        "video_title": "Data centers explained", "narration": "Racks of servers hum in the data center."}


def test_judge_rejects_then_accepts_and_records_verdict(monkeypatch, tmp_path):
    judge, bodies = _install([_resp(CAT), _resp(GOOD)])
    downloads = _patch_stock(monkeypatch, [_hit(1), _hit(2), _hit(3)])
    used = {"hashes": set(), "ids": set()}
    assets = visuals.gather_visuals(dict(SPEC), tmp_path, 1, used, label="t", allow_video=False, min_count=0)
    visuals.reset_breakers()
    assert [a["asset_id"] for a in assets] == ["2"]
    assert assets[0]["judge"]["decision"] == "PASS" and assets[0]["judge"]["judge_model"] == "m1"
    assert len(downloads) == 2 and len(bodies) == 2
    assert "Racks of servers" in bodies[0]["messages"][1]["content"][0]["text"]
    assert not (tmp_path / "t_0_px_1.png").exists()          # rejected candidate deleted
    assert judge.is_rejected("pixabay:1", "server room")       # rejection memory
    # Next job: the rejected asset is skipped before download.
    downloads.clear()
    visuals.gather_visuals(dict(SPEC), tmp_path, 1, {"hashes": set(), "ids": set()}, label="u",
                           allow_video=False, min_count=0)
    visuals.reset_breakers()
    assert "https://example.test/1.jpg" not in downloads
    entry = production.manifest_entry(dict(assets[0], path=str(tmp_path / "x.png")), "long", 1, [0],
                                      {"shots": [{"start": 0, "end": 4}]})
    assert entry["judge"]["score"] == 8 and entry["judge_status"] == "active" and not entry["review_required"]


def test_best_marginal_used_with_review_flag(monkeypatch, tmp_path):
    _install([_resp(CAT), _resp(MEH), _resp(CAT)])
    _patch_stock(monkeypatch, [_hit(1), _hit(2), _hit(3)])
    used = {"hashes": set(), "ids": set()}
    assets = visuals.gather_visuals(dict(SPEC), tmp_path, 1, used, label="t", allow_video=False, min_count=0)
    visuals.reset_breakers()
    assert assets[0]["asset_id"] == "2" and assets[0]["judge"]["decision"] == "MARGINAL"
    assert assets[0]["judge"]["needs_review"]
    entry = production.manifest_entry(dict(assets[0], path="x.png"), "long", 1, [0], {"shots": [{"start": 0, "end": 4}]})
    assert entry["review_required"] and "visual judge" in entry["review_reason"]


def test_five_candidates_then_three_broadened(monkeypatch, tmp_path):
    judge, bodies = _install([_resp(CAT)])
    queried = []
    visuals.trip_breaker("flux", "test")
    monkeypatch.setattr(visuals, "pixabay_video_hits", lambda q: [])

    def stills(q, o):
        queried.append(q)
        base = 100 if q == "server room" else (200 if q == "server" else 300)
        return [_hit(base + i) for i in range(8)]
    monkeypatch.setattr(visuals, "pixabay_still_hits", stills)
    n = iter(range(1000))

    def fake_download(candidate, path):
        k = next(n)
        Image.new("RGB", (64, 36), (k * 13 % 255, k * 29 % 255, k * 7 % 255)).save(path)
        return {"sha256": visuals._sha256(path), "width": 4000, "height": 3000}
    monkeypatch.setattr(visuals, "_download_candidate", fake_download)
    asset = visuals._stock_still("server room", sb.topic_terms(["server room computer"]), tmp_path,
                                 {"hashes": set(), "ids": set()}, "horizontal", "s", ctx=SPEC)
    visuals.reset_breakers()
    assert asset is None
    assert len(bodies) == 8 and queried[:2] == ["server room", "server"]


def test_circuit_open_degrades_to_mechanical(monkeypatch, tmp_path):
    judge, bodies = _install([(402, {}, "payment required")])
    _patch_stock(monkeypatch, [_hit(1), _hit(2)])
    assets = visuals.gather_visuals(dict(SPEC), tmp_path, 1, {"hashes": set(), "ids": set()}, label="t",
                                    allow_video=False, min_count=0)
    visuals.reset_breakers()
    assert assets[0]["asset_id"] == "1" and "judge" not in assets[0]   # mechanical pick, unjudged
    assert len(bodies) == 1
    s = judge.summary()
    assert s["judge_status"] == "circuit_open" and s["needs_review"]


def test_disabled_judge_is_todays_behaviour(monkeypatch, tmp_path):
    downloads = _patch_stock(monkeypatch, [_hit(1), _hit(2)])
    assets = visuals.gather_visuals(dict(SPEC), tmp_path, 1, {"hashes": set(), "ids": set()}, label="t",
                                    allow_video=False, min_count=0)
    visuals.reset_breakers()
    assert assets[0]["asset_id"] == "1" and "judge" not in assets[0] and len(downloads) == 1
    entry = production.manifest_entry(dict(assets[0]), "long", 1, [0], {"shots": [{"start": 0, "end": 4}]})
    assert entry["judge_status"] == "disabled" and entry["review_required"] is False
    assert production._judge_finish("long", [assets[0]], lambda k: "x", [entry], "T", tmp_path)["judge_status"] \
        == "disabled"


def test_contact_sheet_flags_manifest_entries(tmp_path):
    _install([(200, {}, json.dumps({"choices": [{"message": {"content": json.dumps(
        {"bad_shots": [{"n": 2, "reason": "riot police"}], "overall": "needs_review"})}}]}))])
    shot_assets, manifest = [], []
    for i in range(3):
        p = tmp_path / f"img{i}.png"
        Image.new("RGB", (64, 36), (i * 80, 40, 40)).save(p)
        a = {"kind": "image", "provider": "pixabay", "path": str(p)}
        shot_assets.append(a)
        manifest.append({"file": p.name, "review_required": False})
    summary = production._judge_finish("long", shot_assets, lambda k: f"line {k}", manifest, "T", tmp_path)
    assert manifest[1]["review_required"] and "riot police" in manifest[1]["review_reason"]
    assert not manifest[0]["review_required"]
    assert summary["needs_review"] and summary["contact_sheet"]["bad_shots"][0]["n"] == 2
    assert not list(tmp_path.glob("contact_sheet_*"))
