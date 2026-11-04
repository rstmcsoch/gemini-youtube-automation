"""HLS streaming copy (3 Nov 2026): ladder never upscales, aligned keyframes, fMP4 HLS settings,
upload order, backfill pairing and artifact choice."""
import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("dypol_r2", ROOT / "scripts" / "dypol_r2.py")
r2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r2)
import sys  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
_spec2 = importlib.util.spec_from_file_location("dypol_backfill_hls", ROOT / "scripts" / "dypol_backfill_hls.py")
bf = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(bf)


def dims(kind, w, h):
    return [(tw, th) for _, tw, th, _ in r2.hls_ladder(kind, {"w": w, "h": h})]


def test_ladder_short_and_long_never_above_source():
    assert dims("short", 1080, 1920) == [(360, 640), (540, 960), (720, 1280)]
    assert dims("short", 720, 1280) == [(360, 640), (540, 960), (720, 1280)]
    assert dims("long", 1920, 1080) == [(640, 360), (960, 540), (1280, 720)]
    assert dims("long", 1280, 720) == [(640, 360), (960, 540), (1280, 720)]
    # never a 1080p rendition
    assert (1920, 1080) not in dims("long", 1600, 900)
    assert max(h for _, h in dims("long", 1600, 900)) <= 900


def test_long_hls_fits_size_budget():
    # an 11-minute 1080p long video: worst case (every rendition at its cap) stays <= ~100 MB
    lad = r2.hls_ladder("long", {"w": 1920, "h": 1080, "duration": 658})
    worst_mb = (sum(k for *_, k in lad) + r2.HLS_AUDIO_KBPS * len(lad)) * 658 / 8000
    assert worst_mb <= 100, worst_mb
    # a short long video keeps the full caps
    assert [k for *_, k in r2.hls_ladder("long", {"w": 1280, "h": 720, "duration": 60})] == [300, 550, 900]


def test_hls_command_settings():
    cmd, ladder = r2.hls_command("in.mp4", "out", "short", {"w": 1080, "h": 1920, "fps": 30})
    c = " ".join(cmd)
    for part in ("-f hls", "-hls_segment_type fmp4", "-hls_time 2", "-hls_playlist_type vod",
                 "-sc_threshold 0", "expr:gte(t,n_forced*2)", "-g 60", "-preset veryfast", "-crf 25",
                 "independent_segments", "-master_pl_name master.m3u8", "v:0,a:0 v:1,a:1 v:2,a:2"):
        assert part in c, part
    assert len(ladder) == 3 and "v:3" not in c
    cmd, _ = r2.hls_command("in.mp4", "out", "long", {"w": 1280, "h": 720, "fps": 30}, has_audio=False)
    c = " ".join(cmd)
    assert "-hls_time 4" in c and "v:0 v:1 v:2" in c and "0:a:0" not in c


def test_playlists_upload_last(tmp_path):
    for rel in ("master.m3u8", "poster.jpg", "v0/index.m3u8", "v0/init_0.mp4", "v0/seg_00000.m4s"):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")
    order = [rel for _, rel in r2.hls_files(tmp_path)]
    assert order[-1] == "master.m3u8" and order[-2] == "v0/index.m3u8"


def test_long_hls_is_async_by_default(monkeypatch):
    monkeypatch.delenv("DYPOL_HLS_LONG", raising=False)
    assert r2.hls_kinds_default() == {"short"}
    monkeypatch.setenv("DYPOL_HLS_LONG", "inline")
    assert r2.hls_kinds_default() == {"short", "long"}


def test_backfill_pairs_and_artifact_choice():
    pairs, bad = bf.parse_pairs("job-a:41646837309, job-b:123 # note\n bogus")
    assert pairs == [("job-a", "41646837309")] and "bogus" in bad and "job-b:123" in bad
    arts = [{"name": "preview-job-a", "expired": False, "size_in_bytes": 9e6},
            {"name": "dypol-candidate-job-a", "expired": False, "size_in_bytes": 9e7},
            {"name": "dypol-result-1", "expired": False, "size_in_bytes": 300}]
    assert bf.pick_artifact(arts, "job-a")[1] == "full"
    arts[1]["expired"] = True
    assert bf.pick_artifact(arts, "job-a") == (arts[0], "preview")
    arts[0]["expired"] = True
    assert bf.pick_artifact(arts, "job-a") == (None, None)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_real_hls_package(tmp_path):
    src = tmp_path / "in.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=720x1280:r=30:d=5",
                    "-f", "lavfi", "-i", "sine=f=440:d=5", "-shortest", "-c:v", "libx264", "-preset", "ultrafast",
                    "-c:a", "aac", str(src)], check=True)
    entry = r2.make_hls(src, tmp_path / "hls", "short", r2.probe(src))
    master = (tmp_path / "hls" / "master.m3u8").read_text()
    assert master.index("360x640") < master.index("720x1280") and "1080x1920" not in master
    assert entry["segment"] == 2 and entry["segments"] >= 6 and entry["poster"] == "poster.jpg"
    media = (tmp_path / "hls" / "v0" / "index.m3u8").read_text()
    assert "#EXT-X-MAP:URI=" in media and "seg_00000.m4s" in media and "#EXT-X-ENDLIST" in media
