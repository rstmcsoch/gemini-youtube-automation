"""scripts/report_failure.py: failed-run result record for the console."""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import report_failure as rf  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def test_category_line_wins_over_heuristics():
    lines = ["uploading to YouTube", "DYPOL_FAILURE_CATEGORY=llm_unavailable", "Pipeline failed."]
    assert rf.classify(lines) == "llm_unavailable"


def test_heuristics_when_no_category_line():
    assert rf.classify(["NO-PERSONAL-NAMES GATE: BLOCKED"]) == "content_gate"
    assert rf.classify(["googleapiclient.errors.HttpError 403 quotaExceeded"]) == "publish"
    assert rf.classify(["RuntimeError: ffmpeg encode failed with exit code 1"]) == "render"
    assert rf.classify(["nothing useful"]) == "unknown"


def test_secrets_are_redacted_from_excerpt():
    env = {"GOOGLE_API_KEY": "supersecretvalue123", "PATH": "/usr/bin"}
    text = rf.excerpt(["key=supersecretvalue123", "AIzaSyA1234567890abcdefghijklmnop", "ok"], env)
    assert "supersecretvalue123" not in text and "AIzaSy" not in text and "/usr/bin" not in text
    assert "ok" in text


def test_partial_upload_videos_are_kept(tmp_path):
    partial = {"job_id": "job-1", "status": "partial",
               "videos": [{"kind": "long", "youtube_id": "abc", "youtube_url": "https://youtu.be/abc"}]}
    record = rf.build_record("job-1", "42", "publish", ["upload failed for short"], partial, env={})
    assert record["status"] == "partial" and record["video_count"] == 1 and record["youtube_id"] == "abc"
    assert record["failure_category"] == "publish"


def test_cli_writes_failed_record(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("line\n" * 100 + "DYPOL_FAILURE_CATEGORY=llm_unavailable\nPipeline failed.\n")
    out = tmp_path / "dypol-results" / "42.json"
    result = subprocess.run([sys.executable, str(REPO / "scripts/report_failure.py"), "--job-id", "job-9",
                             "--run-id", "42", "--phase", "publish", "--log", str(log),
                             "--partial", str(tmp_path / "missing.json"), "--out", str(out)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    record = json.loads(out.read_text())
    assert record["status"] == "failed" and record["videos"] == [] and record["job_id"] == "job-9"
    assert record["failure_category"] == "llm_unavailable"
    assert len(record["log_excerpt"].splitlines()) <= rf.EXCERPT_LINES
