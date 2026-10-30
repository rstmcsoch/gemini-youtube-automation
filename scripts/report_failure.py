"""Write dypol-results/<run_id>.json for a FAILED run (used by an `if: failure()` step).

The record keeps the console's existing result shape (job_id, run_id, status,
videos[]) and adds failure_category + log_excerpt, so the console can show why a
publish failed without fetching the job log. Any video that was uploaded before
the failure (from a partial result file) is kept in videos[] so the console never
re-uploads it.

Category: the last ``DYPOL_FAILURE_CATEGORY=<category>`` line in the logs, else a
keyword heuristic, else ``unknown``. Secret values from the environment are
redacted from the excerpt before it is written (the file is committed to the repo).
"""
import argparse
import datetime
import json
import os
import re
import sys
from pathlib import Path

CATEGORIES = ("llm_unavailable", "content_gate", "asset", "render", "publish", "unknown")
CATEGORY_LINE = re.compile(r"DYPOL_FAILURE_CATEGORY=([a-z_]+)")
SECRET_ENV_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIALS", "CLIENT_SECRETS")
EXCERPT_LINES = 40
EXCERPT_MAX_CHARS = 6000


def _read_logs(paths):
    lines = []
    for path in paths:
        try:
            lines.extend(Path(path).read_text(encoding="utf-8", errors="replace").splitlines())
        except OSError:
            continue
    return lines


def classify(lines, default="unknown"):
    for line in reversed(lines):
        match = CATEGORY_LINE.search(line)
        if match and match.group(1) in CATEGORIES:
            return match.group(1)
    text = "\n".join(lines[-200:]).lower()
    if "personal-names gate: blocked" in text or "personal-name text detected" in text:
        return "content_gate"
    if any(t in text for t in ("503 unavailable", "server disconnected", "resource_exhausted",
                               "gemini unavailable", "llmunavailableerror")):
        return "llm_unavailable"
    if any(t in text for t in ("ffmpeg encode failed", "ffmpeg chunk encode failed", "uncovered frames")):
        return "render"
    if any(t in text for t in ("youtube", "upload", "oauth", "quotaexceeded", "httperror")):
        return "publish"
    if any(t in text for t in ("pixabay", "pexels", "flux", "tts", "kokoro", "b-roll")):
        return "asset"
    return default if default in CATEGORIES else "unknown"


def _secret_values(env):
    values = []
    for name, value in env.items():
        if value and len(value) >= 8 and any(hint in name.upper() for hint in SECRET_ENV_HINTS):
            values.append(value)
    return sorted(set(values), key=len, reverse=True)


def redact(text, env=None):
    for value in _secret_values(os.environ if env is None else env):
        text = text.replace(value, "***")
    # Generic token shapes (Google API keys, bearer tokens, GitHub tokens).
    text = re.sub(r"AIza[0-9A-Za-z_\-]{20,}", "***", text)
    text = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{16,}", r"\1***", text)
    text = re.sub(r"gh[pousr]_[A-Za-z0-9]{20,}", "***", text)
    return text


def excerpt(lines, env=None):
    tail = [line.rstrip() for line in lines if line.strip()][-EXCERPT_LINES:]
    text = redact("\n".join(tail), env)
    return text[-EXCERPT_MAX_CHARS:]


def build_record(job_id, run_id, phase, lines, partial=None, extra=None, env=None, default_category="unknown"):
    partial = partial if isinstance(partial, dict) else {}
    videos = [v for v in (partial.get("videos") or []) if isinstance(v, dict) and v.get("youtube_id")]
    record = dict(partial)
    record.update({
        "job_id": job_id or partial.get("job_id", ""),
        "run_id": str(run_id),
        "phase": phase,
        "status": "partial" if videos else "failed",
        "videos": videos,
        "video_count": len(videos),
        "youtube_id": videos[0]["youtube_id"] if videos else "",
        "youtube_url": videos[0].get("youtube_url", "") if videos else "",
        "failure_category": classify(lines, default_category),
        "log_excerpt": excerpt(lines, env),
        "recorded_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    })
    record.update(extra or {})
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", default="")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--phase", default="")
    parser.add_argument("--log", action="append", default=[], help="log file(s) captured with tee")
    parser.add_argument("--partial", default="", help="partial result JSON written before the failure")
    parser.add_argument("--default-category", default="unknown")
    parser.add_argument("--extra", action="append", default=[], help="key=value added to the record")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    partial = None
    if args.partial and Path(args.partial).is_file():
        try:
            partial = json.loads(Path(args.partial).read_text(encoding="utf-8"))
        except ValueError:
            partial = None
    extra = dict(item.split("=", 1) for item in args.extra if "=" in item)
    record = build_record(args.job_id, args.run_id, args.phase, _read_logs(args.log), partial, extra,
                          default_category=args.default_category)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Wrote failure record {out} (category={record['failure_category']}, status={record['status']})")
    print(f"DYPOL_FAILURE_CATEGORY={record['failure_category']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
