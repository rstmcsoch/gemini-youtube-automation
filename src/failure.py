"""Map a pipeline exception to a DYPOL failure category for the console.

Categories (shared with Engine B and the console): llm_unavailable, content_gate,
asset, render, publish, unknown. main.py prints one machine-readable line,
``DYPOL_FAILURE_CATEGORY=<category>``, and writes output/dypol_failure.json.
"""
import json
import os
from pathlib import Path

CATEGORIES = ("llm_unavailable", "content_gate", "asset", "render", "publish", "unknown")


def _chain(exc):
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or exc.__context__


def failure_category(exc):
    chain = list(_chain(exc))
    for item in chain:
        category = getattr(item, "failure_category", None)
        if category in CATEGORIES:
            return category
    text = " ".join(f"{type(e).__name__} {e}" for e in chain).lower()
    if "personal-name" in text or "personal name" in text:
        return "content_gate"
    if "ffmpeg" in text or "encode failed" in text or "uncovered frames" in text:
        return "render"
    if "youtube" in text or "upload" in text or "httperror" in text or "oauth" in text:
        return "publish"
    if any(t in text for t in ("pixabay", "pexels", "flux", "wikimedia", "gtts", "tts", "stock")):
        return "asset"
    return "unknown"


def report_failure(exc, out_dir="output"):
    """Print the category line and persist a small JSON record; never raises."""
    category = failure_category(exc)
    print(f"DYPOL_FAILURE_CATEGORY={category}", flush=True)
    try:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "dypol_failure.json").write_text(json.dumps({
            "job_id": os.environ.get("DYPOL_JOB_ID", ""),
            "run_id": os.environ.get("GITHUB_RUN_ID", ""),
            "phase": os.environ.get("DYPOL_PHASE", ""),
            "failure_category": category,
            "error": f"{type(exc).__name__}: {str(exc)[:500]}",
        }, indent=2), encoding="utf-8")
    except OSError:
        pass
    return category
