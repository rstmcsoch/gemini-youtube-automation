#!/usr/bin/env python3
"""Backfill R2 + HLS streams for drafts that only exist as GitHub artifacts.

For each "<job_id>:<run_id>" pair: download that run's draft artifact (the full
render; the small preview-<job_id> artifact only when the full one is gone),
then run the normal R2 step (scripts/dypol_r2.py build + upload): original,
HLS stream (360p/540p/720p), thumbnail strip,
captions, QC marks, manifest.json last.

Never publishes, approves or deletes anything, and never touches YouTube.
A draft whose artifacts have expired is reported as "preview unavailable"
(the console then says so on the check screen) and the run carries on.

Env: GITHUB_TOKEN, GITHUB_REPOSITORY, R2_* (as for dypol_r2.py).
"""
import argparse
import io
import json
import os
import re
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dypol_r2  # noqa: E402

PAIR = re.compile(r"^([A-Za-z0-9._-]{1,100}):(\d{5,20})$")
# Files the R2 step reads; everything else in a draft artifact (slides, work dirs) is skipped.
WANT = re.compile(r"(\.mp4|\.srt|qc\.json|storyboard\.json|thumb[^/]*\.(png|jpe?g)|thumbnails_[^/]+/.+\.(png|jpe?g))$", re.I)
SKIP = re.compile(r"(/work_|/anim_|/shot_|/clip_|broll_|merged_video|slides_short_[^/]+/[^/]+\.mp4$)", re.I)


def gh(path, token, raw=False):
    req = urllib.request.Request(f"https://api.github.com{path}", headers={
        "Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "dypol-backfill-hls"})
    if raw:
        # Artifact zips redirect to signed blob storage, which rejects a forwarded
        # Authorization header (401): resolve the redirect, then fetch it bare.
        try:
            with urllib.request.build_opener(_NoRedirect).open(req, timeout=300) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            loc = e.headers.get("Location") if e.code in (301, 302, 303, 307, 308) else None
            if not loc:
                raise
        with urllib.request.urlopen(urllib.request.Request(loc, headers={"User-Agent": "dypol-backfill-hls"}),
                                    timeout=600) as r:
            return r.read()
    with urllib.request.urlopen(req, timeout=300) as r:
        data = r.read()
    return json.loads(data)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def pick_artifact(artifacts, job_id):
    live = [a for a in artifacts if not a.get("expired") and a.get("size_in_bytes", 0) > 2048]
    full = [a for a in live if job_id in a["name"] and re.match(r"^(dypol-candidate|draft|candidate)-", a["name"])]
    prev = [a for a in live if a["name"] == f"preview-{job_id}"]
    if full:
        return full[0], "full"
    if prev:
        return prev[0], "preview"
    return None, None


def preview_layout(root, engine):
    """A preview-only artifact (preview_long.mp4 / preview_short.mp4): rename so dypol_r2.discover finds it."""
    root = Path(root)
    for f in root.glob("preview_*.mp4"):
        kind = "long" if "long" in f.name else "short"
        if engine == "gemini":
            f.rename(root / f"{kind}_video_preview_copy.mp4")
        else:
            (root / "media").mkdir(exist_ok=True)
            f.rename(root / "media" / "verticals_preview_copy.mp4")


def backfill_one(engine, repo, token, job_id, run_id, work, dry_run=False):
    arts = gh(f"/repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100", token).get("artifacts", [])
    art, which = pick_artifact(arts, job_id)
    if not art:
        print(f"::warning::{job_id}: preview unavailable — the run's artifacts have expired or were never saved")
        return {"job_id": job_id, "run_id": run_id, "status": "preview-unavailable"}
    root = Path(work) / job_id / "src"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    blob = gh(f"/repos/{repo}/actions/artifacts/{art['id']}/zip", token, raw=True)
    n = 0
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for name in z.namelist():
            if name.endswith("/") or not WANT.search(name) or SKIP.search("/" + name):
                continue
            z.extract(name, root)
            n += 1
    del blob
    if which == "preview":
        preview_layout(root, engine)
        print(f"::notice::{job_id}: full draft artifact expired — streaming the small preview copy instead")
    work_r2 = Path(work) / job_id / "r2"
    # Already in R2 (newer drafts): only add the missing HLS streams and merge them into its manifest.
    prefix, existing = (None, None) if dry_run else dypol_r2.read_manifest(job_id)
    if existing and existing.get("files"):
        need = [k for k in ("long", "short") if k in existing["files"] and k not in (existing.get("hls") or {})]
        if not need:
            shutil.rmtree(root, ignore_errors=True)
            print(f"{job_id}: already streams in small parts — nothing to do")
            return {"job_id": job_id, "run_id": run_id, "status": "already-hls"}
        part, uploads, mpath = dypol_r2.build(engine, job_id, root, work_r2, run_id, hls_kinds=need, hls_only=True)
        if not part.get("hls"):
            return {"job_id": job_id, "run_id": run_id, "status": "no-video", "artifact": art["name"]}
        existing["hls"] = {**(existing.get("hls") or {}), **part["hls"]}
        existing["hls_pending"] = [k for k in existing.get("hls_pending") or [] if k not in part["hls"]]
        mpath.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        dypol_r2.upload(uploads, mpath, job_id, manifest_prefix=prefix)
        shutil.rmtree(root, ignore_errors=True)
        return {"job_id": job_id, "run_id": run_id, "status": "ok", "artifact": art["name"], "from": which, "mode": "hls-only",
                "hls": {k: {"renditions": [f"{r['w']}x{r['h']}" for r in v.get("renditions", [])],
                            "segments": v.get("segments"), "mb": round(v.get("bytes", 0) / 1e6, 1)} for k, v in part["hls"].items()}}
    manifest, uploads, mpath = dypol_r2.build(engine, job_id, root, work_r2, run_id, hls_kinds={"long", "short"})
    if which == "preview":
        manifest["source_note"] = "backfilled from the small preview artifact (full draft expired)"
        mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not manifest.get("files"):
        print(f"::warning::{job_id}: no video found in {art['name']}")
        return {"job_id": job_id, "run_id": run_id, "status": "no-video", "artifact": art["name"]}
    if not dry_run:
        dypol_r2.upload(uploads, mpath, job_id)
    shutil.rmtree(root, ignore_errors=True)
    return {"job_id": job_id, "run_id": run_id, "status": "ok", "artifact": art["name"], "from": which,
            "files": n, "hls": {k: {"renditions": [f"{r['w']}x{r['h']}" for r in v.get("renditions", [])],
                                    "segments": v.get("segments"), "mb": round(v.get("bytes", 0) / 1e6, 1)}
                                for k, v in manifest.get("hls", {}).items()}}


def parse_pairs(text):
    pairs, bad = [], []
    for tok in re.split(r"[\s,]+", str(text or "").strip()):
        if not tok or tok.startswith("#"):
            continue
        m = PAIR.match(tok)
        (pairs.append((m.group(1), m.group(2))) if m else bad.append(tok))
    return pairs, bad


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["gemini", "verticals"], required=True)
    ap.add_argument("--jobs", default="", help='"job_id:run_id" pairs, space/comma separated')
    ap.add_argument("--jobs-file", default="")
    ap.add_argument("--work", default=os.path.join(tempfile.gettempdir(), "dypol-backfill"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    text = a.jobs
    if a.jobs_file and Path(a.jobs_file).exists():
        text += "\n" + "\n".join(l.split("#")[0] for l in Path(a.jobs_file).read_text().splitlines())
    pairs, bad = parse_pairs(text)
    for b in bad:
        print(f"::warning::ignored '{b}' (expected job_id:run_id)")
    if not pairs:
        print("::error::no job_id:run_id pairs given")
        return 2
    token, repo = os.environ.get("GITHUB_TOKEN", ""), os.environ.get("GITHUB_REPOSITORY", "")
    if not a.dry_run and not all(os.environ.get(k) for k in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")):
        print("::error::R2 secrets missing")
        return 2
    results = []
    for job_id, run_id in pairs[:30]:
        try:
            results.append(backfill_one(a.engine, repo, token, job_id, run_id, a.work, a.dry_run))
        except Exception as exc:  # noqa: BLE001 - one bad draft never stops the rest
            print(f"::warning::{job_id}: backfill failed: {type(exc).__name__}: {str(exc)[:200]}")
            results.append({"job_id": job_id, "run_id": run_id, "status": "failed", "error": str(exc)[:200]})
    print(json.dumps(results, indent=2))
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write("## HLS backfill\n\n| Job | Result | Streams |\n|---|---|---|\n")
            for r in results:
                streams = "; ".join(f"{k}: {', '.join(v['renditions'])} ({v['mb']} MB)" for k, v in (r.get("hls") or {}).items())
                fh.write(f"| {r['job_id']} | {r['status']}{' (' + r.get('from', '') + ')' if r.get('from') else ''} | {streams} |\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
