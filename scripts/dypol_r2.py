#!/usr/bin/env python3
"""DYPOL draft storage on Cloudflare R2 (shared by Engine A and Engine B).

After a draft renders this script
  1. checks the ORIGINAL render's resolution (long 16:9 between 1280x720 and
     1920x1080, Short 9:16 between 720x1280 and 1080x1920). Nothing is ever
     upscaled: the original is stored untouched.
  2. builds a compressed review copy per video (H.264 CRF 28, -preset veryfast,
     +faststart, a keyframe every 1 s so seeking is instant; long ~540p,
     Short ~720x1280; never larger than the source),
  3. builds a scrubber thumbnail sprite per video (one JPEG grid + its geometry),
  4. converts captions to WebVTT, reads chapters (Engine A storyboard) and QC
     warning times (qc.json silences / black / freezes),
  5. uploads everything to R2 under drafts/<job_id>/ with the S3 API (boto3),
     manifest.json last, so the console only sees complete drafts.

Modes:
  --check   only the resolution check; exit 3 when out of range and
            DYPOL_RESOLUTION_POLICY=fail (default), otherwise warn.
  (default) build + upload; any failure prints a ::warning:: and exits 0 so the
            draft falls back to the GitHub artifacts.
  --dry-run build everything into --work but skip the upload.

Env: R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET.
"""
import argparse
import glob
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

LIMITS = {  # (min_w, min_h, max_w, max_h)
    "long": (1280, 720, 1920, 1080),
    "short": (720, 1280, 1080, 1920),
}
PREVIEW_BOX = {"long": (960, 540), "short": (720, 1280)}
SPRITE_COLS, SPRITE_MAX = 10, 100
TYPES = {".mp4": "video/mp4", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
         ".json": "application/json", ".vtt": "text/vtt", ".srt": "application/x-subrip"}


def log(msg):
    print(f"☁️ R2: {msg}", flush=True)


def gh_annot(level, msg):
    print(f"::{level}::{msg}", flush=True)


def newest(patterns, exclude=None):
    files = []
    for p in patterns:
        files += [f for f in glob.glob(p, recursive=True) if os.path.isfile(f)]
    if exclude:
        files = [f for f in files if not re.search(exclude, os.path.basename(f))]
    files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
    return files[0] if files else None


def discover(engine, root):
    r = str(root).rstrip("/")
    if engine == "gemini":
        return {
            "long": newest([f"{r}/long_video_*.mp4"]),
            "short": newest([f"{r}/short_video_*.mp4"]),
            "thumb_long": newest([f"{r}/thumbnails_long_*/**/*.png", f"{r}/thumbnails_long_*/**/*.jpg"], exclude=r"raw"),
            "thumb_short": newest([f"{r}/thumbnails_short_*/**/*.png", f"{r}/thumbnails_short_*/**/*.jpg"], exclude=r"raw"),
            "qc": newest([f"{r}/qc.json"]),
            "captions_long": newest([f"{r}/slides_long_*/captions_long.srt", f"{r}/slides_long_*/*.srt"]),
            "captions_short": newest([f"{r}/slides_short_*/*.srt"]),
            "storyboard": newest([f"{r}/slides_long_*/storyboard.json"]),
        }
    # verticals: one Short per job
    return {
        "long": None,
        "short": newest([f"{r}/**/verticals_*.mp4"], exclude=r"(merged|preview)") or newest([f"{r}/**/*.mp4"], exclude=r"(merged_video|preview|shot_|clip_)"),
        "thumb_long": None,
        "thumb_short": newest([f"{r}/**/thumb_*.png", f"{r}/**/thumb_*.jpg", f"{r}/**/*thumb*.png"], exclude=r"raw"),
        "qc": newest([f"{r}/**/*_qc.json", f"{r}/**/qc.json"]),
        "captions_long": None,
        "captions_short": newest([f"{r}/**/verticals_*.srt", f"{r}/**/*.srt"]),
        "storyboard": None,
    }


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate:format=duration", "-of", "json", str(path)],
                         capture_output=True, text=True, timeout=120)
    data = json.loads(out.stdout or "{}")
    s = (data.get("streams") or [{}])[0]
    num, _, den = str(s.get("r_frame_rate", "30/1")).partition("/")
    try:
        fps = float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        fps = 30.0
    return {"w": int(s.get("width") or 0), "h": int(s.get("height") or 0),
            "duration": round(float((data.get("format") or {}).get("duration") or 0), 3),
            "fps": round(fps or 30.0, 3)}


def check_resolution(kind, info):
    """(ok, message). Long must be landscape 1280x720..1920x1080, Short portrait 720x1280..1080x1920."""
    min_w, min_h, max_w, max_h = LIMITS[kind]
    w, h = info.get("w", 0), info.get("h", 0)
    label = "long video" if kind == "long" else "Short"
    if not w or not h:
        return False, f"{label}: could not read the resolution"
    if kind == "long" and w < h or kind == "short" and h < w:
        return False, f"{label}: wrong orientation {w}x{h}"
    if w > max_w or h > max_h:
        return False, f"{label}: {w}x{h} is above the {max_w}x{max_h} maximum"
    if w < min_w or h < min_h:
        return False, f"{label}: {w}x{h} is below the {min_w}x{min_h} minimum"
    return True, f"{label}: {w}x{h} ok"


def preview_filter(kind, info):
    """Scale into the preview box, never above the source size; even dimensions."""
    bw, bh = PREVIEW_BOX[kind]
    w, h = info["w"] or bw, info["h"] or bh
    scale = min(1.0, bw / w, bh / h)
    tw, th = max(2, int(w * scale) // 2 * 2), max(2, int(h * scale) // 2 * 2)
    return f"scale={tw}:{th}:flags=bicubic", tw, th


def make_preview(src, dst, kind, info):
    vf, tw, th = preview_filter(kind, info)
    gop = max(1, int(round(info.get("fps") or 30)))
    cmd = ["ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
           "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-profile:v", "main",
           "-pix_fmt", "yuv420p", "-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "0",
           "-force_key_frames", "expr:gte(t,n_forced*1)",
           "-c:a", "aac", "-b:a", "96k", "-ac", "2", "-ar", "44100", "-movflags", "+faststart", str(dst)]
    subprocess.run(cmd, check=True, timeout=1800)
    return {"w": tw, "h": th}


def make_sprite(src, dst, kind, info):
    dur = max(0.1, float(info.get("duration") or 0))
    count = int(min(SPRITE_MAX, max(1, math.ceil(dur))))
    interval = dur / count
    tw = 160 if kind == "long" else 90
    th = int(round(tw * (info["h"] or 9) / (info["w"] or 16))) // 2 * 2
    cols = min(SPRITE_COLS, count)
    rows = math.ceil(count / cols)
    vf = f"fps=1/{interval:.5f},scale={tw}:{th},tile={cols}x{rows}"
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
                    "-vf", vf, "-frames:v", "1", "-q:v", "5", str(dst)], check=True, timeout=900)
    return {"interval": round(interval, 4), "count": count, "cols": cols, "rows": rows, "tw": tw, "th": th}


def srt_to_vtt(srt_text):
    body = srt_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    body = re.sub(r"(\d\d:\d\d:\d\d),(\d\d\d)", r"\1.\2", body)
    body = re.sub(r"^\d+\s*\n(?=\d\d:\d\d:\d\d\.\d\d\d)", "", body, flags=re.M)
    return "WEBVTT\n\n" + body + "\n"


def chapters_from_storyboard(path):
    try:
        sb = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []
    slides = (sb.get("timeline") or {}).get("slides") or []
    titles = sb.get("slides") or []
    out = []
    for k, sl in enumerate(slides):
        title = (titles[k].get("title") if k < len(titles) and isinstance(titles[k], dict) else "") or f"Part {k + 1}"
        t = round(float(sl.get("start") or 0), 2)
        if out and t - out[-1]["t"] < 1:
            continue
        out.append({"t": t, "title": str(title)[:80]})
    return out


def qc_markers(qc):
    """{'long': [...], 'short': [...]} of {t, end, kind, label} from either engine's qc.json."""
    def from_block(block):
        out = []
        for key, kind in (("silences", "silence"), ("black", "black"), ("freezes", "freeze")):
            for s in block.get(key) or []:
                try:
                    t, e = float(s.get("start", 0)), float(s.get("end", s.get("start", 0)))
                except (TypeError, ValueError):
                    continue
                out.append({"t": round(t, 2), "end": round(e, 2), "kind": kind,
                            "label": f"{e - t:.1f} s {kind} at {t:.1f} s"})
        return sorted(out, key=lambda m: m["t"])
    if not isinstance(qc, dict):
        return {}
    vids = qc.get("videos")
    if isinstance(vids, dict):
        return {k: from_block(v) for k, v in vids.items() if k in ("long", "short") and isinstance(v, dict)}
    return {"short": from_block(qc)}


def build(engine, job_id, root, work, run_id=""):
    found = discover(engine, root)
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    manifest = {"v": 1, "job_id": job_id, "engine": engine, "run_id": run_id,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "files": {}, "sprites": {}, "chapters": {}, "markers": {}, "resolution": {}}
    uploads = []  # (local path, key name)

    def add(name, local, **meta):
        key = f"drafts/{job_id}/{name}"
        manifest["files"][name.split(".")[0]] = {"key": key, "size": os.path.getsize(local), **meta}
        uploads.append((str(local), key))

    for kind in ("long", "short"):
        src = found.get(kind)
        if not src:
            continue
        info = probe(src)
        ok, msg = check_resolution(kind, info)
        manifest["resolution"][kind] = {"ok": ok, "message": msg, "w": info["w"], "h": info["h"]}
        add(f"{kind}.mp4", src, **info)
        try:
            pv = work / f"{kind}_preview.mp4"
            dims = make_preview(src, pv, kind, info)
            add(f"{kind}_preview.mp4", pv, **dims, duration=info["duration"])
        except Exception as exc:  # noqa: BLE001
            gh_annot("warning", f"R2 preview encode failed for {kind}: {str(exc)[:160]}")
        try:
            sp = work / f"{kind}_sprite.jpg"
            geo = make_sprite(src, sp, kind, info)
            add(f"{kind}_sprite.jpg", sp)
            manifest["sprites"][kind] = {"key": f"drafts/{job_id}/{kind}_sprite.jpg", **geo}
        except Exception as exc:  # noqa: BLE001
            gh_annot("warning", f"R2 thumbnail strip failed for {kind}: {str(exc)[:160]}")
        cap = found.get(f"captions_{kind}")
        if cap:
            vtt = work / f"{kind}_captions.vtt"
            vtt.write_text(srt_to_vtt(Path(cap).read_text(encoding="utf-8", errors="replace")), encoding="utf-8")
            add(f"{kind}_captions.vtt", vtt)
        th = found.get(f"thumb_{kind}")
        if th:
            add(f"{kind}_thumb{Path(th).suffix.lower()}", th)
    if found.get("storyboard"):
        manifest["chapters"]["long"] = chapters_from_storyboard(found["storyboard"])
    if found.get("qc"):
        add("qc.json", found["qc"])
        try:
            manifest["markers"] = qc_markers(json.loads(Path(found["qc"]).read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            pass
    mpath = work / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest, uploads, mpath


def upload(uploads, mpath, job_id):
    import boto3
    from boto3.s3.transfer import TransferConfig
    from botocore.config import Config
    acct = os.environ["R2_ACCOUNT_ID"].strip()
    s3 = boto3.client("s3", endpoint_url=f"https://{acct}.r2.cloudflarestorage.com",
                      aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"].strip(),
                      aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"].strip(),
                      region_name="auto", config=Config(retries={"max_attempts": 5, "mode": "standard"},
                                                        signature_version="s3v4"))
    bucket = os.environ.get("R2_BUCKET", "dypol-media").strip() or "dypol-media"
    cfg = TransferConfig(multipart_threshold=64 * 1024 * 1024, multipart_chunksize=16 * 1024 * 1024, max_concurrency=4)
    total = 0
    for local, key in uploads + [(str(mpath), f"drafts/{job_id}/manifest.json")]:
        ext = Path(key).suffix.lower()
        s3.upload_file(local, bucket, key, Config=cfg, ExtraArgs={
            "ContentType": TYPES.get(ext, "application/octet-stream"),
            "CacheControl": "private, max-age=31536000, immutable" if ext != ".json" else "private, no-cache"})
        size = os.path.getsize(local)
        total += size
        log(f"uploaded {key} ({size / 1e6:.1f} MB)")
    head = s3.head_object(Bucket=bucket, Key=f"drafts/{job_id}/manifest.json")
    log(f"done: {len(uploads) + 1} objects, {total / 1e6:.1f} MB (manifest {head['ContentLength']} B)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["gemini", "verticals"], required=True)
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--work", default="r2-work")
    ap.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID", ""))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,100}", a.job_id):
        gh_annot("error", "invalid job id")
        return 2
    if a.check:
        policy = os.environ.get("DYPOL_RESOLUTION_POLICY", "fail").strip().lower()
        found = discover(a.engine, a.root)
        bad = []
        for kind in ("long", "short"):
            if found.get(kind):
                ok, msg = check_resolution(kind, probe(found[kind]))
                log(f"resolution {msg}")
                if not ok:
                    bad.append(msg)
        if not found.get("long") and not found.get("short"):
            bad.append("no rendered video found")
        for msg in bad:
            gh_annot("error" if policy == "fail" else "warning", f"Resolution check: {msg} (allowed: long 1280x720-1920x1080, Short 720x1280-1080x1920; never upscaled)")
        return 3 if bad and policy == "fail" else 0
    try:
        manifest, uploads, mpath = build(a.engine, a.job_id, a.root, a.work, a.run_id)
        if not uploads:
            gh_annot("warning", "R2: nothing to upload; the console will use the GitHub artifacts")
            return 0
        if a.dry_run:
            log(json.dumps(manifest, indent=2))
            return 0
        missing = [k for k in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY") if not os.environ.get(k)]
        if missing:
            gh_annot("warning", f"R2 not configured ({', '.join(missing)} missing); the console will use the GitHub artifacts")
            return 0
        upload(uploads, mpath, a.job_id)
    except Exception as exc:  # noqa: BLE001 - never fail the draft; artifacts are the fallback
        gh_annot("warning", f"R2 upload failed, the console will use the GitHub artifacts: {type(exc).__name__}: {str(exc)[:200]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
