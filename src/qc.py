"""Post-render QC for Engine A drafts: never blocks a draft, but hard failures mark it "needs fixes".

One ffmpeg pass per video runs ebur128 (loudness), silencedetect, blackdetect and
freezedetect (on a downscaled copy, so it is cheap); a small contact sheet shows
both videos at a glance. Results go to qc.json and contact.png next to the
candidate in the draft artifact.
"""
import json
import math
import re
import subprocess
from io import BytesIO
from pathlib import Path

TARGET_LUFS = -14.0
LUFS_TOLERANCE = 1.5
MAX_TRUE_PEAK = -1.0
SILENCE_DB, SILENCE_MIN_S = -45, 2.0
BLACK_MIN_S, BLACK_PIX_TH = 0.5, 0.04   # the DYPOL slate (#0F141C) must not count as black
FREEZE_NOISE_DB = -60
MAX_CONTACT_BYTES = 1_500_000
# Hard failures (owner rules): the draft is still saved, but its result says "needs_fixes".
HARD_STATIC_S = 3.5                      # nothing on screen may stay static longer than this
LONG_RES = ((1280, 720), (1920, 1080))   # 16:9, 720p-1080p, never upscaled
SHORT_RES = ((720, 1280), (1080, 1920))  # 9:16   # the preview artifact holds ~1-3 MB mp4s; keep the sheet well under that


def _log(msg):
    print(f"🔍 QC: {msg}", flush=True)


def _freeze_min_s():
    """freezedetect window: the style guide's max_static_s, but never longer than the 3.5 s hard limit."""
    try:
        from src import style
        return min(HARD_STATIC_S, float(style.long_rules()["pacing"]["max_static_s"]))
    except Exception:  # noqa: BLE001
        return HARD_STATIC_S


def _size(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height", "-of", "csv=p=0:s=x", str(path)],
                         capture_output=True, text=True, timeout=60).stdout.strip()
    try:
        w, h = (int(v) for v in out.split("x")[:2])
        return w, h
    except ValueError:
        return None, None


def resolution_problem(label, width, height):
    """'' when the render is 720p-1080p in the right orientation, else what is wrong."""
    if not width or not height:
        return ""
    (lo_w, lo_h), (hi_w, hi_h) = SHORT_RES if label == "short" else LONG_RES
    if not (lo_w <= width <= hi_w and lo_h <= height <= hi_h):
        return (f"{label}: resolution {width}x{height} is outside {lo_w}x{lo_h}-{hi_w}x{hi_h}")
    return ""


def hard_failures(report, relevance=None):
    """Owner-rule failures that make the draft "needs fixes": off-topic script, resolution outside
    720p-1080p, a static picture longer than 3.5 s, a true peak above -1 dBTP (4 Nov 2026)."""
    out = []
    if isinstance(relevance, dict) and relevance.get("verdict") == "off_topic":
        out.append("off-topic script: " + "; ".join(relevance.get("reasons") or [])[:300])
    for label, video in ((report or {}).get("videos") or {}).items():
        problem = resolution_problem(label, video.get("width"), video.get("height"))
        if problem:
            out.append(problem)
        tp = (video.get("loudness") or {}).get("true_peak_dbtp")
        if tp is not None and tp > MAX_TRUE_PEAK:
            out.append(f"{label}: true peak {tp:.1f} dBTP (limit {MAX_TRUE_PEAK:g} dBTP)")
        for f in video.get("freezes") or []:
            if (f.get("duration") or 0) > HARD_STATIC_S:
                out.append(f"{label}: picture static for {f['duration']:.1f} s at {f['start']:.1f} s "
                           f"(limit {HARD_STATIC_S:g} s)")
    return out


def _duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, timeout=60).stdout.strip()
    return float(out or 0.0)


def _pairs(starts, ends, total):
    out = []
    for k, a in enumerate(starts):
        b = ends[k] if k < len(ends) else total
        out.append({"start": round(a, 2), "end": round(b, 2), "duration": round(b - a, 2)})
    return out


def parse_log(err, total):
    """Parse ffmpeg stderr from the combined filter pass into a dict."""
    def nums(pattern):
        return [float(x) for x in re.findall(pattern, err)]
    loud = {}
    summary = err[err.rfind("Summary:"):] if "Summary:" in err else ""
    for key, pat in (("integrated_lufs", r"I:\s+(-?[\d.]+) LUFS"), ("lra_lu", r"LRA:\s+(-?[\d.]+) LU"),
                     ("true_peak_dbtp", r"Peak:\s+(-?[\d.]+|-inf) dBFS")):
        m = re.search(pat, summary)
        if m:
            loud[key] = float(m.group(1)) if m.group(1) != "-inf" else float("-inf")
    silences = _pairs(nums(r"silence_start: (-?[\d.]+)"), nums(r"silence_end: (-?[\d.]+)"), total)
    black = [{"start": round(a, 2), "end": round(b, 2), "duration": round(d, 2)} for a, b, d in
             re.findall(r"black_start:\s*([\d.]+)\s+black_end:\s*([\d.]+)\s+black_duration:\s*([\d.]+)", err)
             for a, b, d in [(float(a), float(b), float(d))]]
    freezes = _pairs(nums(r"freeze_start: ([\d.]+)"), nums(r"freeze_end: ([\d.]+)"), total)
    return {"loudness": loud, "silences": silences, "black": black, "freezes": freezes}


def warnings_for(label, result):
    w = []
    loud = result.get("loudness") or {}
    i = loud.get("integrated_lufs")
    if i is None:
        w.append(f"{label}: loudness not measured")
    elif abs(i - TARGET_LUFS) > LUFS_TOLERANCE:
        w.append(f"{label}: integrated loudness {i:.1f} LUFS (target {TARGET_LUFS:.0f} ±{LUFS_TOLERANCE})")
    tp = loud.get("true_peak_dbtp")
    if tp is not None and tp > MAX_TRUE_PEAK:
        w.append(f"{label}: true peak {tp:.1f} dBTP (> {MAX_TRUE_PEAK})")
    total = result.get("duration", 0.0)
    for s in result.get("silences", []):
        if s["start"] < total - 1.0:      # the padded tail is expected
            w.append(f"{label}: {s['duration']:.1f} s of silence at {s['start']:.1f} s")
    for b in result.get("black", []):
        w.append(f"{label}: {b['duration']:.1f} s black at {b['start']:.1f} s")
    for f in result.get("freezes", []):
        w.append(f"{label}: picture static for {f['duration']:.1f} s at {f['start']:.1f} s")
    return w


def analyze(video_path):
    """Loudness, silences, black and frozen stretches of one video (one decode)."""
    total = _duration(video_path)
    vf = (f"scale=320:-2,blackdetect=d={BLACK_MIN_S}:pix_th={BLACK_PIX_TH},"
          f"freezedetect=n={FREEZE_NOISE_DB}dB:d={_freeze_min_s():g}")
    af = f"ebur128=peak=true,silencedetect=n={SILENCE_DB}dB:d={SILENCE_MIN_S:g}"
    err = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(video_path), "-vf", vf, "-af", af,
                          "-f", "null", "-"], capture_output=True, text=True, timeout=1800).stderr
    result = parse_log(err, total)
    result["duration"] = round(total, 2)
    result["width"], result["height"] = _size(video_path)
    return result


def _grab(video_path, t, width):
    data = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", str(video_path), "-frames:v", "1",
                           "-vf", f"scale={width}:-2", "-f", "image2pipe", "-vcodec", "png", "-"],
                          capture_output=True, timeout=60).stdout
    from PIL import Image
    return Image.open(BytesIO(data)).convert("RGB")


def contact_sheet(long_path=None, short_path=None, out_path="contact.png", width=1280):
    """Long video as a 4x4 grid on top, the Short as one row of 8 below; palette PNG under MAX_CONTACT_BYTES."""
    from PIL import Image
    rows = []
    for attempt in range(2):           # at most one shrink retry
        rows = []
        if long_path and Path(long_path).exists():
            d, tw = _duration(long_path), width // 4
            tiles = [_grab(long_path, d * (k + 0.5) / 16, tw) for k in range(16)]
            th = tiles[0].height
            grid = Image.new("RGB", (tw * 4, th * 4), (15, 20, 28))
            for k, tile in enumerate(tiles):
                grid.paste(tile, ((k % 4) * tw, (k // 4) * th))
            rows.append(grid)
        if short_path and Path(short_path).exists():
            d, tw = _duration(short_path), width // 8
            tiles = [_grab(short_path, d * (k + 0.5) / 8, tw) for k in range(8)]
            strip = Image.new("RGB", (tw * 8, tiles[0].height), (15, 20, 28))
            for k, tile in enumerate(tiles):
                strip.paste(tile, (k * tw, 0))
            rows.append(strip)
        if not rows:
            return None
        sheet = Image.new("RGB", (max(r.width for r in rows), sum(r.height for r in rows) + 8 * (len(rows) - 1)),
                          (15, 20, 28))
        y = 0
        for r in rows:
            sheet.paste(r, (0, y))
            y += r.height + 8
        sheet.quantize(colors=256, method=Image.Quantize.MEDIANCUT).save(out_path, optimize=True)
        if Path(out_path).stat().st_size <= MAX_CONTACT_BYTES:
            break
        width = int(width * 0.7)
    return Path(out_path)


def _clean(value):
    """Strict JSON: -inf / nan (e.g. a silent track's peak) become null."""
    if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def run(long_path, short_path, out_dir, extra=None):
    """Write qc.json + contact.png into out_dir; returns {"warnings": [...], ...}. Never raises."""
    out_dir = Path(out_dir)
    report = {"warnings": [], "videos": {}, "qc_json": "qc.json", "contact": None}
    for label, path in (("long", long_path), ("short", short_path)):
        if not path or not Path(path).exists():
            continue
        try:
            result = analyze(path)
            report["videos"][label] = result
            report["warnings"] += warnings_for(label, result)
        except Exception as exc:  # noqa: BLE001
            report["warnings"].append(f"{label}: QC analysis failed ({str(exc)[:120]})")
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        sheet = contact_sheet(long_path, short_path, out_dir / "contact.png")
        report["contact"] = sheet.name if sheet else None
    except Exception as exc:  # noqa: BLE001
        report["warnings"].append(f"contact sheet failed ({str(exc)[:120]})")
    if extra:
        report.update(extra)
    report["hard_failures"] = hard_failures(report, relevance=(extra or {}).get("script_relevance"))
    report["verdict"] = "needs_fixes" if report["hard_failures"] else "ok"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "qc.json").write_text(json.dumps(_clean(report), indent=2, default=str), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        _log(f"could not write qc.json ({exc})")
    for w in report["warnings"]:
        _log(f"warning: {w}")
    for w in report["hard_failures"]:
        _log(f"NEEDS FIXES: {w}")
    if not report["warnings"]:
        _log("no warnings")
    return report
