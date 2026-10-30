#!/usr/bin/env python3
"""Calibrate the AI visual judge on a labeled golden set.

Folder layout:
    GOLDEN/accepted/*.png|jpg   shots a human accepted
    GOLDEN/rejected/*.png|jpg   shots a human rejected (kitten, riot police, ...)
    GOLDEN/labels.json          optional: {"accepted/x.png": {"video_title", "query", "narration"}}

Prints the catch-rate on rejects (judge did NOT return PASS) and the pass-rate on
accepts (judge returned PASS) against the plan's targets (>=90 % / >=85 %), plus
a lenient view that counts MARGINAL as "not auto-accepted" / "usable".

Needs NVIDIA_API_KEY. Uses an in-memory cache by default so every image is
really judged (pass --use-cache to reuse VISUAL_JUDGE_CACHE_PATH). Never writes
into the repo. Exit code: 0, or 1 with --strict when a target is missed,
2 when the judge is disabled.

    NVIDIA_API_KEY=nvapi-... python scripts/calibrate_judge.py /path/to/judge-golden
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:  # Engine A layout
    from src import visual_judge as vj  # type: ignore
except ImportError:  # Engine B layout
    from verticals import visual_judge as vj  # type: ignore

TARGET_CATCH, TARGET_PASS = 0.90, 0.85
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp"}


def _context(rel, labels, args):
    # Never derive context from the file name: golden-set names describe the label.
    info = labels.get(rel) or {}
    title = info.get("video_title") or args.title
    query = info.get("query") or ""
    narration = info.get("narration") or args.narration or (f"Visual for: {query}" if query else title)
    return title, narration, query or title


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("golden", type=Path)
    p.add_argument("--title", default="Tech news explained")
    p.add_argument("--narration", default="")
    p.add_argument("--limit", type=int, default=0, help="max images per class (0 = all)")
    p.add_argument("--use-cache", action="store_true")
    p.add_argument("--json", type=Path, help="write per-image verdicts here (outside the repo)")
    p.add_argument("--strict", action="store_true")
    args = p.parse_args(argv)

    store = None if args.use_cache else vj.JudgeStore(vj.MemoryBackend())
    judge = vj.VisualJudge(store=store)
    if not judge.enabled:
        print(f"visual judge disabled ({judge.disabled_reason}); set NVIDIA_API_KEY.")
        return 2
    vj.MAX_CALLS_PER_VIDEO = 10 ** 6          # calibration is not a video: lift per-video caps
    vj.MAX_WALL_SECONDS_PER_VIDEO = 10 ** 6
    labels = {}
    if (args.golden / "labels.json").exists():
        labels = json.loads((args.golden / "labels.json").read_text(encoding="utf-8"))

    results = {"accepted": [], "rejected": []}
    for cls in ("accepted", "rejected"):
        files = sorted(f for f in (args.golden / cls).glob("*") if f.suffix.lower() in IMAGE_EXT)
        if args.limit:
            files = files[:args.limit]
        for f in files:
            rel = f"{cls}/{f.name}"
            title, narration, query = _context(rel, labels, args)
            judge.begin_video(rel)
            v = judge.judge_image(f, title, narration, query, asset_key="", record=False)
            decision = v["decision"] if v else "ERROR"
            results[cls].append({"file": rel, "decision": decision, "score": v and v.get("score"),
                                 "issues": v and v.get("issues"), "subject_match": v and v.get("subject_match"),
                                 "reason": v and v.get("reason"), "model": v and v.get("judge_model")})
            print(f"{cls[:3]} {decision:8} {str(v and v.get('score')):>4} "
                  f"{','.join((v or {}).get('issues') or []):30.30} {f.name}", flush=True)
            if judge.circuit_reason:
                print(f"circuit open: {judge.circuit_reason}; stopping.")
                break
        if judge.circuit_reason:
            break

    def rate(rows, ok):
        judged = [r for r in rows if r["decision"] != "ERROR"]
        return (sum(1 for r in judged if ok(r["decision"])) / len(judged)) if judged else 0.0, len(judged)

    catch, n_rej = rate(results["rejected"], lambda d: d != vj.PASS)
    catch_strict, _ = rate(results["rejected"], lambda d: d == vj.FAIL)
    pass_rate, n_acc = rate(results["accepted"], lambda d: d == vj.PASS)
    pass_lenient, _ = rate(results["accepted"], lambda d: d != vj.FAIL)
    print()
    print(f"prompt_version {vj.PROMPT_VERSION}; models {', '.join(judge.models)}; used {judge.summary()['models_used']}")
    print(f"rejects caught (not PASS): {catch:6.1%} of {n_rej}  target >= {TARGET_CATCH:.0%}  "
          f"{'OK' if catch >= TARGET_CATCH else 'MISS'}   (FAIL only: {catch_strict:.1%})")
    print(f"accepts passed (PASS):     {pass_rate:6.1%} of {n_acc}  target >= {TARGET_PASS:.0%}  "
          f"{'OK' if pass_rate >= TARGET_PASS else 'MISS'}   (PASS or MARGINAL: {pass_lenient:.1%})")
    errors = sum(1 for rows in results.values() for r in rows if r["decision"] == "ERROR")
    if errors:
        print(f"{errors} image(s) could not be judged (HTTP/parse errors).")
    misses = [r for r in results["rejected"] if r["decision"] == vj.PASS] + \
             [r for r in results["accepted"] if r["decision"] not in (vj.PASS, "ERROR")]
    if misses:
        print("\nDisagreements:")
        for r in misses:
            print(f"  {r['file']}: {r['decision']} score={r['score']} issues={r['issues']} - {r['reason']}")
    if args.json:
        args.json.write_text(json.dumps({"results": results, "catch_rate": catch, "pass_rate": pass_rate},
                                        indent=2), encoding="utf-8")
    if args.strict and (catch < TARGET_CATCH or pass_rate < TARGET_PASS):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
