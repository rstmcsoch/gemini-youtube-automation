"""Long video + Short production on top of the explainer renderer.

The visual gatherer is injectable so a local sample can be rendered without
network calls (tests and mock renders pass a fake gatherer).
"""
import json
from pathlib import Path

from src import explainer as ex
from src import storyboard as sb

LONG_SIZE = (1920, 1080)
SHORT_SIZE = (1080, 1920)
SLIDE_PAD = 0.45          # seconds of breathing room after each slide's narration
SHORT_MIN, SHORT_MAX = 20.0, 45.0

MANIFEST_FIELDS = ("provider", "source_url", "asset_id", "license", "license_url", "sha256")


def _log(message):
    print(f"🎬 {message}", flush=True)


def _cleanup(narration, audio, shot_assets):
    """Drop bulky intermediates; downloaded stock clips stay recorded (sha256) in the manifest."""
    for path in (narration, audio):
        Path(path).unlink(missing_ok=True)
    for asset in shot_assets:
        if asset and asset.get("kind") == "video" and asset.get("third_party"):
            Path(asset["path"]).unlink(missing_ok=True)


def manifest_entry(asset, video, slide, shots, timeline):
    entry = {
        "video": video, "segment": slide, "shots": shots,
        "kind": asset.get("kind", "image"),
        **{field: asset.get(field, "") for field in MANIFEST_FIELDS},
        "retrieved_at": asset.get("retrieved_at", ""),
        "file": Path(asset.get("path", "")).name,
        "used_seconds": round(sum(timeline["shots"][i]["end"] - timeline["shots"][i]["start"] for i in shots), 3),
        "review_required": False,
    }
    for optional in ("tags", "query", "matched_terms", "prompt", "duration", "width", "height",
                     "fallback_query", "borrowed"):
        if asset.get(optional) not in (None, "", []):
            entry[optional] = asset[optional]
    return entry


def _is_real(asset):
    return bool(asset) and asset.get("provider") != sb.GRAPHIC_PROVIDER


def apply_graphic_cap(shot_assets, manifest, label):
    """Gradient fallbacks may cover at most 10 % of shots; above that flag the manifest."""
    share = sb.graphic_share(shot_assets)
    if share > sb.MAX_GRAPHIC_SHARE:
        _log(f"{label}: {share:.0%} of shots are gradient fallbacks (> {sb.MAX_GRAPHIC_SHARE:.0%}); "
             "marking review_required")
        for entry in manifest:
            if entry.get("provider") == sb.GRAPHIC_PROVIDER:
                entry["review_required"] = True
                entry["review_reason"] = f"gradient fallback share {share:.0%} exceeds 10% cap"
    return share


def _flux_available():
    try:
        from src.visuals import flux_available
        return flux_available()
    except Exception:  # noqa: BLE001 - visuals deps missing in minimal test envs
        return False


def _assign(timeline, per_slide_assets, size, out_dir, label):
    """Resolve every timeline shot to an asset; returns (shot_assets, manifest).

    Fill order for shots beyond a slide's unique assets: own/AI assets, then the
    slide's real stock assets (reused non-consecutively), then real assets borrowed
    from other slides when a slide has none, and only then a gradient graphic.
    """
    from src.visuals import generated_graphic
    shot_assets = [None] * len(timeline["shots"])
    manifest = []
    donors = [(i, a) for i, assets in enumerate(per_slide_assets) for a in assets if _is_real(a)]
    for sl in timeline["slides"]:
        assets = per_slide_assets[sl["index"]]
        if not any(_is_real(a) for a in assets) and donors:
            # Farthest slides first so the borrowed visual does not repeat right after its first use.
            ranked = sorted(donors, key=lambda d: (-abs(d[0] - sl["index"]), d[1].get("kind") == "video"))
            for _, donor in ranked[:2]:
                assets.append(dict(donor, borrowed=True))
        plan = sb.assign_assets(len(sl["shots"]), assets, seed=sl["index"], allow_stock_reuse=True)
        usage = {}
        for local, global_index in enumerate(sl["shots"]):
            pick = plan[local] if local < len(plan) else None
            if pick is None:
                asset = generated_graphic(Path(out_dir) / f"{label}_{sl['index']}_{local}_fill.png", size,
                                          seed=sl["index"] * 13 + local)
                assets.append(asset)
                pick = len(assets) - 1
            shot_assets[global_index] = assets[pick]
            usage.setdefault(pick, []).append(global_index)
        for pick, shots in usage.items():
            manifest.append(manifest_entry(assets[pick], label, sl["index"] + 1, shots, timeline))
    apply_graphic_cap(shot_assets, manifest, label)
    return shot_assets, manifest


def produce_long(slides, narration_paths, out_dir, out_path, gather, used, lesson_terms=()):
    """slides: normalized slide specs (title, content, bullets, visual_queries, image_prompt, kicker)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    narration = out_dir / "narration_long.wav"
    durations = ex.build_narration(narration_paths, [SLIDE_PAD] * len(narration_paths), narration)
    timeline = sb.build_timeline(durations, sb.LONG_SHOT_TARGET)
    gaps = sb.coverage_gaps(timeline)
    if gaps:
        raise RuntimeError(f"Timeline has uncovered frames (would render black): {gaps[:5]}")

    per_slide_assets, bullet_times = [], []
    for sl, slide, path in zip(timeline["slides"], slides, narration_paths):
        n_shots = len(sl["shots"])
        terms = sb.topic_terms(slide.get("visual_queries", []), slide.get("bullets", []), *lesson_terms)
        spec = {"visual_queries": slide.get("visual_queries"), "image_prompt": slide.get("image_prompt"),
                "terms": terms}
        # Without FLUX, ask stock for more unique assets per slide (up to 6) instead of placeholders.
        count = sb.unique_visual_count(n_shots) if _flux_available() else min(n_shots, 6)
        assets = gather(spec, out_dir, count, used, orientation="landscape",
                        label=f"long{sl['index'] + 1:02d}", seed=sl["index"], min_count=0)
        per_slide_assets.append(assets)
        speech = ex.audio_duration(path)
        words = sb.estimate_word_timings(slide.get("content", ""), speech, ex.detect_speech_regions(path, speech))
        bullet_times.append(sb.bullet_reveal_times(slide.get("bullets", []), words, sl["end"] - sl["start"]))
    shot_assets, manifest = _assign(timeline, per_slide_assets, LONG_SIZE, out_dir, "long")

    audio = ex.mix_music_and_normalize(narration, out_dir / "audio_long_final.wav")
    overlay = ex.long_overlay(timeline, slides, bullet_times, LONG_SIZE)
    _log(f"rendering long video: {timeline['total']:.1f}s, {len(timeline['shots'])} shots")
    ex.render(timeline, shot_assets, audio, out_path, LONG_SIZE, overlay, progress=_log)
    storyboard = {"timeline": timeline, "bullet_times": bullet_times,
                  "slides": [{k: s.get(k) for k in ("title", "bullets", "visual_queries")} for s in slides]}
    (out_dir / "storyboard.json").write_text(json.dumps(storyboard, indent=2), encoding="utf-8")
    _cleanup(narration, audio, shot_assets)
    return {"manifest": manifest, "timeline": timeline,
            "own_assets": [a for assets in per_slide_assets for a in assets
                           if a.get("reusable") and _is_real(a)],
            # Stock stills survive cleanup; the Short may reuse them before any gradient.
            "stock_assets": [a for assets in per_slide_assets for a in assets
                             if _is_real(a) and not a.get("reusable") and a.get("kind") == "image"
                             and not a.get("borrowed")]}


def fit_short_audio(tts_fn, script, cta, work_dir):
    """TTS the Short and force its length into 20-45 s (hook-first, silence trimmed)."""
    work_dir = Path(work_dir)
    raw = tts_fn(script, work_dir / "short_voice.mp3")
    voice = ex.trim_silence(raw, work_dir / "short_voice_trim.wav")
    dur = ex.audio_duration(voice)
    if dur < SHORT_MIN and cta and cta not in script:
        script = f"{script} {cta}"
        raw = tts_fn(script, work_dir / "short_voice_cta.mp3")
        voice = ex.trim_silence(raw, work_dir / "short_voice_cta_trim.wav")
        dur = ex.audio_duration(voice)
    if dur > SHORT_MAX:
        factor = dur / (SHORT_MAX - 1.0)
        if factor <= 1.25:
            voice = ex.change_tempo(voice, work_dir / "short_voice_fast.wav", factor)
        else:
            # Far too long for tempo alone: drop the last sentences once, re-voice, then cap tempo.
            sentences = [x for x in script.split(". ") if x]
            keep = max(1, int(len(sentences) * (SHORT_MAX - 3.0) / dur))
            script = ". ".join(sentences[:keep]).rstrip(".") + "."
            raw = tts_fn(script, work_dir / "short_voice_cut.mp3")
            voice = ex.trim_silence(raw, work_dir / "short_voice_cut_trim.wav")
            dur = ex.audio_duration(voice)
            if dur > SHORT_MAX:
                voice = ex.change_tempo(voice, work_dir / "short_voice_fast.wav", min(1.25, dur / (SHORT_MAX - 1.0)))
    return script, voice


def produce_short(script, voice_path, out_dir, out_path, gather, used, spec, own_assets=(),
                  fallback_assets=()):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    narration = out_dir / "narration_short.wav"
    speech = ex.audio_duration(voice_path)
    tail = max(0.5, SHORT_MIN + 0.3 - speech)
    durations = ex.build_narration([voice_path], [tail], narration)
    total = durations[0]
    timeline = sb.build_timeline([total], sb.SHORT_SHOT_TARGET)
    if sb.coverage_gaps(timeline):
        raise RuntimeError("Short timeline has uncovered frames")
    n_shots = len(timeline["shots"])
    # New third-party visuals (each used once), topped up by this job's own FLUX stills re-framed vertically.
    count = min(3, n_shots) if _flux_available() else min(n_shots, 6)
    fresh = gather(spec, out_dir, count, used, orientation="portrait", label="short", seed=97, min_count=0)
    pool = list(fresh)
    for asset in own_assets:
        if len(pool) >= n_shots:
            break
        if _is_real(asset):
            pool.append(dict(asset))
    # Still short of unique visuals (e.g. FLUX quota hit): reuse the long video's stock stills.
    for asset in fallback_assets:
        if len(pool) >= n_shots:
            break
        if _is_real(asset) and Path(asset.get("path", "")).exists() and \
                asset.get("sha256") not in {a.get("sha256") for a in pool}:
            pool.append(dict(asset, borrowed=True))
    # Interleave so stock and AI stills alternate; reuse real assets before any gradient.
    shot_assets_list = sb.assign_assets(n_shots, pool, seed=3, allow_stock_reuse=True)
    shot_assets, manifest_usage = [], {}
    from src.visuals import generated_graphic
    for k, pick in enumerate(shot_assets_list):
        if pick is None:
            pool.append(generated_graphic(out_dir / f"short_fill_{k}.png", SHORT_SIZE, seed=200 + k))
            pick = len(pool) - 1
        shot_assets.append(pool[pick])
        manifest_usage.setdefault(pick, []).append(k)
    manifest = [manifest_entry(pool[p], "short", 1, shots, timeline) for p, shots in manifest_usage.items()]
    apply_graphic_cap(shot_assets, manifest, "short")

    words = sb.estimate_word_timings(script, speech, ex.detect_speech_regions(voice_path, speech))
    audio = ex.mix_music_and_normalize(narration, out_dir / "audio_short_final.wav")
    overlay = ex.short_overlay(words, SHORT_SIZE)
    _log(f"rendering Short: {total:.1f}s, {n_shots} shots")
    ex.render(timeline, shot_assets, audio, out_path, SHORT_SIZE, overlay)
    (out_dir / "captions.json").write_text(json.dumps(words, indent=1), encoding="utf-8")
    _cleanup(narration, audio, shot_assets)
    for leftover in out_dir.glob("short_voice*"):
        leftover.unlink(missing_ok=True)
    return {"manifest": manifest, "timeline": timeline, "words": words}
