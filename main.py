import os
import re
import sys
import json
import datetime
import time
import traceback
from pathlib import Path
from src.generator import (
    generate_curriculum,
    generate_lesson_content,
    text_to_speech,
    generate_visuals,
    create_video,
)
from src.uploader import upload_to_youtube
from dypol_control import (
    resolve_policy, execute_phase, CandidateNotFound, PolicyViolation, save_plan,
)

CONTENT_PLAN_FILE = Path("content_plan.json")
OUTPUT_DIR = Path("output")
LESSONS_PER_RUN = 1

_TAG_STOP = {"a", "an", "the", "of", "to", "in", "on", "by", "or", "and", "is", "are", "how", "what", "why",
             "when", "you", "your", "it", "its", "next", "do", "does", "for", "with", "from", "into"}


def shorts_title(text, limit=100):
    """'<text> #Shorts' within YouTube's 100-char limit, cut on a word boundary (never mid-word)."""
    suffix = " #Shorts"
    text = " ".join(str(text or "").split())
    room = limit - len(suffix)
    if len(text) > room:
        cut = text[:room + 1].rsplit(" ", 1)[0] if " " in text[:room + 1] else text[:room]
        text = cut.rstrip(" ,;:-")
    return f"{text}{suffix}"


def title_tags(title):
    """Content words of the title as tags (no 'How', 'what', 'you'...), de-duplicated."""
    seen, tags = set(), []
    for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9+#.-]*", str(title or "")):
        k = w.casefold()
        if k in _TAG_STOP or k in seen or len(k) < 2:
            continue
        seen.add(k)
        tags.append(w if w.isupper() else w.capitalize())
    return tags


def get_content_plan():
    if not CONTENT_PLAN_FILE.exists():
        print("📄 content_plan.json not found. Generating new plan...")
        new_plan = generate_curriculum()
        with open(CONTENT_PLAN_FILE, 'w') as f:
            json.dump(new_plan, f, indent=2)
        print(f"✅ New curriculum saved to {CONTENT_PLAN_FILE}")
        return new_plan
    else:
        try:
            with open(CONTENT_PLAN_FILE, 'r') as f:
                plan = json.load(f)
            if not plan.get("lessons") or not isinstance(plan["lessons"], list):
                raise ValueError("⚠️ Invalid or empty lesson plan detected.")
            return plan
        except Exception as e:
            print(f"❌ ERROR loading existing plan: {e}. Regenerating...")
            new_plan = generate_curriculum()
            with open(CONTENT_PLAN_FILE, 'w') as f:
                json.dump(new_plan, f, indent=2)
            return new_plan


def update_content_plan(plan):
    save_plan(CONTENT_PLAN_FILE, plan)



def generate_lesson_assets(lesson):
    print(f"\n▶️ Starting production for Lesson: '{lesson['title']}'")
    chapter_safe = re.sub(r'[^\w]', '_', str(lesson['chapter'])).strip('_')
    part_safe = re.sub(r'[^\w]', '_', str(lesson['part'])).strip('_')
    unique_id = f"{datetime.datetime.now().strftime('%Y%m%d')}_{chapter_safe}_{part_safe}"

    lesson_content = generate_lesson_content(lesson['title'])  # reads $REVIEWER_FEEDBACK
    from src import storyboard as sb
    from src.production import produce_long, produce_short, fit_short_audio
    from src.visuals import gather_visuals

    print("\n--- Producing Long-Form Video ---")

    body_slides = [sb.normalize_slide(s, lesson['title']) for s in lesson_content['long_form_slides']]
    lesson_terms = [lesson['title']] + [q for s in body_slides for q in s["visual_queries"]]
    intro_queries = (body_slides[0]["visual_queries"] if body_slides else []) + \
        (body_slides[1]["visual_queries"][:1] if len(body_slides) > 1 else [])
    intro_slide = {
        "title": lesson['title'], "bullets": [], "content": "",
        "kicker": ("AI tech news explained" if lesson.get('origin') == 'dypol-topic-override'
                   else f"Chapter {lesson['chapter']} | Part {lesson['part']}"),
        "visual_queries": intro_queries or sb.normalize_queries([], lesson['title']),
        "image_prompt": body_slides[0]["image_prompt"] if body_slides else lesson['title'],
    }
    outro_slide = {
        "title": "Thanks for Watching!", "kicker": __import__("os").environ.get("CHANNEL_BRAND", "Signal by DYPOL"), "content": "",
        "bullets": ["Like and share if this helped", "Subscribe for daily AI explainers"],
        "visual_queries": body_slides[-1]["visual_queries"] if body_slides else ["technology"],
        "image_prompt": body_slides[-1]["image_prompt"] if body_slides else "abstract technology",
    }
    # Style guide (Engine A): cold-open hook with a number instead of an intro; takeaway + one ask to close.
    from src import shotlist
    hook_line = str(lesson_content.get("hook") or "").strip()
    takeaway = str(lesson_content.get("takeaway") or "").strip()
    ask = str(lesson_content.get("ask") or "").strip()
    if hook_line:
        for issue in shotlist.hook_issues(hook_line):
            print(f"⚠️ hook check: {issue}")
        number = lesson_content.get("hook_number") if isinstance(lesson_content.get("hook_number"), dict) else {}
        intro_slide["shots"] = shotlist.enforce(
            [dict(number, kind="count_up", fallback="kinetic_text", text=lesson['title']),
             {"kind": "kinetic_text", "text": lesson['title']}], chapter=False)
    if takeaway and ask:
        outro_slide["title"] = "The takeaway"
        outro_slide["bullets"] = []
        outro_slide["shots"] = shotlist.enforce(
            [{"kind": "kinetic_text", "text": lesson_content.get("takeaway_short") or takeaway, "label": "Takeaway"},
             {"kind": "kinetic_text", "text": ask, "label": "Your turn"}], chapter=False)
    all_slides = [intro_slide] + body_slides + [outro_slide]

    slide_scripts = [
        hook_line or f"Hello and welcome to {__import__('os').environ.get('CHANNEL_BRAND', 'Signal by DYPOL')}. Today’s lesson is titled {lesson['title']}.",
        *[s['content'] for s in body_slides],
        (f"{takeaway} {ask} Links are in the description." if takeaway and ask else
         "Thanks for watching! If you found this helpful, make sure to subscribe to our channel and hit the like button.")
    ]
    for slide, script in zip(all_slides, slide_scripts):
        slide["content"] = script

    slide_audio_paths = []
    for i, script in enumerate(slide_scripts):
        audio_path = OUTPUT_DIR / f"audio_slide_{i+1}_{unique_id}.mp3"
        wav_path = text_to_speech(script, audio_path)
        slide_audio_paths.append(wav_path)
    print(f"🎧 Total slide audios: {len(slide_audio_paths)}")

    # Third-party assets are tracked job-wide so each is used once across both videos.
    used_assets = {"hashes": set(), "ids": set()}
    slide_dir = OUTPUT_DIR / f"slides_long_{unique_id}"
    long_video_path = OUTPUT_DIR / f"long_video_{unique_id}.mp4"
    print(f"🎥 Creating long-form video at: {long_video_path}")
    long_result = produce_long(all_slides, slide_audio_paths, slide_dir, long_video_path,
                               gather_visuals, used_assets, lesson_terms=lesson_terms,
                               video_title=lesson['title'])
    (slide_dir / "visual_manifest.json").write_text(json.dumps(long_result["manifest"], indent=2), encoding="utf-8")
    (slide_dir / "visual_judge.json").write_text(json.dumps(long_result.get("judge") or {}, indent=2), encoding="utf-8")

    long_thumb_path = generate_visuals(
        output_dir=OUTPUT_DIR / f"thumbnails_long_{unique_id}",
        video_type='long',
        thumbnail_title=lesson['title']
    )

    print("\n--- Producing Short Video ---")
    highlight_text = (lesson_content.get("short_form_highlight") or "").strip()
    hook = sb.clamp_bullet(lesson_content.get("short_hook") or "", 8)
    short_script = " ".join(str(lesson_content.get("short_script") or "").split())
    if len(short_script.split()) < 25:
        short_script = f"{hook + '. ' if hook else ''}{highlight_text}".strip()
    if hook and not short_script.casefold().startswith(hook.casefold()[:12]):
        short_script = f"{hook.rstrip('.?!')}? {short_script}" if hook.endswith("?") else f"{hook.rstrip('.')}. {short_script}"
    short_script = " ".join(short_script.replace("#", " ").split()[:110])
    short_slide_dir = OUTPUT_DIR / f"slides_short_{unique_id}"
    short_slide_dir.mkdir(parents=True, exist_ok=True)
    short_script, short_voice = fit_short_audio(
        text_to_speech, short_script, "The full lesson is on our channel.", short_slide_dir)
    short_queries = sb.normalize_queries(lesson_content.get("short_visual_queries"),
                                         " ".join(q for s in body_slides for q in s["visual_queries"]))
    short_spec = {"visual_queries": short_queries, "image_prompt": intro_slide["image_prompt"],
                  "terms": sb.topic_terms(short_queries, *lesson_terms),
                  "video_title": lesson['title'], "narration": short_script}
    short_video_path = OUTPUT_DIR / f"short_video_{unique_id}.mp4"
    print(f"🎥 Creating short video at: {short_video_path}")
    short_result = produce_short(short_script, short_voice, short_slide_dir, short_video_path,
                                 gather_visuals, used_assets, short_spec, long_result["own_assets"],
                                 fallback_assets=long_result.get("stock_assets", ()))
    (short_slide_dir / "visual_manifest.json").write_text(json.dumps(short_result["manifest"], indent=2), encoding="utf-8")
    (short_slide_dir / "visual_judge.json").write_text(json.dumps(short_result.get("judge") or {}, indent=2), encoding="utf-8")

    short_thumb_path = generate_visuals(
        output_dir=OUTPUT_DIR / f"thumbnails_short_{unique_id}",
        video_type='short',
        thumbnail_title=f"Quick Tip: {lesson['title']}"
    )

    # QC (warnings only): loudness, silences, black / frozen stretches + a contact sheet,
    # written as qc.json and contact.png into output/ (the draft artifact).
    qc_summary = {"warnings": [], "qc_json": None, "contact": None}
    try:
        from src import qc
        qc_report = qc.run(long_video_path, short_video_path, OUTPUT_DIR,
                           extra={"style_check": long_result.get("style_check"), "sfx": long_result.get("sfx")})
        qc_summary = {"warnings": qc_report.get("warnings", []), "qc_json": "qc.json",
                      "contact": qc_report.get("contact")}
    except Exception as exc:  # noqa: BLE001 - QC never blocks a draft
        print(f"⚠️ QC skipped: {exc}")

    hashtags = lesson_content.get("hashtags", "#AI #Developer #LearnAI")
    long_desc = (
        f"Part of the {__import__('os').environ.get('CHANNEL_BRAND', 'Signal by DYPOL')} series.\n\n"
        f"Today's Lesson: {lesson['title']}\n\n{hashtags}"
    )
    long_tags = ", ".join(["AI", "Artificial Intelligence", "Developer", "Programming", "Tutorial"] + title_tags(lesson["title"]))
    highlight = (lesson_content.get("short_form_highlight") or "").strip()
    if not highlight:
        highlight = f"AI Quick Tip: {lesson['title']}"
    short_title = shorts_title(highlight if len(highlight) <= 70 else lesson["title"])
    short_desc = (
        f"{highlight}\n\nWatch the full lesson on our channel.\n\n{hashtags}"
    )

    # Generation is deliberately separate from upload. The candidate and media
    # are saved first; the publish phase can only upload this exact candidate.
    visual_manifests = []
    visual_review_required = False
    for visual_dir in (slide_dir, short_slide_dir):
        manifest_path = visual_dir / "visual_manifest.json"
        if manifest_path.exists():
            try:
                entries = json.loads(manifest_path.read_text(encoding="utf-8"))
                entries = entries if isinstance(entries, list) else [entries]
                visual_manifests.extend(entries)
                visual_review_required = visual_review_required or any(
                    bool(item.get("review_required")) for item in entries
                )
            except (OSError, ValueError):
                visual_review_required = True
    # AI visual judge: circuit open, caps hit, best-MARGINAL picks or contact-sheet flags
    # make the candidate need review, exactly like the gradient cap does.
    visual_judge = {"long": long_result.get("judge") or {"judge_status": "disabled"},
                    "short": short_result.get("judge") or {"judge_status": "disabled"}}
    judge_reasons = [f"{kind}: {reason}" for kind, summary in visual_judge.items()
                     for reason in (summary.get("needs_review_reasons") or [])]
    if judge_reasons:
        visual_review_required = True
    candidate = {
        "job_id": os.environ.get("DYPOL_JOB_ID", ""),
        "lesson_title": lesson["title"],
        "review_required": visual_review_required,
        "visual_manifest": visual_manifests,
        "judge_status": visual_judge["long"].get("judge_status", "disabled"),
        "visual_judge": visual_judge,
        "visual_judge_review_reasons": judge_reasons,
        # Sources skipped this run after a quota/limit hit (e.g. FLUX daily neurons, Pixabay 429).
        "provider_limits": __import__("src.visuals", fromlist=["breaker_status"]).breaker_status(),
        "assets": [
            {
                "kind": "long",
                "path": str(long_video_path),
                "thumb_path": str(long_thumb_path),
                "captions_path": long_result.get("captions_path") or "",
                "title": lesson["title"],
                "description": long_desc,
                "tags": long_tags,
            },
            {
                "kind": "short",
                "path": str(short_video_path),
                "thumb_path": str(short_thumb_path),
                "captions_path": short_result.get("captions_path") or "",
                "title": short_title,
                "description": short_desc,
                "tags": "AI,Shorts,TechTip",
            },
        ],
        # Post-render QC warnings (never sets review_required); full report in output/qc.json.
        "qc": qc_summary,
        # Background music bed (Open Lo-Fi CC0) per video; {"provider": "none"} = rendered without music.
        "music": {"long": long_result.get("music") or {"provider": "none"},
                  "short": short_result.get("music") or {"provider": "none"}},
    }
    # Record the YouTube AI disclosure the publish step will send, so the draft shows it.
    from dypol_control import asset_visual_manifest, resolve_contains_synthetic_media
    for asset in candidate["assets"]:
        asset["contains_synthetic_media"] = resolve_contains_synthetic_media(
            asset_visual_manifest(candidate, asset))
    candidate["contains_synthetic_media"] = any(a["contains_synthetic_media"] for a in candidate["assets"])
    candidate["image_providers"] = sorted({str(e.get("provider")) for e in visual_manifests
                                           if isinstance(e, dict) and e.get("provider")})
    print("containsSyntheticMedia (draft): " + ", ".join(
        f"{a['kind']}={a['contains_synthetic_media']}" for a in candidate["assets"])
        + f"; image providers: {', '.join(candidate['image_providers']) or 'none'}", flush=True)
    return candidate

def upload_candidate_asset(asset, policy):
    """Upload one saved candidate asset and return its verified YouTube record."""
    title_override = os.environ.get("DYPOL_TITLE_OVERRIDE", "").strip()
    description_override = os.environ.get("DYPOL_DESCRIPTION_OVERRIDE", "").strip()
    kind = asset.get("kind", "video")
    title = asset.get("title", "")
    description = asset.get("description", "")
    if kind == "long":
        title = title_override or title
        description = description_override or description
    elif title_override:
        title = shorts_title(title_override)
    return upload_to_youtube(
        asset["path"],
        title,
        description,
        asset.get("tags", ""),
        asset.get("thumb_path"),
        kind=kind,
        visual_manifest=asset.get("visual_manifest"),
        captions_path=asset.get("captions_path") or None,
    )


def main():
    print("DYPOL phase-aware AI Course Generator")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        policy = resolve_policy()
        print(f"Policy: {policy.describe()}")
        plan = get_content_plan()
        try:
            result = execute_phase(
                policy,
                generate_fn=generate_lesson_assets,
                upload_fn=upload_candidate_asset,
                plan=plan,
                candidate_root=OUTPUT_DIR / "candidates",
                results_path=OUTPUT_DIR / "upload_results.json",
                run_id=os.environ.get("GITHUB_RUN_ID", ""),
            )
        except Exception:
            # execute_phase records partial/failed uploads before raising; persist
            # that state too, so the next run does not forget an upload attempt.
            update_content_plan(plan)
            print("Content plan saved after failed or partial phase.")
            raise
        update_content_plan(plan)
        print("Content plan saved.")
        print(json.dumps(result, indent=2))
    except (CandidateNotFound, PolicyViolation) as exc:
        print(f"Phase refused safely: {exc}")
        sys.exit(2)
    except Exception as exc:
        print("Pipeline failed.")
        traceback.print_exc()
        from src.failure import report_failure
        report_failure(exc, OUTPUT_DIR)
        sys.exit(1)

    try:
        for file in OUTPUT_DIR.glob("*.wav"):
            file.unlink()
            print(f"Deleted temporary audio: {file}")
    except Exception as exc:
        print(f"Warning: could not clean temporary WAV files: {exc}")

if __name__ == "__main__":
    main()
