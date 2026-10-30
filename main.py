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

    lesson_content = generate_lesson_content(lesson['title'])
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
    all_slides = [intro_slide] + body_slides + [outro_slide]

    slide_scripts = [
        f"Hello and welcome to {__import__('os').environ.get('CHANNEL_BRAND', 'Signal by DYPOL')}. Today’s lesson is titled {lesson['title']}.",
        *[s['content'] for s in body_slides],
        "Thanks for watching! If you found this helpful, make sure to subscribe to our channel and hit the like button."
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

    hashtags = lesson_content.get("hashtags", "#AI #Developer #LearnAI")
    long_desc = (
        f"Part of the {__import__('os').environ.get('CHANNEL_BRAND', 'Signal by DYPOL')} series.\n\n"
        f"Today's Lesson: {lesson['title']}\n\n{hashtags}"
    )
    long_tags = "AI, Artificial Intelligence, Developer, Programming, Tutorial, " + lesson["title"].replace(" ", ", ")
    highlight = (lesson_content.get("short_form_highlight") or "").strip()
    if not highlight:
        highlight = f"AI Quick Tip: {lesson['title']}"
    short_title = f"{highlight[:90].rstrip()} #Shorts"
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
    return {
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
                "title": lesson["title"],
                "description": long_desc,
                "tags": long_tags,
            },
            {
                "kind": "short",
                "path": str(short_video_path),
                "thumb_path": str(short_thumb_path),
                "title": short_title,
                "description": short_desc,
                "tags": "AI,Shorts,TechTip",
            },
        ],
    }

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
        title = f"{title_override[:88].rstrip()} #Shorts"
    return upload_to_youtube(
        asset["path"],
        title,
        description,
        asset.get("tags", ""),
        asset.get("thumb_path"),
        kind=kind,
        visual_manifest=asset.get("visual_manifest"),
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
    except Exception:
        print("Pipeline failed.")
        traceback.print_exc()
        sys.exit(1)

    try:
        for file in OUTPUT_DIR.glob("*.wav"):
            file.unlink()
            print(f"Deleted temporary audio: {file}")
    except Exception as exc:
        print(f"Warning: could not clean temporary WAV files: {exc}")

if __name__ == "__main__":
    main()
