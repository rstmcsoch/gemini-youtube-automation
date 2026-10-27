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
    YOUR_NAME
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

    print("\n--- Producing Long-Form Video ---")

    intro_slide = {"title": lesson['title'], "content": f"Chapter {lesson['chapter']} | Part {lesson['part']}"}
    outro_slide = {"title": "Thanks for Watching!", "content": "Like, Share & Subscribe for more daily AI content!\n#AIforDevelopers"}
    all_slides = [intro_slide] + lesson_content['long_form_slides'] + [outro_slide]

    slide_scripts = [
        f"Hello and welcome to AI for Developers. I'm {YOUR_NAME} talking bot. Today’s lesson is titled {lesson['title']}.",
        *[s['content'] for s in lesson_content['long_form_slides']],
        "Thanks for watching! If you found this helpful, make sure to subscribe to our channel and hit the like button."
    ]

    slide_audio_paths = []
    for i, script in enumerate(slide_scripts):
        audio_path = OUTPUT_DIR / f"audio_slide_{i+1}_{unique_id}.mp3"
        wav_path = text_to_speech(script, audio_path)
        slide_audio_paths.append(wav_path)
    print(f"🎧 Total slide audios: {len(slide_audio_paths)}")

    slide_dir = OUTPUT_DIR / f"slides_long_{unique_id}"
    slide_paths = []
    for i, slide in enumerate(all_slides):
        path = generate_visuals(
            output_dir=slide_dir,
            video_type='long',
            slide_content=slide,
            slide_number=i + 1,
            total_slides=len(all_slides)
        )
        slide_paths.append(path)

    long_video_path = OUTPUT_DIR / f"long_video_{unique_id}.mp4"
    print(f"🎥 Creating long-form video at: {long_video_path}")
    create_video(slide_paths, slide_audio_paths, long_video_path, 'long')

    long_thumb_path = generate_visuals(
        output_dir=OUTPUT_DIR,
        video_type='long',
        thumbnail_title=lesson['title']
    )

    print("\n--- Producing Short Video ---")
    # short_script = f"{lesson_content['short_form_highlight']}"
    short_script = (f"{lesson_content['short_form_highlight']}\n\n"
    f"Link to the full lesson is in the description below.")
    short_audio_mp3_path = OUTPUT_DIR / f"short_audio_{unique_id}.mp3"
    short_audio_path = text_to_speech(short_script, short_audio_mp3_path)

    short_slide_dir = OUTPUT_DIR / f"slides_short_{unique_id}"
    short_slide_content = {
        "title": "Quick Tip!",
        "content": f"{lesson_content['short_form_highlight']}\n\n#AI for developers by chaitanya"
    }
    short_slide_path = generate_visuals(
        output_dir=short_slide_dir,
        video_type='short',
        slide_content=short_slide_content,
        slide_number=1,
        total_slides=1
    )

    short_video_path = OUTPUT_DIR / f"short_video_{unique_id}.mp4"
    print(f"🎥 Creating short video at: {short_video_path}")
    create_video([short_slide_path], [short_audio_path], short_video_path, 'short')

    short_thumb_path = generate_visuals(
        output_dir=OUTPUT_DIR,
        video_type='short',
        thumbnail_title=f"Quick Tip: {lesson['title']}"
    )

    hashtags = lesson_content.get("hashtags", "#AI #Developer #LearnAI")
    long_desc = (
        f"Part of the 'AI for Developers' series by {YOUR_NAME}.\n\n"
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
    return {
        "job_id": os.environ.get("DYPOL_JOB_ID", ""),
        "lesson_title": lesson["title"],
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
