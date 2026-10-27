# FILE: src/generator.py
# FINAL, CLEAN VERSION: Compatible with per-slide audio sync, dynamic slides, and GitHub Actions.

import os
import json
import time
import random
import requests
from io import BytesIO
from google import genai
from gtts import gTTS
from gtts.tts import gTTSError
from moviepy.editor import AudioFileClip, ImageClip, CompositeAudioClip, concatenate_videoclips, vfx
from moviepy.config import change_settings
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from pathlib import Path
from pydub import AudioSegment

# Gemini text generation fallback: 503/429 responses are transient service or capacity
# conditions. Try a small, documented model cascade instead of failing the entire job.
def _generate_text_with_fallback(client, prompt, task_name="text generation"):
    primary = (os.environ.get("GEMINI_TEXT_MODEL", "gemini-3.8-flash") or "gemini-3.8-flash").strip()
    configured_fallbacks = os.environ.get(
        "GEMINI_FALLBACK_MODELS", "gemini-3.6-flash,gemini-3.5-flash-lite"
    )
    model_names = [primary] + [
        item.strip() for item in configured_fallbacks.split(",") if item.strip()
    ]
    # Keep order stable and avoid calling the same model twice.
    model_names = list(dict.fromkeys(model_names))
    last_error = None

    for index, model_name in enumerate(model_names):
        try:
            response = client.models.generate_content(model=model_name, contents=prompt)
            if not getattr(response, "text", None):
                raise RuntimeError(f"Gemini returned an empty response for model {model_name}.")
            if index:
                print(f"✅ {task_name} succeeded with fallback model: {model_name}")
            return response
        except Exception as exc:
            last_error = exc
            message = str(exc).upper()
            code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
            transient = (
                code in (408, 429, 500, 502, 503, 504)
                or any(token in message for token in (
                    "429", "500", "502", "503", "504",
                    "UNAVAILABLE", "RESOURCE_EXHAUSTED",
                    "INTERNAL SERVER ERROR", "TIMED OUT", "TIMEOUT",
                ))
            )
            if not transient or index == len(model_names) - 1:
                raise
            print(
                f"⚠️ {task_name} model {model_name} returned a transient service error; "
                f"trying fallback {model_names[index + 1]}."
            )
            time.sleep(min(2 ** index, 4))

    if last_error:
        raise last_error
    raise RuntimeError("No Gemini text models were configured.")


# --- Configuration ---
ASSETS_PATH = Path("assets")
FONT_FILE = ASSETS_PATH / "fonts/arial.ttf"
BACKGROUND_MUSIC_PATH = ASSETS_PATH / "music/bg_music.mp3"
FALLBACK_THUMBNAIL_FONT = ImageFont.load_default()
YOUR_NAME = "Chaitanya"

# Google's TTS endpoint throttles bursts from shared CI IPs. When it does, it answers
# 200 OK with no audio stream, which gTTS surfaces as "Probable cause: Unknown".
TTS_MAX_ATTEMPTS = 5
TTS_BACKOFF_SECONDS = 5
TTS_MIN_GAP_SECONDS = 2
_last_tts_request = 0.0

# GitHub Actions compatibility for ImageMagick
if os.name == 'posix':
    change_settings({"IMAGEMAGICK_BINARY": "/usr/bin/convert"})


def get_pexels_image(query, video_type):
    """Searches for a relevant image on Pexels and returns the image object."""
    pexels_api_key = os.getenv("PEXELS_API_KEY")
    if not pexels_api_key:
        print("ℹ️ PEXELS_API_KEY not configured; using the built-in designed background.")
        return None

    orientation = 'landscape' if video_type == 'long' else 'portrait'
    try:
        headers = {"Authorization": pexels_api_key}
        params = {"query": f"abstract {query}", "per_page": 1, "orientation": orientation}
        response = requests.get("https://api.pexels.com/v1/search", headers=headers, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        if data.get('photos'):
            image_url = data['photos'][0]['src']['large2x']
            image_response = requests.get(image_url, timeout=15)
            image_response.raise_for_status()
            return Image.open(BytesIO(image_response.content)).convert("RGBA")
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error fetching Pexels image for query '{query}': {e}")
    except Exception as e:
        print(f"❌ General error fetching Pexels image for query '{query}': {e}")
    return None


def _throttle_tts():
    """Spaces out TTS requests so a burst doesn't trip Google's rate limiter."""
    global _last_tts_request
    elapsed = time.monotonic() - _last_tts_request
    if elapsed < TTS_MIN_GAP_SECONDS:
        time.sleep(TTS_MIN_GAP_SECONDS - elapsed)
    _last_tts_request = time.monotonic()


def text_to_speech(text, output_path):
    """Converts text to speech using gTTS and ensures clean audio using WAV format."""
    print(f"🎤 Converting script to speech...")
    temp_mp3_path = str(output_path).replace('.mp3', '_temp.mp3')
    wav_path = str(output_path.with_suffix('.wav'))

    for attempt in range(1, TTS_MAX_ATTEMPTS + 1):
        try:
            _throttle_tts()
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(temp_mp3_path)

            # A throttled response can still produce a file — just an empty/truncated one.
            if os.path.getsize(temp_mp3_path) < 1024:
                raise gTTSError("gTTS wrote an empty or truncated audio file.")

            audio = AudioSegment.from_mp3(temp_mp3_path)
            audio.export(wav_path, format="wav", codec="pcm_s16le")
            os.remove(temp_mp3_path)

            print(f"✅ Speech generated and converted to WAV successfully!")
            return Path(wav_path)

        except Exception as e:
            if os.path.exists(temp_mp3_path):
                os.remove(temp_mp3_path)

            if attempt == TTS_MAX_ATTEMPTS:
                print(f"❌ ERROR: Failed to generate speech after {TTS_MAX_ATTEMPTS} attempts: {e}")
                raise

            delay = TTS_BACKOFF_SECONDS * (2 ** (attempt - 1)) + random.uniform(0, 2)
            print(f"⚠️ TTS attempt {attempt}/{TTS_MAX_ATTEMPTS} failed ({e}). Retrying in {delay:.1f}s...")
            time.sleep(delay)


def generate_curriculum(previous_titles=None):
    """Generates the entire course curriculum using Gemini."""
    print("🤖 No content plan found. Generating a new curriculum from scratch...")
    try:
        client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

        #Optional: Add prior lesson titles for continuation
        history = ""
        if previous_titles:
            formatted = "\n".join([f"{i+1}. {t}" for i, t in enumerate(previous_titles)])
            history = f"The following lessons have already been created:\n{formatted}\n\nPlease continue from where this series left off.\n"

        prompt = f"""
        You are an expert AI educator. Generate a curriculum for a YouTube series called 'AI for Developers by {YOUR_NAME}'.
        {history}
        The style must be: 'Assume the viewer is a beginner or non-technical person starting their journey into AI as a developer.
        Use simple real-world analogies, relatable examples, and then connect to technical concepts.'

        The curriculum must guide a developer from absolute beginner to advanced AI, covering foundations like Generative AI, LLMs, Vector Databases, and Agentic AI...
        ...then continue into deep AI topics like Reinforcement Learning, Transformers internals, multi-agent systems, tool use, LangGraph, AI architecture, and more.

        Respond with ONLY a valid JSON object. The object must contain a key "lessons" which is a list of 20 lesson objects.
        Each lesson object must have these keys: "chapter", "part", "title", "status" (defaulted to "pending"), and "youtube_id" (defaulted to null).
        """
        response = _generate_text_with_fallback(client, prompt, task_name='curriculum generation')
        json_string = response.text.strip().replace("```json", "").replace("```", "")
        curriculum = json.loads(json_string)
        print("✅ New curriculum generated successfully!")
        return curriculum
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Failed to generate curriculum. {e}")
        raise


def generate_lesson_content(lesson_title):
    """Generates the content for one long-form lesson and its promotional short."""
    print(f"🤖 Generating content for lesson: '{lesson_title}'...")
    try:
        client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        prompt = f"""
        You are creating a lesson for the 'AI for Developers by {YOUR_NAME}' series. The topic is '{lesson_title}'.
        The style is: Assume the viewer is a beginner developer or non-tech person who wants to learn AI from scratch.
        Use analogies and clear, simple language. Each concept must be explained from a developer's perspective, assuming no prior AI or ML knowledge.

        Generate a JSON response with three keys:
        1. "long_form_slides": A list of 7 to 8 slide objects for a longer, more detailed main video. Each object needs a "title" and "content" key.
        2. "short_form_highlight": A single, punchy, 1-2 sentence summary for a YouTube Short.
        3. "hashtags": A string of 5-7 relevant, space-separated hashtags for this lesson (e.g., "#GenerativeAI #LLM #Developer","#NeuralNetworks #BeginnerAI #AIforDevelopers").

        Return only valid JSON.
        """
        response = _generate_text_with_fallback(client, prompt, task_name='lesson content generation')
        json_string = response.text.strip().replace("```json", "").replace("```", "")
        content = json.loads(json_string)
        print("✅ Lesson content generated successfully.")
        return content
    except Exception as e:
        print(f"❌ ERROR: Failed to generate lesson content: {e}")
        raise


def _make_gradient_background(width, height, seed_text=""):
    """Create a polished, dependency-free tech background when stock imagery is unavailable."""
    from PIL import ImageDraw

    top = (7, 13, 29)
    bottom = (20, 25, 49)
    image = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(image)

    # Draw a smooth vertical gradient in horizontal bands without per-pixel Python loops.
    band = 6
    for y in range(0, height, band):
        ratio = y / max(1, height - 1)
        color = tuple(int(top[i] * (1 - ratio) + bottom[i] * ratio) for i in range(3))
        draw.rectangle((0, y, width, min(height, y + band),), fill=color)

    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    deco = ImageDraw.Draw(overlay)
    scale = min(width, height)
    # Large, low-contrast rings and circuit-like rails keep the fallback from looking empty.
    cx, cy = int(width * 0.82), int(height * 0.25)
    for multiplier, alpha in [(0.20, 30), (0.32, 23), (0.45, 16)]:
        radius = int(scale * multiplier)
        deco.ellipse(
            (cx - radius, cy - radius, cx + radius, cy + radius),
            outline=(46, 168, 255, alpha),
            width=max(2, int(scale * 0.002)),
        )
    rail_y = int(height * 0.78)
    for offset in range(0, width, max(80, int(width * 0.055))):
        deco.line(
            (offset, rail_y, min(width, offset + int(width * 0.12)), rail_y),
            fill=(48, 154, 211, 34),
            width=max(1, int(scale * 0.0015)),
        )
    # Small warm accent matches the project's existing visual identity.
    accent_x = int(width * 0.08)
    deco.rounded_rectangle(
        (accent_x, int(height * 0.11), accent_x + max(12, int(width * 0.035)), int(height * 0.11) + max(5, int(height * 0.007))),
        radius=max(2, int(scale * 0.004)),
        fill=(255, 113, 0, 220),
    )
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def _load_visual_font(size):
    try:
        return ImageFont.truetype(str(FONT_FILE), size)
    except Exception:
        return ImageFont.load_default()


def _wrap_visual_text(text, font, draw, max_width):
    """Wrap text to a pixel width using the active PIL font."""
    words = str(text or "").split()
    lines = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        box = draw.textbbox((0, 0), candidate, font=font)
        if box[2] - box[0] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
                current = word
            else:
                # Split a single unusually long token to prevent horizontal clipping.
                fragment = ""
                for char in word:
                    trial = fragment + char
                    bounds = draw.textbbox((0, 0), trial, font=font)
                    if bounds[2] - bounds[0] <= max_width or not fragment:
                        fragment = trial
                    else:
                        lines.append(fragment)
                        fragment = char
                current = fragment
    if current:
        lines.append(current)
    return lines or [""]


def _fit_visual_text(text, draw, max_width, max_height, start_size, min_size, max_lines=4, gap=12):
    """Choose a readable font size and wrapped layout that fits its box."""
    size = start_size
    while size >= min_size:
        font = _load_visual_font(size)
        lines = _wrap_visual_text(text, font, draw, max_width)
        sample = draw.textbbox((0, 0), "Ag", font=font)
        line_height = max(1, sample[3] - sample[1]) + gap
        if len(lines) <= max_lines and len(lines) * line_height <= max_height:
            return font, lines, line_height
        size -= 2
    font = _load_visual_font(min_size)
    lines = _wrap_visual_text(text, font, draw, max_width)
    sample = draw.textbbox((0, 0), "Ag", font=font)
    line_height = max(1, sample[3] - sample[1]) + gap
    return font, lines, line_height


def generate_visuals(output_dir, video_type, slide_content=None, thumbnail_title=None, slide_number=0, total_slides=0):
    """Render readable AI-course slides and distinct, correctly-sized thumbnails."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    is_thumbnail = thumbnail_title is not None
    is_long = video_type == "long"
    width, height = (1920, 1080) if is_long else (1080, 1920)
    title = str(thumbnail_title if is_thumbnail else (slide_content or {}).get("title", "") or "")
    content = str((slide_content or {}).get("content", "") or "")

    background = None
    try:
        # Pexels is optional. If the API is paused or no key is configured, use our
        # built-in designed background instead of a flat black frame.
        background = get_pexels_image(title, video_type)
    except Exception as exc:
        print(f"⚠️ Stock image unavailable; using designed fallback background: {exc}")

    if background is not None:
        try:
            background = background.convert("RGBA").resize((width, height)).filter(ImageFilter.GaussianBlur(12))
            shade = Image.new("RGBA", (width, height), (5, 10, 24, 172))
            final_bg = Image.alpha_composite(background, shade).convert("RGB")
        except Exception:
            final_bg = _make_gradient_background(width, height, title)
    else:
        final_bg = _make_gradient_background(width, height, title)

    draw = ImageDraw.Draw(final_bg)
    accent = (255, 113, 0)
    cyan = (63, 180, 238)
    white = (248, 250, 255)
    muted = (202, 214, 231)

    if is_thumbnail:
        # A strong central card and adaptive title size prevent cropped titles.
        margin_x = int(width * 0.07)
        margin_y = int(height * 0.18)
        draw.rounded_rectangle(
            (margin_x, margin_y, width - margin_x, height - margin_y),
            radius=max(16, int(min(width, height) * 0.035)),
            fill=(8, 15, 31),
            outline=(45, 104, 154),
            width=max(2, int(min(width, height) * 0.003)),
        )
        draw.rounded_rectangle(
            (int(width * 0.18), int(height * 0.32), int(width * 0.18) + int(width * 0.12), int(height * 0.32) + max(8, int(height * 0.012))),
            radius=max(4, int(min(width, height) * 0.008)),
            fill=accent,
        )
        title_top = int(height * 0.38)
        title_bottom = int(height * 0.72)
        font, lines, line_height = _fit_visual_text(
            title, draw, int(width * 0.78), title_bottom - title_top,
            start_size=104 if is_long else 88,
            min_size=40,
            max_lines=4,
            gap=16,
        )
        total_height = len(lines) * line_height
        y = title_top + max(0, (title_bottom - title_top - total_height) // 2)
        for line in lines:
            box = draw.textbbox((0, 0), line, font=font)
            x = (width - (box[2] - box[0])) // 2
            draw.text((x, y), line, font=font, fill=white, stroke_width=1, stroke_fill=(0, 0, 0))
            y += line_height
        kicker = "AI FOR DEVELOPERS  •  VIDEO AUTOMATION"
        kicker_font = _load_visual_font(max(20, int(min(width, height) * 0.022)))
        kicker_box = draw.textbbox((0, 0), kicker, font=kicker_font)
        draw.text(((width - (kicker_box[2] - kicker_box[0])) // 2, int(height * 0.79)), kicker, font=kicker_font, fill=cyan)
    else:
        header_height = int(height * (0.23 if is_long else 0.18))
        outer_margin = int(width * 0.045)
        draw.rounded_rectangle(
            (outer_margin, int(height * 0.045), width - outer_margin, header_height),
            radius=max(12, int(min(width, height) * 0.025)),
            fill=(14, 31, 57),
            outline=(43, 100, 151),
            width=max(2, int(min(width, height) * 0.002)),
        )
        # Keep the accent tab wide enough for Pillow's rounded-corner geometry.
        accent_x0 = outer_margin + 18
        accent_x1 = min(width - outer_margin, accent_x0 + max(64, int(width * 0.055)))
        accent_y0 = int(height * 0.075)
        accent_y1 = min(
            header_height - max(8, int(height * 0.025)),
            accent_y0 + max(18, int(height * 0.045)),
        )
        accent_radius = max(2, min(
            int(min(width, height) * 0.008),
            max(1, (accent_x1 - accent_x0) // 2 - 1),
            max(1, (accent_y1 - accent_y0) // 2 - 1),
        ))
        if accent_x1 <= accent_x0 or accent_y1 <= accent_y0:
            raise ValueError(
                f"Invalid accent-tab geometry: {(accent_x0, accent_y0, accent_x1, accent_y1)}"
            )
        draw.rounded_rectangle(
            (accent_x0, accent_y0, accent_x1, accent_y1),
            radius=accent_radius,
            fill=accent,
        )
        title_left = int(width * 0.09)
        title_width = int(width * 0.82)
        font, title_lines, title_line_height = _fit_visual_text(
            title, draw, title_width, int(header_height * 0.72),
            start_size=82 if is_long else 80,
            min_size=42,
            max_lines=3,
            gap=8,
        )
        total_title_height = len(title_lines) * title_line_height
        title_y = max(int(height * 0.055), int(height * 0.045) + (header_height - int(height * 0.045) - total_title_height) // 2)
        for line in title_lines:
            box = draw.textbbox((0, 0), line, font=font)
            x = (width - (box[2] - box[0])) // 2
            draw.text((x, title_y), line, font=font, fill=white)
            title_y += title_line_height

        footer_height = int(height * 0.065)
        content_top = header_height + int(height * 0.055)
        content_bottom = height - footer_height - int(height * 0.055)
        panel_left = int(width * 0.07)
        panel_right = width - panel_left
        draw.rounded_rectangle(
            (panel_left, content_top, panel_right, content_bottom),
            radius=max(12, int(min(width, height) * 0.025)),
            fill=(7, 14, 30),
            outline=(33, 63, 98),
            width=max(2, int(min(width, height) * 0.0018)),
        )
        is_special_slide = len(content.split()) < 10
        body_start = 64 if is_special_slide and is_long else 68 if is_special_slide else 46 if is_long else 50
        body_min = 34 if is_long else 36
        available_height = content_bottom - content_top - int(height * 0.08)
        body_font, body_lines, body_line_height = _fit_visual_text(
            content, draw, int(width * 0.78), available_height,
            start_size=body_start,
            min_size=body_min,
            max_lines=18 if is_long else 22,
            gap=12,
        )
        # Center compact intro/outro content. Longer content begins near the panel top.
        total_body_height = len(body_lines) * body_line_height
        if is_special_slide:
            body_y = content_top + max(int(height * 0.04), (content_bottom - content_top - total_body_height) // 2)
        else:
            body_y = content_top + int(height * 0.075)
        for line in body_lines:
            box = draw.textbbox((0, 0), line, font=body_font)
            x = (width - (box[2] - box[0])) // 2
            draw.text((x, body_y), line, font=body_font, fill=muted)
            body_y += body_line_height

        # Consistent footer and slide number.
        footer_y = height - footer_height
        draw.rectangle((0, footer_y, width, height), fill=(10, 20, 39))
        draw.rectangle((0, footer_y, width, footer_y + max(4, int(height * 0.004))), fill=accent)
        footer_text = "AI for Developers  •  by Chaitanya"
        footer_font = _load_visual_font(max(18, int(min(width, height) * 0.022)))
        draw.text((int(width * 0.035), footer_y + int(footer_height * 0.30)), footer_text, font=footer_font, fill=(176, 193, 215))
        if total_slides > 0:
            number_text = f"SLIDE {slide_number} / {total_slides}"
            number_font = _load_visual_font(max(16, int(min(width, height) * 0.018)))
            box = draw.textbbox((0, 0), number_text, font=number_font)
            draw.text((width - int(width * 0.035) - (box[2] - box[0]), footer_y + int(footer_height * 0.33)), number_text, font=number_font, fill=cyan)

    file_prefix = "thumbnail_long" if is_thumbnail and is_long else "thumbnail_short" if is_thumbnail else f"slide_{slide_number:02d}"
    out_path = output_dir / f"{file_prefix}.png"
    final_bg.save(out_path)
    return str(out_path)


def create_video(slide_paths, audio_paths, output_path, video_type):
    """Creates a final video from slides and per-slide audio clips with optional background music."""
    print(f"🎬 Creating {video_type} video...")
    try:
        if not slide_paths or not audio_paths or len(slide_paths) != len(audio_paths):
            raise ValueError("Mismatch between slides and audio clips, or no slides provided.")

        image_clips = []
        for i, (img_path, audio_path) in enumerate(zip(slide_paths, audio_paths)):
            audio_clip = AudioFileClip(str(audio_path))
            duration = audio_clip.duration + 0.5  # Padding
            img_clip = (
                ImageClip(img_path)
                .set_duration(duration)
                .set_audio(audio_clip)
                .fadein(0.5)
                .fadeout(0.5)
            )
            image_clips.append(img_clip)

        final_video = concatenate_videoclips(image_clips, method="compose")

        if BACKGROUND_MUSIC_PATH.exists():
            print("🎵 Adding background music...")
            bg_music = AudioFileClip(str(BACKGROUND_MUSIC_PATH)).volumex(0.05)
            if bg_music.duration < final_video.duration:
                bg_music = bg_music.fx(vfx.loop, duration=final_video.duration)
            else:
                bg_music = bg_music.subclip(0, final_video.duration)

            composite_audio = CompositeAudioClip([
                final_video.audio.volumex(1.2),
                bg_music
            ])
            final_video = final_video.set_audio(composite_audio)

        final_video.write_videofile(
            str(output_path),
            fps=24,
            codec="libx264",
            audio_codec="aac",
            audio_bitrate="192k",
            preset="medium",
            threads=4
        )
        print(f"✅ {video_type.capitalize()} video created successfully!")

    except Exception as e:
        print(f"❌ ERROR during video creation: {e}")
        raise
