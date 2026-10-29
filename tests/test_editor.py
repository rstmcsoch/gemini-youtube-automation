"""Tests for the explainer-style editor: text limits, visual relevance, timing, rendering."""
import json
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from src import explainer as ex
from src import production as prod
from src import storyboard as sb

REPO = Path(__file__).resolve().parents[1]


def _tone(path, seconds, freq=440, pause_at=None):
    """Synthetic 'speech': a tone, optionally with a silent gap (a pause between phrases)."""
    if pause_at is None:
        expr = f"sine=frequency={freq}:duration={seconds}"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", expr, "-ar", "24000", str(path)], check=True)
    else:
        a, b = pause_at
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"sine=frequency={freq}:duration={seconds}",
                        "-af", f"volume=enable='between(t,{a},{b})':volume=0", "-ar", "24000", str(path)], check=True)
    return path


def _image(path, color, size=(640, 480)):
    Image.new("RGB", size, color).save(path)
    return path


def _asset(path, reusable=True, kind="image", provider="cloudflare_workers_ai_flux"):
    return {"kind": kind, "provider": provider, "source_url": "https://example.test/a", "asset_id": Path(path).stem,
            "license": "test license", "license_url": "https://example.test/license", "sha256": "abc",
            "path": str(path), "reusable": reusable, "third_party": not reusable}


# ------------------------------------------------------------------ on-screen text

def test_bullets_are_capped_to_three_short_lines():
    raw = ["One two three four five six seven eight nine ten", "Second point here", "Third", "Fourth point", "Fifth"]
    bullets = sb.normalize_bullets(raw)
    assert 1 <= len(bullets) <= 3
    assert all(len(b.split()) <= sb.MAX_BULLET_WORDS for b in bullets)


def test_old_schema_slide_gets_bullets_from_paragraph_and_keeps_narration():
    paragraph = ("Imagine a castle where guards must watch thousands of doors at once. An attacker only needs "
                 "one unlocked latch to break in. Defenders must keep every entrance secured all day and night.")
    slide = sb.normalize_slide({"title": "A New Digital Shield", "content": paragraph})
    assert slide["content"] == paragraph                      # full paragraph stays as narration
    assert 1 <= len(slide["bullets"]) <= 3
    assert all(len(b.split()) <= 8 for b in slide["bullets"])
    assert slide["visual_queries"]


def test_new_schema_fields_are_used_and_sanitised():
    slide = sb.normalize_slide({"title": "T", "content": "Narration.", "bullets": ["AI scans code in seconds"],
                                "visual_queries": ["Google server room", "padlock circuit"],
                                "image_prompt": "a vault door"})
    assert slide["bullets"] == ["AI scans code in seconds"]
    assert slide["visual_queries"] == ["server room", "padlock circuit"]   # brand removed


def test_gemini_prompt_requests_bullets_and_short_fields(monkeypatch):
    import src.generator as gen
    captured = {}

    class Response:
        text = json.dumps({"long_form_slides": [], "short_form_highlight": "x", "hashtags": "#a"})

    def fake(client, prompt, task_name=""):
        captured["prompt"] = prompt
        return Response()

    monkeypatch.setenv("GOOGLE_API_KEY", "test")
    monkeypatch.setattr(gen, "_generate_text_with_fallback", fake)
    monkeypatch.setattr(gen.genai, "Client", lambda api_key: object())
    gen.generate_lesson_content("Topic")
    for key in ('"bullets"', '"visual_queries"', '"image_prompt"', '"short_hook"', '"short_script"', '"content"'):
        assert key in captured["prompt"]


# ------------------------------------------------------------------ visual relevance

def test_tag_gate_rejects_animals_and_brand_only_matches():
    terms = sb.topic_terms(["cyber security", "padlock"], "Google DeepMind releases a new AI model for cybersecurity")
    ok, _, reason = sb.tag_relevance("cat, animal, pet, eyes, tiger, kitty, google pixel", terms)
    assert not ok and "animal" in reason
    ok, _, _ = sb.tag_relevance("imprint, provider, google, internet, company", {"security", "padlock"})
    assert not ok
    ok, _, _ = sb.tag_relevance("cosplay, corset, model, fashion, google", terms)
    assert not ok


def test_tag_gate_rejects_ambiguous_word_matches():
    terms = sb.topic_terms(["human in the loop"], ["model"])
    ok, _, _ = sb.tag_relevance("man, a loop, rope, human, model, depression", terms)
    assert not ok


def test_tag_gate_accepts_on_topic_hits_and_animals_when_relevant():
    terms = sb.topic_terms(["cyber security", "padlock"])
    ok, overlap, _ = sb.tag_relevance("security, padlock, lock, protection", terms)
    assert ok and "padlock" in overlap
    ok, _, _ = sb.tag_relevance("dog, robot, machine", sb.topic_terms(["robot dog"]))
    assert ok


def test_flux_prompt_has_no_brands_logos_or_text():
    prompt = sb.flux_prompt('Google DeepMind logo on a screen showing "SECURE" text and the OpenAI website', "landscape")
    body = prompt.split(". No text")[0].casefold()
    for banned in ("google", "deepmind", "openai", "logo", "secure", "website"):
        assert banned not in body
    assert "No text" in prompt and "no logos" in prompt and "no brand names" in prompt
    assert "16:9" in prompt and "9:16" in sb.flux_prompt("x", "portrait")


def test_gather_visuals_filters_pixabay_hits_by_tags(monkeypatch, tmp_path):
    import src.visuals as visuals
    monkeypatch.setattr(visuals, "pixabay_video_hits", lambda q: [])
    monkeypatch.setattr(visuals, "pixabay_still_hits", lambda q, o: [
        {"id": 1, "tags": "cat, kitten, google pixel", "imageWidth": 4000, "imageHeight": 3000,
         "largeImageURL": "https://example.test/cat.jpg", "pageURL": "https://pixabay.com/cat"},
        {"id": 2, "tags": "security, padlock, protection", "imageWidth": 4000, "imageHeight": 3000,
         "largeImageURL": "https://example.test/lock.jpg", "pageURL": "https://pixabay.com/lock"},
    ])
    downloads = []

    def fake_download(candidate, path):
        downloads.append(candidate["download_url"])
        _image(path, (10, 20, 30))
        return {"sha256": visuals._sha256(path), "width": 4000, "height": 3000}

    monkeypatch.setattr(visuals, "_download_candidate", fake_download)
    monkeypatch.setattr(visuals, "flux_generate", lambda *a, **k: None)
    used = {"hashes": set(), "ids": set()}
    spec = {"visual_queries": ["padlock"], "terms": sb.topic_terms(["padlock", "security"])}
    assets = visuals.gather_visuals(spec, tmp_path, 3, used, label="t")
    assert len(assets) == 3
    assert downloads == ["https://example.test/lock.jpg"]       # the kitten was never downloaded
    stock = [a for a in assets if a["provider"] == "pixabay"]
    assert len(stock) == 1 and stock[0]["license_url"] and stock[0]["source_url"] and stock[0]["sha256"]
    # Missing providers fall back to an original graphic (never black, never None).
    assert all(a["provider"] in {"pixabay", "dypol_generated_graphic"} for a in assets)
    # Used-once rule: a second slide cannot get the same Pixabay still again.
    again = visuals.gather_visuals(spec, tmp_path, 2, used, label="u")
    assert not [a for a in again if a["provider"] == "pixabay"]


# ------------------------------------------------------------------ timing

def test_timeline_has_no_black_gaps_and_cut_lengths():
    durations = [13.6, 32.05, 33.25, 29.25, 8.25]
    tl = sb.build_timeline(durations, sb.LONG_SHOT_TARGET)
    assert abs(tl["total"] - sum(durations)) < 1e-3
    assert sb.coverage_gaps(tl) == []
    for sl in tl["slides"]:
        assert len(sl["shots"]) >= 2
    assert all(s["end"] - s["start"] <= sb.LONG_SHOT_TARGET + 1e-6 for s in tl["shots"])
    # Consecutive shots touch exactly (no timing gap between slides or shots).
    for a, b in zip(tl["shots"], tl["shots"][1:]):
        assert abs(a["end"] - b["start"]) < 1e-3
    short = sb.build_timeline([30.0], sb.SHORT_SHOT_TARGET)
    assert all(3.0 <= s["end"] - s["start"] <= 4.0 for s in short["shots"])


def test_crossfade_length_in_required_range():
    assert 0.3 <= sb.XFADE <= 0.5


def test_assign_assets_never_repeats_third_party():
    assets = [{"reusable": False}, {"reusable": False}, {"reusable": True}, {"reusable": True}]
    plan = sb.assign_assets(7, assets)
    assert plan[:4] == [0, 1, 2, 3]
    assert plan.count(0) == 1 and plan.count(1) == 1
    assert all(p in (2, 3) for p in plan[4:])
    assert sb.assign_assets(3, [{"reusable": False}])[1:] == [None, None]


def test_word_timings_follow_speech_regions():
    words = sb.estimate_word_timings("One two three. Four five six.", 6.0, [(0.0, 2.5), (3.5, 6.0)])
    starts = [s for _, s, _ in words]
    assert starts == sorted(starts)
    assert all(not (2.5 < s < 3.5) for s in starts)          # nothing starts inside the pause
    assert words[-1][2] <= 6.0


def test_bullet_reveal_times_follow_narration_and_stay_ordered():
    text = ("Imagine a castle where guards watch thousands of doors at once and secret tunnels too. "
            "An attacker only needs one unlocked latch. Defenders must secure every entrance. "
            "This model helps teams inspect and seal weak points.")
    words = sb.estimate_word_timings(text, 20.0)
    bullets = ["Attackers need one unlocked door", "Defenders must guard every entrance",
               "AI helps inspect and seal weak points"]
    times = sb.bullet_reveal_times(bullets, words, 20.0)
    attacker = next(s for w, s, _ in words if w.startswith("attacker"))
    assert abs(times[0] - attacker) < 0.01
    assert times == sorted(times) and all(b - a >= 1.2 - 1e-6 for a, b in zip(times, times[1:]))
    assert times[-1] <= 20.0 - 1.5


def test_caption_chunks_are_short():
    words = sb.estimate_word_timings("Defenders must guard thousands of digital doors, but attackers need one.", 5)
    chunks = sb.caption_chunks(words)
    assert all(1 <= len(c) <= 3 for c in chunks)
    assert sum(len(c) for c in chunks) == len(words)


# ------------------------------------------------------------------ audio + render

def test_speech_regions_detect_pause(tmp_path):
    path = _tone(tmp_path / "v.wav", 4.0, pause_at=(1.5, 2.3))
    regions = ex.detect_speech_regions(path, 4.0)
    assert len(regions) == 2
    assert regions[0][1] == pytest.approx(1.5, abs=0.1) and regions[1][0] == pytest.approx(2.3, abs=0.1)


def test_loudnorm_hits_minus_14_lufs(tmp_path):
    voice = _tone(tmp_path / "v.wav", 8.0, freq=300)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(voice), "-af", "volume=-20dB", str(tmp_path / "q.wav")],
                   check=True)
    out = ex.mix_music_and_normalize(tmp_path / "q.wav", tmp_path / "n.wav", music_path=None)
    assert ex.measure_lufs(out) == pytest.approx(-14.0, abs=1.0)


def test_render_has_no_black_frames_and_matches_audio(tmp_path):
    size = (320, 180)
    tl = sb.build_timeline([1.6, 1.6], 0.8)
    imgs = [_image(tmp_path / f"i{k}.png", c) for k, c in enumerate([(200, 60, 40), (40, 160, 200), (90, 200, 90)])]
    shot_assets = [_asset(imgs[k % 3]) for k in range(len(tl["shots"]))]
    audio = _tone(tmp_path / "a.wav", tl["total"])
    out = ex.render(tl, shot_assets, audio, tmp_path / "o.mp4", size, lambda f, t: f, fps=15)
    err = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(out), "-vf", "blackdetect=d=0.03:pix_th=0.1", "-an",
                          "-f", "null", "-"], capture_output=True, text=True).stderr
    assert "black_start" not in err
    durs = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=duration", "-of", "csv=p=0", str(out)],
                          capture_output=True, text=True).stdout.split()
    video_d, audio_d = float(durs[0]), float(durs[1])
    assert abs(video_d - tl["total"]) < 0.1 and abs(audio_d - video_d) < 0.1


def test_produce_long_writes_flat_manifest_and_video(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    monkeypatch.setattr(prod, "LONG_SIZE", (480, 270))
    imgs = [_image(tmp_path / f"i{k}.png", (30 * k, 90, 160)) for k in range(4)]

    def gather(spec, out_dir, count, used, **kw):
        assets = [_asset(imgs[0], reusable=False, provider="pixabay")]
        assets += [_asset(p) for p in imgs[1:count]]
        return assets[:count]

    voices = [_tone(tmp_path / f"s{k}.wav", 2.0 + k) for k in range(2)]
    slides = [sb.normalize_slide({"title": "Intro", "content": "Hello there.", "bullets": []}),
              sb.normalize_slide({"title": "Point", "content": "One two three four.",
                                  "bullets": ["One two", "Three four"]})]
    result = prod.produce_long(slides, voices, tmp_path / "work", tmp_path / "long.mp4", gather,
                               {"hashes": set(), "ids": set()})
    assert (tmp_path / "long.mp4").exists()
    for entry in result["manifest"]:
        for field in ("provider", "source_url", "license", "license_url", "sha256"):
            assert field in entry
    # every shot is accounted for exactly once in the manifest
    shots = sorted(s for e in result["manifest"] for s in e["shots"])
    assert shots == list(range(len(result["timeline"]["shots"])))
    assert not list((tmp_path / "work").glob("*.wav"))          # bulky intermediates cleaned


def test_short_audio_is_forced_into_20_to_45_seconds(tmp_path):
    long_voice = _tone(tmp_path / "long.wav", 50.0)

    def tts(text, path):
        return long_voice

    _, voice = prod.fit_short_audio(tts, "Hook here. Body.", "", tmp_path)
    assert ex.audio_duration(voice) <= prod.SHORT_MAX


def test_caption_render_is_big_and_highlighted():
    img = ex.render_caption(["Defenders", "must", "guard"], 1, (1080, 1920))
    assert img.height >= 100
    colors = {px[:3] for px in img.getdata() if px[3] == 255}
    assert ex.CAPTION_ACTIVE in colors and ex.WHITE in colors


def test_workflow_uses_pip_cache_and_renderer_uses_veryfast():
    workflow = (REPO / ".github/workflows/main.yml").read_text()
    assert "cache: 'pip'" in workflow
    assert '"veryfast"' in (REPO / "src/explainer.py").read_text()
    assert 'preset="veryfast"' in (REPO / "src/generator.py").read_text()
