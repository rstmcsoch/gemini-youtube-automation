"""Explainer-style renderer: moving full-frame visuals, animated lower-third text,
word-level captions for Shorts, crossfades, and loudness-normalised audio.

Frames are composed with Pillow and piped straight into a single ffmpeg encode
(libx264 -preset veryfast), so the video length is derived from the narration
length and every frame has a visible shot (no black gaps).
"""
import json
import math
import random
import re
import subprocess
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pydub import AudioSegment

from src import storyboard as sb

ASSETS = Path("assets")
FONT_REGULAR = [ASSETS / "fonts/arial.ttf", Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")]
FONT_BOLD = [ASSETS / "fonts/arialbd.ttf", Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")]
MUSIC = ASSETS / "music/bg_music.mp3"

FPS = 30
OVERSCAN = 1.18          # source is scaled this much larger than the frame for Ken Burns room
ACCENT = (255, 113, 0)
CYAN = (63, 180, 238)
WHITE = (248, 250, 255)
SUBTLE = (214, 224, 238)
CAPTION_ACTIVE = (255, 214, 0)

_BILINEAR = Image.Resampling.BILINEAR
_LANCZOS = Image.Resampling.LANCZOS

_font_cache = {}


def font(size, bold=False):
    key = (size, bold)
    if key not in _font_cache:
        for candidate in (FONT_BOLD if bold else FONT_REGULAR):
            try:
                _font_cache[key] = ImageFont.truetype(str(candidate), size)
                break
            except OSError:
                continue
        else:
            _font_cache[key] = ImageFont.load_default()
    return _font_cache[key]


def _ease(p):
    p = max(0.0, min(1.0, p))
    return p * p * (3 - 2 * p)


def _wrap(text, fnt, max_width, draw):
    words, lines, current = str(text).split(), [], ""
    for word in words:
        trial = f"{current} {word}".strip()
        if draw.textlength(trial, font=fnt) <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


# ------------------------------------------------------------------ audio

def mean_volume_db(path):
    try:
        err = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af", "volumedetect",
                              "-f", "null", "-"], capture_output=True, text=True, timeout=120).stderr
        return float(re.findall(r"mean_volume: (-?[\d.]+) dB", err)[-1])
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        return -20.0


def detect_speech_regions(path, duration, noise_db=None, min_silence=0.18):
    """Speech regions = complement of ffmpeg silencedetect output.

    The threshold adapts to the clip's mean level (mean - 8 dB), which finds the
    pauses between phrases even under a music bed; tuned against Whisper word
    timestamps on a gTTS sample (mean word-start error ~0.3 s).
    """
    if noise_db is None:
        noise_db = max(-40.0, min(-22.0, mean_volume_db(path) - 8.0))
    try:
        err = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af",
                              f"silencedetect=noise={noise_db}dB:d={min_silence}", "-f", "null", "-"],
                             capture_output=True, text=True, timeout=120).stderr
    except (OSError, subprocess.SubprocessError):
        return [(0.0, duration)]
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", err)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", err)]
    silences = list(zip(starts, ends + [duration] * (len(starts) - len(ends))))
    regions, cursor = [], 0.0
    for a, b in silences:
        if a > cursor + 0.05:
            regions.append((cursor, a))
        cursor = max(cursor, b)
    if duration > cursor + 0.05:
        regions.append((cursor, duration))
    return regions or [(0.0, duration)]


def audio_duration(path):
    return len(AudioSegment.from_file(str(path))) / 1000.0


def trim_silence(path, out_path, keep_ms=60, threshold_db=-40):
    """Trim leading/trailing TTS silence so the hook starts immediately."""
    from pydub.silence import detect_leading_silence
    seg = AudioSegment.from_file(str(path))
    lead = max(0, detect_leading_silence(seg, silence_threshold=threshold_db) - keep_ms)
    tail = max(0, detect_leading_silence(seg.reverse(), silence_threshold=threshold_db) - keep_ms * 3)
    trimmed = seg[lead: len(seg) - tail] if len(seg) - tail > lead + 200 else seg
    trimmed.export(str(out_path), format="wav")
    return Path(out_path)


def change_tempo(path, out_path, factor):
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(path), "-af",
                    f"atempo={factor:.4f}", str(out_path)], check=True)
    return Path(out_path)


def build_narration(segment_paths, pads, out_wav, min_total=None):
    """Concatenate per-slide narration with exact padding; returns per-slide durations (s)."""
    track = AudioSegment.silent(duration=0, frame_rate=48000).set_channels(2)
    durations = []
    for path, pad in zip(segment_paths, pads):
        seg = AudioSegment.from_file(str(path)).set_frame_rate(48000).set_channels(2)
        seg = seg + AudioSegment.silent(duration=int(round(pad * 1000)), frame_rate=48000).set_channels(2)
        durations.append(len(seg) / 1000.0)
        track += seg
    if min_total and len(track) < min_total * 1000:
        extra = int(math.ceil(min_total * 1000 - len(track)))
        track += AudioSegment.silent(duration=extra, frame_rate=48000).set_channels(2)
        durations[-1] += extra / 1000.0
    track.export(str(out_wav), format="wav")
    return durations


def mix_music_and_normalize(narration_wav, out_wav, music_path=MUSIC, music_db=-23, target_lufs=-14.0):
    """Mix quiet music under narration and loudnorm to -14 LUFS (two-pass, true peak -1.5 dBTP)."""
    total = audio_duration(narration_wav)
    mixed = Path(str(out_wav) + ".mix.wav")
    if music_path and Path(music_path).exists():
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(narration_wav),
               "-stream_loop", "-1", "-i", str(music_path), "-filter_complex",
               f"[1:a]volume={music_db}dB,aformat=sample_rates=48000:channel_layouts=stereo,"
               f"afade=t=in:d=1.5,afade=t=out:st={max(0.0, total - 2.0):.3f}:d=2[m];"
               f"[0:a]aformat=sample_rates=48000:channel_layouts=stereo[v];"
               f"[v][m]amix=inputs=2:duration=first:normalize=0[a]",
               "-map", "[a]", "-t", f"{total:.3f}", str(mixed)]
    else:
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(narration_wav), str(mixed)]
    subprocess.run(cmd, check=True)
    params = f"I={target_lufs}:TP=-1.5:LRA=11"
    err = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(mixed), "-af",
                          f"loudnorm={params}:print_format=json", "-f", "null", "-"],
                         capture_output=True, text=True, check=True).stderr
    measured = json.loads(err[err.rindex("{"): err.rindex("}") + 1])
    second = (f"loudnorm={params}:measured_I={measured['input_i']}:measured_TP={measured['input_tp']}"
              f":measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}"
              f":offset={measured['target_offset']}:linear=true")
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(mixed), "-af", second,
                    "-ar", "48000", "-t", f"{total:.3f}", str(out_wav)], check=True)
    mixed.unlink(missing_ok=True)
    return Path(out_wav)


def measure_lufs(path):
    err = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af",
                          "loudnorm=print_format=json", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    return float(json.loads(err[err.rindex("{"): err.rindex("}") + 1])["input_i"])


# ------------------------------------------------------------------ visual sources

def _cover(img, width, height):
    scale = max(width / img.width, height / img.height)
    resized = img.resize((max(width, math.ceil(img.width * scale)), max(height, math.ceil(img.height * scale))),
                         _LANCZOS)
    left = (resized.width - width) // 2
    top = (resized.height - height) // 2
    return resized.crop((left, top, left + width, top + height))


class StillSource:
    """Ken Burns: smooth sub-pixel zoom + pan over an oversized still."""

    def __init__(self, path, size, span, seed=0):
        self.size = size
        self.span = max(0.1, span)
        w, h = size
        with Image.open(path) as im:
            self.src = _cover(im.convert("RGB"), int(w * OVERSCAN), int(h * OVERSCAN))
        rng = random.Random(seed)
        zoom_in = rng.random() < 0.6
        lo, hi = 1.0, OVERSCAN * 0.98
        self.z0, self.z1 = (lo, hi) if zoom_in else (hi, lo)
        self.f0 = (rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7))
        self.f1 = (rng.uniform(0.25, 0.75), rng.uniform(0.25, 0.75))

    def frame(self, local_t):
        p = max(0.0, min(1.0, local_t / self.span))
        z = self.z0 + (self.z1 - self.z0) * p
        fx = self.f0[0] + (self.f1[0] - self.f0[0]) * p
        fy = self.f0[1] + (self.f1[1] - self.f0[1]) * p
        sw, sh = self.src.size
        cw, ch = sw / z, sh / z
        x0 = (sw - cw) * fx
        y0 = (sh - ch) * fy
        # resize(box=...) takes a float box (sub-pixel motion, no judder) and is ~2x faster than transform().
        return self.src.resize(self.size, _BILINEAR, box=(x0, y0, x0 + cw, y0 + ch))

    def close(self):
        pass


class VideoSource:
    """Sequential ffmpeg reader, scaled/cropped to cover the frame, looped if short."""

    def __init__(self, path, size, span, fps=FPS, skip=0.3):
        self.size, self.fps = size, fps
        w, h = size
        self.frame_bytes = w * h * 3
        self.index = -1
        self.last = None
        self.proc = subprocess.Popen(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-stream_loop", "-1", "-ss", f"{skip}",
             "-i", str(path), "-an", "-vf",
             f"scale={w}:{h}:force_original_aspect_ratio=increase:flags=bicubic,crop={w}:{h},fps={fps},format=rgb24",
             "-t", f"{span + 1.0:.3f}", "-f", "rawvideo", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def frame(self, local_t):
        want = int(local_t * self.fps)
        while self.index < want:
            data = self.proc.stdout.read(self.frame_bytes)
            if len(data) < self.frame_bytes:
                break
            self.last = Image.frombytes("RGB", self.size, data)
            self.index += 1
        return self.last if self.last is not None else Image.new("RGB", self.size, (12, 18, 34))

    def close(self):
        try:
            self.proc.stdout.close()
            self.proc.kill()
            self.proc.wait(timeout=5)
        except (OSError, subprocess.SubprocessError):
            pass


# ------------------------------------------------------------------ overlays

def render_lower_third(slide, size):
    """Return (panel RGBA, origin, [(bullet RGBA, origin), ...]) for a long-form slide."""
    w, h = size
    scale = h / 1080
    pad = int(28 * scale)
    panel_w = int(w * 0.43)
    inner = panel_w - 2 * pad - int(14 * scale)
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    kicker = str(slide.get("kicker") or "").upper()
    head_font = font(int((56 if not slide.get("bullets") else 46) * scale), bold=True)
    head_lines = _wrap(slide["title"], head_font, inner, probe)[:3]
    bullet_font = font(int(33 * scale))
    bullet_blocks = [_wrap(b, bullet_font, inner - int(34 * scale), probe)[:2] for b in slide.get("bullets", [])]
    head_lh = int(head_font.size * 1.18)
    bullet_lh = int(bullet_font.size * 1.25)
    kick_font = font(int(24 * scale), bold=True)
    height = pad + (int(kick_font.size * 1.6) if kicker else 0) + head_lh * len(head_lines)
    if bullet_blocks:
        height += int(16 * scale)
        height += sum(bullet_lh * len(b) + int(10 * scale) for b in bullet_blocks)
    height += pad
    # Shrink the card to its content so it covers as little of the picture as possible.
    widest = max([probe.textlength(line, font=head_font) for line in head_lines] +
                 [probe.textlength(line, font=bullet_font) + int(34 * scale) for b in bullet_blocks for line in b] +
                 [probe.textlength(kicker, font=kick_font) if kicker else 0])
    panel_w = min(panel_w, max(int(w * 0.26), int(widest) + 2 * pad + int(14 * scale) + 4))
    x0 = int(w * 0.04)
    y0 = h - int(h * 0.06) - height
    panel = Image.new("RGBA", (panel_w, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(panel)
    d.rounded_rectangle((0, 0, panel_w - 1, height - 1), radius=int(20 * scale), fill=(8, 14, 30, 200))
    d.rounded_rectangle((0, 0, int(8 * scale), height - 1), radius=int(4 * scale), fill=ACCENT + (255,))
    tx = pad + int(14 * scale)
    y = pad
    if kicker:
        d.text((tx, y), kicker, font=kick_font, fill=CYAN + (255,))
        y += int(kick_font.size * 1.6)
    for line in head_lines:
        d.text((tx, y), line, font=head_font, fill=WHITE + (255,))
        y += head_lh
    bullets = []
    if bullet_blocks:
        y += int(16 * scale)
        for block in bullet_blocks:
            bh = bullet_lh * len(block) + int(4 * scale)
            img = Image.new("RGBA", (inner, bh), (0, 0, 0, 0))
            bd = ImageDraw.Draw(img)
            r = int(7 * scale)
            cy = int(bullet_lh * 0.5)
            bd.ellipse((int(4 * scale), cy - r, int(4 * scale) + 2 * r, cy + r), fill=ACCENT + (255,))
            by = 0
            for line in block:
                bd.text((int(34 * scale), by), line, font=bullet_font, fill=SUBTLE + (255,))
                by += bullet_lh
            bullets.append((img, (x0 + tx, y0 + y)))
            y += bullet_lh * len(block) + int(10 * scale)
    return panel, (x0, y0), bullets


def render_bug(size, text=None):
    text = text or __import__("os").environ.get("CHANNEL_BRAND", "Signal by DYPOL")
    w, h = size
    fnt = font(int(min(w, h) * 0.024), bold=True)
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    tw = int(probe.textlength(text, font=fnt))
    pad = int(fnt.size * 0.6)
    img = Image.new("RGBA", (tw + 2 * pad, int(fnt.size * 1.9)), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, img.width - 1, img.height - 1), radius=pad, fill=(8, 14, 30, 150))
    d.text((pad, int(fnt.size * 0.4)), text, font=fnt, fill=(235, 240, 250, 230))
    return img, (int(w * 0.03), int(h * 0.035))


def _with_alpha(img, alpha):
    if alpha >= 0.999:
        return img
    r, g, b, a = img.split()
    a = a.point(lambda v: int(v * alpha))
    return Image.merge("RGBA", (r, g, b, a))


def render_caption(words, active, size):
    """Big bold outlined caption chunk; the spoken word is highlighted."""
    w, h = size
    fnt = font(int(w * 0.095), bold=True)
    stroke = max(4, int(fnt.size * 0.09))
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    space = probe.textlength(" ", font=fnt)
    max_w = int(w * 0.88)
    lines, line, lw = [], [], 0.0
    for i, word in enumerate(words):
        ww = probe.textlength(word, font=fnt)
        if line and lw + space + ww > max_w:
            lines.append(line)
            line, lw = [], 0.0
        line.append((i, word, ww))
        lw += (space if len(line) > 1 else 0) + ww
    if line:
        lines.append(line)
    lh = int(fnt.size * 1.15)
    img = Image.new("RGBA", (w, lh * len(lines) + 2 * stroke), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    y = stroke
    for line in lines:
        total = sum(ww for _, _, ww in line) + space * (len(line) - 1)
        x = (w - total) / 2
        for i, word, ww in line:
            d.text((x, y), word, font=fnt, fill=(CAPTION_ACTIVE if i == active else WHITE) + (255,),
                   stroke_width=stroke, stroke_fill=(0, 0, 0, 255))
            x += ww + space
        y += lh
    return img


# ------------------------------------------------------------------ renderer

def _make_source(asset, size, span, seed):
    if asset.get("kind") == "video":
        return VideoSource(asset["path"], size, span)
    return StillSource(asset["path"], size, span, seed=seed)


def shot_windows(timeline):
    half = sb.XFADE / 2
    total = timeline["total"]
    return [(max(0.0, s["start"] - half), min(total, s["end"] + half)) for s in timeline["shots"]]


def render_workers():
    """Parallel render chunks: $DYPOL_RENDER_WORKERS, else cores - 1 (max 4); 1 = serial."""
    import os
    try:
        value = int(os.environ.get("DYPOL_RENDER_WORKERS", "") or 0)
    except ValueError:
        value = 0
    if value <= 0:
        value = min(4, max(1, (os.cpu_count() or 1) - 1))
    return max(1, value)


def _encoder_cmd(size, fps, crf, out_path, audio_path=None, total=None, threads=None):
    w, h = size
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-"]
    if audio_path is not None:
        cmd += ["-i", str(audio_path), "-map", "0:v", "-map", "1:a"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p"]
    if threads:
        cmd += ["-threads", str(threads)]
    if audio_path is not None:
        cmd += ["-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-t", f"{total:.3f}"]
    else:
        # Chunk files are MPEG-TS: they concatenate cleanly with stream copy.
        cmd += ["-an", "-f", "mpegts"]
    return cmd + [str(out_path)]


def _compose_range(timeline, shot_assets, size, overlay_fn, fps, start, stop, write, progress=None, label=""):
    """Compose frames [start, stop) and hand each RGB frame's bytes to `write`.

    Every frame is a pure function of its time t (stills are Ken Burns over t, clips
    are read sequentially from their window start), so any frame range can be
    rendered independently and the result is identical to a single pass.
    """
    half = sb.XFADE / 2
    shots = timeline["shots"]
    total = timeline["total"]
    windows = shot_windows(timeline)
    sources = {}
    try:
        for f in range(start, stop):
            t = f / fps
            active = [i for i, (a, b) in enumerate(windows) if a <= t < b] or [len(shots) - 1]
            layers = []
            for i in active[-2:]:
                if i not in sources:
                    a, b = windows[i]
                    sources[i] = _make_source(shot_assets[i], size, b - a, seed=1000 + i)
                layers.append(sources[i].frame(t - windows[i][0]))
            frame = layers[0]
            if len(layers) > 1:
                boundary = shots[active[-1]]["start"]
                alpha = _ease((t - (boundary - half)) / sb.XFADE)
                frame = Image.blend(layers[0], layers[1], alpha)
            frame = overlay_fn(frame, t)
            write(frame.convert("RGB").tobytes())
            for i in list(sources):
                if windows[i][1] <= t:
                    sources.pop(i).close()
            if progress and (f - start) % (fps * 30) == 0:
                progress(f"{label}rendered {t:.0f}s / {total:.0f}s")
    finally:
        for src in sources.values():
            src.close()


def _render_serial(timeline, shot_assets, audio_path, out_path, size, overlay_fn, fps, crf, progress):
    total = timeline["total"]
    frames = int(math.ceil(total * fps))
    enc = subprocess.Popen(_encoder_cmd(size, fps, crf, out_path, audio_path, total), stdin=subprocess.PIPE)
    try:
        _compose_range(timeline, shot_assets, size, overlay_fn, fps, 0, frames, enc.stdin.write, progress)
    finally:
        enc.stdin.close()
        code = enc.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg encode failed with exit code {code}")
    return Path(out_path)


_PARALLEL_JOB = {}


def _render_chunk(index):
    """Child process: render one frame range to a video-only H.264 file (fork start method)."""
    job = _PARALLEL_JOB
    start, stop = job["bounds"][index]
    out = job["chunk_paths"][index]
    enc = subprocess.Popen(_encoder_cmd(job["size"], job["fps"], job["crf"], out, threads=job["threads"]),
                           stdin=subprocess.PIPE)
    try:
        _compose_range(job["timeline"], job["shot_assets"], job["size"], job["overlay_fn"], job["fps"],
                       start, stop, enc.stdin.write, job["progress"],
                       label=f"chunk {index + 1}/{len(job['bounds'])}: ")
    finally:
        enc.stdin.close()
        code = enc.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg chunk encode failed with exit code {code}")
    return str(out)


def chunk_bounds(frames, workers):
    """Split [0, frames) into `workers` contiguous, near-equal ranges."""
    step = frames / workers
    edges = [int(round(k * step)) for k in range(workers)] + [frames]
    return [(edges[k], edges[k + 1]) for k in range(workers) if edges[k + 1] > edges[k]]


def _render_parallel(timeline, shot_assets, audio_path, out_path, size, overlay_fn, fps, crf, progress, workers):
    import multiprocessing as mp
    import os
    import shutil
    import tempfile
    total = timeline["total"]
    frames = int(math.ceil(total * fps))
    bounds = chunk_bounds(frames, workers)
    tmp = Path(tempfile.mkdtemp(prefix="render_chunks_", dir=str(Path(out_path).parent)))
    try:
        chunk_paths = [tmp / f"chunk_{k:02d}.ts" for k in range(len(bounds))]
        _PARALLEL_JOB.clear()
        _PARALLEL_JOB.update(timeline=timeline, shot_assets=shot_assets, size=size, overlay_fn=overlay_fn,
                             fps=fps, crf=crf, bounds=bounds, chunk_paths=chunk_paths, progress=progress,
                             threads=max(2, (os.cpu_count() or 2) // len(bounds)))
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=len(bounds)) as pool:
            pool.map(_render_chunk, range(len(bounds)))
        listing = tmp / "chunks.txt"
        listing.write_text("".join(f"file '{p.name}'\n" for p in chunk_paths), encoding="utf-8")
        # Chunks share identical encode settings, so a stream-copy concat is lossless;
        # only the audio is encoded here (same AAC settings as the serial path).
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0",
               "-i", str(listing), "-i", str(audio_path), "-map", "0:v", "-map", "1:a", "-c:v", "copy",
               "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-t", f"{total:.3f}", str(out_path)]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg concat failed with exit code {result.returncode}")
    finally:
        _PARALLEL_JOB.clear()
        shutil.rmtree(tmp, ignore_errors=True)
    return Path(out_path)


def render(timeline, shot_assets, audio_path, out_path, size, overlay_fn, fps=FPS, crf=21, progress=None,
           workers=None):
    """Compose frames and encode. `shot_assets[i]` is the asset dict for timeline shot i.

    With more than one worker the timeline is split into contiguous frame ranges
    rendered by separate processes (Python frame composition is the bottleneck),
    then stream-copy concatenated; same resolution, fps, CRF and preset as the
    single-pass path. Any parallel failure falls back to the single pass.
    """
    frames = int(math.ceil(timeline["total"] * fps))
    workers = render_workers() if workers is None else max(1, int(workers))
    workers = min(workers, max(1, frames // (fps * 4)))  # chunks of at least ~4 s
    if workers > 1:
        try:
            if progress:
                progress(f"parallel render: {workers} chunks")
            return _render_parallel(timeline, shot_assets, audio_path, out_path, size, overlay_fn, fps, crf,
                                    progress, workers)
        except Exception as exc:  # noqa: BLE001 - never lose a render to the speed path
            if progress:
                progress(f"parallel render failed ({exc}); falling back to single pass")
            Path(out_path).unlink(missing_ok=True)
    return _render_serial(timeline, shot_assets, audio_path, out_path, size, overlay_fn, fps, crf, progress)


def long_overlay(timeline, slides, bullet_times, size):
    """Lower-third panel per slide; bullets fade/slide in at their narration times."""
    bug, bug_xy = render_bug(size)
    prepared = [render_lower_third(s, size) for s in slides]
    shift = int(36 * size[1] / 1080)

    def apply(frame, t):
        frame = frame.convert("RGBA")
        frame.alpha_composite(bug, bug_xy)
        for k, sl in enumerate(timeline["slides"]):
            if not (sl["start"] <= t < sl["end"]):
                continue
            local, dur = t - sl["start"], sl["end"] - sl["start"]
            fade = _ease((local - 0.15) / 0.35) * _ease((dur - 0.08 - local) / 0.3)
            if fade <= 0.01:
                break
            panel, origin, bullets = prepared[k]
            frame.alpha_composite(_with_alpha(panel, fade), origin)
            for (img, (bx, by)), reveal in zip(bullets, bullet_times[k]):
                p = _ease((local - reveal) / 0.4)
                if p <= 0.01:
                    continue
                frame.alpha_composite(_with_alpha(img, p * fade), (bx + int(shift * (1 - p)), by))
            break
        return frame.convert("RGB")

    return apply


def short_overlay(word_times, size):
    """Word-level burned-in captions centred ~20% from the bottom."""
    chunks = sb.caption_chunks(word_times)
    starts = [c[0][1] for c in chunks]
    cache = {}
    bug, bug_xy = render_bug(size)
    w, h = size

    def apply(frame, t):
        frame = frame.convert("RGBA")
        frame.alpha_composite(bug, bug_xy)
        idx = None
        for k in range(len(chunks)):
            end = starts[k + 1] if k + 1 < len(chunks) else chunks[k][-1][2] + 0.6
            if (starts[k] - 0.05 if k else 0.0) <= t < end:
                idx = k
                break
        if idx is not None:
            chunk = chunks[idx]
            active = 0
            for j, (_, s, _) in enumerate(chunk):
                if t >= s:
                    active = j
            key = (idx, active)
            if key not in cache:
                cache[key] = render_caption([wd for wd, _, _ in chunk], active, size)
            img = cache[key]
            y = int(h * 0.80) - img.height // 2
            frame.alpha_composite(img, (0, y))
        return frame.convert("RGB")

    return apply


def contact_sheet(video_path, out_path, count=20, columns=5, thumb_width=384):
    """Evenly spaced frames tiled into one JPEG for visual QA."""
    duration = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                                     "csv=p=0", str(video_path)], capture_output=True, text=True).stdout.strip())
    tiles = []
    for k in range(count):
        t = duration * (k + 0.5) / count
        data = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", str(video_path), "-frames:v", "1",
                               "-vf", f"scale={thumb_width}:-2", "-f", "image2pipe", "-vcodec", "png", "-"],
                              capture_output=True).stdout
        tiles.append(Image.open(BytesIO(data)).convert("RGB"))
    tw, th = tiles[0].size
    rows = math.ceil(count / columns)
    sheet = Image.new("RGB", (tw * columns, th * rows))
    for k, tile in enumerate(tiles):
        sheet.paste(tile, ((k % columns) * tw, (k // columns) * th))
    sheet.save(out_path, quality=85)
    return Path(out_path)
