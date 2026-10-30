"""Caption files (SRT) for YouTube, built from the narration word timings the
pipeline already estimates for burned-in captions and bullet reveals."""

from pathlib import Path


def _ts(seconds):
    ms = max(0, int(round(float(seconds) * 1000)))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def cue_groups(word_times, max_words=10, max_chars=42, max_span=5.0):
    """Group (word, start, end) into readable subtitle cues; break after sentence ends."""
    cues, current = [], []
    for item in word_times or []:
        current.append(item)
        text = " ".join(w for w, _, _ in current)
        word = str(item[0])
        if (len(current) >= max_words or len(text) >= max_chars
                or word.endswith((".", "!", "?")) or item[2] - current[0][1] >= max_span):
            cues.append(current)
            current = []
    if current:
        cues.append(current)
    return cues


def words_to_srt(word_times, offset=0.0):
    """SRT text for word timings; ``offset`` shifts every cue (seconds)."""
    blocks = []
    cues = cue_groups([(str(w), float(a) + offset, float(b) + offset) for w, a, b in (word_times or [])
                       if str(w).strip()])
    for n, cue in enumerate(cues, 1):
        start, end = cue[0][1], max(cue[-1][2], cue[0][1] + 0.3)
        if n < len(cues):
            end = min(end, cues[n][0][1])  # never overlap the next cue
        end = max(end, start + 0.1)
        text = " ".join(w for w, _, _ in cue).replace("#", "").strip()
        blocks.append(f"{n}\n{_ts(start)} --> {_ts(end)}\n{text}\n")
    return "\n".join(blocks)


def segments_to_words(segments):
    """Flatten [(slide_start_seconds, [(word, start, end), ...]), ...] onto one timeline."""
    out = []
    for start, words in segments:
        out.extend((w, float(a) + float(start), float(b) + float(start)) for w, a, b in words or [])
    return out


def write_srt(word_times, path):
    """Write an SRT file; returns its path, or None when there is nothing to caption."""
    text = words_to_srt(word_times)
    if not text.strip():
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)
