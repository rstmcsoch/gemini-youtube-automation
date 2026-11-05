"""Long video + Short production on top of the explainer renderer.

The visual gatherer is injectable so a local sample can be rendered without
network calls (tests and mock renders pass a fake gatherer).
"""
import json
import re
from pathlib import Path

from src import explainer as ex
from src import storyboard as sb
from src import captions as caps
from src import pacing

LONG_SIZE = (1920, 1080)
SHORT_SIZE = (1080, 1920)
SLIDE_PAD = 0.45          # seconds of breathing room after each slide's narration
SHORT_MIN, SHORT_MAX = 50.0, 65.0     # a real Short: hook, 2-3 beats, the catch, a takeaway

MANIFEST_FIELDS = ("provider", "source_url", "asset_id", "license", "license_url", "sha256")


def _log(message):
    print(f"🎬 {message}", flush=True)


def _write_captions(words, path):
    """Write the YouTube caption file; a failure here never stops the render."""
    try:
        return caps.write_srt(words, path)
    except Exception as exc:  # noqa: BLE001
        _log(f"WARNING: caption file not written ({path}): {exc}")
        return None


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
                     "fallback_query", "borrowed", "fallback_for"):
        if asset.get(optional) not in (None, "", []):
            entry[optional] = asset[optional]
    entry["judge_status"] = _judge_status()
    if asset.get("judge"):
        entry["judge"] = dict(asset["judge"])
        if asset["judge"].get("needs_review"):
            entry["review_required"] = True
            entry["review_reason"] = f"visual judge: {asset['judge'].get('review_reason', 'needs review')}"
    return entry


# ---------------------------------------------------------------------------
# Visual judge hooks (never raise; a missing/failed judge leaves today's behaviour)
# ---------------------------------------------------------------------------

def _get_judge():
    try:
        from src.visual_judge import get_judge
        return get_judge()
    except Exception:  # noqa: BLE001
        return None


def _judge_status():
    judge = _get_judge()
    return judge.status if judge else "disabled"


def _judge_begin(label):
    judge = _get_judge()
    if judge:
        judge.begin_video(label)


def _judge_finish(label, shot_assets, narration_for_shot, manifest, video_title, out_dir):
    """Post-render contact-sheet check + per-video judge summary (flags review like the gradient cap)."""
    judge = _get_judge()
    if judge is None:
        return {"judge_status": "disabled"}
    try:
        judge.begin_video(label)
        tiles, seen = [], set()
        for k, asset in enumerate(shot_assets):
            if not asset or asset.get("kind") != "image" or not _is_real(asset):
                continue  # contact sheets use stills only
            path = asset.get("path", "")
            if path in seen or not Path(path).exists():
                continue
            seen.add(path)
            tiles.append((path, narration_for_shot(k), asset))
        if judge.active and tiles:
            result = judge.check_contact_sheet([(p, t) for p, t, _ in tiles], video_title, out_dir, label)
            for bad in (result or {}).get("bad_shots", []):
                n = bad["n"] - 1
                if 0 <= n < len(tiles):
                    name = Path(tiles[n][0]).name
                    for entry in manifest:
                        if entry.get("file") == name:
                            entry["review_required"] = True
                            entry["review_reason"] = f"visual judge contact sheet: {bad.get('reason', '')}".strip()
            for sheet in Path(out_dir).glob(f"contact_sheet_{label}_*.jpg"):
                sheet.unlink(missing_ok=True)
        summary = judge.summary(label)
    except Exception as exc:  # noqa: BLE001 - the judge must never block the pipeline
        _log(f"{label}: visual judge summary failed ({exc}); continuing")
        return {"judge_status": _judge_status(), "error": str(exc)[:200]}
    if summary.get("needs_review"):
        _log(f"{label}: visual judge needs review: {'; '.join(summary['needs_review_reasons'])}")
    return summary


_MUSIC = {}


def background_music():
    """This job's Open Lo-Fi (CC0) music bed, fetched once per run; None = render without music."""
    if "record" not in _MUSIC:
        import os
        from src import open_lofi
        _MUSIC["record"] = open_lofi.fetch_track(os.environ.get("DYPOL_JOB_ID", "") or "dypol", log=_log)
    return _MUSIC["record"]


def _mix_audio(narration, out_wav, sfx_path=None):
    music = background_music()
    extra = {"sfx_path": str(sfx_path)} if sfx_path else {}
    if music:
        music = dict(music)
        try:
            from src import open_lofi
            music["applied_volume_db"] = open_lofi.relative_gain_db(
                ex.measure_lufs(narration), ex.measure_lufs(music["path"]), music["gain_db"])
        except Exception:  # noqa: BLE001 - measuring is best effort
            music["applied_volume_db"] = music["gain_db"]
        audio = ex.mix_music_and_normalize(narration, out_wav, music_path=music["path"],
                                           music_db=music["applied_volume_db"], **extra)
    else:
        audio = ex.mix_music_and_normalize(narration, out_wav, music_path=None, **extra)
    return audio, ({k: v for k, v in music.items() if k != "path"} if music else {"provider": "none"})


def _enforce_true_peak(out_path, label):
    """-1 dBTP ceiling on the final file (QC fails the draft above it); never blocks the render."""
    try:
        result = ex.enforce_true_peak(out_path, log=lambda m: _log(f"{label}: {m}"))
        if result.get("after") is not None:
            _log(f"{label}: true peak {result['after']:.2f} dBTP (limit {ex.MAX_TRUE_PEAK_DBTP:g})")
        return result
    except Exception as exc:  # noqa: BLE001 - QC still measures and flags it
        _log(f"{label}: true-peak check failed ({str(exc)[:120]})")
        return {}


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
        from src.visuals import ai_image_available
        return ai_image_available()
    except Exception:  # noqa: BLE001 - visuals deps missing in minimal test envs
        return False


def _assign(timeline, per_slide_assets, size, out_dir, label, fixed=None):
    """Resolve every timeline shot to an asset; returns (shot_assets, manifest).

    Fill order for shots beyond a slide's unique assets: own/AI assets, then the
    slide's real stock assets (reused non-consecutively), then real assets borrowed
    from other slides when a slide has none, and only then a gradient graphic.
    """
    from src.visuals import generated_graphic
    fixed = fixed or {}
    shot_assets = [None] * len(timeline["shots"])
    manifest = []
    _manifest_blocks(fixed, timeline, label, shot_assets, manifest)
    donors = [(i, a) for i, assets in enumerate(per_slide_assets) for a in assets if _is_real(a)]
    for sl in timeline["slides"]:
        assets = per_slide_assets[sl["index"]]
        if not any(_is_real(a) for a in assets) and donors:
            # Farthest slides first so the borrowed visual does not repeat right after its first use.
            ranked = sorted(donors, key=lambda d: (-abs(d[0] - sl["index"]), d[1].get("kind") == "video"))
            for _, donor in ranked[:2]:
                assets.append(dict(donor, borrowed=True))
        free = [g for g in sl["shots"] if g not in fixed]   # block shots are already drawn by a renderer
        plan = sb.assign_assets(len(free), assets, seed=sl["index"], allow_stock_reuse=True)
        usage = {}
        for local, global_index in enumerate(free):
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


RUN_CAP = {"chapter_card": 1, "count_up": 1, "kinetic_text": 1, "code_panel": 3, "diagram": 3, "icon_flow": 2}
SPAN_KINDS = ("code_panel", "diagram", "icon_flow")     # may run on across cuts: they animate throughout


def _max_block_span():
    from src import style
    return float(style.long_rules()["pacing"].get("max_block_span_s", 10.0))


def _groups(n_items, n_slots):
    """Split n_slots consecutive timeline shots into n_items near-equal contiguous groups."""
    edges = [round(k * n_slots / n_items) for k in range(n_items + 1)]
    return [list(range(edges[k], edges[k + 1])) for k in range(n_items)]


def opens_on_chapter(slide):
    from src import shotlist
    return (slide.get("shots") or [{}])[0].get("kind") == shotlist.CHAPTER


def plan_blocks(timeline, slides, out_dir=None, size=LONG_SIZE, label="long", words=None):
    """Map each slide's shot list onto timeline shots.

    Returns ({global_shot_index: block asset}, [kind per timeline shot]). A chapter_card takes
    the slide's first (short, 1.0-1.5 s) shot. A code_panel / diagram may span up to RUN_CAP
    consecutive timeline shots and max_block_span_s (one continuous animation: each shot carries
    the block's time offset, so the crossfade between them is invisible); other blocks take one
    shot. Leftover shots and stock_or_ai_still shots stay with the still sources until
    `_enforce_mix` converts excess stills, then `_govern_runs` breaks runs of 3.
    """
    from src import blocks, shotlist
    windows = ex.shot_windows(timeline)
    fixed, kinds = {}, [shotlist.STILL] * len(timeline["shots"])
    chapters = [i for i, s in enumerate(slides) if opens_on_chapter(s)]
    cap_s = _max_block_span()
    for sl, slide in zip(timeline["slides"], slides):
        shots = list(slide.get("shots") or [])
        tl = sl["shots"]
        if not shots or not tl:
            continue
        pairs = []
        if shots[0].get("kind") == shotlist.CHAPTER and len(tl) > 1:
            pairs.append((shots[0], [0]))
            shots, offset, rest = shots[1:], 1, len(tl) - 1
        else:
            offset, rest = 0, len(tl)
        if shots:
            if len(shots) >= rest:
                pairs += [(shots[int(j * len(shots) / rest)], [offset + j]) for j in range(rest)]
            else:
                pairs += [(shot, [offset + j for j in grp]) for shot, grp in zip(shots, _groups(len(shots), rest))]
        for g_index, (shot, locals_) in enumerate(pairs):
            kind = shot.get("kind")
            if kind not in blocks.SOURCES or not locals_:
                continue
            spec = dict(shot)
            if kind == shotlist.CHAPTER:
                spec["number"] = chapters.index(sl["index"]) + 1 if sl["index"] in chapters else 1
                spec["total"] = max(1, len(chapters))
            span_idx = [tl[locals_[0]]]
            for j in locals_[1:RUN_CAP.get(kind, 1)]:
                if windows[tl[j]][1] - windows[span_idx[0]][0] > cap_s:
                    break
                span_idx.append(tl[j])
            start = windows[span_idx[0]][0]
            span = windows[span_idx[-1]][1] - start
            preview = None
            if out_dir is not None:
                preview = Path(out_dir) / f"{label}_{sl['index']}_{g_index}_{kind}.png"
            try:
                base = blocks.block_asset(spec, preview, size=size, span=span)
            except Exception as exc:  # noqa: BLE001 - a bad block falls back to the still path
                _log(f"{label}: {kind} block for slide {sl['index'] + 1} failed ({exc}); using a still")
                continue
            base["block_id"] = f"{label}_{sl['index']}_{g_index}"
            base["block_span"] = span
            for g in span_idx:
                fixed[g] = dict(base, block_offset=round(windows[g][0] - start, 4))
                kinds[g] = kind
    try:
        _enforce_mix(timeline, slides, fixed, kinds, words, out_dir, size, label)
    except Exception as exc:  # noqa: BLE001 - the mix pass must never lose a render
        _log(f"{label}: shot-mix enforcement failed ({exc}); keeping the planned mix")
    _govern_runs(timeline, slides, fixed, kinds, out_dir, size, label, words=words)
    _govern_text(timeline, slides, fixed, kinds, out_dir, size, label, words=words)
    return fixed, kinds


# ---------------------------------------------------------------------------
# Shot mix: >= 40 % demo blocks (code_panel, diagram, count_up), stills <= 15 %
# ---------------------------------------------------------------------------

_NUM_TOKEN = re.compile(r"^([$€£₹]?)(\d[\d,]*(?:\.\d+)?)(%|x|k|m|b|bn|ms|gb|tb)?$", re.I)


def spoken_number(words):
    """First speakable number in a run of words -> count_up spec fields, else None (years skipped).

    Spelled-out numbers ("sixty-five percent") count too (src/numwords.py)."""
    from src.numwords import normalize
    words = [w for w in normalize(words) if w]
    for k, raw in enumerate(words):
        token = raw.strip(".,;:!?()\"'“”")
        m = _NUM_TOKEN.match(token)
        if not m:
            continue
        prefix, value, suffix = m.group(1), m.group(2).replace(",", ""), (m.group(3) or "")
        nxt = words[k + 1].strip(".,;:!?").lower() if k + 1 < len(words) else ""
        if not suffix and nxt in ("percent", "per-cent"):
            suffix = "%"
        try:
            number = float(value)
        except ValueError:
            continue
        if not suffix and not prefix and ("." not in value) and (1900 <= number <= 2100 or number < 3):
            continue
        rest = words[k + 1 + (1 if nxt in ("percent", "per-cent") and suffix == "%" else 0):]
        spec = {"kind": "count_up", "value": value, "prefix": prefix, "suffix": suffix.lower() if suffix != "%" else "%",
                "label": number_label(rest), "keywords": [], "fallback": "kinetic_text"}
        return spec
    return None


_LABEL_BREAK = {"means", "mean", "is", "are", "was", "were", "be", "that", "which", "who", "to", "and", "but", "so",
                "because", "if", "when", "while", "then", "than", "will", "can", "could", "would", "should",
                "it", "this", "they", "we", "you", "has", "have", "had", "do", "does", "did"}
_LABEL_LEAD = {"of", "the", "a", "an", "in", "on", "at", "for", "per", "out", "about", "over", "nearly", "almost",
               "roughly", "around", "more", "less"}


def number_label(rest, max_words=4):
    """What a spoken number counts, as a label that reads alone: the noun phrase right after it.

    "512 dimensions for every word ..." -> "Dimensions for every word"; "0.0 means complete" -> ""
    (a verb straight after the number: no label, the caller falls back to the slide title)."""
    out = []
    for raw in rest:
        tok = str(raw).strip("()\"'“”")
        bare = tok.strip(".,;:!?").lower()
        if not bare or bare in _LABEL_BREAK or re.match(r"^\d", bare):
            break
        out.append(tok.strip(".,;:!?"))
        if len(out) >= max_words or tok.endswith((".", ",", ";", ":", "!", "?")):
            break
    while out and out[-1].lower() in _KEY_STOP | _LABEL_LEAD:
        out.pop()
    if sum(w.lower() not in _KEY_STOP | _LABEL_LEAD for w in out) < 2:
        return ""
    label = " ".join(out)
    return label[:1].upper() + label[1:]


def _spoken(timeline, words, g):
    """Words (text only) said during timeline shot g, from the slide's word timings."""
    if not words:
        return []
    shot = timeline["shots"][g]
    sl = timeline["slides"][shot["slide"]]
    slide_words = words[shot["slide"]] if shot["slide"] < len(words) else []
    a, b = shot["start"] - sl["start"], shot["end"] - sl["start"]
    return [w for w, s, e in slide_words or [] if e > a + 0.05 and s < b - 0.05]


_KEY_STOP = {"a", "an", "the", "of", "to", "in", "on", "by", "or", "and", "is", "are", "be", "it", "as", "at",
             "that", "this", "with", "for", "from", "into", "if", "so", "but", "then", "than", "when", "while",
             "your", "you", "we", "our", "their", "they", "its", "it's", "was", "were", "has", "have", "can",
             "will", "just", "every", "each", "what", "which", "who", "how", "why", "where", "there",
             "should", "would", "could", "may", "might", "must", "do", "does", "did", "not", "no", "been",
             "being", "about", "more", "also", "very", "these", "those", "some", "any", "us", "me", "my"}


def key_phrase(words, max_words=4):
    """A 2-4 word callout from spoken words, never the clause verbatim (style rule 12).

    Picks the window of <= max_words that starts and ends on a content word and holds the
    most content words (ties: the later window, nearer the point of the sentence)."""
    toks, sentence, sid = [], [], 0
    for w in words:
        raw = str(w)
        t = raw.strip(".,;:!?()\"'“”")
        if t:
            toks.append(t)
            sentence.append(sid)
        if raw.rstrip("\"'”)").endswith((".", "?", "!")):
            sid += 1          # a callout never spans two sentences ("Next build Links")
    content = [t.casefold() not in _KEY_STOP and len(t) > 1 or t[:1].isdigit() for t in toks]
    best, best_score = [], (-1, -1, -1)
    for i in range(len(toks)):
        if not content[i]:
            continue
        for j in range(i, min(len(toks), i + max_words)):
            if sentence[j] != sentence[i]:
                break
            if not content[j]:
                continue
            n_content = sum(content[i:j + 1])
            if n_content < 2 and j > i:
                continue
            score = (n_content, -(j - i + 1 - n_content), i)
            if score > best_score:
                best, best_score = toks[i:j + 1], score
    if len(best) < 2:
        return ""
    phrase = " ".join(best)
    return phrase[:1].upper() + phrase[1:]


def _block_members(fixed, bid):
    return sorted(g for g, a in fixed.items() if a.get("block_id") == bid)


def _would_run(kinds, fixed, g, kind, bid):
    """True when setting shot g to (kind, bid) would create a run of 3 same-kind units."""
    from src import shotlist
    old_kind, old = kinds[g], fixed.get(g)
    lo, hi = max(0, g - 8), min(len(kinds), g + 9)

    def _runs_of(k):
        units = [u for u, _ in _units(kinds[lo:hi], {i - lo: fixed[i] for i in range(lo, hi) if i in fixed})]
        return [r for r in shotlist.runs_over(units, shotlist.max_run()) if r[1] == k]

    # Only a run of the NEW kind counts: an existing run of stills around g must not block
    # breaking it up (the first live draft kept a 7-still hook because of this).
    before = len(_runs_of(kind)) if old_kind != kind else 0
    kinds[g] = kind
    fixed[g] = dict(old or {}, block_id=bid)
    try:
        return len(_runs_of(kind)) > before
    finally:
        kinds[g] = old_kind
        if old is None:
            fixed.pop(g, None)
        else:
            fixed[g] = old


def _enforce_mix(timeline, slides, fixed, kinds, words, out_dir, size, label):
    """Convert excess stills (and, last, kinetic_text) into demo blocks or kinetic_text.

    Per still, in order: a count_up for a number spoken in that shot; extend the neighbouring
    code_panel / diagram over it (within max_block_span_s); a diagram from the slide's bullets
    (once per slide); a kinetic_text of the clause spoken in that shot (only while stills are
    over their cap). Nothing is converted into a run of three of a kind.
    """
    from src import blocks, shotlist, style
    variety = style.long_rules()["variety"]
    demo_kinds = set(variety["demo_kinds"])
    n = len(kinds)
    if not n:
        return
    windows = ex.shot_windows(timeline)
    cap_s = _max_block_span()
    shots = timeline["shots"]
    keep_slides = {0, len(slides) - 1} if len(slides) > 2 else set()   # hook + takeaway keep their own text
    used_numbers, diagram_slides = {}, set()

    def demo_share():
        return sum(k in demo_kinds for k in kinds) / n

    total_s = sum(b - a for a, b in windows) or 1.0

    def still_share():
        # The cap holds by count AND by screen time (stills tend to sit on the longer shots).
        by_time = sum(windows[g][1] - windows[g][0] for g in range(n) if kinds[g] == shotlist.STILL) / total_s
        return max(kinds.count(shotlist.STILL) / n, by_time)

    def done():
        return demo_share() >= variety["min_demo_share"] and still_share() <= variety["max_still_share"]

    def place(g, spec, tag):
        bid = f"{label}_{tag}_{g}"
        if _would_run(kinds, fixed, g, spec["kind"], bid):
            return False
        span = windows[g][1] - windows[g][0]
        preview = Path(out_dir) / f"{label}_{tag}_{g}.png" if out_dir is not None else None
        try:
            asset = blocks.block_asset(spec, preview, size=size, span=span)
        except Exception:  # noqa: BLE001
            return False
        fixed[g] = dict(asset, block_id=bid, block_span=span, block_offset=0.0)
        kinds[g] = spec["kind"]
        return True

    def extend(g):
        for nb in (g - 1, g + 1):
            if not 0 <= nb < n or shots[nb]["slide"] != shots[g]["slide"]:
                continue
            asset = fixed.get(nb)
            if not asset or asset.get("block") not in SPAN_KINDS:
                continue
            members = _block_members(fixed, asset["block_id"])
            new = sorted(members + [g])
            if new != list(range(new[0], new[-1] + 1)) or len(new) > RUN_CAP[asset["block"]] + 1:
                continue
            span = windows[new[-1]][1] - windows[new[0]][0]
            if span > cap_s:
                continue
            if _would_run(kinds, fixed, g, asset["block"], asset["block_id"]):
                continue
            base = {k: v for k, v in asset.items() if k != "block_offset"}
            base["block_span"] = round(span, 4)
            for m in new:
                fixed[m] = dict(base, block_offset=round(windows[m][0] - windows[new[0]][0], 4))
                kinds[m] = asset["block"]
            return True
        return False

    def to_demo(g):
        slide_i = shots[g]["slide"]
        spec = spoken_number(_spoken(timeline, words, g))
        if spec and len(spec["label"].split()) < 2 and slide_i < len(slides):
            spec["label"] = sb.clamp_bullet(slides[slide_i].get("title", ""), 4) or spec["label"]
        if spec and spec["value"] not in used_numbers.setdefault(slide_i, set()) and place(g, spec, "num"):
            used_numbers[slide_i].add(spec["value"])
            return True
        if extend(g):
            return True
        slide = slides[slide_i] if slide_i < len(slides) else {}
        bullets = [b for b in slide.get("bullets") or [] if b]
        title = slide.get("title", "")
        if slide_i == 0 and len(bullets) < 2 and len(slides) > 3:
            # The cold open has no bullets: preview the chapters ahead as a roadmap diagram.
            bullets = [s.get("title", "") for s in slides[1:-1] if s.get("title")][:4]
            title = "What's ahead"
        if slide_i not in diagram_slides and len(bullets) >= 2:
            nodes = [sb.clamp_bullet(b, 3) for b in bullets[:4]]
            nodes = [x for x in nodes if x]
            if len(nodes) >= 2 and place(g, {"kind": "diagram", "nodes": nodes, "title": sb.clamp_bullet(
                    title, 5), "keywords": [], "fallback": "kinetic_text"}, "dia"):
                diagram_slides.add(slide_i)
                return True
        return False

    def to_kinetic(g):
        said = _spoken(timeline, words, g)
        text = key_phrase(said) if len(said) >= 2 else ""
        if len(text.split()) < 2:
            slide = slides[shots[g]["slide"]] if shots[g]["slide"] < len(slides) else {}
            bullets = [b for b in slide.get("bullets") or [] if b]
            text = sb.clamp_bullet(bullets[g % len(bullets)], 6) if bullets else ""
        if not text:
            return False
        return place(g, {"kind": "kinetic_text", "text": text, "keywords": [], "fallback": shotlist.STILL}, "txt")

    # Hook: stock/AI stills never carry the cold open (style rule 1/5). Keep at most one,
    # converted regardless of the global shares, so the hook is never a still slideshow.
    hook_cap = int(variety.get("max_hook_stills", 1))
    hook = [g for g in range(n) if shots[g]["slide"] == 0 and kinds[g] == shotlist.STILL] if len(slides) > 1 else []
    for g in hook[:-hook_cap] if hook_cap > 0 else hook:
        if kinds[g] == shotlist.STILL and not to_demo(g):
            to_kinetic(g)

    if done():
        return

    def still_order():
        idx = [g for g in range(n) if kinds[g] == shotlist.STILL]
        # Break up runs of stills first, then spread conversions through the video.
        return sorted(idx, key=lambda g: (-sum(0 <= j < n and kinds[j] == shotlist.STILL for j in (g - 1, g + 1)),
                                          (g * 7919) % max(1, n)))

    for g in still_order():
        if done():
            break
        if kinds[g] != shotlist.STILL:
            continue
        if to_demo(g):
            continue
        if still_share() > variety["max_still_share"]:
            to_kinetic(g)
    if demo_share() < variety["min_demo_share"]:
        for g in range(n):
            if demo_share() >= variety["min_demo_share"]:
                break
            if kinds[g] == "kinetic_text" and shots[g]["slide"] not in keep_slides:
                old_kind, old = kinds[g], fixed.pop(g, None)
                kinds[g] = shotlist.STILL
                if not to_demo(g):
                    kinds[g] = old_kind
                    if old is not None:
                        fixed[g] = old
    _log(f"{label}: shot mix demo {demo_share():.0%} (>= {variety['min_demo_share']:.0%}), "
         f"stills {still_share():.0%} (<= {variety['max_still_share']:.0%})")


def _units(kinds, fixed):
    """Collapse consecutive shots of one block into a unit: [(kind, [shot indices])]."""
    units = []
    for i, kind in enumerate(kinds):
        bid = fixed.get(i, {}).get("block_id")
        if units and bid and units[-1][2] == bid:
            units[-1][1].append(i)
        else:
            units.append((kind, [i], bid))
    return [(k, idx) for k, idx, _ in units]


def _govern_runs(timeline, slides, fixed, kinds, out_dir, size, label, words=None):
    """Never 3 units of the same kind in a row: the 3rd still becomes a kinetic_text callout
    from the slide's bullets; a 3rd rendered block of the same kind falls back to a still."""
    from src import blocks, shotlist
    limit = shotlist.max_run()
    windows = ex.shot_windows(timeline)
    for _ in range(3):
        units = _units(kinds, fixed)
        violations = shotlist.runs_over([k for k, _ in units], limit)
        if not violations:
            return
        for start, kind, length in violations:
            for u in range(start + limit, start + length, limit + 1):
                idx = units[u][1]
                if kind == shotlist.STILL:
                    g = idx[0]
                    slide = slides[timeline["shots"][g]["slide"]]
                    bullets = [b for b in slide.get("bullets") or [] if b]
                    text = bullets[(g // 2) % len(bullets)] if bullets else ""
                    if not text:
                        text = key_phrase(_spoken(timeline, words, g))
                    if not text:
                        continue
                    spec = {"kind": "kinetic_text", "text": text, "keywords": [], "fallback": shotlist.STILL}
                    span = windows[g][1] - windows[g][0]
                    preview = Path(out_dir) / f"{label}_gov_{g}.png" if out_dir is not None else None
                    fixed[g] = dict(blocks.block_asset(spec, preview, size=size, span=span),
                                    block_id=f"{label}_gov_{g}", block_span=span, block_offset=0.0)
                    kinds[g] = "kinetic_text"
                else:
                    for g in idx:
                        fixed.pop(g, None)
                        kinds[g] = shotlist.STILL


def flow_spec(slide, said=(), text=""):
    """An icon_flow (visual) spec for a beat: nodes from the slide's bullets, else the spoken key phrase."""
    bullets = [sb.clamp_bullet(b, 2) for b in (slide.get("bullets") or []) if b]
    nodes = [b for b in bullets if b][:3]
    if len(nodes) < 2:
        phrase = key_phrase(list(said)) or text or slide.get("title", "")
        nodes = [w.strip(".,;:!?").capitalize() for w in str(phrase).split()
                 if w.strip(".,;:!?").casefold() not in _KEY_STOP and len(w) > 1][:3]
    if len(nodes) < 2:
        nodes = (nodes + [sb.clamp_bullet(slide.get("title", ""), 2) or "Input", "Result"])[:3]
    return {"kind": "icon_flow", "nodes": nodes, "title": sb.clamp_bullet(slide.get("title", ""), 4) or "How it flows",
            "keywords": [], "fallback": "kinetic_text"}


def _govern_text(timeline, slides, fixed, kinds, out_dir, size, label, words=None):
    """At most 2 text-dominant blocks (kinetic_text, chapter_card) in a row: the 3rd (or, when the
    3rd is a chapter card, the 2nd) kinetic_text becomes an icon_flow, a moving visual."""
    from src import blocks
    text = set(blocks.TEXT_DOMINANT)
    windows = ex.shot_windows(timeline)
    keep = {0, len(slides) - 1} if len(slides) > 2 else set()
    for _ in range(len(kinds) + 1):
        units = _units(kinds, fixed)
        run, bad = 0, None
        for u, (k, _) in enumerate(units):
            run = run + 1 if k in text else 0
            if run >= 3:
                bad = u
                break
        if bad is None:
            return
        cands = [u for u in (bad, bad - 1) if units[u][0] == "kinetic_text"]
        cands.sort(key=lambda u: timeline["shots"][units[u][1][0]]["slide"] in keep)
        if not cands:
            _log(f"{label}: 3 text blocks in a row at shot {units[bad][1][0]} (none convertible)")
            return
        idx = units[cands[0]][1]
        g = idx[0]
        slide_i = timeline["shots"][g]["slide"]
        slide = slides[slide_i] if slide_i < len(slides) else {}
        old_text = (fixed.get(g, {}).get("spec") or {}).get("text", "")
        spec = flow_spec(slide, _spoken(timeline, words, g), old_text)
        span = windows[idx[-1]][1] - windows[idx[0]][0]
        preview = Path(out_dir) / f"{label}_flow_{g}.png" if out_dir is not None else None
        try:
            asset = blocks.block_asset(spec, preview, size=size, span=span)
        except Exception as exc:  # noqa: BLE001
            _log(f"{label}: icon_flow for shot {g} failed ({exc})")
            return
        bid = f"{label}_flow_{g}"
        for m in idx:
            fixed[m] = dict(asset, block_id=bid, block_span=span, block_offset=round(windows[m][0] - windows[g][0], 4))
            kinds[m] = "icon_flow"


def text_runs(kinds, fixed):
    """Longest run of text-dominant units (style check: must be <= 2)."""
    from src import blocks
    best = run = 0
    for k, _ in _units(kinds, fixed):
        run = run + 1 if k in blocks.TEXT_DOMINANT else 0
        best = max(best, run)
    return best


def apply_backdrops(timeline, fixed, per_slide_assets, kinds_with_photo=("kinetic_text", "count_up", "chapter_card")):
    """Give text blocks a living photo backdrop: the slide's own topic still, blurred + darkened (Ken Burns)."""
    used = 0
    for sl in timeline["slides"]:
        imgs = [a for a in (per_slide_assets[sl["index"]] if sl["index"] < len(per_slide_assets) else [])
                if a.get("kind") == "image" and _is_real(a) and Path(a.get("path", "")).exists()]
        if not imgs:
            continue
        seen = set()
        for j, g in enumerate(g for g in sl["shots"] if g in fixed):
            asset = fixed[g]
            if asset.get("block") not in kinds_with_photo or asset.get("block_id") in seen:
                continue
            seen.add(asset.get("block_id"))
            pick = imgs[len(seen) % len(imgs)]
            asset.setdefault("spec", {})["backdrop_image"] = pick["path"]
            asset["backdrop_source_url"] = pick.get("source_url", "")
            used += 1
    return used


def _manifest_blocks(fixed, timeline, label, shot_assets, manifest):
    by_id = {}
    for g, asset in sorted(fixed.items()):
        shot_assets[g] = asset
        by_id.setdefault(asset["block_id"], []).append(g)
    for bid, shots in by_id.items():
        asset = fixed[shots[0]]
        entry = manifest_entry(asset, label, timeline["shots"][shots[0]]["slide"] + 1, shots, timeline)
        entry["kind"] = f"block:{asset['block']}"
        if asset.get("backdrop_source_url"):
            entry["backdrop_source_url"] = asset["backdrop_source_url"]
        manifest.append(entry)


def style_report(kinds, fixed, timeline=None):
    """Shares and run checks on the final shot plan (logged + saved in storyboard.json)."""
    from src import shotlist, style
    units = [k for k, _ in _units(kinds, fixed)]
    report = shotlist.shot_kinds_share(kinds)
    if timeline:
        demo = set(style.long_rules()["variety"]["demo_kinds"])
        lengths = [s["end"] - s["start"] for s in timeline["shots"]]
        total = max(1e-6, sum(lengths))
        report["demo_time_share"] = round(sum(d for d, k in zip(lengths, kinds) if k in demo) / total, 3)
        report["still_time_share"] = round(sum(d for d, k in zip(lengths, kinds) if k == shotlist.STILL) / total, 3)
        report["pacing"] = pacing.stats(timeline)
    report["kinds"] = {k: kinds.count(k) for k in sorted(set(kinds))}
    report["runs_over_limit"] = shotlist.runs_over(units, shotlist.max_run())
    report["max_text_run"] = text_runs(kinds, fixed)
    variety = style.long_rules()["variety"]
    report["demo_share_ok"] = report["demo_share"] >= variety["min_demo_share"]
    report["still_share_ok"] = report["still_share"] <= variety["max_still_share"]
    return report


def _sound_design(timeline, kinds, fixed, narration, out_dir, label="long"):
    """SFX track (soft procedural chime / whoosh / swell on block changes); (None, summary) on any failure."""
    try:
        from src import sfx
        events = sfx.plan(timeline, kinds, fixed, seed=len(timeline["shots"]))
        path = sfx.build_track(events, timeline["total"], Path(out_dir) / f"sfx_{label}.wav", narration)
        summary = {"enabled": bool(path), "hits": len(events),
                   "hits_per_min": sfx.per_minute(events, timeline["total"]),
                   "under_voice_db": sfx.audio_rules()["under_db"],
                   "clips": sorted({e["clip"] for e in events}), "license": "procedural (generated in src/sfx.py)"}
        _log(f"{label}: sfx {json.dumps(summary)}")
        return path, summary
    except Exception as exc:  # noqa: BLE001 - sound design never blocks a render
        _log(f"{label}: sfx skipped ({exc})")
        return None, {"enabled": False, "reason": str(exc)[:200]}


def _finish(words, ass_path, size, mode):
    """(encoder filter, captions burned in?) — grade + word-level ASS karaoke captions."""
    ass = None
    try:
        ass = caps.write_ass(words, ass_path, size, mode)
    except Exception as exc:  # noqa: BLE001
        _log(f"{mode}: caption script not written ({exc}); rendering without burned-in captions")
    vf = ex.finish_filter(ass)
    burned = bool(ass) and "subtitles=" in vf
    if ass and not burned:
        _log(f"{mode}: ffmpeg has no subtitles filter (libass); captions not burned in")
    return vf, burned


def narration_word_timings(text, path, speech=None):
    """([(word, start, end)], source) for one narration clip. The Kokoro timestamped model's
    word timings (src/voice.py sidecar, exact to the audio) when they exist for this text;
    else storyboard.estimate_word_timings (silence-region guess, ~0.2-0.3 s off)."""
    from src import voice
    speech = ex.audio_duration(path) if speech is None else speech
    words = voice.load_word_timings(path, text, speech)
    if words:
        return words, "kokoro"
    return sb.estimate_word_timings(text, speech, ex.detect_speech_regions(path, speech)), "estimate"


def _log_timing_sources(label, sources):
    est = sum(1 for s in sources if s != "kokoro")
    if est:
        _log(f"{label}: WORD_TIMING_FALLBACK {est}/{len(sources)} clip(s) have no Kokoro word timings; "
             f"using estimate_word_timings (approximate)")
    else:
        _log(f"{label}: word timings from Kokoro phoneme durations ({len(sources)} clip(s))")


def produce_long(slides, narration_paths, out_dir, out_path, gather, used, lesson_terms=(), video_title=None):
    """slides: normalized slide specs (title, content, bullets, visual_queries, image_prompt, kicker)."""
    video_title = video_title or (slides[0].get("title", "") if slides else "")
    _judge_begin("long")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    narration = out_dir / "narration_long.wav"
    durations = ex.build_narration(narration_paths, [SLIDE_PAD] * len(narration_paths), narration)
    # Word timings first: cuts land on narration clause boundaries (variable 2-4 s shots).
    slide_words, sources = [], []
    for slide, path in zip(slides, narration_paths):
        words, source = narration_word_timings(slide.get("content", ""), path)
        slide_words.append(words)
        sources.append(source)
    _log_timing_sources("long", sources)
    timeline = pacing.build_timeline(durations, slide_words, [opens_on_chapter(s) for s in slides])
    gaps = sb.coverage_gaps(timeline)
    if gaps:
        raise RuntimeError(f"Timeline has uncovered frames (would render black): {gaps[:5]}")
    _log(f"long: pacing {json.dumps(pacing.stats(timeline))}")

    fixed, kinds = plan_blocks(timeline, slides, out_dir, LONG_SIZE, "long", words=slide_words)
    per_slide_assets, bullet_times, caption_segments = [], [], []
    for sl, slide, path, words in zip(timeline["slides"], slides, narration_paths, slide_words):
        free_shots = len([g for g in sl["shots"] if g not in fixed])
        if not free_shots:
            # Every shot of this slide is a rendered block: no stock / AI request needed.
            per_slide_assets.append([])
            bullet_times.append(sb.bullet_reveal_times(slide.get("bullets", []), words, sl["end"] - sl["start"]))
            caption_segments.append((sl["start"], words))
            continue
        n_shots = free_shots
        terms = sb.topic_terms(slide.get("visual_queries", []), slide.get("bullets", []), *lesson_terms)
        spec = {"visual_queries": slide.get("visual_queries"), "image_prompt": slide.get("image_prompt"),
                "terms": terms, "video_title": video_title,
                "narration": slide.get("content") or slide.get("title", "")}
        # Without FLUX, ask stock for more unique assets per slide (up to 6) instead of placeholders.
        count = sb.unique_visual_count(n_shots) if _flux_available() else min(n_shots, 6)
        assets = gather(spec, out_dir, count, used, orientation="landscape",
                        label=f"long{sl['index'] + 1:02d}", seed=sl["index"], min_count=0)
        per_slide_assets.append(assets)
        bullet_times.append(sb.bullet_reveal_times(slide.get("bullets", []), words, sl["end"] - sl["start"]))
        caption_segments.append((sl["start"], words))
    photo_backdrops = apply_backdrops(timeline, fixed, per_slide_assets)
    _log(f"long: {photo_backdrops} text blocks on a blurred topic-still backdrop; the rest on the drifting field")
    shot_assets, manifest = _assign(timeline, per_slide_assets, LONG_SIZE, out_dir, "long", fixed=fixed)
    style_check = style_report(kinds, fixed, timeline)
    _log(f"long: style check {json.dumps(style_check)}")

    all_words = caps.segments_to_words(caption_segments)
    sfx_path, sfx_summary = _sound_design(timeline, kinds, fixed, narration, out_dir)
    audio, music = _mix_audio(narration, out_dir / "audio_long_final.wav", sfx_path=sfx_path)
    vf, burned = _finish(all_words, out_dir / "captions_long.ass", LONG_SIZE, "long")
    overlay = ex.long_overlay(timeline, slides, bullet_times, LONG_SIZE,
                              hide_windows=[(timeline["shots"][g]["start"], timeline["shots"][g]["end"])
                                            for g in sorted(fixed)], captions=burned)
    _log(f"rendering long video: {timeline['total']:.1f}s, {len(timeline['shots'])} shots")
    ex.render(timeline, shot_assets, audio, out_path, LONG_SIZE, overlay, progress=_log, vf=vf)
    true_peak = _enforce_true_peak(out_path, "long video")
    judge_summary = _judge_finish(
        "long", shot_assets,
        lambda k: (slides[timeline["shots"][k]["slide"]].get("content")
                   or slides[timeline["shots"][k]["slide"]].get("title", "")),
        manifest, video_title, out_dir)
    storyboard = {"timeline": timeline, "bullet_times": bullet_times, "shot_kinds": kinds,
                  "style_check": style_check, "sfx": sfx_summary, "captions_burned_in": burned,
                  "slides": [{k: s.get(k) for k in ("title", "bullets", "visual_queries", "shots")} for s in slides]}
    (out_dir / "storyboard.json").write_text(json.dumps(storyboard, indent=2), encoding="utf-8")
    captions_path = _write_captions(all_words, out_dir / "captions_long.srt")
    _cleanup(narration, audio, shot_assets)
    if sfx_path:
        Path(sfx_path).unlink(missing_ok=True)
    return {"manifest": manifest, "timeline": timeline, "judge": judge_summary, "captions_path": captions_path,
            "true_peak": true_peak, "music": music, "sfx": sfx_summary, "style_check": style_check,
            "own_assets": [a for assets in per_slide_assets for a in assets
                           if a.get("reusable") and _is_real(a)],
            # Stock stills survive cleanup; the Short may reuse them before any gradient.
            "stock_assets": [a for assets in per_slide_assets for a in assets
                             if _is_real(a) and not a.get("reusable") and a.get("kind") == "image"
                             and not a.get("borrowed")]}


def fit_short_audio(tts_fn, script, cta, work_dir, lo=None, hi=None, expand_fn=None):
    """TTS the Short and bring it into [lo, hi] s (default 50-65 s): silence trimmed; too long ->
    tempo up to 25 % faster or sentences cut once. Too short -> ONE script rewrite with more
    content (`expand_fn(script, extra_words) -> longer script`), re-voiced; only what is still
    missing may be closed by slowing, and the voice is never slowed below 150 WPM (owner rule,
    4 Nov 2026: the old x0.90 slow-down took it to ~125 WPM). `cta` is only appended when given
    (the DYPOL Short has no CTA)."""
    from src import short_rules as sr
    lo = SHORT_MIN if lo is None else lo
    hi = SHORT_MAX if hi is None else hi
    work_dir = Path(work_dir)
    raw = tts_fn(script, work_dir / "short_voice.mp3")
    voice = ex.trim_silence(raw, work_dir / "short_voice_trim.wav")
    dur = ex.audio_duration(voice)
    if dur < lo and cta and cta not in script:
        script = f"{script} {cta}"
        raw = tts_fn(script, work_dir / "short_voice_cta.mp3")
        voice = ex.trim_silence(raw, work_dir / "short_voice_cta_trim.wav")
        dur = ex.audio_duration(voice)
    if dur < lo and expand_fn is not None:
        words = len(script.split())
        extra = sr.words_needed(words, dur, lo + 1.0)
        try:
            longer = " ".join(str(expand_fn(script, extra) or "").split())
        except Exception as exc:  # noqa: BLE001 - the first script still makes a Short
            longer = ""
            _log(f"short: content rewrite failed ({str(exc)[:120]})")
        if len(longer.split()) > words:
            raw2 = tts_fn(longer, work_dir / "short_voice_more.mp3")
            voice2 = ex.trim_silence(raw2, work_dir / "short_voice_more_trim.wav")
            dur2 = ex.audio_duration(voice2)
            if dur2 > dur and dur2 <= hi + 8.0:
                _log(f"short: {dur:.1f}s < {lo:.0f}s; rewrite with more content -> {dur2:.1f}s "
                     f"({words} -> {len(longer.split())} words)")
                script, voice, dur = longer, voice2, dur2
            else:
                _log(f"short: rewrite gave {dur2:.1f}s; keeping the first script")
        else:
            _log("short: rewrite was not longer; keeping the first script")
    if dur < lo:
        words = len(script.split())
        factor = sr.slow_factor(words, dur, lo + 0.3)
        if factor < 0.995:
            voice = ex.change_tempo(voice, work_dir / "short_voice_slow.wav", factor)
            _log(f"short: voice {dur:.1f}s < {lo:.0f}s; slowed x{factor:.3f} "
                 f"({sr.wpm(words, dur):.0f} -> {sr.wpm(words, dur / factor):.0f} WPM, floor {sr.MIN_WPM:.0f})")
            dur = dur / factor
        if dur < lo - 0.05:
            _log(f"short: {dur:.1f}s stays under {lo:.0f}s; the voice is not slowed below {sr.MIN_WPM:.0f} WPM")
    elif dur > hi:
        factor = dur / (hi - 1.0)
        if factor <= 1.25:
            voice = ex.change_tempo(voice, work_dir / "short_voice_fast.wav", factor)
        else:
            # Far too long for tempo alone: drop the last sentences once, re-voice, then cap tempo.
            sentences = [x for x in script.split(". ") if x]
            keep = max(1, int(len(sentences) * (hi - 3.0) / dur))
            script = ". ".join(sentences[:keep]).rstrip(".") + "."
            raw = tts_fn(script, work_dir / "short_voice_cut.mp3")
            voice = ex.trim_silence(raw, work_dir / "short_voice_cut_trim.wav")
            dur = ex.audio_duration(voice)
            if dur > hi:
                voice = ex.change_tempo(voice, work_dir / "short_voice_fast.wav", min(1.25, dur / (hi - 1.0)))
    return script, voice


_SENT = re.compile(r"(?<=[.!?])\s+")
_CATCH = re.compile(r"\b(but|however|catch|careful|limit|fails?|wrong|halluc|mistake|risk)", re.I)


def short_sections(plan, script):
    """[{role, text, nodes, icons}] covering the script in order: hook, 2-3 beats, catch, takeaway.

    Built from the model's structured plan when its texts add up to the script; otherwise split
    from the script's sentences (first = hook, last = takeaway, the one with "but/catch" = catch)."""
    plan = plan if isinstance(plan, dict) else {}
    secs = []
    if plan.get("hook"):
        secs.append({"role": "hook", "text": str(plan["hook"])})
    for b in plan.get("beats") or []:
        if isinstance(b, dict) and b.get("text"):
            secs.append({"role": "beat", "text": str(b["text"]), "nodes": list(b.get("nodes") or []),
                         "icons": list(b.get("icons") or ([b["icon"]] if b.get("icon") else []))})
    c = plan.get("catch")
    if isinstance(c, dict) and c.get("text"):
        secs.append({"role": "catch", "text": str(c["text"]), "nodes": list(c.get("nodes") or []),
                     "icons": list(c.get("icons") or ([c["icon"]] if c.get("icon") else []))})
    elif isinstance(c, str) and c:
        secs.append({"role": "catch", "text": c})
    if plan.get("takeaway"):
        secs.append({"role": "takeaway", "text": str(plan["takeaway"])})
    norm = lambda x: re.sub(r"[^a-z0-9]+", " ", x.lower()).strip()
    if len(secs) >= 4 and norm(" ".join(x["text"] for x in secs)) == norm(script):
        return secs
    sents = [x for x in _SENT.split(" ".join(str(script).split())) if x]
    if len(sents) < 3:
        return [{"role": "hook", "text": sents[0] if sents else script}] + \
               ([{"role": "beat", "text": " ".join(sents[1:])}] if len(sents) > 1 else [])
    hook, take, body = sents[0], sents[-1], sents[1:-1]
    ci = next((i for i in range(len(body) - 1, -1, -1) if _CATCH.search(body[i])), None)
    catch = body.pop(ci) if ci is not None and len(body) > 2 else None
    beats_n = 3 if len(body) >= 6 else 2 if len(body) >= 2 else 1
    groups = _groups(beats_n, len(body)) if body else []
    out = [{"role": "hook", "text": hook}]
    pbeats = [b for b in plan.get("beats") or [] if isinstance(b, dict)]
    for k, grp in enumerate(groups):
        nodes = list(pbeats[k].get("nodes") or []) if k < len(pbeats) else []
        out.append({"role": "beat", "text": " ".join(body[i] for i in grp), "nodes": nodes})
    if catch:
        out.append({"role": "catch", "text": catch})
    out.append({"role": "takeaway", "text": take})
    return out


def _find_number(words, value):
    """Index of the spoken word carrying `value` (digits only compare: '1,000' == '1000'), else None."""
    want = re.sub(r"[^\d.]", "", str(value)).rstrip(".")
    if not want:
        return None
    from src.numwords import normalize
    for i, w in enumerate(normalize([w for w, _, _ in words])):   # index-preserving
        if re.sub(r"[^\d.]", "", str(w)).rstrip(".") == want:
            return i
    return None


def plan_short(timeline, words, sections, number=None, photos=(), out_dir=None, size=SHORT_SIZE, n_stills=None):
    """Shot plan for the Short: ({shot: block asset}, kinds). Motion blocks carry it; at most a few
    full-frame stills (when photos exist); text blocks sit on a blurred topic still or the field."""
    from src import blocks, shotlist
    shots = timeline["shots"]
    n = len(shots)
    windows = ex.shot_windows(timeline)
    cap_s = _max_block_span()
    # word index -> section
    bounds, k = [], 0
    for sec in sections:
        cnt = len(sec["text"].split())
        bounds.append((k, k + cnt))
        k += cnt
    def sec_of_time(t):
        # A section runs until the next section's first word starts (word starts drive cues).
        for si, (a, b) in enumerate(bounds):
            if a < len(words) and (b >= len(words) or t < words[b][1]):
                return si
        return len(sections) - 1
    shot_sec = [sec_of_time((s["start"] + s["end"]) / 2) for s in shots]
    photos = [p for p in photos if p]
    n_stills = len(photos) if n_stills is None else n_stills
    still_budget = min(n_stills, max(1, int(round(n * 0.18)))) if n_stills else 0
    beat_no = 0
    plan = [None] * n          # (kind, spec, group_id)
    for si, sec in enumerate(sections):
        idx = [g for g in range(n) if shot_sec[g] == si]
        if not idx:
            continue
        a, b = bounds[si]
        said = [w for w, _, _ in words[a:b]]
        phrase = key_phrase(said) or sb.clamp_bullet(sec["text"], 5)
        nodes = [sb.clamp_bullet(x, 2) for x in sec.get("nodes") or [] if x][:3]
        slide = {"title": sec.get("title") or phrase, "bullets": nodes}
        role = sec["role"]
        if role == "hook":
            units = [("kinetic_text", {"text": phrase, "label": "Quick explainer"}, 1),
                     (shotlist.STILL, {}, 1),
                     ("icon_flow", dict(flow_spec(slide, said, phrase), title="In 60 seconds"), 2)]
        elif role == "beat":
            beat_no += 1
            dia = {"kind": "diagram", "nodes": nodes if len(nodes) >= 2 else flow_spec(slide, said)["nodes"],
                   "icons": sec.get("icons") or [], "title": f"Step {beat_no}"}
            flow = dict(flow_spec(slide, said, phrase), title=f"Step {beat_no}", icons=sec.get("icons") or [])
            kin = {"text": phrase, "label": f"Step {beat_no}"}
            units = [("diagram", dia, 3), ("kinetic_text", kin, 1), (shotlist.STILL, {}, 1), ("icon_flow", flow, 2)] \
                if beat_no % 2 else \
                [("icon_flow", flow, 2), ("kinetic_text", kin, 1), ("diagram", dia, 3), (shotlist.STILL, {}, 1)]
        elif role == "catch":
            flow = dict(flow_spec(slide, said, phrase), title="The catch", icons=["warn"] * 3)
            units = [("kinetic_text", {"text": phrase, "label": "The catch", "icon": "warn"}, 1),
                     ("icon_flow", flow, 2), (shotlist.STILL, {}, 1)]
        else:
            units = [("kinetic_text", {"text": phrase, "label": "Takeaway", "icon": "check"}, 1),
                     ("icon_flow", dict(flow_spec(slide, said, phrase), title="Remember"), 2)]
        u, j = 0, 0
        while j < len(idx):
            kind, spec, span_n = units[u % len(units)]
            u += 1
            if kind == shotlist.STILL:
                if still_budget <= 0:
                    continue
                still_budget -= 1
                plan[idx[j]] = (kind, None, None)
                j += 1
                continue
            grp = [idx[j]]
            while len(grp) < span_n and j + len(grp) < len(idx) and \
                    windows[idx[j + len(grp)]][1] - windows[grp[0]][0] <= cap_s:
                grp.append(idx[j + len(grp)])
            gid = f"short_{si}_{u}"
            for g in grp:
                plan[g] = (kind, dict(spec, kind=kind), gid)
            j += len(grp)
    # The count_up: a sourced number at the shot it is spoken (else the hook's first shot).
    if number:
        at = _find_number(words, number.get("value"))
        g = 0
        if at is not None:
            t = words[at][1]            # the shot on screen when the number starts being spoken
            g = next((i for i, s in enumerate(shots) if s["start"] - 1e-6 <= t < s["end"]), 0)
        old = plan[g]
        if old and old[2]:
            for m in range(n):          # the unit it replaces shrinks to its other shots
                if m != g and plan[m] and plan[m][2] == old[2]:
                    plan[m] = (old[0], old[1], old[2] + ("b" if m < g else "c"))
        plan[g] = ("count_up", dict(number, kind="count_up", title="By the numbers"), f"num_{g}")
    fixed, kinds = {}, [shotlist.STILL] * n
    photo_i = 0
    g = 0
    while g < n:
        item = plan[g]
        if not item or item[0] == shotlist.STILL:
            g += 1
            continue
        kind, spec, gid = item
        grp = [g]
        while grp[-1] + 1 < n and plan[grp[-1] + 1] and plan[grp[-1] + 1][2] == gid:
            grp.append(grp[-1] + 1)
        if kind in ("kinetic_text", "count_up") and photos:
            spec = dict(spec, backdrop_image=photos[photo_i % len(photos)])
            photo_i += 1
        span = windows[grp[-1]][1] - windows[grp[0]][0]
        preview = Path(out_dir) / f"short_{g}_{kind}.png" if out_dir is not None else None
        try:
            asset = blocks.block_asset(spec, preview, size=size, span=span)
        except Exception as exc:  # noqa: BLE001 - a bad block falls back to a still
            _log(f"short: {kind} block at shot {g} failed ({exc}); using a still")
            g = grp[-1] + 1
            continue
        for m in grp:
            fixed[m] = dict(asset, block_id=f"short_{gid}", block_span=round(span, 4),
                            block_offset=round(windows[m][0] - windows[grp[0]][0], 4))
            kinds[m] = kind
        g = grp[-1] + 1
    slides = [{"title": sections[0]["text"] if sections else "", "bullets": []}]
    _govern_runs(timeline, slides, fixed, kinds, out_dir, size, "short", words=[words])
    _govern_text(timeline, slides, fixed, kinds, out_dir, size, "short", words=[words])
    return fixed, kinds


def short_number(number, words):
    """The Short's count_up spec: the model's number only with a named source (never invented);
    else a number actually spoken in the script (shown without a source line); else None."""
    from src.shotlist import parse_number
    if isinstance(number, dict) and str(number.get("source") or "").strip() and parse_number(number.get("value")):
        label = sb.clamp_bullet(str(number.get("label") or ""), 5)
        return {"value": str(number["value"]), "prefix": str(number.get("prefix") or ""),
                "suffix": str(number.get("suffix") or ""), "label": label,
                "source": sb.clamp_bullet(str(number["source"]), 8), "sourced": True}
    spec = spoken_number([w for w, _, _ in words])
    if spec:
        spec.pop("fallback", None)
        spec.pop("keywords", None)
        spec["sourced"] = False
        return spec
    return None


def produce_short(script, voice_path, out_dir, out_path, gather, used, spec, own_assets=(),
                  fallback_assets=(), plan=None):
    """The DYPOL Short (1080x1920, 50-65 s): motion blocks (diagram, icon_flow, count_up, kinetic_text)
    on living backgrounds, a few topic stills, word-level karaoke captions, SFX on block changes."""
    from src import shotlist
    _judge_begin("short")
    spec = dict(spec or {})
    spec.setdefault("narration", script)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    narration = out_dir / "narration_short.wav"
    speech = ex.audio_duration(voice_path)
    durations = ex.build_narration([voice_path], [0.6], narration)
    total = durations[0]
    words, source = narration_word_timings(script, voice_path, speech)
    _log_timing_sources("short", [source])
    timeline = pacing.build_timeline([total], [words], [False])
    if sb.coverage_gaps(timeline):
        raise RuntimeError("Short timeline has uncovered frames")
    n_shots = len(timeline["shots"])
    sections = short_sections(plan, script)
    number = short_number((plan or {}).get("number") if isinstance(plan, dict) else None, words)
    if number is None:
        _log("short: no sourced or spoken number; no count_up (never invented)")
    elif not number.get("sourced"):
        _log(f"short: count_up uses the spoken number {number['value']} (the model gave no source)")
    number = {k: v for k, v in (number or {}).items() if k != "sourced"} or None
    # Topic stills (portrait): a few full-frame shots + blurred backdrops for the text blocks.
    count = min(4, n_shots)
    fresh = gather(spec, out_dir, count, used, orientation="portrait", label="short", seed=97, min_count=0)
    pool = list(fresh)
    for asset in own_assets:
        if len(pool) >= count:
            break
        if _is_real(asset):
            pool.append(dict(asset))
    for asset in fallback_assets:
        if len(pool) >= count:
            break
        if _is_real(asset) and Path(asset.get("path", "")).exists() and \
                asset.get("sha256") not in {a.get("sha256") for a in pool}:
            pool.append(dict(asset, borrowed=True))
    photos = [a["path"] for a in pool if a.get("kind") == "image" and _is_real(a) and Path(a.get("path", "")).exists()]
    fixed, kinds = plan_short(timeline, words, sections, number, photos=photos, out_dir=out_dir,
                               n_stills=len(pool))
    still_shots = [g for g in range(n_shots) if g not in fixed]
    if still_shots and not pool:
        # No topic visuals at all: those shots become icon_flows rather than placeholder art.
        from src import blocks
        windows = ex.shot_windows(timeline)
        for g in still_shots:
            fs = flow_spec({"title": sections[0]["text"] if sections else ""}, [w for w, _, _ in words])
            fixed[g] = dict(blocks.block_asset(fs, None, size=SHORT_SIZE, span=windows[g][1] - windows[g][0]),
                            block_id=f"short_fill_{g}", block_span=windows[g][1] - windows[g][0], block_offset=0.0)
            kinds[g] = "icon_flow"
        still_shots = []
    shot_assets = [None] * n_shots
    manifest = []
    _manifest_blocks(fixed, timeline, "short", shot_assets, manifest)
    plan_assets = sb.assign_assets(len(still_shots), pool, seed=3, allow_stock_reuse=True)
    usage = {}
    from src.visuals import generated_graphic
    for k, g in enumerate(still_shots):
        pick = plan_assets[k] if k < len(plan_assets) else None
        if pick is None:
            pool.append(generated_graphic(out_dir / f"short_fill_{k}.png", SHORT_SIZE, seed=200 + k))
            pick = len(pool) - 1
        shot_assets[g] = pool[pick]
        usage.setdefault(pick, []).append(g)
    manifest += [manifest_entry(pool[p], "short", 1, shots, timeline) for p, shots in usage.items()]
    apply_graphic_cap(shot_assets, manifest, "short")
    style_check = style_report(kinds, fixed, timeline)
    style_check["sections"] = [s["role"] for s in sections]
    style_check["count_up"] = number
    _log(f"short: style check {json.dumps(style_check)}")

    sfx_path, sfx_summary = _sound_design(timeline, kinds, fixed, narration, out_dir, label="short")
    audio, music = _mix_audio(narration, out_dir / "audio_short_final.wav", sfx_path=sfx_path)
    vf, burned = _finish(words, out_dir / "captions_short.ass", SHORT_SIZE, "short")
    overlay = ex.short_overlay(words, SHORT_SIZE, captions=not burned)
    _log(f"rendering Short: {total:.1f}s, {n_shots} shots")
    ex.render(timeline, shot_assets, audio, out_path, SHORT_SIZE, overlay, vf=vf)
    _enforce_true_peak(out_path, "short")

    def _short_line(k):
        shot = timeline["shots"][k]
        said = [w for w, a, b in words if b > shot["start"] and a < shot["end"]]
        return " ".join(said) or script
    judge_summary = _judge_finish("short", shot_assets, _short_line, manifest,
                                  spec.get("video_title", ""), out_dir)
    (out_dir / "captions.json").write_text(json.dumps(words, indent=1), encoding="utf-8")
    (out_dir / "storyboard.json").write_text(json.dumps({"timeline": timeline, "shot_kinds": kinds,
                                                         "style_check": style_check, "sfx": sfx_summary,
                                                         "sections": sections}, indent=2), encoding="utf-8")
    captions_path = _write_captions(words, out_dir / "captions_short.srt")
    _cleanup(narration, audio, shot_assets)
    if sfx_path:
        Path(sfx_path).unlink(missing_ok=True)
    for leftover in out_dir.glob("short_voice*"):
        leftover.unlink(missing_ok=True)
    return {"manifest": manifest, "timeline": timeline, "words": words, "judge": judge_summary,
            "captions_path": captions_path, "music": music, "style_check": style_check, "sfx": sfx_summary}
