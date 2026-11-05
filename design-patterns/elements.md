# Design pattern: elements (Elements & animation) — status: ready

The current Engine A look. Source of truth: style/style.guide.yaml (`look`, `blocks`, `long`) and src/blocks.py.

- Canvas: 1920x1080 at 30 fps for the long video; the Short is a 9:16 vertical cut of the same system.
- Palette: deep slate background (#0F141C), panels #161D28 with #26303D borders, off-white body text (#EEF2F6),
  muted labels (#8A97A8); accents lime #C8F560 (the point / winner), cyan #4CC9F0 (the product / system),
  amber #FFB547 (the caveat / catch).
- Motion: CPU-rendered Pillow blocks with eased in/out motion over an animated particle backdrop (src/backdrop.py).
- Block kinds: kinetic_text, count_up, code_panel (typed at ~32 chars/s, max 10 lines), diagram (2-5 nodes),
  chapter_card, icon_flow; stock_or_ai_still (stock first, then FLUX) is the fallback and capped at 15% of shots.
- Pacing: a screen change every 2-4 s (target ~3 s, cut on narration clause boundaries), never more than 6 s static;
  code panels and diagrams may run up to 10 s because they animate throughout.
- Variety: never 3 blocks of the same kind in a row, at least 40% demo-style blocks, at most 2 text-heavy blocks in a row.
- Hook: no intro or logo card; first word by 0.5 s, stakes by 3 s, a number by 8 s, opening on a proof block.
- Text: at most 25 words on screen, numbers at least 120 px, 6% safe margin; word-timed captions.
- Chapters every 60-120 s with short chapter-card stingers; ends with one takeaway and exactly one comment question.
- Sound: Kokoro narration at 150-165 WPM, a quiet CC0 lo-fi music bed (about -26 dB) and whooshes/pops ~12 dB under the voice.
