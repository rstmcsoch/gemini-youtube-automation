"""Piece 3 (first live draft review): hook stills, key-phrase callouts, Shorts title, tags."""
import main
from src import production


def test_key_phrase_is_short_and_not_verbatim():
    words = "When user B watches a 6th sci-fi video the system".split()
    phrase = production.key_phrase(words)
    assert 2 <= len(phrase.split()) <= 4
    assert phrase.casefold() != " ".join(words[:6]).casefold()
    assert phrase.split()[-1].casefold() not in production._KEY_STOP


def test_key_phrase_needs_two_words():
    assert production.key_phrase(["the"]) == ""
    assert production.key_phrase([]) == ""


def test_shorts_title_cuts_on_word_boundary():
    long = "Discover how recommendation engines process user behavior with matrix factorization, vector embeddings"
    t = main.shorts_title(long)
    assert t.endswith(" #Shorts") and len(t) <= 100
    body = t[: -len(" #Shorts")]
    assert long.startswith(body) and (len(body) == len(long) or long[len(body)] == " ")


def test_title_tags_drop_function_words():
    tags = main.title_tags("How AI recommends what you watch next")
    assert tags == ["AI", "Recommends", "Watch"]


def _timeline(slide_shots):
    shots, slides, t = [], [], 0.0
    for si, n in enumerate(slide_shots):
        idx = []
        start = t
        for j in range(n):
            shots.append({"slide": si, "shot": j, "start": t, "end": t + 3.0})
            idx.append(len(shots) - 1)
            t += 3.0
        slides.append({"index": si, "start": start, "end": t, "shots": idx})
    return {"total": t, "slides": slides, "shots": shots}


def test_hook_stills_are_converted(tmp_path):
    slides = [{"title": "Hook", "bullets": [], "shots": [
        {"kind": "count_up", "value": "80", "suffix": "%", "label": "of watch time", "keywords": [],
         "fallback": "kinetic_text"}]}]
    slides += [{"title": f"Chapter {i}", "bullets": ["Users map to items", "Scores fill gaps"],
                "shots": [{"kind": "chapter_card", "title": f"Chapter {i}", "keywords": [], "fallback": "kinetic_text"},
                          {"kind": "code_panel", "code": "x = 1\ny = 2", "title": "a.py", "keywords": [],
                           "fallback": "diagram"},
                          {"kind": "diagram", "nodes": ["A", "B", "C"], "title": "Flow", "keywords": [],
                           "fallback": "code_panel"}]} for i in range(1, 5)]
    slides.append({"title": "The takeaway", "bullets": [], "shots": [
        {"kind": "kinetic_text", "text": "Vectors drive it", "keywords": [], "fallback": "stock_or_ai_still"}]})
    tl = _timeline([8] + [6] * 4 + [2])
    said = "recommendation algorithms drive eighty percent of watch time across modern video platforms today".split()
    words = [[(w, i * 0.4, i * 0.4 + 0.35) for i, w in enumerate(said * 4)]] + [[] for _ in slides[1:]]
    fixed, kinds = production.plan_blocks(tl, slides, out_dir=tmp_path, size=(640, 360), label="t", words=words)
    hook = [kinds[g] for g in tl["slides"][0]["shots"]]
    assert hook.count("stock_or_ai_still") <= 2, hook          # was 7 of 8 before the fix
    assert "stock_or_ai_still" not in hook[:4], hook           # the cold open is never stills
    assert kinds[0] == "count_up"
    assert any(fixed.get(g, {}).get("block") == "diagram" for g in tl["slides"][0]["shots"])  # roadmap


def test_key_phrase_never_spans_sentences():
    phrase = production.key_phrase("should we test in our next build? Links are in the description.".split())
    assert "Links" not in phrase.split() or "build" not in phrase.split()
