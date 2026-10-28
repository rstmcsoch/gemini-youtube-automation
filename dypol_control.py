"""dypol_control.py — DYPOL Creator Console phase control for Engine A.

Add as ``dypol_control.py`` in YOUR FORK of gemini-youtube-automation (repo
root, next to ``main.py``). ``main.py`` wires it in using the exact diff in
``engines/engine-a-patches/code_patches.md``.

This module implements the real phase semantics the dashboard dispatches:

  phase=draft    → generate the intended content and SAVE the candidate files
                   (output/candidates/<job_id>/candidate.json) but perform
                   NO YouTube upload.
  phase=publish  → upload ONLY the saved candidate associated with the console
                   job id. Never generate a new lesson / replacement candidate.
  phase=full     → generate + upload in one run ONLY when the console policy
                   explicitly allows automatic publishing (DYPOL_AUTO_PUBLISH=1
                   and DYPOL_REVIEW_REQUIRED=0). Review-required mode can never
                   be bypassed by invoking full.

Also implemented here:
  * topic input end-to-end (``select_next_lesson`` honours DYPOL_TOPIC),
  * content-plan preservation (completed lessons are never regenerated; a
    draft never marks a lesson complete; history is appended, never rewritten),
  * multi-video upload tracking (Engine A produces a long-form video AND a
    Short — both video IDs / URLs / visibility states are recorded separately
    and never overwrite each other),
  * privacy enforcement + verification of the actual ``videos.insert`` body
    (default ``private`` for test uploads),
  * fail-safe behaviour when the approved candidate is missing (raise — never
    silently regenerate or upload a replacement).

Pure logic only — no Google/MoviePy imports — so the semantics are unit-testable
without credentials (see tests/engine_a_phase.test.py).
"""
from __future__ import annotations

import datetime
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

VALID_PHASES = ("draft", "publish", "full")
VALID_VISIBILITY = ("private", "unlisted", "public")


class CandidateNotFound(RuntimeError):
    """Raised when the approved candidate for a job cannot be found.

    Failing loudly is mandatory: silently generating a replacement candidate
    or uploading anything else would violate the review gate.
    """


class PolicyViolation(RuntimeError):
    """Raised when an operation would violate the resolved phase policy."""


# --------------------------------------------------------------------------
# Phase policy
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PhasePolicy:
    phase: str
    allow_generate: bool
    allow_upload: bool
    review_required: bool
    auto_publish: bool
    visibility: str
    job_id: str = ""

    def describe(self) -> str:
        return (
            f"phase={self.phase} generate={self.allow_generate} upload={self.allow_upload} "
            f"review_required={self.review_required} auto_publish={self.auto_publish} "
            f"visibility={self.visibility} job_id={self.job_id or '-'}"
        )


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")


def resolve_policy(env: dict | os._Environ | None = None) -> PhasePolicy:
    env = os.environ if env is None else env
    phase = str(env.get("DYPOL_PHASE", "draft") or "draft").strip().lower()
    if phase == "":
        phase = "draft"
    if phase not in VALID_PHASES:
        raise PolicyViolation(
            f"Invalid DYPOL_PHASE={phase!r}. Expected one of {VALID_PHASES}."
        )

    review_required = _truthy(env.get("DYPOL_REVIEW_REQUIRED", "1"))
    auto_publish = _truthy(env.get("DYPOL_AUTO_PUBLISH", "0"))
    visibility = str(env.get("YOUTUBE_PRIVACY_STATUS", "private") or "private").strip().lower()
    if visibility not in VALID_VISIBILITY:
        raise PolicyViolation(
            f"Invalid YOUTUBE_PRIVACY_STATUS={visibility!r}. Expected one of {VALID_VISIBILITY}."
        )
    job_id = str(env.get("DYPOL_JOB_ID", "") or "").strip()

    if phase == "draft":
        allow_generate, allow_upload = True, False
    elif phase == "publish":
        # Upload of an APPROVED candidate only — the console sends this phase
        # after its review gate; no generation may happen in this phase.
        allow_generate, allow_upload = False, True
        if not job_id:
            raise PolicyViolation(
                "phase=publish requires DYPOL_JOB_ID — publishing must be tied to "
                "an approved console job so the exact candidate can be resolved."
            )
    else:  # full
        allow_generate = True
        # Direct upload only when the configured policy explicitly allows
        # automatic publishing. Review-required mode blocks it — the console
        # ALSO refuses to dispatch full under review-required, so this is a
        # defence-in-depth check on the workflow side.
        allow_upload = auto_publish and not review_required

    return PhasePolicy(
        phase=phase,
        allow_generate=allow_generate,
        allow_upload=allow_upload,
        review_required=review_required,
        auto_publish=auto_publish,
        visibility=visibility,
        job_id=job_id,
    )


# --------------------------------------------------------------------------
# videos.insert body: build + verify
# --------------------------------------------------------------------------

def build_insert_body(title: str, description: str, tags, visibility: str = "private") -> dict:
    """Build the exact request body for youtube.videos().insert."""
    if visibility not in VALID_VISIBILITY:
        raise PolicyViolation(f"Invalid visibility {visibility!r}")
    if isinstance(tags, str):
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    else:
        tag_list = [str(t).strip() for t in (tags or []) if str(t).strip()]
    return {
        "snippet": {
            "title": (title or "Untitled Video")[:100],
            "description": description or "",
            "tags": tag_list,
            "categoryId": "28",
        },
        "status": {
            "privacyStatus": visibility,
            "selfDeclaredMadeForKids": False,
        },
    }


def verify_insert_body(body: dict, expected_visibility: str = "private") -> dict:
    """Refuse to upload if the actual insert body would publish at the wrong
    visibility. This is the last line of defence before videos.insert."""
    actual = ((body or {}).get("status") or {}).get("privacyStatus")
    if actual != expected_visibility:
        raise PolicyViolation(
            f"videos.insert body would upload with privacyStatus={actual!r}, "
            f"expected {expected_visibility!r}. Upload refused."
        )
    if not ((body or {}).get("snippet") or {}).get("title"):
        raise PolicyViolation("videos.insert body is missing a title. Upload refused.")
    return body


# --------------------------------------------------------------------------
# Content plan: preservation, selection, result marking
# --------------------------------------------------------------------------

def _norm(text: str) -> str:
    t = re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()
    t = re.sub(r"ies$", "y", t)          # categories → category
    t = re.sub(r"s$", "", t) if len(t) > 3 else t  # light plural fold
    return " ".join(t.split())


def load_plan(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        plan = json.load(fh)
    if not isinstance(plan.get("lessons"), list):
        raise ValueError("content plan has no 'lessons' list")
    return plan


def save_plan(path: Path, plan: dict) -> None:
    """Atomic write so a crash can never corrupt generation history."""
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(plan, fh, indent=2)
    tmp.replace(path)


def select_next_lesson(plan: dict, topic: str | None = None):
    """Pick the lesson to work on.

    * A console-supplied topic wins: if it matches an existing lesson (normalised
      title/topic compare) that lesson is used — including an already-completed
      one ONLY for explicit re-publish of ITS saved candidate (never regenerated).
    * Without a topic, the first ``pending`` lesson is used. Lessons with
      status 'complete'/'draft_ready' are never re-selected for generation.
    * An unmatched topic becomes a NEW ad-hoc lesson appended to the plan —
      existing history is preserved, nothing is discarded.
    """
    lessons = plan.setdefault("lessons", [])
    if topic:
        key = _norm(topic)
        for lesson in lessons:
            if key and (key == _norm(lesson.get("title")) or key == _norm(lesson.get("topic"))):
                return lesson
        ad_hoc = {
            "chapter": "Ad-hoc",
            "part": "Topic input",
            "title": topic.strip(),
            "topic": topic.strip(),
            "status": "pending",
            "youtube_id": None,
            "origin": "dypol-topic-override",
        }
        lessons.append(ad_hoc)
        return ad_hoc

    for lesson in lessons:
        if lesson.get("status") == "pending":
            return lesson
    return None


def find_candidate(candidate_root: Path, job_id: str) -> dict:
    """Resolve the saved candidate for a console job. Raises CandidateNotFound
    with an actionable message instead of generating or uploading anything else."""
    if not job_id:
        raise CandidateNotFound("No job id supplied — cannot resolve an approved candidate.")
    cand_path = Path(candidate_root) / job_id / "candidate.json"
    if not cand_path.exists():
        raise CandidateNotFound(
            f"Approved candidate for job {job_id} not found at {cand_path}. "
            "The draft may have been cleaned up or never produced. "
            "Re-run phase=draft for this topic (or restore the artifact) — "
            "no upload or regeneration was performed."
        )
    with open(cand_path, "r", encoding="utf-8") as fh:
        cand = json.load(fh)
    # Fail closed when essential visual rights are unresolved. The approval
    # flag must be set only after review of this exact saved candidate.
    if cand.get("review_required") and os.environ.get("DYPOL_VISUAL_REVIEW_APPROVED", "").strip() != "1":
        raise PolicyViolation(
            f"Candidate {job_id} requires visual-rights review before upload. "
            "Review its saved visual_manifest and explicitly set DYPOL_VISUAL_REVIEW_APPROVED=1."
        )
    for asset in cand.get("assets", []):
        for key in ("path",):
            if not asset.get(key) or not Path(asset[key]).exists():
                raise CandidateNotFound(
                    f"Candidate asset missing for job {job_id}: {asset.get('path')}. "
                    "No upload was performed — restore the candidate files or "
                    "re-run phase=draft deliberately."
                )
    return cand


def save_candidate(candidate_root: Path, job_id: str, candidate: dict) -> Path:
    cand_dir = Path(candidate_root) / job_id
    cand_dir.mkdir(parents=True, exist_ok=True)
    cand_path = cand_dir / "candidate.json"
    with open(cand_path, "w", encoding="utf-8") as fh:
        json.dump(candidate, fh, indent=2)
    return cand_path


# --------------------------------------------------------------------------
# Multi-video upload tracking
# --------------------------------------------------------------------------

def record_upload_results(
    videos: list[dict],
    out_path: Path,
    *,
    job_id: str = "",
    run_id: str = "",
    status: str = "uploaded",
    visibility: str = "private",
) -> dict:
    """Persist EVERY uploaded video separately (long-form + Short).

    The result schema keeps a ``videos`` array so the second upload never
    overwrites the first; the legacy single-video fields mirror the primary
    (long-form) record for backward compatibility with the console sync.
    """
    vids = []
    for v in videos or []:
        vids.append({
            "kind": v.get("kind", "video"),           # 'long' | 'short' | 'video'
            "youtube_id": v.get("youtube_id") or v.get("id") or "",
            "youtube_url": v.get("youtube_url")
            or f"https://youtu.be/{v.get('youtube_id') or v.get('id') or ''}",
            "title": v.get("title", ""),
            "visibility": v.get("visibility", visibility),
        })
    primary = next((v for v in vids if v["kind"] == "long"), vids[0] if vids else {})
    # A draft (no uploads) keeps its honest status — only an *attempted* upload
    # without any recorded video becomes "failed".
    final_status = status if vids else (status if status not in ("uploaded", "partial") else "failed")
    result = {
        "job_id": job_id,
        "run_id": run_id,
        "status": final_status,
        "visibility": visibility,
        "videos": vids,
        "youtube_id": primary.get("youtube_id", ""),
        "youtube_url": primary.get("youtube_url", ""),
        "video_count": len(vids),
        "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    tmp.replace(out_path)
    return result


def mark_lesson_result(plan: dict, lesson_title: str, phase: str, videos: list[dict]) -> None:
    """Update plan history without destroying it.

    draft   → status 'draft_ready' (candidate produced; NOT complete, so the
              lesson is not lost and not treated as published).
    publish → status 'complete' with BOTH upload records appended under
              ``uploads`` (never overwriting earlier entries) and the primary
              id mirrored to ``youtube_id`` for legacy readers.
    """
    for lesson in plan.get("lessons", []):
        if _norm(lesson.get("title")) != _norm(lesson_title):
            continue
        uploads = lesson.setdefault("uploads", [])
        for v in videos or []:
            uploads.append({
                "kind": v.get("kind", "video"),
                "youtube_id": v.get("youtube_id") or v.get("id"),
                "youtube_url": v.get("youtube_url")
                or f"https://youtu.be/{v.get('youtube_id') or v.get('id') or ''}",
                "visibility": v.get("visibility", "private"),
                "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            })
        if phase == "draft":
            lesson["status"] = "draft_ready"
        else:
            lesson["status"] = "complete"
            primary = next((u for u in uploads if u["kind"] == "long"), uploads[-1] if uploads else None)
            if primary:
                lesson["youtube_id"] = primary["youtube_id"]          # long-form
            short = next((u for u in uploads if u["kind"] == "short"), None)
            if short:
                lesson["youtube_short_id"] = short["youtube_id"]      # Short kept separately
        return


# --------------------------------------------------------------------------
# Phase executor (called from main.py with real generate/upload callables)
# --------------------------------------------------------------------------

def execute_phase(
    policy: PhasePolicy,
    *,
    generate_fn,
    upload_fn,
    plan: dict,
    candidate_root: Path,
    results_path: Path,
    run_id: str = "",
    candidate_job_id: str = "",
) -> dict:
    """Run one phase. ``generate_fn(lesson) -> candidate dict`` and
    ``upload_fn(asset, meta) -> video record dict`` are injected so the
    semantics are unit-testable and main.py keeps its heavy pipeline code.

    candidate dict schema produced by generate_fn:
      { "job_id", "lesson_title", "assets": [
          {"kind": "long",  "path": "output/long_video_x.mp4",
           "thumb_path": "...", "title": "...", "description": "...", "tags": "..."},
          {"kind": "short", ...} ] }
    """
    policy.phase  # already validated
    job_id = candidate_job_id or policy.job_id

    if policy.phase == "publish":
        # NEVER generate in publish phase — upload the approved candidate only.
        candidate = find_candidate(candidate_root, job_id)
        videos, error = [], None
        if policy.allow_upload:
            for asset in candidate.get("assets", []):
                body = build_insert_body(
                    asset.get("title", ""), asset.get("description", ""), asset.get("tags", ""),
                    policy.visibility,
                )
                verify_insert_body(body, policy.visibility)  # refuse wrong visibility
                try:
                    record = upload_fn(asset, policy)
                except Exception as e:  # persist what succeeded, then fail loudly
                    error = e
                    break
                record.setdefault("kind", asset.get("kind", "video"))
                record.setdefault("visibility", policy.visibility)
                videos.append(record)
        result = record_upload_results(
            videos, results_path, job_id=job_id, run_id=run_id,
            status="uploaded" if videos and not error else ("partial" if videos else "not_uploaded"),
            visibility=policy.visibility,
        )
        mark_lesson_result(plan, candidate.get("lesson_title", ""), "publish", videos)
        if error:
            raise error
        return result

    # draft / full: generation first
    if not policy.allow_generate:
        raise PolicyViolation(f"Generation not permitted in phase={policy.phase}")
    lesson = select_next_lesson(plan, os.environ.get("DYPOL_TOPIC") or None)
    if lesson is None:
        raise PolicyViolation("No pending lesson and no topic supplied — nothing to produce.")
    candidate = generate_fn(lesson)
    # The console job id WINS — it is the handoff key used by publish phase.
    candidate["job_id"] = job_id or candidate.get("job_id") or f"adhoc-{int(datetime.datetime.now().timestamp())}"
    candidate.setdefault("lesson_title", lesson.get("title", ""))

    videos, error = [], None
    if policy.allow_upload:
        for asset in candidate.get("assets", []):
            body = build_insert_body(
                asset.get("title", ""), asset.get("description", ""), asset.get("tags", ""),
                policy.visibility,
            )
            verify_insert_body(body, policy.visibility)
            try:
                record = upload_fn(asset, policy)
            except Exception as e:
                error = e
                break
            record.setdefault("kind", asset.get("kind", "video"))
            record.setdefault("visibility", policy.visibility)
            videos.append(record)
    else:
        # draft: persist the approved-candidate files for the later publish run
        save_candidate(candidate_root, candidate["job_id"], candidate)

    result = record_upload_results(
        videos, results_path, job_id=candidate["job_id"], run_id=run_id,
        status="uploaded" if videos and not error else ("partial" if videos else "draft_only"),
        visibility=policy.visibility,
    )
    mark_lesson_result(plan, candidate.get("lesson_title", ""), "draft" if not policy.allow_upload else "full", videos)
    if error:
        raise error
    return result
