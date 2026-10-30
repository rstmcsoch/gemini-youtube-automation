"""DYPOL-safe YouTube uploader for Engine A. Default visibility is private."""

import os
from pathlib import Path

from google.oauth2.credentials import Credentials
from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

from dypol_control import (
    build_insert_body, verify_insert_body, resolve_contains_synthetic_media, VALID_VISIBILITY,
)

CLIENT_SECRETS_FILE = Path("client_secrets.json")
CREDENTIALS_FILE = Path("credentials.json")
YOUTUBE_UPLOAD_SCOPE_URL = "https://www.googleapis.com/auth/youtube.upload"
YOUTUBE_FORCE_SSL_SCOPE_URL = "https://www.googleapis.com/auth/youtube.force-ssl"
# The stored refresh token was granted both scopes; force-ssl is what captions.insert needs.
YOUTUBE_SCOPES = [YOUTUBE_UPLOAD_SCOPE_URL, YOUTUBE_FORCE_SSL_SCOPE_URL]
YOUTUBE_UPLOAD_SCOPE = [YOUTUBE_UPLOAD_SCOPE_URL]  # fallback for an upload-only token
CAPTION_LANGUAGE = "en"
CAPTION_NAME = "English"


def sanitize_title(title, max_length=100):
    clean = str(title or "").replace("&lt;", "").replace("&gt;", "").replace("<", "").replace(">", "").strip()
    if not clean:
        clean = "Untitled Video"
    if len(clean) > max_length:
        clean = clean[:max(1, max_length - 3)].rstrip() + ("..." if max_length > 3 else "")
    return clean[:max_length]


def get_privacy_status():
    visibility = str(os.environ.get("YOUTUBE_PRIVACY_STATUS", "private") or "private").strip().lower()
    if visibility not in VALID_VISIBILITY:
        print("WARNING: invalid YOUTUBE_PRIVACY_STATUS; falling back to private.")
        return "private"
    return visibility


def _load_credentials(scopes):
    """Credentials from the saved refresh token, asking for ``scopes`` on refresh.

    Never re-consents: the refresh request only narrows to scopes the token
    already holds, so no browser/consent screen is involved."""
    credentials = Credentials.from_authorized_user_file(str(CREDENTIALS_FILE), scopes)
    if not credentials.valid:
        # A token can be invalid even if its expiry is absent or malformed.
        # Refresh with the saved refresh token instead of an interactive flow.
        if not credentials.refresh_token:
            return None
        credentials.refresh(Request())
    return credentials


def get_authenticated_service(scopes=None):
    """Return (youtube_service, granted_scopes).

    Requests upload + force-ssl. If the stored token turns out to be upload-only
    (refresh fails with invalid_scope), it falls back to upload-only so the video
    still uploads; captions are then skipped with a warning."""
    scopes = list(scopes or YOUTUBE_SCOPES)
    credentials = None
    if CREDENTIALS_FILE.exists():
        try:
            credentials = _load_credentials(scopes)
        except RefreshError as exc:
            if "invalid_scope" in str(exc) and set(scopes) != set(YOUTUBE_UPLOAD_SCOPE):
                print("WARNING: stored YouTube token lacks youtube.force-ssl; "
                      "uploading with youtube.upload only (captions will be skipped).")
                scopes = list(YOUTUBE_UPLOAD_SCOPE)
                credentials = _load_credentials(scopes)
            else:
                raise
        if credentials is not None:
            CREDENTIALS_FILE.write_text(credentials.to_json(), encoding="utf-8")
    if credentials is None:
        if not CLIENT_SECRETS_FILE.exists():
            raise FileNotFoundError("client_secrets.json is missing; configure YouTube OAuth first.")
        flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRETS_FILE), scopes=scopes)
        credentials = flow.run_local_server(port=0)
        CREDENTIALS_FILE.write_text(credentials.to_json(), encoding="utf-8")
    granted = list(getattr(credentials, "granted_scopes", None) or credentials.scopes or scopes)
    return build("youtube", "v3", credentials=credentials), granted


def upload_captions(youtube, video_id, captions_path, language=CAPTION_LANGUAGE, name=CAPTION_NAME):
    """Attach an SRT/VTT caption track to an uploaded video. Never raises.

    Returns the caption track id, or None (with a logged warning) on any failure."""
    path = Path(captions_path) if captions_path else None
    if not path or not path.is_file() or path.stat().st_size == 0:
        print(f"WARNING: no caption file for video {video_id}; captions skipped.")
        return None
    try:
        body = {"snippet": {"videoId": video_id, "language": language, "name": name, "isDraft": False}}
        media = MediaFileUpload(str(path), mimetype="application/octet-stream", resumable=False)
        response = youtube.captions().insert(part="snippet", body=body, media_body=media).execute()
        caption_id = (response or {}).get("id")
        if not caption_id:
            print(f"WARNING: caption upload for video {video_id} returned no id.")
            return None
        print(f"Captions uploaded for video {video_id}: {path.name} (track {caption_id})")
        return caption_id
    except Exception as exc:  # noqa: BLE001 - captions must never fail the publish
        print(f"WARNING: caption upload failed for video {video_id}: {exc}")
        return None


def upload_to_youtube(video_path, title, description, tags, thumbnail_path=None, kind="video",
                      visual_manifest=None, captions_path=None):
    """Upload one video and return its ID, URL, title, visibility and kind.

    ``captions_path`` (SRT/VTT) is attached after upload; a caption failure only
    logs a warning. ``visual_manifest`` is this video's visual manifest; when it includes an
    AI-generated asset (FLUX / Cloudflare / AI-generated) the upload declares
    status.containsSyntheticMedia=True, unless YOUTUBE_CONTAINS_SYNTHETIC_MEDIA
    (auto|true|false, default auto) forces the value."""
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    safe_title = sanitize_title(title)
    visibility = get_privacy_status()
    contains_synthetic_media = resolve_contains_synthetic_media(visual_manifest)
    body = build_insert_body(safe_title, description, tags, visibility, contains_synthetic_media)
    verify_insert_body(body, visibility)
    print(f"containsSyntheticMedia={contains_synthetic_media} ({kind})")
    service = get_authenticated_service()
    youtube, granted_scopes = service if isinstance(service, tuple) else (service, list(YOUTUBE_SCOPES))
    media = MediaFileUpload(str(video_path), chunksize=-1, resumable=True)
    request = youtube.videos().insert(part=",".join(body.keys()), body=body, media_body=media)

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"Uploaded {int(status.progress() * 100)}%")

    video_id = (response or {}).get("id")
    if not video_id:
        raise RuntimeError("YouTube API returned no video ID; refusing to report success.")

    thumbnail_uploaded = False
    if thumbnail_path and Path(thumbnail_path).is_file():
        try:
            youtube.thumbnails().set(
                videoId=video_id,
                media_body=MediaFileUpload(str(thumbnail_path))
            ).execute()
            thumbnail_uploaded = True
        except Exception as exc:
            # Thumbnail failure does not undo the successful video upload.
            print(f"WARNING: thumbnail upload failed for video {video_id}: {exc}")

    caption_id = None
    if captions_path:
        if YOUTUBE_FORCE_SSL_SCOPE_URL in granted_scopes:
            caption_id = upload_captions(youtube, video_id, captions_path)
        else:
            print(f"WARNING: captions skipped for video {video_id}: token lacks youtube.force-ssl.")

    print(f"Video uploaded with visibility={visibility}; ID={video_id}")
    return {
        "youtube_id": video_id,
        "youtube_url": f"https://youtu.be/{video_id}",
        "title": safe_title,
        "visibility": visibility,
        "kind": kind,
        "thumbnail_uploaded": thumbnail_uploaded,
        "contains_synthetic_media": contains_synthetic_media,
        "captions_uploaded": bool(caption_id),
        "caption_id": caption_id or "",
    }
