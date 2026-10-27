"""DYPOL-safe YouTube uploader for Engine A. Default visibility is private."""

import os
from pathlib import Path

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

from dypol_control import build_insert_body, verify_insert_body, VALID_VISIBILITY

CLIENT_SECRETS_FILE = Path("client_secrets.json")
CREDENTIALS_FILE = Path("credentials.json")
YOUTUBE_UPLOAD_SCOPE = ["https://www.googleapis.com/auth/youtube.upload"]


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


def get_authenticated_service():
    credentials = None
    if CREDENTIALS_FILE.exists():
        credentials = Credentials.from_authorized_user_file(str(CREDENTIALS_FILE), YOUTUBE_UPLOAD_SCOPE)

    if not credentials or not credentials.valid:
        # A token can be invalid even if its expiry is absent or malformed.
        # Prefer refreshing with the saved refresh token instead of opening an
        # interactive OAuth flow inside a headless GitHub Actions runner.
        if credentials and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            if not CLIENT_SECRETS_FILE.exists():
                raise FileNotFoundError("client_secrets.json is missing; configure YouTube OAuth first.")
            flow = InstalledAppFlow.from_client_secrets_file(
                str(CLIENT_SECRETS_FILE), scopes=YOUTUBE_UPLOAD_SCOPE
            )
            credentials = flow.run_local_server(port=0)
        CREDENTIALS_FILE.write_text(credentials.to_json(), encoding="utf-8")
    return build("youtube", "v3", credentials=credentials)


def upload_to_youtube(video_path, title, description, tags, thumbnail_path=None, kind="video"):
    """Upload one video and return its ID, URL, title, visibility and kind."""
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    safe_title = sanitize_title(title)
    visibility = get_privacy_status()
    body = build_insert_body(safe_title, description, tags, visibility)
    verify_insert_body(body, visibility)
    youtube = get_authenticated_service()
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

    print(f"Video uploaded with visibility={visibility}; ID={video_id}")
    return {
        "youtube_id": video_id,
        "youtube_url": f"https://youtu.be/{video_id}",
        "title": safe_title,
        "visibility": visibility,
        "kind": kind,
        "thumbnail_uploaded": thumbnail_uploaded,
    }
