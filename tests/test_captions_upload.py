"""Captions: SRT generation from narration timings and the non-fatal caption upload
(YouTube HTTP mocked with googleapiclient's HttpMockSequence; no network)."""
import json
from pathlib import Path

import pytest
from googleapiclient.discovery import build
from googleapiclient.http import HttpMockSequence

from src import captions, uploader


WORDS = [("Hello", 0.1, 0.4), ("world.", 0.45, 0.9), ("Next", 1.2, 1.5), ("sentence", 1.55, 2.0),
         ("here.", 2.05, 2.4)]


def test_srt_cues_follow_sentences_and_timestamps():
    srt = captions.words_to_srt(WORDS)
    blocks = srt.strip().split("\n\n")
    assert len(blocks) == 2
    assert blocks[0].splitlines() == ["1", "00:00:00,100 --> 00:00:00,900", "Hello world."]
    assert blocks[1].splitlines()[1] == "00:00:01,200 --> 00:00:02,400"


def test_long_captions_offset_each_slide_by_its_start():
    words = captions.segments_to_words([(0.0, [("Intro.", 0.2, 0.8)]), (12.5, [("Body.", 0.3, 1.0)])])
    srt = captions.words_to_srt(words)
    assert "00:00:12,800 --> 00:00:13,500" in srt
    assert "00:00:00,200 --> 00:00:00,800" in srt


def test_cues_never_overlap_and_split_long_runs():
    words = [(f"w{i}", i * 0.3, i * 0.3 + 0.5) for i in range(30)]
    cues = captions.cue_groups(words)
    assert all(len(c) <= 10 for c in cues)
    srt = captions.words_to_srt(words)
    times = [line for line in srt.splitlines() if "-->" in line]

    def secs(t):
        h, m, rest = t.split(":")
        s, ms = rest.split(",")
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
    spans = [tuple(secs(x.strip()) for x in t.split("-->")) for t in times]
    assert all(a[1] <= b[0] + 1e-9 for a, b in zip(spans, spans[1:]))


def test_write_srt_skips_empty(tmp_path):
    assert captions.write_srt([], tmp_path / "x.srt") is None
    path = captions.write_srt(WORDS, tmp_path / "sub" / "c.srt")
    assert Path(path).read_text(encoding="utf-8").startswith("1\n")


class _Resp(dict):
    def __init__(self, status):
        super().__init__(status=str(status))
        self.status = status
        self.reason = "OK"


def _youtube(responses):
    return build("youtube", "v3", http=HttpMockSequence(responses), static_discovery=True)


def test_upload_captions_posts_snippet_and_file(tmp_path):
    srt = tmp_path / "c.srt"
    srt.write_text(captions.words_to_srt(WORDS), encoding="utf-8")
    seq = [({"status": "200"}, json.dumps({"id": "cap123"}))]
    yt = _youtube(seq)
    assert uploader.upload_captions(yt, "vid1", srt) == "cap123"
    sent = yt._http.request_sequence
    assert len(sent) == 1 and "/upload/youtube/v3/captions" in sent[0][0]


def test_upload_captions_request_shape(tmp_path, monkeypatch):
    srt = tmp_path / "c.srt"
    srt.write_text(captions.words_to_srt(WORDS), encoding="utf-8")
    seen = {}

    class Recorder(HttpMockSequence):
        def request(self, uri, method="GET", body=None, headers=None, **kw):
            seen.update(uri=uri, method=method, body=body if isinstance(body, (str, bytes)) else str(body))
            return super().request(uri, method, body, headers, **kw)
    yt = build("youtube", "v3", http=Recorder([({"status": "200"}, json.dumps({"id": "c1"}))]),
               static_discovery=True)
    assert uploader.upload_captions(yt, "vid9", srt) == "c1"
    assert seen["method"] == "POST"
    assert "/upload/youtube/v3/captions" in seen["uri"] and "part=snippet" in seen["uri"]
    body = seen["body"].decode() if isinstance(seen["body"], bytes) else seen["body"]
    assert '"videoId": "vid9"' in body and '"language": "en"' in body
    assert "Hello world." in body


def test_upload_captions_http_error_is_non_fatal(tmp_path, capsys):
    srt = tmp_path / "c.srt"
    srt.write_text(captions.words_to_srt(WORDS), encoding="utf-8")
    yt = _youtube([({"status": "403"}, json.dumps({"error": {"code": 403, "message": "insufficientPermissions"}}))])
    assert uploader.upload_captions(yt, "vid1", srt) is None
    assert "WARNING: caption upload failed" in capsys.readouterr().out


def test_upload_captions_missing_file_is_non_fatal(tmp_path, capsys):
    assert uploader.upload_captions(object(), "vid1", tmp_path / "nope.srt") is None
    assert "captions skipped" in capsys.readouterr().out


def _creds_file(tmp_path, scopes):
    path = tmp_path / "credentials.json"
    path.write_text(json.dumps({"client_id": "cid", "client_secret": "sec", "refresh_token": "rt",
                                "token_uri": "https://oauth2.googleapis.com/token", "scopes": scopes}))
    return path


class _TokenRequest:
    """Mock google.auth transport: records the refresh POST and answers it."""
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def __call__(self, url, method="GET", body=None, headers=None, **kw):
        self.calls.append({"url": url, "body": body})
        status, payload = self.replies.pop(0)

        class R:
            pass
        r = R()
        r.status, r.headers, r.data = status, {}, json.dumps(payload).encode()
        return r


def test_service_requests_upload_and_force_ssl_without_reconsent(tmp_path, monkeypatch):
    monkeypatch.setattr(uploader, "CREDENTIALS_FILE", _creds_file(tmp_path, [uploader.YOUTUBE_UPLOAD_SCOPE_URL]))
    req = _TokenRequest([(200, {"access_token": "at", "expires_in": 3600,
                                "scope": " ".join(uploader.YOUTUBE_SCOPES)})])
    monkeypatch.setattr(uploader, "Request", lambda: req)
    monkeypatch.setattr(uploader, "build", lambda *a, **k: "yt")
    monkeypatch.setattr(uploader.InstalledAppFlow, "from_client_secrets_file",
                        lambda *a, **k: pytest.fail("must not open a consent flow"))
    yt, granted = uploader.get_authenticated_service()
    assert yt == "yt"
    assert uploader.YOUTUBE_FORCE_SSL_SCOPE_URL in granted
    body = req.calls[0]["body"]
    body = body.decode() if isinstance(body, bytes) else str(body)
    assert "grant_type=refresh_token" in body
    assert "youtube.upload" in body and "youtube.force-ssl" in body
    saved = json.loads(uploader.CREDENTIALS_FILE.read_text())
    assert uploader.YOUTUBE_FORCE_SSL_SCOPE_URL in saved["scopes"]


def test_service_falls_back_to_upload_only_on_invalid_scope(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(uploader, "CREDENTIALS_FILE", _creds_file(tmp_path, [uploader.YOUTUBE_UPLOAD_SCOPE_URL]))
    req = _TokenRequest([(400, {"error": "invalid_scope", "error_description": "bad scope"}),
                         (200, {"access_token": "at", "expires_in": 3600,
                                "scope": uploader.YOUTUBE_UPLOAD_SCOPE_URL})])
    monkeypatch.setattr(uploader, "Request", lambda: req)
    monkeypatch.setattr(uploader, "build", lambda *a, **k: "yt")
    _, granted = uploader.get_authenticated_service()
    assert uploader.YOUTUBE_FORCE_SSL_SCOPE_URL not in granted
    assert "lacks youtube.force-ssl" in capsys.readouterr().out


class _FakeInsert:
    def __init__(self, vid):
        self.vid = vid

    def next_chunk(self):
        return None, {"id": self.vid}


class _FakeYT:
    def __init__(self):
        self.caption_calls, self.insert_bodies = [], []

    def videos(self):
        outer = self

        class V:
            def insert(self, part, body, media_body):
                outer.insert_bodies.append(body)
                return _FakeInsert("vidX")
        return V()

    def captions(self):
        outer = self

        class C:
            def insert(self, part, body, media_body):
                outer.caption_calls.append(body)

                class E:
                    def execute(self_inner):
                        return {"id": "capX"}
                return E()
        return C()


@pytest.fixture
def fake_upload(tmp_path, monkeypatch):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"\x00" * 64)
    srt = tmp_path / "c.srt"
    srt.write_text(captions.words_to_srt(WORDS), encoding="utf-8")
    yt = _FakeYT()
    monkeypatch.setattr(uploader, "MediaFileUpload", lambda *a, **k: object())
    monkeypatch.setenv("YOUTUBE_PRIVACY_STATUS", "unlisted")
    return video, srt, yt


def test_upload_attaches_captions_after_video(fake_upload, monkeypatch):
    video, srt, yt = fake_upload
    monkeypatch.setattr(uploader, "get_authenticated_service", lambda: (yt, uploader.YOUTUBE_SCOPES))
    rec = uploader.upload_to_youtube(video, "T", "D", "a,b", kind="short", captions_path=srt)
    assert rec["youtube_id"] == "vidX" and rec["visibility"] == "unlisted"
    assert rec["captions_uploaded"] is True and rec["caption_id"] == "capX"
    assert yt.caption_calls[0]["snippet"]["videoId"] == "vidX"
    assert yt.insert_bodies[0]["status"]["privacyStatus"] == "unlisted"


def test_upload_skips_captions_without_force_ssl(fake_upload, monkeypatch, capsys):
    video, srt, yt = fake_upload
    monkeypatch.setattr(uploader, "get_authenticated_service", lambda: (yt, uploader.YOUTUBE_UPLOAD_SCOPE))
    rec = uploader.upload_to_youtube(video, "T", "D", "a", kind="long", captions_path=srt)
    assert rec["youtube_id"] == "vidX" and rec["captions_uploaded"] is False
    assert yt.caption_calls == []
    assert "lacks youtube.force-ssl" in capsys.readouterr().out


def test_caption_failure_does_not_fail_upload(fake_upload, monkeypatch):
    video, srt, yt = fake_upload

    def boom(*a, **k):
        raise RuntimeError("quota")
    yt.captions = boom
    monkeypatch.setattr(uploader, "get_authenticated_service", lambda: (yt, uploader.YOUTUBE_SCOPES))
    rec = uploader.upload_to_youtube(video, "T", "D", "a", kind="long", captions_path=srt)
    assert rec["youtube_id"] == "vidX" and rec["captions_uploaded"] is False


def test_upload_results_keep_caption_flag(tmp_path):
    from dypol_control import record_upload_results
    res = record_upload_results([{"kind": "long", "youtube_id": "a", "captions_uploaded": True},
                                 {"kind": "short", "youtube_id": "b"}], tmp_path / "r.json")
    assert res["videos"][0]["captions_uploaded"] is True
    assert "captions_uploaded" not in res["videos"][1]
