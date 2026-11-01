"""Open Lo-Fi (CC0) background music: deterministic pick, toggle, non-fatal download, mix level."""
import hashlib

from src import open_lofi
from src import production as prod


def _fake_pool(monkeypatch, data=b"fake-mp3-bytes"):
    pool = (("a.mp3", hashlib.sha256(data).hexdigest()), ("b.mp3", hashlib.sha256(data).hexdigest()))
    monkeypatch.setattr(open_lofi, "POOL", pool)
    return data


def test_choice_is_deterministic_per_job():
    assert open_lofi.choose_track("job-1") == open_lofi.choose_track("job-1")
    picks = {open_lofi.choose_track(f"job-{i}")[0] for i in range(40)}
    assert len(picks) > 3  # spreads over the pool
    assert all(name.endswith(".mp3") and len(sha) == 64 for name, sha in open_lofi.POOL)


def test_enabled_by_default_and_toggle():
    assert open_lofi.music_enabled({}) is True
    assert open_lofi.music_enabled({"DYPOL_MUSIC_ENABLED": "0"}) is False
    assert open_lofi.music_enabled({"MUSIC_ENABLED": "off"}) is False
    assert open_lofi.music_enabled({"DYPOL_MUSIC_ENABLED": "1", "MUSIC_ENABLED": "0"}) is True


def test_gain_is_well_below_voice():
    assert open_lofi.music_gain_db({}) == -26.0
    assert open_lofi.music_gain_db({"DYPOL_MUSIC_GAIN_DB": "-10"}) == -24.0
    assert open_lofi.music_gain_db({"DYPOL_MUSIC_GAIN_DB": "-40"}) == -28.0


def test_fetch_downloads_once_then_uses_cache(monkeypatch, tmp_path):
    data = _fake_pool(monkeypatch)
    calls = []

    def downloader(name):
        calls.append(name)
        return data

    rec = open_lofi.fetch_track("job-x", cache_dir=tmp_path, downloader=downloader, env={})
    assert rec["provider"] == "open_lofi" and rec["license"] == "CC0-1.0" and rec["attribution_required"] is False
    assert (tmp_path / rec["file"]).read_bytes() == data
    again = open_lofi.fetch_track("job-x", cache_dir=tmp_path, downloader=downloader, env={})
    assert again["file"] == rec["file"] and len(calls) == 1


def test_download_failure_or_bad_hash_means_no_music(monkeypatch, tmp_path):
    _fake_pool(monkeypatch)

    def boom(name):
        raise OSError("network down")

    assert open_lofi.fetch_track("job-x", cache_dir=tmp_path, downloader=boom, env={}) is None
    assert open_lofi.fetch_track("job-x", cache_dir=tmp_path, downloader=lambda n: b"tampered", env={}) is None
    assert open_lofi.fetch_track("job-x", cache_dir=tmp_path, downloader=boom,
                                 env={"DYPOL_MUSIC_ENABLED": "0"}) is None


def test_mix_uses_track_and_gain_or_renders_without(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(prod.ex, "mix_music_and_normalize",
                        lambda n, o, music_path=None, music_db=-26: seen.append((music_path, music_db)) or o)
    monkeypatch.setattr(prod.ex, "measure_lufs", lambda p: -20.0 if str(p).endswith("n.wav") else -14.0)
    monkeypatch.setitem(prod._MUSIC, "record", {"provider": "open_lofi", "file": "a.mp3",
                                                "path": str(tmp_path / "a.mp3"), "gain_db": -26.0})
    _, record = prod._mix_audio(tmp_path / "n.wav", tmp_path / "o.wav")
    # bed lands 26 dB under the voice: -20 - 26 - (-14) = -32 dB on the track
    assert seen[-1] == (str(tmp_path / "a.mp3"), -32.0) and record["file"] == "a.mp3" and "path" not in record
    assert record["applied_volume_db"] == -32.0 and record["gain_db"] == -26.0
    monkeypatch.setitem(prod._MUSIC, "record", None)
    _, record = prod._mix_audio(tmp_path / "n.wav", tmp_path / "o.wav")
    assert seen[-1][0] is None and record == {"provider": "none"}


def test_relative_gain():
    assert open_lofi.relative_gain_db(-20.0, -14.0, -26.0) == -32.0
    assert open_lofi.relative_gain_db(float("-inf"), -14.0, -26.0) == -26.0
    assert open_lofi.relative_gain_db("x", None, -25.0) == -25.0
    assert open_lofi.relative_gain_db(-10.0, -60.0, -26.0) == -6.0   # never louder than -6 dB on the track
