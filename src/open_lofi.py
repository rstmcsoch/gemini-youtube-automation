"""Background music from the Open Lo-Fi pack (CC0 1.0, public domain).

Source: https://github.com/btahir/open-lofi (LICENSE = CC0 1.0 Universal; README:
"Use them in apps, videos, streams, games, podcasts - anything. No attribution
required."). The audio lives only in the v1.0.0 release zip (~550 MB), so one
track is pulled out of it with HTTP range requests (~2-5 MB, a handful of
requests) at build time and cached; nothing large is committed.

* On by default. DYPOL_MUSIC_ENABLED (or MUSIC_ENABLED) = 0/false/no/off turns it off.
* The bed sits DYPOL_MUSIC_GAIN_DB (default -26, clamped to -28..-24) dB below the
  narration's integrated loudness, with a short fade in / out.
* The track is chosen deterministically from the job id (same job -> same track).
* Every pool track is pinned by sha256; a mismatch or any download problem means
  the video renders without music (never fatal).
* Pool: calm, instrumental-sounding tracks from the Focus / Ambient / Chillhop
  categories. Each was screened with Whisper on a 30 s sample for vocals and the
  ones that looked sung were dropped (screening, not a guarantee).
"""
import hashlib
import io
import os
import zipfile
from pathlib import Path

OPEN_LOFI_REPO = "https://github.com/btahir/open-lofi"
OPEN_LOFI_RELEASE = "v1.0.0"
OPEN_LOFI_ZIP_URL = f"{OPEN_LOFI_REPO}/releases/download/{OPEN_LOFI_RELEASE}/openlofi.zip"
LICENSE = "CC0-1.0"
LICENSE_URL = f"{OPEN_LOFI_REPO}/blob/main/LICENSE"
PROVIDER = "open_lofi"
DEFAULT_GAIN_DB = -26.0
GAIN_RANGE = (-28.0, -24.0)
OFF_VALUES = {"0", "false", "no", "off"}

# (file inside openlofi.zip, sha256 of the mp3)
POOL = (
    ("2-am-debug-loop.mp3", "26d25dda9658677ddccdfdef6c0d316761d9882447c329b86294416f8e44d4be"),
    ("aurora-on-mute.mp3", "a476ca2262812f64e8bc06c06d71fa19c77a8d83dbe7c6f024020ea9a96559f6"),
    ("coffee-ring-notebook.mp3", "e9b6bbafe0c720f38d5df09b84afaf7f29075667ee55c2514d21e0982991149e"),
    ("continue-screen-dreams.mp3", "8bca2116e603b858cffa96a830d2a932d445a3cdb81a2cdfc9d42cec10e6bf83"),
    ("cursor-after-midnight.mp3", "711e8ebe696ade8a24b5bff3231c6f2466e7293278dbfaee98805743ba35c92d"),
    ("deep-space-loop.mp3", "90f897194420809f1752f716611852e85fc0e2c54e1e2002bc07cbd701e87d85"),
    ("drifting-through-fog.mp3", "6e367cdd324e72984deae5bb507c868e6f8551d90f93986f321a2083303bee0c"),
    ("dust-on-the-morning-keys.mp3", "819a31ecf338fd0256493824209267061a37963c52bbee71a84dcbedce42bae6"),
    ("first-coffee-thoughts.mp3", "981f25068f969e2940f9fbab5a35ab092f4d1af303e7ad63912ebf24ca33e156"),
    ("glow-on-the-overpass.mp3", "ac833d9a0bd67d7481cfd78053a66e55124afc766e8fccb2697963416140ed11"),
    ("graphite-in-the-quiet.mp3", "bbc4be5c0923c0c34e0a9c8dd0cd5d0096af358f3c49f380de0fc0b06cc825b0"),
    ("morning-pages.mp3", "f1c6b44455d61ed3b641a72717e0e0d1379d0dd11a49c260011657fca759d228"),
    ("orbiting-in-silence.mp3", "4a8f7eb8c2eeec408fd70f74c6072ea79d1fdd73384b6720a4ae020ac5af052f"),
    ("pixel-quest-save-point.mp3", "536ed1ba4d8274b02bf48a6dab5377914029234e29669cee6e452004295d433d"),
    ("soft-gold-sky.mp3", "c3a0a146a63dc53c7aafbf7aa100d3b5188abcb0ba7ed9312ada1a0e6bffb1d3"),
    ("stacks-of-quiet-hours.mp3", "fde17d22455d1a38d926c0d1f77da86d1c01619ff7038a92d5addac338cf99c5"),
    ("terminal-rain.mp3", "fa8b129cb9deb2225db5bdfed55b531d0fcfae050bf62af551c30d8d5cd806ee"),
    ("warm-constellations.mp3", "abcc5a7ebe752617ef72b3e205950eedff76e8d985424a0fc646879ca43a343c"),
)


def music_enabled(env=None) -> bool:
    env = os.environ if env is None else env
    for name in ("DYPOL_MUSIC_ENABLED", "MUSIC_ENABLED"):
        value = str(env.get(name, "") or "").strip().casefold()
        if value:
            return value not in OFF_VALUES
    return True


def music_gain_db(env=None) -> float:
    env = os.environ if env is None else env
    try:
        gain = float(env.get("DYPOL_MUSIC_GAIN_DB") or env.get("MUSIC_GAIN_DB") or DEFAULT_GAIN_DB)
    except (TypeError, ValueError):
        gain = DEFAULT_GAIN_DB
    return min(GAIN_RANGE[1], max(GAIN_RANGE[0], gain))


def relative_gain_db(voice_lufs, music_lufs, target_db=DEFAULT_GAIN_DB):
    """Volume (dB) to apply to the track so the bed sits ``target_db`` below the voice
    (integrated loudness). Unmeasurable input falls back to ``target_db`` itself."""
    import math
    try:
        voice, music = float(voice_lufs), float(music_lufs)
        if not (math.isfinite(voice) and math.isfinite(music)):
            raise ValueError
    except (TypeError, ValueError):
        return float(target_db)
    return round(min(-6.0, max(-45.0, voice + float(target_db) - music)), 2)


def choose_track(job_id, pool=None):
    """Deterministic pick: the same job id always gets the same track."""
    pool = pool or POOL
    digest = hashlib.sha256(str(job_id or "dypol").encode("utf-8")).hexdigest()
    return pool[int(digest[:12], 16) % len(pool)]


def default_cache_dir(env=None) -> Path:
    env = os.environ if env is None else env
    return Path(env.get("DYPOL_MUSIC_CACHE") or (Path.home() / ".cache" / "dypol-music"))


class _RangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file over HTTP range requests."""

    def __init__(self, url, session=None, timeout=(8, 60)):
        import requests
        self.session = session or requests.Session()
        self.timeout = timeout
        head = self.session.head(url, allow_redirects=True, timeout=timeout)
        head.raise_for_status()
        self.url = head.url
        self.size = int(head.headers["Content-Length"])
        self.pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        base = {0: 0, 1: self.pos, 2: self.size}[whence]
        self.pos = max(0, base + offset)
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n <= 0 or self.pos >= self.size:
            return b""
        end = min(self.size, self.pos + n) - 1
        r = self.session.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=self.timeout)
        if r.status_code != 206:
            raise OSError(f"range request answered HTTP {r.status_code}")
        self.pos += len(r.content)
        return r.content


def _download_member(name, url=OPEN_LOFI_ZIP_URL):
    with zipfile.ZipFile(_RangeFile(url)) as archive:
        return archive.read(name)


def fetch_track(job_id, cache_dir=None, downloader=None, env=None, log=print):
    """Return the music record for this job (with a local ``path``) or None.

    Never raises: disabled, offline, a changed release (sha256 mismatch) or any
    other problem all mean "render without music"."""
    if not music_enabled(env):
        log("Background music disabled (DYPOL_MUSIC_ENABLED=0)")
        return None
    name, sha = choose_track(job_id)
    try:
        cache = Path(cache_dir or default_cache_dir(env))
        cache.mkdir(parents=True, exist_ok=True)
        path = cache / name
        if not (path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == sha):
            data = (downloader or _download_member)(name)
            if hashlib.sha256(data).hexdigest() != sha:
                log(f"WARNING: Open Lo-Fi track {name} failed its sha256 check; rendering without music")
                return None
            tmp = path.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.replace(path)
    except Exception as exc:  # noqa: BLE001 - music is optional, never fatal
        log(f"WARNING: background music unavailable ({str(exc)[:160]}); rendering without music")
        return None
    record = {
        "provider": PROVIDER, "file": name, "sha256": sha, "path": str(path),
        "license": LICENSE, "license_url": LICENSE_URL, "source_url": OPEN_LOFI_ZIP_URL,
        "attribution_required": False, "gain_db": music_gain_db(env),
        # gain_db is the target level of the bed RELATIVE to the narration (dB).
    }
    log(f"Background music: {name} (Open Lo-Fi, CC0), {record['gain_db']:g} dB below the voice")
    return record
