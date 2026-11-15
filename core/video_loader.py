"""Input validation, probing and (authorised) YouTube download."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from core.errors import CorruptMediaError, DownloadError, UnsupportedFormatError
from core.ffmpeg_utils import probe_json

SUPPORTED_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
_YT_RE = re.compile(r"^(https?://)?(www\.|m\.)?(youtube\.com/(watch\?v=|shorts/|live/)|youtu\.be/)[\w\-]{6,}", re.I)


@dataclass
class MediaInfo:
    path: str
    filename: str
    size_bytes: int
    duration: float
    width: int
    height: int
    fps: float
    video_codec: str
    has_audio: bool
    audio_codec: str
    sample_rate: int
    channels: int

    def summary(self) -> str:
        m, s = divmod(int(self.duration), 60)
        h, m = divmod(m, 60)
        a = f"{self.audio_codec}, {self.sample_rate} Hz, {self.channels} ch" if self.has_audio else "no audio"
        return (f"{self.filename} | {h:d}:{m:02d}:{s:02d} | {self.size_bytes/2**20:.1f} MB | "
                f"{self.width}x{self.height} @ {self.fps:.2f} fps | {a}")


def _fps(r: str) -> float:
    try:
        n, d = r.split("/")
        return float(n) / float(d) if float(d) else 0.0
    except Exception:
        try:
            return float(r)
        except Exception:
            return 0.0


def probe(path: str | Path) -> MediaInfo:
    p = Path(path)
    if not p.is_file():
        raise CorruptMediaError(f"not a file: {p}", "The selected file does not exist.")
    if p.suffix.lower() not in SUPPORTED_EXTS:
        raise UnsupportedFormatError(f"unsupported extension {p.suffix}",
                                     f"Unsupported format '{p.suffix}'. Supported: MP4, MKV, AVI, MOV, WEBM.")
    j = probe_json(p)
    streams = j.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video" and s.get("disposition", {}).get("attached_pic") != 1), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if v is None:
        raise CorruptMediaError("no video stream", "No video track was found in this file.")
    dur = float(j.get("format", {}).get("duration") or v.get("duration") or 0)
    if dur <= 0:
        raise CorruptMediaError("zero duration", "The video has no readable duration (file may be corrupted).")
    return MediaInfo(
        path=str(p), filename=p.name, size_bytes=p.stat().st_size, duration=dur,
        width=int(v.get("width", 0)), height=int(v.get("height", 0)),
        fps=_fps(v.get("avg_frame_rate") or v.get("r_frame_rate", "0/1")),
        video_codec=v.get("codec_name", ""), has_audio=a is not None,
        audio_codec=(a or {}).get("codec_name", ""), sample_rate=int((a or {}).get("sample_rate", 0) or 0),
        channels=int((a or {}).get("channels", 0) or 0))


def validate_for_dubbing(info: MediaInfo) -> None:
    if not info.has_audio:
        raise CorruptMediaError("no audio", "This video has no audio track, so there is no dialogue to dub.")


def is_youtube_url(url: str) -> bool:
    return bool(_YT_RE.match(url.strip()))


def download_youtube(url: str, dest_dir: str | Path, on_progress: Optional[Callable[[float], None]] = None) -> Path:
    """Download with yt-dlp. Only for content you own or are allowed to download.
    No DRM/access-control circumvention is attempted; errors are reported as-is."""
    if not is_youtube_url(url):
        raise DownloadError("invalid url", "This does not look like a valid YouTube link.")
    try:
        import yt_dlp
    except ImportError:
        raise DownloadError("yt-dlp missing", "yt-dlp is not installed (run install.bat).")
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    def hook(d):
        if on_progress and d.get("status") == "downloading" and d.get("total_bytes"):
            on_progress(d["downloaded_bytes"] / d["total_bytes"])

    opts = {"outtmpl": str(dest / "%(title).80s [%(id)s].%(ext)s"), "format": "bv*+ba/b",
            "merge_output_format": "mp4", "noplaylist": True, "progress_hooks": [hook], "quiet": True}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            path = Path(ydl.prepare_filename(info)).with_suffix(".mp4")
            if not path.exists():
                path = Path(ydl.prepare_filename(info))
    except Exception as e:
        msg = str(e)
        user = "The video could not be downloaded."
        if "private" in msg.lower() or "sign in" in msg.lower():
            user = "This video is private or requires sign-in; it cannot be downloaded."
        elif "unable to download" in msg.lower() or "urlopen" in msg.lower() or "network" in msg.lower():
            user = "Network error while downloading. Check your internet connection."
        raise DownloadError(msg, user)
    return path
