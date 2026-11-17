"""Safe video URL importing for GeoDubber's online API.

Supported sources:
- YouTube video URLs (single video only)
- Direct HTTPS URLs to public video files

Magnet/torrent URLs and arbitrary third-party page extraction are intentionally
not supported. Users must only import content they own or are authorized to process.
"""
from __future__ import annotations

import ipaddress
import socket
import shutil
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be"}
REDIRECT_CODES = {301, 302, 303, 307, 308}


class VideoUrlError(ValueError):
    pass


def _validate_youtube_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        return False
    host = parsed.hostname.lower().rstrip(".")
    return host in YOUTUBE_HOSTS


def _validate_public_https_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise VideoUrlError("პირდაპირი ვიდეოს ბმული უნდა იყოს HTTPS მისამართი.")
    if parsed.username or parsed.password:
        raise VideoUrlError("ვიდეოს ბმულში მომხმარებლის სახელი ან პაროლი დაუშვებელია.")
    if parsed.port not in (None, 443):
        raise VideoUrlError("ვიდეოს ბმულისთვის დაშვებულია მხოლოდ HTTPS პორტი 443.")
    extension = Path(parsed.path).suffix.lower()
    if extension not in VIDEO_EXTENSIONS:
        raise VideoUrlError("პირდაპირი ბმული უნდა მიუთითებდეს MP4, MKV, AVI, MOV ან WEBM ფაილზე.")

    hostname = parsed.hostname
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(hostname, 443, type=socket.SOCK_STREAM)}
    except OSError as exc:
        raise VideoUrlError("ვიდეოს მისამართის DNS ჩანაწერი ვერ მოიძებნა.") from exc
    if not addresses:
        raise VideoUrlError("ვიდეოს სერვერის მისამართი ვერ განისაზღვრა.")
    for value in addresses:
        try:
            ip = ipaddress.ip_address(value)
        except ValueError as exc:
            raise VideoUrlError("ვიდეოს სერვერის IP მისამართი არასწორია.") from exc
        if not ip.is_global:
            raise VideoUrlError("ლოკალურ ან კერძო ქსელში განთავსებული მისამართებიდან ჩამოტვირთვა აკრძალულია.")
    return extension


def _download_direct(url: str, destination: Path, max_bytes: int, progress) -> Path:
    current = url
    session = requests.Session()
    session.headers.update({"User-Agent": "GeoDubber/1.0", "Accept": "video/*,application/octet-stream;q=0.9,*/*;q=0.5"})
    temporary = destination.with_suffix(destination.suffix + ".uploading")
    try:
        for redirect_count in range(6):
            extension = _validate_public_https_url(current)
            response = session.get(current, stream=True, timeout=(15, 60), allow_redirects=False)
            if response.status_code in REDIRECT_CODES:
                location = response.headers.get("Location")
                response.close()
                if not location or redirect_count >= 5:
                    raise VideoUrlError("ვიდეოს სერვერმა ზედმეტად ბევრი გადამისამართება დააბრუნა.")
                current = urljoin(current, location)
                continue
            response.raise_for_status()
            length = response.headers.get("Content-Length")
            if length:
                try:
                    if int(length) > max_bytes:
                        raise VideoUrlError("ვიდეო ფაილი დაშვებულ 8 GB ზომას აჭარბებს.")
                except ValueError:
                    raise VideoUrlError("ვიდეოს სერვერმა არასწორი Content-Length დააბრუნა.")
            total = 0
            last_update = 0.0
            destination.parent.mkdir(parents=True, exist_ok=True)
            with temporary.open("wb") as stream:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > max_bytes:
                        raise VideoUrlError("ვიდეო ფაილი დაშვებულ 8 GB ზომას აჭარბებს.")
                    stream.write(chunk)
                    now = time.monotonic()
                    if now - last_update >= 0.5:
                        progress(total, int(length) if length and length.isdigit() else 0, "downloading")
                        last_update = now
            response.close()
            if total < 128:
                raise VideoUrlError("ვიდეო ფაილი ცარიელია ან დაზიანებულია.")
            temporary.replace(destination.with_suffix(extension))
            result = destination.with_suffix(extension)
            progress(total, total, "finished")
            return result
        raise VideoUrlError("ვიდეოს გადამისამართება ვერ დასრულდა.")
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        session.close()


def _download_youtube(url: str, destination_dir: Path, max_bytes: int, progress) -> Path:
    try:
        from yt_dlp import YoutubeDL
    except ImportError as exc:
        raise VideoUrlError("YouTube ბმულისთვის yt-dlp მოდული არ არის დაყენებული. განაახლე requirements-online.txt.") from exc

    destination_dir.mkdir(parents=True, exist_ok=True)
    outtmpl = str(destination_dir / "source.%(ext)s")

    def hook(data):
        status = data.get("status")
        downloaded = int(data.get("downloaded_bytes") or 0)
        total = int(data.get("total_bytes") or data.get("total_bytes_estimate") or 0)
        progress(downloaded, total, status or "downloading")

    ffmpeg_location = shutil.which("ffmpeg")
    options = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "outtmpl": outtmpl,
        "noplaylist": True,
        "max_filesize": max_bytes,
        "socket_timeout": 30,
        "retries": 5,
        "fragment_retries": 5,
        "continuedl": True,
        "overwrites": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [hook],
    }
    if ffmpeg_location:
        options["ffmpeg_location"] = ffmpeg_location

    try:
        with YoutubeDL(options) as downloader:
            info = downloader.extract_info(url, download=True)
        if not info or info.get("_type") == "playlist":
            raise VideoUrlError("მხარდაჭერილია მხოლოდ ერთი YouTube ვიდეო, არა playlist.")
    except VideoUrlError:
        raise
    except Exception as exc:
        raise VideoUrlError("YouTube ვიდეოს ჩამოტვირთვა ვერ მოხერხდა. გადაამოწმე ბმული და ვიდეოს ხელმისაწვდომობა.") from exc

    candidates = [
        path for path in destination_dir.glob("source.*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS and not path.name.endswith(".part")
    ]
    if not candidates:
        raise VideoUrlError("YouTube-დან მიღებული ვიდეოფაილი ვერ მოიძებნა.")
    result = max(candidates, key=lambda path: path.stat().st_size)
    if result.stat().st_size > max_bytes:
        result.unlink(missing_ok=True)
        raise VideoUrlError("ვიდეო ფაილი დაშვებულ 8 GB ზომას აჭარბებს.")
    progress(result.stat().st_size, result.stat().st_size, "finished")
    return result


def download_video_url(url: str, destination_dir: str | Path, max_bytes: int, progress) -> Path:
    raw = str(url or "").strip()
    if len(raw) > 4096:
        raise VideoUrlError("ვიდეოს ბმული ზედმეტად გრძელია.")
    parsed = urlparse(raw)
    if parsed.scheme.lower() == "magnet":
        raise VideoUrlError("Magnet/Torrent ბმულები არ არის მხარდაჭერილი. გამოიყენე ვიდეოფაილი ან პირდაპირი HTTPS ბმული იმ მასალაზე, რომლის გამოყენების უფლებაც გაქვს.")
    if _validate_youtube_url(raw):
        return _download_youtube(raw, Path(destination_dir), max_bytes, progress)
    extension = _validate_public_https_url(raw)
    destination = Path(destination_dir) / ("source" + extension)
    return _download_direct(raw, destination, max_bytes, progress)
