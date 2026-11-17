"""Authorized video URL importing for GeoDubber's online API.

Supported sources:
- YouTube single-video URLs
- Direct HTTPS video-file URLs
- BitTorrent magnet URIs (one video file selected from the torrent)

Only import videos you own or are authorized to process.
"""
from __future__ import annotations

import ipaddress
import re
import shutil
import socket
import time
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urljoin, urlparse

import requests

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be"}
REDIRECT_CODES = {301, 302, 303, 307, 308}
MAGNET_INFO_HASH = re.compile(r"^(?:[0-9a-fA-F]{40}|[A-Z2-7]{32})$")


class VideoUrlError(ValueError):
    pass


class VideoUrlCancelled(VideoUrlError):
    """Raised when a user cancels an in-progress source download."""


def _check_cancelled(cancelled: Callable[[], bool] | None) -> None:
    if cancelled is not None and cancelled():
        raise VideoUrlCancelled("ჩამოტვირთვა გაუქმდა.")


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


def _tracker_is_public(tracker: str) -> bool:
    parsed = urlparse(str(tracker))
    if parsed.scheme.lower() not in {"udp", "http", "https"} or not parsed.hostname:
        return False
    if parsed.username or parsed.password:
        return False
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80 if parsed.scheme == "http" else 6969)
        kind = socket.SOCK_DGRAM if parsed.scheme == "udp" else socket.SOCK_STREAM
        addresses = {item[4][0] for item in socket.getaddrinfo(parsed.hostname, port, type=kind)}
    except (OSError, ValueError):
        return False
    if not addresses:
        return False
    try:
        return all(ipaddress.ip_address(value).is_global for value in addresses)
    except ValueError:
        return False


def _download_magnet(magnet: str, destination_dir: Path, max_bytes: int, progress, cancelled: Callable[[], bool] | None = None) -> Path:
    query = parse_qs(urlparse(magnet).query)
    info_hashes = [
        item.split(":", 2)[-1].strip()
        for item in query.get("xt", [])
        if item.lower().startswith("urn:btih:")
    ]
    if not any(MAGNET_INFO_HASH.fullmatch(value) for value in info_hashes):
        raise VideoUrlError("Magnet ბმულში არასწორი ან დაკარგული BitTorrent info-hash არის.")

    try:
        import libtorrent as lt
    except ImportError as exc:
        raise VideoUrlError("Magnet/Torrent ჩამოტვირთვისთვის libtorrent მოდული არ არის დაყენებული. განაახლე requirements-online.txt.") from exc

    destination_dir.mkdir(parents=True, exist_ok=True)
    torrent_dir = destination_dir / "_torrent_download"
    torrent_dir.mkdir(parents=True, exist_ok=True)
    session = None
    handle = None
    selected_path: Path | None = None
    selected_size = 0

    try:
        try:
            params = lt.parse_magnet_uri(magnet)
        except Exception as exc:
            raise VideoUrlError("Magnet ბმულის ფორმატი არასწორია.") from exc

        # Reject tracker endpoints that resolve to localhost, private or reserved IP ranges.
        try:
            params.trackers = [tracker for tracker in params.trackers if _tracker_is_public(tracker)]
        except Exception:
            params.trackers = []

        params.save_path = str(torrent_dir)
        params.storage_mode = lt.storage_mode_t.storage_mode_sparse
        session = lt.session({"listen_interfaces": "0.0.0.0:0", "enable_dht": True})
        handle = session.add_torrent(params)

        metadata_deadline = time.monotonic() + 300
        last_update = 0.0
        while True:
            _check_cancelled(cancelled)
            try:
                torrent_info = handle.torrent_file()
            except Exception:
                torrent_info = None
            if torrent_info is not None:
                break
            now = time.monotonic()
            if now >= metadata_deadline:
                raise VideoUrlError("ტორენტის ინფორმაცია ვერ მოიძებნა. შეამოწმე magnet ბმული, seed-ები და tracker-ები.")
            if now - last_update >= 1:
                progress(0, 0, "metadata")
                last_update = now
            time.sleep(1)

        file_storage = torrent_info.files()
        candidates: list[tuple[int, int, Path]] = []
        root = torrent_dir.resolve()
        for index in range(file_storage.num_files()):
            relative_name = file_storage.file_path(index)
            extension = Path(relative_name).suffix.lower()
            if extension not in VIDEO_EXTENSIONS:
                continue
            size = int(file_storage.file_size(index))
            if size <= 0:
                continue
            candidate = (torrent_dir / relative_name).resolve()
            if not candidate.is_relative_to(root):
                continue
            candidates.append((size, index, candidate))

        if not candidates:
            raise VideoUrlError("ტორენტში მხარდაჭერილი ვიდეოფაილი ვერ მოიძებნა (MP4, MKV, AVI, MOV, WEBM).")

        # Download one main video only; skip extras, samples, subtitles and archives.
        viable = [item for item in candidates if item[0] <= max_bytes]
        if not viable:
            raise VideoUrlError("ტორენტში არსებული ვიდეოფაილები 8 GB დაშვებულ ზომას აჭარბებს.")
        non_sample = [item for item in viable if "sample" not in item[2].name.lower()]
        selected_size, selected_index, selected_path = max(non_sample or viable, key=lambda item: item[0])

        priorities = [0] * file_storage.num_files()
        priorities[selected_index] = 4
        handle.prioritize_files(priorities)
        handle.resume()

        deadline = time.monotonic() + max(3600, int(__import__("os").getenv("GEODUBBER_TORRENT_TIMEOUT", "86400")))
        while True:
            _check_cancelled(cancelled)
            now = time.monotonic()
            if now >= deadline:
                raise VideoUrlError("ტორენტის ჩამოტვირთვის დროის ლიმიტი ამოიწურა.")
            try:
                downloaded = int(handle.file_progress()[selected_index])
            except Exception:
                downloaded = 0
            progress(downloaded, selected_size, "downloading")
            if downloaded >= selected_size and selected_path.is_file() and selected_path.stat().st_size >= selected_size:
                break
            time.sleep(1)

        handle.pause()
        session.remove_torrent(handle)
        handle = None
        session = None

        final_path = destination_dir / ("source" + selected_path.suffix.lower())
        if final_path.exists():
            final_path.unlink()
        shutil.move(str(selected_path), str(final_path))
        shutil.rmtree(torrent_dir, ignore_errors=True)
        progress(selected_size, selected_size, "finished")
        return final_path

    except VideoUrlError:
        raise
    except Exception as exc:
        raise VideoUrlError("ტორენტიდან ვიდეოს ჩამოტვირთვა ვერ მოხერხდა. შეამოწმე seed-ები, tracker-ები და დისკზე თავისუფალი ადგილი.") from exc
    finally:
        if handle is not None and session is not None:
            try:
                handle.pause()
                session.remove_torrent(handle)
            except Exception:
                pass


def _download_direct(url: str, destination: Path, max_bytes: int, progress, cancelled: Callable[[], bool] | None = None) -> Path:
    current = url
    session = requests.Session()
    session.headers.update({"User-Agent": "GeoDubber/1.0", "Accept": "video/*,application/octet-stream;q=0.9,*/*;q=0.5"})
    temporary = destination.with_suffix(destination.suffix + ".uploading")
    try:
        for redirect_count in range(6):
            _check_cancelled(cancelled)
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
                    _check_cancelled(cancelled)
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


def _download_youtube(url: str, destination_dir: Path, max_bytes: int, progress, cancelled: Callable[[], bool] | None = None) -> Path:
    try:
        from yt_dlp import YoutubeDL
    except ImportError as exc:
        raise VideoUrlError("YouTube ბმულისთვის yt-dlp მოდული არ არის დაყენებული. განაახლე requirements-online.txt.") from exc

    destination_dir.mkdir(parents=True, exist_ok=True)
    outtmpl = str(destination_dir / "source.%(ext)s")

    def hook(data):
        _check_cancelled(cancelled)
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
        if cancelled is not None and cancelled():
            raise VideoUrlCancelled("ჩამოტვირთვა გაუქმდა.") from exc
        raise VideoUrlError("YouTube ვიდეოს ჩამოტვირთვა ვერ მოხერხდა. გადაამოწმე ბმული და ვიდეოს ხელმისაწვდომობა.") from exc
    _check_cancelled(cancelled)

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


def download_video_url(url: str, destination_dir: str | Path, max_bytes: int, progress, cancelled: Callable[[], bool] | None = None) -> Path:
    raw = str(url or "").strip()
    if len(raw) > 4096:
        raise VideoUrlError("ვიდეოს ბმული ზედმეტად გრძელია.")
    parsed = urlparse(raw)
    _check_cancelled(cancelled)
    if parsed.scheme.lower() == "magnet":
        return _download_magnet(raw, Path(destination_dir), max_bytes, progress, cancelled)
    if _validate_youtube_url(raw):
        return _download_youtube(raw, Path(destination_dir), max_bytes, progress, cancelled)
    extension = _validate_public_https_url(raw)
    destination = Path(destination_dir) / ("source" + extension)
    return _download_direct(raw, destination, max_bytes, progress, cancelled)
