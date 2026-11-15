"""Locate and run FFmpeg/FFprobe with progress, cancellation and readable errors."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Optional, Sequence

from config.settings import app_dir
from core.errors import CancelledError, CorruptMediaError, MediaProcessingError, MissingDependencyError

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def _find(name: str) -> str:
    env = os.environ.get(f"GEODUBBER_{name.upper()}")
    if env and Path(env).exists():
        return env
    exe = name + (".exe" if os.name == "nt" else "")
    for cand in (app_dir() / "tools" / "ffmpeg" / "bin" / exe, app_dir() / "tools" / exe):
        if cand.is_file():
            return str(cand)
    found = shutil.which(name)
    if found:
        return found
    raise MissingDependencyError(
        f"{name} not found",
        f"{name} was not found. Install FFmpeg (see README, 'FFmpeg setup') or put it in tools\\ffmpeg\\bin.",
    )


def ffmpeg_exe() -> str:
    return _find("ffmpeg")


def ffprobe_exe() -> str:
    return _find("ffprobe")


def have_filter(name: str) -> bool:
    try:
        out = subprocess.run([ffmpeg_exe(), "-hide_banner", "-filters"], capture_output=True, text=True,
                             creationflags=_NO_WINDOW).stdout
    except Exception:
        return False
    return any(line.split()[1:2] == [name] for line in out.splitlines() if len(line.split()) > 1)


def probe_json(path: str | Path) -> dict:
    cmd = [ffprobe_exe(), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)]
    p = subprocess.run(cmd, capture_output=True, text=True, creationflags=_NO_WINDOW)
    if p.returncode != 0:
        raise CorruptMediaError(p.stderr.strip(), "The file could not be read. It may be corrupted or not a video.")
    try:
        return json.loads(p.stdout)
    except ValueError as e:
        raise CorruptMediaError(str(e), "The file could not be analysed.")


class Control:
    """Cooperative pause/cancel. Long loops call `checkpoint()` between work units."""

    def __init__(self):
        import threading
        self._cancel = threading.Event()
        self._run = threading.Event()
        self._run.set()

    def cancel(self):
        self._cancel.set()
        self._run.set()  # wake paused workers so they can exit

    def pause(self):
        self._run.clear()

    def resume(self):
        self._run.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    @property
    def paused(self) -> bool:
        return not self._run.is_set()

    def checkpoint(self):
        while not self._run.wait(0.2):
            if self._cancel.is_set():
                break
        if self._cancel.is_set():
            raise CancelledError()


def run_ffmpeg(args: Sequence[str], duration: Optional[float] = None,
               on_progress: Optional[Callable[[float], None]] = None,
               control: Optional[Control] = None) -> None:
    """Run ffmpeg. `on_progress` receives a 0..1 fraction when `duration` is known."""
    cmd = [ffmpeg_exe(), "-hide_banner", "-nostdin", "-y", "-threads", "0", "-loglevel", "error",
           "-progress", "pipe:1", "-nostats", *map(str, args)]
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, text=True, creationflags=_NO_WINDOW)
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                if control is not None and control.cancelled:
                    proc.kill()
                    proc.wait()
                    raise CancelledError()
                if on_progress and duration and line.startswith(("out_time_us=", "out_time_ms=")):
                    try:
                        us = int(line.split("=", 1)[1])
                        on_progress(max(0.0, min(1.0, us / 1e6 / duration)))
                    except ValueError:
                        pass
            proc.wait()
        except BaseException:
            if proc.poll() is None:
                proc.kill()
            raise
        if proc.returncode != 0:
            err.seek(0)
            tail = err.read().decode("utf-8", "replace")[-1500:]
            raise MediaProcessingError(f"ffmpeg failed ({proc.returncode}): {tail}",
                                       "FFmpeg could not process the media. See the log for details.")
