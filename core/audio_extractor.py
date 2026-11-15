"""Audio extraction and chunk planning (never loads the whole video into RAM)."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from core.ffmpeg_utils import Control, run_ffmpeg
from core.video_loader import MediaInfo


def extract_full_audio(video: str | Path, out_wav: str | Path, info: MediaInfo,
                       on_progress: Optional[Callable[[float], None]] = None,
                       control: Optional[Control] = None) -> Path:
    """High-quality intermediate: PCM16, source sample rate, up to 2 channels (streamed by ffmpeg)."""
    sr = info.sample_rate or 48000
    ch = 2 if info.channels >= 2 else 1
    run_ffmpeg(["-i", str(video), "-map", "0:a:0", "-vn", "-ac", ch, "-ar", sr, "-c:a", "pcm_s16le", str(out_wav)],
               duration=info.duration, on_progress=on_progress, control=control)
    return Path(out_wav)


def extract_mono16k(src_audio: str | Path, out_wav: str | Path,
                    on_progress: Optional[Callable[[float], None]] = None, duration: Optional[float] = None,
                    control: Optional[Control] = None) -> Path:
    """Compact mono 16 kHz copy used by ASR / diarization / gender estimation."""
    run_ffmpeg(["-i", str(src_audio), "-vn", "-ac", 1, "-ar", 16000, "-c:a", "pcm_s16le", str(out_wav)],
               duration=duration, on_progress=on_progress, control=control)
    return Path(out_wav)


def cut_wav(src: str | Path, start: float, end: float, out: str | Path, control: Optional[Control] = None) -> Path:
    run_ffmpeg(["-ss", f"{start:.3f}", "-t", f"{max(0.01, end - start):.3f}", "-i", str(src), "-c:a", "pcm_s16le", str(out)],
               control=control)
    return Path(out)


def plan_windows(duration: float, chunk_s: float, overlap_s: float) -> list[dict]:
    """Nominal chunks [start,end) plus an overlapped read window around each.

    Segments are attributed to the chunk containing their *midpoint*, so every sentence is
    kept exactly once while still being transcribed with context from both sides."""
    chunk_s = max(30.0, chunk_s)
    out, s = [], 0.0
    while s < duration - 1e-6:
        e = min(duration, s + chunk_s)
        out.append({"index": len(out), "start": s, "end": e,
                    "win_start": max(0.0, s - overlap_s), "win_end": min(duration, e + overlap_s)})
        s = e
    return out


def keep_by_midpoint(segments: list[dict], start: float, end: float, last: bool = False) -> list[dict]:
    res = []
    for sg in segments:
        mid = (sg["start"] + sg["end"]) / 2
        if start <= mid < end or (last and mid >= start):
            res.append(sg)
    return res
