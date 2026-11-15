"""Mux the dubbed audio (and subtitles) into the final MP4."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from core.errors import MediaProcessingError
from core.ffmpeg_utils import Control, have_filter, run_ffmpeg
from core.video_loader import MediaInfo

_COPYABLE = {"h264", "hevc", "mpeg4", "av1", "vp9"}


def _esc_filter_path(p: Path) -> str:
    s = str(p.resolve()).replace("\\", "/")
    return s.replace(":", "\\:").replace("'", "\\'")


def export_video(video: str | Path, audio: str | Path, out_mp4: str | Path, info: MediaInfo,
                 srt: Optional[str | Path] = None, burn: bool = False, crf: int = 20, preset: str = "veryfast",
                 on_progress: Optional[Callable[[float], None]] = None, control: Optional[Control] = None,
                 sub_lang: str = "kat") -> Path:
    """Default is a fast stream-copy of the original video (no quality loss, keeps resolution/fps) with
    the new AAC track and an optional *soft* subtitle track. Burn-in re-encodes with libx264."""
    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_mp4.with_name(out_mp4.stem + ".partial.mp4")
    args = ["-i", str(video), "-i", str(audio)]
    if srt and not burn:
        args += ["-i", str(srt)]
    args += ["-map", "0:v:0", "-map", "1:a:0"]
    if srt and not burn:
        args += ["-map", "2:0"]
    reencode = burn and srt is not None or info.video_codec not in _COPYABLE
    if burn and srt:
        if not have_filter("subtitles"):
            raise MediaProcessingError("libass missing", "This FFmpeg build cannot burn subtitles (needs libass). "
                                       "Disable 'burn subtitles' to use a soft subtitle track.")
        style = "FontName=Sylfaen,FontSize=22,Outline=2"
        args += ["-vf", f"subtitles='{_esc_filter_path(Path(srt))}':force_style='{style}'"]
    if reencode:
        args += ["-c:v", "libx264", "-preset", preset, "-crf", crf, "-pix_fmt", "yuv420p"]
    else:
        args += ["-c:v", "copy"]
    args += ["-c:a", "copy"]                       # audio is already AAC from the mixer
    if srt and not burn:
        args += ["-c:s", "mov_text", f"-metadata:s:s:0", f"language={sub_lang}"]
    args += ["-metadata:s:a:0", f"language={sub_lang}", "-movflags", "+faststart", str(tmp)]
    try:
        run_ffmpeg(args, duration=info.duration, on_progress=on_progress, control=control)
        tmp.replace(out_mp4)
    except BaseException:
        tmp.unlink(missing_ok=True)      # interrupted export never leaves a half-written final file
        raise
    return out_mp4
