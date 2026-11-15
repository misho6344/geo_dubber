"""Final audio mix (dub + background + optional original dialogue) with ducking and loudness control."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from core.ffmpeg_utils import Control, run_ffmpeg

_ST = "aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo"


def build_filter(has_background: bool, has_original: bool, voice_vol: float, bg_vol: float,
                 orig_vol: float, duck: bool, normalize: bool = True) -> str:
    """Input 0 = Georgian dub track, 1 = background (if any), then original dialogue (if any)."""
    parts, mix_in = [], []
    bg_idx = 1 if has_background else None
    orig_idx = (2 if has_background else 1) if has_original else None

    use_duck = duck and has_background
    if use_duck:
        parts.append(f"[0:a]{_ST},volume={voice_vol:.3f},asplit=2[dub][dubkey]")
    else:
        parts.append(f"[0:a]{_ST},volume={voice_vol:.3f}[dub]")
    mix_in.append("[dub]")

    if has_background:
        parts.append(f"[{bg_idx}:a]{_ST},volume={bg_vol:.3f}[bg]")
        if use_duck:
            # background dips ~ -9 dB while the Georgian voice speaks, then recovers smoothly
            parts.append("[bg][dubkey]sidechaincompress=threshold=0.03:ratio=3:attack=30:release=500:makeup=1[bgd]")
            mix_in.append("[bgd]")
        else:
            mix_in.append("[bg]")
    if has_original:
        parts.append(f"[{orig_idx}:a]{_ST},volume={orig_vol:.3f}[orig]")
        mix_in.append("[orig]")

    tail = f"{''.join(mix_in)}amix=inputs={len(mix_in)}:normalize=0:duration=longest:dropout_transition=0"
    tail += ",alimiter=limit=0.95:level=disabled"
    if normalize:
        tail += ",loudnorm=I=-16:TP=-1.5:LRA=11"
    parts.append(tail + "[out]")
    return ";".join(parts)


def mix_audio(dub_wav: str | Path, out_audio: str | Path, background: Optional[str | Path] = None,
              original: Optional[str | Path] = None, voice_vol: float = 1.0, bg_vol: float = 1.0,
              orig_vol: float = 0.0, duck: bool = True, bitrate: str = "192k", duration: Optional[float] = None,
              on_progress: Optional[Callable[[float], None]] = None, control: Optional[Control] = None) -> Path:
    inputs = ["-i", str(dub_wav)]
    if background:
        inputs += ["-i", str(background)]
    use_orig = original is not None and orig_vol > 0.001
    if use_orig:
        inputs += ["-i", str(original)]
    flt = build_filter(bool(background), use_orig, voice_vol, bg_vol, orig_vol, duck)
    run_ffmpeg([*inputs, "-filter_complex", flt, "-map", "[out]", "-c:a", "aac", "-b:a", bitrate, "-ar", 48000,
                "-ac", 2, str(out_audio)], duration=duration, on_progress=on_progress, control=control)
    return Path(out_audio)
