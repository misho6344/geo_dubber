"""Timeline placement of Georgian clips and incremental rendering of the dub track."""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from config.settings import sub_dir
from core.audio_io import WavWriter, read_wav, resample, to_mono, wav_duration
from core.ffmpeg_utils import Control, run_ffmpeg


@dataclass
class Placement:
    seg_id: int
    speaker_id: str
    start: float          # where the clip starts on the timeline
    duration: float       # clip duration AFTER speed-up
    orig_duration: float
    speed: float
    overflow: float       # seconds the clip runs past its available window (0 = fits)
    audio_path: str = ""

    @property
    def end(self) -> float:
        return self.start + self.duration


def plan_placements(items: list[dict], max_speedup: float = 1.3, max_spill: float = 1.5,
                    min_gap: float = 0.05, max_shift: float = 0.35) -> list[Placement]:
    """Place clips without overlapping dialogue from the same speaker.

    Different speakers may overlap (as they can in the source film). A clip may
    spill into silence after its subtitle window and is sped up only when needed
    to fit before that speaker's next line. If it still cannot fit, it is kept
    in full and its overflow is reported instead of silently truncating speech.

    max_shift is retained for call-site compatibility; preventing self-overlap
    takes priority over limiting the delay of a later line.
    """
    if not items:
        return []
    items = sorted(
        items,
        key=lambda item: (float(item["start"]), float(item["end"]), str(item.get("speaker_id", ""))),
    )
    cursor: dict[str, float] = {}
    out: list[Placement] = []
    gap = max(0.0, float(min_gap))
    spill = max(0.0, float(max_spill))
    speed_limit = max(1.0, float(max_speedup))

    for idx, item in enumerate(items):
        speaker = str(item.get("speaker_id") or "speaker_1")
        original_start = max(0.0, float(item["start"]))
        original_end = max(original_start, float(item["end"]))
        start = original_start
        if speaker in cursor:
            # Never place two clips belonging to one speaker on top of each other.
            start = max(start, cursor[speaker] + gap)

        next_same_start = next(
            (
                float(other["start"])
                for other in items[idx + 1:]
                if str(other.get("speaker_id") or "speaker_1") == speaker
                and float(other["start"]) > original_start + 1e-6
            ),
            None,
        )
        limit = original_end + spill
        if next_same_start is not None:
            # Other speakers are allowed to overlap naturally and must not force
            # this clip to be shortened. Only this speaker's next line is a boundary.
            limit = min(limit, max(original_end, next_same_start - gap))

        available = max(0.05, limit - start)
        duration = max(0.0, float(item["duration"]))
        speed = 1.0 if duration <= available or duration <= 0 else min(
            duration / available, speed_limit
        )
        fitted_duration = duration / speed if speed > 0 else duration
        overflow = max(0.0, fitted_duration - available)

        out.append(Placement(
            seg_id=item["id"],
            speaker_id=speaker,
            start=start,
            duration=fitted_duration,
            orig_duration=duration,
            speed=speed,
            overflow=overflow,
        ))
        cursor[speaker] = start + fitted_duration

    return out

def apply_speed(src: str | Path, speed: float, cache_dir: Optional[Path] = None,
                control: Optional[Control] = None) -> Path:
    """Pitch-preserving tempo change via ffmpeg atempo (cached)."""
    src = Path(src)
    if speed <= 1.005:
        return src
    d = cache_dir or (sub_dir("cache") / "speed")
    d.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{src}|{src.stat().st_mtime_ns}|{speed:.4f}".encode()).hexdigest()
    out = d / f"{key}.wav"
    if not out.exists():
        run_ffmpeg(["-i", str(src), "-filter:a", f"atempo={speed:.4f}", "-c:a", "pcm_s16le", str(out)], control=control)
    return out


def _load_clip(path: str, sr: int, gain: float = 1.0, fade_s: float = 0.01) -> np.ndarray:
    x, s = read_wav(path)
    x = resample(to_mono(x), s, sr) * gain
    n = min(int(fade_s * sr), len(x) // 2)
    if n > 0:
        ramp = np.linspace(0, 1, n, dtype=np.float32)
        x[:n] *= ramp
        x[-n:] *= ramp[::-1]
    return x


def render_dub_track(placements: list[Placement], out_path: str | Path, total_duration: float,
                     sr: int = 24000, block_s: float = 30.0, gain: float = 1.0,
                     on_progress: Optional[Callable[[float], None]] = None,
                     control: Optional[Control] = None) -> float:
    """Render clips block-by-block and extend beyond source duration when needed.
    Returns the greater of the source duration and the final clip end."""
    placements = sorted(placements, key=lambda p: p.start)
    max_p_end = max([p.end for p in placements], default=0.0)
    # Never trim the final spoken line when it extends beyond the source video.
    total = max(max(0.0, float(total_duration)), max_p_end)
    n_blocks = max(1, math.ceil(total / block_s))
    loaded: dict[int, np.ndarray] = {}
    with WavWriter(out_path, sr, 1) as w:
        for b in range(n_blocks):
            if control:
                control.checkpoint()
            b0, b1 = b * block_s, min(total, (b + 1) * block_s)
            s0, s1 = int(round(b0 * sr)), int(round(b1 * sr))
            buf = np.zeros(s1 - s0, dtype=np.float32)
            for p in placements:
                if p.start >= b1:
                    break
                if p.end <= b0:
                    loaded.pop(id(p), None)
                    continue
                clip = loaded.get(id(p))
                if clip is None:
                    clip = loaded[id(p)] = _load_clip(p.audio_path, sr, gain)
                c0 = int(round(p.start * sr))
                lo, hi = max(c0, s0), min(c0 + len(clip), s1)
                if hi > lo:
                    buf[lo - s0:hi - s0] += clip[lo - c0:hi - c0]
            w.write(np.clip(buf, -1.0, 1.0))
            if on_progress:
                on_progress((b + 1) / n_blocks)
    return total