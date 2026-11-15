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
                    min_gap: float = 0.05) -> list[Placement]:
    """items: dicts with id, start, end, duration (TTS clip length), speaker_id.

    Rules: keep the original start; same speaker never overlaps themself; a clip may use
    up to `max_spill` seconds of silence after its original window (never into the next
    line); otherwise speed up to `max_speedup`; anything still too long is placed in full
    (never truncated or dropped) and reported via `overflow`."""
    items = sorted(items, key=lambda i: (i["start"], i["end"]))
    cursor: dict[str, float] = {}
    out: list[Placement] = []
    for idx, it in enumerate(items):
        start = max(it["start"], cursor.get(it["speaker_id"], 0.0))
        nxt = next((j["start"] for j in items[idx + 1:] if j["start"] > start + 1e-6), None)
        limit = it["end"] + max_spill
        if nxt is not None:
            limit = min(limit, max(it["end"], nxt - min_gap))
        avail = max(limit - start, 0.05)
        dur = float(it["duration"])
        speed = 1.0 if dur <= avail else min(dur / avail, max_speedup)
        new_dur = dur / speed
        out.append(Placement(seg_id=it["id"], speaker_id=it["speaker_id"], start=start, duration=new_dur,
                             orig_duration=dur, speed=speed, overflow=max(0.0, new_dur - avail)))
        cursor[it["speaker_id"]] = start + new_dur + min_gap
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
    """Render all clips into one mono WAV block by block (memory stays ~1 block).
    Returns the final track duration (extended if dialogue runs past the video end)."""
    placements = sorted(placements, key=lambda p: p.start)
    total = max([total_duration] + [p.end for p in placements])
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
