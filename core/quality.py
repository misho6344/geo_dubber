"""Automatic quality control for generated speech and its timing.

Pure functions (no models, no files) so they can be tested. The pipeline calls them:
  * trim_silence     - removes dead air at the start/end of each generated clip, so the dub follows the original rhythm
  * fit_needed       - which lines are still too long for their window once their REAL audio length is known
  * clip_problems    - clipping, silent or truncated-sounding clips, suspicious durations
  * timeline_problems- overlaps between lines of different speakers, lines that spill into the next line
"""
from __future__ import annotations

import numpy as np


def clip_stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return {"peak": 0.0, "rms_db": -120.0, "seconds": 0.0}
    peak = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    return {"peak": peak, "rms_db": 20 * np.log10(max(rms, 1e-9))}


def trim_silence(x: np.ndarray, sr: int, thresh_db: float = -42.0, keep_s: float = 0.06) -> tuple[np.ndarray, float, float]:
    """Remove leading and trailing samples quieter than thresh_db, keeping keep_s on each side.
    Returns (trimmed, seconds_removed_at_start, seconds_removed_at_end)."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return x, 0.0, 0.0
    thr = 10 ** (thresh_db / 20)
    loud = np.flatnonzero(np.abs(x) > thr)
    if loud.size == 0:                       # nothing audible: leave it for the silence check to report
        return x, 0.0, 0.0
    keep = int(keep_s * sr)
    first = max(0, loud[0] - keep)
    last = min(x.size, loud[-1] + 1 + keep)
    return x[first:last], first / sr, (x.size - last) / sr


def fit_needed(durations: dict[int, float], windows: dict[int, float], max_speedup: float, max_spill: float) -> list[int]:
    """Line ids whose generated audio cannot fit even with the largest allowed speed-up and spill.

    durations: {seg_id: generated clip seconds}; windows: {seg_id: original (end - start) seconds}.
    A line fits when duration <= window * max_speedup + max_spill (the same rule as plan_placements)."""
    out = []
    for sid, dur in durations.items():
        win = windows.get(sid)
        if win is None:
            continue
        if dur > win * max_speedup + max_spill:
            out.append(sid)
    return sorted(out)


def clip_problems(clips: dict[int, dict], texts: dict[int, str]) -> list[dict]:
    """clips: {seg_id: {"peak", "rms_db", "duration"}} ; texts: {seg_id: georgian text}."""
    issues = []
    for sid, c in clips.items():
        text = texts.get(sid, "")
        n_chars = len("".join(text.split()))
        if c.get("peak", 0) >= 0.99:
            issues.append({"id": sid, "kind": "clipping", "detail": f"peak {c['peak']:.2f}"})
        if c.get("rms_db", 0) < -50 and n_chars >= 3:
            issues.append({"id": sid, "kind": "silent", "detail": f"rms {c['rms_db']:.0f} dB for '{text[:30]}'"})
        dur = float(c.get("duration", 0))
        # Georgian speech is about 12 characters per second; much less audio than that means words were lost.
        if n_chars >= 15 and dur < n_chars / 20.0:
            issues.append({"id": sid, "kind": "too_short", "detail": f"{dur:.1f}s for {n_chars} characters"})
    return issues


def timeline_problems(placements: list[dict]) -> list[dict]:
    """placements: [{"id", "speaker_id", "start", "end"}]. Reports overlaps between lines of the SAME speaker
    (always a bug) and spill of more than 0.3 s into a line of another speaker."""
    issues = []
    ordered = sorted(placements, key=lambda p: p["start"])
    for a, b in zip(ordered, ordered[1:]):
        overlap = a["end"] - b["start"]
        if overlap <= 0.02:
            continue
        if a["speaker_id"] == b["speaker_id"]:
            issues.append({"id": a["id"], "kind": "same_speaker_overlap", "detail": f"{overlap:.2f}s into line {b['id']}"})
        elif overlap > 0.3:
            issues.append({"id": a["id"], "kind": "spill_into_other", "detail": f"{overlap:.2f}s into line {b['id']}"})
    return issues
