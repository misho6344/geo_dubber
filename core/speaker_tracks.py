"""Keep every character's voice stable over the whole film.

Problem this solves: speaker labels were decided per line (Gemini's per-line answer or a pitch clustering
that can flip). One wrong line sent that line to the wrong voice, and a man could speak with a woman's voice
later in the film. The fix works in four steps and only changes lines where the measured pitch clearly
contradicts the character's gender:

  1. Measure the median F0 of each line (at most MAX_F0_SECONDS of audio).
  2. Give each line an effective gender: its own pitch when that is clear, otherwise the majority of the
     clear lines within SMOOTH_RADIUS lines in time (so an unclear line does not flip on its own).
  3. Check each character's gender against the voiced duration of its lines. Only a strong, consistent
     disagreement (FLIP_SHARE of the speech, and a median pitch clearly on the other side) changes the
     character's gender, and that is always reported as a warning.
  4. Move a line to a character of the matching gender when its own speaker has the other gender. Among
     same-gender characters, a line moves only when it is clearly closer to another character's pitch.

Pure functions (apart from reading the audio) so they can be tested. The pipeline calls `stabilise`.
"""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np

from core.audio_io import read_wav_block, to_mono
from core.speaker_diarization import estimate_f0

BOUNDARY_HZ = 165.0      # male below, female above
MIN_CONF = 0.5           # a line needs |F0 - boundary| >= 20 Hz to count on its own
CONF_SCALE_HZ = 40.0     # confidence reaches 1.0 at 40 Hz from the boundary
MAX_F0_SECONDS = 3.0     # audio read per line for pitch
MIN_LINE_SECONDS = 0.35  # shorter lines are not measured
SMOOTH_RADIUS = 3        # neighbouring lines used to fill in unclear lines
FLIP_SHARE = 0.75        # share of a character's voiced time that must contradict its gender
FLIP_MIN_SECONDS = 3.0   # voiced time needed before a character's gender may change
FLIP_MARGIN_HZ = 15.0    # median pitch must be this far on the other side of the boundary
MOVE_CUR_LOG = 0.15      # same-gender move: current character must be off by about 2.5 semitones...
MOVE_RATIO = 0.5         # ...and the other character must be at most half as far off
ROUNDS = 2               # re-measure character pitch after moving lines
OCTAVE_TOL = 0.06        # within 6% of an exact octave counts as a pitch-tracker octave error
MAX_MOVE_SHARE = 0.2     # more moves than this share of lines means the pitch data is unreliable


def line_gender(f0: Optional[float]) -> tuple[Optional[str], float]:
    """(gender, confidence) for one line's median F0. Confidence grows with distance from the boundary."""
    if f0 is None or f0 <= 0:
        return None, 0.0
    gender = "female" if f0 > BOUNDARY_HZ else "male"
    conf = min(1.0, abs(f0 - BOUNDARY_HZ) / CONF_SCALE_HZ)
    return gender, round(conf, 2)


def segment_f0(wav16k: Path, segs: list[dict]) -> list[Optional[float]]:
    """Median F0 per segment, in the same order as segs. None when the line is too short or unvoiced."""
    out: list[Optional[float]] = []
    for s in segs:
        dur = min(MAX_F0_SECONDS, float(s["end"]) - float(s["start"]))
        if dur < MIN_LINE_SECONDS:
            out.append(None)
            continue
        blk, sr = read_wav_block(wav16k, float(s["start"]), dur)
        out.append(estimate_f0(to_mono(blk), sr))
    return out


def effective_genders(f0s: list[Optional[float]]) -> list[Optional[str]]:
    """Own gender for clear lines; for unclear lines, the majority of clear lines nearby in time (or None)."""
    own = [line_gender(f) for f in f0s]
    clear = [g if (g is not None and c >= MIN_CONF) else None for g, c in own]
    eff: list[Optional[str]] = []
    for i, g in enumerate(clear):
        if g is not None:
            eff.append(g)
            continue
        lo, hi = max(0, i - SMOOTH_RADIUS), min(len(clear), i + SMOOTH_RADIUS + 1)
        votes = Counter(clear[j] for j in range(lo, hi) if j != i and clear[j] is not None).most_common(2)
        if votes and (len(votes) == 1 or votes[0][1] > votes[1][1]):
            eff.append(votes[0][0])
        else:
            eff.append(None)
    return eff


def speaker_medians(speakers: list[str], f0s: list[Optional[float]]) -> dict[str, float]:
    by: dict[str, list[float]] = {}
    for spk, f in zip(speakers, f0s):
        if f:
            by.setdefault(spk, []).append(float(f))
    return {k: float(np.median(v)) for k, v in by.items() if v}


def check_speaker_genders(speakers: list[str], f0s: list[Optional[float]], durs: list[float],
                          eff: list[Optional[str]], info: dict[str, dict]) -> list[dict]:
    """Change a character's gender only on strong, consistent acoustic disagreement. Mutates info."""
    changes = []
    for spk, meta in info.items():
        cur = meta.get("gender")
        if cur not in ("male", "female") or meta.get("gender_source") == "user":
            continue
        other = "male" if cur == "female" else "female"
        total = against = 0.0
        pitches = []
        for s, f, d, g in zip(speakers, f0s, durs, eff):
            if s != spk or not f or g is None:
                continue
            total += d
            pitches.append(float(f))
            if g == other:
                against += d
        if total < FLIP_MIN_SECONDS or not pitches:
            continue
        share = against / total
        med = float(np.median(pitches))
        beyond = (med - BOUNDARY_HZ) if other == "female" else (BOUNDARY_HZ - med)
        if share >= FLIP_SHARE and beyond >= FLIP_MARGIN_HZ:
            meta["gender"] = other
            meta["gender_source"] = "pitch_check"
            meta["gender_confidence"] = round(min(0.9, share), 2)
            changes.append({"speaker": spk, "from": cur, "to": other, "share": round(share, 2),
                            "median_f0": round(med, 1)})
    return changes


def _octave_explains(f: float, ref: float) -> bool:
    """True when f is a half or double of the character's pitch (a common pitch-tracker error)."""
    return any(abs(math.log(f / (ref * k))) < OCTAVE_TOL for k in (0.5, 2.0))


def reassign_lines(speakers: list[str], f0s: list[Optional[float]], eff: list[Optional[str]],
                   info: dict[str, dict]) -> tuple[list[str], int, int]:
    """Return (new speaker per line, lines moved, lines that had no matching character to move to)."""
    medians = speaker_medians(speakers, f0s)
    gender_of = {k: v.get("gender") for k, v in info.items()}
    out = list(speakers)
    moved = unresolved = 0
    for i, (spk, f, g) in enumerate(zip(speakers, f0s, eff)):
        if not f or g is None:
            continue
        cur_g = gender_of.get(spk)
        if cur_g not in ("male", "female"):
            continue
        same = [k for k, v in gender_of.items() if v == g and k in medians]
        if g != cur_g:
            if not same:
                unresolved += 1
                continue
            if spk in medians and _octave_explains(f, medians[spk]):
                continue   # half/double pitch error: the line still matches its own character
            out[i] = min(same, key=lambda k: abs(math.log(f / medians[k])))
            moved += 1
            continue
        if spk not in medians or len(same) < 2:
            continue
        d_cur = abs(math.log(f / medians[spk]))
        if d_cur < MOVE_CUR_LOG:
            continue
        best = min(same, key=lambda k: abs(math.log(f / medians[k])))
        if best != spk and abs(math.log(f / medians[best])) < MOVE_RATIO * d_cur:
            out[i] = best
            moved += 1
    return out, moved, unresolved


def stabilise(segs: list[dict], wav16k: Path, info: dict[str, dict]) -> tuple[dict[str, dict], list[str]]:
    """Make character assignment stable over time. Updates segs[i]["speaker_id"] in place.

    Returns (updated character info, messages for the user). info is the state["speakers"] dict whose
    genders have already been set (pitch, Gemini, or user)."""
    msgs: list[str] = []
    if not segs or not info:
        return info, msgs
    f0s = segment_f0(wav16k, segs)
    speakers = [str(s["speaker_id"]) for s in segs]
    durs = [float(s["end"]) - float(s["start"]) for s in segs]
    eff = effective_genders(f0s)

    for ch in check_speaker_genders(speakers, f0s, durs, eff, info):
        msgs.append(f"{info[ch['speaker']].get('label', ch['speaker'])}: most of the voiced speech is "
                    f"{ch['to']} ({ch['median_f0']:.0f} Hz median), so its gender was changed from "
                    f"{ch['from']} to {ch['to']}. Check this character in Review.")

    unresolved = 0
    trial = list(speakers)
    trial_moved = 0
    for _ in range(ROUNDS):
        trial, moved, unresolved = reassign_lines(trial, f0s, eff, info)
        trial_moved += moved
        if moved == 0:
            break
    measured = sum(1 for f in f0s if f)
    if measured and trial_moved > MAX_MOVE_SHARE * measured:
        msgs.append(f"The pitch check would move {trial_moved} of {measured} measured lines, which is too many to "
                    "be safe, so no voices were changed. Gemini's speaker analysis (add the API key) or Review "
                    "is needed to fix the characters.")
        trial = speakers
        unresolved = 0
    else:
        speakers = trial
    changed = 0
    for s, spk in zip(segs, speakers):
        if s["speaker_id"] != spk:
            s["speaker_id"] = spk
            s["speaker_confidence"] = min(float(s.get("speaker_confidence") or 0.9), 0.8)
            changed += 1
    if changed:
        msgs.append(f"Pitch check moved {changed} line(s) to a character of the matching gender, "
                    "so no line keeps the wrong voice.")
    if unresolved:
        msgs.append(f"{unresolved} line(s) sound like the other gender but no character of that gender exists. "
                    "They keep their voice; check them in Review.")

    new_info: dict[str, dict] = {}
    f_med = speaker_medians(speakers, f0s)
    for spk in sorted(set(speakers)):
        lines = [i for i, s in enumerate(speakers) if s == spk]
        meta = dict(info.get(spk, {}))
        meta.setdefault("gender", "unknown")
        meta.setdefault("gender_confidence", 0.0)
        meta["f0"] = round(f_med[spk], 1) if spk in f_med else meta.get("f0")
        meta["speech_seconds"] = round(sum(durs[i] for i in lines), 1)
        meta["segments"] = len(lines)
        meta["label"] = "Speaker " + spk.split("_")[-1]
        new_info[spk] = meta
    return new_info, msgs
