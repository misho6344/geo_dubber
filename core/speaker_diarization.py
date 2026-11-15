"""Speaker diarization, speaker assignment and (heuristic) gender estimation.

Diarization separates VOICES, it does not know character names.
Gender is estimated from median fundamental frequency (F0): a heuristic with a confidence
value, NOT a certainty - the GUI lets the user correct it.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

from core.audio_io import read_wav_block, to_mono, wav_duration
from core.errors import MissingDependencyError, ModelMissingError

log = logging.getLogger(__name__)
Turn = tuple[float, float, str]          # (start, end, speaker label)


class Diarizer(ABC):
    @abstractmethod
    def diarize(self, wav16k: Path) -> list[Turn]:
        ...


class PyannoteDiarizer(Diarizer):
    """Needs `pyannote.audio` and a Hugging Face token with the gated model terms accepted
    (pyannote/speaker-diarization-3.1). Runs locally after the one-time download."""

    def __init__(self, device: str = "cpu", hf_token: str = "", model: str = "pyannote/speaker-diarization-3.1",
                 num_speakers: Optional[int] = None):
        self.device, self.token, self.model, self.num_speakers = device, hf_token, model, num_speakers

    def diarize(self, wav16k: Path) -> list[Turn]:
        try:
            import torch
            from pyannote.audio import Pipeline
        except ImportError as e:
            raise MissingDependencyError(str(e), "pyannote.audio is not installed (run install.bat).")
        try:
            pipe = Pipeline.from_pretrained(self.model, use_auth_token=self.token or None)
        except Exception as e:
            raise ModelMissingError(str(e), "The speaker model could not be loaded. Accept the model terms on "
                                    "huggingface.co/pyannote/speaker-diarization-3.1 and enter your token in Settings.")
        if pipe is None:
            raise ModelMissingError("pipeline None", "The speaker model needs a Hugging Face token (Settings > Models).")
        pipe.to(torch.device(self.device))
        kw = {"num_speakers": self.num_speakers} if self.num_speakers else {}
        ann = pipe(str(wav16k), **kw)
        return [(float(t.start), float(t.end), str(lab)) for t, _, lab in ann.itertracks(yield_label=True)]


class SingleSpeakerDiarizer(Diarizer):
    """Fallback when speaker detection is off/unavailable: everything is one speaker (and says so)."""

    def diarize(self, wav16k: Path) -> list[Turn]:
        return []


def assign_speakers(segments: list[dict], turns: list[Turn], max_gap: float = 1.5) -> list[dict]:
    """Give every segment a stable speaker id ('speaker_1', ...) by maximum time overlap with the turns.
    Ids are numbered by order of first appearance. Segments with no overlapping turn get the nearest turn
    within `max_gap` seconds and are flagged low-confidence."""
    if not turns:
        for sg in segments:
            sg["speaker_id"], sg["speaker_confidence"] = "speaker_1", 0.0
        return segments
    order: dict[str, str] = {}

    def canon(lab: str) -> str:
        if lab not in order:
            order[lab] = f"speaker_{len(order) + 1}"
        return order[lab]

    for t in sorted(turns):
        canon(t[2])
    ts = sorted(turns)
    last = None
    for sg in segments:
        overlap: dict[str, float] = {}
        for a, b, lab in ts:
            if b <= sg["start"]:
                continue
            if a >= sg["end"]:
                break
            overlap[lab] = overlap.get(lab, 0.0) + min(b, sg["end"]) - max(a, sg["start"])
        dur = max(1e-6, sg["end"] - sg["start"])
        if overlap:
            lab, ov = max(overlap.items(), key=lambda kv: kv[1])
            sg["speaker_id"], sg["speaker_confidence"] = canon(lab), round(min(1.0, ov / dur), 3)
        else:
            near = min(ts, key=lambda t: min(abs(t[0] - sg["end"]), abs(t[1] - sg["start"])))
            gap = min(abs(near[0] - sg["end"]), abs(near[1] - sg["start"]))
            if gap <= max_gap:
                sg["speaker_id"], sg["speaker_confidence"] = canon(near[2]), 0.2
            else:
                sg["speaker_id"], sg["speaker_confidence"] = last or canon(ts[0][2]), 0.0
        last = sg["speaker_id"]
    return segments


def merge_minor_speakers(segments: list[dict], min_seconds: float = 6.0) -> dict[str, str]:
    """Speakers with very little speech are usually diarization noise, not characters:
    reassign them to the temporally closest major speaker. Returns {old_id: new_id}."""
    total: dict[str, float] = {}
    for sg in segments:
        total[sg["speaker_id"]] = total.get(sg["speaker_id"], 0.0) + sg["end"] - sg["start"]
    major = {s for s, t in total.items() if t >= min_seconds}
    if not major or len(major) == len(total):
        return {}
    mapping: dict[str, str] = {}
    for minor in set(total) - major:
        centers = [(sg["start"] + sg["end"]) / 2 for sg in segments if sg["speaker_id"] == minor]
        c = float(np.mean(centers))
        best = min((sg for sg in segments if sg["speaker_id"] in major),
                   key=lambda sg: abs((sg["start"] + sg["end"]) / 2 - c))
        mapping[minor] = best["speaker_id"]
    for sg in segments:
        if sg["speaker_id"] in mapping:
            sg["speaker_id"] = mapping[sg["speaker_id"]]
            sg["speaker_confidence"] = min(sg.get("speaker_confidence", 0.0), 0.3)
    return mapping


# --------------------------------------------------------------------------- gender
def estimate_f0(x: np.ndarray, sr: int, fmin: float = 70.0, fmax: float = 380.0) -> Optional[float]:
    """Median F0 over voiced frames using normalised autocorrelation. None if no voiced speech."""
    frame, hop = int(0.04 * sr), int(0.01 * sr)
    if len(x) < frame * 2:
        return None
    lag_min, lag_max = int(sr / fmax), int(sr / fmin)
    thr = 0.08 * float(np.sqrt(np.mean(x ** 2)) + 1e-9)
    f0s = []
    win = np.hanning(frame).astype(np.float32)
    for i in range(0, len(x) - frame - lag_max, hop * 2):
        seg = x[i:i + frame + lag_max]
        f = seg[:frame] * win
        if np.sqrt(np.mean(f ** 2)) < thr:
            continue
        n = frame
        ac = np.correlate(seg[:n + lag_max], f, mode="valid")[: lag_max + 1]
        e0 = float(np.dot(f, f)) + 1e-9
        if lag_max >= len(ac):
            continue
        r = ac / e0
        rng = r[lag_min:lag_max + 1]
        # smallest strong peak (avoids octave errors where 2*period correlates equally well)
        k = int(np.argmax(rng >= 0.88 * rng.max()))
        while k + 1 < len(rng) and rng[k + 1] >= rng[k]:
            k += 1
        lag = lag_min + k
        if r[lag] > 0.45:
            f0s.append(sr / lag)
    if len(f0s) < 4:
        return None
    # Reject extreme outliers before taking median
    med = float(np.median(f0s))
    filtered = [v for v in f0s if abs(v - med) < 70]
    return float(np.median(filtered)) if len(filtered) >= 3 else med


def estimate_spectral_centroid(x: np.ndarray, sr: int) -> Optional[float]:
    """Brightness cue that helps separate similar-pitch male voices."""
    if x is None or x.size < sr // 10:
        return None
    x = x.astype(np.float64)
    x = x - x.mean()
    n = 1
    while n < len(x):
        n *= 2
    n = min(n, 4096)
    if len(x) < n:
        return None
    seg = x[:n] * np.hanning(n)
    spec = np.abs(np.fft.rfft(seg)) + 1e-12
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    return float(np.sum(freqs * spec) / np.sum(spec))


def gender_from_f0(f0: Optional[float], centroid: Optional[float] = None) -> tuple[str, float]:
    """Heuristic gender from median F0 + optional spectral centroid.
    Male typical F0 ~85-160 Hz, female ~165-255 Hz. Confidence drops near the boundary.
    Higher spectral centroid slightly biases toward female when F0 is ambiguous.
    """
    if f0 is None:
        return "unknown", 0.0
    boundary = 158.0  # slightly lower than classic 165 to reduce male→female errors on higher male voices
    conf = float(min(1.0, abs(f0 - boundary) / 50.0))
    gender = "female" if f0 > boundary else "male"
    # When F0 is close to the boundary, use brightness as a weak secondary cue
    if conf < 0.45 and centroid is not None:
        # Female speech usually brighter; typical centroids ~1500-2500 Hz vs male ~800-1800
        if centroid > 1950 and gender == "male":
            gender = "female"
            conf = max(conf, 0.35)
        elif centroid < 1400 and gender == "female":
            gender = "male"
            conf = max(conf, 0.35)
    return gender, round(conf, 2)


def estimate_speaker_genders(segments: list[dict], wav16k: Path, max_segments: int = 30,
                             max_seg_seconds: float = 8.0) -> dict[str, dict]:
    """Per speaker: use their longest segments, median F0 + spectral centroid. Returns speaker info dict."""
    info: dict[str, dict] = {}
    by: dict[str, list[dict]] = {}
    for sg in segments:
        by.setdefault(sg["speaker_id"], []).append(sg)
    for spk, segs in by.items():
        speech = sum(s["end"] - s["start"] for s in segs)
        # Prefer longer, clearer segments; also take a few mid-length ones for robustness
        picked = sorted(segs, key=lambda s: s["end"] - s["start"], reverse=True)[:max_segments]
        f0s = []
        cents = []
        for s in picked:
            dur = min(max_seg_seconds, s["end"] - s["start"])
            if dur < 0.35:
                continue
            blk, sr = read_wav_block(wav16k, s["start"], dur)
            mono = to_mono(blk)
            f = estimate_f0(mono, sr)
            if f:
                f0s.append(f)
            c = estimate_spectral_centroid(mono, sr)
            if c:
                cents.append(c)
        f0 = float(np.median(f0s)) if f0s else None
        cent = float(np.median(cents)) if cents else None
        gender, conf = gender_from_f0(f0, cent)
        # Extra safety: if we have very few F0 samples, lower confidence
        if len(f0s) < 4:
            conf = min(conf, 0.4)
        info[spk] = {"gender": gender, "gender_confidence": conf,
                     "f0": round(f0, 1) if f0 else None,
                     "centroid": round(cent, 0) if cent else None,
                     "speech_seconds": round(speech, 1), "segments": len(segs),
                     "label": "Speaker " + spk.split("_")[-1]}
    return info



def stabilize_speakers_by_pitch(segments: list[dict], wav16k: Path, turns: list[Turn]) -> list[dict]:
    """Re-label each ASR segment by its own F0 against the two pitch clusters, then smooth flips.

    Fixes the common failure where male/female assignment starts correctly but drifts mid-video
    when short pitch windows near the boundary get the wrong label.
    """
    if not segments or not turns:
        return segments
    labs = {t[2] for t in turns}
    if len(labs) < 2:
        return segments
    # Recover approximate centers from turns by sampling midpoints
    samples: dict[str, list[float]] = {lab: [] for lab in labs}
    for a, b, lab in turns:
        mid = (a + b) / 2
        try:
            blk, sr = read_wav_block(wav16k, max(0.0, mid - 0.2), 0.45)
            f0 = estimate_f0(to_mono(blk), sr)
            if f0:
                samples[lab].append(f0)
        except Exception:
            continue
    centers = {}
    for lab, vals in samples.items():
        if vals:
            centers[lab] = float(np.median(vals))
    if len(centers) < 2:
        return segments
    labels = sorted(centers.keys(), key=lambda L: centers[L])  # low pitch first
    lo_lab, hi_lab = labels[0], labels[-1]
    lo_c, hi_c = centers[lo_lab], centers[hi_lab]
    if hi_c - lo_c < 12:
        return segments

    # Map turn labels to stable speaker_1 (low/male-ish) / speaker_2 (high/female-ish)
    id_for = {lo_lab: "speaker_1", hi_lab: "speaker_2"}

    raw: list[str] = []
    confs: list[float] = []
    for sg in segments:
        try:
            dur = min(6.0, max(0.25, sg["end"] - sg["start"]))
            blk, sr = read_wav_block(wav16k, sg["start"], dur)
            f0 = estimate_f0(to_mono(blk), sr)
        except Exception:
            f0 = None
        if f0 is None:
            # keep existing id if already set
            raw.append(sg.get("speaker_id") or id_for[lo_lab])
            confs.append(0.15)
            continue
        d_lo, d_hi = abs(f0 - lo_c), abs(f0 - hi_c)
        pick = id_for[lo_lab] if d_lo <= d_hi else id_for[hi_lab]
        margin = abs(d_lo - d_hi)
        confs.append(round(min(1.0, margin / 40.0), 3))
        raw.append(pick)

    # Temporal majority smooth: isolated flips (A B A -> A A A) over window of 3
    fixed = list(raw)
    for i in range(1, len(fixed) - 1):
        if fixed[i - 1] == fixed[i + 1] and fixed[i] != fixed[i - 1]:
            # only flip if this segment was low confidence
            if confs[i] < 0.35:
                fixed[i] = fixed[i - 1]
    # Second pass: runs of length 1 between same labels
    for i in range(1, len(fixed) - 1):
        if fixed[i - 1] == fixed[i + 1] and fixed[i] != fixed[i - 1]:
            fixed[i] = fixed[i - 1]

    for sg, spk, c in zip(segments, fixed, confs):
        sg["speaker_id"] = spk
        sg["speaker_confidence"] = max(float(sg.get("speaker_confidence", 0)), c)
    return segments




def cluster_segments_by_pitch(segments: list[dict], wav16k: Path, n_speakers: int = 2,
                              min_gap_hz: float = 10.0) -> list[dict]:
    """Label each ASR line with speaker_1..N using F0 + brightness, then smooth turns.

    For 2 men + 1 woman, female is usually high F0; the two men are split by residual
    pitch/brightness difference and by alternating-turn continuity (A-B-A patterns).
    """
    n_speakers = max(1, min(4, int(n_speakers)))
    if not segments:
        return segments
    if n_speakers == 1:
        for sg in segments:
            sg["speaker_id"] = "speaker_1"
            sg["speaker_confidence"] = 1.0
        return segments

    feats: list[tuple[Optional[float], Optional[float]]] = []
    for sg in segments:
        try:
            dur = min(6.0, max(0.35, float(sg["end"]) - float(sg["start"])))
            blk, sr = read_wav_block(wav16k, float(sg["start"]), dur)
            mono = to_mono(blk)
            feats.append((estimate_f0(mono, sr), estimate_spectral_centroid(mono, sr)))
        except Exception:
            feats.append((None, None))

    # Build feature matrix for lines with F0
    idxs = [i for i, (f0, _) in enumerate(feats) if f0 is not None]
    if len(idxs) < max(2, n_speakers):
        for sg in segments:
            sg.setdefault("speaker_id", "speaker_1")
            sg.setdefault("speaker_confidence", 0.1)
        return segments

    f0_arr = np.array([feats[i][0] for i in idxs], dtype=np.float64)
    sc_arr = np.array([feats[i][1] if feats[i][1] is not None else float(np.median([feats[j][1] for j in idxs if feats[j][1] is not None] or [1500.0])) for i in idxs], dtype=np.float64)
    # normalize
    f0_n = (f0_arr - f0_arr.mean()) / (f0_arr.std() + 1e-6)
    sc_n = (sc_arr - sc_arr.mean()) / (sc_arr.std() + 1e-6)
    X = np.stack([f0_n, 0.55 * sc_n], axis=1)

    # k-means
    rng = np.random.default_rng(0)
    # seed centers at F0 quantiles
    qs = np.linspace(12, 88, n_speakers)
    centers = []
    for q in qs:
        fi = int(np.argmin(np.abs(f0_arr - np.percentile(f0_arr, q))))
        centers.append(X[fi].copy())
    centers = np.array(centers, dtype=np.float64)
    labels_idx = np.zeros(len(X), dtype=int)
    for _ in range(25):
        d = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels_idx = d.argmin(axis=1)
        new_c = centers.copy()
        for k in range(n_speakers):
            pts = X[labels_idx == k]
            if len(pts):
                new_c[k] = pts.mean(axis=0)
        if np.allclose(new_c, centers, atol=1e-3):
            centers = new_c
            break
        centers = new_c

    # rank clusters by mean F0 -> speaker_1 lowest pitch
    mean_f0 = []
    for k in range(n_speakers):
        pts = f0_arr[labels_idx == k]
        mean_f0.append(float(pts.mean()) if len(pts) else 9999.0)
    order = list(np.argsort(mean_f0))
    rank = {order[i]: i + 1 for i in range(len(order))}

    raw_lab = [""] * len(segments)
    confs = [0.15] * len(segments)
    for j, i in enumerate(idxs):
        k = int(labels_idx[j])
        raw_lab[i] = f"speaker_{rank[k]}"
        # confidence from distance margin
        dd = ((X[j] - centers) ** 2).sum(axis=1)
        ordered = np.sort(dd)
        margin = float(ordered[1] - ordered[0]) if len(ordered) > 1 else 0.5
        confs[i] = round(min(1.0, margin / 1.5), 3)

    # fill missing with nearest temporal neighbour
    for i in range(len(raw_lab)):
        if raw_lab[i]:
            continue
        left = next((raw_lab[j] for j in range(i - 1, -1, -1) if raw_lab[j]), None)
        right = next((raw_lab[j] for j in range(i + 1, len(raw_lab)) if raw_lab[j]), None)
        raw_lab[i] = left or right or "speaker_1"

    # Temporal continuity: isolated flips with low conf become neighbour
    for _ in range(2):
        for i in range(1, len(raw_lab) - 1):
            if raw_lab[i - 1] == raw_lab[i + 1] and raw_lab[i] != raw_lab[i - 1] and confs[i] < 0.45:
                raw_lab[i] = raw_lab[i - 1]

    # For 3+ speakers: discourage rapid A-A-A collapse of two males —
    # if consecutive lines have very different F0 but same label, and a free male cluster exists, split
    if n_speakers >= 3:
        for i in range(1, len(segments)):
            if not feats[i][0] or not feats[i - 1][0]:
                continue
            if raw_lab[i] == raw_lab[i - 1] and abs(feats[i][0] - feats[i - 1][0]) > 25 and confs[i] < 0.55:
                # switch to the other non-female (not highest pitch) speaker
                female = f"speaker_{n_speakers}"  # highest F0 rank
                alts = [f"speaker_{k}" for k in range(1, n_speakers + 1) if f"speaker_{k}" != raw_lab[i] and f"speaker_{k}" != female]
                if alts:
                    raw_lab[i] = alts[0]

    for sg, sp, c in zip(segments, raw_lab, confs):
        sg["speaker_id"] = sp
        sg["speaker_confidence"] = c
    n_found = len(set(raw_lab))
    log.info("Pitch/brightness clustering: asked %d, found %d speakers", n_speakers, n_found)
    return segments



def pick_preview_window(segments: list[dict], target_seconds: float = 5.0,
                        others: Optional[list[dict]] = None) -> tuple[float, float]:
    """Choose about target_seconds of speech from ONE speaker.

    segments: the speaker's own lines. others: every line of the other speakers. The window never overlaps
    another speaker's line, so the preview holds one voice only (the old version chained lines across the
    gap, which put other characters into the 5-second sample)."""
    if not segments:
        return 0.0, 0.0
    other_spans = [(float(o["start"]), float(o["end"])) for o in (others or [])]

    def clean(a: float, b: float) -> bool:
        return not any(min(b, oe) - max(a, os_) > 0.1 for os_, oe in other_spans)

    ordered = sorted(segments, key=lambda s: float(s["start"]))
    by_len = sorted(ordered, key=lambda s: -(float(s["end"]) - float(s["start"])))
    for s in by_len:
        start, end = float(s["start"]), float(s["end"])
        if end - start < target_seconds:
            break
        mid = (start + end) / 2.0
        a = max(start, mid - target_seconds / 2.0)
        b = min(end, a + target_seconds)
        if clean(a, b):
            return a, b
    # No single line is long enough: grow from the longest line through this speaker's neighbouring lines,
    # stopping at a gap of more than 1.2 s or at any other speaker's speech.
    for s in by_len:
        i = next(k for k, o in enumerate(ordered) if o is s)
        a, b = float(s["start"]), float(s["end"])
        lo = i
        while lo > 0 and b - a < target_seconds:
            prv = ordered[lo - 1]
            if a - float(prv["end"]) > 1.2 or not clean(float(prv["start"]), b):
                break
            a = float(prv["start"])
            lo -= 1
        j = i + 1
        while j < len(ordered) and b - a < target_seconds:
            nxt = ordered[j]
            if float(nxt["start"]) - b > 1.2 or not clean(a, float(nxt["end"])):
                break
            b = float(nxt["end"])
            j += 1
        if b - a >= 0.3:
            return a, min(b, a + target_seconds)
    s = by_len[0]
    a = float(s["start"])
    return a, min(float(s["end"]), a + target_seconds)


def local_two_speaker_turns(wav16k: Path, min_gap_hz: float = 18.0, min_cluster_seconds: float = 1.2) -> list[Turn]:
    """Split a track into up to two speakers using pitch only. No model and no Hugging Face token.

    Returns turns [(start, end, "pitch_low"|"pitch_high")] when the two pitch groups are clearly
    different (typical male + female). Returns [] when the video is one speaker, so the caller
    keeps a single voice instead of inventing a second one.
    """
    try:
        dur = wav_duration(wav16k)
    except Exception:
        log.exception("could not read audio for local speaker split")
        return []
    win, step = 0.50, 0.25
    if dur < win:
        return []
    voiced: list[tuple[float, float, float]] = []
    t = 0.0
    rms_vals: list[float] = []
    windows: list[tuple[float, float, np.ndarray, int, float]] = []
    while t + win <= dur + 1e-3:
        blk, sr = read_wav_block(wav16k, t, win)
        x = to_mono(blk)
        rms = float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0
        windows.append((t, min(dur, t + win), x, sr, rms))
        rms_vals.append(rms)
        t += step
        if len(windows) > 20000:          # safety cap (~80 min at 0.25s step)
            break
    if not rms_vals:
        return []
    thr = max(0.008, float(np.percentile(rms_vals, 55)) * 0.40)
    for a, b, x, sr, rms in windows:
        if rms < thr or x.size < sr // 5:
            continue
        f0 = estimate_f0(x, sr)
        if f0:
            voiced.append((a, b, f0))
    if len(voiced) < 6:
        return []
    vals = np.array([v[2] for v in voiced], dtype=np.float64)
    c_lo = float(np.percentile(vals, 30))
    c_hi = float(np.percentile(vals, 70))
    if abs(c_hi - c_lo) < 1.0:
        return []
    for _ in range(15):
        d_lo = np.abs(vals - c_lo)
        d_hi = np.abs(vals - c_hi)
        low = d_lo <= d_hi
        if low.all() or not low.any():
            return []
        c_lo = float(vals[low].mean())
        c_hi = float(vals[~low].mean())
        if c_lo > c_hi:
            c_lo, c_hi = c_hi, c_lo
    if c_hi - c_lo < min_gap_hz:
        return []

    def lab_of(f0: float) -> str:
        return "pitch_low" if abs(f0 - c_lo) <= abs(f0 - c_hi) else "pitch_high"

    span = {"pitch_low": 0.0, "pitch_high": 0.0}
    for a, b, f0 in voiced:
        span[lab_of(f0)] += b - a
    if min(span.values()) < min_cluster_seconds:
        return []

    turns: list[Turn] = []
    cur_a = cur_b = 0.0
    cur_lab = ""
    for a, b, f0 in voiced:
        lab = lab_of(f0)
        if cur_lab == lab and a - cur_b <= 0.65:
            cur_b = b
        else:
            if cur_lab:
                end = min(cur_b, a)
                if end - cur_a >= 0.30:
                    turns.append((cur_a, end, cur_lab))
            cur_a, cur_b, cur_lab = a, b, lab
    if cur_lab and cur_b - cur_a >= 0.30:
        turns.append((cur_a, cur_b, cur_lab))
    if len({t[2] for t in turns}) < 2:
        return []
    log.info("Local two-voice split: centers %.0f Hz / %.0f Hz, %d turns", c_lo, c_hi, len(turns))
    return turns
