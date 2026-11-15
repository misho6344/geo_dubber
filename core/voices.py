"""Voice library and speaker->voice assignment.

A *voice* is either built into a TTS engine (single fixed voice) or a user-supplied
reference recording used by a reference-conditioned engine (e.g. Georgian F5-TTS).
Reference recordings must be authorised by their speaker / licence holder: adding one
requires an explicit consent flag, and voices without it are never used.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from config.settings import sub_dir
from core.audio_io import read_wav, to_mono, write_wav, resample
from core.errors import VoiceUnavailableError
from core.ffmpeg_utils import run_ffmpeg


@dataclass
class Voice:
    id: str
    name: str
    gender: str                      # "male" | "female"
    engine: str                      # which TTS engine can use it ("*" = any reference engine)
    ref_audio: str = ""              # absolute path (reference engines)
    ref_text: str = ""               # exact transcript of the reference audio
    builtin: bool = False
    consent: bool = False            # user confirmed they may use this recording
    note: str = ""                   # licence / source note
    builtin_params: dict = field(default_factory=dict)  # e.g. {"speaker_id": 0}

    def fingerprint(self) -> str:
        h = hashlib.sha1()
        h.update(f"{self.id}|{self.engine}|{self.ref_text}|{json.dumps(self.builtin_params, sort_keys=True)}".encode())
        if self.ref_audio and Path(self.ref_audio).exists():
            h.update(Path(self.ref_audio).read_bytes())
        return h.hexdigest()[:16]


def voices_dir() -> Path:
    return sub_dir("voices")


def add_reference_voice(name: str, gender: str, source_audio: str | Path, ref_text: str,
                        consent: bool, note: str = "", engine: str = "*",
                        max_seconds: float = 12.0, root: Optional[Path] = None) -> Voice:
    """Import a reference clip: converted to mono 24 kHz WAV, trimmed, stored with metadata."""
    if gender not in ("male", "female"):
        raise ValueError("gender must be 'male' or 'female'")
    if not consent:
        raise VoiceUnavailableError("consent not confirmed",
                                    "You must confirm that you are allowed to use this recording as a voice reference.")
    root = root or voices_dir()
    vid = re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_") or "voice"
    d = root / vid
    n = 2
    while d.exists():
        d = root / f"{vid}_{n}"
        n += 1
    d.mkdir(parents=True)
    ref = d / "ref.wav"
    run_ffmpeg(["-i", str(source_audio), "-t", str(max_seconds), "-ac", "1", "-ar", "24000",
                "-c:a", "pcm_s16le", str(ref)])
    v = Voice(id=d.name, name=name, gender=gender, engine=engine, ref_audio=str(ref),
              ref_text=ref_text.strip(), consent=True, note=note)
    (d / "voice.json").write_text(json.dumps(asdict(v), indent=2, ensure_ascii=False), encoding="utf-8")
    return v


def load_user_voices(root: Optional[Path] = None) -> list[Voice]:
    root = root or voices_dir()
    out = []
    for meta in sorted(root.glob("*/voice.json")):
        try:
            raw = json.loads(meta.read_text(encoding="utf-8"))
            v = Voice(**raw)
            ref = meta.parent / "ref.wav"
            if not ref.exists():
                for ext in ("wav", "mp3", "flac", "m4a", "ogg"):
                    cand = meta.parent / f"ref.{ext}"
                    if cand.exists():
                        ref = cand
                        break
            v.ref_audio = str(ref)   # keep portable if the folder moved (nino is stored as ref.mp3)
            out.append(v)
        except (OSError, ValueError, TypeError):
            continue
    return out


def delete_voice(voice_id: str, root: Optional[Path] = None) -> None:
    d = (root or voices_dir()) / voice_id
    if d.exists():
        shutil.rmtree(d)


def usable_voices(all_voices: list[Voice], engine: str) -> list[Voice]:
    res = []
    for v in all_voices:
        if v.engine not in ("*", engine):
            continue
        if not v.builtin and not v.consent:
            continue
        if not v.builtin and not Path(v.ref_audio).exists():
            continue
        res.append(v)
    return res


def assign_voices(speakers: dict[str, dict], voices: list[Voice], current: Optional[dict[str, str]] = None,
                  allow_sharing: bool = False) -> tuple[dict[str, str], list[str]]:
    """Return ({speaker_id: voice_id}, warnings).

    * Existing valid assignments are kept only if the voice gender still matches the speaker gender.
    * Gender is strictly respected; unknown gender is warned about.
    * Distinct speakers get distinct voices. If none is left, the speaker stays UNASSIGNED
      (with a warning) unless `allow_sharing` is on — we never silently reuse a voice.
    * Low-confidence gender detections get a stronger warning so the user reviews them.
    """
    current = dict(current or {})
    by_id = {v.id: v for v in voices}
    result: dict[str, str] = {}
    warnings: list[str] = []
    used: set[str] = set()

    # Keep existing assignments only when gender still matches
    for spk, vid in current.items():
        if spk not in speakers or vid not in by_id:
            continue
        g = speakers[spk].get("gender", "unknown")
        v = by_id[vid]
        if g in ("male", "female") and v.gender != g:
            # Mismatch: drop so it can be re-assigned correctly
            continue
        result[spk] = vid
        used.add(vid)

    order = sorted((s for s in speakers if s not in result),
                   key=lambda s: -float(speakers[s].get("speech_seconds", 0.0)))
    for spk in order:
        info = speakers[spk]
        g = info.get("gender", "unknown")
        conf = float(info.get("gender_confidence") or 0)
        label = info.get("label", spk)
        if g not in ("male", "female"):
            warnings.append(f"{label}: gender could not be determined (listen to original and set it).")
            pool = voices
        else:
            pool = [v for v in voices if v.gender == g]
            if conf < 0.4:
                warnings.append(
                    f"{label}: gender is uncertain (F0 {info.get('f0')} Hz) — please verify by listening."
                )
        free = [v for v in pool if v.id not in used]
        if free:
            result[spk] = free[0].id
            used.add(free[0].id)
        elif pool and allow_sharing:
            result[spk] = pool[0].id
            warnings.append(f"{label}: not enough distinct {g or ''} voices; sharing '{pool[0].name}'.")
        else:
            warnings.append(f"{label}: no free {g if g in ('male','female') else ''} Georgian voice available. "
                            f"Add more voices (Settings > Voices) or assign one manually.")
    return result, warnings
