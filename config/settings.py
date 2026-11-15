"""Application paths and user settings (portable: everything lives next to the app)."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

APP_NAME = "GeoDubber"
APP_VERSION = "0.1.1"


def app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    """Root for models/cache/projects/output/logs/voices. Override with GEODUBBER_HOME."""
    env = os.environ.get("GEODUBBER_HOME")
    root = Path(env).expanduser() if env else app_dir()
    root.mkdir(parents=True, exist_ok=True)
    return root


def sub_dir(name: str) -> Path:
    p = data_dir() / name
    p.mkdir(parents=True, exist_ok=True)
    return p


# Processing presets. Each step is faster than before; quality still uses a larger speech model.
# beam>1 is the slow part of Whisper; it is 1 for fast/balanced and 5 for quality, where accuracy matters most
# (the v1 quality preset used beam 1, which produced many garbled and repeated lines).
PRIORITY_PRESETS = {
    "fast": {"asr_model": "tiny", "asr_beam": 1, "nfe_step": 8, "nllb_beam": 1,
             "video_preset": "ultrafast", "tts_workers": 8, "demucs_segment": "6"},
    "balanced": {"asr_model": "base", "asr_beam": 1, "nfe_step": 12, "nllb_beam": 1,
                 "video_preset": "veryfast", "tts_workers": 6, "demucs_segment": "7"},
    "quality": {"asr_model": "small", "asr_beam": 5, "nfe_step": 16, "nllb_beam": 1,
                "video_preset": "veryfast", "tts_workers": 4, "demucs_segment": "7"},
}


@dataclass
class Settings:
    priority: str = "fast"                      # fast | balanced | quality
    asr_model: str = ""                         # empty -> taken from preset
    translation_model: str = "gemini-3.5-flash-lite"   # the model that answered in the v1 logs; the translator rotates on quota errors
    tts_engine: str = "edge_georgian"           # edge_georgian | f5_georgian | piper_ka | mms_kat
    device: str = "auto"                        # auto | cuda | cpu
    preserve_background: bool = True
    auto_speakers: bool = True
    georgian_subtitles: bool = True
    original_subtitles: bool = False
    burn_subtitles: bool = False                # False -> soft (selectable) subtitle track, no re-encode
    video_crf: int = 20
    video_preset: str = "veryfast"
    audio_bitrate: str = "192k"
    output_dir: str = ""
    cache_dir: str = ""
    temp_dir: str = ""
    chunk_minutes: float = 10.0
    chunk_overlap_seconds: float = 5.0
    translation_batch: int = 16                 # lines per Gemini request (JSON batch)
    translation_glossary_file: str = ""         # "Source = Georgian" lines; empty -> <app folder>/glossary.txt
    translation_style_note: str = ""            # e.g. "Sci-fi comedy, dry humour, casual speech; keep profanity strong"
    translation_context_lines: int = 8          # earlier English=>Georgian pairs shown to the model for consistency
    translation_sequential: bool = True         # batches run in order so each one sees the previous translations
    timing_fit_rounds: int = 2                  # re-write lines whose REAL audio is too long, up to this many rounds
    trim_clip_silence: bool = True              # cut dead air from the start/end of each generated line
    voice_qc: bool = True                       # automatic checks: clipping, silent/short clips, overlaps, spill
    edge_character_variants: bool = True        # extra Edge voices (slightly different pitch/speed) for 2nd, 3rd... characters of the same gender
    max_speedup: float = 1.3
    max_spill_seconds: float = 1.5
    voice_volume: float = 1.0
    background_volume: float = 1.0
    original_dialogue_volume: float = 0.0
    duck_background: bool = True
    allow_voice_sharing: bool = True
    min_speaker_seconds: float = 2.0
    # Pre-set character genders before generation: "auto" | "male" | "female"
    speaker_count: int = 0          # 0 = Gemini decides how many characters; else force 1–4
    gemini_speakers: bool = True    # Gemini listens to the audio: number of characters, gender, voice
    speaker1_gender: str = "auto"
    speaker2_gender: str = "auto"
    speaker3_gender: str = "auto"
    speaker4_gender: str = "auto"
    gemini_api_key: str = ""                    # prefer the GEMINI_API_KEY environment variable; a key here is stored in plain text
    hf_token: str = ""                          # needed for gated pyannote models; stored locally only
    app_language: str = "English"
    review_translation: bool = True
    check_updates: bool = False
    # Path to user-supplied timed subtitles (SRT/VTT/TXT). When set, ASR is skipped
    # and the file is used as the dubbing script. If the text is already Georgian,
    # translation is also skipped.
    imported_subtitles: str = ""

    def preset(self) -> dict:
        return PRIORITY_PRESETS.get(self.priority, PRIORITY_PRESETS["fast"])

    def effective_asr_model(self) -> str:
        return self.asr_model or self.preset()["asr_model"]

    def resolved_output_dir(self) -> Path:
        p = Path(self.output_dir) if self.output_dir else sub_dir("output")
        p.mkdir(parents=True, exist_ok=True)
        return p

    def resolved_cache_dir(self) -> Path:
        p = Path(self.cache_dir) if self.cache_dir else sub_dir("cache")
        p.mkdir(parents=True, exist_ok=True)
        return p

    def resolved_temp_dir(self) -> Path:
        p = Path(self.temp_dir) if self.temp_dir else sub_dir("cache") / "tmp"
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ---- persistence -------------------------------------------------
    @staticmethod
    def path() -> Path:
        return data_dir() / "settings.json"

    @classmethod
    def load(cls) -> "Settings":
        s = cls()
        try:
            raw = json.loads(cls.path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return s
        known = {f.name for f in fields(cls)}
        for k, v in raw.items():
            if k in known:
                setattr(s, k, v)
        return s

    def save(self) -> None:
        tmp = self.path().with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path())
