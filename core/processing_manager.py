"""The dubbing pipeline: staged, checkpointed, pausable, resumable."""
from __future__ import annotations

import json
import logging
import os
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

from config.settings import APP_VERSION, Settings, data_dir
from core import audio_extractor as AX
from core import gemini_speakers as GS
from core import speaker_diarization as SD
from core import speaker_tracks as ST
from core import subtitles as SUB
from core import video_loader as VL
from core import quality as Q
from core.audio_io import read_wav, to_mono, wav_duration, write_wav
from core.audio_mixer import mix_audio
from core.audio_synchronizer import apply_speed, plan_placements, render_dub_track
from core.background_separator import Separator, create_separator
from core.errors import CancelledError, GeoDubberError, VoiceUnavailableError
from core.ffmpeg_utils import Control
from core.georgian_tts import TTSBackend, create_backend, synthesize_cached
from core.hardware import check_disk_space, resolve_device
from core.speech_recognition import FasterWhisperRecognizer, Recognizer, merge_fragments, transcribe_chunked
from core.subtitles import read_subtitle_file
from core.translation import (NLLBTranslator, Translator, estimate_speech_seconds, is_valid_georgian,
                              load_glossary, translate_segments)
from core.video_exporter import export_video
from core.voices import Voice, assign_voices, load_user_voices, usable_voices

log = logging.getLogger(__name__)

# key, label, weight (share of overall progress)
STAGES = [
    ("extract", "Audio extraction", 5),
    ("asr", "Speech recognition", 25),
    ("speakers", "Speaker detection", 8),
    ("translate", "Translation", 10),
    ("review", "Review translation & voices", 0),
    ("tts", "Voice generation", 30),
    ("fit", "Timing fit (re-write lines that are too long)", 3),
    ("separate", "Background separation", 8),
    ("sync", "Audio synchronization", 5),
    ("mix", "Final audio mix", 4),
    ("export", "Final export", 5),
]
BG_FALLBACK_GAIN = 0.25      # original track level under the dub when no separation model is available


def middle_speaker_gender(info: dict) -> tuple[str, float]:
    """Gender of a speaker that is neither the lowest nor the highest pitch (no Gemini answer available).
    Uses the speaker's own median F0 when it was measured; otherwise 'unknown' is replaced by a male/female
    guess from the pitch threshold used everywhere else (165 Hz)."""
    f0 = info.get("f0")
    if f0:
        return ("male" if float(f0) < 165.0 else "female"), 0.6
    if info.get("gender") in ("male", "female"):
        return info["gender"], float(info.get("gender_confidence") or 0.5)
    return "male", 0.3


@dataclass
class Progress:
    stage: str
    stage_label: str
    stage_fraction: float
    overall: float
    message: str = ""
    eta_seconds: Optional[float] = None       # only for the CURRENT stage, only when it can be estimated
    speed: str = ""
    stage_status: dict = field(default_factory=dict)


@dataclass
class Backends:
    recognizer: Recognizer
    diarizer: SD.Diarizer
    translator: Translator
    tts: TTSBackend
    separator: Separator


def default_backends(settings: Settings, project_dir: Path) -> tuple[Backends, str]:
    device, note = resolve_device(settings.device)
    pr = settings.preset()
    be = Backends(
        recognizer=FasterWhisperRecognizer(settings.effective_asr_model(), device, pr["asr_beam"]),
        diarizer=(SD.PyannoteDiarizer(device, settings.hf_token) if settings.auto_speakers else SD.SingleSpeakerDiarizer()),
        translator=NLLBTranslator(settings.translation_model, device, pr["nllb_beam"], settings.translation_batch,
                                  getattr(settings, "gemini_api_key", ""),
                                  style_note=settings.translation_style_note,
                                  context_lines=settings.translation_context_lines,
                                  sequential=settings.translation_sequential),
        tts=create_backend(settings.tts_engine, settings, device),
        separator=create_separator(device, settings.resolved_temp_dir(), segment=str(pr.get("demucs_segment") or "8")),
    )
    return be, f"{device} {note}".strip()


class Pipeline:
    def __init__(self, project_dir: str | Path, settings: Settings, backends: Backends,
                 control: Optional[Control] = None, on_progress: Optional[Callable[[Progress], None]] = None,
                 on_review: Optional[Callable[["Pipeline"], None]] = None, cache_root: Optional[Path] = None,
                 extra_voices: Optional[list[Voice]] = None,
                 on_characters: Optional[Callable[[int, list], None]] = None):
        self.dir = Path(project_dir)
        self.work = self.dir / "work"
        self.work.mkdir(parents=True, exist_ok=True)
        self.s, self.be = settings, backends
        self.control = control or Control()
        self.on_progress, self.on_review = on_progress, on_review
        self.cache_root = cache_root
        self.extra_voices = extra_voices
        self.on_characters = on_characters       # called with (count, [genders]) once the characters are known
        self.state_path = self.dir / "state.json"
        self.state = self._load()
        self._stage_key, self._stage_t0 = "", 0.0

    # ----------------------------------------------------------- state
    def _load(self) -> dict:
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"version": 1, "done": [], "asr_chunks": {}, "segments": [], "speakers": {}, "assignments": {},
                    "warnings": [], "tts": {}, "timings": {}, "turns": []}

    def save(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.state_path)       # atomic: a crash never leaves a half-written checkpoint

    def warn(self, msg: str) -> None:
        if msg not in self.state["warnings"]:
            self.state["warnings"].append(msg)
        log.warning(msg)

    def invalidate(self, from_stage: str) -> None:
        """Forget completed stages from `from_stage` on (e.g. 'sync' to re-export after edits)."""
        keys = [k for k, _, _ in STAGES]
        drop = set(keys[keys.index(from_stage):])
        self.state["done"] = [k for k in self.state["done"] if k not in drop]
        self.save()

    # ----------------------------------------------------------- progress
    def _emit(self, frac: float, message: str = "", speed: str = "") -> None:
        if not self.on_progress:
            return
        frac = max(0.0, min(1.0, frac))
        total_w = sum(w for k, _, w in STAGES if w and not (k == "separate" and not self.s.preserve_background))
        done_w, label = 0.0, ""
        status = {}
        for key, lab, w in STAGES:
            if key == self._stage_key:
                label = lab
                status[key] = f"{int(frac * 100)}%"
            elif key in self.state["done"]:
                status[key] = "done"
            else:
                status[key] = "waiting"
            if key == "separate" and not self.s.preserve_background:
                status[key] = "skipped"
                continue
            if key in self.state["done"]:
                done_w += w
            elif key == self._stage_key:
                done_w += w * frac
        elapsed = time.time() - self._stage_t0
        eta = (elapsed * (1 - frac) / frac) if (frac >= 0.05 and elapsed >= 10) else None
        self.on_progress(Progress(self._stage_key, label, frac, done_w / total_w if total_w else 0.0, message, eta, speed, status))

    def _begin(self, key: str) -> None:
        self._stage_key, self._stage_t0 = key, time.time()
        self._emit(0.0, "starting")

    # ----------------------------------------------------------- run
    def run(self, video_path: str | Path) -> dict:
        self.state["video"] = str(video_path)
        # Carry imported subtitles path into state (from settings or prior run)
        sub = (getattr(self.s, "imported_subtitles", "") or "").strip()
        if sub:
            self.state["imported_subtitles"] = sub
        elif "imported_subtitles" not in self.state:
            self.state["imported_subtitles"] = ""
        info = VL.probe(video_path)
        VL.validate_for_dubbing(info)
        self.state["info"] = asdict(info)
        self.save()
        ch = self.state.get("characters")
        if ch and self.on_characters:             # resumed project: show the characters found earlier
            self.on_characters(ch["count"], ch["genders"])
        for key, _, _ in STAGES:
            if key in self.state["done"]:
                continue
            if key == "separate" and not self.s.preserve_background:
                continue
            self.control.checkpoint()
            self._begin(key)
            t0 = time.time()
            getattr(self, f"_stage_{key}")(info)
            self.state["timings"][key] = round(time.time() - t0, 2)
            self.state["done"].append(key)
            self._emit(1.0, "done")
            self.save()
        self._stage_key = ""
        self._emit(1.0, "finished")
        return self.state["outputs"]

    # ----------------------------------------------------------- stages
    def _stage_extract(self, info: VL.MediaInfo):
        check_disk_space(self.dir, info.duration / 3600 * 4.0 + 0.5)
        a, a16 = self.work / "audio.wav", self.work / "audio16k.wav"
        AX.extract_full_audio(info.path, a, info, lambda f: self._emit(f * 0.8, "extracting audio"), self.control)
        AX.extract_mono16k(a, a16, lambda f: self._emit(0.8 + f * 0.2, "preparing speech audio"), info.duration, self.control)

    def _stage_asr(self, info: VL.MediaInfo):
        # --- Import path: user-supplied timed subtitles replace speech recognition ---
        sub_path = (self.state.get("imported_subtitles") or getattr(self.s, "imported_subtitles", "") or "").strip()
        if sub_path and Path(sub_path).exists():
            self._emit(0.2, "loading imported subtitles")
            try:
                segs = read_subtitle_file(sub_path)
            except Exception as e:
                raise GeoDubberError(str(e), f"Could not read subtitle file: {Path(sub_path).name}")
            # Normalize speaker tags: male/female → speaker_1 / speaker_2 later; keep speaker_N
            for i, s in enumerate(segs):
                s.setdefault("id", i + 1)
                s["confidence"] = 1.0
            geo_count = sum(1 for s in segs if is_valid_georgian(s.get("text", "")))
            already_georgian = geo_count >= max(1, len(segs) * 0.6)
            if already_georgian:
                # Text is already the voice-over script in Georgian — use as translation, skip translate stage content
                for s in segs:
                    s["translation"] = s["text"]
                self.state["language"] = "ka"
                self.state["subtitles_already_georgian"] = True
                self.warn(f"Imported {len(segs)} Georgian lines from {Path(sub_path).name} — skipping speech recognition and translation.")
            else:
                self.state["language"] = "en"
                self.state["subtitles_already_georgian"] = False
                self.warn(f"Imported {len(segs)} lines from {Path(sub_path).name} — will translate to Georgian.")
            self.state["raw_segments"] = segs
            self.state["imported_subtitles"] = str(sub_path)
            self._emit(1.0, f"{len(segs)} subtitle lines")
            return

        t0 = time.time()
        done = {int(k): v for k, v in self.state["asr_chunks"].items()}

        def on_chunk(i, n, segs):
            self.state["asr_chunks"][str(i)] = segs
            self.save()
            processed = min(info.duration, (len(self.state["asr_chunks"])) * self.s.chunk_minutes * 60)
            self._emit(len(self.state["asr_chunks"]) / n, f"chunk {i + 1}/{n}",
                       f"{processed / max(1e-6, time.time() - t0):.1f}x realtime")

        raw = transcribe_chunked(self.be.recognizer, self.work / "audio16k.wav", info.duration, self.work,
                                 self.s.chunk_minutes * 60, self.s.chunk_overlap_seconds, done, on_chunk, self.control)
        if not raw:
            raise GeoDubberError("no speech", "No speech was detected in this video.")
        self.state["language"] = self.be.recognizer.language or "en"
        self.state["raw_segments"] = raw
        self.state["subtitles_already_georgian"] = False

    def _stage_speakers(self, info: VL.MediaInfo):
        raw = [dict(s) for s in self.state["raw_segments"]]
        a16 = self.work / "audio16k.wav"
        turns: list = []
        want = int(getattr(self.s, "speaker_count", 0) or 0)
        if want <= 0:
            want = 2  # auto default; user can set 3 or 4 in Options

        # If imported subtitles already carry speaker tags, honour them and skip diarization
        tagged = [s for s in raw if s.get("speaker_id")]
        use_imported_speakers = bool(tagged) and len(tagged) >= max(1, len(raw) // 2)

        # Gemini listens to the audio and decides how many characters there are, their gender and voice.
        gem = None
        if not use_imported_speakers and self.s.auto_speakers and getattr(self.s, "gemini_speakers", True):
            api_key = getattr(self.be.translator, "api_key", "")
            if not api_key:
                self.warn("Gemini key missing: character genders are guessed from pitch only. Check them in Review.")
            if api_key:
                self._emit(0.1, "Gemini is listening to the characters")
                forced_hint = [g for g in ((getattr(self.s, k, "auto") or "auto").lower() for k in
                                           ("speaker1_gender", "speaker2_gender", "speaker3_gender", "speaker4_gender"))
                               if g in ("male", "female")]
                errs: list = []
                gem = GS.analyze_speakers(GS.make_client(api_key), a16, raw,
                                          want=int(getattr(self.s, "speaker_count", 0) or 0),
                                          forced_genders=forced_hint, control=self.control, errors=errs)
                if gem is None:
                    why = (" Reasons: " + " | ".join(errs[:3])) if errs else ""
                    self.warn("Gemini could not analyse the speakers - using local pitch detection instead." + why)

        if use_imported_speakers:
            gender_map = {"male": "speaker_1", "female": "speaker_2"}
            order: dict[str, str] = {}
            for s in raw:
                tag = (s.get("speaker_id") or "").strip().lower()
                if tag in gender_map:
                    s["speaker_id"] = gender_map[tag]
                elif tag.startswith("speaker_"):
                    pass
                elif tag:
                    if tag not in order:
                        order[tag] = f"speaker_{len(order) + 1}"
                    s["speaker_id"] = order[tag]
                else:
                    s["speaker_id"] = "speaker_1"
                s["speaker_confidence"] = float(s.get("speaker_confidence") or 1.0)
            for s in raw:
                if not s.get("speaker_id"):
                    s["speaker_id"] = "speaker_1"
                    s["speaker_confidence"] = 0.3
            self.warn(f"Using speaker tags from imported subtitles ({len({s['speaker_id'] for s in raw})} characters).")
        elif self.s.auto_speakers and gem:
            for sg, spk in zip(raw, gem["lines"]):
                sg["speaker_id"], sg["speaker_confidence"] = spk, 0.9
            turns = [(float(s["start"]), float(s["end"]), s["speaker_id"]) for s in raw]
            self.warn(f"Gemini found {len(gem['speakers'])} character(s): "
                      + ", ".join(f"{k.split('_')[-1]}={v['gender']} -> {v['voice'].capitalize()}"
                                  for k, v in gem["speakers"].items()))
        elif self.s.auto_speakers:
            if (self.s.hf_token or "").strip():
                try:
                    turns = self.be.diarizer.diarize(a16)
                except CancelledError:
                    raise
                except GeoDubberError as e:
                    self.warn(f"Speaker model failed ({e.user_message}). Using local pitch clustering.")
                except Exception as e:
                    log.exception("diarization failed")
                    self.warn(f"Speaker model failed ({e}). Using local pitch clustering.")
            n_spk = len({t[2] for t in turns}) if turns else 0
            # Prefer model only if it found at least as many speakers as requested
            if n_spk >= max(2, want):
                self._emit(0.5, "assigning speakers")
                SD.assign_speakers(raw, turns)
            else:
                self._emit(0.4, "clustering speakers by pitch")
                SD.cluster_segments_by_pitch(raw, a16, n_speakers=want)
                # Build synthetic turns for preview/stabilize compatibility
                turns = []
                for s in raw:
                    turns.append((float(s["start"]), float(s["end"]), s["speaker_id"]))
                self.warn(
                    f"Split into {want} character(s) by pitch. "
                    "In Review, play each ~5s sample and set Male/Female so voices stay correct."
                )
        elif not use_imported_speakers:
            self.warn("Automatic speaker detection is off: all dialogue is treated as ONE speaker.")
            for sg in raw:
                sg["speaker_id"], sg["speaker_confidence"] = "speaker_1", 0.0
            turns = []

        self._emit(0.55, "assigning speakers")
        if not use_imported_speakers:
            if turns and any(not s.get("speaker_id") for s in raw):
                SD.assign_speakers(raw, turns)
            merged = {} if gem else SD.merge_minor_speakers(raw, self.s.min_speaker_seconds)
            if merged:
                self.warn(f"Merged {len(merged)} very short speaker(s) into neighbours: {merged}")
        segs = merge_fragments(raw)
        # Re-cluster fragments if we still have too few distinct speakers vs requested
        n_now = len({s["speaker_id"] for s in segs})
        if not use_imported_speakers and self.s.auto_speakers and not gem and n_now < want:
            SD.cluster_segments_by_pitch(segs, a16, n_speakers=want)
            n_now = len({s["speaker_id"] for s in segs})
        self.state["speakers"] = SD.estimate_speaker_genders(segs, a16)
        if gem:
            for spk, info in self.state["speakers"].items():
                g = gem["speakers"].get(spk)
                if g:
                    info["gender"], info["gender_confidence"], info["gender_source"] = g["gender"], 0.95, "gemini"
                    info["voice_hint"] = g["voice"]            # "giorgi" or "nino"
            # Cross-check Gemini's gender with the measured pitch. A mismatch does not change the voice
            # automatically (Gemini may be right), but it is shown in the warnings and the Review window.
            for spk, info in self.state["speakers"].items():
                f0, g = info.get("f0"), info.get("gender")
                if f0 and g in ("male", "female") and GS.pitch_contradicts(g, float(f0)):
                    self.warn(f"{info.get('label', spk)}: Gemini says {g}, but the median pitch is "
                              f"{float(f0):.0f} Hz. Check this character's voice in Review.")
                    info["gender_check"] = "pitch_mismatch"
        # Keep each character's voice stable over the whole film: a line whose pitch contradicts its
        # character's gender moves to a character of the right gender (see core/speaker_tracks.py).
        if not use_imported_speakers and self.s.auto_speakers and self.state["speakers"]:
            self._emit(0.56, "checking voices over time")
            info, msgs = ST.stabilise(segs, a16, self.state["speakers"])
            self.state["speakers"] = info
            for m in msgs:
                self.warn(m)
        # User pre-set genders: Character 1..4 by speech duration rank
        forced = []
        for key in ("speaker1_gender", "speaker2_gender", "speaker3_gender", "speaker4_gender"):
            g = (getattr(self.s, key, "auto") or "auto").lower()
            if g in ("male", "female"):
                forced.append(g)
        by_speech = sorted(
            self.state["speakers"].items(),
            key=lambda kv: -float(kv[1].get("speech_seconds") or 0),
        )
        if forced:
            for (spk, info), gender in zip(by_speech, forced):
                info["gender"] = gender
                info["gender_confidence"] = 1.0
                info["gender_source"] = "user"
            self.warn(
                "Using your pre-set character genders: "
                + ", ".join(
                    f"{self.state['speakers'][s]['label']}={self.state['speakers'][s]['gender']}"
                    for s, _ in by_speech[: len(forced)]
                )
            )
        elif not gem:
            # CRITICAL: after pitch clustering, speaker_1 = lowest F0, speaker_N = highest F0.
            # Force gender by rank so independent F0 estimation cannot flip male/female.
            # This is the main fix for "woman gets male voice / man gets female voice".
            by_f0 = sorted(
                self.state["speakers"].items(),
                key=lambda kv: (float(kv[1].get("f0") or 9999.0), -float(kv[1].get("speech_seconds") or 0)),
            )
            n = len(by_f0)
            if n >= 2:
                for i, (spk, info) in enumerate(by_f0):
                    if info.get("gender_source") == "user":
                        continue
                    if i == 0:
                        # lowest pitch → male
                        info["gender"] = "male"
                        info["gender_confidence"] = max(0.7, float(info.get("gender_confidence") or 0))
                        info["gender_source"] = "pitch_rank"
                    elif i == n - 1:
                        # highest pitch → female
                        info["gender"] = "female"
                        info["gender_confidence"] = max(0.7, float(info.get("gender_confidence") or 0))
                        info["gender_source"] = "pitch_rank"
                    else:
                        # middle speakers (3+): decide by their own measured pitch, not by default. (v1 forced
                        # every middle speaker to male, which gave female characters male voices.)
                        g, conf = middle_speaker_gender(info)
                        info["gender"] = g
                        info["gender_confidence"] = max(0.55, conf)
                        info["gender_source"] = "pitch_rank"
                self.warn(
                    "Genders set by pitch rank (lowest→male, highest→female). "
                    "In Review, play each sample and correct if needed."
                )
            else:
                # Single speaker or no F0: fall back to classic threshold
                for spk, info in self.state["speakers"].items():
                    if info.get("gender") == "unknown" and info.get("f0"):
                        f0 = float(info["f0"])
                        cent = info.get("centroid")
                        g = "male" if f0 < 158 else "female"
                        conf = max(0.35, float(info.get("gender_confidence") or 0))
                        if conf < 0.45 and cent is not None:
                            if float(cent) > 1950 and g == "male":
                                g = "female"
                            elif float(cent) < 1400 and g == "female":
                                g = "male"
                        info["gender"] = g
                        info["gender_confidence"] = conf
        self.state["segments"] = segs
        ranked = sorted(self.state["speakers"].values(), key=lambda i: -float(i.get("speech_seconds") or 0))
        chars = {"count": len(ranked), "genders": [i.get("gender", "unknown") for i in ranked],
                 "source": "gemini" if gem else "local"}
        self.state["characters"] = chars
        if self.on_characters:
            try:
                self.on_characters(chars["count"], chars["genders"])
            except Exception:
                log.exception("on_characters callback failed")
        self._assign()
        # Ensure male speakers actually got male voices (and female → female)
        voices = {v.id: v for v in self.all_voices()}
        used = set()
        for spk, info in sorted(self.state["speakers"].items(), key=lambda kv: -float(kv[1].get("speech_seconds") or 0)):
            g = info.get("gender")
            vid = self.state["assignments"].get(spk)
            ok = vid and vid in voices and vid not in used and (g not in ("male", "female") or voices[vid].gender == g)
            if ok:
                used.add(vid)
                continue
            # A voice of the right gender that no other character uses yet (main voices come first in the list).
            pool = [v for v in self.all_voices() if v.id not in used and (g not in ("male", "female") or v.gender == g)]
            if not pool and g in ("male", "female"):
                pool = [v for v in self.all_voices() if v.gender == g]
            if pool:
                self.state["assignments"][spk] = pool[0].id
                used.add(pool[0].id)

        # Final safety: if any female-labeled speaker got a male voice (or vice versa), force fix
        for spk, info in self.state["speakers"].items():
            g = info.get("gender")
            vid = self.state["assignments"].get(spk)
            if g not in ("male", "female") or not vid or vid not in voices:
                continue
            if voices[vid].gender == g:
                continue
            # Wrong gender voice — steal a free correct one
            free = [v for v in self.all_voices() if v.gender == g and v.id not in used]
            if not free:
                free = [v for v in self.all_voices() if v.gender == g]
            if free:
                old = vid
                self.state["assignments"][spk] = free[0].id
                used.add(free[0].id)
                self.warn(f"{info.get('label', spk)}: corrected voice gender ({old} → {free[0].id})")

    def all_voices(self) -> list[Voice]:
        vs = list(self.be.tts.builtin_voices())
        # Reference clips only change the sound on engines that actually clone a voice.
        # On Edge TTS they would all collapse into the same Giorgi/Eka voice.
        if getattr(self.be.tts, "needs_reference", False):
            extra = self.extra_voices if self.extra_voices is not None else load_user_voices()
            vs.extend(extra)
        return usable_voices(vs, self.be.tts.name)

    def _assign(self) -> None:
        voices = self.all_voices()
        mapping, warns = assign_voices(self.state["speakers"], voices, self.state.get("assignments"), self.s.allow_voice_sharing)
        self.state["assignments"] = mapping
        self.state["voice_warnings"] = warns
        for w in warns:
            self.warn(w)

    def _glossary(self) -> dict:
        """Names and terms that must always be translated the same way (glossary.txt, see README)."""
        path = self.s.translation_glossary_file or str(data_dir() / "glossary.txt")
        return load_glossary(path)

    def _stage_translate(self, info: VL.MediaInfo):
        segs = self.state["segments"]
        if self.state.get("subtitles_already_georgian"):
            # Imported Georgian script — already in translation field
            for s in segs:
                if not s.get("translation"):
                    s["translation"] = s.get("text", "")
            self._emit(1.0, "skipped (imported Georgian subtitles)")
            return
        # Give the translator what it needs: each line's speaker gender, the glossary, and the cancel hook
        # (so waiting out a Gemini quota limit can still be cancelled).
        for s in segs:
            s["gender"] = (self.state["speakers"].get(s.get("speaker_id")) or {}).get("gender", "unknown")
        if hasattr(self.be.translator, "checkpoint"):
            self.be.translator.checkpoint = self.control.checkpoint
        translate_segments(self.be.translator, segs, self.state["language"], self.s.translation_batch,
                           self._glossary(),
                           on_batch=lambda d, t: (self.save(), self._emit(d / t, f"{d}/{t} segments")), control=self.control)
        for s in segs:
            s["too_long"] = estimate_speech_seconds(s["translation"]) > (s["end"] - s["start"]) * self.s.max_speedup + self.s.max_spill_seconds
        if getattr(self.be.translator, "client", True) is None:
            self.warn("Gemini key missing: all lines were translated by the WEB fallback, which is much lower quality. "
                      "Set GEMINI_API_KEY and dub again (clear the translations first with scripts/reset_translations.py).")
        sources = Counter(s.get("mt_source") for s in segs)
        if sources.get("failed"):
            self.warn(f"{sources['failed']} line(s) could not be translated and stay silent in the dub. "
                      "Run Start Dubbing again later (the quota may have reset) or type them in Review.")
        if sources.get("web"):
            self.warn(f"{sources['web']} line(s) came from the word-level web fallback (lower quality). Check them in Review.")
        notes = Counter(s.get("tr_note") for s in segs if s.get("tr_note"))
        if notes:
            self.warn("Flagged in Review (idiom / joke / pun / culture / unclear): "
                      + ", ".join(f"{k}={v}" for k, v in notes.most_common()))
        if sources.get("gemini_retry"):
            self.warn(f"{sources['gemini_retry']} line(s) needed a second translation attempt. Check them in Review.")
        n_long = sum(1 for s in segs if s["too_long"])
        if n_long:
            self.warn(f"{n_long} translated line(s) are probably longer than their time window; consider shortening them in the editor.")

    def _stage_review(self, info: VL.MediaInfo):
        self._assign()                       # pick up voices added since detection
        # Build ~5s original-audio previews per speaker so the user can hear who is male/female
        previews = {}
        try:
            by: dict = {}
            for s in self.state.get("segments") or []:
                by.setdefault(s["speaker_id"], []).append(s)
            prev_dir = self.work / "speaker_previews"
            prev_dir.mkdir(parents=True, exist_ok=True)
            src = self.work / "audio.wav"
            if not src.exists():
                src = self.work / "audio16k.wav"
            for spk, segs in by.items():
                others = [o for o in self.state.get("segments") or [] if o["speaker_id"] != spk]
                a, b = SD.pick_preview_window(segs, 5.0, others=others)
                if b <= a:
                    continue
                out = prev_dir / f"{spk}.wav"
                try:
                    AX.cut_wav(src, a, b, out, self.control)
                    if out.exists() and out.stat().st_size > 100:
                        previews[spk] = str(out)
                        if spk in self.state.get("speakers", {}):
                            self.state["speakers"][spk]["preview_path"] = str(out)
                            self.state["speakers"][spk]["preview_start"] = a
                            self.state["speakers"][spk]["preview_end"] = b
                except Exception:
                    log.exception("preview extract failed for %s", spk)
        except Exception:
            log.exception("speaker preview extraction failed")
        self.state["speaker_previews"] = previews
        self.save()
        if self.s.review_translation and self.on_review:
            self.on_review(self)             # blocks until the user continues; may edit self.state in place
            self.control.checkpoint()
            self._assign()                   # re-assign voices after user set genders from listening
            self.save()
        missing = [self.state["speakers"][sp]["label"] for sp in {s["speaker_id"] for s in self.state["segments"] if (s.get("translation") or "").strip()}
                   if sp not in self.state["assignments"]]
        if missing:
            raise VoiceUnavailableError(f"unassigned: {missing}",
                                        f"No Georgian voice is assigned to: {', '.join(sorted(missing))}. Add voices or assign them manually, then resume.")

    def _stage_tts(self, info: VL.MediaInfo):
        voices = {v.id: v for v in self.all_voices()}
        segs = [s for s in self.state["segments"] if (s.get("translation") or "").strip()]
        if not segs:
            return
        pending = []
        for s in segs:
            vid = self.state["assignments"].get(s["speaker_id"])
            voice = voices.get(vid) if vid else None
            if voice is None:
                raise VoiceUnavailableError(f"voice {vid} missing", f"Assigned voice '{vid}' is no longer available.")
            entry = self.state["tts"].get(str(s["id"]))
            if entry and entry["text"] == s["translation"] and entry["voice"] == vid and Path(entry["path"]).exists():
                continue
            pending.append((s, voice, vid))
        if not pending:
            return
        workers = max(1, int(self.s.preset().get("tts_workers", 4)))
        t0, failed, done = time.time(), [], 0

        def _one(item):
            s, voice, vid = item
            self.control.checkpoint()
            last = None
            for attempt in (1, 2):
                try:
                    p = synthesize_cached(self.be.tts, voice, s["translation"], self.cache_root)
                    return s["id"], {"path": str(p), "duration": wav_duration(p), "text": s["translation"], "voice": vid}, None
                except CancelledError:
                    raise
                except Exception as e:
                    last = e
                    log.warning("TTS failed for segment %s (attempt %d): %s", s["id"], attempt, e)
            return s["id"], None, str(last)

        with ThreadPoolExecutor(max_workers=min(workers, len(pending))) as pool:
            futures = [pool.submit(_one, item) for item in pending]
            for fut in as_completed(futures):
                self.control.checkpoint()
                try:
                    sid, entry, err = fut.result()
                except CancelledError:
                    pool.shutdown(wait=False, cancel_futures=True)
                    raise
                if err or entry is None:
                    failed.append((sid, err or "unknown"))
                else:
                    self.state["tts"][str(sid)] = entry
                done += 1
                if done % 8 == 0:
                    self.save()
                self._emit(done / len(pending), f"segment {done}/{len(pending)}",
                           f"{done / max(1e-6, time.time() - t0):.2f} lines/s")
        self.save()
        if failed:
            self.state["tts_failures"] = failed
            raise GeoDubberError(f"TTS failed for {len(failed)} segments: {failed[:3]}",
                                 f"Voice generation failed for {len(failed)} line(s). Finished lines are saved - resume to retry. "
                                 f"First error: {failed[0][1]}")
        self.state["tts_failures"] = []

    # ----------------------------------------------------------- timing
    def _trimmed(self, entry: dict) -> tuple[Path, float]:
        """A generated line with dead air removed from both ends (cached). Returns (path, seconds)."""
        src = Path(entry["path"])
        if not self.s.trim_clip_silence:
            return src, float(entry.get("duration") or wav_duration(src))
        out = self.work / "trim" / src.name
        if out.exists() and out.stat().st_size > 44:
            return out, wav_duration(out)
        x, sr = read_wav(src)
        y, _, _ = Q.trim_silence(to_mono(x), sr)
        out.parent.mkdir(parents=True, exist_ok=True)
        write_wav(out, y, sr)
        return out, len(y) / sr

    def _fit_durations(self) -> tuple[dict, dict, dict]:
        """(durations of the real generated audio, original windows, segments) for lines with audio."""
        segs = {s["id"]: s for s in self.state["segments"] if (s.get("translation") or "").strip()}
        durs, wins = {}, {}
        for sid, s in segs.items():
            t = self.state["tts"].get(str(sid))
            if t:
                durs[sid] = self._trimmed(t)[1]
                wins[sid] = max(0.4, float(s["end"]) - float(s["start"]))
        return durs, wins, segs

    def _stage_fit(self, info: VL.MediaInfo):
        """Closed timing loop: lines whose REAL audio does not fit their window are re-written shorter
        (same meaning) and voiced again. Only the changed lines are synthesised again."""
        if getattr(self.be.translator, "client", True) is None:
            self.state["fit"] = {"shortened": 0, "still_long": [], "skipped": "no Gemini key"}
            self.warn("Timing fit skipped: no Gemini key, so long lines cannot be rewritten.")
            return
        if not self.s.timing_fit_rounds or not hasattr(self.be.translator, "shorten"):
            self.state["fit"] = {"shortened": 0, "still_long": [], "skipped": True}
            return
        shortened = 0
        for rnd in range(int(self.s.timing_fit_rounds)):
            self.control.checkpoint()
            durs, wins, segs = self._fit_durations()
            long_ids = Q.fit_needed(durs, wins, self.s.max_speedup, self.s.max_spill_seconds)
            if not long_ids:
                break
            items = [{"i": sid, "source": segs[sid]["text"], "ka": segs[sid]["translation"],
                      # aim below the hard limit so the shorter version really fits
                      "max_seconds": wins[sid] * self.s.max_speedup * 0.95 + self.s.max_spill_seconds * 0.5}
                     for sid in long_ids]
            self._emit(rnd / max(1, self.s.timing_fit_rounds), f"round {rnd + 1}: {len(items)} line(s) too long")
            new = self.be.translator.shorten(items)
            changed = 0
            for sid, ka in new.items():
                if ka and ka != segs[sid]["translation"]:
                    segs[sid]["translation"] = ka
                    segs[sid]["mt_source"] = "shortened"
                    changed += 1
            self.save()
            if not changed:
                break
            shortened += changed
            self._stage_tts(info)               # only lines whose text changed are voiced again
        durs, wins, _ = self._fit_durations()
        still = Q.fit_needed(durs, wins, self.s.max_speedup, self.s.max_spill_seconds)
        self.state["fit"] = {"shortened": shortened, "still_long": still}
        if still:
            self.warn(f"{len(still)} line(s) stay longer than their window after re-writing (ids {still[:15]}). "
                      "They are kept in full; shorten them in Review if they overlap.")
        elif shortened:
            self.warn(f"{shortened} line(s) were re-written shorter so they fit their time window.")

    def _stage_separate(self, info: VL.MediaInfo):
        out = self.work / "background.wav"
        if out.exists() and out.stat().st_size > 1000:
            log.info("Reusing existing background track: %s", out)
            self.state["background"] = str(out)
            return
        res = None
        try:
            res = self.be.separator.separate(self.work / "audio.wav", info.duration, out, lambda f: self._emit(f, "separating"), self.control)
        except CancelledError:
            raise
        except Exception as e:
            log.exception("Background separation failed")
            self.warn(f"Background separation failed ({e}). Using original audio quietly under the dub.")
            res = None
        if res is None:
            self.state["background"] = "fallback"
            self.warn("Dialogue/background separation is not available: the original soundtrack is kept quietly under the "
                      "Georgian dub, so some ORIGINAL SPEECH may remain faintly audible.")
        else:
            self.state["background"] = str(out)

    def _stage_sync(self, info: VL.MediaInfo):
        items, clip_src = [], {}
        for s in self.state["segments"]:
            t = self.state["tts"].get(str(s["id"]))
            if t:
                path, sec = self._trimmed(t)                 # dead air removed: the dub follows the rhythm
                clip_src[s["id"]] = str(path)
                items.append({"id": s["id"], "start": s["start"], "end": s["end"], "duration": sec, "speaker_id": s["speaker_id"]})
        places = plan_placements(items, self.s.max_speedup, self.s.max_spill_seconds)
        for i, p in enumerate(places):
            self.control.checkpoint()
            src = clip_src[p.seg_id]
            p.audio_path = str(apply_speed(src, p.speed, self.work / "speed", self.control))
            self._emit(0.5 * (i + 1) / len(places), "adjusting speech speed")
        total = render_dub_track(places, self.work / "dub.wav", info.duration, gain=1.0,
                                 on_progress=lambda f: self._emit(0.5 + f * 0.5, "building dub track"), control=self.control)
        over = [p for p in places if p.overflow > 0.05]
        self.state["sync"] = {"lines": len(places), "sped_up": sum(1 for p in places if p.speed > 1.001),
                              "overflow_ids": [p.seg_id for p in over], "dub_duration": total}
        if over:
            self.warn(f"{len(over)} line(s) are longer than their window even after speed-up (kept in full, may overlap the next line): "
                      f"ids {[p.seg_id for p in over][:15]}")
        if total > info.duration + 0.5:
            self.warn(f"Dub audio runs {total - info.duration:.1f}s past the end of the video.")
        if self.s.voice_qc:
            self._quality_control(places, clip_src)

    def _quality_control(self, places, clip_src):
        """Automatic checks of the generated audio and the timeline. Results go to state['qc'] and warnings."""
        texts = {s["id"]: s.get("translation", "") for s in self.state["segments"]}
        clips = {}
        for p in places:
            x, sr = read_wav(clip_src[p.seg_id])
            x = to_mono(x)
            st = Q.clip_stats(x)
            st["duration"] = len(x) / sr
            clips[p.seg_id] = st
        issues = Q.clip_problems(clips, texts)
        issues += Q.timeline_problems([{"id": p.seg_id, "speaker_id": p.speaker_id, "start": p.start, "end": p.end}
                                       for p in places])
        counts = Counter(i["kind"] for i in issues)
        self.state["qc"] = {"counts": dict(counts), "issues": issues[:300]}
        for kind, n in counts.items():
            ex = [i["id"] for i in issues if i["kind"] == kind][:8]
            self.warn(f"Quality check: {n} x {kind} (ids {ex}). Listen to them in Review.")

    def _stage_mix(self, info: VL.MediaInfo):
        bg = None
        bg_vol = self.s.background_volume
        if self.s.preserve_background:
            if self.state.get("background") == "fallback":
                bg, bg_vol = self.work / "audio.wav", bg_vol * BG_FALLBACK_GAIN
            else:
                bg = self.state.get("background")
        orig = self.work / "audio.wav" if self.s.original_dialogue_volume > 0.001 else None
        mix_audio(self.work / "dub.wav", self.work / "mix.m4a", bg, orig, self.s.voice_volume, bg_vol,
                  self.s.original_dialogue_volume, self.s.duck_background, self.s.audio_bitrate, info.duration,
                  self._emit, self.control)

    def _stage_export(self, info: VL.MediaInfo):
        out_dir = self.s.resolved_output_dir()
        stem = Path(info.path).stem
        segs = self.state["segments"]
        outs = {}
        outs["georgian_srt"] = str(SUB.write_srt(segs, out_dir / f"{stem}_GE.srt", "translation"))
        outs["original_srt"] = str(SUB.write_srt(segs, out_dir / f"{stem}_original.srt", "text"))
        outs["translation_txt"] = str(SUB.write_txt(segs, out_dir / f"{stem}_translation.txt"))
        srt = outs["georgian_srt"] if self.s.georgian_subtitles else None
        vpreset = self.s.preset().get("video_preset") or self.s.video_preset
        outs["video"] = str(export_video(info.path, self.work / "mix.m4a", out_dir / f"{stem}_GE_DUBBED.mp4", info, srt,
                                         self.s.burn_subtitles, self.s.video_crf, vpreset,
                                         self._emit, self.control))
        report = {"app_version": APP_VERSION, "video": asdict(info), "language": self.state.get("language"),
                  "settings": asdict(self.s) | {"hf_token": "***" if self.s.hf_token else ""},
                  "tts_engine": self.be.tts.name, "speakers": self.state["speakers"], "assignments": self.state["assignments"],
                  "sync": self.state.get("sync"), "timings_seconds": self.state["timings"], "warnings": self.state["warnings"],
                  "segments": len(segs)}
        rp = out_dir / f"{stem}_report.json"
        rp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        outs["report"] = str(rp)
        self.state["outputs"] = outs

    # ----------------------------------------------------------- housekeeping
    def cleanup_temp(self) -> None:
        shutil.rmtree(self.work, ignore_errors=True)
        self.work.mkdir(parents=True, exist_ok=True)
