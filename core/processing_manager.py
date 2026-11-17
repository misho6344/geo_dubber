"""The dubbing pipeline: staged, checkpointed, pausable, resumable."""
from __future__ import annotations

import json
import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

from config.settings import APP_VERSION, Settings
from core import audio_extractor as AX
from core import speaker_diarization as SD
from core import gemini_speakers as GS
from core import speaker_tracks as ST
from core import subtitles as SUB
from core import video_loader as VL
from core.audio_io import wav_duration
from core.audio_mixer import mix_audio
from core.audio_synchronizer import apply_speed, plan_placements, render_dub_track
from core.background_separator import FallbackSeparator, Separator, create_separator
from core.errors import CancelledError, GeoDubberError, VoiceUnavailableError
from core.ffmpeg_utils import Control
from core.georgian_tts import TTSBackend, create_backend, synthesize_cached
from core.hardware import check_disk_space, resolve_device
from core.speech_recognition import FasterWhisperRecognizer, Recognizer, merge_fragments, transcribe_chunked
from core.translation import NLLBTranslator, Translator, estimate_speech_seconds, translate_segments, is_valid_georgian
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
    ("separate", "Background separation", 8),
    ("sync", "Audio synchronization", 5),
    ("mix", "Final audio mix", 4),
    ("export", "Final export", 5),
]
BG_FALLBACK_GAIN = 0.25      # original track level under the dub when no separation model is available


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
    total_eta_seconds: Optional[float] = None  # estimated time remaining for the whole project


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
        translator=NLLBTranslator(settings.translation_model, device, pr["nllb_beam"], settings.translation_batch),
        tts=create_backend(settings.tts_engine, settings, device),
        separator=(create_separator(device, settings.resolved_temp_dir(), segment=str(pr.get("demucs_segment") or "8"))
                  if settings.preserve_background else FallbackSeparator()),
    )
    return be, f"{device} {note}".strip()


class Pipeline:
    def __init__(self, project_dir: str | Path, settings: Settings, backends: Backends,
                 control: Optional[Control] = None, on_progress: Optional[Callable[[Progress], None]] = None,
                 on_review: Optional[Callable[["Pipeline"], None]] = None, cache_root: Optional[Path] = None,
                 extra_voices: Optional[list[Voice]] = None,
                 on_characters: Optional[Callable] = None):
        self.dir = Path(project_dir)
        self.work = self.dir / "work"
        self.work.mkdir(parents=True, exist_ok=True)
        self.s, self.be = settings, backends
        self.control = control or Control()
        self.on_progress, self.on_review = on_progress, on_review
        self.cache_root = cache_root
        self.extra_voices = extra_voices
        self.on_characters = on_characters
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
        overall = done_w / total_w if total_w else 0.0
        timings = self.state.get("timings") or {}
        done_keys = set(self.state.get("done") or [])
        measured_elapsed = sum(
            max(0.0, float(timings.get(k, 0.0) or 0.0))
            for k in done_keys if k != "review"
        )
        if self._stage_key and self._stage_key != "review" and self._stage_key not in done_keys:
            measured_elapsed += max(0.0, elapsed)
        total_eta = None
        if self._stage_key != "review" and overall >= 0.03 and measured_elapsed >= 8.0:
            total_eta = max(0.0, measured_elapsed * (1.0 - overall) / overall)
        self.on_progress(Progress(
            self._stage_key, label, frac, overall, message, eta, speed, status, total_eta
        ))

    def _begin(self, key: str) -> None:
        self._stage_key, self._stage_t0 = key, time.time()
        self._emit(0.0, "starting")

    # ----------------------------------------------------------- run
    def run(self, video_path: str | Path) -> dict:
        self.state["video"] = str(video_path)
        info = VL.probe(video_path)
        VL.validate_for_dubbing(info)
        self.state["info"] = asdict(info)
        self.save()
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

    def _stage_speakers(self, info: VL.MediaInfo):
        raw = [dict(s) for s in self.state["raw_segments"]]
        a16 = self.work / "audio16k.wav"
        turns: list = []
        requested_want = int(getattr(self.s, "speaker_count", 0) or 0)
        fallback_want = requested_want or 2
        gemini_speaker_info: dict = {}
        used_gemini = False

        if self.s.auto_speakers:
            # Gemini analyses the actual audio and transcript together. This is more useful for
            # voice-gender selection than assigning gender from a single pitch threshold.
            client = GS.make_client()
            if client is not None:
                self._emit(0.15, "analysing character voices with Gemini")
                errors: list[str] = []
                result = GS.analyze_speakers(
                    client,
                    self.work / "audio.wav",
                    raw,
                    want=requested_want,
                    control=self.control,
                    errors=errors,
                )
                if result and len(result.get("lines", [])) == len(raw):
                    gemini_speaker_info = result.get("speakers") or {}
                    for segment, speaker_id in zip(raw, result["lines"]):
                        segment["speaker_id"] = speaker_id
                    used_gemini = True
                    self.warn(
                        "Gemini analysed speaker turns and perceived voice gender from the original audio. "
                        "Confirm each sample in Review before voice generation."
                    )
                else:
                    reason = errors[0] if errors else "Gemini returned an incomplete speaker map"
                    self.warn(f"Gemini speaker analysis was unavailable ({reason[:180]}). Using local speaker detection.")

            if not used_gemini:
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
                if n_spk >= max(2, fallback_want):
                    self._emit(0.5, "assigning speakers")
                    SD.assign_speakers(raw, turns)
                else:
                    self._emit(0.4, "clustering speakers by pitch")
                    SD.cluster_segments_by_pitch(raw, a16, n_speakers=fallback_want)
                    turns = []
                    for segment in raw:
                        turns.append((float(segment["start"]), float(segment["end"]), segment["speaker_id"]))
                    self.warn(
                        f"Split into {fallback_want} character(s) by pitch. "
                        "This is an acoustic estimate; listen to each sample and confirm Male/Female in Review."
                    )
            else:
                self._emit(0.65, "Gemini speaker analysis completed")
        else:
            self.warn("Automatic speaker detection is off: all dialogue is treated as ONE speaker.")
            for segment in raw:
                segment["speaker_id"], segment["speaker_confidence"] = "speaker_1", 0.0
            turns = []

        self._emit(0.55, "assigning speakers")
        if turns and any(not segment.get("speaker_id") for segment in raw):
            SD.assign_speakers(raw, turns)
        merged = SD.merge_minor_speakers(raw, self.s.min_speaker_seconds)
        if merged:
            self.warn(f"Merged {len(merged)} very short speaker(s) into neighbours: {merged}")
        segs = merge_fragments(raw)
        n_now = len({segment["speaker_id"] for segment in segs})
        if self.s.auto_speakers and not used_gemini and n_now < fallback_want:
            SD.cluster_segments_by_pitch(segs, a16, n_speakers=fallback_want)

        self.state["speakers"] = SD.estimate_speaker_genders(segs, a16)
        if used_gemini:
            for speaker_id, model_info in gemini_speaker_info.items():
                info_row = self.state["speakers"].get(speaker_id)
                gender = str(model_info.get("gender", "")).lower()
                if info_row is not None and gender in {"male", "female"}:
                    info_row["gender"] = gender
                    info_row["gender_source"] = "gemini"
                    if model_info.get("gender_confidence") is not None:
                        try:
                            info_row["gender_confidence"] = max(0.0, min(1.0, float(model_info["gender_confidence"])))
                        except (TypeError, ValueError):
                            pass

        # User pre-set genders: Character 1..4 by speech duration rank
        forced = []
        for key in ("speaker1_gender", "speaker2_gender", "speaker3_gender", "speaker4_gender"):
            gender = (getattr(self.s, key, "auto") or "auto").lower()
            if gender in ("male", "female"):
                forced.append(gender)
        by_speech = sorted(
            self.state["speakers"].items(),
            key=lambda kv: -float(kv[1].get("speech_seconds") or 0),
        )
        if forced:
            for (speaker_id, speaker_info), gender in zip(by_speech, forced):
                speaker_info["gender"] = gender
                speaker_info["gender_confidence"] = 1.0
                speaker_info["gender_source"] = "user"
            self.warn(
                "Using your pre-set character genders: "
                + ", ".join(
                    f"{self.state['speakers'][speaker_id]['label']}={self.state['speakers'][speaker_id]['gender']}"
                    for speaker_id, _ in by_speech[: len(forced)]
                )
            )

        # The local pitch stabilizer corrects isolated line swaps, but never overrides a
        # manually confirmed gender. Gemini's full-audio speaker map is kept intact.
        if not used_gemini:
            try:
                self.state["speakers"], messages = ST.stabilise(segs, a16, self.state["speakers"])
                for message in messages:
                    self.warn(message)
            except Exception:
                log.exception("speaker pitch stabilisation failed")

        self.state["segments"] = segs
        self._assign()
        # Ensure male speakers actually got male voices (and female → female)
        voices = {v.id: v for v in self.all_voices()}
        used = set()
        male_pref = ["edge_ka_giorgi", "edge_ka_giorgi_deep", "edge_ka_giorgi_high"]
        female_pref = ["edge_ka_eka", "edge_ka_eka_warm", "edge_ka_eka_bright"]
        for speaker_id, speaker_info in sorted(
            self.state["speakers"].items(),
            key=lambda kv: -float(kv[1].get("speech_seconds") or 0),
        ):
            gender = speaker_info.get("gender")
            voice_id = self.state["assignments"].get(speaker_id)
            valid = (
                voice_id and voice_id in voices and voice_id not in used
                and (gender not in ("male", "female") or voices[voice_id].gender == gender)
            )
            if valid:
                used.add(voice_id)
                continue
            preferred = male_pref if gender == "male" else female_pref if gender == "female" else []
            pool = [voices[voice] for voice in preferred if voice in voices and voice not in used]
            if not pool:
                pool = [
                    voice for voice in self.all_voices()
                    if voice.id not in used and (gender not in ("male", "female") or voice.gender == gender)
                ]
            if not pool and gender in ("male", "female"):
                pool = [voice for voice in self.all_voices() if voice.gender == gender]
            if pool:
                self.state["assignments"][speaker_id] = pool[0].id
                used.add(pool[0].id)

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

    def _stage_translate(self, info: VL.MediaInfo):
        segs = self.state["segments"]
        total = len([s for s in segs if not s.get("translation") or not is_valid_georgian(s.get("translation"))]) or 1
        translate_segments(self.be.translator, segs, self.state["language"], self.s.translation_batch,
                           self.state.get("glossary"),
                           on_batch=lambda d, t, msg=None: (self.save(), self._emit(d / t, msg or f"{d}/{t} segments")),
                           control=self.control)
        for s in segs:
            s["too_long"] = estimate_speech_seconds(s["translation"]) > (s["end"] - s["start"]) * self.s.max_speedup + self.s.max_spill_seconds
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
                a, b = SD.pick_preview_window(segs, 5.0)
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
        items = []
        for s in self.state["segments"]:
            t = self.state["tts"].get(str(s["id"]))
            if t:
                items.append({"id": s["id"], "start": s["start"], "end": s["end"], "duration": t["duration"], "speaker_id": s["speaker_id"]})
        places = plan_placements(items, self.s.max_speedup, self.s.max_spill_seconds)
        for i, p in enumerate(places):
            self.control.checkpoint()
            src = self.state["tts"][str(p.seg_id)]["path"]
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
        if getattr(self.s, "export_subtitle_files", True):
            outs["georgian_srt"] = str(SUB.write_srt(segs, out_dir / f"{stem}_GE.srt", "translation"))
            outs["original_srt"] = str(SUB.write_srt(segs, out_dir / f"{stem}_original.srt", "text"))
        else:
            # Remove stale subtitle files when an existing project is rerun with subtitles disabled.
            for subtitle_file in (out_dir / f"{stem}_GE.srt", out_dir / f"{stem}_original.srt"):
                try:
                    subtitle_file.unlink(missing_ok=True)
                except OSError:
                    self.log.warning("Could not remove old subtitle file: %s", subtitle_file)
        outs["translation_txt"] = str(SUB.write_txt(segs, out_dir / f"{stem}_translation.txt"))
        srt = outs.get("georgian_srt") if self.s.georgian_subtitles else None
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
