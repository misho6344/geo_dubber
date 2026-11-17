"""Chunked speech recognition with faster-whisper (model loaded once, reused for every chunk)."""
from __future__ import annotations

import logging
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

from config.settings import sub_dir
from core.audio_extractor import cut_wav, keep_by_midpoint, plan_windows
from core.errors import MissingDependencyError, ModelMissingError
from core.ffmpeg_utils import Control

log = logging.getLogger(__name__)


def _cpu_thread_count() -> int:
    """Use the explicit server CPU budget when set; preserve desktop auto-tuning otherwise."""
    configured = os.getenv("GEODUBBER_CPU_THREADS", "").strip()
    if configured:
        try:
            return max(1, min(64, int(configured)))
        except ValueError:
            log.warning("Invalid GEODUBBER_CPU_THREADS=%r; falling back to automatic detection", configured)
    return max(1, (os.cpu_count() or 4) - 1)


# Whisper sometimes gets stuck and repeats one word or phrase ("word- word- word- ..."). Runs of 3+ repeats are
# collapsed to a single occurrence; 2 repeats are kept because people really say "no, no".
_REPEAT_RUN = re.compile(r"(?<![\w])([^\s\-,.!?]+)(?:[\s\-,]+\1(?![\w]))(?:[\s\-,]+\1(?![\w]))+", re.UNICODE)


def collapse_repetitions(text: str) -> str:
    """Collapse hallucinated loops such as 'a- a- a- a-' into 'a'."""
    if not text:
        return text
    out = _REPEAT_RUN.sub(lambda m: m.group(1), text)
    return re.sub(r"\s{2,}", " ", out).strip()


class Recognizer(ABC):
    language: Optional[str] = None

    @abstractmethod
    def transcribe(self, wav16k: Path, offset: float) -> list[dict]:
        """Return segments [{start,end,text}] with times shifted by `offset` seconds."""


class FasterWhisperRecognizer(Recognizer):
    def __init__(self, model: str = "small", device: str = "cpu", beam: int = 1, language: Optional[str] = None):
        self.model_name, self.device, self.beam, self.language = model, device, beam, language
        self._m = None

    def _load(self):
        if self._m is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as e:
                raise MissingDependencyError(str(e), "faster-whisper is not installed (run install.bat).")
            compute = "float16" if self.device == "cuda" else "int8"
            threads = _cpu_thread_count()
            try:
                local = sub_dir("models") / f"faster-whisper-{self.model_name}"      # from the Settings download manager
                src = str(local) if local.exists() else self.model_name
                self._m = WhisperModel(src, device=self.device, compute_type=compute,
                                       cpu_threads=threads, num_workers=1,
                                       download_root=str(sub_dir("models") / "whisper"))
            except Exception as e:
                if self.device == "cuda":      # CUDA init failure -> automatic CPU fallback
                    log.warning("CUDA init failed (%s); falling back to CPU", e)
                    self.device = "cpu"
                    self._m = WhisperModel(src, device="cpu", compute_type="int8",
                                           cpu_threads=threads, num_workers=1,
                                           download_root=str(sub_dir("models") / "whisper"))
                else:
                    raise ModelMissingError(str(e), f"Speech model '{self.model_name}' could not be loaded. "
                                            "Check your internet connection for the first download.")
            self._batch = None
        return self._m

    def transcribe(self, wav16k: Path, offset: float) -> list[dict]:
        m = self._load()
        kwargs = dict(
            language=self.language,
            beam_size=max(1, int(self.beam)),
            best_of=1,
            # A tuple lets Whisper re-decode a window with higher temperature when it looks like a loop
            # (compression ratio / low log-probability). A single 0.0 disabled that safety net.
            temperature=(0.0, 0.2, 0.4),
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 250, "speech_pad_ms": 150},
            condition_on_previous_text=False,
            word_timestamps=False,
            without_timestamps=False,
        )
        segs = info = None
        if self.device == "cuda":
            try:
                from faster_whisper import BatchedInferencePipeline
                if self._batch is None:
                    self._batch = BatchedInferencePipeline(model=m)
                segs, info = self._batch.transcribe(str(wav16k), batch_size=8, **kwargs)
            except Exception as e:
                log.info("Batched Whisper unavailable (%s); using standard transcribe", e)
                segs = None
        if segs is None:
            segs, info = m.transcribe(str(wav16k), **kwargs)
        if self.language is None and info is not None:
            self.language = info.language          # detect once (first chunk), reuse for the rest
        out = []
        for s in segs:
            text = collapse_repetitions(s.text.strip())
            if text:
                out.append({"start": offset + s.start, "end": offset + s.end, "text": text})
        return out


def transcribe_chunked(rec: Recognizer, wav16k: Path, duration: float, work_dir: Path, chunk_s: float, overlap_s: float,
                       done: Optional[dict[int, list[dict]]] = None,
                       on_chunk: Optional[Callable[[int, int, list[dict]], None]] = None,
                       control: Optional[Control] = None) -> list[dict]:
    """Transcribe in overlapped windows; keep each segment in exactly one chunk (midpoint rule).
    `done` = already finished chunks (from a checkpoint); `on_chunk` is called after each new chunk
    so the caller can persist progress."""
    wins = plan_windows(duration, chunk_s, overlap_s)
    done = dict(done or {})
    work_dir.mkdir(parents=True, exist_ok=True)
    for w in wins:
        if w["index"] in done:
            continue
        if control:
            control.checkpoint()
        part = work_dir / f"asr_{w['index']:04d}.wav"
        cut_wav(wav16k, w["win_start"], w["win_end"], part, control)
        try:
            segs = rec.transcribe(part, w["win_start"])
        finally:
            part.unlink(missing_ok=True)
        done[w["index"]] = keep_by_midpoint(segs, w["start"], w["end"], last=w["index"] == len(wins) - 1)
        if on_chunk:
            on_chunk(w["index"], len(wins), done[w["index"]])
    out = [s for i in sorted(done) for s in done[i]]
    out.sort(key=lambda s: s["start"])
    return out


_SENT_END = (".", "!", "?", "…", "。", "！", "？", '."', '?"', '!"')


def merge_fragments(segments: list[dict], max_gap: float = 0.6, max_len: float = 12.0) -> list[dict]:
    """Whisper often cuts sentences in the middle. Merge consecutive fragments of the SAME speaker that do not
    end a sentence, so translation sees the whole sentence (context) and TTS speaks it naturally."""
    out: list[dict] = []
    for sg in segments:
        if out:
            p = out[-1]
            if (p["speaker_id"] == sg["speaker_id"] and not p["text"].rstrip().endswith(_SENT_END)
                    and sg["start"] - p["end"] <= max_gap and sg["end"] - p["start"] <= max_len):
                p["text"] = (p["text"].rstrip() + " " + sg["text"].lstrip()).strip()
                p["end"] = sg["end"]
                continue
        out.append(dict(sg))
    for i, sg in enumerate(out):
        sg["id"] = i
    return out
