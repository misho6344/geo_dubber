"""Dialogue / background separation.

DemucsSeparator runs `python -m demucs --two-stems=vocals` per chunk and concatenates the
"no_vocals" stems (music + effects). NOT run in this build environment (no model access):
test it on your PC. FallbackSeparator does no separation and says so.
"""
from __future__ import annotations

import logging
import os
import re
import time
import shutil
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

from core.audio_extractor import cut_wav, plan_windows
from core.audio_io import wav_duration
from core.errors import MediaProcessingError, MissingDependencyError
from core.ffmpeg_utils import Control, run_ffmpeg

log = logging.getLogger(__name__)


CPU_BACKGROUND_SEPARATION_SKIP_SECONDS = 3 * 60 * 60


def should_skip_cpu_background_separation(device: str, duration: float) -> bool:
    """Avoid extremely long CPU Demucs runs; films up to 3 hours still get real separation."""
    return str(device or "").lower() == "cpu" and float(duration or 0) > CPU_BACKGROUND_SEPARATION_SKIP_SECONDS


class Separator(ABC):
    reliable = True

    @abstractmethod
    def separate(self, audio_wav: Path, duration: float, out_background: Path,
                 on_progress: Optional[Callable[[float], None]] = None, control: Optional[Control] = None) -> Optional[Path]:
        """Write the background-only track to out_background (or return None if impossible)."""


class DemucsSeparator(Separator):
    def __init__(self, device: str = "cpu", model: str = "htdemucs", chunk_s: float = 600.0,
                 work_dir: Optional[Path] = None, segment: str = "8"):
        self.device, self.model, self.chunk_s, self.work_dir = device, model, chunk_s, work_dir
        # Demucs parses --segment as an integer. Passing "7.0" (from a float-valued
        # setting) makes argparse fail and silently disables background separation.
        # htdemucs supports up to 7.8 seconds, so use the largest safe integer.
        try:
            self.segment = str(max(1, min(7, int(float(segment or 7)))))
        except (TypeError, ValueError, OverflowError):
            self.segment = "7"

    @staticmethod
    def _progress_from_output(output: str) -> Optional[float]:
        """Read the latest tqdm percentage from Demucs output, if available."""
        matches = re.findall(r"(?<!\d)(\d{1,3})%", output)
        if not matches:
            return None
        return max(0.0, min(1.0, int(matches[-1]) / 100.0))

    def _run_demucs_chunk(self, cmd, log_path: Path, control=None,
                          on_progress=None, chunk_base=0.0, chunk_weight=1.0):
        """Run Demucs while exposing its progress and checking cancellation."""
        from core.errors import CancelledError
        with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
            proc = subprocess.Popen(
                cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            read_position = 0
            last_fraction = -1.0
            output_tail = ""
            try:
                while proc.poll() is None:
                    if control is not None:
                        if control.cancelled:
                            proc.terminate()
                            try:
                                proc.wait(timeout=8)
                            except subprocess.TimeoutExpired:
                                proc.kill()
                                proc.wait()
                            raise CancelledError()
                        control.checkpoint()
                    with log_path.open("r", encoding="utf-8", errors="replace") as reader:
                        reader.seek(read_position)
                        fresh = reader.read()
                        read_position = reader.tell()
                    if fresh:
                        output_tail = (output_tail + fresh)[-2000:]
                        fraction = self._progress_from_output(output_tail)
                        if fraction is not None and fraction > last_fraction:
                            last_fraction = fraction
                            if on_progress:
                                on_progress(chunk_base + chunk_weight * fraction)
                    time.sleep(0.35)
                with log_path.open("r", encoding="utf-8", errors="replace") as reader:
                    output = reader.read()
            except BaseException:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=8)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                raise
        if proc.returncode != 0:
            log.warning("Demucs failed (code %s): %s", proc.returncode, output[-1200:])
            return False
        return True

    def separate(self, audio_wav, duration, out_background, on_progress=None, control=None):
        """Return a verified music/effects stem, or None for the documented fallback."""
        try:
            import demucs  # noqa: F401
        except ImportError:
            log.warning("demucs not installed; using fallback background")
            return None

        out_background = Path(out_background)
        out_background.parent.mkdir(parents=True, exist_ok=True)
        temporary_output = out_background.with_name(
            f"{out_background.stem}.partial{out_background.suffix or '.wav'}"
        )
        temporary_output.unlink(missing_ok=True)
        work = Path(tempfile.mkdtemp(prefix="sep_", dir=self.work_dir))
        try:
            wins = plan_windows(duration, self.chunk_s, 0.0)
            stems = []
            for index, w in enumerate(wins):
                if control:
                    control.checkpoint()
                chunk = work / f"c{w['index']:04d}.wav"
                cut_wav(audio_wav, w["start"], w["end"], chunk, control)
                cmd = [sys.executable, "-m", "demucs", "--two-stems=vocals", "-n", self.model, "-d", self.device,
                       "--shifts", "0", "--overlap", "0.05", "--segment", str(self.segment),
                       "-j", "1" if self.device != "cuda" else "2",
                       "-o", str(work / "out"), str(chunk)]
                log_path = work / f"demucs_{index:04d}.log"
                base = index / len(wins)
                weight = 1.0 / len(wins)
                progress_callback = (
                    lambda inner, b=base, wgt=weight: on_progress(b + wgt * inner)
                ) if on_progress else None
                ok = self._run_demucs_chunk(
                    cmd, log_path, control, progress_callback, base, weight
                )
                stem = work / "out" / self.model / chunk.stem / "no_vocals.wav"
                if not ok or not stem.exists() or stem.stat().st_size < 1000:
                    log.warning("Demucs produced no usable background stem for chunk %s; using fallback", index + 1)
                    return None
                stems.append(stem)
                chunk.unlink(missing_ok=True)
                if on_progress:
                    on_progress((index + 1) / len(wins))
            lst = work / "list.txt"
            lst.write_text("".join(f"file '{s.as_posix()}'\n" for s in stems), encoding="utf-8")
            run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(lst), "-c:a", "pcm_s16le",
                        str(temporary_output)], control=control)
            if not temporary_output.exists() or temporary_output.stat().st_size < 1000:
                raise MediaProcessingError(
                    "empty background stem",
                    "Demucs completed without creating a usable background audio file.",
                )
            actual_duration = wav_duration(temporary_output)
            tolerance = max(2.0, float(duration) * 0.005)
            if duration > 0 and abs(actual_duration - float(duration)) > tolerance:
                raise MediaProcessingError(
                    "incomplete background stem",
                    f"Demucs background audio is {actual_duration:.2f}s; expected about {float(duration):.2f}s.",
                )
            os.replace(temporary_output, out_background)
            return out_background
        except Exception as e:
            from core.errors import CancelledError
            if isinstance(e, CancelledError):
                raise
            log.exception("Demucs separation failed; using fallback")
            return None
        finally:
            temporary_output.unlink(missing_ok=True)
            shutil.rmtree(work, ignore_errors=True)

class FallbackSeparator(Separator):
    """No separation available: the pipeline keeps the original track under the dub at low volume
    (so original speech stays faintly audible) and shows a warning."""
    reliable = False

    def separate(self, audio_wav, duration, out_background, on_progress=None, control=None):
        return None


def create_separator(device: str, work_dir: Optional[Path] = None, segment: str = "8") -> Separator:
    try:
        import demucs  # noqa: F401
        return DemucsSeparator(device=device, work_dir=work_dir, segment=segment)
    except ImportError:
        return FallbackSeparator()
