"""Dialogue / background separation.

DemucsSeparator runs `python -m demucs --two-stems=vocals` per chunk and concatenates the
"no_vocals" stems (music + effects). NOT run in this build environment (no model access):
test it on your PC. FallbackSeparator does no separation and says so.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

from core.audio_extractor import cut_wav, plan_windows
from core.errors import MediaProcessingError, MissingDependencyError
from core.ffmpeg_utils import Control, run_ffmpeg

log = logging.getLogger(__name__)


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
        # The htdemucs model refuses segments longer than 7.8 s ("Maximum segment is: 7.8"); clamp it.
        try:
            self.segment = str(min(7.8, float(segment or 7.0)))
        except ValueError:
            self.segment = "7"

    def separate(self, audio_wav, duration, out_background, on_progress=None, control=None):
        """Return background path on success, or None so the pipeline keeps a quiet original track."""
        try:
            import demucs  # noqa: F401
        except ImportError:
            log.warning("demucs not installed; using fallback background")
            return None
        work = Path(tempfile.mkdtemp(prefix="sep_", dir=self.work_dir))
        try:
            wins = plan_windows(duration, self.chunk_s, 0.0)
            stems = []
            for w in wins:
                if control:
                    control.checkpoint()
                chunk = work / f"c{w['index']:04d}.wav"
                cut_wav(audio_wav, w["start"], w["end"], chunk, control)
                cmd = [sys.executable, "-m", "demucs", "--two-stems=vocals", "-n", self.model, "-d", self.device,
                       "--shifts", "0", "--overlap", "0.05", "--segment", str(self.segment),
                       "-j", "0" if self.device != "cuda" else "2",
                       "-o", str(work / "out"), str(chunk)]
                p = subprocess.run(cmd, capture_output=True, text=True)
                stem = work / "out" / self.model / chunk.stem / "no_vocals.wav"
                if p.returncode != 0 or not stem.exists():
                    log.warning("Demucs failed (code %s): %s", p.returncode, (p.stderr or "")[-500:])
                    return None
                stems.append(stem)
                chunk.unlink(missing_ok=True)
                if on_progress:
                    on_progress((w["index"] + 1) / len(wins))
            lst = work / "list.txt"
            lst.write_text("".join(f"file '{s.as_posix()}'\n" for s in stems), encoding="utf-8")
            run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(lst), "-c:a", "pcm_s16le", str(out_background)], control=control)
            return out_background
        except Exception as e:
            from core.errors import CancelledError
            if isinstance(e, CancelledError):
                raise
            log.exception("Demucs separation failed; using fallback")
            return None
        finally:
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
