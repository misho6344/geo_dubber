"""Runs the pipeline off the GUI thread. Pause/resume/cancel go through core.ffmpeg_utils.Control."""
from __future__ import annotations

import logging
import threading
import traceback
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal

from config.settings import Settings
from core import video_loader as VL
from core.errors import CancelledError, GeoDubberError
from core.ffmpeg_utils import Control
from core.processing_manager import Pipeline, Progress, default_backends

log = logging.getLogger(__name__)


class DubWorker(QThread):
    progress = Signal(object)          # Progress
    review_needed = Signal()           # GUI must show the review dialog, then call review_done()
    finished_ok = Signal(dict)
    failed = Signal(str)               # user-facing message
    cancelled = Signal()
    log_line = Signal(str)
    characters_detected = Signal(int, list)   # (number of characters, [gender per character])

    def __init__(self, source: str, settings: Settings, project_dir: Path, parent=None):
        super().__init__(parent)
        self.source, self.settings, self.project_dir = source, settings, project_dir
        self.control = Control()
        self._review_event = threading.Event()
        self.pipeline: Pipeline | None = None

    # GUI-thread API
    def pause(self): self.control.pause()
    def resume(self): self.control.resume()
    def cancel(self):
        self.control.cancel()
        self._review_event.set()
    def review_done(self): self._review_event.set()

    def _on_review(self, pl: Pipeline):
        self._review_event.clear()
        self.review_needed.emit()
        while not self._review_event.wait(0.2):
            if self.control.cancelled:
                break

    def run(self):
        try:
            subtitle_path = (self.settings.imported_subtitles or "").strip().strip('"')
            if subtitle_path:
                self.log_line.emit(
                    f"Subtitle script selected: {subtitle_path}. "
                    "Audio transcription and automatic translation will be skipped."
                )
            else:
                self.log_line.emit(
                    "No subtitle script selected. Audio transcription and automatic Georgian translation will run."
                )
            src = self.source
            if VL.is_youtube_url(src):
                self.log_line.emit("Downloading video (authorised content only)...")
                src = str(VL.download_youtube(src, self.project_dir / "download"))
            be, dev = default_backends(self.settings, self.project_dir)
            self.log_line.emit(f"Compute device: {dev}")
            self.pipeline = Pipeline(self.project_dir, self.settings, be, self.control, self.progress.emit, self._on_review,
                                     on_characters=self.characters_detected.emit)
            out = self.pipeline.run(src)
            self.finished_ok.emit(out)
        except CancelledError:
            self.cancelled.emit()
        except GeoDubberError as e:
            log.error("pipeline error: %s", e)
            self.failed.emit(e.user_message)
        except Exception as e:  # never crash silently
            log.error("unexpected error\n%s", traceback.format_exc())
            self.failed.emit(f"Unexpected error: {e}\nSee logs/geodubber.log for details.")
