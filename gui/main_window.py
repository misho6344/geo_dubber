"""Main window."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QLayout,
                               QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
                               QScrollArea, QVBoxLayout, QWidget)

from config.settings import APP_NAME, APP_VERSION, Settings, sub_dir
from core import hardware
from core import video_loader as VL
from core.errors import GeoDubberError
from core.processing_manager import STAGES, Pipeline
from core.subtitles import write_srt
from database.project_database import ProjectDatabase
from gui.settings_window import SettingsDialog
from gui.translation_editor import ReviewDialog
from gui.voice_manager import AddVoiceDialog
from gui.worker import DubWorker

log = logging.getLogger(__name__)


def panel(title: str | None = None) -> tuple[QFrame, QVBoxLayout]:
    f = QFrame()
    f.setObjectName("panel")
    v = QVBoxLayout(f)
    v.setContentsMargins(15, 12, 15, 13)
    v.setSpacing(8)
    if title:
        l = QLabel(title)
        l.setObjectName("sectionTitle")
        v.addWidget(l)
    return f, v


def fmt_eta(sec):
    if sec is None:
        return "estimating…"
    m, s = divmod(int(sec), 60); h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d} left (this stage)" if h else f"{m}:{s:02d} left (this stage)"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {APP_VERSION}")
        self.resize(1080, 860)
        self.settings = Settings.load()
        self.db = ProjectDatabase()
        self.worker: DubWorker | None = None
        self.info: VL.MediaInfo | None = None
        self.project_dir: Path | None = None
        self.project_id: str | None = None
        self.last_pipeline: Pipeline | None = None

        root = QWidget()
        root.setObjectName("root")
        lay = QVBoxLayout(root)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(9)
        lay.setSizeConstraint(QLayout.SetMinimumSize)

        # Keep the full interface usable on shorter displays instead of
        # compressing panels until their controls overlap.
        scroll = QScrollArea()
        scroll.setObjectName("mainScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setWidget(root)
        self.setCentralWidget(scroll)

        # --- A. header
        h = QHBoxLayout()
        t = QLabel("GEODUBBER"); t.setObjectName("title")
        v = QLabel(f"v{APP_VERSION}"); v.setObjectName("muted")
        h.addWidget(t); h.addWidget(v); h.addStretch(1)
        for text, fn in (("Projects", self.open_projects), ("Settings", self.open_settings), ("Help", self.show_help)):
            b = QPushButton(text); b.clicked.connect(fn); h.addWidget(b)
        lay.addLayout(h)

        # --- B. video input
        f, v = panel("1 · Video")
        f.setMinimumHeight(255)
        row = QHBoxLayout()
        self.path_edit = QLineEdit(); self.path_edit.setPlaceholderText("Select a video file (MP4, MKV, AVI, MOV, WEBM)…")
        self.path_edit.editingFinished.connect(self.load_source)
        b = QPushButton("Browse…"); b.clicked.connect(self.browse)
        row.addWidget(self.path_edit, 1); row.addWidget(b); v.addLayout(row)
        self.url_edit = QLineEdit(); self.url_edit.setPlaceholderText("…or paste a YouTube URL (only for videos you own or may download)")
        v.addWidget(self.url_edit)
        # Imported timed script / subtitles
        sub_row = QHBoxLayout()
        self.sub_edit = QLineEdit()
        self.sub_edit.setPlaceholderText("Optional: Georgian timed subtitles (SRT / VTT / TXT) — use as-is, skip ASR & translation")
        if getattr(self.settings, "imported_subtitles", ""):
            self.sub_edit.setText(self.settings.imported_subtitles)
        sub_btn = QPushButton("Import subtitles…")
        sub_btn.clicked.connect(self.browse_subtitles)
        sub_clear = QPushButton("Clear")
        sub_clear.clicked.connect(lambda: self.sub_edit.setText(""))
        sub_row.addWidget(self.sub_edit, 1)
        sub_row.addWidget(sub_btn)
        sub_row.addWidget(sub_clear)
        v.addLayout(sub_row)
        sub_hint = QLabel(
            "When a subtitle file is attached, GeoDubber uses its text and timings exactly as supplied: "
            "audio transcription and automatic translation are skipped. Use a Georgian subtitle file for "
            "Georgian voice output. Without a file, GeoDubber transcribes the audio and translates it into Georgian. "
            "Optional speaker tags: [speaker_1], [male], [female] at the start of a line."
        )
        sub_hint.setObjectName("muted")
        sub_hint.setWordWrap(True)
        v.addWidget(sub_hint)
        self.info_label = QLabel("No video selected"); self.info_label.setObjectName("muted"); self.info_label.setWordWrap(True)
        v.addWidget(self.info_label)
        lay.addWidget(f)

        # --- D. processing options (voices are reviewed after detection)
        f, v = panel("2 · Options")
        f.setMinimumHeight(315)
        g = QGridLayout()
        g.setContentsMargins(0, 2, 0, 2)
        g.setHorizontalSpacing(14)
        g.setVerticalSpacing(7)
        v.addLayout(g)
        self.priority = QComboBox(); self.priority.addItems(["fast", "balanced", "quality"]); self.priority.setCurrentText(self.settings.priority)
        self.device = QComboBox(); self.device.addItems(["auto", "cuda", "cpu"]); self.device.setCurrentText(self.settings.device)
        g.addWidget(QLabel("Priority"), 0, 0); g.addWidget(self.priority, 0, 1)
        g.addWidget(QLabel("GPU"), 0, 2); g.addWidget(self.device, 0, 3)
        self.cb = {}
        for i, (key, text) in enumerate([("preserve_background", "Preserve background music & effects"),
                                         ("auto_speakers", "Detect characters & gender (Gemini)"),
                                         ("georgian_subtitles", "Georgian subtitles"),
                                         ("original_subtitles", "Original-language subtitle file"),
                                         ("review_translation", "Review translation & voices before voice generation")]):
            c = QCheckBox(text); c.setChecked(getattr(self.settings, key)); self.cb[key] = c
            g.addWidget(c, 1 + i // 2, (i % 2) * 2, 1, 2)
        g.addWidget(QLabel("Characters in video"), 4, 0)
        self.speaker_count = QComboBox()
        for n, lab in [(0, "Auto — Gemini decides"), (1, "1 character"), (2, "2 characters"),
                       (3, "3 characters"), (4, "4 characters")]:
            self.speaker_count.addItem(lab, n)
        sc = int(getattr(self.settings, "speaker_count", 0) or 0)
        self.speaker_count.setCurrentIndex(max(0, self.speaker_count.findData(sc)))
        g.addWidget(self.speaker_count, 4, 1)
        self.char_hint = QLabel()
        self.char_hint.setObjectName("muted"); self.char_hint.setWordWrap(True)
        g.addWidget(self.char_hint, 4, 2, 1, 2)

        # Character rows are shown only as many as the video needs (chosen by Gemini or by you).
        self.char_labels, self.spk_boxes = [], []
        for i in range(4):
            lab = QLabel(f"Character {i + 1}")
            box = QComboBox()
            box.addItem("Auto", "auto"); box.addItem("Male (კაცი)", "male"); box.addItem("Female (ქალი)", "female")
            box.setCurrentIndex(max(0, box.findData(getattr(self.settings, f"speaker{i + 1}_gender", "auto") or "auto")))
            r, c = 5 + i // 2, (i % 2) * 2
            g.addWidget(lab, r, c); g.addWidget(box, r, c + 1)
            self.char_labels.append(lab); self.spk_boxes.append(box)
            box.activated.connect(self._characters_edited_by_user)
        self.spk1_gender, self.spk2_gender, self.spk3_gender, self.spk4_gender = self.spk_boxes
        self._auto_detected = False          # True while the character settings show what Gemini detected
        self._run_auto = False
        self._detect_source = ""
        self.speaker_count.activated.connect(self._characters_edited_by_user)
        self.speaker_count.currentIndexChanged.connect(lambda _i: self._refresh_character_rows())
        self._refresh_character_rows()
        lay.addWidget(f)

        # --- progress
        f, v = panel("3 · Progress")
        f.setMinimumHeight(210)
        self.overall = QProgressBar(); self.overall.setRange(0, 1000); v.addWidget(self.overall)
        self.stage_labels = {}
        sg = QGridLayout()
        sg.setContentsMargins(0, 2, 0, 2)
        sg.setHorizontalSpacing(22)
        sg.setVerticalSpacing(5)
        for i, (key, label, _w) in enumerate(STAGES):
            l = QLabel(f"{i + 1}. {label} — waiting"); l.setObjectName("muted"); self.stage_labels[key] = (l, label)
            sg.addWidget(l, i % 5, i // 5)
        v.addLayout(sg)
        self.status = QLabel(""); v.addWidget(self.status)
        self.warn_label = QLabel(""); self.warn_label.setObjectName("warn"); self.warn_label.setWordWrap(True); v.addWidget(self.warn_label)
        lay.addWidget(f)

        # --- E. controls
        row = QHBoxLayout()
        self.start_btn = QPushButton("Start Dubbing"); self.start_btn.setObjectName("primary"); self.start_btn.clicked.connect(self.start)
        self.pause_btn = QPushButton("Pause"); self.resume_btn = QPushButton("Resume")
        self.stop_btn = QPushButton("Stop (keep progress)"); self.cancel_btn = QPushButton("Cancel and delete project")
        self._delete_after_stop = False
        self.pause_btn.clicked.connect(lambda: self.worker and self.worker.pause() or self._paused(True))
        self.resume_btn.clicked.connect(lambda: self.worker and self.worker.resume() or self._paused(False))
        self.stop_btn.clicked.connect(self.stop)
        self.cancel_btn.clicked.connect(self.cancel_and_delete)
        self.open_btn = QPushButton("Open Output Folder"); self.open_btn.clicked.connect(self.open_output)
        self.preview_btn = QPushButton("Preview Translation"); self.preview_btn.clicked.connect(self.preview_translation)
        self.srt_btn = QPushButton("Export Subtitles"); self.srt_btn.clicked.connect(self.export_subs)
        for b in (self.start_btn, self.pause_btn, self.resume_btn, self.stop_btn, self.cancel_btn, self.preview_btn, self.srt_btn, self.open_btn):
            row.addWidget(b)
        lay.addLayout(row)

        # --- hardware + log
        f, v = panel("System")
        f.setMinimumHeight(82)
        self.hw = QLabel(); self.hw.setObjectName("muted"); self.hw.setWordWrap(True); v.addWidget(self.hw)
        lay.addWidget(f)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        self.log.setMinimumHeight(72)
        self.log.setMaximumHeight(96)
        lay.addWidget(self.log)

        self.refresh_hw()
        self._buttons(running=False)
        self.statusBar().showMessage("Ready")

    # ------------------------------------------------------------ helpers
    def refresh_hw(self):
        d = hardware.detect(sub_dir("cache"))
        dev, note = hardware.resolve_device(self.settings.device)
        self.hw.setText(f"CPU: {d['cpu']} ({d['cpu_cores']} threads)   RAM: {d['ram_available_gb']}/{d['ram_total_gb']} GB free   "
                        f"GPU: {d['gpu']}   CUDA: {'yes' if d['cuda_available'] else 'no'}   Disk free: {d['disk_free_gb']} GB   → using {dev} {note}")

    def _buttons(self, running: bool, paused: bool = False):
        self.start_btn.setEnabled(not running)
        self.pause_btn.setEnabled(running and not paused)
        self.resume_btn.setEnabled(running and paused)
        self.stop_btn.setEnabled(running)
        # Cancel-and-delete works whenever a project is running or loaded (also after Preview Translation).
        self.cancel_btn.setEnabled(running or self.project_id is not None)
        self.preview_btn.setEnabled(not running and self.last_pipeline is not None)
        self.srt_btn.setEnabled(not running and self.last_pipeline is not None)

    def _paused(self, p):
        self._buttons(True, p)
        self.status.setText("Paused — will stop at the next safe point" if p else "")

    def _apply_ui_to_settings(self):
        s = self.settings
        s.priority, s.device = self.priority.currentText(), self.device.currentText()
        for k, c in self.cb.items():
            setattr(s, k, c.isChecked())
        if self._auto_detected:
            # The boxes only show what Gemini detected; keep the saved setting on "Auto" for the next video.
            s.speaker_count = 0
            for i in range(4):
                setattr(s, f"speaker{i + 1}_gender", "auto")
        else:
            s.speaker_count = int(self.speaker_count.currentData() or 0)
            n = s.speaker_count
            for i, box in enumerate(self.spk_boxes):
                setattr(s, f"speaker{i + 1}_gender", (box.currentData() or "auto") if (n == 0 or i < n) else "auto")
        s.imported_subtitles = (self.sub_edit.text() or "").strip().strip('"')
        s.save()

    def append_log(self, text: str):
        self.log.appendPlainText(text)

    # ------------------------------------------------------------ characters
    def _refresh_character_rows(self):
        """Show only the characters the project needs; the rest of the settings follow from that."""
        n = int(self.speaker_count.currentData() or 0)
        for i, (lab, box) in enumerate(zip(self.char_labels, self.spk_boxes)):
            show = n > 0 and i < n
            lab.setVisible(show); box.setVisible(show)
        if self._auto_detected:
            self.char_hint.setText(f"Detected by Gemini: {n} character(s). Voices: Giorgi (male), Nino (female). "
                                   "Change the number to override.")
        elif n == 0:
            self.char_hint.setText("Gemini will listen to the video and set the number of characters, "
                                   "their gender and voice (Giorgi / Nino) automatically.")
        else:
            self.char_hint.setText("Set each character's gender, or leave Auto. Voices: Giorgi (male), Nino (female).")

    def _characters_edited_by_user(self, *_):
        self._auto_detected = False
        self._refresh_character_rows()

    def _reset_characters(self):
        self._auto_detected = False
        self.speaker_count.blockSignals(True)
        self.speaker_count.setCurrentIndex(max(0, self.speaker_count.findData(0)))
        self.speaker_count.blockSignals(False)
        for box in self.spk_boxes:
            box.setCurrentIndex(0)
        self._refresh_character_rows()

    def on_characters_detected(self, count: int, genders: list):
        """Gemini decided how many characters there are -> update the project settings to match."""
        if not self._run_auto:
            return
        count = max(1, min(4, int(count)))
        self._auto_detected = True
        self.speaker_count.blockSignals(True)
        self.speaker_count.setCurrentIndex(max(0, self.speaker_count.findData(count)))
        self.speaker_count.blockSignals(False)
        for i, box in enumerate(self.spk_boxes):
            g = genders[i] if i < len(genders) else "auto"
            box.setCurrentIndex(max(0, box.findData(g if g in ("male", "female") else "auto")))
        self._refresh_character_rows()
        names = ", ".join(f"{i + 1}: {'Giorgi' if g == 'male' else 'Nino' if g == 'female' else '?'}"
                          for i, g in enumerate(genders[:count]))
        self.append_log(f"Characters: {count} ({names})")

    # ------------------------------------------------------------ input
    def browse(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select video", "", "Video (*.mp4 *.mkv *.avi *.mov *.webm)")
        if f:
            self.path_edit.setText(f); self.load_source()

    def browse_subtitles(self):
        f, _ = QFileDialog.getOpenFileName(
            self,
            "Import timed script / subtitles",
            "",
            "Subtitles (*.srt *.vtt *.txt);;SRT (*.srt);;WebVTT (*.vtt);;Timed text (*.txt);;All (*.*)",
        )
        if f:
            self.sub_edit.setText(f)

    def load_source(self):
        p = self.path_edit.text().strip().strip('"')
        if not p:
            return
        if self._auto_detected and p != self._detect_source:
            self._reset_characters()             # a different video: Gemini has to decide again
        try:
            self.info = VL.probe(p)
            VL.validate_for_dubbing(self.info)
            self.info_label.setText(self.info.summary())
        except GeoDubberError as e:
            self.info = None
            self.info_label.setText("⚠ " + e.user_message)

    # ------------------------------------------------------------ run
    def _project_dir_for(self, source: str) -> Path:
        h = hashlib.sha1(source.encode("utf-8")).hexdigest()[:8]
        stem = "".join(c for c in Path(source).stem if c.isalnum() or c in "-_ ")[:40].strip() or "project"
        d = sub_dir("projects") / f"{stem}_{h}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def start(self, source: str | None = None, project_dir: Path | None = None):
        source = source if isinstance(source, str) else (self.url_edit.text().strip() or self.path_edit.text().strip().strip('"'))
        if not source:
            QMessageBox.information(self, APP_NAME, "Select a video file or paste a YouTube URL first.")
            return
        if not VL.is_youtube_url(source):
            self.load_source()
            if not self.info:
                QMessageBox.warning(self, APP_NAME, self.info_label.text())
                return
        if self._auto_detected and source != self._detect_source:
            self._reset_characters()
        self._apply_ui_to_settings()
        self._run_auto = int(self.settings.speaker_count or 0) == 0
        self._detect_source = source
        self.project_dir = project_dir or self._project_dir_for(source)
        existing = self.db.projects()
        pid = next((p["id"] for p in existing if p["project_dir"] == str(self.project_dir)), None)
        self.project_id = pid or self.db.create_project(Path(source).stem if not VL.is_youtube_url(source) else source, source, str(self.project_dir))
        self.db.set_status(self.project_id, "running")
        self.worker = DubWorker(source, self.settings, self.project_dir, self)
        self.worker.progress.connect(self.on_progress)
        self.worker.review_needed.connect(self.on_review)
        self.worker.finished_ok.connect(self.on_done)
        self.worker.failed.connect(self.on_failed)
        self.worker.cancelled.connect(self.on_cancelled)
        self.worker.log_line.connect(self.append_log)
        self.worker.characters_detected.connect(self.on_characters_detected)
        self._buttons(True)
        self.append_log(f"Project: {self.project_dir}")
        self.worker.start()

    def stop(self):
        if self.worker and QMessageBox.question(self, APP_NAME, "Stop processing? Finished steps are saved and can be resumed later.") == QMessageBox.Yes:
            self.worker.cancel()

    def cancel_and_delete(self):
        if not self.project_id:
            return
        msg = ("Delete this project completely?\n\n"
               "The dubbed video, subtitles, report and all work files are deleted.\n"
               "The ORIGINAL video is not deleted.\n\nThis cannot be undone.")
        if QMessageBox.question(self, APP_NAME, msg) != QMessageBox.Yes:
            return
        if self.worker and self.worker.isRunning():
            self._delete_after_stop = True     # purged in _finish, once the worker has stopped
            self.worker.cancel()
            self.append_log("Stopping, then deleting this project...")
        else:
            self._purge_project()

    def _project_outputs(self, project_dir: Path, video_path: str) -> list[Path]:
        """Files the export wrote (dubbed video, subtitles, report). The original video is never included."""
        try:
            st = json.loads((project_dir / "state.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        original = Path(video_path).resolve() if video_path and not video_path.startswith("http") else None
        out = []
        for v in (st.get("outputs") or {}).values():
            if not v:
                continue
            path = Path(v)
            if original and path.resolve() == original:
                continue
            out.append(path)
        return out

    def _purge_project(self) -> None:
        row = next((r for r in self.db.projects() if r["id"] == self.project_id), None)
        if not row:
            return
        pdir = Path(row["project_dir"])
        keep = pdir / "download"               # a YouTube video is downloaded here: it is the original, keep it
        for path in self._project_outputs(pdir, row.get("video_path") or ""):
            try:
                path.unlink()
            except OSError:
                pass
        if pdir.exists():
            for child in pdir.iterdir():
                if child == keep:
                    continue
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    try:
                        child.unlink()
                    except OSError:
                        pass
        self.db.delete_project(row["id"])
        self.append_log(f"Project deleted: {pdir}")
        self.project_id = None
        self.last_pipeline = None
        self.status.setText("Project deleted")
        self._buttons(running=False)

    def on_progress(self, p):
        self.overall.setValue(int(p.overall * 1000))
        for key, (lab, name) in self.stage_labels.items():
            st = p.stage_status.get(key, "waiting")
            lab.setText(f"{[k for k, *_ in STAGES].index(key) + 1}. {name} — {st}")
            lab.setStyleSheet("color:#e0a526;font-weight:600;" if key == p.stage and st not in ("done",) else
                              ("color:#7fd18b;" if st == "done" else ""))
        self.status.setText(f"{p.stage_label}: {int(p.stage_fraction * 100)}%  {p.message}  {('· ' + p.speed) if p.speed else ''}  "
                            f"{('· ' + fmt_eta(p.eta_seconds)) if p.stage_fraction < 1 else ''}")
        if self.worker and self.worker.pipeline:
            w = self.worker.pipeline.state.get("warnings", [])
            self.warn_label.setText("⚠ " + "\n⚠ ".join(w[-3:]) if w else "")

    def on_review(self):
        pl = self.worker.pipeline
        dlg = ReviewDialog(pl, pl.all_voices(), pl.be.tts, self, voice_add_cb=lambda: self._add_voice(dlg))
        dlg.exec()
        dlg.apply()
        self.worker.review_done()

    def _add_voice(self, dlg: ReviewDialog):
        if AddVoiceDialog(dlg).exec():
            pl = dlg.pl
            pl.state["assignments"] = dlg.voice_panel.assignments
            pl._assign()
            dlg.voice_panel.load(pl.state["speakers"], pl.all_voices(), pl.state["assignments"], pl.be.tts)

    def _finish(self, status, detail=""):
        self.last_pipeline = self.worker.pipeline if self.worker else self.last_pipeline
        if self.project_id:
            self.db.set_status(self.project_id, status, detail)
            if self.last_pipeline:
                self.db.save_assignments(self.project_id, self.last_pipeline.state.get("assignments", {}))
        self._buttons(False)
        if self._delete_after_stop:
            self._delete_after_stop = False
            self._purge_project()

    def on_done(self, outs: dict):
        self._finish("done", outs.get("video", ""))
        self.append_log("Finished: " + outs.get("video", ""))
        QMessageBox.information(self, APP_NAME, f"Dubbing finished.\n\n{outs.get('video', '')}")

    def on_failed(self, msg: str):
        self._finish("failed", msg)
        self.append_log("ERROR: " + msg)
        QMessageBox.critical(self, APP_NAME, msg + "\n\nFinished steps are saved: fix the problem and press Start Dubbing to resume.")

    def on_cancelled(self):
        self._finish("cancelled")
        self.append_log("Cancelled. Press Start Dubbing to resume this project.")
        self.status.setText("Cancelled")

    # ------------------------------------------------------------ after-run tools
    def preview_translation(self):
        pl = self.last_pipeline
        if not pl or not pl.state.get("segments"):
            return
        dlg = ReviewDialog(pl, pl.all_voices(), pl.be.tts, self)
        dlg.exec(); dlg.apply()
        if QMessageBox.question(self, APP_NAME, "Re-generate changed lines and re-export the video now?") == QMessageBox.Yes:
            pl.invalidate("tts")
            self.start(pl.state["video"], pl.dir)

    def export_subs(self):
        pl = self.last_pipeline
        if not pl:
            return
        out = self.settings.resolved_output_dir()
        stem = Path(pl.state["video"]).stem
        a = write_srt(pl.state["segments"], out / f"{stem}_GE.srt", "translation")
        b = write_srt(pl.state["segments"], out / f"{stem}_original.srt", "text")
        QMessageBox.information(self, APP_NAME, f"Saved:\n{a}\n{b}")

    def open_output(self):
        d = self.settings.resolved_output_dir()
        if sys.platform.startswith("win"):
            os.startfile(str(d))                      # noqa
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(d)])
        else:
            subprocess.Popen(["xdg-open", str(d)])

    def open_settings(self):
        SettingsDialog(self.settings, self).exec()
        self.refresh_hw()

    def open_projects(self):
        rows = self.db.projects()
        if not rows:
            QMessageBox.information(self, "Projects", "No projects yet.")
            return
        import shutil
        import time
        from PySide6.QtGui import QColor
        from PySide6.QtWidgets import QDialog, QListWidget, QListWidgetItem

        d = QDialog(self)
        d.setWindowTitle("Projects / history")
        d.resize(900, 560)
        lay = QVBoxLayout(d)
        lay.setContentsMargins(18, 16, 18, 16)
        lay.setSpacing(10)
        title = QLabel("Projects / history"); title.setObjectName("sectionTitle")
        hint = QLabel("Select a project, then press Resume / open to continue it (finished steps are reused).")
        hint.setObjectName("muted"); hint.setWordWrap(True)
        lay.addWidget(title); lay.addWidget(hint)

        lst = QListWidget()
        lst.setObjectName("projectList")
        lst.setTextElideMode(Qt.ElideMiddle)
        lst.setWordWrap(False)
        lst.setUniformItemSizes(True)
        lay.addWidget(lst, 1)

        status_look = {"done": "✔ Done", "failed": "✖ Failed", "running": "● Running", "cancelled": "■ Cancelled"}

        def fill():
            lst.clear()
            for r in rows:
                label = status_look.get(r["status"], f"○ {r['status']}")
                when = time.strftime("%Y-%m-%d  %H:%M", time.localtime(r["updated"]))
                it = QListWidgetItem(f"{r['name']}\n{label}   ·   {when}")
                it.setData(Qt.UserRole, r["id"])
                it.setToolTip(f"{r['name']}\n{r['video_path']}")
                it.setForeground(QColor("#F3F6FC"))
                lst.addItem(it)
        fill()
        lst.setCurrentRow(0)

        row = QHBoxLayout()
        res = QPushButton("Resume / open"); res.setObjectName("primary")
        dele = QPushButton("Delete project data")
        close = QPushButton("Close")
        row.addWidget(res, 2); row.addWidget(dele, 2); row.addWidget(close, 1)
        lay.addLayout(row)

        def current():
            i = lst.currentRow()
            if i < 0:
                return None, -1
            pid = lst.item(i).data(Qt.UserRole)
            return next((r for r in rows if r["id"] == pid), None), i

        def update_buttons():
            ok = current()[0] is not None
            res.setEnabled(ok); dele.setEnabled(ok)

        def resume():
            p, _ = current()
            if p:
                d.accept(); self.path_edit.setText(p["video_path"]); self.start(p["video_path"], Path(p["project_dir"]))

        def delete():
            p, i = current()
            if p and QMessageBox.question(d, "Delete", "Delete this project and its temporary files?") == QMessageBox.Yes:
                shutil.rmtree(p["project_dir"], ignore_errors=True); self.db.delete_project(p["id"])
                rows.remove(p); lst.takeItem(i)
                update_buttons()

        res.clicked.connect(resume); dele.clicked.connect(delete); close.clicked.connect(d.reject)
        lst.itemDoubleClicked.connect(lambda _it: resume())
        lst.currentRowChanged.connect(lambda _r: update_buttons())
        update_buttons()
        d.exec()

    def show_help(self):
        QMessageBox.information(self, "Help", (
            "1. Select a video (or paste a URL you are allowed to download).\n"
            "2. Choose Fast / Balanced / Quality and press Start Dubbing.\n"
            "3. After detection, review characters, genders, voices and translations, then Continue.\n"
            "4. Wait for completion; open the output folder.\n\n"
            "First run: Settings > Models to download the models, and Settings > Voices to add reference voices "
            "(one per character you need, at least one male and one female).\n\n"
            "Pause stops at the next safe point; Cancel keeps finished steps so you can resume."))
