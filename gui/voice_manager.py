"""Voice assignment panel + dialog for adding reference voices."""
from __future__ import annotations

import tempfile
from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QLabel, QLineEdit,
                               QMessageBox, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from core.errors import GeoDubberError
from core.georgian_tts import TTSBackend, synthesize_cached
from core.voices import Voice, add_reference_voice


class VoicePanel(QTableWidget):
    """Character | Gender | Voice | ▶ 5s original — user hears original, sets gender, picks Georgian voice."""

    def __init__(self, parent=None):
        super().__init__(0, 4, parent)
        self.setHorizontalHeaderLabels(["Character", "Gender", "Assigned Georgian voice", "Original (~5s)"])
        self.horizontalHeader().setStretchLastSection(True)
        self.setColumnWidth(0, 150), self.setColumnWidth(1, 130), self.setColumnWidth(2, 280)
        self.speakers: dict = {}
        self.voices: list[Voice] = []
        self.assignments: dict = {}
        self.tts: TTSBackend | None = None
        self.previews: dict = {}
        self.player = QMediaPlayer(self)
        self.audio_out = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_out)

    def load(self, speakers: dict, voices: list[Voice], assignments: dict, tts: TTSBackend | None = None,
             previews: dict | None = None):
        self.speakers, self.voices, self.assignments = speakers, voices, dict(assignments)
        self.tts = tts
        self.previews = dict(previews or {})
        for spk, info in self.speakers.items():
            if info.get("preview_path") and spk not in self.previews:
                self.previews[spk] = info["preview_path"]
        self.setRowCount(0)
        for spk, info in sorted(speakers.items()):
            r = self.rowCount()
            self.insertRow(r)
            label = QTableWidgetItem(f"{info.get('label', spk)}  ({info.get('speech_seconds', 0):.0f}s)")
            label.setData(Qt.UserRole, spk)
            label.setFlags(label.flags() & ~Qt.ItemIsEditable)
            self.setItem(r, 0, label)
            g = QComboBox()
            g.addItems(["male", "female", "unknown"])
            g.setCurrentText(info.get("gender", "unknown"))
            if info.get("gender_confidence", 1) < 0.35 and info.get("gender_source") != "user":
                g.setToolTip(f"Uncertain (F0 {info.get('f0')} Hz) — listen to original and set gender")
                g.setStyleSheet("border: 1px solid #ffb454;")
            g.currentTextChanged.connect(lambda t, s=spk: self._gender(s, t))
            self.setCellWidget(r, 1, g)
            v = QComboBox()
            self._fill_voices(v, info.get("gender", "unknown"), self.assignments.get(spk, ""))
            if not self.assignments.get(spk):
                v.setStyleSheet("border: 1px solid #ff6b6b;")
            v.currentIndexChanged.connect(lambda _i, s=spk, w=v: self._voice(s, w.currentData()))
            self.setCellWidget(r, 2, v)
            b = QPushButton("▶ Play 5s")
            b.setToolTip("Play ~5 seconds of this character from the original video")
            b.clicked.connect(lambda _c=False, s=spk: self.preview_original(s))
            self.setCellWidget(r, 3, b)

    def _fill_voices(self, combo: QComboBox, gender: str, selected: str):
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("— none (unassigned) —", "")
        pool = self.voices
        if gender in ("male", "female"):
            matched = [vo for vo in self.voices if vo.gender == gender]
            if matched:
                pool = matched
        for vo in pool:
            combo.addItem(f"{vo.name}  [{vo.gender}]", vo.id)
        idx = combo.findData(selected)
        combo.setCurrentIndex(max(0, idx))
        combo.blockSignals(False)

    def _gender(self, spk, g):
        self.speakers[spk]["gender"] = g
        self.speakers[spk]["gender_source"] = "user"
        self.speakers[spk]["gender_confidence"] = 1.0
        # Prefer a free voice of this gender
        used = {vid for s, vid in self.assignments.items() if s != spk}
        pool = [vo for vo in self.voices if vo.gender == g and vo.id not in used] if g in ("male", "female") else list(self.voices)
        if not pool and g in ("male", "female"):
            pool = [vo for vo in self.voices if vo.gender == g]
        if pool:
            self.assignments[spk] = pool[0].id
        # refresh the voice combo in this row
        for r in range(self.rowCount()):
            it = self.item(r, 0)
            if it and it.data(Qt.UserRole) == spk:
                w = self.cellWidget(r, 2)
                if isinstance(w, QComboBox):
                    self._fill_voices(w, g, self.assignments.get(spk, ""))
                break

    def _voice(self, spk, vid):
        if vid:
            self.assignments[spk] = vid
        else:
            self.assignments.pop(spk, None)

    def preview_original(self, spk):
        path = self.previews.get(spk) or (self.speakers.get(spk) or {}).get("preview_path")
        if not path or not Path(path).exists():
            QMessageBox.information(
                self, "Preview",
                "No original sample for this character yet. Re-run with Review enabled after speaker detection.",
            )
            return
        self.player.setSource(QUrl.fromLocalFile(str(path)))
        self.player.play()

    def preview(self, spk):
        """Backward-compatible: play original sample."""
        self.preview_original(spk)


class AddVoiceDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Georgian reference voice")
        self.resize(520, 300)
        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.name = QLineEdit()
        self.gender = QComboBox()
        self.gender.addItems(["male", "female"])
        self.file = QLineEdit()
        pick = QPushButton("Browse…")
        pick.clicked.connect(self._browse)
        self.text = QLineEdit()
        self.text.setPlaceholderText("Exact Georgian transcript of the clip")
        self.note = QLineEdit()
        self.note.setPlaceholderText("Source / licence (e.g. my own recording, CC0 Common Voice)")
        self.consent = QCheckBox("I own this recording or have permission to use this voice for dubbing")
        form.addRow("Name", self.name), form.addRow("Gender", self.gender)
        form.addRow("Audio clip (5–12 s)", self.file), form.addRow("", pick)
        form.addRow("Transcript", self.text), form.addRow("Source / licence", self.note)
        lay.addLayout(form), lay.addWidget(self.consent)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._ok), bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.voice: Voice | None = None

    def _browse(self):
        f, _ = QFileDialog.getOpenFileName(self, "Reference clip", "", "Audio (*.wav *.mp3 *.flac *.m4a *.ogg)")
        if f:
            self.file.setText(f)

    def _ok(self):
        try:
            self.voice = add_reference_voice(self.name.text() or "voice", self.gender.currentText(), self.file.text(),
                                             self.text.text(), self.consent.isChecked(), self.note.text())
            self.accept()
        except GeoDubberError as e:
            QMessageBox.warning(self, "Cannot add voice", e.user_message)
        except Exception as e:
            QMessageBox.warning(self, "Cannot add voice", str(e))
