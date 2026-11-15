"""Editable translation preview (original | Georgian), highlights lines likely too long."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QAbstractItemView, QDialog, QDialogButtonBox, QLabel, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout

from core.subtitles import fmt_ts
from gui.voice_manager import VoicePanel


class ReviewDialog(QDialog):
    """Shown between translation and voice generation: fix translations, genders and voices."""

    def __init__(self, pipeline, voices, tts, parent=None, voice_add_cb=None):
        super().__init__(parent)
        self.pl = pipeline
        self.setWindowTitle("Review translation and voices")
        self.resize(1100, 720)
        lay = QVBoxLayout(self)
        warns = pipeline.state.get("warnings", [])
        if warns:
            w = QLabel("⚠ " + "\n⚠ ".join(warns[-6:]))
            w.setObjectName("warn")
            w.setWordWrap(True)
            lay.addWidget(w)
        help_l = QLabel(
            "1) Press ▶ Play 5s for each character and listen to the ORIGINAL voice.\n"
            "2) Set Gender (Male / Female) from what you heard.\n"
            "3) Check the Georgian voice assignment, edit translations if needed, then Continue."
        )
        help_l.setWordWrap(True)
        help_l.setObjectName("muted")
        lay.addWidget(help_l)
        self.voice_panel = VoicePanel()
        self.voice_panel.setMinimumHeight(160)
        self.voice_panel.setMaximumHeight(240)
        previews = dict(pipeline.state.get("speaker_previews") or {})
        for spk, info in (pipeline.state.get("speakers") or {}).items():
            if info.get("preview_path"):
                previews.setdefault(spk, info["preview_path"])
        self.voice_panel.load(pipeline.state["speakers"], voices, pipeline.state["assignments"], tts, previews=previews)
        lay.addWidget(self.voice_panel)
        if voice_add_cb:
            b = QPushButton("＋ Add voice…")
            b.clicked.connect(voice_add_cb)
            lay.addWidget(b)
        spk_ids = sorted({s["speaker_id"] for s in pipeline.state["segments"]})
        if not spk_ids:
            spk_ids = ["speaker_1"]
        self._spk_ids = spk_ids
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Time", "Speaker (edit)", "Original", "Georgian (editable)", "Long?"])
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.setColumnWidth(0, 90), self.table.setColumnWidth(1, 120), self.table.setColumnWidth(2, 340)
        self.table.setColumnWidth(3, 400), self.table.setColumnWidth(4, 50)
        self.table.setWordWrap(True)
        self.table.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        from PySide6.QtWidgets import QComboBox
        for s in pipeline.state["segments"]:
            r = self.table.rowCount()
            self.table.insertRow(r)
            t_item = QTableWidgetItem(fmt_ts(s["start"])[:8])
            t_item.setFlags(t_item.flags() & ~Qt.ItemIsEditable)
            t_item.setData(Qt.UserRole, s["id"])
            self.table.setItem(r, 0, t_item)
            combo = QComboBox()
            for sid in spk_ids:
                lab = sid.replace("speaker_", "S")
                gender = (pipeline.state.get("speakers") or {}).get(sid, {}).get("gender", "")
                if gender in ("male", "female"):
                    lab = f"{lab} ({gender[0].upper()})"
                combo.addItem(lab, sid)
            # allow extra slots if user needs to create distinction
            for extra in ("speaker_1", "speaker_2", "speaker_3", "speaker_4"):
                if extra not in spk_ids:
                    combo.addItem(extra.replace("speaker_", "S"), extra)
            ix = combo.findData(s["speaker_id"])
            combo.setCurrentIndex(max(0, ix))
            self.table.setCellWidget(r, 1, combo)
            o_item = QTableWidgetItem(s["text"])
            o_item.setFlags(o_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(r, 2, o_item)
            self.table.setItem(r, 3, QTableWidgetItem(s.get("translation", "")))
            long_item = QTableWidgetItem("⚠" if s.get("too_long") else "")
            long_item.setFlags(long_item.flags() & ~Qt.ItemIsEditable)
            if s.get("too_long"):
                long_item.setForeground(QColor("#ffb454"))
            self.table.setItem(r, 4, long_item)
        lay.addWidget(self.table, 1)
        bb = QDialogButtonBox()
        bb.addButton("Continue dubbing", QDialogButtonBox.AcceptRole).setObjectName("primary")
        bb.accepted.connect(self.accept)
        lay.addWidget(bb)

    def apply(self):
        from PySide6.QtWidgets import QComboBox
        by_id = {s["id"]: s for s in self.pl.state["segments"]}
        used_spk = set()
        for r in range(self.table.rowCount()):
            sid = self.table.item(r, 0).data(Qt.UserRole)
            by_id[sid]["translation"] = self.table.item(r, 3).text().strip()
            w = self.table.cellWidget(r, 1)
            if isinstance(w, QComboBox) and w.currentData():
                by_id[sid]["speaker_id"] = w.currentData()
                used_spk.add(w.currentData())
        # Ensure speaker meta exists for any newly referenced ids
        speakers = dict(self.voice_panel.speakers)
        for spk in used_spk:
            if spk not in speakers:
                speakers[spk] = {
                    "gender": "unknown", "gender_confidence": 0.0, "f0": None,
                    "speech_seconds": 0.0, "segments": 0,
                    "label": "Speaker " + spk.split("_")[-1],
                }
        # refresh speech seconds
        for spk in list(speakers.keys()):
            secs = sum(s["end"] - s["start"] for s in self.pl.state["segments"] if s.get("speaker_id") == spk)
            speakers[spk]["speech_seconds"] = round(secs, 1)
            speakers[spk]["segments"] = sum(1 for s in self.pl.state["segments"] if s.get("speaker_id") == spk)
        # keep voice panel genders
        for spk, info in self.voice_panel.speakers.items():
            if spk in speakers:
                speakers[spk]["gender"] = info.get("gender", speakers[spk].get("gender"))
                speakers[spk]["gender_source"] = info.get("gender_source")
                speakers[spk]["gender_confidence"] = info.get("gender_confidence", 1.0)
        self.pl.state["speakers"] = speakers
        self.pl.state["assignments"] = self.voice_panel.assignments
        self.pl.save()
