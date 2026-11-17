"""Settings window: processing, audio, folders, models (download manager) and voices."""
from __future__ import annotations

import shutil

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout,
                               QLabel, QLineEdit, QListWidget, QMessageBox, QPushButton, QSpinBox, QTabWidget, QVBoxLayout, QWidget)

from config.settings import Settings, get_gemini_api_key, sub_dir
from core.georgian_tts import F5_REPO
from core.voices import delete_voice, load_user_voices
from gui.voice_manager import AddVoiceDialog

# (label, hf repo or None, approx size note). Sizes are approximate / not verified here.
MODELS = [
    ("Georgian F5-TTS (voice generation)", F5_REPO, "size not verified - shown by Hugging Face while downloading"),
    ("Whisper small (fast speech recognition)", "Systran/faster-whisper-small", "≈0.5 GB"),
    ("Whisper medium", "Systran/faster-whisper-medium", "≈1.5 GB"),
    ("Whisper large-v3 (best quality)", "Systran/faster-whisper-large-v3", "≈3 GB"),
    ("NLLB-200 600M (translation to Georgian)", "facebook/nllb-200-distilled-600M", "≈2.5 GB"),
]


class DownloadThread(QThread):
    done = Signal(str, str)

    def __init__(self, repo: str, token: str):
        super().__init__()
        self.repo, self.token = repo, token

    def run(self):
        try:
            from huggingface_hub import snapshot_download
            name = self.repo.split("/")[-1]
            target = sub_dir("models") / name
            snapshot_download(self.repo, local_dir=str(target), token=self.token or None)
            self.done.emit(self.repo, "")
        except Exception as e:
            self.done.emit(self.repo, str(e))


class SettingsDialog(QDialog):
    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self.s = settings
        self.setWindowTitle("Settings")
        self.resize(640, 560)
        tabs = QTabWidget()
        lay = QVBoxLayout(self)
        lay.addWidget(tabs)

        g = QWidget(),
        # ---- processing
        w = QWidget(); f = QFormLayout(w)
        self.priority = QComboBox(); self.priority.addItems(["fast", "balanced", "quality"]); self.priority.setCurrentText(settings.priority)
        self.asr = QComboBox(); self.asr.addItems(["(from priority)", "tiny", "base", "small", "medium", "large-v3", "distil-large-v3"])
        self.asr.setCurrentText(settings.asr_model or "(from priority)")
        self.tr = QLineEdit(settings.translation_model)
        self.tts = QComboBox(); self.tts.addItems(["edge_georgian", "f5_georgian", "piper_ka", "mms_kat"]); self.tts.setCurrentText(settings.tts_engine)
        self.device = QComboBox(); self.device.addItems(["auto", "cuda", "cpu"]); self.device.setCurrentText(settings.device)
        self.chunk = QDoubleSpinBox(); self.chunk.setRange(0.5, 60); self.chunk.setValue(settings.chunk_minutes); self.chunk.setSuffix(" min")
        self.tbatch = QSpinBox(); self.tbatch.setRange(1, 128); self.tbatch.setValue(settings.translation_batch)
        self.lang = QComboBox(); self.lang.addItems(["English"]);
        # Gemini credentials are environment-only; never expose or persist the secret in the UI.
        key_configured = bool(get_gemini_api_key())
        self.gemini_hint = QLabel(
            "GEMINI_API_KEY is configured in this environment."
            if key_configured else
            "GEMINI_API_KEY is not configured. Add it to the server Environment Variables; do not paste it here or store it in settings.json."
        )
        self.gemini_hint.setWordWrap(True)
        self.token = QLineEdit(settings.hf_token); self.token.setEchoMode(QLineEdit.Password)
        self.token.setPlaceholderText("Hugging Face token (only for gated speaker model)")
        for lab, wd in (("Processing priority", self.priority), ("Speech recognition model", self.asr), ("Translation model", self.tr),
                        ("TTS engine", self.tts), ("GPU acceleration", self.device), ("Chunk length", self.chunk),
                        ("Translation batch size", self.tbatch), ("Interface language", self.lang),
                        ("Gemini API", self.gemini_hint), ("HF token", self.token)):
            f.addRow(lab, wd)
        self.share = QCheckBox("Allow two characters to share a voice when not enough voices exist"); self.share.setChecked(settings.allow_voice_sharing)
        f.addRow(self.share)
        tabs.addTab(w, "Processing")

        # ---- audio / output
        w = QWidget(); f = QFormLayout(w)
        def vol(v):
            d = QDoubleSpinBox(); d.setRange(0, 2); d.setSingleStep(0.05); d.setValue(v); return d
        self.vv, self.bv, self.ov = vol(settings.voice_volume), vol(settings.background_volume), vol(settings.original_dialogue_volume)
        self.duck = QCheckBox("Lower background while the Georgian voice speaks"); self.duck.setChecked(settings.duck_background)
        self.crf = QSpinBox(); self.crf.setRange(14, 30); self.crf.setValue(settings.video_crf)
        self.burn = QCheckBox("Burn subtitles into the video (re-encodes; slower)"); self.burn.setChecked(settings.burn_subtitles)
        self.abr = QComboBox(); self.abr.addItems(["128k", "160k", "192k", "256k", "320k"]); self.abr.setCurrentText(settings.audio_bitrate)
        self.out = QLineEdit(settings.output_dir); self.cache = QLineEdit(settings.cache_dir); self.tmp = QLineEdit(settings.temp_dir)
        for lab, wd in (("Georgian voice volume", self.vv), ("Background volume", self.bv), ("Original dialogue volume", self.ov),
                        ("", self.duck), ("Video quality (CRF, lower = better; used when re-encoding)", self.crf), ("", self.burn),
                        ("Audio bitrate", self.abr), ("Output folder", self.out), ("Cache folder", self.cache), ("Temp folder", self.tmp)):
            f.addRow(lab, wd)
        b = QPushButton("Delete cache and temporary files"); b.clicked.connect(self._clear)
        f.addRow(b)
        tabs.addTab(w, "Audio & folders")

        # ---- models
        w = QWidget(); v = QVBoxLayout(w)
        v.addWidget(QLabel("Models are downloaded from their official Hugging Face repositories and stored in the 'models' folder.\n"
                           "Check each model's licence (several are non-commercial)."))
        self.model_rows = {}
        for lab, repo, size in MODELS:
            h = QHBoxLayout(); l = QLabel(f"{lab}\n{size}"); btn = QPushButton("Download")
            btn.clicked.connect(lambda _c=False, r=repo, bt=btn: self._download(r, bt))
            h.addWidget(l, 1); h.addWidget(btn); v.addLayout(h)
        v.addStretch(1)
        tabs.addTab(w, "Models")

        # ---- voices
        w = QWidget(); v = QVBoxLayout(w)
        v.addWidget(QLabel("Reference voices for the Georgian F5-TTS engine. Use only recordings you are allowed to use."))
        self.vlist = QListWidget(); v.addWidget(self.vlist)
        h = QHBoxLayout(); a = QPushButton("＋ Add voice…"); d = QPushButton("Delete selected")
        a.clicked.connect(self._add_voice); d.clicked.connect(self._del_voice); h.addWidget(a); h.addWidget(d); v.addLayout(h)
        tabs.addTab(w, "Voices"); self._reload_voices()

        bb = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._save); bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self._threads = []

    def _reload_voices(self):
        self.vlist.clear()
        for v in load_user_voices():
            self.vlist.addItem(f"{v.id} — {v.name} [{v.gender}]")

    def _add_voice(self):
        if AddVoiceDialog(self).exec():
            self._reload_voices()

    def _del_voice(self):
        it = self.vlist.currentItem()
        if it and QMessageBox.question(self, "Delete voice", f"Delete {it.text()}?") == QMessageBox.Yes:
            delete_voice(it.text().split(" — ")[0]); self._reload_voices()

    def _download(self, repo, btn):
        btn.setEnabled(False); btn.setText("Downloading…")
        t = DownloadThread(repo, self.token.text()); self._threads.append(t)
        def done(r, err):
            btn.setEnabled(True)
            if err:
                btn.setText("Retry"); QMessageBox.warning(self, "Download failed", f"{r}\n\n{err}")
            else:
                btn.setText("Downloaded ✓")
        t.done.connect(done); t.start()

    def _clear(self):
        if QMessageBox.question(self, "Delete", "Delete all cached audio and temporary files? (Projects are kept.)") == QMessageBox.Yes:
            shutil.rmtree(self.s.resolved_cache_dir(), ignore_errors=True)
            self.s.resolved_cache_dir()

    def _save(self):
        s = self.s
        s.priority = self.priority.currentText(); s.asr_model = "" if self.asr.currentIndex() == 0 else self.asr.currentText()
        s.translation_model = self.tr.text().strip(); s.tts_engine = self.tts.currentText(); s.device = self.device.currentText()
        s.chunk_minutes = self.chunk.value(); s.translation_batch = self.tbatch.value(); s.hf_token = self.token.text().strip()
        s.allow_voice_sharing = self.share.isChecked(); s.voice_volume, s.background_volume = self.vv.value(), self.bv.value()
        s.original_dialogue_volume = self.ov.value(); s.duck_background = self.duck.isChecked(); s.video_crf = self.crf.value()
        s.burn_subtitles = self.burn.isChecked(); s.audio_bitrate = self.abr.currentText()
        s.output_dir, s.cache_dir, s.temp_dir = self.out.text().strip(), self.cache.text().strip(), self.tmp.text().strip()
        s.save(); self.accept()
