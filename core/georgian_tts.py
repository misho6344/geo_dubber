"""Georgian text-to-speech engines.

VERIFIED BY RESEARCH (not yet run in this build environment, which blocks model downloads):
  * f5_georgian - community Georgian fine-tune of F5-TTS v1 Base (HF: tabluka90/georgian-f5-tts).
                  Reference-conditioned / zero-shot: one reference clip per voice -> many voices.
  * piper_ka    - Piper 'ka_GE' voice (single female voice 'natia'). Needs the piper binary + .onnx.
  * mms_kat     - Meta MMS-TTS 'facebook/mms-tts-kat' (single voice, CC-BY-NC).
NOT supported: Coqui XTTS-v2 (its official language list has no Georgian).

Run scripts/verify_georgian_tts.py on your PC BEFORE trusting any engine.
Engine licences differ (several are non-commercial) - check them before commercial use.
"""
from __future__ import annotations

import hashlib
import logging
import os
import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

from config.settings import Settings, data_dir, sub_dir
from core.audio_io import read_wav, resample, to_mono, write_wav
from core.errors import MissingDependencyError, ModelMissingError, VoiceUnavailableError
from core.voices import Voice

log = logging.getLogger(__name__)

F5_REPO = "tabluka90/georgian-f5-tts"
F5_CKPT = "georgian_f5_34520.safetensors"
OUT_SR = 24000


class TTSBackend(ABC):
    name = "base"
    needs_reference = False

    @abstractmethod
    def synthesize(self, text: str, voice: Voice) -> tuple[np.ndarray, int]:
        """Return (mono float32 samples, sample_rate)."""

    def builtin_voices(self) -> list[Voice]:
        return []

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
class F5GeorgianBackend(TTSBackend):
    name = "f5_georgian"
    needs_reference = True

    def __init__(self, device: str = "cpu", nfe_step: int = 16, model_dir: Optional[Path] = None):
        self.device = device
        self.nfe_step = nfe_step
        self.model_dir = Path(model_dir) if model_dir else sub_dir("models") / "georgian-f5-tts"
        self._model = None

    def _ensure_files(self) -> tuple[Path, Path]:
        ckpt, vocab = self.model_dir / F5_CKPT, self.model_dir / "vocab.txt"
        if not (ckpt.exists() and vocab.exists()):
            raise ModelMissingError(
                f"missing {ckpt} / {vocab}",
                f"The Georgian F5-TTS model is not installed. Open Settings > Models and download '{F5_REPO}'.")
        return ckpt, vocab

    def _load(self):
        if self._model is not None:
            return self._model
        ckpt, vocab = self._ensure_files()
        try:
            from f5_tts.api import F5TTS
        except ImportError as e:
            raise MissingDependencyError(str(e), "The 'f5-tts' package is not installed (run install.bat).")
        log.info("Loading Georgian F5-TTS on %s", self.device)
        self._model = F5TTS(model="F5TTS_v1_Base", ckpt_file=str(ckpt), vocab_file=str(vocab), device=self.device)
        return self._model

    def synthesize(self, text: str, voice: Voice) -> tuple[np.ndarray, int]:
        if not voice.ref_audio or not Path(voice.ref_audio).exists():
            raise VoiceUnavailableError("no reference", f"Voice '{voice.name}' has no reference recording.")
        m = self._load()
        seed = int(hashlib.sha1(voice.id.encode()).hexdigest()[:8], 16) % (2 ** 31)  # stable per voice
        wav, sr, _ = m.infer(ref_file=voice.ref_audio, ref_text=voice.ref_text, gen_text=text,
                             nfe_step=self.nfe_step, remove_silence=False, seed=seed)
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        return wav, int(sr)

    def close(self):
        self._model = None
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


# ---------------------------------------------------------------------------
class PiperGeorgianBackend(TTSBackend):
    """Piper 'ka_GE' voice via the piper executable. One fixed (female) voice."""
    name = "piper_ka"

    def __init__(self, model_path: Optional[Path] = None, exe: Optional[str] = None):
        self.model_path = Path(model_path) if model_path else sub_dir("models") / "piper" / "ka_GE-natia-medium.onnx"
        self.exe = exe or os.environ.get("GEODUBBER_PIPER") or "piper"

    def builtin_voices(self) -> list[Voice]:
        return [Voice(id="piper_ka_natia", name="Piper Georgian (Natia)", gender="female", engine=self.name,
                      builtin=True, consent=True, note="Piper ka_GE; check model card licence")]

    def synthesize(self, text: str, voice: Voice) -> tuple[np.ndarray, int]:
        if not self.model_path.exists():
            raise ModelMissingError(str(self.model_path), "The Piper Georgian voice file (.onnx) is not installed.")
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "o.wav"
            try:
                p = subprocess.run([self.exe, "--model", str(self.model_path), "--output_file", str(out)],
                                   input=text, text=True, encoding="utf-8", capture_output=True)
            except FileNotFoundError:
                raise MissingDependencyError("piper not found", "The 'piper' executable was not found.")
            if p.returncode != 0 or not out.exists():
                raise VoiceUnavailableError(p.stderr[-500:], "Piper failed to synthesize speech.")
            x, sr = read_wav(out)
        return to_mono(x), sr


# ---------------------------------------------------------------------------
class MMSGeorgianBackend(TTSBackend):
    """facebook/mms-tts-kat via transformers (single voice, CC-BY-NC)."""
    name = "mms_kat"

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._m = self._t = None

    def builtin_voices(self) -> list[Voice]:
        return [Voice(id="mms_kat_default", name="MMS Georgian", gender="female", engine=self.name,
                      builtin=True, consent=True, note="Meta MMS-TTS, CC-BY-NC 4.0; gender not documented - verify by ear")]

    def _load(self):
        if self._m is None:
            try:
                import torch
                from transformers import AutoTokenizer, VitsModel
            except ImportError as e:
                raise MissingDependencyError(str(e), "transformers/torch are not installed (run install.bat).")
            cache = str(sub_dir("models") / "hf")
            self._t = AutoTokenizer.from_pretrained("facebook/mms-tts-kat", cache_dir=cache)
            self._m = VitsModel.from_pretrained("facebook/mms-tts-kat", cache_dir=cache).to(self.device).eval()
        return self._m, self._t

    def synthesize(self, text: str, voice: Voice) -> tuple[np.ndarray, int]:
        import torch
        m, t = self._load()
        ids = t(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            wav = m(**ids).waveform[0].cpu().numpy()
        return wav.astype(np.float32), int(m.config.sampling_rate)


# ---------------------------------------------------------------------------
class EdgeGeorgianBackend(TTSBackend):
    """Microsoft Edge Neural TTS Georgian voices (ka-GE-GiorgiNeural = male, ka-GE-EkaNeural = female).
    Ultra-fast (1-2s per line), zero model weights on disk, studio-quality neural voices.

    Edge has only ONE male and ONE female Georgian voice. With `variants=True` each gender gets two extra
    versions that differ slightly in pitch and speed, so a second or third character of the same gender sounds
    different instead of sharing the same voice. The variants are untested by ear: listen to them in Review and
    switch them off in Settings if they sound wrong.
    """
    name = "edge_georgian"
    needs_reference = False

    # (suffix, label, pitch, rate). Base voice first: assignment prefers it.
    MALE_VARIANTS = [("", "Giorgi", "+0Hz", "+0%"),
                     ("_deep", "Giorgi (deeper)", "-10Hz", "-3%"),
                     ("_bright", "Giorgi (brighter)", "+8Hz", "+2%")]
    FEMALE_VARIANTS = [("", "Nino", "+0Hz", "+0%"),
                       ("_low", "Nino (lower)", "-8Hz", "-2%"),
                       ("_high", "Nino (higher)", "+8Hz", "+2%")]

    def __init__(self, variants: bool = True):
        self.variants = bool(variants)

    def builtin_voices(self) -> list[Voice]:
        out = []
        groups = [("giorgi", "male", "ka-GE-GiorgiNeural", self.MALE_VARIANTS),
                  ("eka", "female", "ka-GE-EkaNeural", self.FEMALE_VARIANTS)]
        for base, gender, tag, variants in groups:
            chosen = variants if self.variants else variants[:1]
            for suffix, label, pitch, rate in chosen:
                out.append(Voice(
                    id=f"edge_ka_{base}{suffix}", name=label, gender=gender, engine=self.name,
                    builtin=True, consent=True,
                    note=f"Microsoft {tag}" + ("" if not suffix else f" (pitch {pitch}, rate {rate})"),
                    builtin_params={"voice": tag, "pitch": pitch, "rate": rate}))
        return out

    def synthesize(self, text: str, voice: Voice) -> tuple[np.ndarray, int]:
        import asyncio
        import io
        try:
            import edge_tts
            import soundfile as sf
        except ImportError:
            raise MissingDependencyError("edge-tts missing", "The 'edge-tts' package is required. Run install.bat.")

        params = voice.builtin_params or {}
        voice_tag = params.get("voice") or ("ka-GE-EkaNeural" if voice.gender == "female" else "ka-GE-GiorgiNeural")
        pitch = params.get("pitch", "+0Hz")
        rate = params.get("rate", "+0%")

        async def _run():
            c = edge_tts.Communicate(text, voice_tag, pitch=pitch, rate=rate)
            data = b""
            async for chunk in c.stream():
                if chunk["type"] == "audio":
                    data += chunk["data"]
            if not data:
                raise VoiceUnavailableError("empty audio", "Edge TTS returned empty audio data.")
            wav, sr = sf.read(io.BytesIO(data), dtype="float32")
            if wav.ndim > 1:
                wav = np.mean(wav, axis=1)
            return np.asarray(wav, dtype=np.float32), int(sr)

        # A fresh loop per call so several lines can be spoken at once from worker threads.
        return asyncio.run(_run())


# ---------------------------------------------------------------------------
ENGINES = {
    "edge_georgian": EdgeGeorgianBackend,
    "f5_georgian": F5GeorgianBackend,
    "piper_ka": PiperGeorgianBackend,
    "mms_kat": MMSGeorgianBackend,
}


def create_backend(name: str, settings: Settings, device: str) -> TTSBackend:
    if name not in ENGINES:
        raise ValueError(f"Unknown TTS engine '{name}'")
    if name == "edge_georgian":
        return EdgeGeorgianBackend(variants=getattr(settings, "edge_character_variants", True))
    if name == "f5_georgian":
        return F5GeorgianBackend(device=device, nfe_step=settings.preset()["nfe_step"])
    if name == "mms_kat":
        return MMSGeorgianBackend(device=device)
    return PiperGeorgianBackend()


def synthesize_cached(backend: TTSBackend, voice: Voice, text: str, cache_root: Optional[Path] = None) -> Path:
    """Generate speech for `text` (or reuse the cached file). Stored as mono 24 kHz PCM16."""
    text = text.strip()
    if not text:
        raise ValueError("empty text")
    key = hashlib.sha1(f"{backend.name}|{voice.fingerprint()}|{text}".encode("utf-8")).hexdigest()
    base = (cache_root or (sub_dir("cache") / "tts")) / backend.name
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{key}.wav"
    if path.exists() and path.stat().st_size > 44:
        return path
    wav, sr = backend.synthesize(text, voice)
    if wav.size == 0 or not np.isfinite(wav).all():
        raise VoiceUnavailableError("empty/invalid audio", "The TTS engine returned invalid audio.")
    peak = float(np.max(np.abs(wav)))
    if peak > 0:
        wav = wav * min(1.0, 0.95 / peak)
    write_wav(path, resample(wav, sr, OUT_SR), OUT_SR)
    return path
