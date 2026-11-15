"""WAV helpers. All intermediates are PCM16 so they can be read block-wise (low RAM)."""
from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


def write_wav(path: str | Path, data: np.ndarray, sr: int) -> None:
    x = np.asarray(data, dtype=np.float32)
    x = np.clip(x, -1.0, 1.0)
    wavfile.write(str(path), int(sr), (x * 32767.0).astype(np.int16))


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    sr, x = wavfile.read(str(path))
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    elif x.dtype == np.int32:
        x = x.astype(np.float32) / 2147483648.0
    elif x.dtype == np.uint8:
        x = (x.astype(np.float32) - 128.0) / 128.0
    else:
        x = x.astype(np.float32)
    return x, int(sr)


def to_mono(x: np.ndarray) -> np.ndarray:
    return x if x.ndim == 1 else x.mean(axis=1)


def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return x
    g = math.gcd(int(sr_from), int(sr_to))
    return resample_poly(x, sr_to // g, sr_from // g, axis=0).astype(np.float32)


def wav_info(path: str | Path) -> tuple[int, int, int]:
    """(sample_rate, channels, frames) for PCM WAV."""
    with wave.open(str(path), "rb") as w:
        return w.getframerate(), w.getnchannels(), w.getnframes()


def wav_duration(path: str | Path) -> float:
    sr, _, n = wav_info(path)
    return n / sr


def read_wav_block(path: str | Path, start_s: float, dur_s: float) -> tuple[np.ndarray, int]:
    """Read [start, start+dur) from a PCM16 WAV without loading the whole file. Returns (n, ch) float32."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError("read_wav_block supports PCM16 only")
        sr, ch = w.getframerate(), w.getnchannels()
        w.setpos(min(max(0, int(start_s * sr)), w.getnframes()))
        raw = w.readframes(int(dur_s * sr))
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return x.reshape(-1, ch), sr


class WavWriter:
    """Incremental PCM16 WAV writer."""

    def __init__(self, path: str | Path, sr: int, channels: int = 1):
        self._w = wave.open(str(path), "wb")
        self._w.setnchannels(channels)
        self._w.setsampwidth(2)
        self._w.setframerate(int(sr))
        self.channels = channels

    def write(self, x: np.ndarray) -> None:
        x = np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0)
        self._w.writeframes((x * 32767.0).astype(np.int16).tobytes())

    def close(self) -> None:
        self._w.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
