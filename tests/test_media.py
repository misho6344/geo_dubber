import json
import subprocess

import numpy as np
import pytest

from core import audio_extractor as AX
from core import video_loader as VL
from core.audio_io import read_wav, read_wav_block, wav_duration, write_wav
from core.audio_mixer import build_filter, mix_audio
from core.errors import CorruptMediaError, UnsupportedFormatError
from core.ffmpeg_utils import probe_json
from core.subtitles import fmt_ts, write_srt
from core.video_exporter import export_video


def test_probe_and_validate(sample_video):
    info = VL.probe(sample_video)
    assert (info.width, info.height) == (320, 240)
    assert 19.5 < info.duration < 20.5 and info.has_audio and abs(info.fps - 25) < 0.1
    VL.validate_for_dubbing(info)


def test_unsupported_and_corrupt(tmp_path):
    bad = tmp_path / "x.txt"
    bad.write_text("hi")
    with pytest.raises(UnsupportedFormatError):
        VL.probe(bad)
    corrupt = tmp_path / "c.mp4"
    corrupt.write_bytes(b"not a video at all" * 100)
    with pytest.raises(CorruptMediaError):
        VL.probe(corrupt)


def test_youtube_url_check():
    assert VL.is_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    assert VL.is_youtube_url("https://youtu.be/dQw4w9WgXcQ")
    assert not VL.is_youtube_url("https://example.com/video.mp4")


def test_extract_and_chunks(sample_video, tmp_path):
    info = VL.probe(sample_video)
    prog = []
    wav = AX.extract_full_audio(sample_video, tmp_path / "a.wav", info, on_progress=prog.append)
    assert abs(wav_duration(wav) - info.duration) < 0.2
    m16 = AX.extract_mono16k(wav, tmp_path / "m.wav")
    x, sr = read_wav(m16)
    assert sr == 16000 and x.ndim == 1
    blk, bsr = read_wav_block(wav, 5.0, 2.0)
    assert abs(len(blk) / bsr - 2.0) < 0.01
    c = AX.cut_wav(wav, 3.0, 6.0, tmp_path / "c.wav")
    assert abs(wav_duration(c) - 3.0) < 0.05


def test_plan_windows_and_midpoint():
    w = AX.plan_windows(100.0, 30.0, 5.0)
    assert [(a["start"], a["end"]) for a in w] == [(0, 30), (30, 60), (60, 90), (90, 100)]
    assert w[1]["win_start"] == 25 and w[1]["win_end"] == 65 and w[0]["win_start"] == 0
    segs = [{"start": 28, "end": 33}, {"start": 25, "end": 28}, {"start": 60, "end": 62}]
    kept0 = AX.keep_by_midpoint(segs, 0, 30)
    kept1 = AX.keep_by_midpoint(segs, 30, 60)
    kept2 = AX.keep_by_midpoint(segs, 60, 90)
    all_kept = kept0 + kept1 + kept2
    assert len(all_kept) == 3 and len({id(s) for s in all_kept}) == 3     # each exactly once


def test_srt(tmp_path):
    assert fmt_ts(3661.5) == "01:01:01,500"
    p = write_srt([{"start": 1, "end": 2, "text": "გამარჯობა"}, {"start": 3, "end": 3.1, "text": ""}], tmp_path / "a.srt")
    t = p.read_text(encoding="utf-8")
    assert "გამარჯობა" in t and t.count("-->") == 1


def test_mix_and_export(sample_video, tmp_path):
    info = VL.probe(sample_video)
    sr = 24000
    t = np.arange(int(sr * 20)) / sr
    dub = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    dub_wav = tmp_path / "dub.wav"
    write_wav(dub_wav, dub, sr)
    orig = AX.extract_full_audio(sample_video, tmp_path / "o.wav", info)
    mixed = mix_audio(dub_wav, tmp_path / "mix.m4a", background=orig, original=orig, orig_vol=0.1, duck=True,
                      duration=20)
    ja = probe_json(mixed)
    assert ja["streams"][0]["codec_name"] == "aac" and ja["streams"][0]["channels"] == 2
    srt = write_srt([{"start": 1, "end": 3, "text": "ტესტი"}], tmp_path / "s.srt")
    out = export_video(sample_video, mixed, tmp_path / "final.mp4", info, srt=srt)
    j = probe_json(out)
    kinds = [s["codec_type"] for s in j["streams"]]
    assert kinds.count("video") == 1 and kinds.count("audio") == 1 and kinds.count("subtitle") == 1
    v = next(s for s in j["streams"] if s["codec_type"] == "video")
    assert (v["width"], v["height"]) == (320, 240) and v["codec_name"] == "h264"
    assert abs(float(j["format"]["duration"]) - 20) < 0.6
    assert not (tmp_path / "final.partial.mp4").exists()


def test_mix_loudness_not_clipping(tmp_path, sample_video):
    info = VL.probe(sample_video)
    sr = 24000
    loud = np.ones(sr * 5, dtype=np.float32) * 0.99
    write_wav(tmp_path / "l.wav", loud, sr)
    mix_audio(tmp_path / "l.wav", tmp_path / "m.m4a", duck=False, duration=5)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp_path / "m.m4a"), str(tmp_path / "m.wav")], check=True)
    x, _ = read_wav(tmp_path / "m.wav")
    assert np.max(np.abs(x)) <= 1.0


def test_filter_variants():
    for bg in (False, True):
        for orig in (False, True):
            for duck in (False, True):
                f = build_filter(bg, orig, 1, 1, 0.2, duck)
                assert "[out]" in f and "amix" in f
