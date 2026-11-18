"""End-to-end pipeline tests on REAL media/FFmpeg/audio code with scripted STAND-IN model backends.
The stand-ins (Stub*) replace only the AI models (ASR, diarization, translation, TTS) - they prove the pipeline
logic, not model quality."""
import json
import subprocess
import threading
import time

import numpy as np
import pytest

from config.settings import Settings
from core import video_loader as VL
from core.audio_io import read_wav, write_wav
from core.background_separator import FallbackSeparator
from core.errors import CancelledError, GeoDubberError, VoiceUnavailableError
from core.ffmpeg_utils import Control, probe_json
from core.georgian_tts import TTSBackend
from core.processing_manager import Backends, Pipeline
from core.speaker_diarization import Diarizer, SingleSpeakerDiarizer
from core.speech_recognition import Recognizer
from core.translation import Translator
from core.voices import Voice

SCRIPT = [  # (start, end, text, speaker f0)
    (1.0, 4.0, "Where are you going?", 110), (6.0, 9.0, "To the market, I will be back soon.", 230),
    (11.0, 14.0, "Be careful out there.", 110), (35.0, 41.0, "A line crossing the chunk boundary at thirty seconds.", 230),
    (50.0, 53.0, "Goodbye my friend.", 110)]


def voiced(f0, sr, dur):
    t = np.arange(int(sr * dur)) / sr
    x = sum((1 / h) * np.sin(2 * np.pi * f0 * h * t) for h in range(1, 8))
    return 0.3 * x / np.max(np.abs(x))


@pytest.fixture(scope="module")
def dialog_video(tmp_path_factory):
    d = tmp_path_factory.mktemp("dlg")
    sr, total = 48000, 60
    a = np.random.default_rng(1).normal(0, 0.01, sr * total).astype(np.float32)       # faint 'music' noise
    for s, e, _, f0 in SCRIPT:
        a[int(s * sr):int(e * sr)] += voiced(f0, sr, e - s).astype(np.float32)
    write_wav(d / "a.wav", a, sr)
    out = d / "movie.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"testsrc=size=320x240:rate=10:duration={total}",
                    "-i", str(d / "a.wav"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out)], check=True)
    return out


class StubRecognizer(Recognizer):
    language = "en"

    def transcribe(self, wav16k, offset):
        from core.audio_io import wav_duration
        end = offset + wav_duration(wav16k)
        return [{"start": s, "end": e, "text": t} for s, e, t, _ in SCRIPT if e > offset and s < end and offset <= (s + e) / 2 < end + 0.001 or (s < end and e > offset)]


class StubDiarizer(Diarizer):
    def diarize(self, wav16k):
        return [(s, e, "A" if f == 110 else "B") for s, e, _, f in SCRIPT]


class StubTranslator(Translator):
    def translate(self, texts, src_lang, **kwargs):
        self.last_sources = ["gemini"] * len(texts)
        return ["გა:" + t for t in texts]


class StubTTS(TTSBackend):
    """Test-only: tone whose frequency identifies the voice, length ~ text length. NOT speech."""
    name = "stub"
    FREQ = {"m1": 700, "m2": 800, "f1": 900, "f2": 1000}

    def __init__(self, voices=("m1", "f1", "m2", "f2")):
        self.calls, self.voice_ids, self.cancel_after, self.control = [], voices, None, None

    def builtin_voices(self):
        return [Voice(id=i, name=i, gender="male" if i.startswith("m") else "female", engine="stub", builtin=True, consent=True)
                for i in self.voice_ids]

    def synthesize(self, text, voice):
        self.calls.append((voice.id, text))
        if self.cancel_after and len(self.calls) >= self.cancel_after:
            self.control.cancel()
        sr = 24000
        dur = 0.4 + 0.05 * len(text)
        t = np.arange(int(sr * dur)) / sr
        return (0.3 * np.sin(2 * np.pi * self.FREQ[voice.id] * t)).astype(np.float32), sr


def make(tmp_path, tts=None, diarizer=None, **kw):
    s = Settings(chunk_minutes=0.5, chunk_overlap_seconds=3.0, output_dir=str(tmp_path / "out"), **kw)
    tts = tts or StubTTS()
    be = Backends(StubRecognizer(), diarizer or StubDiarizer(), StubTranslator(), tts, FallbackSeparator())
    return s, be


def pipe(tmp_path, s, be, **kw):
    ctl = Control()
    tts_ctl = getattr(be.tts, "control", None)
    be.tts.control = ctl
    return Pipeline(tmp_path / "proj", s, be, control=ctl, cache_root=tmp_path / "ttscache", extra_voices=[], **kw), ctl


def dominant_freq(x, sr):
    sp = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return np.fft.rfftfreq(len(x), 1 / sr)[np.argmax(sp)]


def test_full_pipeline(tmp_path, dialog_video):
    s, be = make(tmp_path)
    events = []
    p, _ = pipe(tmp_path, s, be, on_progress=events.append)
    outs = p.run(dialog_video)
    st = p.state
    # chunking: 60 s with 30 s chunks -> 2 chunks, the boundary-crossing line appears exactly once
    assert len(st["asr_chunks"]) == 2
    texts = [x["text"] for x in st["segments"]]
    assert texts.count("A line crossing the chunk boundary at thirty seconds.") == 1 and len(texts) == 5
    # speakers + gender estimated from the REAL audio, voices consistent
    sp = st["speakers"]
    assert sp["speaker_1"]["gender"] == "male" and sp["speaker_2"]["gender"] == "female"
    a = st["assignments"]
    assert a == {"speaker_1": "m1", "speaker_2": "f1"}
    assert [x["speaker_id"] for x in st["segments"]] == ["speaker_1", "speaker_2", "speaker_1", "speaker_2", "speaker_1"]
    # TTS lines run in parallel threads, so compare voice per line, not the order the calls finished in.
    spk_of = {x["translation"]: x["speaker_id"] for x in st["segments"]}
    voice_of = {text: vid for vid, text in be.tts.calls}
    assert all(voice_of[t] == a[spk] for t, spk in spk_of.items()) and len(be.tts.calls) == 5
    # outputs
    j = probe_json(outs["video"])
    kinds = [x["codec_type"] for x in j["streams"]]
    assert kinds.count("video") == 1 and kinds.count("audio") == 1 and kinds.count("subtitle") == 1
    assert abs(float(j["format"]["duration"]) - 60) < 1
    assert "გა:Where are you going?" in open(outs["georgian_srt"], encoding="utf-8").read()
    rep = json.load(open(outs["report"], encoding="utf-8"))
    assert rep["assignments"] == a and any("separation" in w.lower() for w in rep["warnings"])
    # progress is real and monotone, ends at 100 %
    overall = [e.overall for e in events]
    assert overall == sorted(overall) and overall[-1] == 1.0
    assert all(e.eta_seconds is None or e.eta_seconds >= 0 for e in events)
    # the final soundtrack really contains the Georgian voices at the right times (m1=700 Hz @ ~2 s, f1=900 Hz @ ~7 s)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", outs["video"], "-vn", "-ac", "1", "-ar", "16000", str(tmp_path / "f.wav")], check=True)
    x, sr = read_wav(tmp_path / "f.wav")
    f1 = dominant_freq(x[int(1.6 * sr):int(2.4 * sr)], sr)
    f2 = dominant_freq(x[int(6.6 * sr):int(7.4 * sr)], sr)
    assert abs(f1 - 700) < 25 and abs(f2 - 900) < 25
    # background kept (fallback: original track quietly present in a silent stretch where only 'music' noise exists)
    assert np.abs(x[int(20 * sr):int(25 * sr)]).mean() > 1e-4


def test_imported_subtitles_are_used_as_is_without_asr_or_translation(tmp_path, dialog_video):
    srt = tmp_path / "georgian.srt"
    expected = [
        "Where are you going?",
        "To the market, I will be back soon.",
        "Be careful out there.",
        "A line crossing the chunk boundary at thirty seconds.",
        "Goodbye my friend.",
    ]
    blocks = []
    for index, (start, end, text) in enumerate(zip(
        ("00:00:01,000", "00:00:06,000", "00:00:11,000", "00:00:35,000", "00:00:50,000"),
        ("00:00:04,000", "00:00:09,000", "00:00:14,000", "00:00:41,000", "00:00:53,000"),
        expected,
    ), start=1):
        blocks.extend([str(index), f"{start} --> {end}", text, ""])
    srt.write_text("\n".join(blocks), encoding="utf-8")

    s, be = make(tmp_path, imported_subtitles=str(srt))
    be.recognizer.transcribe = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("ASR must not run when an imported subtitle file is attached")
    )
    be.translator.translate = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("Automatic translation must not run when an imported subtitle file is attached")
    )

    p, _ = pipe(tmp_path, s, be)
    p.run(dialog_video)

    assert [segment["text"] for segment in p.state["segments"]] == expected
    assert [segment["translation"] for segment in p.state["segments"]] == expected
    assert all(segment["mt_source"] == "source_srt" for segment in p.state["segments"])
    assert p.state["input_script"] == "subtitles"
    assert p.state["asr_chunks"] == {}


def test_removing_imported_subtitles_reenables_asr_and_translation(tmp_path, dialog_video):
    srt = tmp_path / "georgian.srt"
    srt.write_text(
        "1\n00:00:01,000 --> 00:00:04,000\nგამარჯობა\n",
        encoding="utf-8",
    )
    s1, be1 = make(tmp_path, imported_subtitles=str(srt))
    p1, _ = pipe(tmp_path, s1, be1)
    p1.run(dialog_video)

    s2, be2 = make(tmp_path)
    translation_calls = []
    original_translate = be2.translator.translate

    def tracked_translate(*args, **kwargs):
        translation_calls.append(1)
        return original_translate(*args, **kwargs)

    be2.translator.translate = tracked_translate
    p2, _ = pipe(tmp_path, s2, be2)
    p2.run(dialog_video)

    assert p2.state["input_script"] == "audio"
    assert p2.state["asr_chunks"]
    assert translation_calls


def test_cancel_then_resume_does_not_redo_work(tmp_path, dialog_video):
    s, be = make(tmp_path)
    be.tts.cancel_after = 3
    p, ctl = pipe(tmp_path, s, be)
    with pytest.raises(CancelledError):
        p.run(dialog_video)
    assert "asr" in p.state["done"] and "tts" not in p.state["done"]
    first_calls = len(be.tts.calls)
    # "restart the app": brand-new objects, same project folder
    s2, be2 = make(tmp_path)
    p2, _ = pipe(tmp_path, s2, be2)
    be2.recognizer.transcribe = lambda *a, **k: (_ for _ in ()).throw(AssertionError("ASR must not rerun"))
    p2.run(dialog_video)
    assert "export" in p2.state["done"]
    assert len(be2.tts.calls) <= 5 - 2          # lines finished before the crash are reused (cached), not regenerated
    assert first_calls + len(be2.tts.calls) <= 6


def test_pause_blocks_and_resume_continues(tmp_path, dialog_video):
    s, be = make(tmp_path)
    p, ctl = pipe(tmp_path, s, be)
    ctl.pause()
    res = {}
    th = threading.Thread(target=lambda: res.setdefault("o", p.run(dialog_video)))
    th.start()
    time.sleep(1.5)
    assert th.is_alive() and "extract" not in p.state["done"]          # really paused before doing anything
    ctl.resume()
    th.join(120)
    assert "o" in res and "export" in p.state["done"]


def test_review_edit_regenerates_only_that_line(tmp_path, dialog_video):
    s, be = make(tmp_path)
    p, _ = pipe(tmp_path, s, be)

    def review(pl):
        pl.state["segments"][1]["translation"] = "ჩემი შესწორებული ტექსტი"
    p.on_review = review
    p.run(dialog_video)
    assert any(c[1] == "ჩემი შესწორებული ტექსტი" and c[0] == "f1" for c in be.tts.calls)
    # re-export after a later edit regenerates just that one line
    p.state["segments"][1]["translation"] = "კიდევ ერთი ვერსია"
    p.invalidate("tts")
    n = len(be.tts.calls)
    p.run(dialog_video)
    assert len(be.tts.calls) == n + 1


def test_missing_voice_fails_clearly_then_resumes(tmp_path, dialog_video):
    s, be = make(tmp_path, tts=StubTTS(voices=("m1",)))               # no female voice at all
    p, _ = pipe(tmp_path, s, be)
    with pytest.raises(VoiceUnavailableError) as e:
        p.run(dialog_video)
    assert "Speaker 2" in e.value.user_message
    assert any("no free female" in w for w in p.state["warnings"])
    assert be.tts.calls == []                                           # nothing synthesized with a wrong voice
    be.tts.voice_ids = ("m1", "f1")                                      # user adds a voice, resumes
    p2 = Pipeline(tmp_path / "proj", s, be, control=Control(), cache_root=tmp_path / "ttscache", extra_voices=[])
    p2.run(dialog_video)
    assert p2.state["assignments"]["speaker_2"] == "f1"


def test_diarization_failure_degrades_with_warning(tmp_path, dialog_video):
    class Bad(Diarizer):
        def diarize(self, w):
            raise GeoDubberError("boom", "model failed")
    # The speaker model only runs when a Hugging Face token is set; with it, a failure must fall back to
    # local pitch clustering (two characters by default) and say so.
    s, be = make(tmp_path, diarizer=Bad(), hf_token="hf_test_token")
    p, _ = pipe(tmp_path, s, be)
    p.run(dialog_video)
    assert {x["speaker_id"] for x in p.state["segments"]} == {"speaker_1", "speaker_2"}
    assert any("Using local pitch clustering" in w for w in p.state["warnings"])


def test_no_audio_video_rejected(tmp_path):
    v = tmp_path / "silent.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(v)], check=True)
    s, be = make(tmp_path)
    p, _ = pipe(tmp_path, s, be)
    with pytest.raises(GeoDubberError) as e:
        p.run(v)
    assert "no audio" in e.value.user_message.lower()


def test_tts_cache_and_invalid_audio(tmp_path):
    from core.georgian_tts import synthesize_cached
    tts = StubTTS()
    v = tts.builtin_voices()[0]
    a = synthesize_cached(tts, v, "hello", tmp_path / "c")
    b = synthesize_cached(tts, v, "hello", tmp_path / "c")
    assert a == b and len(tts.calls) == 1

    class Bad(StubTTS):
        def synthesize(self, text, voice):
            return np.zeros(0, dtype=np.float32), 24000
    with pytest.raises(GeoDubberError):
        synthesize_cached(Bad(), v, "x", tmp_path / "c2")


def test_database(tmp_path):
    from database.project_database import ProjectDatabase
    db = ProjectDatabase(tmp_path / "t.db")
    pid = db.create_project("movie", "/v/movie.mp4", "/p/1")
    db.set_status(pid, "running")
    db.save_assignments(pid, {"speaker_1": "m1", "speaker_2": "f1"})
    assert db.assignments(pid) == {"speaker_1": "m1", "speaker_2": "f1"}
    assert db.projects()[0]["status"] == "running" and len(db.history(pid)) == 2
    db.delete_project(pid)
    assert db.projects() == [] and db.assignments(pid) == {}


def test_settings_roundtrip():
    s = Settings(priority="quality", device="cpu")
    s.save()
    t = Settings.load()
    assert t.priority == "quality" and t.device == "cpu" and Settings().priority == "fast"
    # presets: fast -> tiny (default), quality -> small (see config/settings.py)
    assert Settings().effective_asr_model() == "tiny" and t.effective_asr_model() == "small"



def test_fit_stage_rewrites_lines_whose_real_audio_is_too_long(tmp_path, dialog_video):
    from core.processing_manager import STAGES

    class LongThenShort(StubTranslator):
        """First translation is much longer than any window; shorten() returns a short line."""
        def translate(self, texts, src_lang, **kwargs):
            self.last_sources = ["gemini"] * len(texts)
            return ["ძალიან გრძელი ხაზი ყველა სიტყვით " * 6 for _ in texts]

        def shorten(self, items):
            return {it["i"]: "მოკლე" for it in items}

    s, be = make(tmp_path)
    be.translator = LongThenShort()
    p, _ = pipe(tmp_path, s, be)
    p.run(dialog_video)
    assert p.state["fit"]["shortened"] >= 1
    assert any(x["translation"] == "მოკლე" for x in p.state["segments"])
    assert "fit" in [k for k, _, _ in STAGES]
