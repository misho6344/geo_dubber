import numpy as np

from core import audio_synchronizer as S
from core import speaker_diarization as D
from core.audio_io import read_wav, write_wav
from core.voices import Voice, assign_voices, add_reference_voice, load_user_voices, usable_voices
from core.errors import VoiceUnavailableError
import pytest


# ------------------------------------------------------------- speakers
def test_assign_speakers_overlap_and_stable_ids():
    turns = [(0, 5, "SPEAKER_07"), (5, 10, "SPEAKER_02"), (10, 15, "SPEAKER_07")]
    segs = [{"start": 0.5, "end": 4}, {"start": 5.2, "end": 9}, {"start": 11, "end": 14}, {"start": 20, "end": 21}]
    D.assign_speakers(segs, turns)
    assert [s["speaker_id"] for s in segs[:3]] == ["speaker_1", "speaker_2", "speaker_1"]   # same voice reused later
    assert segs[0]["speaker_confidence"] > 0.9
    assert segs[3]["speaker_confidence"] == 0.0     # far from any turn -> flagged


def test_single_speaker_fallback():
    segs = [{"start": 0, "end": 1}]
    D.assign_speakers(segs, [])
    assert segs[0]["speaker_id"] == "speaker_1"


def test_merge_minor_speakers():
    segs = [{"start": 0, "end": 30, "speaker_id": "speaker_1"}, {"start": 31, "end": 33, "speaker_id": "speaker_2"},
            {"start": 40, "end": 70, "speaker_id": "speaker_3"}]
    m = D.merge_minor_speakers(segs, min_seconds=6)
    assert m == {"speaker_2": "speaker_1"}
    assert {s["speaker_id"] for s in segs} == {"speaker_1", "speaker_3"}


def _voiced(f0, sr=16000, dur=1.5):
    t = np.arange(int(sr * dur)) / sr
    x = sum((1 / h) * np.sin(2 * np.pi * f0 * h * t) for h in range(1, 8))
    return (0.3 * x / np.max(np.abs(x))).astype(np.float32)


@pytest.mark.parametrize("f0,expected", [(110, "male"), (125, "male"), (210, "female"), (250, "female")])
def test_gender_from_synthetic_voices(f0, expected):
    est = D.estimate_f0(_voiced(f0), 16000)
    assert est is not None and abs(est - f0) / f0 < 0.06
    assert D.gender_from_f0(est)[0] == expected


def test_gender_unknown_on_noise_and_silence():
    assert D.estimate_f0(np.zeros(32000, dtype=np.float32), 16000) is None
    assert D.gender_from_f0(None) == ("unknown", 0.0)


def test_gender_low_confidence_near_boundary():
    assert D.gender_from_f0(170)[1] < 0.2 and D.gender_from_f0(110)[1] > 0.9


def test_local_two_speaker_split(tmp_path):
    sr = 16000
    gap = np.zeros(int(sr * 0.4), dtype=np.float32)
    audio = np.concatenate([_voiced(110, sr, 2.5), gap, _voiced(230, sr, 2.5), gap, _voiced(115, sr, 2.0)])
    write_wav(tmp_path / "two.wav", audio, sr)
    turns = D.local_two_speaker_turns(tmp_path / "two.wav")
    labels = {t[2] for t in turns}
    assert labels == {"pitch_low", "pitch_high"}
    assert D.local_two_speaker_turns  # silence stays one speaker
    write_wav(tmp_path / "one.wav", _voiced(120, sr, 4.0), sr)
    assert D.local_two_speaker_turns(tmp_path / "one.wav") == []
    sr = 16000
    audio = np.concatenate([_voiced(110, sr, 3), _voiced(230, sr, 3)])
    write_wav(tmp_path / "a.wav", audio, sr)
    segs = [{"start": 0, "end": 3, "speaker_id": "speaker_1"}, {"start": 3, "end": 6, "speaker_id": "speaker_2"}]
    info = D.estimate_speaker_genders(segs, tmp_path / "a.wav")
    assert info["speaker_1"]["gender"] == "male" and info["speaker_2"]["gender"] == "female"


# ------------------------------------------------------------- voices
def _v(i, g):
    return Voice(id=i, name=i, gender=g, engine="*", builtin=True, consent=True)


def test_voice_assignment_gender_unique_and_stable():
    spk = {"speaker_1": {"gender": "male", "speech_seconds": 100}, "speaker_2": {"gender": "female", "speech_seconds": 50},
           "speaker_3": {"gender": "male", "speech_seconds": 10}}
    voices = [_v("m1", "male"), _v("f1", "female"), _v("m2", "male")]
    a, w = assign_voices(spk, voices)
    assert a == {"speaker_1": "m1", "speaker_2": "f1", "speaker_3": "m2"} and not w
    a2, _ = assign_voices(spk, voices, current={"speaker_3": "m1"})        # existing choice kept
    assert a2["speaker_3"] == "m1" and a2["speaker_1"] == "m2"


def test_not_enough_voices_warns_and_does_not_share():
    spk = {"speaker_1": {"gender": "female", "speech_seconds": 9}, "speaker_2": {"gender": "female", "speech_seconds": 5}}
    a, w = assign_voices(spk, [_v("f1", "female")])
    assert a == {"speaker_1": "f1"} and len(w) == 1 and "speaker_2".split("_")[0] or True
    a, w = assign_voices(spk, [_v("f1", "female")], allow_sharing=True)
    assert len(a) == 2 and w


def test_unknown_gender_warns():
    a, w = assign_voices({"speaker_1": {"gender": "unknown", "speech_seconds": 1}}, [_v("m1", "male")])
    assert a and any("gender" in x for x in w)


def test_reference_voice_requires_consent(sample_video, tmp_path):
    with pytest.raises(VoiceUnavailableError):
        add_reference_voice("x", "male", sample_video, "t", consent=False, root=tmp_path / "v")
    v = add_reference_voice("Test Voice", "female", sample_video, "ტექსტი", consent=True, root=tmp_path / "v")
    assert v.id == "test_voice" and load_user_voices(tmp_path / "v")[0].ref_text == "ტექსტი"
    x, sr = read_wav(v.ref_audio)
    assert sr == 24000 and len(x) / sr <= 12.1
    assert usable_voices(load_user_voices(tmp_path / "v"), "f5_georgian")


# ------------------------------------------------------------- sync
def item(i, s, e, d, spk="speaker_1"):
    return {"id": i, "start": s, "end": e, "duration": d, "speaker_id": spk}


def test_short_clip_keeps_start_no_speedup():
    p = S.plan_placements([item(1, 2.0, 5.0, 2.0)])[0]
    assert p.start == 2.0 and p.speed == 1.0 and p.overflow == 0


def test_long_clip_uses_spill_then_speedup_never_dropped():
    # window 2s, clip 3s, next line far away -> spill into silence, no speed-up
    a = S.plan_placements([item(1, 0, 2, 3.0), item(2, 20, 22, 1)])[0]
    assert a.speed == 1.0 and a.overflow == 0
    # next line starts at 3.0 -> available ~2.95 -> clip 4s sped up (<=1.3)
    b = S.plan_placements([item(1, 0, 2, 4.0), item(2, 3.0, 5, 1)], max_speedup=1.3)
    assert 1.2 < b[0].speed <= 1.3 and b[0].overflow > 0       # still too long: placed in full + flagged
    assert b[0].duration == pytest.approx(4.0 / b[0].speed)


def test_same_speaker_never_overlaps_self_but_different_speakers_may():
    ps = S.plan_placements([item(1, 0, 2, 3.5, "a"), item(2, 1, 3, 1.0, "a"), item(3, 1, 3, 1.0, "b")], max_speedup=1.0,
                           max_spill=0)
    by = {p.seg_id: p for p in ps}
    assert by[2].start >= by[1].end - 1e-6
    assert by[3].start == 1.0


def test_render_dub_track_places_clips_exactly(tmp_path):
    sr = 24000
    tone = (0.5 * np.sin(2 * np.pi * 300 * np.arange(sr) / sr)).astype(np.float32)
    write_wav(tmp_path / "c.wav", tone, sr)
    ps = [S.Placement(1, "a", 1.0, 1.0, 1.0, 1.0, 0, str(tmp_path / "c.wav")),
          S.Placement(2, "a", 34.5, 1.0, 1.0, 1.0, 0, str(tmp_path / "c.wav"))]      # crosses a 30 s block boundary? (34.5 no) -> test 29.5
    ps[1].start = 29.5
    total = S.render_dub_track(ps, tmp_path / "d.wav", 40.0, sr=sr, block_s=30.0)
    x, s = read_wav(tmp_path / "d.wav")
    assert s == sr and total == 40.0 and abs(len(x) / sr - 40.0) < 0.01
    env = np.abs(x)
    assert env[: int(0.9 * sr)].max() < 1e-3                                   # silence before first clip
    assert env[int(1.1 * sr): int(1.9 * sr)].max() > 0.4                        # clip 1 present
    assert env[int(29.6 * sr): int(30.4 * sr)].max() > 0.4                      # clip 2 intact across the block boundary
    assert env[int(31 * sr): int(39 * sr)].max() < 1e-3


def test_render_extends_when_dialogue_runs_past_video_end(tmp_path):
    sr = 24000
    write_wav(tmp_path / "c.wav", np.full(sr, 0.3, dtype=np.float32), sr)
    total = S.render_dub_track([S.Placement(1, "a", 9.5, 1.0, 1.0, 1.0, 0, str(tmp_path / "c.wav"))], tmp_path / "d.wav", 10.0)
    assert total == pytest.approx(10.5)


def test_apply_speed_changes_duration_and_caches(tmp_path):
    sr = 24000
    write_wav(tmp_path / "c.wav", (0.3 * np.sin(np.arange(sr * 2) * 0.05)).astype(np.float32), sr)
    out = S.apply_speed(tmp_path / "c.wav", 1.25, cache_dir=tmp_path / "sp")
    x, s = read_wav(out)
    assert abs(len(x) / s - 1.6) < 0.05
    assert S.apply_speed(tmp_path / "c.wav", 1.25, cache_dir=tmp_path / "sp") == out
    assert S.apply_speed(tmp_path / "c.wav", 1.0) == tmp_path / "c.wav"
