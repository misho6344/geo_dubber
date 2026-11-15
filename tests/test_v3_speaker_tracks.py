"""Tests for v3.5: stable character voices over time and the single-speaker 5 s gender preview."""
import wave

import numpy as np

from core import speaker_diarization as SD
from core import speaker_tracks as ST

SR = 16000


def _voice(f0, seconds):
    """Harmonic tone at f0: a simple voice-like signal that pitch detection can read."""
    t = np.arange(int(seconds * SR)) / SR
    x = sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, 6))
    return (0.3 * x / np.max(np.abs(x))).astype(np.float32)


def _write_track(path, pieces):
    """pieces: [(start_s, f0 or None, seconds)]. Silence is filled between them."""
    total = max(s + d for s, _, d in pieces) + 0.5
    x = np.zeros(int(total * SR), dtype=np.float32)
    for start, f0, dur in pieces:
        if f0:
            a = int(start * SR)
            v = _voice(f0, dur)
            x[a:a + len(v)] = v
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return path


# ---- pure logic --------------------------------------------------------------------------------

def test_line_gender_boundary_and_confidence():
    assert ST.line_gender(None) == (None, 0.0)
    g, c = ST.line_gender(120.0)
    assert g == "male" and c >= 0.5
    g, c = ST.line_gender(230.0)
    assert g == "female" and c >= 0.5
    g, c = ST.line_gender(160.0)          # near the boundary: low confidence
    assert g == "male" and c < ST.MIN_CONF


def test_effective_gender_fills_unclear_line_from_neighbours():
    f0s = [120.0, 118.0, 160.0, 122.0, 119.0]   # middle line is unclear
    eff = ST.effective_genders(f0s)
    assert eff == ["male", "male", "male", "male", "male"]


def test_effective_gender_keeps_clear_lines_as_they_are():
    f0s = [120.0, 230.0, 118.0, 225.0]
    assert ST.effective_genders(f0s) == ["male", "female", "male", "female"]


def test_wrong_gender_line_moves_to_matching_character():
    speakers = ["speaker_1", "speaker_2", "speaker_2", "speaker_2"]
    f0s = [120.0, 220.0, 118.0, 225.0]          # line 2 is male but labelled speaker_2 (female)
    info = {"speaker_1": {"gender": "male"}, "speaker_2": {"gender": "female"}}
    eff = ST.effective_genders(f0s)
    out, moved, unresolved = ST.reassign_lines(speakers, f0s, eff, info)
    assert out == ["speaker_1", "speaker_2", "speaker_1", "speaker_2"]
    assert moved == 1 and unresolved == 0


def test_no_matching_character_is_reported_not_forced():
    speakers = ["speaker_1", "speaker_1"]
    f0s = [120.0, 230.0]
    info = {"speaker_1": {"gender": "male"}}
    out, moved, unresolved = ST.reassign_lines(speakers, f0s, ST.effective_genders(f0s), info)
    assert out == speakers and moved == 0 and unresolved == 1


def test_same_gender_line_moves_only_when_clearly_closer_to_another_character():
    # Two female characters: 190 Hz and 260 Hz. A 255 Hz line labelled as the 190 Hz one moves.
    speakers = ["speaker_2", "speaker_3", "speaker_2", "speaker_3", "speaker_2"]
    f0s = [190.0, 260.0, 255.0, 258.0, 192.0]
    info = {"speaker_2": {"gender": "female"}, "speaker_3": {"gender": "female"}}
    out, moved, _ = ST.reassign_lines(speakers, f0s, ST.effective_genders(f0s), info)
    assert out[2] == "speaker_3"
    assert moved == 1
    # A line only a little off its character stays where it is (no jitter between similar voices).
    speakers2 = ["speaker_2", "speaker_3", "speaker_2"]
    f0s2 = [190.0, 260.0, 200.0]
    out2, moved2, _ = ST.reassign_lines(speakers2, f0s2, ST.effective_genders(f0s2), info)
    assert out2 == speakers2 and moved2 == 0


def test_strong_contrary_speech_flips_character_gender_with_warning_data():
    speakers = ["speaker_1"] * 4
    f0s = [230.0, 235.0, 228.0, 232.0]           # all female pitch, but labelled male
    durs = [1.5, 1.5, 1.5, 1.5]
    info = {"speaker_1": {"gender": "male", "gender_source": "pitch", "gender_confidence": 0.5}}
    changes = ST.check_speaker_genders(speakers, f0s, durs, ST.effective_genders(f0s), info)
    assert changes and changes[0]["to"] == "female"
    assert info["speaker_1"]["gender"] == "female"
    assert info["speaker_1"]["gender_source"] == "pitch_check"


def test_user_set_gender_is_never_flipped():
    speakers = ["speaker_1"] * 4
    f0s = [230.0, 235.0, 228.0, 232.0]
    durs = [1.5] * 4
    info = {"speaker_1": {"gender": "male", "gender_source": "user"}}
    assert ST.check_speaker_genders(speakers, f0s, durs, ST.effective_genders(f0s), info) == []
    assert info["speaker_1"]["gender"] == "male"


def test_mixed_speech_does_not_flip_character():
    speakers = ["speaker_1"] * 4
    f0s = [120.0, 230.0, 118.0, 232.0]           # half and half: no strong disagreement
    durs = [1.5] * 4
    info = {"speaker_1": {"gender": "male", "gender_source": "pitch"}}
    assert ST.check_speaker_genders(speakers, f0s, durs, ST.effective_genders(f0s), info) == []


# ---- audio-level stabilise -----------------------------------------------------------------------

def test_stabilise_fixes_a_man_labelled_as_the_woman_in_a_real_track(tmp_path):
    wav = _write_track(tmp_path / "a16.wav", [
        (0.0, 120.0, 1.5),   # man
        (2.0, 220.0, 1.5),   # woman
        (4.0, 118.0, 1.5),   # man, but labelled speaker_2 (wrong)
        (6.0, 225.0, 1.5),   # woman
        (8.0, 122.0, 1.5),   # man
    ])
    segs = [
        {"start": 0.0, "end": 1.5, "speaker_id": "speaker_1"},
        {"start": 2.0, "end": 3.5, "speaker_id": "speaker_2"},
        {"start": 4.0, "end": 5.5, "speaker_id": "speaker_2"},
        {"start": 6.0, "end": 7.5, "speaker_id": "speaker_2"},
        {"start": 8.0, "end": 9.5, "speaker_id": "speaker_1"},
    ]
    info = {"speaker_1": {"gender": "male", "gender_source": "pitch", "gender_confidence": 0.8},
            "speaker_2": {"gender": "female", "gender_source": "pitch", "gender_confidence": 0.8}}
    new_info, msgs = ST.stabilise(segs, wav, info)
    assert [s["speaker_id"] for s in segs] == ["speaker_1", "speaker_2", "speaker_1", "speaker_2", "speaker_1"]
    assert new_info["speaker_1"]["gender"] == "male"
    assert new_info["speaker_2"]["gender"] == "female"
    assert any("moved" in m for m in msgs)


def test_stabilise_keeps_consistent_labels_untouched(tmp_path):
    wav = _write_track(tmp_path / "b16.wav", [(0.0, 120.0, 1.5), (2.0, 230.0, 1.5), (4.0, 121.0, 1.5)])
    segs = [{"start": 0.0, "end": 1.5, "speaker_id": "speaker_1"},
            {"start": 2.0, "end": 3.5, "speaker_id": "speaker_2"},
            {"start": 4.0, "end": 5.5, "speaker_id": "speaker_1"}]
    info = {"speaker_1": {"gender": "male"}, "speaker_2": {"gender": "female"}}
    new_info, msgs = ST.stabilise(segs, wav, info)
    assert [s["speaker_id"] for s in segs] == ["speaker_1", "speaker_2", "speaker_1"]
    assert msgs == []
    assert set(new_info) == {"speaker_1", "speaker_2"}


def test_stabilise_drops_a_character_left_without_lines(tmp_path):
    wav = _write_track(tmp_path / "c16.wav", [(0.0, 120.0, 1.5), (2.0, 230.0, 1.5)])
    segs = [{"start": 0.0, "end": 1.5, "speaker_id": "speaker_1"},
            {"start": 2.0, "end": 3.5, "speaker_id": "speaker_1"}]
    info = {"speaker_1": {"gender": "male"}, "speaker_2": {"gender": "female"}}
    new_info, _ = ST.stabilise(segs, wav, info)
    assert set(new_info) == {"speaker_1"}


# ---- 5 s preview window -----------------------------------------------------------------------------

def test_preview_never_includes_another_speakers_line():
    own = [{"start": 0.0, "end": 2.0}, {"start": 3.2, "end": 6.0}]
    other = [{"start": 2.2, "end": 3.0}]          # sits in the gap between the two own lines
    a, b = SD.pick_preview_window(own, 5.0, others=other)
    for o in other:
        assert min(b, o["end"]) - max(a, o["start"]) <= 0.1
    assert b - a <= 5.0 + 1e-6


def test_preview_chains_own_neighbours_when_no_other_speaker_is_between():
    own = [{"start": 0.0, "end": 2.0}, {"start": 2.3, "end": 4.5}]
    a, b = SD.pick_preview_window(own, 5.0, others=[])
    assert a == 0.0 and b >= 4.4


def test_preview_single_long_line_is_centred_and_clipped():
    own = [{"start": 10.0, "end": 20.0}]
    a, b = SD.pick_preview_window(own, 5.0, others=[])
    assert 10.0 <= a and b <= 20.0 and abs((b - a) - 5.0) < 1e-6


def test_preview_without_others_argument_still_works():
    own = [{"start": 0.0, "end": 1.0}, {"start": 1.5, "end": 2.5}]
    a, b = SD.pick_preview_window(own, 5.0)
    assert a == 0.0 and b >= 2.4


def test_octave_error_does_not_move_a_woman_to_the_man():
    # Woman at 230 Hz, one line tracked at half pitch (115 Hz): it stays with her.
    speakers = ["speaker_2", "speaker_2", "speaker_2", "speaker_1"]
    f0s = [230.0, 115.0, 228.0, 120.0]
    info = {"speaker_1": {"gender": "male"}, "speaker_2": {"gender": "female"}}
    out, moved, _ = ST.reassign_lines(speakers, f0s, ST.effective_genders(f0s), info)
    assert out[1] == "speaker_2" and moved == 0


def test_mass_moves_are_refused(tmp_path):
    # Every line is labelled wrongly: the pitch check would move far too many lines, so nothing changes.
    # Both characters are a mix of male and female pitch (no character is clearly the other gender),
    # so no gender flip applies and the line moves would touch half of all lines.
    pieces = [(i * 2.0, 120.0 if i % 4 < 2 else 230.0, 1.5) for i in range(10)]
    wav = _write_track(tmp_path / "m16.wav", pieces)
    segs = [{"start": p[0], "end": p[0] + 1.5, "speaker_id": "speaker_1" if i % 2 == 0 else "speaker_2"}
            for i, p in enumerate(pieces)]
    before = [s["speaker_id"] for s in segs]
    info = {"speaker_1": {"gender": "male"}, "speaker_2": {"gender": "female"}}
    _, msgs = ST.stabilise(segs, wav, info)
    assert [s["speaker_id"] for s in segs] == before
    assert any("too many" in m for m in msgs)
