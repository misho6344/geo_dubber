import numpy as np

from core.audio_io import read_wav, write_wav
from core.audio_synchronizer import Placement, plan_placements, render_dub_track


def _item(identifier, start, end, duration, speaker="speaker_1"):
    return {
        "id": identifier,
        "start": start,
        "end": end,
        "duration": duration,
        "speaker_id": speaker,
    }


def test_same_speaker_lines_are_serialized_but_other_speakers_can_overlap():
    placements = plan_placements(
        [
            _item(1, 0.0, 2.0, 3.5, "speaker_a"),
            _item(2, 1.0, 3.0, 1.0, "speaker_a"),
            _item(3, 1.0, 3.0, 1.0, "speaker_b"),
        ],
        max_speedup=1.0,
        max_spill=0.0,
    )
    by_id = {placement.seg_id: placement for placement in placements}
    assert by_id[2].start >= by_id[1].end + 0.05 - 1e-6
    assert by_id[3].start == 1.0


def test_other_speaker_does_not_force_unnecessary_speedup():
    placements = plan_placements(
        [_item(1, 0.0, 1.0, 2.0, "speaker_a"), _item(2, 0.5, 1.5, 0.5, "speaker_b")],
        max_speedup=1.3,
        max_spill=1.0,
    )
    by_id = {placement.seg_id: placement for placement in placements}
    assert by_id[1].speed == 1.0
    assert by_id[2].start == 0.5


def test_rendered_track_keeps_speech_after_source_video_ends(tmp_path):
    sample_rate = 24000
    clip = np.full(sample_rate, 0.3, dtype=np.float32)
    clip_path = tmp_path / "last-line.wav"
    write_wav(clip_path, clip, sample_rate)

    placement = Placement(
        seg_id=1,
        speaker_id="speaker_1",
        start=9.5,
        duration=1.0,
        orig_duration=1.0,
        speed=1.0,
        overflow=0.0,
        audio_path=str(clip_path),
    )
    output = tmp_path / "dub.wav"
    total = render_dub_track([placement], output, total_duration=10.0, sr=sample_rate)
    samples, decoded_rate = read_wav(output)

    assert decoded_rate == sample_rate
    assert total == 10.5
    assert abs(len(samples) / decoded_rate - 10.5) < 0.01
