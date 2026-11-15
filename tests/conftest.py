import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("GEODUBBER_HOME", str(tmp_path / "home"))
    yield


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory):
    """Real MP4 made with ffmpeg: 20 s test pattern + 440 Hz 'dialogue' tone + quiet noise 'music'."""
    d = tmp_path_factory.mktemp("media")
    out = d / "sample.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=20",
           "-f", "lavfi", "-i", "sine=frequency=440:duration=20",
           "-f", "lavfi", "-i", "anoisesrc=color=pink:amplitude=0.05:duration=20",
           "-filter_complex", "[1:a][2:a]amix=inputs=2:normalize=0[a]", "-map", "0:v", "-map", "[a]",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out)]
    subprocess.run(cmd, check=True)
    return out
