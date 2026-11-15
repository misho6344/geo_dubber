import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import reset_translations as R  # noqa: E402


def _state(tmp_path):
    data = {"segments": [
        {"id": 0, "translation": "ერთი", "mt_source": "gemini"},
        {"id": 1, "translation": "ორი", "mt_source": "web"},
        {"id": 2, "translation": "", "mt_source": "failed"},
    ], "tts": {"0": {"path": "a"}, "1": {"path": "b"}}}
    p = tmp_path / "state.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


def test_reset_all_clears_translations_and_keeps_backup(tmp_path):
    p = _state(tmp_path)
    cleared, kept = R.reset(p)
    data = json.loads(p.read_text(encoding="utf-8"))
    assert cleared == 2 and kept == 0
    assert all(not s["translation"] for s in data["segments"][:2])
    assert data["tts"] == {}                                   # TTS cache for cleared lines dropped
    assert json.loads(p.with_suffix(".json.bak").read_text(encoding="utf-8"))["segments"][0]["translation"] == "ერთი"


def test_only_web_keeps_gemini_lines(tmp_path):
    p = _state(tmp_path)
    cleared, kept = R.reset(p, only_web=True)
    data = json.loads(p.read_text(encoding="utf-8"))
    assert cleared == 1 and kept == 1
    assert data["segments"][0]["translation"] == "ერთი"
    assert data["segments"][1]["translation"] == ""
