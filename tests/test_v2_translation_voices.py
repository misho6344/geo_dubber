"""Tests for the v2 translation, voice-variant and speaker-check changes.

Gemini is replaced by a fake client, so these tests run without the google-genai SDK or network access.
"""
import json
import re
import shutil
from pathlib import Path

import pytest

from core import translation as T
from core.gemini_speakers import pitch_contradicts
from core.georgian_tts import EdgeGeorgianBackend
from core.speech_recognition import collapse_repetitions


GE = "მიკი ამბობს ხმამაღლა და გარკვევით"          # plain Georgian, short enough for any window


class _Resp:
    def __init__(self, text):
        self.text = text


class FakeModels:
    def __init__(self, responder):
        self.responder = responder
        self.calls = []

    def generate_content(self, model, contents, config=None):
        self.calls.append({"model": model, "prompt": contents, "config": config})
        return self.responder(model, contents, len(self.calls))


class FakeClient:
    def __init__(self, responder):
        self.models = FakeModels(responder)


def _input_items(prompt):
    """The JSON array the translator embeds after 'INPUT:' in every prompt."""
    raw = prompt.split("INPUT:\n", 1)[1]
    return json.loads(raw)


def answer_all(text_for=lambda item: GE):
    """Responder that answers every input line of a batch with `text_for(item)`."""
    def responder(model, prompt, n):
        lines = [{"i": it["i"], "ka": text_for(it)} for it in _input_items(prompt)]
        return _Resp(json.dumps({"lines": lines}, ensure_ascii=False))
    return responder


def make_translator(responder, *, web=False, **kw):
    t = T.GeminiTranslator(model="gemini-3.5-flash-lite", api_key="test-key", batch=16,
                           sleep=lambda s: None, web_fallback=web, **kw)
    t.client = FakeClient(responder)
    return t


# ----------------------------------------------------------------------------- pure helpers
def test_glossary_parses_pairs_comments_and_keep_latin_entries():
    g = T.parse_glossary("# names\nMickey = მიკი\nNasha=ნაშა\nBen = Ben\n\nbroken line\n")
    assert g == {"Mickey": "მიკი", "Nasha": "ნაშა", "Ben": "Ben"}


def test_batch_answer_parser_tolerates_code_fences_and_prose():
    raw = 'Sure! ```json\n{"lines":[{"i":0,"ka":"ერთი"},{"i":1,"ka":"ორი"}]}\n```'
    assert T.parse_batch_answer(raw) == {0: "ერთი", 1: "ორი"}
    assert T.parse_batch_answer("not json at all") == {}


def test_line_problem_rules():
    allowed = {"Ben"}
    assert T.line_problem(GE, 5.0, allowed) is None
    assert T.line_problem("", 5.0, allowed) == "empty"
    assert T.line_problem("hello there", 5.0, allowed) == "no Georgian text"
    assert T.line_problem(GE + " slacked", 5.0, allowed).startswith("English left")
    assert T.line_problem(GE + " Ben", 5.0, allowed) is None           # glossary keeps this word in Latin
    long_text = " ".join(["სიტყვა"] * 40)
    assert T.line_problem(long_text, 2.0, allowed, check_length=True) == "too long for its window"
    assert T.line_problem(long_text, 2.0, allowed) is None              # length only checked when asked


def test_very_short_windows_are_not_length_checked():
    # a 0.5 s window can never hold a whole Georgian word; flagging it would waste API calls
    assert not T.too_long("მიკი", 0.46)


# ----------------------------------------------------------------------------- batching + validation
def test_whole_batch_is_one_request_and_keeps_order():
    t = make_translator(answer_all(lambda it: f"ხაზი {it['i']} ჩვენი"))
    texts = [f"line {i} hello" for i in range(16)]
    out = t.translate(texts, "en", max_seconds_list=[10] * 16, genders=["male"] * 16,
                      glossary={"Mickey": "მიკი"})
    assert len(t.client.models.calls) == 1
    assert out == [f"ხაზი {i} ჩვენი" for i in range(16)]
    assert t.last_sources == ["gemini"] * 16


def test_prompt_contains_glossary_gender_and_time_window():
    t = make_translator(answer_all())
    t.translate(["Mickey, hello"], "en", max_seconds_list=[3.0], genders=["female"],
                glossary={"Mickey": "მიკი", "Ben": "Ben"})
    prompt = t.client.models.calls[0]["prompt"]
    assert '"Mickey" -> "მიკი"' in prompt
    assert "Keep these words in Latin letters: Ben" in prompt
    items = _input_items(prompt)
    assert items[0]["speaker"] == "female" and items[0]["max_seconds"] == 3.0


def test_missing_and_english_lines_are_repaired_in_one_second_request():
    def responder(model, prompt, n):
        items = _input_items(prompt)
        if n == 1:   # first batch: line 3 comes back in English, line 5 is missing
            lines = [{"i": it["i"], "ka": ("slacked off" if it["i"] == 3 else GE)} for it in items if it["i"] != 5]
        else:        # repair request: only the two bad lines are sent
            assert sorted(it["i"] for it in items) == [3, 5]
            lines = [{"i": it["i"], "ka": GE} for it in items]
        return _Resp(json.dumps({"lines": lines}, ensure_ascii=False))

    t = make_translator(responder)
    out = t.translate([f"line {i}" for i in range(8)], "en", max_seconds_list=[10] * 8)
    assert len(t.client.models.calls) == 2
    assert all(out)
    assert t.last_sources[3] == "gemini_retry" and t.last_sources[5] == "gemini_retry"
    assert t.last_sources[0] == "gemini"


def test_overlong_line_gets_a_shorter_version_when_it_fits():
    long_ka = " ".join(["სიტყვა"] * 30)
    def responder(model, prompt, n):
        items = _input_items(prompt)
        text = long_ka if n == 1 else GE          # the 'shorter' request returns a fitting line
        return _Resp(json.dumps({"lines": [{"i": it["i"], "ka": text} for it in items]}, ensure_ascii=False))
    t = make_translator(responder)
    out = t.translate(["a fairly long sentence here"], "en", max_seconds_list=[3.0])
    assert out == [GE]
    assert t.last_sources == ["gemini_retry"]
    assert "shorter" in t.client.models.calls[1]["prompt"].lower()


# ----------------------------------------------------------------------------- quota + fallbacks
def test_model_rotation_absorbs_a_single_model_quota_error():
    def responder(model, prompt, n):
        if n == 1:
            raise RuntimeError("429 RESOURCE_EXHAUSTED. Please retry in 2.0s")
        return answer_all()(model, prompt, n)
    t = make_translator(responder)
    assert t.translate(["hello"], "en", max_seconds_list=[5]) == [GE]
    assert t.stats["quota_waits"] == 0                       # another model answered, no waiting needed
    assert t.client.models.calls[1]["model"] != t.client.models.calls[0]["model"]


def test_quota_error_waits_for_the_server_delay_then_succeeds():
    # The first round hits the quota on all three models (calls 1-3); the second round succeeds (call 4).
    k = len(T.GeminiTranslator.CANDIDATE_MODELS)   # every model is over quota in the first round
    def responder(model, prompt, n):
        if n <= k:
            raise RuntimeError("429 RESOURCE_EXHAUSTED. Please retry in 2.0s")
        return answer_all()(model, prompt, n)
    waits = []
    t = make_translator(responder)
    t.sleep = waits.append
    out = t.translate(["hello"], "en", max_seconds_list=[5])
    assert out == [GE]
    assert t.stats["quota_waits"] == 1
    assert sum(waits) == pytest.approx(2.0)


def test_failed_lines_stay_empty_when_web_fallback_is_off():
    def responder(model, prompt, n):
        raise RuntimeError("500 internal")
    t = make_translator(responder, web=False)
    out = t.translate(["a", "b"], "en", max_seconds_list=[5, 5])
    assert out == ["", ""]
    assert t.last_sources == ["failed", "failed"]


def test_web_fallback_is_used_and_flagged(monkeypatch):
    def responder(model, prompt, n):
        raise RuntimeError("500 internal")
    monkeypatch.setattr(T, "_web_google_translate", lambda text, src="en": GE)
    t = make_translator(responder, web=True)
    out = t.translate(["hello"], "en", max_seconds_list=[5])
    assert out == [GE]
    assert t.last_sources == ["web"]


def test_cancel_during_quota_wait_is_honoured():
    from core.errors import CancelledError

    def responder(model, prompt, n):
        raise RuntimeError("429 RESOURCE_EXHAUSTED. retry in 30s")

    calls = {"n": 0}

    def checkpoint():
        calls["n"] += 1
        if calls["n"] > 3:
            raise CancelledError()

    t = make_translator(responder, checkpoint=checkpoint)
    with pytest.raises(CancelledError):
        t.translate(["hello"], "en", max_seconds_list=[5])


# ----------------------------------------------------------------------------- pipeline entry point
def test_translate_segments_passes_genders_glossary_and_sets_sources():
    segs = [
        {"id": 0, "start": 0.0, "end": 2.0, "text": "Mickey, hello", "speaker_id": "speaker_1", "gender": "male"},
        {"id": 1, "start": 2.0, "end": 4.0, "text": "Nasha says hi", "speaker_id": "speaker_2", "gender": "female"},
    ]
    t = make_translator(answer_all())
    T.translate_segments(t, segs, "en", 16, {"Mickey": "მიკი", "Nasha": "ნაშა"})
    assert all(s["translation"] == GE for s in segs)
    assert [s["mt_source"] for s in segs] == ["gemini", "gemini"]
    prompt = t.client.models.calls[0]["prompt"]
    assert '"Nasha" -> "ნაშა"' in prompt
    assert [it["speaker"] for it in _input_items(prompt)] == ["male", "female"]


def test_georgian_source_is_copied_not_translated():
    segs = [{"id": 0, "start": 0, "end": 2, "text": GE}]
    t = make_translator(answer_all())
    T.translate_segments(t, segs, "ka")
    assert segs[0]["translation"] == GE and segs[0]["mt_source"] == "source"
    assert t.client.models.calls == []


# ----------------------------------------------------------------------------- voices
def test_edge_variants_give_each_gender_distinct_voices():
    voices = EdgeGeorgianBackend(variants=True).builtin_voices()
    males = [v for v in voices if v.gender == "male"]
    females = [v for v in voices if v.gender == "female"]
    assert len(males) == 3 and len(females) == 3
    assert {v.id for v in voices} >= {"edge_ka_giorgi", "edge_ka_eka"}      # old ids still exist (saved projects)
    assert len({(v.builtin_params["pitch"], v.builtin_params["rate"]) for v in voices if v.gender == "male"}) == 3


def test_edge_variants_can_be_switched_off():
    voices = EdgeGeorgianBackend(variants=False).builtin_voices()
    assert sorted(v.id for v in voices) == ["edge_ka_eka", "edge_ka_giorgi"]


def test_three_male_characters_get_three_different_voices():
    from core.voices import assign_voices
    voices = EdgeGeorgianBackend(variants=True).builtin_voices()
    speakers = {f"speaker_{i}": {"gender": "male", "speech_seconds": 10 - i, "label": f"S{i}",
                                 "gender_confidence": 0.95} for i in range(1, 4)}
    mapping, warnings = assign_voices(speakers, voices, allow_sharing=False)
    assert len(set(mapping.values())) == 3
    assert not warnings


def test_shipped_reference_voice_files_parse_and_load():
    from core.voices import load_user_voices
    root = Path(__file__).resolve().parent.parent / "voices"
    for name in ("giorgi", "nino"):
        json.loads((root / name / "voice.json").read_text(encoding="utf-8"))   # must be valid JSON
    loaded = load_user_voices(root)
    assert {v.id for v in loaded} == {"giorgi", "nino"}


# ----------------------------------------------------------------------------- speaker check + ASR
def test_pitch_contradiction_flags_only_clear_outliers():
    assert pitch_contradicts("male", 230.0)
    assert not pitch_contradicts("male", 120.0)
    assert pitch_contradicts("female", 110.0)
    assert not pitch_contradicts("female", 210.0)
    assert not pitch_contradicts("unknown", 300.0)


@pytest.mark.parametrize("src,expected", [
    ("ძვირფასი- ძვირფასი- ძვირფასი- ძვირფასი-", "ძვირფასი-"),
    ("no, no, no, no", "no"),
    ("yes yes", "yes yes"),          # two repeats are real speech
    ("ok ok ok done", "ok done"),
])
def test_repetition_loops_are_collapsed(src, expected):
    assert collapse_repetitions(src) == expected


def test_middle_speakers_follow_their_own_pitch():
    from core.processing_manager import middle_speaker_gender
    assert middle_speaker_gender({"f0": 210.0}) == ("female", 0.6)     # v1 would have said male
    assert middle_speaker_gender({"f0": 120.0}) == ("male", 0.6)
    assert middle_speaker_gender({"gender": "female", "gender_confidence": 0.8}) == ("female", 0.8)



def test_speaker_analysis_retries_a_busy_model_then_succeeds(monkeypatch, tmp_path):
    import sys
    import types as pytypes
    from core import gemini_speakers as GS

    # fake google.genai.types (the SDK is not installed here)
    fake_types = pytypes.SimpleNamespace(
        Part=type("Part", (), {"from_bytes": staticmethod(lambda data, mime_type: ("part", len(data)))}),
        GenerateContentConfig=lambda **kw: kw)
    fake_genai = pytypes.ModuleType("google.genai")
    fake_genai.types = fake_types
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)

    # the audio compression step is replaced: no ffmpeg needed for this test
    def fake_compress(audio, out_dir, control):
        p = Path(out_dir) / "x.mp3"
        p.write_bytes(b"x" * 2000)
        return p, "audio/mp3"
    monkeypatch.setattr(GS, "_compress", fake_compress)

    class Busy(Exception):
        pass

    calls = {"n": 0}

    class Models:
        def generate_content(self, model, contents, config=None):
            calls["n"] += 1
            if calls["n"] <= 2:
                raise Busy("503 UNAVAILABLE: high demand")
            payload = {"speakers": [{"id": "s1", "gender": "female"}],
                       "lines": [{"n": i, "s": "s1"} for i in range(3)]}
            return type("R", (), {"text": json.dumps(payload)})()

    client = type("C", (), {"models": Models()})()
    segs = [{"start": i, "end": i + 1, "text": "x"} for i in range(3)]
    waits, errors = [], []
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF" + b"\0" * 100)
    res = GS.analyze_speakers(client, wav, segs, models=["m1"], errors=errors, sleep=waits.append)
    assert res is not None and res["speakers"]["speaker_1"]["gender"] == "female"
    assert calls["n"] == 3 and sum(waits) == pytest.approx(5.0 + 15.0)
    assert errors == []
