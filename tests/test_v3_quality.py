"""Tests for v3: context-aware translation, timing fit and automatic quality control."""
import json

import numpy as np
import pytest

from core import quality as Q
from core import translation as T

GE = "მიკი ამბობს ხმამაღლა"


class _Resp:
    def __init__(self, text):
        self.text = text


class FakeModels:
    def __init__(self, responder):
        self.responder, self.calls = responder, []

    def generate_content(self, model, contents, config=None):
        self.calls.append(contents)
        return self.responder(contents, len(self.calls))


def _items(prompt):
    return json.loads(prompt.split("INPUT:\n", 1)[1])


def make(responder, **kw):
    t = T.GeminiTranslator(model="gemini-3.5-flash-lite", api_key="k", batch=2, sleep=lambda s: None,
                           web_fallback=False, **kw)
    t.client = type("C", (), {"models": FakeModels(responder)})()
    return t


def plain(prompt, n):
    return _Resp(json.dumps({"lines": [{"i": it["i"], "ka": GE, "note": ""} for it in _items(prompt)]}, ensure_ascii=False))


# ------------------------------------------------------------------ quality.py
def test_trim_silence_keeps_margins_and_reports_removed_time():
    sr = 1000
    x = np.concatenate([np.zeros(500), np.ones(200) * 0.5, np.zeros(300)]).astype(np.float32)
    y, start, end = Q.trim_silence(x, sr, thresh_db=-42, keep_s=0.02)
    assert len(y) == 200 + 2 * 20
    assert start == pytest.approx(0.48) and end == pytest.approx(0.28)


def test_trim_silence_leaves_all_silent_clip_alone():
    y, a, b = Q.trim_silence(np.zeros(100, dtype=np.float32), 1000)
    assert len(y) == 100 and a == 0 and b == 0


def test_fit_needed_uses_the_same_rule_as_placement():
    # window 2 s, speed-up 1.3 and spill 1.5 s -> limit 4.1 s
    assert Q.fit_needed({1: 4.0, 2: 4.2}, {1: 2.0, 2: 2.0}, 1.3, 1.5) == [2]


def test_clip_problems_detects_clipping_silence_and_truncation():
    clips = {1: {"peak": 1.0, "rms_db": -20, "duration": 2},
             2: {"peak": 0.2, "rms_db": -70, "duration": 2},
             3: {"peak": 0.3, "rms_db": -20, "duration": 0.5}}
    texts = {1: "a", 2: GE, 3: "ყველაფერი რაც უნდა თქვა აქ"}
    kinds = {(i["id"], i["kind"]) for i in Q.clip_problems(clips, texts)}
    assert kinds == {(1, "clipping"), (2, "silent"), (3, "too_short")}


def test_timeline_flags_same_speaker_overlap_and_long_spill():
    p = [{"id": 1, "speaker_id": "a", "start": 0, "end": 3},
         {"id": 2, "speaker_id": "a", "start": 2.5, "end": 4},      # same speaker overlaps itself: always a bug
         {"id": 3, "speaker_id": "b", "start": 3.9, "end": 5},      # other speaker, small overlap: fine
         {"id": 4, "speaker_id": "b", "start": 5.0, "end": 6}]
    kinds = [(i["id"], i["kind"]) for i in Q.timeline_problems(p)]
    assert (2, "same_speaker_overlap") not in kinds                # overlap is reported on the earlier line
    assert (1, "same_speaker_overlap") in kinds
    assert not any(k == "spill_into_other" for _, k in kinds)


# ------------------------------------------------------------------ translation.py
def test_style_note_and_memory_reach_the_prompt():
    t = make(plain, style_note="dry sci-fi comedy", context_lines=8, sequential=False)
    t.translate(["Hello there"], "en", max_seconds_list=[5])
    prompt = t.client.models.calls[0]
    assert "dry sci-fi comedy" in prompt


def test_sequential_batches_see_earlier_translations():
    def responder(prompt, n):
        items = _items(prompt)
        return _Resp(json.dumps({"lines": [{"i": it["i"], "ka": f"{GE} ნომერი{it['i']}", "note": ""} for it in items]},
                                ensure_ascii=False))
    t = make(responder, context_lines=8, sequential=True)
    t.translate([f"line {k}" for k in range(4)], "en", max_seconds_list=[10] * 4)   # batch size 2 -> 2 requests
    second = t.client.models.calls[1]
    assert "Earlier lines of this film" in second and "line 0 =>" in second
    assert len(t.client.models.calls) == 2                       # no repair requests were needed


def test_notes_are_recorded_per_line():
    def responder(prompt, n):
        items = _items(prompt)
        lines = [{"i": it["i"], "ka": GE, "note": "idiom" if it["i"] == 1 else ""} for it in items]
        return _Resp(json.dumps({"lines": lines}, ensure_ascii=False))
    t = make(responder)
    t.translate(["a", "b"], "en", max_seconds_list=[5, 5])
    assert t.last_notes == ["", "idiom"]


def test_shorten_returns_only_lines_that_now_fit():
    def responder(prompt, n):
        items = _items(prompt)
        return _Resp(json.dumps({"lines": [{"i": it["i"], "ka": GE, "note": ""} for it in items]}, ensure_ascii=False))
    t = make(responder)
    out = t.shorten([{"i": 3, "source": "a long sentence", "ka": " ".join(["სიტყვა"] * 30), "max_seconds": 3.0}])
    assert out == {3: GE}
    assert "CURRENT GEORGIAN VERSIONS" in t.client.models.calls[0]


def test_shorten_rejects_an_answer_that_is_still_too_long():
    def responder(prompt, n):
        long_ka = " ".join(["სიტყვა"] * 30)
        return _Resp(json.dumps({"lines": [{"i": 3, "ka": long_ka}]}, ensure_ascii=False))
    t = make(responder)
    assert t.shorten([{"i": 3, "source": "x", "ka": "y", "max_seconds": 2.0}]) == {}


def test_translate_segments_stores_notes():
    segs = [{"id": 0, "start": 0, "end": 3, "text": "a"}, {"id": 1, "start": 3, "end": 6, "text": "b"}]

    def responder(prompt, n):
        items = _items(prompt)
        return _Resp(json.dumps({"lines": [{"i": it["i"], "ka": GE, "note": "joke" if it["i"] == 0 else ""}
                                           for it in items]}, ensure_ascii=False))
    T.translate_segments(make(responder), segs, "en", 2)
    assert segs[0].get("tr_note") == "joke" and "tr_note" not in segs[1]


def test_busy_service_503_is_retried_not_sent_to_web(monkeypatch):
    calls = {"n": 0}

    class Busy(Exception):
        pass

    def responder(prompt, n):
        calls["n"] += 1
        if calls["n"] <= len(T.GeminiTranslator.CANDIDATE_MODELS):   # every model is busy on the first round
            raise Busy("503 UNAVAILABLE: This model is currently experiencing high demand.")
        items = _items(prompt)
        return _Resp(json.dumps({"lines": [{"i": it["i"], "ka": GE, "note": ""} for it in items]}, ensure_ascii=False))

    waits = []
    t = make(responder)
    t.sleep = waits.append
    monkeypatch.setattr(T, "_web_google_translate", lambda *a, **k: pytest.fail("must not use web fallback"))
    out = t.translate(["hello", "there"], "en", max_seconds_list=[5, 5])
    assert out == [GE, GE]
    assert t.last_sources == ["gemini", "gemini"]
    assert sum(waits) >= 9.0                      # it waited before retrying
