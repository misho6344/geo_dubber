from core import gemini_speakers as G

SEGS = [{"start": 0, "end": 5, "text": "a"}, {"start": 5, "end": 6, "text": "b"},
        {"start": 6, "end": 20, "text": "c"}, {"start": 20, "end": 21, "text": "d"}]


def test_gender_decides_voice_and_order_by_speaking_time():
    data = {"speakers": [{"id": "s1", "gender": "female", "voice": "giorgi"}, {"id": "s2", "gender": "male"}],
            "lines": [{"n": 0, "s": "s1"}, {"n": 2, "s": "s2"}, {"n": 3, "s": "s2"}]}
    r = G.normalize(data, SEGS)
    assert r["speakers"]["speaker_1"] == {"gender": "male", "voice": "giorgi"}     # talks most -> speaker_1
    assert r["speakers"]["speaker_2"] == {"gender": "female", "voice": "nino"}
    assert r["lines"] == ["speaker_2", "speaker_2", "speaker_1", "speaker_1"]       # skipped line 1 inherits previous


def test_unusable_answers_are_rejected():
    assert G.normalize({}, SEGS) is None
    assert G.normalize({"speakers": [{"id": "s1", "gender": "robot"}], "lines": []}, SEGS) is None


def test_more_than_four_characters_are_folded():
    data = {"speakers": [{"id": f"s{i}", "gender": "male" if i % 2 else "female"} for i in range(1, 7)],
            "lines": [{"n": i, "s": f"s{i + 1}"} for i in range(4)]}
    r = G.normalize(data, SEGS)
    assert len(r["speakers"]) <= G.MAX_CHARACTERS


def test_only_two_edge_voices_when_variants_are_off():
    from core.georgian_tts import EdgeGeorgianBackend
    v = EdgeGeorgianBackend(variants=False).builtin_voices()
    assert [(x.name, x.gender) for x in v] == [("Giorgi", "male"), ("Nino", "female")]


def test_base_edge_voices_come_first_with_variants_on():
    from core.georgian_tts import EdgeGeorgianBackend
    v = EdgeGeorgianBackend(variants=True).builtin_voices()
    assert [x.id for x in v if x.builtin_params["pitch"] == "+0Hz"] == ["edge_ka_giorgi", "edge_ka_eka"]
