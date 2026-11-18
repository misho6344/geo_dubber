from core.subtitles import read_subtitle_file


def test_txt_file_with_srt_blocks_is_parsed_as_srt(tmp_path):
    path = tmp_path / "captions.txt"
    path.write_text(
        "1\n"
        "00:00:01,000 --> 00:00:02,000\n"
        "გამარჯობა\n"
        "\n"
        "2\n"
        "00:00:02,100 --> 00:00:03,000\n"
        "როგორ ხარ?\n",
        encoding="utf-8",
    )

    segments = read_subtitle_file(path)

    assert len(segments) == 2
    assert [segment["text"] for segment in segments] == ["გამარჯობა", "როგორ ხარ?"]
    assert segments[0]["start"] == 1.0
    assert segments[0]["end"] == 2.0


def test_malformed_srt_arrow_line_is_not_treated_as_timed_txt(tmp_path):
    path = tmp_path / "captions.txt"
    path.write_text(
        "1\n"
        "00:00:01,000 --> 00:00:02,000\n"
        "ქართული ტექსტი\n"
        "\n"
        "2\n"
        "not-a-valid-time --> 00:00:04,000\n"
        "ეს არ უნდა გახდეს სუბტიტრი\n",
        encoding="utf-8",
    )

    segments = read_subtitle_file(path)

    assert len(segments) == 1
    assert segments[0]["text"] == "ქართული ტექსტი"
