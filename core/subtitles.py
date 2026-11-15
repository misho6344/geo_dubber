"""SRT/VTT/TXT subtitle readers and writers."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


def fmt_ts(t: float) -> str:
    t = max(0.0, t)
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _parse_ts(s: str) -> Optional[float]:
    """Parse 00:01:02,345 or 00:01:02.345 or 01:02,345 into seconds."""
    s = s.strip().replace(",", ".")
    m = re.match(r"^(?:(\d+):)?(\d{1,2}):(\d{1,2}(?:\.\d+)?)$", s)
    if not m:
        return None
    h = int(m.group(1) or 0)
    mi = int(m.group(2))
    sec = float(m.group(3))
    return h * 3600 + mi * 60 + sec


_SPEAKER_RE = re.compile(
    r"^\s*(?:\[?\s*(speaker[_\s-]?\d+|spk[_\s-]?\d+|character[_\s-]?\d+|პერსონაჟი\s*\d+|კაცი|ქალი|male|female)\s*\]?)\s*[:\-–]?\s*",
    re.IGNORECASE,
)


def _split_speaker(text: str) -> tuple[Optional[str], str]:
    """Extract optional leading speaker tag; return (speaker_id_or_None, clean_text)."""
    text = text.strip()
    m = _SPEAKER_RE.match(text)
    if not m:
        return None, text
    raw = m.group(1).strip().lower().replace(" ", "_").replace("-", "_")
    rest = text[m.end():].strip()
    if raw in ("კაცი", "male"):
        return "male", rest
    if raw in ("ქალი", "female"):
        return "female", rest
    # speaker_1 / spk1 / character_2 → speaker_N
    digits = re.findall(r"\d+", raw)
    if digits:
        return f"speaker_{int(digits[0])}", rest
    return raw, rest


def read_srt(path: str | Path) -> list[dict]:
    """Parse an .srt file into [{id, start, end, text, speaker_id?}]."""
    raw = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    blocks = re.split(r"\n\s*\n", raw.strip())
    out: list[dict] = []
    sid = 0
    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        # Optional index line
        if re.match(r"^\d+$", lines[0]) and len(lines) >= 2:
            lines = lines[1:]
        if not lines:
            continue
        # Timecode line
        tc = re.match(
            r"^(\d{1,2}:\d{2}:\d{2}[,\.]\d{1,3}|\d{1,2}:\d{2}[,\.]\d{1,3})\s*-->\s*"
            r"(\d{1,2}:\d{2}:\d{2}[,\.]\d{1,3}|\d{1,2}:\d{2}[,\.]\d{1,3})",
            lines[0],
        )
        if not tc:
            continue
        start, end = _parse_ts(tc.group(1)), _parse_ts(tc.group(2))
        if start is None or end is None or end <= start:
            continue
        body = " ".join(lines[1:]).strip()
        if not body:
            continue
        # Strip simple HTML tags from some exports
        body = re.sub(r"<[^>]+>", "", body).strip()
        spk, body = _split_speaker(body)
        if not body:
            continue
        sid += 1
        seg = {"id": sid, "start": float(start), "end": float(end), "text": body}
        if spk:
            seg["speaker_id"] = spk
        out.append(seg)
    return out


def read_vtt(path: str | Path) -> list[dict]:
    """Parse a .vtt file (WebVTT) into segments."""
    raw = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    # Drop header
    raw = re.sub(r"^WEBVTT[^\n]*\n", "", raw, count=1, flags=re.IGNORECASE)
    blocks = re.split(r"\n\s*\n", raw.strip())
    out: list[dict] = []
    sid = 0
    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip() and not ln.strip().startswith("NOTE")]
        if not lines:
            continue
        if re.match(r"^\d+$", lines[0]) and len(lines) >= 2:
            lines = lines[1:]
        tc = re.match(
            r"^(\d{1,2}:\d{2}:\d{2}\.\d{1,3}|\d{1,2}:\d{2}\.\d{1,3})\s*-->\s*"
            r"(\d{1,2}:\d{2}:\d{2}\.\d{1,3}|\d{1,2}:\d{2}\.\d{1,3})",
            lines[0],
        )
        if not tc:
            continue
        start, end = _parse_ts(tc.group(1)), _parse_ts(tc.group(2))
        if start is None or end is None or end <= start:
            continue
        body = " ".join(lines[1:]).strip()
        body = re.sub(r"<[^>]+>", "", body).strip()
        spk, body = _split_speaker(body)
        if not body:
            continue
        sid += 1
        seg = {"id": sid, "start": float(start), "end": float(end), "text": body}
        if spk:
            seg["speaker_id"] = spk
        out.append(seg)
    return out


def read_timed_txt(path: str | Path) -> list[dict]:
    """Parse lines like: 00:01:02,100  text  or  00:01:02,100 --> 00:01:05,000  text"""
    raw = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    out: list[dict] = []
    sid = 0
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    for i, ln in enumerate(lines):
        # range form
        m = re.match(
            r"^(\d{1,2}:\d{2}:\d{2}[,\.]\d{1,3}|\d{1,2}:\d{2}[,\.]\d{1,3})\s*-->\s*"
            r"(\d{1,2}:\d{2}:\d{2}[,\.]\d{1,3}|\d{1,2}:\d{2}[,\.]\d{1,3})\s+(.+)$",
            ln,
        )
        if m:
            start, end, body = _parse_ts(m.group(1)), _parse_ts(m.group(2)), m.group(3).strip()
        else:
            m = re.match(
                r"^(\d{1,2}:\d{2}:\d{2}[,\.]\d{1,3}|\d{1,2}:\d{2}[,\.]\d{1,3})\s+(.+)$",
                ln,
            )
            if not m:
                continue
            start = _parse_ts(m.group(1))
            body = m.group(2).strip()
            # end = start of next timed line, or +3s
            end = None
            for j in range(i + 1, len(lines)):
                m2 = re.match(
                    r"^(\d{1,2}:\d{2}:\d{2}[,\.]\d{1,3}|\d{1,2}:\d{2}[,\.]\d{1,3})",
                    lines[j],
                )
                if m2:
                    end = _parse_ts(m2.group(1))
                    break
            if end is None and start is not None:
                end = start + 3.0
        if start is None or end is None or end <= start or not body:
            continue
        spk, body = _split_speaker(body)
        if not body:
            continue
        sid += 1
        seg = {"id": sid, "start": float(start), "end": float(end), "text": body}
        if spk:
            seg["speaker_id"] = spk
        out.append(seg)
    return out


def read_subtitle_file(path: str | Path) -> list[dict]:
    """Auto-detect format by extension and parse. Raises ValueError if empty/unsupported."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    ext = path.suffix.lower()
    if ext == ".srt":
        segs = read_srt(path)
    elif ext in (".vtt", ".webvtt"):
        segs = read_vtt(path)
    elif ext in (".txt", ".csv"):
        # try timed txt first; if empty, try as srt-like
        segs = read_timed_txt(path)
        if not segs:
            segs = read_srt(path)
    else:
        # try all parsers
        segs = read_srt(path) or read_vtt(path) or read_timed_txt(path)
    if not segs:
        raise ValueError(f"No timed lines found in {path.name}")
    return segs


def write_srt(segments: list[dict], path: str | Path, key: str = "text") -> Path:
    lines = []
    n = 0
    for sg in segments:
        txt = (sg.get(key) or "").strip()
        if not txt:
            continue
        n += 1
        end = max(sg["end"], sg["start"] + 0.4)
        lines += [str(n), f"{fmt_ts(sg['start'])} --> {fmt_ts(end)}", txt, ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    return Path(path)


def write_txt(segments: list[dict], path: str | Path, key: str = "translation", with_speaker: bool = True) -> Path:
    out = []
    for sg in segments:
        txt = (sg.get(key) or "").strip()
        if txt:
            spk = f"[{sg.get('speaker_id', '')}] " if with_speaker else ""
            out.append(f"{fmt_ts(sg['start'])}  {spk}{txt}")
    Path(path).write_text("\n".join(out), encoding="utf-8")
    return Path(path)
