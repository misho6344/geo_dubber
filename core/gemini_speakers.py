"""Let Gemini listen to the video's audio and decide:

* how many different characters speak,
* the gender of each character,
* which Georgian voice each one needs (only two exist: Giorgi = male, Nino = female),
* which character speaks each subtitle line.

Anything that goes wrong returns ``None`` so the pipeline falls back to local pitch clustering.
"""
from __future__ import annotations

import json
import logging
import re
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

from core.ffmpeg_utils import Control, run_ffmpeg
from config.settings import get_gemini_api_key

log = logging.getLogger(__name__)

MAX_CHARACTERS = 5
INLINE_LIMIT_BYTES = 14 * 1024 * 1024        # larger audio goes through the Files API
VOICE_BY_GENDER = {"male": "giorgi", "female": "nino"}
# Prefer the model with the largest free daily quota; keep lower-quota models as fallbacks.
MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.8-flash",
]
TRANSIENT_MARKERS = ("503", "UNAVAILABLE", "500", "INTERNAL", "502", "504", "DEADLINE")
RETRY_WAITS = (5.0, 15.0, 30.0)     # seconds between attempts on the same model when the service is busy
QUOTA_MARKERS = ("429", "RESOURCE_EXHAUSTED", "exceeded your current quota", "check your plan and billing")


def choose_gender(gemini_gender: str, gemini_confidence: float,
                  acoustic_gender: str, acoustic_confidence: float) -> tuple[str, str]:
    """Resolve Gemini/acoustic disagreement without blindly trusting either estimate.

    A clear acoustic estimate (>=0.65) wins over a conflicting Gemini label.
    Otherwise Gemini is used only when it reports reasonable confidence (>=0.60).
    Returns (gender, source).
    """
    gg = str(gemini_gender or "").strip().lower()
    ag = str(acoustic_gender or "").strip().lower()
    try:
        gc = max(0.0, min(1.0, float(gemini_confidence)))
    except (TypeError, ValueError):
        gc = 0.0
    try:
        ac = max(0.0, min(1.0, float(acoustic_confidence)))
    except (TypeError, ValueError):
        ac = 0.0

    if ag in {"male", "female"} and ac >= 0.65 and (gg != ag or gc < 0.60):
        return ag, "audio_pitch"
    if gg in {"male", "female"} and gc >= 0.60:
        return gg, "gemini"
    if ag in {"male", "female"}:
        return ag, "audio_pitch"
    return "unknown", "uncertain"


def pitch_contradicts(gender: str, median_f0: float) -> bool:
    """True when a measured median pitch is clearly outside the usual range for the stated gender.
    Adult male voices are mostly 85-180 Hz, female 165-255 Hz; only clear outliers are flagged."""
    if gender == "male":
        return median_f0 >= 200.0
    if gender == "female":
        return median_f0 <= 140.0
    return False


def _wait(seconds: float, sleep: Callable[[float], None], control: Optional[Control]) -> None:
    left = float(seconds)
    while left > 0:
        if control is not None:
            control.checkpoint()          # cancel is honoured while waiting
        step = min(1.0, left)
        sleep(step)
        left -= step


def make_client(api_key: str | None = None, timeout_ms: int = 600000):
    """Create a Gemini client using an explicit key or GEMINI_API_KEY from the environment."""
    resolved_key = (api_key or get_gemini_api_key()).strip()
    if not resolved_key:
        log.warning("GEMINI_API_KEY is not configured; Gemini speaker analysis is unavailable.")
        return None
    try:
        from google import genai
        return genai.Client(
            api_key=resolved_key,
            http_options={"timeout": timeout_ms}
        )
    except Exception as e:
        log.warning(
            "Could not create Gemini client for speaker analysis: %s", e
        )
        return None


def _ts(sec: float) -> str:
    m, s = divmod(max(0.0, float(sec)), 60)
    return f"{int(m):02d}:{s:04.1f}"


def build_prompt(segments: list[dict], want: int = 0, forced_genders: Optional[list[str]] = None) -> str:
    lines = []
    for i, s in enumerate(segments):
        text = " ".join(str(s.get("text", "")).split())[:160]
        lines.append(f"{i} | {_ts(s['start'])}-{_ts(s['end'])} | {text}")
    hint = ""
    if want and want > 0:
        hint += f"The user says there are exactly {want} different speaker(s). Use exactly that number.\n"
    if forced_genders:
        hint += ("The user says the main speakers, from the one who talks most to the one who talks least, are: "
                 + ", ".join(forced_genders) + ".\n")
    return (
        "Analyze this Georgian-language audio recording. Listen to the whole recording before answering.\n"
        "1. Count the distinct speakers, from 1 to 5. Do not count music, crowd noise, or background sounds.\n"
        "2. For each speaker, estimate perceived voice gender from the audio only: \"male\" or \"female\". Base this on vocal characteristics such as pitch and timbre, compared with the other speakers in the recording. Do not infer gender from names, roles, text, appearance, or stereotypes.\n"
        "3. Give a confidence score from 0.0 to 1.0 for each gender estimate. If you are uncertain, still choose the closest category and give a score below 0.6.\n"
        "4. For each numbered line below, identify which speaker says it. Use vocal identity, not gender alone, so the same person stays consistent throughout. If two different people sound alike, treat them as separate speakers.\n"
        "5. Assign each speaker a Georgian voice: \"giorgi\" for a male-sounding voice, \"nino\" for a female-sounding voice.\n"
        "%s"
        "Name speakers s1, s2, ... The same person must always keep the same name.\n"
        "Answer with JSON only, in exactly this shape:\n"
        "{\"speakers\":[{\"id\":\"s1\",\"gender\":\"male\",\"gender_confidence\":0.85,\"voice\":\"giorgi\"}],"
        "\"lines\":[{\"n\":0,\"s\":\"s1\"}]}\n"
        "Every transcript line number must appear once in \"lines\".\n\n"
        "TRANSCRIPT (number | time | text):\n%s\n"
    ) % (hint, "\n".join(lines))


def _parse_json(text: str) -> Optional[dict]:
    if not text:
        return None
    text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text.strip())
    try:
        data = json.loads(text)
    except ValueError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except ValueError:
            return None
    return data if isinstance(data, dict) else None


def normalize(data: dict, segments: list[dict]) -> Optional[dict]:
    """Validate Gemini's answer and convert it to speaker_1..speaker_N (ordered by speaking time).

    Returns {"speakers": {"speaker_1": {"gender", "voice"}}, "lines": [speaker_id per segment index]}.
    """
    raw_spk = data.get("speakers")
    raw_lines = data.get("lines")
    if not isinstance(raw_spk, list) or not raw_spk or not isinstance(raw_lines, list):
        return None
    gender_of: dict[str, tuple[str, float]] = {}
    for sp in raw_spk:
        if not isinstance(sp, dict):
            continue
        sid, g = str(sp.get("id", "")).strip(), str(sp.get("gender", "")).strip().lower()
        if sid and g in ("male", "female"):
            try:
                confidence = max(0.0, min(1.0, float(sp.get("gender_confidence", sp.get("confidence", 0.7)))))
            except (TypeError, ValueError):
                confidence = 0.7
            gender_of[sid] = (g, confidence)
    if not gender_of:
        return None
    gender_labels = {sid: value[0] for sid, value in gender_of.items()}

    per_line: list[Optional[str]] = [None] * len(segments)
    for ln in raw_lines:
        if not isinstance(ln, dict):
            continue
        try:
            n = int(ln.get("n"))
        except (TypeError, ValueError):
            continue
        sid = str(ln.get("s", "")).strip()
        if 0 <= n < len(segments) and sid in gender_of:
            per_line[n] = sid
    # Lines Gemini skipped: continue with the previous speaker (or the next known one).
    last = None
    for i, v in enumerate(per_line):
        if v is None:
            per_line[i] = last
        else:
            last = v
    nxt = None
    for i in range(len(per_line) - 1, -1, -1):
        if per_line[i] is None:
            per_line[i] = nxt
        else:
            nxt = per_line[i]
    if not any(per_line):
        return None

    secs: dict[str, float] = {}
    for sid, seg in zip(per_line, segments):
        secs[sid] = secs.get(sid, 0.0) + max(0.0, seg["end"] - seg["start"])
    order = sorted(secs, key=lambda k: -secs[k])

    # More than MAX_CHARACTERS: fold the quietest into the loudest speaker of the same gender.
    keep = order[:MAX_CHARACTERS]
    remap = {}
    for sid in order:
        if sid in keep:
            remap[sid] = sid
        else:
            same = [k for k in keep if gender_labels[k] == gender_labels[sid]]
            remap[sid] = (same or keep)[0]
    per_line = [remap[v] for v in per_line]

    final_ids = {old: f"speaker_{i + 1}" for i, old in enumerate(keep)}
    speakers = {}
    for old, new in final_ids.items():
        g, confidence = gender_of[old]
        speakers[new] = {"gender": g, "gender_confidence": confidence, "voice": VOICE_BY_GENDER[g]}     # gender always decides the voice
    return {"speakers": speakers, "lines": [final_ids[v] for v in per_line]}


def _compress(audio: Path, out_dir: Path, control: Optional[Control]) -> tuple[Path, str]:
    """Small mono file for Gemini. Tries mp3, then aac, then flac (depends on the bundled ffmpeg)."""
    last = None
    for name, mime, codec in (("speakers.mp3", "audio/mp3", []), ("speakers.aac", "audio/aac", ["-c:a", "aac"]),
                              ("speakers.flac", "audio/flac", ["-c:a", "flac"])):
        out = out_dir / name
        try:
            run_ffmpeg(["-i", str(audio), "-vn", "-ac", "1", "-ar", "16000", *codec,
                        *(["-b:a", "32k"] if name != "speakers.flac" else []), str(out)], control=control)
            if out.exists() and out.stat().st_size > 1000:
                return out, mime
        except Exception as e:
            last = e
    raise RuntimeError(f"ffmpeg could not compress the audio: {last}")


def analyze_speakers(client, audio_path: str | Path, segments: list[dict], *, want: int = 0,
                     forced_genders: Optional[list[str]] = None, models: Optional[list[str]] = None,
                     control: Optional[Control] = None, errors: Optional[list] = None,
                     sleep: Callable[[float], None] = time.sleep) -> Optional[dict]:
    """Ask Gemini. Failure reasons are appended to `errors` (if given). `client` is a google.genai Client. Returns normalize()'s result or None."""
    if client is None or not segments:
        if errors is not None:
            errors.append("no Gemini client" if client is None else "no speech lines")
        return None
    try:
        from google.genai import types
    except Exception as e:                                   # pragma: no cover - SDK missing
        log.warning("google-genai not available for speaker analysis: %s", e)
        return None

    uploaded = None
    try:
        with tempfile.TemporaryDirectory() as td:
            mp3, mime = _compress(Path(audio_path), Path(td), control)
            data = mp3.read_bytes()
            if len(data) <= INLINE_LIMIT_BYTES:
                audio_part = types.Part.from_bytes(data=data, mime_type=mime)
            else:
                uploaded = client.files.upload(file=str(mp3), config={"mime_type": mime})
                import time
                for _ in range(60):
                    state = str(getattr(getattr(uploaded, "state", None), "name", getattr(uploaded, "state", "")))
                    if state in ("", "ACTIVE"):
                        break
                    if state == "FAILED":
                        raise RuntimeError("Gemini could not process the audio file")
                    time.sleep(2)
                    uploaded = client.files.get(name=uploaded.name)
                audio_part = uploaded
        prompt = build_prompt(segments, want, forced_genders)
        cfg = types.GenerateContentConfig(response_mime_type="application/json", temperature=0)
        for model in (models or MODELS):
            # Busy or over-quota services are retried on the same model before moving on to the next one.
            for attempt in range(len(RETRY_WAITS) + 1):
                try:
                    resp = client.models.generate_content(model=model, contents=[audio_part, prompt], config=cfg)
                    result = normalize(_parse_json(getattr(resp, "text", "") or "") or {}, segments)
                    if result:
                        return result
                    log.warning("Gemini (%s) returned an unusable speaker analysis", model)
                    if errors is not None:
                        errors.append(f"{model}: unusable answer: {(getattr(resp, 'text', '') or '')[:120]!r}")
                    break
                except Exception as e:
                    msg = str(e)
                    if any(marker.lower() in msg.lower() for marker in QUOTA_MARKERS):
                        # A 429 is project/model quota pressure. Do not retry it four times
                        # or probe other models: that only burns more quota on the same project.
                        log.warning("Gemini (%s) speaker analysis stopped after quota response: %s", model, msg[:200])
                        if errors is not None:
                            errors.append(f"{model}: {msg[:160]}")
                        return None
                    transient = any(m in msg for m in TRANSIENT_MARKERS)
                    log.warning("Gemini (%s) speaker analysis failed (attempt %d): %s", model, attempt + 1, msg[:200])
                    if transient and attempt < len(RETRY_WAITS):
                        _wait(RETRY_WAITS[attempt], sleep, control)
                        continue
                    if errors is not None:
                        errors.append(f"{model}: {msg[:160]}")
                    break
    except Exception as e:
        log.warning("Gemini speaker analysis failed: %s", e)
        if errors is not None:
            errors.append(str(e)[:200])
    finally:
        if uploaded is not None:
            try:
                client.files.delete(name=uploaded.name)
            except Exception:
                pass
    return None
