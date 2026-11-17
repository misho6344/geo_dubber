"""Translation to spoken Georgian.

Design (v2):
  * Lines are sent to Gemini in BATCHES (one request per ~16 lines, JSON in / JSON out). The old version sent one
    request per line, which hit the free-tier quota after a few hundred lines and silently fell back to a
    word-level web translator.
  * Every answer is validated: every line must come back, contain Georgian, contain no stray English words
    (unless the glossary keeps the word on purpose) and fit the time window. Failing lines are retried once in a
    small "make it shorter / fix it" batch.
  * Quota errors (HTTP 429 / RESOURCE_EXHAUSTED) back off and honour the server's retry delay. Models that are
    cooling down are skipped.
  * Names and terms come from a glossary and are given to the model in the prompt (the old glossary was applied
    to the English source text and was never filled in).
  * Each line records where its text came from (`mt_source`): gemini, gemini_retry, web or failed. Lines that
    could not be translated are left EMPTY (silence in the dub, listed as a warning) instead of being spoken as
    English by a Georgian voice.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

from config.settings import get_gemini_api_key

log = logging.getLogger(__name__)

_GEORGIAN_CHARS = re.compile(r"[Ⴀ-ჿⴀ-⴯]")
_LATIN_WORD = re.compile(r"[A-Za-z]{3,}")
_RETRY_DELAY = re.compile(r"(?:retry (?:in|after)|retryDelay[\"']?\s*[:=]\s*[\"']?)\s*([0-9]+(?:\.[0-9]+)?)\s*s", re.I)


# ----------------------------------------------------------------------------- helpers
def is_valid_georgian(text: str) -> bool:
    """True if the text contains Georgian script (Mkhedruli / Asomtavruli / Nuskhuri)."""
    return bool(text) and bool(_GEORGIAN_CHARS.search(text))


def estimate_speech_seconds(text: str) -> float:
    """Rough spoken duration of Georgian text (about 12 characters per second)."""
    if not text:
        return 0.0
    chars = sum(len(w) for w in text.strip().split())
    return max(0.5, chars / 12.0)


def parse_glossary(text: str) -> dict[str, str]:
    """Parse 'Source = Georgian' lines. Lines starting with # are comments. A line with the same
    source and target (e.g. 'Ben = Ben') means: keep this word in Latin letters."""
    out: dict[str, str] = {}
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        src, dst = (p.strip() for p in line.split("=", 1))
        if src and dst:
            out[src] = dst
    return out


def load_glossary(path: str | os.PathLike | None) -> dict[str, str]:
    if not path:
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            return parse_glossary(fh.read())
    except OSError as e:
        log.warning("Glossary file not readable (%s): %s", path, e)
        return {}


def _clean_georgian_output(text: str) -> str:
    """Extract the Georgian sentence from a single-line model answer (used by the one-line fallback)."""
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"^```[a-z]*\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip('"\'*` \n\r')
    for prefix in (
        r"^(translation|translated|georgian|here is|here's|output|result)\s*[:\-–]\s*",
        r"^(თარგმანი|ქართულად|აქ არის)\s*[:\-–]\s*",
    ):
        text = re.sub(prefix, "", text, flags=re.IGNORECASE).strip()
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    georgian_lines = [ln for ln in lines if is_valid_georgian(ln)]
    if georgian_lines:
        cand = max(georgian_lines, key=len)
        cand = re.sub(r"^[\*\s\"\'«»]+", "", cand)
        cand = re.sub(r"[\*\s\"\'«»]+$", "", cand)
        cand = re.split(r"\s{2,}|\s*\([^)]*\)\s*$", cand)[0].strip()
        return cand.strip()
    return text


def _web_google_translate(text: str, src_lang: str = "en") -> Optional[str]:
    """Last-resort machine translation via public Google endpoints (word-level quality; always flagged)."""
    ua = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    urls = [
        (f"https://clients5.google.com/translate_a/t?client=dict-chrome-ex&sl={src_lang}&tl=ka&q="
         + urllib.parse.quote(text)),
        (f"https://translate.googleapis.com/translate_a/single?client=webapp&sl={src_lang}&tl=ka&dt=t&q="
         + urllib.parse.quote(text)),
        f"https://api.mymemory.translated.net/get?q={urllib.parse.quote(text)}&langpair={src_lang}|ka",
    ]
    for i, url in enumerate(urls):
        try:
            req = urllib.request.Request(url, headers=ua)
            with urllib.request.urlopen(req, timeout=8) as res:
                data = json.loads(res.read().decode("utf-8"))
            if i == 0 and isinstance(data, list) and data:
                cand = data[0] if isinstance(data[0], str) else data[0][0]
            elif i == 1 and isinstance(data, list) and data and isinstance(data[0], list):
                cand = "".join(item[0] for item in data[0] if item and item[0])
            elif i == 2:
                cand = (data.get("responseData") or {}).get("translatedText", "")
            else:
                cand = ""
            if is_valid_georgian(cand):
                return cand.strip()
        except Exception as e:                       # network / format errors: try the next endpoint
            log.debug("web translate endpoint %d failed for '%s': %s", i, text[:40], e)
    return None


def _is_quota_error(e: Exception) -> bool:
    code = getattr(e, "code", None) or getattr(getattr(e, "response", None), "status_code", None)
    return code == 429 or "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e)[:40]


TRANSIENT_MARKERS = ("503", "504", "500", "502", "UNAVAILABLE", "DEADLINE_EXCEEDED", "INTERNAL", "high demand")


def _is_transient_error(e: Exception) -> bool:
    """Busy service or timeout: worth waiting and trying again instead of giving the lines to the web fallback."""
    code = getattr(e, "code", None)
    return code in (500, 502, 503, 504) or any(m in str(e)[:300] for m in TRANSIENT_MARKERS)


def _retry_delay(e: Exception) -> Optional[float]:
    m = _RETRY_DELAY.search(str(e))
    return float(m.group(1)) if m else None


class QuotaExhausted(RuntimeError):
    def __init__(self, delay: Optional[float]):
        super().__init__("Gemini quota exhausted")
        self.delay = delay


DAILY_COOLDOWN_S = 1800.0   # a server delay this long means the day's quota is used up, not a short burst


class DailyQuotaExhausted(RuntimeError):
    """Every model is over its daily quota. Waiting is pointless: the lines are left for a later run."""
    def __init__(self, seconds: float):
        super().__init__(f"Gemini daily quota used up (about {seconds / 3600:.1f} h until reset)")
        self.seconds = seconds


# ----------------------------------------------------------------------------- prompts
def build_batch_prompt(items: list[dict], *, src_lang: str, glossary: dict[str, str],
                       prev: str = "", nxt: str = "", shorter: bool = False,
                       style_note: str = "", memory: list[tuple[str, str]] | None = None, extra: str = "") -> str:
    """items: [{"i": int, "text": str, "gender": "male|female|unknown", "max_seconds": float}]"""
    terms = [f'- "{k}" -> "{v}"' for k, v in glossary.items() if k.strip() != v.strip()]
    keep = [k for k, v in glossary.items() if k.strip() == v.strip()]
    payload = [{"i": it["i"], "speaker": it.get("gender", "unknown"),
                "max_seconds": round(float(it.get("max_seconds") or 0), 1), "text": it["text"]} for it in items]
    rules = [
        f"You are an experienced translator who dubs films and series into Georgian (source language: {src_lang}).",
        "Translate every input line into natural, spoken Georgian for voice-over.",
        "Rules:",
        "1. Meaning first: keep every piece of information, tone, irony, emotion and humour. Never omit or add content.",
        "2. Profanity and insults stay at the same strength, using Georgian equivalents. Do not soften them.",
        "3. Idioms and slang: translate the intended meaning, not word by word.",
        "4. Speech must be understandable when heard once: short, simple, natural sentences. "
        "Fix transcription noise (broken words, repeated words) from the source sense.",
        "5. Timing: each line must be speakable within its max_seconds (Georgian speaks about 12 characters per second). "
        "If the literal translation is longer, shorten it while keeping the meaning.",
        "6. speaker = male/female/unknown. Use it only to choose gender-specific address words "
        "(e.g. masculine or feminine forms of address). Do not add gender where the source has none.",
        "7. Names and terms: use EXACTLY the Georgian forms listed below, every time.",
        "8. Non-speech sounds (laughs, coughs, grunts) may be written briefly, e.g. 'ჰა-ჰა'.",
        "9. The output must contain Georgian text only. No English words, no notes, no quotes around the whole line. "
        "Keep Latin letters only for the words that the glossary keeps unchanged.",
        "10. Context lines (if given) are only for understanding. Do NOT translate or output them.",
        "11. Idioms, jokes, puns and cultural references: find the Georgian expression with the same effect and "
        "register. If none exists, translate the meaning and set \"note\" to one of: idiom, joke, pun, culture, unclear.",
        "12. Consistency: if the memory below shows how a phrase or joke was already translated, keep the same wording.",
        "13. If a line is unclear or looks like a transcription error, give your best reading and set \"note\" to unclear.",
    ]
    if style_note:
        rules.append("Style of this film/series (follow it): " + style_note.strip())
    if shorter:
        rules.insert(2, "IMPORTANT: the previous attempt was too long or wrong. Make each line clearly shorter "
                        "than its max_seconds, and make sure there is no English left.")
    if terms:
        rules.append("Glossary (source -> Georgian):\n" + "\n".join(terms))
    if keep:
        rules.append("Keep these words in Latin letters: " + ", ".join(keep))
    ctx = ""
    if memory:
        ctx += ("Earlier lines of this film, with their Georgian translations (memory only, do NOT output them):\n"
                + "\n".join(f"- {en} => {ka}" for en, ka in memory[-12:]) + "\n")
    if prev or nxt:
        ctx = (f"Context before the batch (not to translate): {prev or '-'}\n"
               f"Context after the batch (not to translate): {nxt or '-'}\n")
    out_shape = '{"lines":[{"i":<same i as input>,"ka":"<Georgian line>","note":"<empty or idiom|joke|pun|culture|unclear>"}]}'
    return ("\n".join(rules) + "\n\n" + ctx + (extra + "\n\n" if extra else "") +
            f"Answer with JSON only, exactly in this shape, one entry per input line, same order: {out_shape}\n\n"
            "INPUT:\n" + json.dumps(payload, ensure_ascii=False, indent=1))


def build_single_prompt(text: str, *, src_lang: str, glossary: dict[str, str], max_seconds: Optional[float]) -> str:
    items = [{"i": 0, "text": text, "gender": "unknown", "max_seconds": max_seconds or 0}]
    return build_batch_prompt(items, src_lang=src_lang, glossary=glossary, shorter=True)


def parse_batch_answer(text: str) -> dict[int, str]:
    """Return {i: georgian_text}. Tolerates code fences and extra prose around the JSON object."""
    return {i: v[0] for i, v in parse_batch_answer_full(text).items()}


def parse_batch_answer_full(text: str) -> dict[int, tuple[str, str]]:
    """Return {i: (georgian_text, note)}; note is '' when the model had nothing to flag."""
    if not text:
        return {}
    cleaned = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text.strip())
    try:
        data = json.loads(cleaned)
    except ValueError:
        m = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not m:
            return {}
        try:
            data = json.loads(m.group(0))
        except ValueError:
            return {}
    rows = data.get("lines") if isinstance(data, dict) else None
    out: dict[int, tuple[str, str]] = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        try:
            i = int(row.get("i"))
        except (TypeError, ValueError):
            continue
        ka = row.get("ka")
        if isinstance(ka, str):
            out[i] = (ka.strip(), str(row.get("note") or "").strip().lower())
    return out


LENGTH_TOLERANCE = 1.25          # a line may be this much longer than its window before we ask for a shorter one
MIN_WINDOW_FOR_LENGTH = 1.5      # very short windows are not length-checked (a single word would always fail)


def line_problem(ka: str, max_seconds: float, allowed_latin: set[str], check_length: bool = False) -> Optional[str]:
    """Why a translated line is unacceptable, or None if it is fine."""
    if not ka or not ka.strip():
        return "empty"
    if not is_valid_georgian(ka):
        return "no Georgian text"
    stray = [w for w in _LATIN_WORD.findall(ka) if w not in allowed_latin]
    if stray:
        return "English left: " + ", ".join(stray[:3])
    if check_length and too_long(ka, max_seconds):
        return "too long for its window"
    return None


def too_long(ka: str, max_seconds: float) -> bool:
    return bool(max_seconds) and max_seconds >= MIN_WINDOW_FOR_LENGTH and \
        estimate_speech_seconds(ka) > max_seconds * LENGTH_TOLERANCE


# ----------------------------------------------------------------------------- translators
class Translator:
    """Base Translator interface."""
    last_sources: list[str] = []

    def translate(self, texts: list[str], src_lang: str,
                  contexts: list | None = None, max_seconds_list: list | None = None,
                  genders: list | None = None, glossary: dict | None = None) -> list[str]:
        raise NotImplementedError


class GeminiTranslator(Translator):
    """Batched Gemini translation with validation, quota back-off and a flagged web fallback."""

    CANDIDATE_MODELS = [
        "gemini-3.5-flash-lite",       # the model that actually answered in the field logs
        "gemini-3.5-flash",
        "gemini-3.8-flash",
    ]

    def __init__(self, model: str = "gemini-3.5-flash-lite", device: str = "cpu", beam: int = 1, batch: int = 16,
                 api_key: str = "", *, max_rounds: int = 6, max_workers: int = 2,
                 sleep: Callable[[float], None] = time.sleep, checkpoint: Optional[Callable[[], None]] = None,
                 web_fallback: bool = True, style_note: str = "", context_lines: int = 8,
                 sequential: bool = True):
        self.model_name = model if (model and "nllb" not in model.lower()) else self.CANDIDATE_MODELS[0]
        self.models = [self.model_name] + [m for m in self.CANDIDATE_MODELS if m != self.model_name]
        self.batch = max(1, int(batch or 16))
        self.api_key = (api_key or get_gemini_api_key()).strip()
        self.max_rounds = max_rounds
        self.max_workers = max(1, max_workers)
        self.sleep = sleep
        self.checkpoint = checkpoint or (lambda: None)
        self.web_fallback = web_fallback
        self.style_note = style_note
        self.context_lines = max(0, int(context_lines))
        self.sequential = bool(sequential)
        self.last_notes: list[str] = []
        self._cool_until: dict[str, float] = {}
        self._progress_callback = None
        self._progress_done = 0
        self._progress_total = 0
        self.client = None
        if not self.api_key:
            log.warning("NO GEMINI KEY: translation will use the word-level web fallback (much lower quality). "
                        "Set the GEMINI_API_KEY environment variable.")
        self.stats = {"gemini": 0, "gemini_retry": 0, "web": 0, "failed": 0, "requests": 0, "quota_waits": 0}
        try:
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY environment variable is not set")
            from google import genai                    # imported lazily so tests run without the SDK
            self.client = genai.Client(api_key=self.api_key, http_options={"timeout": 120000})  # milliseconds
        except Exception as e:
            log.warning("Failed to initialise Gemini client: %s", e)

    # -- low level ---------------------------------------------------------
    def _wait(self, seconds: float) -> None:
        """Sleep in 1-second steps so a cancel is noticed quickly."""
        remaining = max(0.0, float(seconds))
        while remaining > 0:
            self.checkpoint()                            # raises CancelledError when the user cancels
            if self._progress_callback:
                try:
                    self._progress_callback(
                        self._progress_done,
                        self._progress_total,
                        f"Gemini დროებით მიუწვდომელია — ხელახლა ცდა {int(remaining + 0.999)} წმ-ში",
                        None,
                    )
                except Exception:
                    log.exception("Translation progress callback failed during retry wait")
            step = min(1.0, remaining)
            self.sleep(step)
            remaining -= step

    def _generate(self, prompt: str) -> str:
        """One request, trying the models that are not cooling down. Raises QuotaExhausted or RuntimeError."""
        # google-genai accepts a plain dict as the request config (no SDK import needed here).
        cfg = {"response_mime_type": "application/json", "temperature": 0.3}
        now = time.time()
        available = [m for m in self.models if self._cool_until.get(m, 0) <= now]
        if not available:
            soonest = min(self._cool_until.get(m, now) for m in self.models) - now
            if soonest > DAILY_COOLDOWN_S:
                raise DailyQuotaExhausted(soonest)
        models = available or self.models[:1]
        last_quota: Optional[Exception] = None
        last_err: Optional[Exception] = None
        for model in models:
            self.stats["requests"] += 1
            try:
                resp = self.client.models.generate_content(model=model, contents=prompt, config=cfg)
                text = (getattr(resp, "text", "") or "").strip()
                if text:
                    return text
                last_err = RuntimeError(f"{model}: empty answer")
            except Exception as e:
                if _is_quota_error(e) or _is_transient_error(e):
                    delay = _retry_delay(e) or (20.0 if _is_quota_error(e) else 10.0)
                    self._cool_until[model] = time.time() + delay
                    last_quota = e
                    log.warning("Gemini (%s) %s, cooling down %.0fs", model,
                                "quota reached" if _is_quota_error(e) else "busy/unavailable", delay)
                else:
                    last_err = e
                    log.warning("Gemini (%s) failed: %s", model, str(e)[:200])
        if last_quota is not None:
            now = time.time()
            soonest = min(self._cool_until.get(m, now) for m in self.models) - now
            if soonest > DAILY_COOLDOWN_S:
                raise DailyQuotaExhausted(soonest)
            raise QuotaExhausted(min(_retry_delay(last_quota) or 20.0, 120.0))
        raise RuntimeError(str(last_err) if last_err else "no model answered")

    def _ask(self, prompt: str) -> Optional[str]:
        """Ask with quota back-off. Returns the raw answer or None if the quota never recovered."""
        if not self.client:
            return None
        for round_ in range(self.max_rounds):
            try:
                return self._generate(prompt)
            except QuotaExhausted as q:
                self.stats["quota_waits"] += 1
                wait = q.delay if q.delay else min(60.0, 5.0 * (2 ** round_))
                log.warning("All Gemini models over quota; waiting %.0fs (round %d/%d)", wait, round_ + 1, self.max_rounds)
                self._wait(wait)
            except DailyQuotaExhausted as d:
                self.stats["daily_quota"] = True
                log.warning("%s. The remaining lines are left empty: resume after the reset.", str(d))
                return None
            except RuntimeError as e:
                log.warning("Gemini request failed: %s", str(e)[:200])
                return None
        return None

    # -- batch level -------------------------------------------------------
    def _translate_batch(self, items: list[dict], src_lang: str, glossary: dict[str, str],
                         prev: str, nxt: str, allowed_latin: set[str], *, shorter: bool = False,
                         check_length: bool = False, memory: list[tuple[str, str]] | None = None
                         ) -> dict[int, tuple[str, str]]:
        """One request for `items`. Returns {i: (ka, note)} for the items that passed validation."""
        prompt = build_batch_prompt(items, src_lang=src_lang, glossary=glossary, prev=prev, nxt=nxt,
                                    shorter=shorter, style_note=self.style_note, memory=memory)
        answers = parse_batch_answer_full(self._ask(prompt) or "")
        ok: dict[int, tuple[str, str]] = {}
        for it in items:
            ka, note = answers.get(it["i"], ("", ""))
            if ka and line_problem(ka, it["max_seconds"], allowed_latin, check_length) is None:
                ok[it["i"]] = (ka, note)
        return ok

    def _translate_chunk(self, chunk: list[dict], src_lang: str, glossary: dict[str, str],
                         allowed_latin: set[str], memory: list[tuple[str, str]] | None = None
                         ) -> dict[int, tuple[str, str, str]]:
        """Translate one batch. Returns {i: (text, source_tag, note)}; empty text = failed."""
        prev, nxt = chunk[0].get("prev", ""), chunk[-1].get("nxt", "")
        result: dict[int, tuple[str, str, str]] = {}
        # Pass 1: the whole batch in one request.
        for i, (ka, note) in self._translate_batch(chunk, src_lang, glossary, prev, nxt, allowed_latin,
                                                   memory=memory).items():
            result[i] = (ka, "gemini", note)
        # Pass 2: lines that were missing, empty, or still contained English, in a repair request.
        missing = [it for it in chunk if it["i"] not in result]
        if missing:
            for i, (ka, note) in self._translate_batch(missing, src_lang, glossary, prev, nxt, allowed_latin,
                                                       shorter=True, memory=memory).items():
                result[i] = (ka, "gemini_retry", note)
        # Pass 3: lines that are clearly too long for their window get one "shorter" request.
        long_items = [it for it in chunk if it["i"] in result and too_long(result[it["i"]][0], it["max_seconds"])]
        if long_items:
            for i, (ka, note) in self._translate_batch(long_items, src_lang, glossary, prev, nxt, allowed_latin,
                                                       shorter=True, check_length=True, memory=memory).items():
                result[i] = (ka, "gemini_retry", note)
        # Pass 4: still missing -> accept Gemini's text if it at least contains Georgian letters.
        still = [it for it in chunk if it["i"] not in result]
        if still:
            answers = parse_batch_answer_full(self._ask(build_batch_prompt(
                still, src_lang=src_lang, glossary=glossary, prev=prev, nxt=nxt, shorter=True,
                style_note=self.style_note, memory=memory)) or "")
            for it in still:
                ka, note = answers.get(it["i"], ("", ""))
                if ka and is_valid_georgian(ka):
                    result[it["i"]] = (ka, "gemini_retry", note)
        for it in chunk:
            if it["i"] in result:
                continue
            web = _web_google_translate(it["text"], src_lang) if self.web_fallback else None
            if web:
                result[it["i"]] = (web, "web", "")
                log.warning("Fell back to web translate for: '%s'", it["text"][:40])
            else:
                result[it["i"]] = ("", "failed", "")
                log.error("Could not translate into Georgian: '%s'", it["text"][:80])
        return result

    def translate(self, texts: list[str], src_lang: str, contexts: list | None = None,
                  max_seconds_list: list | None = None, genders: list | None = None,
                  glossary: dict | None = None, progress_callback=None) -> list[str]:
        if not texts:
            return []
        glossary = dict(glossary or {})
        allowed_latin = {k for k, v in glossary.items() if k.strip() == v.strip()} | {
            w for k, v in glossary.items() for w in _LATIN_WORD.findall(v)}
        contexts = contexts or [("", "")] * len(texts)
        max_seconds_list = max_seconds_list or [None] * len(texts)
        genders = genders or ["unknown"] * len(texts)

        items = []
        for i, t in enumerate(texts):
            prev, nxt = contexts[i] if i < len(contexts) else ("", "")
            items.append({"i": i, "text": t, "gender": genders[i] if i < len(genders) else "unknown",
                          "max_seconds": float(max_seconds_list[i] or 0) or 0.0, "prev": prev, "nxt": nxt})
        chunks = [items[k:k + self.batch] for k in range(0, len(items), self.batch)]
        results: dict[int, tuple[str, str, str]] = {}
        completed = 0
        self._progress_callback = progress_callback
        self._progress_done = 0
        self._progress_total = len(texts)

        def notify(done: int, message: str, partial: dict | None = None) -> None:
            self._progress_done = done
            if progress_callback:
                try:
                    progress_callback(done, len(texts), message, partial)
                except Exception:
                    log.exception("Translation progress callback failed")

        def _job(chunk, memory=None):
            if self.client is None:
                out = {}
                for it in chunk:
                    web = _web_google_translate(it["text"], src_lang) if self.web_fallback else None
                    out[it["i"]] = (web or "", "web" if web else "failed", "")
                return out
            return self._translate_chunk(chunk, src_lang, glossary, allowed_latin, memory)

        def _memory_from(part: dict, chunk: list[dict]) -> list[tuple[str, str]]:
            # Pairs (English, Georgian) of the last accepted lines: lets later batches keep names,
            # running jokes and tone consistent with what was already said.
            pairs = [(it["text"], part[it["i"]][0]) for it in chunk if part.get(it["i"], ("",))[0]]
            return pairs[-self.context_lines:] if self.context_lines else []

        if self.sequential and self.client is not None and len(chunks) > 1:
            memory: list[tuple[str, str]] = []
            for chunk_index, chunk in enumerate(chunks, start=1):
                notify(
                    completed,
                    f"Gemini პარტია {chunk_index}/{len(chunks)} — მუშავდება {len(chunk)} ხაზი",
                )
                part = _job(chunk, memory)
                results.update(part)
                completed += len(chunk)
                notify(
                    completed,
                    f"თარგმნილია {completed}/{len(texts)} ხაზი · პარტია {chunk_index}/{len(chunks)}",
                    part,
                )
                memory = (memory + _memory_from(part, chunk))[-self.context_lines:] if self.context_lines else []
        else:
            workers = min(self.max_workers, max(1, len(chunks)))
            if workers == 1:
                for chunk_index, chunk in enumerate(chunks, start=1):
                    notify(completed, f"თარგმნის პარტია {chunk_index}/{len(chunks)} მუშავდება")
                    part = _job(chunk)
                    results.update(part)
                    completed += len(chunk)
                    notify(completed, f"თარგმნილია {completed}/{len(texts)} ხაზი · პარტია {chunk_index}/{len(chunks)}", part)
            else:
                from concurrent.futures import as_completed
                completed_by_index: dict[int, dict] = {}
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = {pool.submit(_job, chunk): i for i, chunk in enumerate(chunks)}
                    for future in as_completed(futures):
                        chunk_index = futures[future]
                        part = future.result()
                        completed_by_index[chunk_index] = part
                        results.update(part)
                        completed += len(chunks[chunk_index])
                        notify(
                            completed,
                            f"თარგმნილია {completed}/{len(texts)} ხაზი · დასრულდა პარტია {chunk_index + 1}/{len(chunks)}",
                            part,
                        )

        out, sources, notes = [], [], []
        for i in range(len(texts)):
            ka, src, note = results.get(i, ("", "failed", ""))
            out.append(ka)
            sources.append(src)
            notes.append(note)
            self.stats[src] = self.stats.get(src, 0) + 1
        self.last_sources = sources
        self.last_notes = notes
        self._progress_callback = None
        return out

    def shorten(self, items: list[dict], src_lang: str = "en") -> dict[int, str]:
        """Timing repair: rewrite Georgian lines that are too long for their window, keeping the meaning.
        items: [{"i", "source" (original English), "ka", "max_seconds"}]. Returns {i: shorter_ka} for the lines
        that passed validation; the others are not returned and keep their current text."""
        if not items or self.client is None:
            return {}
        prompt_items = [{"i": it["i"], "text": it["source"], "gender": "unknown", "max_seconds": it["max_seconds"]}
                        for it in items]
        current = {it["i"]: it["ka"] for it in items}
        extra = ("CURRENT GEORGIAN VERSIONS (too long; shorten these, keep the meaning):\n"
                 + json.dumps([{"i": i, "ka": current[i]} for i in current], ensure_ascii=False))
        prompt = build_batch_prompt(prompt_items, src_lang=src_lang, glossary={}, shorter=True,
                                    style_note=self.style_note, extra=extra)
        answers = parse_batch_answer_full(self._ask(prompt) or "")
        out = {}
        allowed = set()
        for it in items:
            ka = answers.get(it["i"], ("", ""))[0]
            if ka and line_problem(ka, it["max_seconds"], allowed, check_length=True) is None:
                out[it["i"]] = ka
        return out


def _strip_allowed(text: str, allowed: set[str]) -> str:
    return " ".join(w for w in text.split() if w.strip(".,!?;:") not in allowed)


# Backwards-compatible name used by processing_manager and the tests.
NLLBTranslator = GeminiTranslator


# ----------------------------------------------------------------------------- pipeline entry point
def translate_segments(
    translator: Translator,
    segments: list[dict],
    src_lang: str,
    batch_size: int = 16,
    glossary: dict | None = None,
    on_batch=None,
    control=None,
):
    """Translate ASR segments into spoken Georgian with neighbour context, speaker genders and the glossary.

    Sets seg["translation"] and seg["mt_source"] (gemini | gemini_retry | web | failed). Failed lines are
    left empty so the dub stays silent there, and they are translated again on the next run."""
    to_translate = [s for s in segments
                    if not s.get("translation") or (src_lang != "ka" and not is_valid_georgian(s.get("translation")))]
    if not to_translate:
        return
    if src_lang == "ka":                       # the source is already Georgian: speak it as it is
        for s in to_translate:
            s["translation"], s["mt_source"] = s.get("text", ""), "source"
        if on_batch:
            on_batch(len(to_translate), len(to_translate), "წყარო უკვე ქართულადაა — თარგმნა გამოტოვებულია")
        return

    ordered = sorted(segments, key=lambda s: s.get("start", 0))
    idx_of = {id(s): i for i, s in enumerate(ordered)}

    texts, contexts, max_secs, genders = [], [], [], []
    speaker_gender = {}
    for s in to_translate:
        texts.append(s["text"])
        i = idx_of.get(id(s))
        prev = ordered[i - 1]["text"] if i is not None and i > 0 else ""
        nxt = ordered[i + 1]["text"] if i is not None and i + 1 < len(ordered) else ""
        contexts.append((prev, nxt))
        win = max(0.4, float(s.get("end", 0) - s.get("start", 0)))
        max_secs.append(win * 1.15)
        genders.append(s.get("gender") or speaker_gender.get(s.get("speaker_id"), "unknown"))

    if hasattr(translator, "batch") and batch_size:
        translator.batch = max(1, int(batch_size))

    def _translation_progress(done: int, total: int, message: str, partial: dict | None = None) -> None:
        # Save each completed batch immediately. If the job is cancelled later, those
        # translations remain in the checkpoint instead of being lost with the batch list.
        if partial:
            for index, value in partial.items():
                if not 0 <= int(index) < len(to_translate):
                    continue
                ka, source, note = value
                segment = to_translate[int(index)]
                segment["translation"] = ka
                segment["mt_source"] = source
                if note:
                    segment["tr_note"] = note
        if on_batch:
            on_batch(done, total, message)

    if isinstance(translator, GeminiTranslator):
        translated = translator.translate(
            texts, src_lang, contexts=contexts, max_seconds_list=max_secs,
            genders=genders, glossary=glossary or {}, progress_callback=_translation_progress,
        )
    else:
        translated = translator.translate(texts, src_lang, contexts=contexts, max_seconds_list=max_secs,
                                           genders=genders, glossary=glossary or {})
    sources = getattr(translator, "last_sources", None) or ["gemini"] * len(to_translate)

    notes = getattr(translator, "last_notes", None) or [""] * len(to_translate)
    for seg, tr, src, note in zip(to_translate, translated, sources, notes):
        if control:
            control.checkpoint()
        seg["translation"] = tr
        seg["mt_source"] = src
        if note:
            seg["tr_note"] = note          # idiom | joke | pun | culture | unclear: shown in Review
    if on_batch:
        on_batch(len(to_translate), len(to_translate), f"თარგმნა დასრულდა: {len(to_translate)} ხაზი")
