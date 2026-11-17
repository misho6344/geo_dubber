"""Private web API for running the existing GeoDubber pipeline online.

The desktop PySide6 UI is deliberately not imported here. Jobs execute in a single
background thread, use the same checkpointed Pipeline, and store uploads/projects
on GEODUBBER_HOME (attach persistent storage in production).
"""
from __future__ import annotations

import hmac
import json
import logging
import os
import re
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

LOG = logging.getLogger("geodubber.online")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

DATA_ROOT = Path(os.getenv("GEODUBBER_HOME", "/var/data/geodubber")).expanduser()
PROJECTS_ROOT = DATA_ROOT / "projects"
CACHE_ROOT = DATA_ROOT / "cache"
MAX_UPLOAD_BYTES = int(os.getenv("GEODUBBER_MAX_UPLOAD_BYTES", str(8 * 1024**3)))
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
ALLOWED_SCRIPT_EXTENSIONS = {".srt", ".vtt", ".txt"}
JOB_ID_RE = re.compile(r"^[a-f0-9]{24}$")
SPEAKER_ID_RE = re.compile(r"^speaker_[1-4]$")

app = FastAPI(title="GeoDubber API", docs_url=None, redoc_url=None)
_lock = threading.RLock()
_jobs: dict[str, dict[str, Any]] = {}
_controls: dict[str, Any] = {}
_review_events: dict[str, threading.Event] = {}
_pipelines: dict[str, Any] = {}
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="geodubber-job")


def _now() -> float:
    return round(time.time(), 3)


def _job_dir(job_id: str) -> Path:
    if not JOB_ID_RE.fullmatch(job_id):
        raise HTTPException(404, "პროექტი ვერ მოიძებნა.")
    return PROJECTS_ROOT / job_id


def _write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _public_job(job: dict) -> dict:
    return {
        "id": job["id"],
        "name": job.get("name", "GeoDubber project"),
        "status": job.get("status", "created"),
        "createdAt": job.get("createdAt"),
        "updatedAt": job.get("updatedAt"),
        "fileName": job.get("fileName", ""),
        "fileSize": int(job.get("fileSize") or 0),
        "progress": job.get("progress") or {},
        "message": job.get("message", ""),
        "warnings": job.get("warnings") or [],
        "error": job.get("error", ""),
        "outputs": job.get("outputs") or {},
        "reviewRequired": job.get("status") == "review_required",
        "canResume": job.get("status") in {"paused", "cancelled", "failed"},
    }


def _save_job(job_id: str) -> None:
    job = _jobs.get(job_id)
    if not job:
        return
    job["updatedAt"] = _now()
    safe = {k: v for k, v in job.items() if not k.startswith("_")}
    _write_json_atomic(_job_dir(job_id) / "job.json", safe)


def _require_api_key(authorization: str | None) -> None:
    expected = os.getenv("GEODUBBER_API_TOKEN", "").strip()
    if len(expected) < 32:
        raise HTTPException(503, "სერვერზე GEODUBBER_API_TOKEN არ არის კონფიგურირებული.")
    supplied = ""
    if authorization and authorization.lower().startswith("bearer "):
        supplied = authorization[7:].strip()
    if not supplied or not hmac.compare_digest(supplied, expected):
        raise HTTPException(401, "წვდომა აკრძალულია.")


def _job_or_404(job_id: str) -> dict:
    _job_dir(job_id)
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(404, "პროექტი ვერ მოიძებნა.")
        return job


def _validate_options(raw: Any) -> dict:
    if not isinstance(raw, dict):
        raise HTTPException(400, "პარამეტრების ფორმატი არასწორია.")
    priority = str(raw.get("priority", "fast")).lower()
    if priority not in {"fast", "balanced", "quality"}:
        raise HTTPException(400, "დამუშავების პრიორიტეტი არასწორია.")
    tts = str(raw.get("ttsEngine", "edge_georgian"))
    if tts != "edge_georgian":
        raise HTTPException(400, "უფასო ონლაინ სერვერზე ხელმისაწვდომია მხოლოდ Edge Georgian ხმა.")
    speakers = int(raw.get("speakerCount", 0) or 0)
    if speakers not in {0, 1, 2, 3, 4}:
        raise HTTPException(400, "პერსონაჟების რაოდენობა უნდა იყოს 0-დან 4-მდე.")
    batch = int(raw.get("translationBatch", 16) or 16)
    if not 1 <= batch <= 64:
        raise HTTPException(400, "თარგმნის ბლოკის ზომა არასწორია.")
    style = str(raw.get("translationStyleNote", "") or "").strip()
    if len(style) > 1200:
        raise HTTPException(400, "თარგმნის სტილის აღწერა ზედმეტად გრძელია.")
    if bool(raw.get("preserveBackground", False)):
        raise HTTPException(400, "უფასო სერვერზე Demucs ფონური აუდიოს გამოყოფა გამორთულია.")
    return {
        "priority": priority,
        "tts_engine": tts,
        "speaker_count": speakers,
        "translation_batch": batch,
        "translation_style_note": style,
        "preserve_background": bool(raw.get("preserveBackground", False)),
        "auto_speakers": bool(raw.get("autoSpeakers", True)),
        "georgian_subtitles": bool(raw.get("georgianSubtitles", True)),
        "original_subtitles": bool(raw.get("originalSubtitles", False)),
        "review_translation": bool(raw.get("reviewTranslation", True)),
        "burn_subtitles": bool(raw.get("burnSubtitles", False)),
        "original_dialogue_volume": max(0.0, min(1.0, float(raw.get("originalDialogueVolume", 0.0) or 0.0))),
    }


def _progress_dict(progress: Any) -> dict:
    return {
        "stage": str(progress.stage or ""),
        "stageLabel": str(progress.stage_label or ""),
        "stageFraction": round(float(progress.stage_fraction or 0), 4),
        "overall": round(float(progress.overall or 0), 4),
        "message": str(progress.message or "")[:300],
        "etaSeconds": progress.eta_seconds,
        "speed": str(progress.speed or "")[:80],
        "stages": progress.stage_status or {},
    }


def _run_job(job_id: str) -> None:
    from config.settings import Settings
    from core.errors import CancelledError
    from core.ffmpeg_utils import Control
    from core.processing_manager import Pipeline, default_backends

    with _lock:
        job = _jobs.get(job_id)
        if not job:
            return
        control = _controls.setdefault(job_id, Control())
        options = dict(job["options"])
        source_path = Path(job["sourcePath"])
        project_dir = _job_dir(job_id)
        _save_job(job_id)
    settings = Settings()
    settings.priority = options["priority"]
    settings.device = "cpu"
    settings.tts_engine = options["tts_engine"]
    settings.preserve_background = options["preserve_background"]
    settings.auto_speakers = options["auto_speakers"]
    settings.georgian_subtitles = options["georgian_subtitles"]
    settings.original_subtitles = options["original_subtitles"]
    settings.review_translation = options["review_translation"]
    settings.burn_subtitles = options["burn_subtitles"]
    settings.speaker_count = options["speaker_count"]
    settings.translation_batch = options["translation_batch"]
    settings.translation_style_note = options["translation_style_note"]
    settings.original_dialogue_volume = options["original_dialogue_volume"]
    settings.hf_token = os.getenv("HF_TOKEN", "").strip()
    script_path = job.get("scriptPath")
    settings.imported_subtitles = str(script_path) if script_path else ""
    settings.output_dir = str(project_dir / "output")
    settings.cache_dir = str(CACHE_ROOT / "tts")
    settings.temp_dir = str(project_dir / "temp")
    settings.edge_character_variants = True
    settings.max_speedup = 1.3
    settings.max_spill_seconds = 1.5
    settings.voice_qc = True
    settings.trim_clip_silence = True
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "temp").mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    pipeline = None
    review_event = threading.Event()
    with _lock:
        _review_events[job_id] = review_event
        job["status"] = "running"
        job["message"] = "დამუშავება დაიწყო."
        job["error"] = ""
        _save_job(job_id)

    def on_progress(progress: Any) -> None:
        with _lock:
            current = _jobs.get(job_id)
            if not current:
                return
            current["progress"] = _progress_dict(progress)
            current["message"] = current["progress"]["message"] or current["progress"]["stageLabel"]
            current["status"] = "paused" if control.paused else "running"
            if pipeline is not None:
                current["warnings"] = list(pipeline.state.get("warnings") or [])
            _save_job(job_id)

    def on_review(pl: Any) -> None:
        voice_rows = []
        try:
            voice_rows = [
                {"id": v.id, "name": v.name, "gender": v.gender}
                for v in pl.all_voices()
            ]
        except Exception:
            LOG.exception("Could not build voice list for review")
        review = {
            "segments": [
                {
                    "id": int(s.get("id", 0)),
                    "start": float(s.get("start", 0)),
                    "end": float(s.get("end", 0)),
                    "text": str(s.get("text", "")),
                    "translation": str(s.get("translation", "")),
                    "speakerId": str(s.get("speaker_id", "")),
                    "mtSource": str(s.get("mt_source", "")),
                    "note": str(s.get("tr_note", "")),
                }
                for s in (pl.state.get("segments") or [])
            ],
            "speakers": {
                sid: {
                    key: info.get(key)
                    for key in ("label", "gender", "gender_confidence", "speech_seconds", "f0", "gender_source")
                    if key in info
                }
                for sid, info in (pl.state.get("speakers") or {}).items()
            },
            "assignments": pl.state.get("assignments") or {},
            "voices": voice_rows,
        }
        with _lock:
            current = _jobs.get(job_id)
            if current:
                current["review"] = review
                current["status"] = "review_required"
                current["message"] = "თარგმანი და ხმები გადასახედია."
                _save_job(job_id)
        while not review_event.wait(0.25):
            control.checkpoint()
        control.checkpoint()
        with _lock:
            current = _jobs.get(job_id)
            if current:
                current["review"] = None
                current["status"] = "running"
                current["message"] = "დამუშავება გრძელდება."
                _save_job(job_id)

    try:
        backends, device_note = default_backends(settings, project_dir)
        LOG.info("Starting GeoDubber job %s using %s", job_id, device_note)
        pipeline = Pipeline(
            project_dir,
            settings,
            backends,
            control=control,
            on_progress=on_progress,
            on_review=on_review if settings.review_translation else None,
            cache_root=CACHE_ROOT / "tts",
        )
        with _lock:
            _pipelines[job_id] = pipeline
        outputs = pipeline.run(source_path)
        video_path = Path(outputs.get("video", ""))
        if not video_path.is_file():
            raise RuntimeError("ფინალური MP4 ფაილი ვერ შეიქმნა.")
        with _lock:
            current = _jobs.get(job_id)
            if current:
                current["status"] = "completed"
                current["message"] = "ქართული გახმოვანება დასრულდა."
                current["outputs"] = {
                    key: Path(value).name
                    for key, value in outputs.items()
                    if value and Path(value).is_file()
                }
                current["videoPath"] = str(video_path)
                current["warnings"] = list(pipeline.state.get("warnings") or [])
                current["progress"] = {
                    **(current.get("progress") or {}),
                    "overall": 1.0,
                    "stage": "complete",
                    "stageLabel": "დასრულებულია",
                    "stageFraction": 1.0,
                }
                _save_job(job_id)
    except CancelledError:
        with _lock:
            current = _jobs.get(job_id)
            if current:
                current["status"] = "cancelled"
                current["message"] = "დამუშავება შეჩერდა. პროგრესი შენახულია; შეგიძლია გააგრძელო."
                _save_job(job_id)
    except Exception as exc:
        LOG.exception("GeoDubber job %s failed", job_id)
        with _lock:
            current = _jobs.get(job_id)
            if current:
                current["status"] = "failed"
                current["error"] = str(exc)[:1200]
                current["message"] = "დამუშავება შეცდომით დასრულდა."
                if pipeline is not None:
                    current["warnings"] = list(pipeline.state.get("warnings") or [])
                _save_job(job_id)
    finally:
        try:
            if pipeline is not None:
                pipeline.be.tts.close()
        except Exception:
            LOG.exception("Could not close TTS backend")
        with _lock:
            _pipelines.pop(job_id, None)
            _review_events.pop(job_id, None)


@app.on_event("startup")
def startup() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    PROJECTS_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    # Resume queued/interrupted jobs from their persisted checkpoints.
    for folder in PROJECTS_ROOT.iterdir():
        metadata = folder / "job.json"
        if not folder.is_dir() or not metadata.is_file():
            continue
        try:
            job = json.loads(metadata.read_text(encoding="utf-8"))
            job_id = str(job.get("id", ""))
            if not JOB_ID_RE.fullmatch(job_id):
                continue
            job["status"] = "queued" if job.get("sourcePath") and Path(job["sourcePath"]).is_file() and job.get("status") in {
                "running", "paused", "review_required", "queued"
            } else job.get("status", "failed")
            job["message"] = "სერვერის გადატვირთვის შემდეგ პროექტი რიგში დაბრუნდა." if job["status"] == "queued" else job.get("message", "")
            _jobs[job_id] = job
            if job["status"] == "queued":
                _executor.submit(_run_job, job_id)
        except Exception:
            LOG.exception("Could not load GeoDubber job metadata: %s", metadata)


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "service": "geodubber-api"}


@app.get("/v1/status")
def status(authorization: str | None = Header(default=None)) -> dict:
    _require_api_key(authorization)
    with _lock:
        active = sum(1 for j in _jobs.values() if j.get("status") in {"queued", "running", "paused", "review_required"})
        return {
            "ok": True,
            "activeJobs": active,
            "totalJobs": len(_jobs),
            "storageRoot": str(DATA_ROOT),
            "geminiConfigured": bool(os.getenv("GEMINI_API_KEY", "").strip()),
        }


@app.post("/v1/jobs")
async def create_job(request: Request, authorization: str | None = Header(default=None)) -> dict:
    _require_api_key(authorization)
    try:
        raw = await request.json()
    except Exception:
        raise HTTPException(400, "პარამეტრების JSON ფორმატი არასწორია.")
    options = _validate_options(raw.get("options", raw))
    project_id = uuid.uuid4().hex[:24]
    folder = _job_dir(project_id)
    folder.mkdir(parents=True, exist_ok=False)
    job = {
        "id": project_id,
        "name": str(raw.get("name") or "GeoDubber project")[:100],
        "status": "upload_pending",
        "createdAt": _now(),
        "updatedAt": _now(),
        "fileName": "",
        "fileSize": 0,
        "sourcePath": "",
        "scriptPath": "",
        "options": options,
        "progress": {"overall": 0, "stage": "upload", "stageLabel": "ვიდეოს ატვირთვა", "stageFraction": 0},
        "message": "ატვირთე ვიდეო დამუშავების დასაწყებად.",
        "warnings": [],
        "error": "",
        "outputs": {},
        "review": None,
    }
    with _lock:
        _jobs[project_id] = job
        _save_job(project_id)
    return _public_job(job)


async def _stream_to_file(request: Request, destination: Path, limit: int) -> int:
    length = request.headers.get("content-length")
    if length:
        try:
            if int(length) > limit:
                raise HTTPException(413, "ფაილი დაშვებულ ზომას აჭარბებს.")
        except ValueError:
            raise HTTPException(400, "Content-Length არასწორია.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".uploading")
    total = 0
    try:
        with temporary.open("wb") as out:
            async for chunk in request.stream():
                if not chunk:
                    continue
                total += len(chunk)
                if total > limit:
                    raise HTTPException(413, "ფაილი დაშვებულ ზომას აჭარბებს.")
                out.write(chunk)
        if total < 128:
            raise HTTPException(400, "ატვირთული ფაილი ცარიელია ან დაზიანებულია.")
        os.replace(temporary, destination)
        return total
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


@app.put("/v1/jobs/{job_id}/video")
async def upload_video(job_id: str, request: Request, authorization: str | None = Header(default=None)) -> dict:
    _require_api_key(authorization)
    job = _job_or_404(job_id)
    if job.get("status") != "upload_pending":
        raise HTTPException(409, "ამ პროექტისთვის ვიდეო უკვე ატვირთულია.")
    raw_name = unquote(request.headers.get("x-file-name", "video.mp4"))
    safe_name = Path(raw_name.replace("\\", "/")).name[:180]
    extension = Path(safe_name).suffix.lower()
    if extension not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(415, "მხარდაჭერილია MP4, MKV, AVI, MOV და WEBM ფორმატები.")
    destination = _job_dir(job_id) / ("source" + extension)
    size = await _stream_to_file(request, destination, MAX_UPLOAD_BYTES)
    with _lock:
        job = _jobs[job_id]
        job["fileName"] = safe_name
        job["fileSize"] = size
        job["sourcePath"] = str(destination)
        job["status"] = "queued"
        job["message"] = "ვიდეო რიგში დაემატა."
        _controls[job_id] = __import__("core.ffmpeg_utils", fromlist=["Control"]).Control()
        _save_job(job_id)
        _executor.submit(_run_job, job_id)
    return _public_job(job)


@app.put("/v1/jobs/{job_id}/script")
async def upload_script(job_id: str, request: Request, authorization: str | None = Header(default=None)) -> dict:
    _require_api_key(authorization)
    job = _job_or_404(job_id)
    if job.get("status") != "upload_pending":
        raise HTTPException(409, "სუბტიტრები ვიდეოს ატვირთვამდე უნდა აიტვირთოს.")
    raw_name = unquote(request.headers.get("x-file-name", "script.srt"))
    safe_name = Path(raw_name.replace("\\", "/")).name[:180]
    extension = Path(safe_name).suffix.lower()
    if extension not in ALLOWED_SCRIPT_EXTENSIONS:
        raise HTTPException(415, "მხარდაჭერილია SRT, VTT და TXT ფორმატები.")
    destination = _job_dir(job_id) / ("script" + extension)
    await _stream_to_file(request, destination, 20 * 1024 * 1024)
    with _lock:
        job["scriptPath"] = str(destination)
        job["message"] = "სუბტიტრები აიტვირთა. ახლა ატვირთე ვიდეო."
        _save_job(job_id)
    return {"ok": True}


@app.get("/v1/jobs")
def list_jobs(authorization: str | None = Header(default=None)) -> list[dict]:
    _require_api_key(authorization)
    with _lock:
        return [_public_job(job) for job in sorted(_jobs.values(), key=lambda j: j.get("createdAt", 0), reverse=True)]


@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str, authorization: str | None = Header(default=None)) -> dict:
    _require_api_key(authorization)
    job = _job_or_404(job_id)
    return _public_job(job)


@app.get("/v1/jobs/{job_id}/review")
def get_review(job_id: str, authorization: str | None = Header(default=None)) -> dict:
    _require_api_key(authorization)
    job = _job_or_404(job_id)
    if job.get("status") != "review_required" or not job.get("review"):
        raise HTTPException(409, "ამ პროექტს ამ ეტაპზე გადახედვა არ სჭირდება.")
    return job["review"]


@app.post("/v1/jobs/{job_id}/review")
async def save_review(job_id: str, request: Request, authorization: str | None = Header(default=None)) -> dict:
    _require_api_key(authorization)
    job = _job_or_404(job_id)
    if job.get("status") != "review_required":
        raise HTTPException(409, "პროექტი გადახედვის ეტაპზე არ არის.")
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(400, "გადახედვის მონაცემები არასწორია.")
    with _lock:
        pipeline = _pipelines.get(job_id)
        event = _review_events.get(job_id)
    if pipeline is None or event is None:
        raise HTTPException(409, "გადახედვის სესია აღარ არის აქტიური.")
    changes = payload.get("segments", [])
    if not isinstance(changes, list) or len(changes) > 10000:
        raise HTTPException(400, "სუბტიტრების სია არასწორია.")
    by_id = {int(s.get("id", 0)): s for s in pipeline.state.get("segments", [])}
    for change in changes:
        if not isinstance(change, dict):
            continue
        seg = by_id.get(int(change.get("id", 0) or 0))
        if seg is None:
            continue
        translation = str(change.get("translation", ""))[:2000].strip()
        seg["translation"] = translation
        seg["mt_source"] = "manual"
    speaker_changes = payload.get("speakers", {})
    if isinstance(speaker_changes, dict):
        for speaker_id, update in speaker_changes.items():
            if not SPEAKER_ID_RE.fullmatch(str(speaker_id)) or speaker_id not in pipeline.state.get("speakers", {}):
                continue
            if isinstance(update, dict):
                gender = str(update.get("gender", "")).lower()
                if gender in {"male", "female", "unknown"}:
                    pipeline.state["speakers"][speaker_id]["gender"] = gender
                    pipeline.state["speakers"][speaker_id]["gender_source"] = "user"
    assignment_changes = payload.get("assignments", {})
    if isinstance(assignment_changes, dict):
        allowed = {v.id for v in pipeline.all_voices()}
        for speaker_id, voice_id in assignment_changes.items():
            if speaker_id in pipeline.state.get("speakers", {}) and voice_id in allowed:
                pipeline.state.setdefault("assignments", {})[speaker_id] = voice_id
    pipeline.save()
    with _lock:
        job["review"] = None
        job["message"] = "გადახედვა შენახულია."
        _save_job(job_id)
    event.set()
    return {"ok": True, "message": "ცვლილებები შენახულია. გახმოვანება გაგრძელდება."}


@app.post("/v1/jobs/{job_id}/pause")
def pause_job(job_id: str, authorization: str | None = Header(default=None)) -> dict:
    _require_api_key(authorization)
    job = _job_or_404(job_id)
    with _lock:
        control = _controls.get(job_id)
        if not control or job.get("status") not in {"running", "queued"}:
            raise HTTPException(409, "პროექტის შეჩერება ამ ეტაპზე შეუძლებელია.")
        control.pause()
        job["status"] = "paused"
        job["message"] = "დამუშავება შეჩერებულია."
        _save_job(job_id)
    return _public_job(job)


@app.post("/v1/jobs/{job_id}/resume")
def resume_job(job_id: str, authorization: str | None = Header(default=None)) -> dict:
    _require_api_key(authorization)
    job = _job_or_404(job_id)
    with _lock:
        status = job.get("status")
        if status == "paused":
            control = _controls.get(job_id)
            if not control:
                raise HTTPException(409, "შეჩერებული სამუშაო ვერ მოიძებნა.")
            control.resume()
            job["status"] = "running"
            job["message"] = "დამუშავება გრძელდება."
            _save_job(job_id)
        elif status in {"cancelled", "failed"} and job.get("sourcePath") and Path(job["sourcePath"]).is_file():
            from core.ffmpeg_utils import Control
            _controls[job_id] = Control()
            job["status"] = "queued"
            job["error"] = ""
            job["message"] = "პროექტი პროგრესის შენარჩუნებით რიგში დაბრუნდა."
            _save_job(job_id)
            _executor.submit(_run_job, job_id)
        else:
            raise HTTPException(409, "პროექტის გაგრძელება ამ ეტაპზე შეუძლებელია.")
    return _public_job(job)


@app.post("/v1/jobs/{job_id}/cancel")
def cancel_job(job_id: str, authorization: str | None = Header(default=None)) -> dict:
    _require_api_key(authorization)
    job = _job_or_404(job_id)
    with _lock:
        control = _controls.get(job_id)
        if not control or job.get("status") not in {"running", "paused", "queued", "review_required"}:
            raise HTTPException(409, "პროექტის შეჩერება ამ ეტაპზე შეუძლებელია.")
        control.cancel()
        event = _review_events.get(job_id)
        if event:
            event.set()
        job["status"] = "cancelling"
        job["message"] = "მიმდინარე ეტაპი უსაფრთხოდ ჩერდება..."
        _save_job(job_id)
    return _public_job(job)


@app.delete("/v1/jobs/{job_id}")
def delete_job(job_id: str, authorization: str | None = Header(default=None)) -> dict:
    _require_api_key(authorization)
    job = _job_or_404(job_id)
    with _lock:
        if job.get("status") in {"running", "paused", "queued", "review_required", "cancelling"}:
            raise HTTPException(409, "ჯერ გააჩერე დამუშავება, შემდეგ წაშალე პროექტი.")
        _jobs.pop(job_id, None)
        _controls.pop(job_id, None)
        _review_events.pop(job_id, None)
    shutil.rmtree(_job_dir(job_id), ignore_errors=True)
    return {"ok": True}


@app.get("/v1/jobs/{job_id}/download")
def download_video(job_id: str, authorization: str | None = Header(default=None)):
    _require_api_key(authorization)
    job = _job_or_404(job_id)
    if job.get("status") != "completed":
        raise HTTPException(409, "ფინალური ვიდეო ჯერ მზად არ არის.")
    path = Path(job.get("videoPath") or "")
    project = _job_dir(job_id).resolve()
    if not path.is_file() or not path.resolve().is_relative_to(project):
        raise HTTPException(404, "ფინალური ვიდეო ვერ მოიძებნა.")
    return FileResponse(path, media_type="video/mp4", filename=path.name)


@app.get("/v1/jobs/{job_id}/previews/{speaker_id}")
def preview_audio(job_id: str, speaker_id: str, authorization: str | None = Header(default=None)):
    _require_api_key(authorization)
    _job_or_404(job_id)
    if not SPEAKER_ID_RE.fullmatch(speaker_id):
        raise HTTPException(404, "აუდიო ნიმუში ვერ მოიძებნა.")
    path = _job_dir(job_id) / "work" / "speaker_previews" / f"{speaker_id}.wav"
    if not path.is_file():
        raise HTTPException(404, "აუდიო ნიმუში ჯერ მზად არ არის.")
    return FileResponse(path, media_type="audio/wav", filename=f"{speaker_id}.wav")


@app.exception_handler(HTTPException)
async def http_error_handler(_request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": str(exc.detail)})
