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
import secrets
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

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
_job_futures: dict[str, Any] = {}
_download_tickets: dict[str, dict[str, Any]] = {}
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
        "downloadProgress": job.get("downloadProgress") or {},
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


def _submit_job(job_id: str, target: Any, *args: Any) -> Any:
    """Submit and retain the worker future so deletion can stop it safely."""
    future = _executor.submit(target, *args)
    with _lock:
        _job_futures[job_id] = future
    return future


def _finish_delete_after_worker(job_id: str, future: Any | None) -> bool:
    """Delete a project's files now, or after its active worker exits.

    Returns True when cleanup was completed synchronously; otherwise a daemon
    cleanup thread waits for the worker to stop before removing its files.
    """
    def cleanup() -> None:
        if future is not None:
            try:
                future.result()
            except Exception:
                LOG.debug("Worker for deleted job %s finished with an exception", job_id, exc_info=True)
        with _lock:
            _job_futures.pop(job_id, None)
            _controls.pop(job_id, None)
            _review_events.pop(job_id, None)
            _pipelines.pop(job_id, None)
        shutil.rmtree(_job_dir(job_id), ignore_errors=True)
        LOG.info("Deleted GeoDubber project %s", job_id)

    if future is not None and not future.done():
        threading.Thread(
            target=cleanup,
            name=f"geodubber-delete-{job_id[:8]}",
            daemon=True,
        ).start()
        return False
    cleanup()
    return True


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
    def number(name: str, default: float, low: float, high: float) -> float:
        try:
            value = float(raw.get(name, default))
        except (TypeError, ValueError):
            raise HTTPException(400, f"{name}: არასწორი რიცხვითი მნიშვნელობა.")
        return max(low, min(high, value))
    genders = {}
    for i in range(1, 5):
        gender = str(raw.get(f"speaker{i}Gender", "auto")).lower()
        if gender not in {"auto", "male", "female"}:
            raise HTTPException(400, "პერსონაჟის სქესის პარამეტრი არასწორია.")
        genders[f"speaker{i}_gender"] = gender
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
        "original_dialogue_volume": number("originalDialogueVolume", 0.0, 0.0, 1.0),
        "voice_volume": number("voiceVolume", 1.0, 0.0, 2.0),
        "background_volume": number("backgroundVolume", 1.0, 0.0, 2.0),
        "duck_background": bool(raw.get("duckBackground", True)),
        "video_crf": int(number("videoCrf", 20, 14, 30)),
        "audio_bitrate": str(raw.get("audioBitrate", "192k")) if str(raw.get("audioBitrate", "192k")) in {"128k", "160k", "192k", "256k", "320k"} else "192k",
        **genders,
    }


def _progress_dict(progress: Any) -> dict:
    return {
        "stage": str(progress.stage or ""),
        "stageLabel": str(progress.stage_label or ""),
        "stageFraction": round(float(progress.stage_fraction or 0), 4),
        "overall": round(float(progress.overall or 0), 4),
        "message": str(progress.message or "")[:300],
        "etaSeconds": progress.eta_seconds,
        "totalEtaSeconds": getattr(progress, "total_eta_seconds", None),
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
    settings.voice_volume = options["voice_volume"]
    settings.background_volume = options["background_volume"]
    settings.duck_background = options["duck_background"]
    settings.video_crf = options["video_crf"]
    settings.audio_bitrate = options["audio_bitrate"]
    for key in ("speaker1_gender", "speaker2_gender", "speaker3_gender", "speaker4_gender"):
        setattr(settings, key, options[key])
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
        # Extract one representative frame for each detected speaker. This is only a
        # visual aid for the user's manual identity check; no gender is inferred from appearance.
        frame_dir = pl.work / "speaker_frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        frame_available = {}
        try:
            from core.ffmpeg_utils import run_ffmpeg
            by_speaker = {}
            for segment in (pl.state.get("segments") or []):
                sid = str(segment.get("speaker_id", ""))
                if not SPEAKER_ID_RE.fullmatch(sid):
                    continue
                duration = max(0.0, float(segment.get("end", 0)) - float(segment.get("start", 0)))
                if duration >= 0.8:
                    by_speaker.setdefault(sid, []).append(segment)
            for sid, rows in by_speaker.items():
                sample = max(rows, key=lambda row: float(row.get("end", 0)) - float(row.get("start", 0)))
                at = max(0.0, (float(sample["start"]) + float(sample["end"])) / 2.0)
                frame = frame_dir / f"{sid}.jpg"
                try:
                    run_ffmpeg(["-ss", str(at), "-i", str(source_path), "-frames:v", "1",
                                "-vf", "scale=480:-2", "-q:v", "4", str(frame)], control=control)
                    frame_available[sid] = frame.is_file() and frame.stat().st_size > 1000
                except Exception:
                    LOG.exception("Could not extract review frame for %s", sid)
                    frame_available[sid] = False
        except Exception:
            LOG.exception("Could not prepare speaker review frames")

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
            "frameAvailable": frame_available,
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
        # Use the same separator selection as the desktop pipeline:
        # Demucs when installed, otherwise FallbackSeparator.
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
                _submit_job(job_id, _run_job, job_id)
        except Exception:
            LOG.exception("Could not load GeoDubber job metadata: %s", metadata)


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "service": "geodubber-api"}


@app.get("/v1/status")
def status(authorization: str | None = Header(default=None)) -> dict:
    _require_api_key(authorization)
    with _lock:
        active = sum(1 for j in _jobs.values() if j.get("status") in {"queued", "downloading", "running", "paused", "review_required"})
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
        _submit_job(job_id, _run_job, job_id)
    return _public_job(job)


def _run_url_import(job_id: str, source_url: str) -> None:
    """Download an authorized source URL, then hand the file to the normal pipeline."""
    from core.url_import import VideoUrlCancelled, VideoUrlError, download_video_url
    job = _job_or_404(job_id)
    folder = _job_dir(job_id)

    def is_cancelled() -> bool:
        with _lock:
            current = _jobs.get(job_id)
            return not current or current.get("status") in {"cancelled", "cancelling"}

    def on_download(downloaded: int, total: int, state: str) -> None:
        with _lock:
            current = _jobs.get(job_id)
            if not current or current.get("status") in {"cancelled", "cancelling"}:
                return
            fraction = min(1.0, downloaded / total) if total else 0.0
            current["downloadProgress"] = {
                "downloadedBytes": downloaded,
                "totalBytes": total,
                "fraction": fraction,
                "state": state,
            }
            current["progress"] = {
                "overall": 0.0,
                "stage": "download",
                "stageLabel": "ვიდეოს ბმულიდან ჩამოტვირთვა",
                "stageFraction": fraction,
                "message": f"ჩამოტვირთულია {downloaded // (1024 * 1024)} MB"
                    + (f" / {total // (1024 * 1024)} MB" if total else ""),
            }
            current["message"] = current["progress"]["message"]
            _save_job(job_id)

    try:
        on_download(0, 0, "starting")
        source_path = download_video_url(source_url, folder, MAX_UPLOAD_BYTES, on_download, is_cancelled)
        if not source_path.is_file():
            raise VideoUrlError("ვიდეოს ჩამოტვირთვა დასრულდა, მაგრამ ფაილი ვერ მოიძებნა.")
        with _lock:
            current = _jobs.get(job_id)
            if not current or current.get("status") in {"cancelled", "cancelling"}:
                source_path.unlink(missing_ok=True)
                return
            current["fileName"] = source_path.name
            current["fileSize"] = source_path.stat().st_size
            current["sourcePath"] = str(source_path)
            current["downloadProgress"] = {
                "downloadedBytes": source_path.stat().st_size,
                "totalBytes": source_path.stat().st_size,
                "fraction": 1.0,
                "state": "finished",
            }
            current["status"] = "queued"
            current["message"] = "ვიდეო ჩამოიტვირთა და დამუშავების რიგში დაემატა."
            from core.ffmpeg_utils import Control
            _controls[job_id] = Control()
            _save_job(job_id)
        _run_job(job_id)
    except VideoUrlCancelled:
        with _lock:
            current = _jobs.get(job_id)
            if current:
                current["status"] = "cancelled"
                current["message"] = "ჩამოტვირთვა გაუქმებულია. პროექტის წაშლა ახლა შეგიძლია."
                current["downloadProgress"] = {**(current.get("downloadProgress") or {}), "state": "cancelled"}
                _save_job(job_id)
        shutil.rmtree(folder / "_torrent_download", ignore_errors=True)
    except Exception as exc:
        LOG.exception("GeoDubber URL import failed for job %s", job_id)
        with _lock:
            current = _jobs.get(job_id)
            if current:
                if current.get("status") in {"cancelled", "cancelling"}:
                    current["status"] = "cancelled"
                    current["message"] = "ჩამოტვირთვა გაუქმებულია. პროექტის წაშლა ახლა შეგიძლია."
                else:
                    current["status"] = "failed"
                    current["error"] = str(exc)[:1200]
                    current["message"] = "ვიდეოს ბმულიდან მიღება ვერ მოხერხდა."
                _save_job(job_id)
        shutil.rmtree(folder / "_torrent_download", ignore_errors=True)


@app.post("/v1/jobs/{job_id}/source-url")
async def import_video_url(
    job_id: str,
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict:
    _require_api_key(authorization)
    job = _job_or_404(job_id)
    if job.get("status") != "upload_pending":
        raise HTTPException(409, "ამ პროექტისთვის ვიდეოს წყარო უკვე მითითებულია.")
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(400, "ვიდეოს ბმულის JSON ფორმატი არასწორია.")
    source_url = str(payload.get("url") or "").strip()
    if not source_url or len(source_url) > 4096:
        raise HTTPException(400, "შეიყვანე სწორი ვიდეოს ბმული.")
    from urllib.parse import urlparse
    parsed = urlparse(source_url)
    if parsed.scheme.lower() not in {"magnet", "https"}:
        raise HTTPException(415, "მხარდაჭერილია Magnet ბმული, YouTube HTTPS ბმული ან პირდაპირი HTTPS ვიდეოფაილი.")
    with _lock:
        job["status"] = "downloading"
        job["message"] = "ვიდეოს ბმული მიღებულია. ჩამოტვირთვა იწყება."
        job["sourceUrlType"] = "torrent" if parsed.scheme.lower() == "magnet" else ("youtube" if (parsed.hostname or "").lower().removeprefix("www.") in {"youtube.com", "m.youtube.com", "youtu.be"} else "direct")
        job["downloadProgress"] = {"downloadedBytes": 0, "totalBytes": 0, "fraction": 0.0, "state": "starting"}
        job["progress"] = {
            "overall": 0.0,
            "stage": "download",
            "stageLabel": "ვიდეოს ბმულიდან ჩამოტვირთვა",
            "stageFraction": 0.0,
            "message": "ჩამოტვირთვა იწყება...",
        }
        _save_job(job_id)
        _submit_job(job_id, _run_url_import, job_id, source_url)
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


@app.get("/v1/jobs/{job_id}/frames/{speaker_id}")
def preview_frame(job_id: str, speaker_id: str, authorization: str | None = Header(default=None)):
    _require_api_key(authorization)
    _job_or_404(job_id)
    if not SPEAKER_ID_RE.fullmatch(speaker_id):
        raise HTTPException(404, "კადრი ვერ მოიძებნა.")
    path = _job_dir(job_id) / "work" / "speaker_frames" / f"{speaker_id}.jpg"
    if not path.is_file():
        raise HTTPException(404, "პერსონაჟის კადრი ჯერ მზად არ არის.")
    return FileResponse(path, media_type="image/jpeg", filename=f"{speaker_id}.jpg")


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
        speaker_id = str(change.get("speakerId", "") or "")
        if SPEAKER_ID_RE.fullmatch(speaker_id):
            seg["speaker_id"] = speaker_id
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
            _submit_job(job_id, _run_job, job_id)
        else:
            raise HTTPException(409, "პროექტის გაგრძელება ამ ეტაპზე შეუძლებელია.")
    return _public_job(job)


@app.post("/v1/jobs/{job_id}/cancel")
def cancel_job(job_id: str, authorization: str | None = Header(default=None)) -> dict:
    _require_api_key(authorization)
    job = _job_or_404(job_id)
    with _lock:
        status = job.get("status")
        if status == "upload_pending":
            job["status"] = "cancelled"
            job["message"] = "პროექტი გაუქმებულია. წაშლა ახლა შეგიძლია."
            _save_job(job_id)
            return _public_job(job)
        if status == "downloading":
            job["status"] = "cancelling"
            job["message"] = "ჩამოტვირთვა ჩერდება..."
            _save_job(job_id)
            return _public_job(job)
        control = _controls.get(job_id)
        if not control or status not in {"running", "paused", "queued", "review_required"}:
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
    """Cancel any active work and delete the project without a separate Stop action."""
    _require_api_key(authorization)
    job = _job_or_404(job_id)
    with _lock:
        future = _job_futures.get(job_id)
        control = _controls.get(job_id)
        event = _review_events.get(job_id)
        status = str(job.get("status", ""))
        worker_active = future is not None and not future.done()
        active_status = status in {"running", "paused", "queued", "review_required", "downloading", "cancelling"}

        if active_status or worker_active:
            # Signal graceful cancellation before making the project disappear
            # from the UI/API list. The worker's cancellation callbacks also
            # treat a missing job as a cancellation request.
            if control is not None:
                try:
                    control.cancel()
                except Exception:
                    LOG.exception("Could not signal cancellation for job %s", job_id)
            if event is not None:
                event.set()
            job["status"] = "cancelled"
            job["message"] = "წაშლის მოთხოვნა მიღებულია — მიმდინარე სამუშაო ჩერდება და ფაილები იშლება."
            try:
                _save_job(job_id)
            except Exception:
                LOG.exception("Could not persist delete state for job %s", job_id)

        # Cancel a job that is still waiting in the single-worker queue. If it
        # has already started, the cleanup thread waits for it to exit.
        if future is not None and not future.done():
            future.cancel()
        _jobs.pop(job_id, None)
        for ticket, record in list(_download_tickets.items()):
            if record.get("jobId") == job_id:
                _download_tickets.pop(ticket, None)

    completed_now = _finish_delete_after_worker(job_id, future)
    return {
        "ok": True,
        "deleting": not completed_now,
        "message": "პროექტი წაიშალა." if completed_now else "მიმდინარე სამუშაო ავტომატურად ჩერდება; პროექტის ფაილები წაიშლება პროცესის უსაფრთხოდ დასრულებისთანავე.",
    }


@app.post("/v1/jobs/{job_id}/download-ticket")
def create_download_ticket(job_id: str, authorization: str | None = Header(default=None)) -> dict:
    """Create a short-lived URL so the browser can download large MP4s directly.

    This avoids buffering the complete video in browser memory and avoids relying
    on a synthetic click after a multi-gigabyte fetch.
    """
    _require_api_key(authorization)
    job = _job_or_404(job_id)
    if job.get("status") != "completed":
        raise HTTPException(409, "ფინალური ვიდეო ჯერ მზად არ არის.")
    path = Path(job.get("videoPath") or "")
    project = _job_dir(job_id).resolve()
    if not path.is_file() or not path.resolve().is_relative_to(project):
        raise HTTPException(404, "ფინალური ვიდეო ვერ მოიძებნა.")

    now = time.time()
    ticket = secrets.token_urlsafe(32)
    with _lock:
        expired = [key for key, value in _download_tickets.items() if value.get("expiresAt", 0) <= now]
        for key in expired:
            _download_tickets.pop(key, None)
        _download_tickets[ticket] = {"jobId": job_id, "expiresAt": now + 900}
    return {"url": f"/v1/download/{ticket}", "filename": path.name, "expiresIn": 900}


@app.get("/v1/download/{ticket}")
def download_with_ticket(ticket: str):
    """Stream the MP4 using an expiring, unguessable ticket (no API token in URL)."""
    now = time.time()
    with _lock:
        record = _download_tickets.get(ticket)
        if not record:
            raise HTTPException(404, "ჩამოტვირთვის ბმული არასწორია ან ვადაგასულია.")
        if record.get("expiresAt", 0) <= now:
            _download_tickets.pop(ticket, None)
            raise HTTPException(410, "ჩამოტვირთვის ბმულს ვადა გაუვიდა. დააჭირე Download Output-ს ხელახლა.")
        job = _jobs.get(record.get("jobId", ""))
        if not job or job.get("status") != "completed":
            raise HTTPException(404, "პროექტი ან მისი საბოლოო ვიდეო ვერ მოიძებნა.")
        job_id = job["id"]
        path = Path(job.get("videoPath") or "")
    project = _job_dir(job_id).resolve()
    if not path.is_file() or not path.resolve().is_relative_to(project):
        raise HTTPException(404, "ფინალური ვიდეო ვერ მოიძებნა.")
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=path.name,
        headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"},
    )


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



@app.get("/v1/jobs/{job_id}/segments")
def get_segments(job_id: str, authorization: str | None = Header(default=None)) -> dict:
    _require_api_key(authorization)
    job = _job_or_404(job_id)
    with _lock:
        pipeline = _pipelines.get(job_id)
        if pipeline is not None:
            segments = pipeline.state.get("segments") or []
        else:
            state_file = _job_dir(job_id) / "state.json"
            if not state_file.is_file():
                raise HTTPException(409, "სუბტიტრები ჯერ მზად არ არის.")
            try:
                state = json.loads(state_file.read_text(encoding="utf-8"))
                segments = state.get("segments") or []
            except (OSError, ValueError):
                raise HTTPException(500, "პროექტის სუბტიტრების წაკითხვა ვერ მოხერხდა.")
    return {"segments": [
        {"id": int(x.get("id", 0)), "start": float(x.get("start", 0)),
         "end": float(x.get("end", 0)), "text": str(x.get("text", "")),
         "translation": str(x.get("translation", "")),
         "speaker_id": str(x.get("speaker_id", ""))}
        for x in segments
    ]}


@app.get("/v1/jobs/{job_id}/subtitles/{kind}")
def download_subtitles(job_id: str, kind: str, authorization: str | None = Header(default=None)):
    _require_api_key(authorization)
    job = _job_or_404(job_id)
    if job.get("status") != "completed":
        raise HTTPException(409, "სუბტიტრები დასრულებული პროექტისთვის იქნება ხელმისაწვდომი.")
    if kind not in {"georgian", "original"}:
        raise HTTPException(404, "სუბტიტრების ტიპი ვერ მოიძებნა.")
    state_file = _job_dir(job_id) / "state.json"
    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
        segments = state.get("segments") or []
    except (OSError, ValueError):
        raise HTTPException(404, "სუბტიტრები ვერ მოიძებნა.")
    from core.subtitles import write_srt
    key = "translation" if kind == "georgian" else "text"
    target = _job_dir(job_id) / ("subtitles_ka.srt" if kind == "georgian" else "subtitles_original.srt")
    write_srt(segments, target, key)
    return FileResponse(target, media_type="application/x-subrip; charset=utf-8", filename=target.name)


@app.exception_handler(HTTPException)
async def http_error_handler(_request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": str(exc.detail)})


# Serve the browser interface from the same origin as the authenticated API.
WEB_ROOT = Path(__file__).resolve().parent / "web"
if WEB_ROOT.is_dir():
    app.mount("/", StaticFiles(directory=str(WEB_ROOT), html=True), name="geodubber-web")
