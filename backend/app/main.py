"""FastAPI application for the perimeter anomaly detection service.

Endpoints (all JSON unless noted):

- `GET  /health`                              model loaded?
- `GET  /clips/{clip_id}`                     clip info + its zone, from config
- `GET  /clips/{clip_id}/reference-frame`     JPEG the zone is drawn over
- `POST /clips/{clip_id}/process`             run the clip through the pipeline
- `GET  /clips/{clip_id}/alerts`              alerts from the last run
- `GET  /clips/{clip_id}/snapshots/{name}`    JPEG snapshot referenced by an alert

Uploads (Phase 1, see docs/PHASE1.md and app/services/jobs.py):

- `POST /uploads`                             upload a clip -> 202 + job id
- `GET  /jobs/{job_id}`                       job status and progress
- `GET  /jobs/{job_id}/alerts`                alerts, once the job is complete
- `GET  /jobs/{job_id}/snapshots/{name}`      JPEG snapshot referenced by an alert
- `GET  /jobs/{job_id}/reference-frame`       the upload's first frame

The committed sample clip keeps its original synchronous endpoint: it's a
fixed 5 s clip, a demo, and the regression check. Uploads are processed in
the background because their length is up to the user. Results live in
memory — a restart forgets them. No live input (see docs/PROJECT.md).
"""

import logging
import os
import shutil
import threading
from pathlib import PurePosixPath
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import cv2

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi import Path as Path_  # pathlib.Path is used too
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from app.models.schemas import Alert, ClipResult, Zone
from app.services.clip_processor import (
    ClipReadError,
    FrameDetector,
    SELECTABLE_CLASSES,
    FrameSizeMismatchError,
    classes_for,
    load_zone,
    process_clip,
    whole_frame_zone,
)
from app.services.detector import Detector
from app.services.jobs import (
    Job,
    JobRunner,
    JobState,
    QueueFull,
    new_job_id,
    reference_key,
    snapshot_key,
)
from app.services.storage import LocalObjectStore, ObjectStore, StorageError, store_from_env
from app.services.uploads import (
    DEFAULT_LIMITS,
    UploadRejected,
    check_probe,
    probe_clip,
    sniff_container,
)

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
_SAMPLE_DIR = Path(__file__).resolve().parent / "data" / "sample_clip"

# Where snapshots are written. Outside the source tree's data/ directory so a
# run never dirties committed files; gitignored.
OUTPUT_DIR = Path(os.environ.get("PERIMETER_OUTPUT_DIR", _BACKEND_ROOT / "output"))

# Browser origins allowed to call the API. The dashboard runs on a different
# origin (localhost:3000 in dev, Vercel in production), so without this the
# browser blocks every request. Comma-separated.
CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get("PERIMETER_CORS_ORIGINS", "http://localhost:3000").split(",")
    if o.strip()
]


@dataclass(frozen=True)
class ClipSource:
    clip_path: Path
    zone_path: Path
    reference_frame_path: Path


CLIPS: dict[str, ClipSource] = {
    "sample": ClipSource(
        clip_path=_SAMPLE_DIR / "crosswalk.mp4",
        zone_path=_SAMPLE_DIR / "zone.json",
        reference_frame_path=_SAMPLE_DIR / "reference_frame.jpg",
    ),
}

logger = logging.getLogger(__name__)

detector = Detector()
# Why the model failed to load at startup, if it did. Reported by /health
# and in the 503 that processing requests get.
model_load_error: str | None = None

# Last result per clip. Guarded by _processing_lock for writes.
_results: dict[str, ClipResult] = {}
# One run at a time: processing is CPU-bound, the model instance is shared,
# and two runs would write into the same snapshot directory. Shared by the
# sample-clip endpoint and the upload job runner.
_processing_lock = threading.Lock()

# --- uploads (Phase 1) ---
UPLOAD_LIMITS = DEFAULT_LIMITS
# Uploads are sampled at 2 frames per second of video, not every frame: at
# ~0.57 s per processed frame on the free-tier host, every frame of a 60 s
# 30 fps clip would take ~17 minutes; 2 fps keeps it to ~120 processed frames.
# A walking person moves ~0.7 m between samples, so zone entries are still
# caught (Day 2 reasoning). The sample clip keeps every frame.
UPLOAD_SAMPLE_FPS = float(os.environ.get("PERIMETER_UPLOAD_SAMPLE_FPS", 2.0))
# Allowance for the multipart framing around the file in the request body.
_MULTIPART_OVERHEAD = 1024 * 1024

# Upload results go to Cloudflare R2 when PERIMETER_R2_* is configured (see
# app/services/storage.py), else to a local folder for development. Chosen at
# startup; until then (and in tests) a local store is in place.
storage_error: str | None = None
jobs = JobRunner(
    model_lock=_processing_lock,
    store=LocalObjectStore(OUTPUT_DIR / "store"),
    work_root=OUTPUT_DIR / "work",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model at startup rather than on the first request.

    Loading YOLO26-N takes a few seconds; paying that at startup means the
    first real request isn't the one that waits for it.

    If loading fails (weights missing or corrupt), the service still starts,
    in a degraded state: /health reports `model_loaded: false` with the
    reason, processing requests get a 503 saying the same, and the error is
    logged. That is louder than refusing to boot — a host polling /health
    (Day 5) sees why, instead of a crash loop with the reason only in logs.
    """
    global model_load_error, storage_error, jobs
    # The work folder holds uploaded videos only while their job runs; after
    # a restart nothing can be running, so anything there is left over.
    shutil.rmtree(_work_dir(), ignore_errors=True)
    store: ObjectStore
    try:
        store = store_from_env(OUTPUT_DIR / "store")
        store.check()
        storage_error = None
        logger.info("upload results are stored in: %s", store.kind)
    except Exception as e:
        # Uploads are refused (503) rather than falling back to local disk,
        # where results would vanish at the next restart.
        storage_error = f"{type(e).__name__}: {e}"
        store = LocalObjectStore(OUTPUT_DIR / "store")
        logger.error("result storage unavailable; uploads disabled: %s", storage_error)
    jobs = JobRunner(model_lock=_processing_lock, store=store, work_root=_work_dir())
    try:
        detector.load()
        model_load_error = None
    except Exception as e:  # ultralytics raises TypeError, UnpicklingError, ...
        model_load_error = f"{type(e).__name__}: {e}"
        logger.error("model failed to load; service is degraded: %s", model_load_error)
    yield
    jobs.shutdown()


app = FastAPI(
    title="Perimeter Anomaly Detection",
    description=(
        "Detects people and vehicles in recorded video clips and raises an "
        "alert when one enters a configured zone."
    ),
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def get_detector() -> FrameDetector:
    """The detector endpoints use. Overridden in tests with a fake.

    Refuses with 503 when the model isn't loaded, rather than letting
    `Detector.detect` retry the load mid-request and fail with a bare 500.
    """
    if not detector.is_loaded:
        reason = model_load_error or "it has not been loaded"
        raise HTTPException(
            503,
            f"model not loaded ({reason}); fix {detector.weights_path.name} "
            f"and restart the service",
        )
    return detector


def get_clip(clip_id: str) -> ClipSource:
    source = CLIPS.get(clip_id)
    if source is None:
        raise HTTPException(404, f"unknown clip {clip_id!r}; available: {sorted(CLIPS)}")
    return source


def _snapshot_dir(clip_id: str) -> Path:
    return OUTPUT_DIR / clip_id / "snapshots"


def _work_dir() -> Path:
    return OUTPUT_DIR / "work"


@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    """Refuse an oversized upload from its declared length, before reading it.

    FastAPI reads a form body completely before any endpoint code runs, so
    this is the only place the size can be refused up front. Uploads without
    a declared length (chunked) are refused too: they'd have no bound at all.
    Browsers always declare the length of a form upload. The endpoint still
    counts the real bytes as a second check.
    """
    if request.method == "POST" and request.url.path == "/uploads":
        declared = request.headers.get("content-length")
        if declared is None:
            return JSONResponse(
                {"detail": "uploads must declare their size (Content-Length)"}, status_code=411
            )
        if int(declared) > UPLOAD_LIMITS.max_bytes + _MULTIPART_OVERHEAD:
            return JSONResponse({"detail": _too_big_message()}, status_code=413)
    return await call_next(request)


def _too_big_message() -> str:
    return f"the file is over the {UPLOAD_LIMITS.max_bytes // (1024 * 1024)} MB size limit"


# --- response models ---------------------------------------------------------


class HealthResponse(BaseModel):
    """Response body for `GET /health`."""

    status: str
    model_loaded: bool
    model_weights: str
    model_error: str | None = Field(
        description="Why the model failed to load at startup; null when loaded."
    )
    inference_threads: int | None = Field(
        description=(
            "CPU threads PyTorch uses for inference; set from the container's CPU "
            "limit when there is one. Null when the model isn't loaded."
        )
    )
    storage: str = Field(description='Where upload results are kept: "r2" or "local".')
    storage_error: str | None = Field(
        description="Why result storage is unavailable (uploads disabled); null when fine."
    )


class ClipInfo(BaseModel):
    clip_id: str
    zone: Zone = Field(description="The configured zone, read from the clip's zone.json.")
    reference_frame_url: str
    processed: bool = Field(description="Whether alerts from a run are available.")


class ProcessSummary(BaseModel):
    clip_id: str
    frames_read: int
    frames_processed: int
    processing_time_s: float
    alert_count: int


class AlertOut(Alert):
    snapshot_url: str = Field(description="API path serving this alert's snapshot.")


class ClipInfoOut(BaseModel):
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float


class JobResponse(BaseModel):
    job_id: str
    status: JobState
    filename: str = Field(description="The uploaded file's name, for display only.")
    classes: list[str] = Field(description='What alerts, e.g. ["person", "dog"].')
    clip: ClipInfoOut
    zone: Zone
    sample_fps: float | None
    frames_to_process: int
    frames_processed: int
    queue_position: int = Field(description="1 = next to run; 0 when not waiting.")
    created_at: float
    started_at: float | None
    finished_at: float | None
    processing_time_s: float | None
    error: str | None
    status_url: str
    alerts_url: str
    reference_frame_url: str


class AlertsResponse(BaseModel):
    clip_id: str
    zone_name: str
    frame_width: int = Field(description="Frame size that bbox/anchor coordinates refer to.")
    frame_height: int
    clip_fps: float
    sample_fps: float | None
    frames_read: int
    frames_processed: int
    processing_time_s: float
    alerts: list[AlertOut]


# --- endpoints ---------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness/readiness check.

    Reports whether the model is actually loaded, not just whether the
    process is up — a service that is running but can't do inference is not
    healthy in any useful sense, and Day 5 deploys this to a host that will
    poll it.
    """
    return HealthResponse(
        status="ok" if detector.is_loaded and storage_error is None else "degraded",
        model_loaded=detector.is_loaded,
        model_weights=detector.weights_path.name,
        model_error=model_load_error,
        inference_threads=(
            getattr(detector, "inference_threads", None) if detector.is_loaded else None
        ),
        storage=jobs.store.kind,
        storage_error=storage_error,
    )


@app.get("/clips/{clip_id}", response_model=ClipInfo)
def clip_info(clip_id: str) -> ClipInfo:
    source = get_clip(clip_id)
    return ClipInfo(
        clip_id=clip_id,
        zone=load_zone(source.zone_path),
        reference_frame_url=f"/clips/{clip_id}/reference-frame",
        processed=clip_id in _results,
    )


@app.get("/clips/{clip_id}/reference-frame", response_class=FileResponse)
def reference_frame(clip_id: str) -> FileResponse:
    return FileResponse(get_clip(clip_id).reference_frame_path, media_type="image/jpeg")


@app.post("/clips/{clip_id}/process", response_model=ProcessSummary)
def process(
    clip_id: str,
    detector: Annotated[FrameDetector, Depends(get_detector)],
    sample_fps: Annotated[
        float | None,
        Query(gt=0, description="Samples per second of clip time; omit for every frame."),
    ] = None,
) -> ProcessSummary:
    """Run the clip through detection + zone check, replacing any earlier result.

    Synchronous: the response arrives when processing finishes (~4-5 s for
    the sample clip on a laptop CPU; slower on a free-tier host). A plain
    `def` endpoint, so FastAPI runs it in a worker thread and the server stays
    responsive to other requests meanwhile.
    """
    source = get_clip(clip_id)
    if not _processing_lock.acquire(blocking=False):
        raise HTTPException(409, "a clip is already being processed; try again when it finishes")
    try:
        # Drop the old result first: its snapshot files are about to be deleted.
        _results.pop(clip_id, None)
        snapshot_dir = _snapshot_dir(clip_id)
        shutil.rmtree(snapshot_dir, ignore_errors=True)
        try:
            result = process_clip(
                source.clip_path,
                load_zone(source.zone_path),
                detector,
                snapshot_dir,
                sample_fps=sample_fps,
            )
        except FrameSizeMismatchError as e:
            # The committed clip and its zone disagree: a server config
            # error, not something the caller did wrong.
            raise HTTPException(500, str(e)) from e
        except (ClipReadError, FileNotFoundError) as e:
            # Also server-side: the clip is a file the server is configured
            # with, not something the caller sent. 500, but with the reason.
            raise HTTPException(500, f"clip could not be read: {e}") from e
        except OSError as e:
            # After the clip branch above, which catches FileNotFoundError (a
            # subclass): what's left is the output side — disk full,
            # permissions — while writing snapshots.
            raise HTTPException(500, f"could not write snapshots: {e}") from e
        _results[clip_id] = result
    finally:
        _processing_lock.release()

    return ProcessSummary(
        clip_id=clip_id,
        frames_read=result.frames_read,
        frames_processed=result.frames_processed,
        processing_time_s=result.processing_time_s,
        alert_count=len(result.alerts),
    )


@app.get("/clips/{clip_id}/alerts", response_model=AlertsResponse)
def alerts(clip_id: str) -> AlertsResponse:
    source = get_clip(clip_id)
    result = _results.get(clip_id)
    if result is None:
        raise HTTPException(404, f"clip {clip_id!r} has not been processed yet")
    zone = load_zone(source.zone_path)
    return AlertsResponse(
        clip_id=clip_id,
        zone_name=zone.name,
        frame_width=zone.frame_width,
        frame_height=zone.frame_height,
        clip_fps=result.clip_fps,
        sample_fps=result.sample_fps,
        frames_read=result.frames_read,
        frames_processed=result.frames_processed,
        processing_time_s=result.processing_time_s,
        alerts=[
            AlertOut(
                **a.model_dump(),
                snapshot_url=f"/clips/{clip_id}/snapshots/{a.snapshot}",
            )
            for a in result.alerts
        ],
    )


# --- uploads and jobs (Phase 1) ---------------------------------------------------


def _job_response(job: Job) -> JobResponse:
    base = f"/jobs/{job.job_id}"
    return JobResponse(
        job_id=job.job_id,
        status=job.status,
        filename=job.filename,
        classes=job.classes,
        clip=ClipInfoOut(
            width=job.probe.width,
            height=job.probe.height,
            fps=job.probe.fps,
            frame_count=job.probe.frame_count,
            duration_s=job.probe.duration_s,
        ),
        zone=job.zone,
        sample_fps=job.sample_fps,
        frames_to_process=job.frames_to_process,
        frames_processed=job.frames_processed,
        queue_position=jobs.queue_position(job),
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        processing_time_s=job.processing_time_s,
        error=job.error,
        status_url=base,
        alerts_url=f"{base}/alerts",
        # A signed, expiring R2 link in production; an API route locally.
        reference_frame_url=jobs.store.url(reference_key(job.job_id)),
    )


def _save_upload(source, destination: Path) -> None:
    """Copy the uploaded file, enforcing the size limit on the actual bytes."""
    written = 0
    with destination.open("wb") as out:
        while chunk := source.read(1024 * 1024):
            written += len(chunk)
            if written > UPLOAD_LIMITS.max_bytes:
                raise UploadRejected(_too_big_message(), status_code=413)
            out.write(chunk)


def _write_reference_frame(frame, path: Path) -> None:
    height, width = frame.shape[:2]
    if width > 1280:  # a view to draw the zone over, not evidence
        frame = cv2.resize(frame, (1280, round(height * 1280 / width)), interpolation=cv2.INTER_AREA)
    if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 80]):
        raise OSError(f"failed to write reference frame: {path}")


class UploadLimitsResponse(BaseModel):
    max_bytes: int
    max_duration_s: float
    max_resolution: str = Field(description="Long x short side, either orientation.")
    max_fps: float
    classes: list[str]
    sample_fps: float = Field(description="Frames per second of video that are processed.")
    formats: str


@app.get("/uploads/limits", response_model=UploadLimitsResponse)
def upload_limits() -> UploadLimitsResponse:
    """The limits POST /uploads enforces, so the dashboard can check first."""
    return UploadLimitsResponse(
        max_bytes=UPLOAD_LIMITS.max_bytes,
        max_duration_s=UPLOAD_LIMITS.max_duration_s,
        max_resolution=f"{UPLOAD_LIMITS.max_long_side}x{UPLOAD_LIMITS.max_short_side}",
        max_fps=UPLOAD_LIMITS.max_fps,
        classes=list(SELECTABLE_CLASSES),
        sample_fps=UPLOAD_SAMPLE_FPS,
        formats="MP4/MOV, WebM/MKV, AVI, MPEG-TS",
    )


@app.post("/uploads", status_code=202, response_model=JobResponse)
def create_upload(
    file: Annotated[UploadFile, File(description="The video clip.")],
    classes: Annotated[
        list[str],
        Form(
            description=(
                "What to alert on; repeat the field for several: person, vehicle, "
                "bicycle, dog, cat, backpack, handbag, suitcase."
            )
        ),
    ],
    detector: Annotated[FrameDetector, Depends(get_detector)],
) -> JobResponse:
    """Validate and store an uploaded clip, then queue it for processing.

    Returns 202 immediately with a job id; poll `status_url` for progress.
    The whole frame is the zone (Phase 1 has no zone editor). A plain `def`
    endpoint: copying the file and probing it with OpenCV block, so FastAPI
    runs this in a worker thread instead of on the event loop.
    """
    if storage_error is not None:
        raise HTTPException(503, f"result storage is unavailable, so uploads are disabled ({storage_error})")
    # "both" (Phase 1's choice) still accepted; stored as what it means.
    selection = [c for choice in classes for c in (["person", "vehicle"] if choice == "both" else [choice])]
    try:
        classes_for(selection)
    except ValueError as e:
        raise HTTPException(422, f"{e}; choose from: {', '.join(SELECTABLE_CLASSES)}") from e
    selection = list(dict.fromkeys(selection))  # de-duplicate, keep order
    if jobs.active_count() >= jobs.max_active:
        raise HTTPException(429, _busy_message())

    job_id = new_job_id()
    # Local and temporary: the video lives here only until its job ends.
    directory = jobs.work_root / job_id
    directory.mkdir(parents=True)
    video_path = directory / "upload"
    try:
        _save_upload(file.file, video_path)
        with video_path.open("rb") as f:
            if sniff_container(f.read(512)) is None:
                raise UploadRejected(
                    "the file is not a supported video (MP4/MOV, WebM/MKV, AVI or MPEG-TS)",
                    status_code=415,
                )
        probe, first_frame = probe_clip(video_path)
        check_probe(probe, UPLOAD_LIMITS)
        _write_reference_frame(first_frame, directory / "reference.jpg")
        try:
            job = jobs.submit(
                job_id=job_id,
                work_dir=directory,
                reference_frame=directory / "reference.jpg",
                # Display only: never used to build a path.
                filename=PurePosixPath((file.filename or "upload").replace("\\", "/")).name[:120]
                or "upload",
                classes=selection,
                probe=probe,
                zone=whole_frame_zone(probe.width, probe.height),
                sample_fps=UPLOAD_SAMPLE_FPS,
                detector=detector,
            )
        except (QueueFull, UploadRejected):
            raise
        except Exception as e:  # the store refused the reference frame / record
            logger.exception("could not register job %s in storage", job_id)
            raise UploadRejected(
                f"the upload couldn't be saved to result storage ({type(e).__name__}); "
                "try again shortly",
                status_code=503,
            ) from e
    except UploadRejected as e:
        shutil.rmtree(directory, ignore_errors=True)
        raise HTTPException(e.status_code, str(e)) from e
    except QueueFull as e:
        shutil.rmtree(directory, ignore_errors=True)
        raise HTTPException(429, _busy_message()) from e
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    return _job_response(job)


def _busy_message() -> str:
    return (
        f"the server is busy with {jobs.max_active} clips already; "
        "try again when one has finished"
    )


_JOB_ID = Path_(pattern=r"^[0-9a-f]{32}$")


def _get_job(job_id: str) -> Job:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(
            404,
            f"unknown job {job_id!r}: it doesn't exist, or its results have expired "
            "(they are kept for 7 days)",
        )
    return job


@app.get("/jobs/{job_id}", response_model=JobResponse)
def job_status(job_id: Annotated[str, _JOB_ID]) -> JobResponse:
    return _job_response(_get_job(job_id))


@app.get("/jobs/{job_id}/alerts", response_model=AlertsResponse)
def job_alerts(job_id: Annotated[str, _JOB_ID]) -> AlertsResponse:
    job = _get_job(job_id)
    if job.status is not JobState.COMPLETE:
        detail = f"job is {job.status.value}; results are available once it is complete"
        if job.error:
            detail += f" (error: {job.error})"
        raise HTTPException(409, detail)
    result = jobs.result(job)
    if result is None:
        raise HTTPException(500, f"job {job_id!r} is complete but its results are missing from storage")
    return AlertsResponse(
        clip_id=job.job_id,
        zone_name=job.zone.name,
        frame_width=job.zone.frame_width,
        frame_height=job.zone.frame_height,
        clip_fps=result.clip_fps,
        sample_fps=result.sample_fps,
        frames_read=result.frames_read,
        frames_processed=result.frames_processed,
        processing_time_s=result.processing_time_s,
        alerts=[
            # `snapshot` is the object's key in the store; the URL is a
            # signed, expiring R2 link (or an API route locally), issued
            # fresh on every request.
            AlertOut(**a.model_dump(), snapshot_url=jobs.store.url(a.snapshot))
            for a in result.alerts
        ],
    )


def _serve_stored(key: str):
    """Serve an object from the local store; with R2, redirect to a signed link."""
    store = jobs.store
    if store.kind != "local":
        return RedirectResponse(store.url(key), status_code=307)
    path = store.local_path(key)
    if not path.is_file():
        raise HTTPException(404, f"not found: {key}")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/jobs/{job_id}/reference.jpg", response_class=FileResponse)
def job_reference_frame(job_id: Annotated[str, _JOB_ID]):
    _get_job(job_id)
    return _serve_stored(reference_key(job_id))


@app.get("/jobs/{job_id}/snapshots/{name}", response_class=FileResponse)
def job_snapshot(
    job_id: Annotated[str, _JOB_ID],
    name: Annotated[str, Path_(pattern=r"^frame_\d{5}\.jpg$")],
):
    job = _get_job(job_id)
    if job.status is not JobState.COMPLETE:
        raise HTTPException(404, f"no snapshot {name!r} for job {job_id!r}")
    return _serve_stored(snapshot_key(job_id, name))


@app.get("/clips/{clip_id}/snapshots/{name}", response_class=FileResponse)
def snapshot(
    clip_id: str,
    # Only names clip_processor generates. Rejecting anything else up front
    # means no request can reach outside the snapshot directory.
    name: Annotated[str, Path_(pattern=r"^frame_\d{5}\.jpg$")],
) -> FileResponse:
    get_clip(clip_id)
    path = _snapshot_dir(clip_id) / name
    if clip_id not in _results or not path.is_file():
        raise HTTPException(404, f"no snapshot {name!r} for clip {clip_id!r}")
    return FileResponse(path, media_type="image/jpeg")
