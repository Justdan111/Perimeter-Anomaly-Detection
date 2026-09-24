"""FastAPI application for the perimeter anomaly detection service.

Endpoints (all JSON unless noted):

- `GET  /health`                              model loaded?
- `GET  /clips/{clip_id}`                     clip info + its zone, from config
- `GET  /clips/{clip_id}/reference-frame`     JPEG the zone is drawn over
- `POST /clips/{clip_id}/process`             run the clip through the pipeline
- `GET  /clips/{clip_id}/alerts`              alerts from the last run
- `GET  /clips/{clip_id}/snapshots/{name}`    JPEG snapshot referenced by an alert

MVP scope: the only clip is the committed sample (`clip_id = "sample"`), and
results live in memory — a restart forgets them, and processing again
replaces them. No upload, no persistence, no live input (see docs/PROJECT.md).
"""

import logging
import os
import shutil
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi import Path as Path_  # pathlib.Path is used too
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.models.schemas import Alert, ClipResult, Zone
from app.services.clip_processor import (
    ClipReadError,
    FrameDetector,
    FrameSizeMismatchError,
    load_zone,
    process_clip,
)
from app.services.detector import Detector

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
# and two runs would write into the same snapshot directory.
_processing_lock = threading.Lock()


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
    global model_load_error
    try:
        detector.load()
        model_load_error = None
    except Exception as e:  # ultralytics raises TypeError, UnpicklingError, ...
        model_load_error = f"{type(e).__name__}: {e}"
        logger.error("model failed to load; service is degraded: %s", model_load_error)
    yield


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


# --- response models ---------------------------------------------------------


class HealthResponse(BaseModel):
    """Response body for `GET /health`."""

    status: str
    model_loaded: bool
    model_weights: str
    model_error: str | None = Field(
        description="Why the model failed to load at startup; null when loaded."
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
        status="ok" if detector.is_loaded else "degraded",
        model_loaded=detector.is_loaded,
        model_weights=detector.weights_path.name,
        model_error=model_load_error,
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
