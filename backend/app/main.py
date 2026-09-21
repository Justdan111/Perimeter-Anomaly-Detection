"""FastAPI application for the perimeter anomaly detection service.

Day 1 scope: the skeleton and `/health` only. Clip processing and alert
retrieval endpoints arrive on Day 2, once `clip_processor.py` exists.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from app.services.detector import Detector

detector = Detector()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model at startup rather than on the first request.

    Loading YOLO26-N takes a few seconds. Paying that cost at startup means
    the first real request isn't the one that waits for it, and it means a
    missing weights file fails loudly at boot instead of halfway through
    processing a clip.
    """
    detector.load()
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


class HealthResponse(BaseModel):
    """Response body for `GET /health`."""

    status: str
    model_loaded: bool
    model_weights: str


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
    )
