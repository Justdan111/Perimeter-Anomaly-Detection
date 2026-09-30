"""Live frame ingestion (Phase 4, Stage 4a — validation, see docs/PHASE4.md).

A local agent next to the camera samples frames and POSTs them one at a time;
each is run through the existing detector and zone check here, and the result
is pushed to connected dashboards over a WebSocket. `clip_processor` is not
involved: a live frame is not a clip — it has no fps, no frame index, no
snapshot directory, and the stream never ends.

Stage 4a decisions, stated so they're decisions rather than accidents:

- **The whole frame is the zone** (as for uploads): there is no zone editor,
  and a webcam has no committed zone.
- **Same model instance, same lock** as uploads and the sample clip. The
  model isn't safe to share across threads (docs/PHASE1.md), and a second
  instance is exactly the kind of extra memory Phase 3 showed the free tier
  can't take. Whether sharing the lock makes live and uploads conflict is the
  thing Stage 4a has to observe, so it's left exactly as it is.
- **A frame waits at most `LIVE_LOCK_WAIT_S` for the model, then is dropped**
  (503), rather than queueing behind an upload that can hold the model for
  minutes. A live frame that arrives late is worthless; a queue of them is a
  memory leak.
- **Nothing is stored.** Frames are decoded, checked and dropped; the
  WebSocket message carries a downscaled JPEG for the dashboard and is not
  kept. No debouncing: every frame's detections are reported (Stage 4b).
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import resource
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from app.models.schemas import Detection
from app.services.zone_check import point_in_polygon

logger = logging.getLogger(__name__)

LIVE_CLASSES: frozenset[str] = frozenset({"person", "car", "truck", "bus", "motorcycle"})
LIVE_LOCK_WAIT_S = float(os.environ.get("PERIMETER_LIVE_LOCK_WAIT_S", 1.0))
MAX_FRAME_BYTES = 2 * 1024 * 1024
MAX_FRAME_SIDE = 1920
PREVIEW_WIDTH = 480


class FrameRejected(ValueError):
    """The request body isn't a usable frame."""


def decode_frame(body: bytes) -> np.ndarray:
    """JPEG/PNG bytes -> BGR frame, or FrameRejected with the reason."""
    if not body:
        raise FrameRejected("empty body; send one JPEG frame")
    if len(body) > MAX_FRAME_BYTES:
        raise FrameRejected(f"frame is {len(body)} bytes; the limit is {MAX_FRAME_BYTES}")
    frame = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise FrameRejected("body is not a decodable image (send a JPEG)")
    h, w = frame.shape[:2]
    if max(h, w) > MAX_FRAME_SIDE:
        raise FrameRejected(f"frame is {w}x{h}; the longest side may be at most {MAX_FRAME_SIDE}")
    return frame


def whole_frame(width: int, height: int) -> list[tuple[float, float]]:
    w, h = float(width), float(height)
    return [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]


def live_alerts(
    detections: list[Detection],
    zone_points: list[tuple[float, float]],
    allowed: frozenset[str] = LIVE_CLASSES,
) -> list[dict]:
    """Detections of an allowed class whose anchor (bottom-centre of the box,
    the same point clip processing tests) is inside the zone."""
    out = []
    for d in detections:
        if d.class_name not in allowed:
            continue
        x1, y1, x2, y2 = d.bbox
        anchor = ((x1 + x2) / 2, y2)
        if point_in_polygon(anchor, zone_points):
            out.append(
                {
                    "class_name": d.class_name,
                    "confidence": round(d.confidence, 3),
                    "bbox": [round(v, 1) for v in d.bbox],
                    "anchor": [round(anchor[0], 1), round(anchor[1], 1)],
                }
            )
    return out


def preview_jpeg_b64(frame: np.ndarray, width: int = PREVIEW_WIDTH) -> str:
    h, w = frame.shape[:2]
    if w > width:
        frame = cv2.resize(frame, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""


# --- measurement --------------------------------------------------------------


def memory_mb() -> dict:
    """Process peak RSS and the container's own count (what an OOM kill is
    based on — see Phase 3: RSS overstates it). Missing files -> omitted."""
    # ru_maxrss is KiB on Linux (the deployed host) but bytes on macOS.
    scale = 2**20 if sys.platform == "darwin" else 2**10
    out = {"peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale, 1)}
    for key, name in (("cgroup_current_mb", "memory.current"), ("cgroup_peak_mb", "memory.peak")):
        try:
            out[key] = round(int(Path("/sys/fs/cgroup", name).read_text()) / 2**20, 1)
        except (OSError, ValueError):
            pass
    return out


class LiveStats:
    """Counters for the live path. `process_started_at` changes on restart,
    which is how an agent polling /live/stats notices the instance died."""

    def __init__(self) -> None:
        self.process_started_at = time.time()
        self.pid = os.getpid()
        self._lock = threading.Lock()
        self.received = 0
        self.processed = 0
        self.dropped_busy = 0
        self.rejected = 0
        self.alerts = 0
        self._detect_ms: deque[float] = deque(maxlen=120)
        self._wait_ms: deque[float] = deque(maxlen=120)

    def count(self, field: str, n: int = 1) -> None:
        with self._lock:
            setattr(self, field, getattr(self, field) + n)

    def timing(self, wait_ms: float, detect_ms: float) -> None:
        with self._lock:
            self._wait_ms.append(wait_ms)
            self._detect_ms.append(detect_ms)

    def snapshot(self) -> dict:
        def med(xs):
            s = sorted(xs)
            return round(s[len(s) // 2], 1) if s else None

        with self._lock:
            return {
                "process_started_at": self.process_started_at,
                "pid": self.pid,
                "uptime_s": round(time.time() - self.process_started_at, 1),
                "frames_received": self.received,
                "frames_processed": self.processed,
                "frames_dropped_model_busy": self.dropped_busy,
                "frames_rejected": self.rejected,
                "alerts": self.alerts,
                "recent_detect_ms_median": med(self._detect_ms),
                "recent_detect_ms_max": round(max(self._detect_ms), 1) if self._detect_ms else None,
                "recent_lock_wait_ms_median": med(self._wait_ms),
                **memory_mb(),
            }


# --- WebSocket fan-out --------------------------------------------------------


class LiveHub:
    """Dashboards subscribed to the live feed. Sends are best-effort: a
    client that can't keep up or has gone away is dropped, never waited on."""

    def __init__(self) -> None:
        self._clients: set = set()

    @property
    def client_count(self) -> int:
        return len(self._clients)

    def add(self, ws) -> None:
        self._clients.add(ws)

    def remove(self, ws) -> None:
        self._clients.discard(ws)

    async def broadcast(self, message: dict, timeout_s: float = 2.0) -> int:
        """Send to every client; returns how many received it."""
        clients = list(self._clients)
        if not clients:
            return 0

        async def send(ws) -> bool:
            try:
                await asyncio.wait_for(ws.send_json(message), timeout_s)
                return True
            except Exception:
                self.remove(ws)
                return False

        results = await asyncio.gather(*(send(ws) for ws in clients))
        return sum(results)
