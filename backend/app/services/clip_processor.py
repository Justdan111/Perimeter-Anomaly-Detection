"""Run a recorded clip through the detector and the zone check, producing alerts.

Structure: a thin OpenCV read loop (`process_clip`) around small pure
functions that hold all the actual decisions — which classes count, which
point of a box is tested, when a frame is sampled, what timestamp a frame
has, whether the clip's resolution matches the zone. The pure functions are
unit tested with plain fixtures (tests/test_clip_processor.py); the loop was
verified by hand against the real sample clip.

Scope notes:

- Frames are independent. There is no tracking: the same person standing in
  the zone for two seconds is one alert per processed frame, not one alert.
  Cross-frame tracking is permanently out of scope (docs/PROJECT.md).
- Recorded clips only. `process_clip` takes a file path. The Day 6 live-feed
  plan swaps the `cv2.VideoCapture(path)` source for a `FrameSource`; the
  pure functions below don't care where frames come from.

Anchor-point decision
---------------------
A detection is "in the zone" when the BOTTOM-CENTRE of its bounding box is
inside the polygon. The zone is drawn on the ground (a roadway, a yard), and
the bottom of a box is where a person's feet or a vehicle's tyres meet that
ground. The box centre would put a pedestrian's waist, projected in
perspective, well above where they stand, so a person on the pavement behind
the zone can have a centre inside it.

KNOWN LIMITATION: YOLO clips boxes to the frame. A person walking out of the
bottom of the frame gets a box whose bottom edge is exactly the frame edge
(y = frame height), so their anchor is on the frame edge, not at their real,
off-screen feet. Their reported position is therefore wrong, and can be
nearer to or farther from the zone than they really are. For the sample
clip's zone this is harmless, because the zone's lowest vertex is at y=628,
above the frame edge, so a clipped anchor at y=720 is never inside it.
It would matter for a zone that reaches the bottom edge of the frame. Fixing
it properly (e.g. estimating foot position from box height) is out of scope
for this sprint.

Sampling-rate decision
----------------------
`sample_fps` is a rate in clip-time (samples per second of video), not a
frame stride. "Check twice a second" means the same thing on a 24 fps and a
60 fps clip; "every 12th frame" does not. `None` (the default) processes
every frame.

The default is every frame because, with no cross-frame state, sampling can
only remove alerts, never change or add one: a sampled run's alerts are
exactly the full run's alerts for the frames it kept. So a full run is the
reference answer that any sampled run can be checked against, and on the
5 s / 120-frame sample clip it costs a few seconds on a laptop CPU. The
trade-off: CPU cost and alert count both scale linearly with frames
processed. For long clips or a slow host (Render free tier, Day 5), pass
something like `sample_fps=2`. A person walking at ~1.4 m/s moves ~0.7 m
between samples at 2 fps, far less than the width of any sensible zone, so
someone crossing the zone is still caught, at up to 0.5 s of timestamp
error. The right value for the deployed service is a Day 5 measurement,
not a guess made here.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from app.models.schemas import Alert, ClipResult, Detection, Point, Zone
from app.services.zone_check import point_in_polygon

logger = logging.getLogger(__name__)

# COCO class labels that count as an intrusion. Everything else the model
# sees — on the sample clip that is mostly "traffic light", "handbag" and
# "backpack", all above the detector's confidence threshold — is dropped
# before the zone check. "bicycle" is deliberately absent: a ridden bicycle
# comes with a detected "person", which already alerts.
PERSON_CLASSES: frozenset[str] = frozenset({"person"})
VEHICLE_CLASSES: frozenset[str] = frozenset({"car", "truck", "bus", "motorcycle"})
ALLOWED_CLASSES: frozenset[str] = PERSON_CLASSES | VEHICLE_CLASSES

# What an uploader chooses between (Phase 1). The same list as above, split
# in two — class selection exposes the existing filter, it adds no classes.
CLASS_GROUPS: dict[str, frozenset[str]] = {
    "person": PERSON_CLASSES,
    "vehicle": VEHICLE_CLASSES,
    "both": ALLOWED_CLASSES,
}

# Absorbs float error when a frame's clip time lands exactly on a sample
# boundary (e.g. frame 3 of a 30 fps clip at 10 fps is t = 0.1 s exactly).
# Far below one frame at any real frame rate.
_SAMPLE_EPSILON = 1e-9

# Snapshots are review thumbnails, not evidence-grade stills: half of the
# sample clip's 1280x720 is enough to recognise a person or a car, and at
# quality 70 a snapshot is ~53 KB vs ~262 KB at full size and quality 90
# (measured on the sample clip: 5.8 MB vs 28.3 MB for its 108 snapshots).
# Width is capped rather than forced to 640x360, so another aspect ratio isn't
# squashed and a smaller clip isn't upscaled. Alert bboxes stay in FULL-frame
# pixel coordinates; the dashboard scales them onto the thumbnail.
_SNAPSHOT_MAX_WIDTH = 640
_SNAPSHOT_JPEG_QUALITY = 70


class FrameSizeMismatchError(ValueError):
    """The clip's frames are not the size the zone was drawn against."""


class ClipReadError(ValueError):
    """The clip file can't be opened or decoded to the end."""


class FrameDetector(Protocol):
    """Anything with `detect(frame) -> list[Detection]`, e.g. `Detector`."""

    def detect(self, frame: np.ndarray) -> list[Detection]: ...


# --- pure logic -----------------------------------------------------------


def filter_allowed_classes(
    detections: Iterable[Detection],
    allowed: Iterable[str] = ALLOWED_CLASSES,
) -> list[Detection]:
    """Keep only detections whose class is in `allowed`, preserving order.

    Matching is exact: "carrot" is not "car" and "Person" is not "person".
    """
    allowed = frozenset(allowed)
    return [d for d in detections if d.class_name in allowed]


def check_frame_size(width: int, height: int, zone: Zone) -> None:
    """Raise if a frame's size differs from the size the zone was drawn on.

    The zone's points are pixel coordinates. Applied to a frame of any other
    size, every point is in the wrong place and the alert list is silently
    wrong, so a mismatch is an error, not something to rescale around.
    Same aspect ratio is not enough: a zone drawn at 1280x720 applied to a
    640x360 frame is 2x off on both axes.
    """
    if (width, height) != (zone.frame_width, zone.frame_height):
        raise FrameSizeMismatchError(
            f"clip frames are {width}x{height} but zone {zone.name!r} was "
            f"drawn against {zone.frame_width}x{zone.frame_height}; redraw "
            f"the zone for this resolution or resize the clip"
        )


def anchor_point(bbox: tuple[float, float, float, float]) -> Point:
    """The point of a box tested against the zone: bottom-centre.

    See "Anchor-point decision" in the module docstring, including the
    limitation for boxes clipped by the bottom of the frame.
    """
    x1, _, x2, y2 = bbox
    return ((x1 + x2) / 2, y2)


def frame_timestamp(frame_index: int, clip_fps: float) -> float:
    """Clip time of a frame, in seconds: frame_index / fps.

    Assumes a constant frame rate. True of the sample clip (re-encoded by
    ffmpeg to H.264 CFR); a variable-frame-rate phone recording would need
    per-frame timestamps from the container instead.
    """
    if clip_fps <= 0:
        raise ValueError(f"clip fps must be positive, got {clip_fps}")
    return frame_index / clip_fps


def should_sample(
    frame_index: int, clip_fps: float, sample_fps: float | None
) -> bool:
    """Whether to run the detector on this frame.

    Picks the first frame at or after each sample instant (0, 1/sample_fps,
    2/sample_fps, ... seconds of clip time). Rates at or above the clip's
    own fps can't invent frames, so they mean "every frame".
    """
    if sample_fps is None:
        return True
    if sample_fps <= 0:
        raise ValueError(f"sample_fps must be positive or None, got {sample_fps}")
    if sample_fps >= clip_fps:
        return True

    # Number of sample instants at or before this frame's time vs. the
    # previous frame's. If it went up, a sample instant fell in between and
    # this is the first frame at or after it.
    def samples_due(i: int) -> int:
        return math.floor(i * sample_fps / clip_fps + _SAMPLE_EPSILON)

    return frame_index == 0 or samples_due(frame_index) > samples_due(frame_index - 1)


def alerts_for_frame(
    detections: Iterable[Detection],
    zone: Zone,
    frame_index: int,
    clip_fps: float,
    snapshot: str,
    allowed_classes: Iterable[str] = ALLOWED_CLASSES,
) -> list[Alert]:
    """Turn one frame's detections into alerts for those inside the zone.

    Filters to the allowed classes, tests each survivor's bottom-centre
    anchor against the zone, and builds one `Alert` per hit, in detection
    order. No de-duplication (no tracking).
    """
    timestamp_s = frame_timestamp(frame_index, clip_fps)
    alerts: list[Alert] = []
    for d in filter_allowed_classes(detections, allowed_classes):
        anchor = anchor_point(d.bbox)
        if point_in_polygon(anchor, zone.points):
            alerts.append(
                Alert(
                    timestamp_s=timestamp_s,
                    frame_index=frame_index,
                    class_name=d.class_name,
                    confidence=d.confidence,
                    bbox=d.bbox,
                    anchor=anchor,
                    zone_name=zone.name,
                    snapshot=snapshot,
                )
            )
    return alerts


def count_sampled_frames(frame_count: int, clip_fps: float, sample_fps: float | None) -> int:
    """How many of `frame_count` frames `should_sample` picks (for progress)."""
    return sum(should_sample(i, clip_fps, sample_fps) for i in range(frame_count))


def whole_frame_zone(width: int, height: int, name: str = "Whole frame") -> Zone:
    """A zone covering the entire frame: every allowed detection alerts.

    The Phase 1 default for uploads, which have no drawn zone. Because the
    boundary counts as inside, this includes boxes cut off by the frame
    edge — so the anchor-clipping limitation can't change a verdict here.
    """
    w, h = float(width), float(height)
    return Zone(
        name=name,
        frame_width=width,
        frame_height=height,
        points=[(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)],
    )


def snapshot_filename(frame_index: int) -> str:
    return f"frame_{frame_index:05d}.jpg"


# --- I/O ------------------------------------------------------------------


def load_zone(path: Path | str) -> Zone:
    """Load and validate a zone from its JSON config file."""
    return Zone.model_validate_json(Path(path).read_text())


def process_clip(
    clip_path: Path | str,
    zone: Zone,
    detector: FrameDetector,
    snapshot_dir: Path | str,
    sample_fps: float | None = None,
    allowed_classes: Iterable[str] = ALLOWED_CLASSES,
    on_progress: Callable[[int], None] | None = None,
) -> ClipResult:
    """Run a clip frame by frame and return every in-zone alert.

    Args:
        clip_path: A video file OpenCV can read.
        zone: The watched area. Its frame size must match the clip's.
        detector: Usually `app.services.detector.Detector`, already loaded
            (model load time is not counted in `processing_time_s`).
        snapshot_dir: Where to write one JPEG per frame that produced at
            least one alert. Created if missing. Frames without alerts are
            not written.
        sample_fps: Samples per second of clip time; None for every frame.
            See "Sampling-rate decision" in the module docstring.
        allowed_classes: Which detected classes can alert (default: people
            and vehicles). See `CLASS_GROUPS`.
        on_progress: Called with the number of frames processed so far,
            after each processed frame. For reporting only.

    Raises:
        FileNotFoundError: The clip doesn't exist.
        ClipReadError: OpenCV can't open the clip, it reports no usable
            fps, or it stops decoding before the frame count in its own
            header (a truncated or corrupt file).
        FrameSizeMismatchError: The clip's frame size isn't the zone's.

    On any failure, snapshots this call already wrote are deleted, so a
    failed run never leaves a partial set behind for something to serve.
    """
    clip_path = Path(clip_path)
    snapshot_dir = Path(snapshot_dir)
    if not clip_path.exists():
        raise FileNotFoundError(f"clip not found: {clip_path}")
    if sample_fps is not None and sample_fps <= 0:
        raise ValueError(f"sample_fps must be positive or None, got {sample_fps}")

    capture = cv2.VideoCapture(str(clip_path))
    written: list[Path] = []
    try:
        if not capture.isOpened():
            raise ClipReadError(f"OpenCV could not open {clip_path.name} as a video")

        clip_fps = capture.get(cv2.CAP_PROP_FPS)
        if not clip_fps or clip_fps <= 0:
            raise ClipReadError(f"{clip_path.name} reports no usable fps ({clip_fps})")

        # Fail before any inference if the container's declared size is
        # wrong. Each decoded frame is checked too (below): the header can
        # disagree with the actual frames, and it costs nothing.
        check_frame_size(
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            zone,
        )
        declared_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        snapshot_dir.mkdir(parents=True, exist_ok=True)
        alerts: list[Alert] = []
        frames_read = 0
        frames_processed = 0
        started = time.perf_counter()

        # Read until the decoder runs out; the header's frame count is only
        # used afterwards, as a check. Every frame is decoded even when
        # sampling: seeking is unreliable across codecs, and decoding is
        # cheap next to inference.
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames_read += 1

            if should_sample(frame_index, clip_fps, sample_fps):
                height, width = frame.shape[:2]
                check_frame_size(width, height, zone)

                snapshot = snapshot_filename(frame_index)
                frame_alerts = alerts_for_frame(
                    detector.detect(frame),
                    zone,
                    frame_index,
                    clip_fps,
                    snapshot,
                    allowed_classes,
                )
                frames_processed += 1
                if on_progress is not None:
                    on_progress(frames_processed)

                if frame_alerts:
                    _write_snapshot(snapshot_dir / snapshot, frame)
                    written.append(snapshot_dir / snapshot)
                    alerts.extend(frame_alerts)

            frame_index += 1

        # `read()` returning False means "no more frames" and "can't decode
        # the rest" alike. A truncated file opens fine and its header still
        # declares the full length, so without this check it would return
        # a result for the first part as if that were the whole clip.
        # Checked against re-encodes of the sample clip as MP4, WebM, MKV,
        # AVI (MJPEG), MPEG-TS and a variable-frame-rate MP4: in every case
        # the header count matched the decoded count exactly. If some other
        # file's header over-estimates its length, this raises a false
        # error: loud, and the message says what was compared. A count of
        # 0 or less means "unknown" and skips the check.
        if declared_frames > 0 and frames_read < declared_frames:
            raise ClipReadError(
                f"{clip_path.name} decoded {frames_read} of {declared_frames} frames "
                f"declared in its header; the file is truncated or corrupt"
            )
    except BaseException:
        _remove_snapshots(written)
        raise
    finally:
        capture.release()

    return ClipResult(
        clip_fps=clip_fps,
        sample_fps=sample_fps,
        frames_read=frames_read,
        frames_processed=frames_processed,
        processing_time_s=time.perf_counter() - started,
        alerts=alerts,
    )


def _remove_snapshots(paths: list[Path]) -> None:
    """Best-effort delete of a failed run's snapshots.

    Never raises: this runs while another exception is propagating, and if
    the directory has become unwritable (the likely cause of a write
    failure in the first place), deleting fails too. That second error must
    not replace the first in what the caller sees. Files that can't be
    removed are logged; they are also never served, because the API only
    serves snapshots belonging to a successful run.
    """
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError as e:
            logger.warning("could not remove snapshot from failed run: %s (%s)", path, e)


def _write_snapshot(path: Path, frame: np.ndarray) -> None:
    """Write a downscaled copy of the frame, with no boxes drawn on it.

    The dashboard draws the zone and box on top from the alert's data.
    """
    height, width = frame.shape[:2]
    scale = min(1.0, _SNAPSHOT_MAX_WIDTH / width)
    if scale < 1.0:
        size = (round(width * scale), round(height * scale))
        # INTER_AREA is OpenCV's recommended filter for shrinking: it averages
        # source pixels instead of skipping them, so thin edges don't alias.
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    ok = cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, _SNAPSHOT_JPEG_QUALITY])
    if not ok:
        raise OSError(f"failed to write snapshot: {path}")
