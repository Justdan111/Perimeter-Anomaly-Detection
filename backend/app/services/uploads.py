"""Validation for user-uploaded clips.

Runs at upload time, before a job is created, so a bad file gets an HTTP
error the uploader sees immediately rather than a job that fails later.

What can be checked cheaply here: the container type (from the first bytes),
that OpenCV opens the file and decodes its first frame, and the limits below
(from the container header). What can't: damage further into the file — a
truncated upload decodes fine at the start. That surfaces during processing,
where Day 4's frame-count check fails the job with a clear reason.

Limits — see "Uploads" in the README for how these were chosen and measured.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


@dataclass(frozen=True)
class UploadLimits:
    max_bytes: int
    max_duration_s: float
    # Long/short side rather than width/height, so an upright phone video
    # (1080 wide, 1920 tall) is judged the same as a landscape one.
    max_long_side: int
    max_short_side: int
    min_short_side: int
    max_fps: float
    min_frames: int


DEFAULT_LIMITS = UploadLimits(
    max_bytes=_env_int("PERIMETER_MAX_UPLOAD_MB", 100) * 1024 * 1024,
    max_duration_s=_env_float("PERIMETER_MAX_UPLOAD_SECONDS", 60.0),
    max_long_side=1920,
    max_short_side=1080,
    min_short_side=120,
    max_fps=60.0,
    min_frames=2,
)


class UploadRejected(ValueError):
    """The upload isn't acceptable. Carries the HTTP status to answer with."""

    def __init__(self, message: str, status_code: int = 422):
        super().__init__(message)
        self.status_code = status_code


# --- container sniffing ------------------------------------------------------------


def sniff_container(head: bytes) -> str | None:
    """Identify a video container from the file's first bytes, or None.

    An allow-list of real video containers. OpenCV can't be the judge on its
    own: it happily opens a JPEG (as a one-frame clip) and an animated GIF.
    Needs at least the first 193 bytes to recognise MPEG-TS.
    """
    if len(head) >= 12 and head[4:8] == b"ftyp":
        return "mp4"  # MP4, MOV (QuickTime), M4V, 3GP — the ISO base media family
    if head.startswith(b"\x1a\x45\xdf\xa3"):
        return "mkv"  # Matroska, including WebM
    if head.startswith(b"RIFF") and head[8:12] == b"AVI ":
        return "avi"
    # MPEG transport stream: 188-byte packets, each starting with 0x47.
    if len(head) >= 189 and head[0] == 0x47 and head[188] == 0x47:
        return "ts"
    return None


# --- probing and limits ------------------------------------------------------------


@dataclass(frozen=True)
class ClipProbe:
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps


def probe_clip(path: Path) -> tuple[ClipProbe, np.ndarray]:
    """Open the clip and decode its first frame; return both.

    Width and height come from the decoded frame, not the header: for a
    phone video with a rotation flag, the frame is what processing will see.
    The frame itself becomes the job's reference image for the zone view.
    """
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise UploadRejected("the file could not be opened as a video")
        ok, frame = capture.read()
        if not ok or frame is None:
            raise UploadRejected("the video's first frame could not be decoded")
        height, width = frame.shape[:2]
        probe = ClipProbe(
            width=width,
            height=height,
            fps=float(capture.get(cv2.CAP_PROP_FPS)),
            frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
        return probe, frame
    finally:
        capture.release()


def check_probe(probe: ClipProbe, limits: UploadLimits = DEFAULT_LIMITS) -> None:
    """Raise `UploadRejected` if the clip is outside the limits."""
    if not probe.fps or probe.fps <= 0:
        raise UploadRejected("the video reports no usable frame rate")
    if probe.fps > limits.max_fps:
        raise UploadRejected(
            f"frame rate {probe.fps:.0f} fps is above the {limits.max_fps:.0f} fps limit"
        )
    if probe.frame_count <= 0:
        raise UploadRejected(
            "the video's length can't be determined from its header, so processing "
            "time can't be bounded; re-encode it (e.g. to MP4) and try again"
        )
    if probe.frame_count < limits.min_frames:
        raise UploadRejected("the file is a single image, not a video")

    long_side, short_side = max(probe.width, probe.height), min(probe.width, probe.height)
    if long_side > limits.max_long_side or short_side > limits.max_short_side:
        raise UploadRejected(
            f"resolution {probe.width}x{probe.height} is above the 1080p limit "
            f"({limits.max_long_side}x{limits.max_short_side}, either orientation)"
        )
    if short_side < limits.min_short_side:
        raise UploadRejected(
            f"resolution {probe.width}x{probe.height} is too small "
            f"(shorter side under {limits.min_short_side} px)"
        )

    if probe.duration_s > limits.max_duration_s:
        raise UploadRejected(
            f"the video is {probe.duration_s:.1f} s long; the limit is "
            f"{limits.max_duration_s:.0f} s"
        )
