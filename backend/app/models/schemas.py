"""Pydantic data models shared across the service.

Everything that crosses a component boundary is one of these, so no component
has to know another's internals. In particular `Detection` is what `detector.py`
returns — downstream code works with that, never with an Ultralytics `Results`
object, so swapping the model out later touches exactly one file.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

# A point in frame pixel coordinates: x right, y down (OpenCV convention).
Point = tuple[float, float]

# Dominant-colour names (app/services/colors.py). "mixed" = no colour clearly
# dominates; "unknown" = too small to judge.
ColorName = Literal[
    "black", "white", "gray", "red", "orange", "yellow",
    "green", "blue", "purple", "pink", "brown", "mixed", "unknown",
]


class Zone(BaseModel):
    """A polygon defining the watched area, in frame pixel coordinates.

    This is configuration, not code: zones are loaded from a JSON file
    (see `app/data/sample_clip/zone.json`) so the watched area can change
    without touching the pipeline.
    """

    name: str = Field(
        description="Human-readable label for the zone, used in alerts.",
    )
    points: list[Point] = Field(
        min_length=3,
        description=(
            "Polygon vertices in order. The polygon is implicitly closed — do "
            "not repeat the first vertex at the end."
        ),
    )
    # Pixel coordinates are meaningless without knowing what they are relative
    # to. Recording the frame size the zone was drawn against makes a
    # resolution mismatch (a zone drawn on 1080p applied to a 720p frame, with
    # every coordinate silently 1.5x off) a loud failure rather than a subtly
    # wrong alert list. NOTE: this is one field beyond docs/SPEC.md section 7,
    # added deliberately for that reason.
    frame_width: int = Field(
        gt=0, description="Width of the frame the points were drawn against."
    )
    frame_height: int = Field(
        gt=0, description="Height of the frame the points were drawn against."
    )

    @field_validator("points")
    @classmethod
    def reject_degenerate_polygon(cls, points: list[Point]) -> list[Point]:
        """Reject polygons that enclose no area.

        A zone with every vertex identical, or all vertices on one line, has
        zero area and can never contain anything — it would silently never
        alert, which is the failure mode this project most needs to avoid.
        """
        distinct = {(round(x, 6), round(y, 6)) for x, y in points}
        if len(distinct) < 3:
            raise ValueError(
                f"zone polygon needs at least 3 distinct vertices, "
                f"got {len(distinct)}"
            )

        # Shoelace formula — twice the signed area. Zero means collinear.
        area2 = 0.0
        for i in range(len(points)):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % len(points)]
            area2 += x1 * y2 - x2 * y1
        if abs(area2) < 1e-9:
            raise ValueError(
                "zone polygon encloses no area (all vertices are collinear)"
            )

        return points


class Detection(BaseModel):
    """One object the model found in one frame.

    Deliberately plain data. No frame, no model handle, no tracking identity —
    detections are independent per frame, which is an explicit scope decision
    (see docs/PROJECT.md, "What's explicitly OUT of scope").
    """

    class_name: str = Field(description='Model class label, e.g. "person", "car".')
    confidence: float = Field(
        ge=0.0, le=1.0, description="Model confidence in this detection, 0..1."
    )
    bbox: tuple[float, float, float, float] = Field(
        description="Bounding box as (x1, y1, x2, y2) in frame pixel coordinates."
    )

    @field_validator("bbox")
    @classmethod
    def reject_inverted_bbox(
        cls, bbox: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        """Require x1 <= x2 and y1 <= y2 so downstream geometry is safe."""
        x1, y1, x2, y2 = bbox
        if x2 < x1 or y2 < y1:
            raise ValueError(
                f"bbox must be (x1, y1, x2, y2) with x1 <= x2 and y1 <= y2, got {bbox}"
            )
        return bbox


class Alert(BaseModel):
    """A person or vehicle detected inside the zone, in one frame of a clip.

    One alert per in-zone detection per processed frame. There is no
    de-duplication across frames: a car parked in the zone for 5 seconds is
    ~120 alerts at full frame rate, because detections are independent per
    frame and cross-frame tracking is explicitly out of scope.

    NOTE: SPEC section 7 lists `timestamp_s`, `class_name` and `snapshot`.
    The extra fields are deliberate. `frame_index`, `bbox` and `anchor`
    make each alert checkable by hand: you can open the frame, see the box,
    and see exactly which point was tested against the zone. `confidence`
    lets a reviewer tell a solid detection from a marginal one.
    """

    timestamp_s: float = Field(
        ge=0.0,
        description="Position in the clip, computed as frame_index / clip fps.",
    )
    frame_index: int = Field(ge=0, description="0-based index of the frame in the clip.")
    class_name: str = Field(description='Detected class, e.g. "person", "car".')
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: tuple[float, float, float, float] = Field(
        description="Bounding box as (x1, y1, x2, y2) in frame pixel coordinates."
    )
    anchor: Point = Field(
        description=(
            "The point that was tested against the zone: bottom-centre of the "
            "bbox. For a box clipped by the bottom of the frame, this sits on "
            "the frame edge rather than at the object's real (off-screen) feet."
        )
    )
    zone_name: str = Field(description="Name of the zone the anchor fell inside.")
    # Phase 2. Optional so that alerts recorded before colour extraction
    # existed (Phase 1 results in R2) still load: missing means "not recorded".
    color: ColorName | None = Field(
        default=None,
        description="Dominant colour of a vehicle or object (not set for people).",
    )
    upper_color: ColorName | None = Field(
        default=None, description="A person's upper-body clothing colour."
    )
    lower_color: ColorName | None = Field(
        default=None, description="A person's lower-body clothing colour."
    )
    snapshot: str = Field(
        description=(
            "Filename of a downscaled (max 640 px wide), unannotated JPEG of the "
            "frame, relative to "
            "the snapshot directory the clip was processed with. Alerts from "
            "the same frame share one file."
        )
    )


class ClipResult(BaseModel):
    """Everything produced by processing one clip."""

    clip_fps: float = Field(gt=0.0, description="Frame rate reported by the clip.")
    sample_fps: float | None = Field(
        description="Requested sampling rate; None means every frame was processed."
    )
    frames_read: int = Field(ge=0, description="Frames decoded from the clip.")
    frames_processed: int = Field(ge=0, description="Frames actually run through the detector.")
    processing_time_s: float = Field(
        ge=0.0, description="Wall-clock time for the whole clip, model load excluded."
    )
    alerts: list[Alert]
