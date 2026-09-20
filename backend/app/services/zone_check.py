"""Point-in-polygon zone check.

A pure function: coordinates in, boolean out. No model, no video file, no I/O,
no imports from anywhere else in this project. That isolation is deliberate —
it is what lets this logic be tested on Day 1 with nothing but plain numbers,
and it is the piece whose correctness every alert depends on.

Chosen approach: the ray casting (even-odd) rule, with an explicit on-boundary
check in front of it. Ray casting is used because it is easy to verify by hand
against a drawing and works on concave polygons, which matter here — a zone
drawn around a yard minus a building corner is not convex.
"""

from collections.abc import Sequence

Point = tuple[float, float]
Polygon = Sequence[Point]

# Tolerance for "the point lies exactly on this edge". Bounding-box anchors are
# produced by division ((x1 + x2) / 2), so exact equality against an edge is
# fragile; 1e-9 is many orders of magnitude below one pixel, so it can never
# change a real detection's verdict, it only absorbs float representation error.
_ON_EDGE_TOLERANCE = 1e-9


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    """Return whether `point` falls inside `polygon`.

    Args:
        point: (x, y) in frame pixel coordinates.
        polygon: The zone's vertices as (x, y) pairs, in order. The polygon is
            treated as closed — the last vertex joins back to the first, so
            there is no need to repeat the opening vertex at the end.

    Returns:
        True if the point is inside the polygon or lies on its boundary.

    Raises:
        ValueError: If the polygon has fewer than 3 vertices. A malformed zone
            is raised loudly rather than treated as empty, because an empty
            zone would silently never alert — the worst possible failure mode
            for this tool.

    Boundary rule: a point exactly on an edge or vertex counts as INSIDE.
    Ray casting alone answers boundary cases arbitrarily depending on which
    edge the ray happens to strike, so the decision is made explicitly here.
    For a perimeter alarm, something touching the zone edge is worth
    surfacing: a false positive costs a human one glance, a missed intrusion
    defeats the purpose of the tool.
    """
    if len(polygon) < 3:
        raise ValueError(
            f"A zone polygon needs at least 3 points, got {len(polygon)}"
        )

    px, py = point

    # Pass 1: on the boundary? Decided before ray casting so the answer is the
    # documented one rather than an artefact of the crossing count.
    for (ax, ay), (bx, by) in _edges(polygon):
        if _is_on_segment(px, py, ax, ay, bx, by):
            return True

    # Pass 2: ray casting. Cast a ray from the point in the +x direction and
    # count how many edges it crosses. Odd means inside, even means outside.
    inside = False
    for (ax, ay), (bx, by) in _edges(polygon):
        # Half-open comparison: an edge counts only if the point's y is at or
        # above one endpoint and strictly below the other. This is what stops a
        # vertex shared by two edges being counted twice when the ray passes
        # exactly through it — the bug that makes naive implementations report
        # outside points as inside.
        if (ay > py) != (by > py):
            # x coordinate where this edge crosses the horizontal line y = py.
            crossing_x = ax + (py - ay) / (by - ay) * (bx - ax)
            if px < crossing_x:
                inside = not inside

    return inside


def _edges(polygon: Polygon):
    """Yield each (start, end) vertex pair, closing the last back to the first."""
    for i in range(len(polygon)):
        yield polygon[i], polygon[(i + 1) % len(polygon)]


def _is_on_segment(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> bool:
    """Return whether (px, py) lies on the line segment (ax, ay) -> (bx, by)."""
    # Collinear? The cross product of (b - a) and (p - a) is zero when the
    # three points lie on one line.
    cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    if abs(cross) > _ON_EDGE_TOLERANCE:
        return False

    # Collinear with the segment's line, but possibly beyond either end — so
    # confirm it sits within the segment's bounding box.
    return (
        min(ax, bx) - _ON_EDGE_TOLERANCE <= px <= max(ax, bx) + _ON_EDGE_TOLERANCE
        and min(ay, by) - _ON_EDGE_TOLERANCE <= py <= max(ay, by) + _ON_EDGE_TOLERANCE
    )
