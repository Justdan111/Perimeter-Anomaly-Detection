"""Unit tests for the point-in-polygon zone check.

These tests deliberately need no model weights, no video file, and no I/O —
plain coordinates in, boolean out. That is the whole reason this component is
tested on Day 1 rather than deferred.

Coordinates are frame pixel coordinates: x right, y *down* (OpenCV convention).
Ray casting is indifferent to the y direction, but the fixtures are written the
way a real zone would be drawn over a frame so they read correctly.
"""

import pytest

from app.services.zone_check import point_in_polygon

# A plain axis-aligned square, 0,0 -> 100,100. Easy to reason about by hand.
SQUARE = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]

# A concave "L" shape. The notch matters: a point can sit inside the shape's
# bounding box, and inside its convex hull, while being genuinely outside the
# zone. Real drawn zones (a yard minus a building corner) are rarely convex.
#
#   (0,0) ---------------- (100,0)
#     |                        |
#     |        notch  +--------+ (100,40)
#     |               |
#     |     (40,40) --+
#     |     |
#   (0,100) -- (40,100)
L_SHAPE = [
    (0.0, 0.0),
    (100.0, 0.0),
    (100.0, 40.0),
    (40.0, 40.0),
    (40.0, 100.0),
    (0.0, 100.0),
]

# A rectangle with a triangular bite taken out of its left side, the bite's tip
# reaching (60, 50). That tip is a vertex the boundary passes straight through
# in the y direction rather than turning back at — the exact configuration that
# breaks a ray-casting implementation without a half-open edge rule.
#
#   (20,0) ------------------ (100,0)
#      \                          |
#       \  (60,50)  <- tip        |
#       /                         |
#   (20,100) ---------------- (100,100)
CHEVRON = [(20.0, 0.0), (100.0, 0.0), (100.0, 100.0), (20.0, 100.0), (60.0, 50.0)]

# A realistic zone as it would actually be configured: a quadrilateral covering
# a driveway in the lower-left of a 1280x720 frame, drawn in perspective.
DRIVEWAY_ZONE = [(120.0, 700.0), (540.0, 380.0), (760.0, 400.0), (480.0, 715.0)]


class TestClearlyInsideOrOutside:
    """The two cases that must never be wrong, whatever else is."""

    def test_point_in_the_middle_of_the_square_is_inside(self):
        assert point_in_polygon((50.0, 50.0), SQUARE) is True

    def test_point_well_outside_the_square_is_outside(self):
        assert point_in_polygon((150.0, 50.0), SQUARE) is False

    def test_point_above_the_square_is_outside(self):
        # Outside on the y axis rather than x — a ray cast horizontally must not
        # accidentally report points that are merely x-aligned with the polygon.
        assert point_in_polygon((50.0, -20.0), SQUARE) is False

    def test_point_inside_a_perspective_driveway_zone_is_inside(self):
        # A vehicle sitting on the driveway, using a realistic frame coordinate.
        assert point_in_polygon((450.0, 600.0), DRIVEWAY_ZONE) is True

    def test_point_on_the_pavement_outside_the_driveway_zone_is_outside(self):
        # Same frame, but up and to the left of the zone — a person walking past
        # on the street must not alert.
        assert point_in_polygon((200.0, 300.0), DRIVEWAY_ZONE) is False


class TestBoundary:
    """Boundary behaviour is a decision, not an accident.

    A point exactly on an edge or vertex counts as INSIDE. For a perimeter
    alarm, something touching the zone boundary is worth surfacing: a false
    positive costs a human one glance, a missed intrusion costs the whole point
    of the tool. Naive ray casting answers this arbitrarily depending on which
    edge you happen to hit, so it is pinned down here.
    """

    def test_point_exactly_on_a_horizontal_edge_is_inside(self):
        assert point_in_polygon((50.0, 0.0), SQUARE) is True

    def test_point_exactly_on_a_vertical_edge_is_inside(self):
        assert point_in_polygon((100.0, 50.0), SQUARE) is True

    def test_point_exactly_on_a_vertex_is_inside(self):
        assert point_in_polygon((0.0, 0.0), SQUARE) is True

    def test_point_exactly_on_a_diagonal_edge_is_inside(self):
        # Midpoint of the diagonal edge (0,0)->(100,100) of a triangle. A
        # diagonal is the case an axis-aligned-only boundary check gets wrong.
        triangle = [(0.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
        assert point_in_polygon((50.0, 50.0), triangle) is True

    def test_point_a_hair_inside_the_edge_is_inside(self):
        assert point_in_polygon((99.999, 50.0), SQUARE) is True

    def test_point_a_hair_outside_the_edge_is_outside(self):
        assert point_in_polygon((100.001, 50.0), SQUARE) is False


class TestRayCastingEdgeCases:
    """The cases where a naive implementation silently gives a wrong answer.

    Each case here was checked against a deliberately broken implementation to
    confirm it actually discriminates — a test that passes against the bug it
    is named for is decoration, not coverage.
    """

    def test_point_level_with_a_pass_through_vertex_is_not_double_counted(self):
        # The classic ray-casting bug. A ray cast from (30, 50) passes exactly
        # through the notch tip (60, 50) — a vertex the boundary passes
        # straight through rather than turning back at. An implementation that
        # counts both edges meeting there gets an even crossing count and
        # reports this outside point as inside. Verified: the naive version
        # answers True (wrong) for this exact point.
        assert point_in_polygon((30.0, 50.0), CHEVRON) is False

    def test_point_inside_a_zone_level_with_a_pass_through_vertex_is_inside(self):
        # Same shape, same troublesome y, but genuinely inside the zone.
        assert point_in_polygon((80.0, 50.0), CHEVRON) is True

    def test_point_in_the_notch_of_a_concave_zone_is_outside(self):
        # Inside the zone's bounding box, outside the actual L-shaped zone.
        # Verified: a bounding-box shortcut answers True (wrong) here.
        assert point_in_polygon((70.0, 70.0), L_SHAPE) is False

    def test_point_in_the_bounding_box_but_off_the_driveway_is_outside(self):
        # Same trap on the realistic perspective zone: this point sits within
        # the quad's bounding box but off to its left, on grass. Verified: a
        # bounding-box shortcut answers True (wrong) here too.
        assert point_in_polygon((200.0, 600.0), DRIVEWAY_ZONE) is False

    def test_point_in_the_arm_of_a_concave_zone_is_inside(self):
        assert point_in_polygon((20.0, 70.0), L_SHAPE) is True

    def test_point_in_the_body_of_a_concave_zone_is_inside(self):
        assert point_in_polygon((70.0, 20.0), L_SHAPE) is True


class TestInvalidPolygons:
    """A malformed zone config should fail loudly, not silently never alert."""

    def test_polygon_with_two_points_is_rejected(self):
        with pytest.raises(ValueError, match="at least 3"):
            point_in_polygon((0.0, 0.0), [(0.0, 0.0), (10.0, 10.0)])

    def test_empty_polygon_is_rejected(self):
        with pytest.raises(ValueError, match="at least 3"):
            point_in_polygon((0.0, 0.0), [])
