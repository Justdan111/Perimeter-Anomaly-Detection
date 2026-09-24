"""Validation on the shared data models.

These validators exist since Day 1 but were never exercised by a test — found
by measuring coverage on Day 4. A zone that encloses no area would silently
never alert, so rejecting it is the point, and needs pinning.
"""

import pytest
from pydantic import ValidationError

from app.models.schemas import Detection, Zone


def zone(points, w=1280, h=720):
    return Zone(name="z", frame_width=w, frame_height=h, points=points)


class TestZone:
    def test_valid_concave_zone_is_accepted(self):
        z = zone([(0, 0), (100, 0), (100, 40), (40, 40), (40, 100), (0, 100)])
        assert len(z.points) == 6

    def test_fewer_than_three_points_is_rejected(self):
        with pytest.raises(ValidationError):
            zone([(0, 0), (10, 10)])

    def test_repeated_vertices_that_leave_fewer_than_three_distinct_are_rejected(self):
        # Four points, but only two distinct: passes a plain length check.
        with pytest.raises(ValidationError, match="3 distinct"):
            zone([(0, 0), (10, 10), (0, 0), (10, 10)])

    def test_collinear_vertices_are_rejected(self):
        # Three distinct points on one line enclose no area.
        with pytest.raises(ValidationError, match="collinear"):
            zone([(0, 0), (50, 50), (100, 100)])

    @pytest.mark.parametrize("w,h", [(0, 720), (1280, 0), (-1, 720)])
    def test_non_positive_frame_size_is_rejected(self, w, h):
        with pytest.raises(ValidationError):
            zone([(0, 0), (10, 0), (0, 10)], w=w, h=h)


class TestDetection:
    def test_valid_detection_is_accepted(self):
        d = Detection(class_name="person", confidence=0.5, bbox=(1, 2, 3, 4))
        assert d.bbox == (1, 2, 3, 4)

    @pytest.mark.parametrize("bbox", [(10, 0, 5, 10), (0, 10, 10, 5)])
    def test_inverted_bbox_is_rejected(self, bbox):
        with pytest.raises(ValidationError, match="x1 <= x2"):
            Detection(class_name="person", confidence=0.5, bbox=bbox)

    @pytest.mark.parametrize("confidence", [-0.1, 1.1])
    def test_confidence_outside_0_1_is_rejected(self, confidence):
        with pytest.raises(ValidationError):
            Detection(class_name="person", confidence=confidence, bbox=(0, 0, 1, 1))
