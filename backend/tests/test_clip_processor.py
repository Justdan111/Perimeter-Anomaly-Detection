"""Unit tests for the pure logic inside `clip_processor.py`.

Same discipline as `test_zone_check.py`: no model weights, no video file, no
I/O. Every function tested here takes plain values or `Detection` fixtures and
returns plain values. The one thing in `clip_processor.py` that is NOT covered
here is the OpenCV read loop itself, which was verified by hand against the
real sample clip (see the Day 2 notes in the module docstring).

Several tests below exist because a specific plausible mistake would otherwise
pass silently. Each of those names the mistake it guards against, and was run
against a deliberately broken implementation to confirm it goes red.
"""

import numpy as np
import pytest

from app.models.schemas import Detection, Zone
from app.services.clip_processor import (
    ALLOWED_CLASSES,
    FrameSizeMismatchError,
    alerts_for_frame,
    anchor_point,
    check_frame_size,
    filter_allowed_classes,
    frame_timestamp,
    should_sample,
)


def det(class_name: str, bbox=(0.0, 0.0, 10.0, 10.0), confidence=0.9) -> Detection:
    return Detection(class_name=class_name, confidence=confidence, bbox=bbox)


# A 1280x720 zone covering a rectangle in the middle of the frame.
ZONE = Zone(
    name="test zone",
    frame_width=1280,
    frame_height=720,
    points=[(400.0, 300.0), (800.0, 300.0), (800.0, 600.0), (400.0, 600.0)],
)


class TestFilterAllowedClasses:
    def test_allowed_set_is_exactly_person_and_the_four_vehicle_classes(self):
        # Pinned explicitly so widening the set (e.g. adding "bicycle") is a
        # deliberate, reviewed change rather than a drive-by edit.
        assert ALLOWED_CLASSES == {"person", "car", "truck", "bus", "motorcycle"}

    def test_keeps_every_allowed_class(self):
        detections = [det(c) for c in ["person", "car", "truck", "bus", "motorcycle"]]
        assert filter_allowed_classes(detections) == detections

    def test_drops_the_classes_the_real_clip_actually_produced(self):
        # These are the non-target classes YOLO26-N really emitted on the
        # sample clip, at confidences above the detector threshold. A handbag
        # in the zone must not become an intrusion alert.
        noise = [det(c) for c in ["traffic light", "handbag", "backpack", "suitcase"]]
        assert filter_allowed_classes(noise) == []

    def test_keeps_order_and_drops_only_the_disallowed_items_from_a_mixed_list(self):
        person, bag, car, light = (
            det("person"), det("handbag"), det("car"), det("traffic light")
        )
        assert filter_allowed_classes([person, bag, car, light]) == [person, car]

    def test_match_is_exact_not_substring_or_case_insensitive(self):
        # Guards against `"car" in class_name`-style matching: "carrot" is a
        # real COCO class and must not pass as a vehicle.
        assert filter_allowed_classes([det("carrot"), det("Person")]) == []

    def test_empty_input_gives_empty_output(self):
        assert filter_allowed_classes([]) == []

    def test_custom_allowed_set_overrides_the_default(self):
        assert filter_allowed_classes(
            [det("person"), det("car")], allowed={"car"}
        ) == [det("car")]


class TestCheckFrameSize:
    def test_matching_size_passes_silently(self):
        check_frame_size(width=1280, height=720, zone=ZONE)  # no exception

    def test_width_mismatch_raises(self):
        with pytest.raises(FrameSizeMismatchError, match="1920x720"):
            check_frame_size(width=1920, height=720, zone=ZONE)

    def test_height_mismatch_raises(self):
        with pytest.raises(FrameSizeMismatchError, match="1280x1080"):
            check_frame_size(width=1280, height=1080, zone=ZONE)

    def test_swapped_dimensions_raise(self):
        # A portrait phone clip of the "same" resolution. Guards against a
        # check that compares pixel count or a sorted pair instead of each
        # axis separately.
        with pytest.raises(FrameSizeMismatchError):
            check_frame_size(width=720, height=1280, zone=ZONE)

    def test_proportional_scale_still_raises(self):
        # 640x360 has the same aspect ratio as 1280x720. The zone's pixel
        # coordinates are still wrong by 2x, so this must not be accepted as
        # "close enough" by an aspect-ratio check.
        with pytest.raises(FrameSizeMismatchError):
            check_frame_size(width=640, height=360, zone=ZONE)

    def test_error_is_a_value_error(self):
        # Callers (the API layer, Day 3) can catch it as a bad-input error.
        assert issubclass(FrameSizeMismatchError, ValueError)


class TestAnchorPoint:
    def test_anchor_is_bottom_centre_of_the_box(self):
        assert anchor_point((100.0, 200.0, 140.0, 380.0)) == (120.0, 380.0)

    def test_anchor_uses_the_bottom_edge_not_the_top(self):
        # Image y grows downward, so the bottom of the box is y2, the larger
        # value. Guards against using y1 (the head) as the anchor.
        _, y = anchor_point((0.0, 100.0, 10.0, 500.0))
        assert y == 500.0

    def test_clipped_box_anchors_at_the_frame_edge(self):
        # KNOWN LIMITATION, pinned rather than hidden: a person walking off
        # the bottom of a 720-high frame gets a box that stops at y=720, so
        # the anchor sits on the frame edge, not at their real (off-screen)
        # feet. This test documents the behaviour; it does not endorse it.
        assert anchor_point((500.0, 300.0, 600.0, 720.0)) == (550.0, 720.0)


class TestAnchorDecidesZoneMembership:
    """The anchor choice changes real verdicts, so the behaviour is pinned."""

    def test_person_standing_in_zone_with_head_outside_it_alerts(self):
        # Box spans y 200..500: the head is above the zone (starts at y=300),
        # the feet are inside. A bottom-centre anchor alerts; a top or centre
        # anchor may not. For "where is this person standing", feet win.
        alerts = alerts_for_frame(
            [det("person", bbox=(580.0, 200.0, 620.0, 500.0))],
            ZONE, frame_index=0, clip_fps=25.0, snapshot="f.jpg",
        )
        assert len(alerts) == 1

    def test_person_whose_body_overlaps_zone_but_feet_are_below_it_does_not_alert(self):
        # Box spans y 450..700: the torso overlaps the zone, the feet are
        # below it at y=700. Guards against using the box centre (y=575,
        # inside) as the anchor.
        alerts = alerts_for_frame(
            [det("person", bbox=(580.0, 450.0, 620.0, 700.0))],
            ZONE, frame_index=0, clip_fps=25.0, snapshot="f.jpg",
        )
        assert alerts == []


class TestFrameTimestamp:
    def test_first_frame_is_time_zero(self):
        assert frame_timestamp(0, 25.0) == 0.0

    def test_timestamp_is_index_over_fps(self):
        assert frame_timestamp(50, 25.0) == pytest.approx(2.0)

    def test_ntsc_rate_is_not_rounded_to_24(self):
        # The sample clip is 23.976 fps (24000/1001). Assuming 24 would put
        # frame 119 at 4.958 s instead of 4.963 s, and the error grows with
        # clip length (about 3.6 s per hour).
        assert frame_timestamp(119, 24000 / 1001) == pytest.approx(4.963, abs=1e-3)

    @pytest.mark.parametrize("bad_fps", [0.0, -25.0])
    def test_non_positive_fps_raises(self, bad_fps):
        # OpenCV reports fps=0 for some broken or unusual files. Dividing by
        # it must be a loud error, not inf/nan timestamps.
        with pytest.raises(ValueError, match="fps"):
            frame_timestamp(10, bad_fps)


class TestShouldSample:
    def test_none_means_every_frame(self):
        assert all(should_sample(i, 30.0, None) for i in range(100))

    def test_first_frame_is_always_sampled(self):
        assert should_sample(0, 30.0, 1.0) is True

    def test_integer_ratio_picks_evenly_spaced_frames(self):
        picked = [i for i in range(30) if should_sample(i, 30.0, 10.0)]
        assert picked == [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]

    def test_one_per_second_on_the_real_clip_rate(self):
        # 23.976 fps: one sample per second of clip time.
        fps = 24000 / 1001
        picked = [i for i in range(120) if should_sample(i, fps, 1.0)]
        assert picked == [0, 24, 48, 72, 96]

    def test_sampling_follows_clip_time_not_a_fixed_frame_stride(self):
        # Over the 120-frame sample clip, "every 24th frame" picks exactly
        # the same frames as "once per second", so the test above cannot
        # tell them apart. They diverge after ~42 s at 23.976 fps: a stride
        # drifts ahead of clip time. Checking 50 s of frames catches it.
        fps = 24000 / 1001
        picked = [i for i in range(1200) if should_sample(i, fps, 1.0)]
        assert len(picked) == 51  # seconds 0..50
        # Each picked frame is the first one at or after a whole second.
        for second, i in enumerate(picked):
            assert frame_timestamp(i, fps) >= second
            if i > 0:
                assert frame_timestamp(i - 1, fps) < second

    def test_sample_count_matches_rate_times_duration(self):
        # 10 s at 30 fps sampled at 2 fps is 20 samples, not 21 or 19.
        picked = [i for i in range(300) if should_sample(i, 30.0, 2.0)]
        assert len(picked) == 20

    def test_rate_at_or_above_clip_fps_samples_every_frame(self):
        # Asking for more samples than there are frames can't invent frames;
        # it degrades to "every frame" rather than raising.
        assert all(should_sample(i, 24.0, 24.0) for i in range(50))
        assert all(should_sample(i, 24.0, 60.0) for i in range(50))

    @pytest.mark.parametrize("bad_rate", [0.0, -1.0])
    def test_non_positive_sample_rate_raises(self, bad_rate):
        with pytest.raises(ValueError, match="sample_fps"):
            should_sample(0, 30.0, bad_rate)


class TestAlertsForFrame:
    def test_detection_inside_zone_produces_one_alert_with_all_fields(self):
        d = det("car", bbox=(500.0, 350.0, 700.0, 450.0), confidence=0.8)
        alerts = alerts_for_frame(
            [d], ZONE, frame_index=48, clip_fps=24.0, snapshot="frame_00048.jpg"
        )
        assert len(alerts) == 1
        a = alerts[0]
        assert a.class_name == "car"
        assert a.timestamp_s == pytest.approx(2.0)
        assert a.frame_index == 48
        assert a.confidence == 0.8
        assert a.bbox == d.bbox
        assert a.anchor == (600.0, 450.0)
        assert a.snapshot == "frame_00048.jpg"
        assert a.zone_name == "test zone"

    def test_detection_outside_zone_produces_no_alert(self):
        alerts = alerts_for_frame(
            [det("person", bbox=(0.0, 0.0, 50.0, 100.0))],
            ZONE, frame_index=0, clip_fps=24.0, snapshot="f.jpg",
        )
        assert alerts == []

    def test_disallowed_class_inside_zone_produces_no_alert(self):
        # The false-positive the Day 1 inference run proved was possible: a
        # handbag sitting in the zone is not an intrusion.
        alerts = alerts_for_frame(
            [det("handbag", bbox=(550.0, 400.0, 600.0, 450.0))],
            ZONE, frame_index=0, clip_fps=24.0, snapshot="f.jpg",
        )
        assert alerts == []

    def test_one_alert_per_in_zone_detection_not_one_per_frame(self):
        # Two people and a car in the zone, one person outside, one handbag
        # inside. Three alerts, in detection order. No de-duplication: each
        # frame's detections are independent (no tracking, by design).
        dets = [
            det("person", bbox=(450.0, 400.0, 480.0, 500.0)),
            det("person", bbox=(10.0, 10.0, 40.0, 100.0)),
            det("handbag", bbox=(600.0, 450.0, 620.0, 480.0)),
            det("car", bbox=(600.0, 400.0, 780.0, 550.0)),
            det("person", bbox=(700.0, 400.0, 730.0, 590.0)),
        ]
        alerts = alerts_for_frame(
            dets, ZONE, frame_index=0, clip_fps=24.0, snapshot="f.jpg"
        )
        assert [a.class_name for a in alerts] == ["person", "car", "person"]
        assert [a.bbox for a in alerts] == [dets[0].bbox, dets[3].bbox, dets[4].bbox]

    def test_no_detections_gives_no_alerts(self):
        assert alerts_for_frame(
            [], ZONE, frame_index=0, clip_fps=24.0, snapshot="f.jpg"
        ) == []


# --- Phase 1: class selection, whole-frame zone, progress -------------------------------

from app.services.clip_processor import (  # noqa: E402
    CLASS_GROUPS,
    count_sampled_frames,
    whole_frame_zone,
)


class TestClassGroups:
    def test_groups_are_exactly_the_existing_allowed_list_split_in_two(self):
        # Class selection exposes the existing filter list; it must not
        # quietly add or drop a class.
        assert CLASS_GROUPS["person"] == {"person"}
        assert CLASS_GROUPS["vehicle"] == {"car", "truck", "bus", "motorcycle"}
        # "both" is now a legacy alias (Phase 1 records), resolved by
        # classes_for; the guarantee is the same.
        assert classes_for(["both"]) == ALLOWED_CLASSES
        assert CLASS_GROUPS["person"] | CLASS_GROUPS["vehicle"] == ALLOWED_CLASSES
        assert not CLASS_GROUPS["person"] & CLASS_GROUPS["vehicle"]

    def test_alerts_for_frame_honours_the_selected_classes(self):
        dets = [
            det("person", bbox=(450.0, 400.0, 480.0, 500.0)),
            det("car", bbox=(600.0, 400.0, 780.0, 550.0)),
            det("bus", bbox=(500.0, 350.0, 700.0, 450.0)),
        ]
        vehicles = alerts_for_frame(
            dets, ZONE, 0, 24.0, "f.jpg", allowed_classes=CLASS_GROUPS["vehicle"]
        )
        people = alerts_for_frame(
            dets, ZONE, 0, 24.0, "f.jpg", allowed_classes=CLASS_GROUPS["person"]
        )
        assert [a.class_name for a in vehicles] == ["car", "bus"]
        assert [a.class_name for a in people] == ["person"]

    def test_default_is_still_both(self):
        # The sample clip path passes nothing and must behave as before.
        dets = [det("person", bbox=(450.0, 400.0, 480.0, 500.0)), det("car", bbox=(600.0, 400.0, 780.0, 550.0))]
        assert len(alerts_for_frame(dets, ZONE, 0, 24.0, "f.jpg")) == 2


class TestWholeFrameZone:
    def test_covers_the_frame_exactly(self):
        z = whole_frame_zone(1920, 1080)
        assert z.points == [(0.0, 0.0), (1920.0, 0.0), (1920.0, 1080.0), (0.0, 1080.0)]
        assert (z.frame_width, z.frame_height) == (1920, 1080)

    @pytest.mark.parametrize(
        "bbox",
        [
            (0.0, 0.0, 10.0, 10.0),  # top-left corner
            (1900.0, 900.0, 1920.0, 1080.0),  # cut off by the bottom-right edge
            (900.0, 500.0, 1000.0, 1080.0),  # cut off by the bottom edge
        ],
    )
    def test_any_detection_anywhere_in_frame_alerts(self, bbox):
        # "Anywhere in frame" includes boxes clipped by the frame edge, whose
        # anchor sits exactly on the bottom boundary.
        z = whole_frame_zone(1920, 1080)
        assert len(alerts_for_frame([det("person", bbox=bbox)], z, 0, 30.0, "f.jpg")) == 1


class TestCountSampledFrames:
    def test_matches_should_sample_exactly(self):
        fps = 30000 / 1001
        for frames, rate in [(120, None), (1800, 2.0), (1799, 2.0), (300, 1.0), (10, 60.0)]:
            expected = sum(should_sample(i, fps, rate) for i in range(frames))
            assert count_sampled_frames(frames, fps, rate) == expected


# --- Phase 2: more classes, colour on every alert ------------------------------------------

from app.services.clip_processor import SELECTABLE_CLASSES, add_colors, classes_for  # noqa: E402


class TestPhase2Classes:
    def test_new_classes_are_selectable(self):
        for name in ("bicycle", "dog", "cat", "backpack", "handbag", "suitcase"):
            assert CLASS_GROUPS[name] == {name}
        assert set(SELECTABLE_CLASSES) == {
            "person", "vehicle", "bicycle", "dog", "cat", "backpack", "handbag", "suitcase",
        }

    def test_default_filter_is_unchanged_so_the_sample_clip_still_ignores_bags(self):
        # Day 2: a handbag in the zone must not raise an intrusion alert on
        # the sample clip. New classes are opt-in at upload, not the default.
        assert ALLOWED_CLASSES == {"person", "car", "truck", "bus", "motorcycle"}
        dets = [det("handbag", bbox=(550.0, 400.0, 600.0, 450.0))]
        assert alerts_for_frame(dets, ZONE, 0, 24.0, "f.jpg") == []

    def test_classes_for_unions_the_selection(self):
        assert classes_for(["person", "dog"]) == {"person", "dog"}
        assert classes_for(["vehicle", "suitcase"]) == {"car", "truck", "bus", "motorcycle", "suitcase"}

    def test_legacy_phase1_choices_still_resolve(self):
        # Phase 1 job records stored "person" | "vehicle" | "both".
        assert classes_for(["both"]) == ALLOWED_CLASSES
        assert classes_for("vehicle") == {"car", "truck", "bus", "motorcycle"}

    @pytest.mark.parametrize("bad", [[], ["giraffe"], ["person", "carrot"]])
    def test_unknown_or_empty_selection_is_an_error(self, bad):
        with pytest.raises(ValueError):
            classes_for(bad)


def frame_with(regions, h=720, w=1280):
    """A grey frame with solid-colour rectangles: {(x1,y1,x2,y2): bgr}."""
    frame = np.full((h, w, 3), 120, dtype=np.uint8)
    for (x1, y1, x2, y2), bgr in regions.items():
        frame[y1:y2, x1:x2] = bgr
    return frame


def alert_for(cls, bbox):
    return alerts_for_frame(
        [det(cls, bbox=bbox)], whole_frame_zone(1280, 720), 0, 24.0, "f.jpg",
        allowed_classes=CLASS_GROUPS["vehicle"] | CLASS_GROUPS["person"] | {"handbag"},
    )[0]


class TestAddColors:
    def test_vehicle_gets_its_body_colour_from_the_raw_frame(self):
        frame = frame_with({(100, 100, 400, 300): (20, 20, 200)})
        [a] = add_colors([alert_for("car", (100.0, 100.0, 400.0, 300.0))], frame)
        assert a.color == "red"
        assert a.upper_color is None and a.lower_color is None

    def test_person_gets_upper_and_lower_not_a_single_colour(self):
        frame = frame_with({
            (600, 100, 680, 280): (200, 90, 20),  # shirt, blue
            (600, 280, 680, 420): (15, 15, 15),  # trousers, black
        })
        [a] = add_colors([alert_for("person", (600.0, 60.0, 680.0, 440.0))], frame)
        assert (a.upper_color, a.lower_color) == ("blue", "black")
        assert a.color is None

    def test_any_other_object_uses_the_same_primitive(self):
        frame = frame_with({(900, 500, 960, 560): (0, 215, 230)})
        [a] = add_colors([alert_for("handbag", (900.0, 500.0, 960.0, 560.0))], frame)
        assert a.color == "yellow"

    def test_each_alert_is_coloured_from_its_own_box(self):
        frame = frame_with({(100, 100, 300, 250): (20, 20, 200), (700, 100, 900, 250): (200, 90, 20)})
        a, b = add_colors(
            [alert_for("car", (100.0, 100.0, 300.0, 250.0)), alert_for("car", (700.0, 100.0, 900.0, 250.0))],
            frame,
        )
        assert (a.color, b.color) == ("red", "blue")

    def test_box_partly_outside_the_frame_is_clipped_not_an_error(self):
        frame = frame_with({(1200, 600, 1280, 720): (20, 20, 200)})
        [a] = add_colors([alert_for("car", (1200.0, 600.0, 1280.0, 720.0))], frame)
        assert a.color == "red"

    def test_alert_without_colour_fields_still_validates(self):
        # Phase 1 result.json records in R2 have no colour fields.
        from app.models.schemas import Alert

        legacy = {k: v for k, v in alert_for("car", (1.0, 1.0, 50.0, 50.0)).model_dump().items()
                  if k not in ("color", "upper_color", "lower_color")}
        a = Alert.model_validate(legacy)
        assert a.color is None and a.upper_color is None and a.lower_color is None


class TestMonochromeFramesInThePipeline:
    def test_alerts_in_a_black_and_white_frame_get_unknown_colours(self):
        frame = np.full((720, 1280, 3), 200, dtype=np.uint8)  # a white "car", no colour anywhere
        frame[300:500, 300:500] = 30  # and a dark "person"
        car, person = add_colors(
            [alert_for("car", (100.0, 100.0, 250.0, 200.0)), alert_for("person", (300.0, 300.0, 500.0, 500.0))],
            frame,
        )
        assert car.color == "unknown"  # not "white": there's no colour to read
        assert (person.upper_color, person.lower_color) == ("unknown", "unknown")

    def test_colour_frames_are_unaffected(self):
        frame = frame_with({(100, 100, 400, 300): (20, 20, 200)})
        [a] = add_colors([alert_for("car", (100.0, 100.0, 400.0, 300.0))], frame)
        assert a.color == "red"
