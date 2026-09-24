"""Dominant-colour extraction: pure functions, crop in, colour name out.

Synthetic fixtures here pin the rules (every bucket, noise, borders, mixed
crops, tiny crops). Real crops cut from the sample footage are tested in
TestRealCrops — those are what the rules have to survive.

Crops are BGR, as OpenCV delivers them.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.services.colors import COLOR_NAMES, clothing_colors, dominant_color

FIXTURES = Path(__file__).parent / "fixtures" / "colors"


def solid(bgr, h=80, w=60):
    return np.full((h, w, 3), bgr, dtype=np.uint8)


def noisy(bgr, sigma=12, seed=0, h=80, w=60):
    rng = np.random.default_rng(seed)
    img = solid(bgr, h, w).astype(np.int16) + rng.normal(0, sigma, (h, w, 3)).astype(np.int16)
    return np.clip(img, 0, 255).astype(np.uint8)


class TestBuckets:
    @pytest.mark.parametrize(
        "bgr,expected",
        [
            ((20, 20, 200), "red"),
            ((30, 30, 140), "red"),  # dark red, still clearly red
            ((0, 140, 255), "orange"),
            ((0, 215, 230), "yellow"),
            ((40, 170, 40), "green"),
            ((200, 90, 20), "blue"),
            ((110, 30, 20), "blue"),  # navy
            ((150, 40, 120), "purple"),
            ((180, 105, 255), "pink"),
            ((40, 75, 125), "brown"),
            ((245, 245, 245), "white"),
            ((15, 15, 15), "black"),
            ((128, 128, 128), "gray"),
            ((190, 190, 195), "gray"),  # silver
        ],
    )
    def test_solid_colour_maps_to_its_bucket(self, bgr, expected):
        assert dominant_color(solid(bgr)) == expected

    def test_every_bucket_is_reachable_and_nothing_else_is_returned(self):
        # Guards the list itself: a bucket no rule can produce is dead, and a
        # rule producing a name outside the list would break the UI filter.
        assert set(COLOR_NAMES) == {
            "black", "white", "gray", "red", "orange", "yellow",
            "green", "blue", "purple", "pink", "brown",
        }

    @pytest.mark.parametrize("bgr,expected", [((20, 20, 200), "red"), ((200, 90, 20), "blue"), ((245, 245, 245), "white")])
    def test_sensor_noise_does_not_change_the_answer(self, bgr, expected):
        assert dominant_color(noisy(bgr)) == expected


def shaded_panel(lit_v, shade_from_v, shade_to_v, h=80, w=80):
    """A neutral surface half in sun, half in shade — the pattern measured on
    real white cars in this footage: a sunlit part near full brightness and a
    shaded part grading down through the gray range, split roughly evenly
    between the white and gray buckets."""
    column = np.empty(h)
    column[: h // 2] = lit_v
    column[h // 2 :] = np.linspace(shade_from_v, shade_to_v, h - h // 2)
    column = column.astype(np.uint8)
    return np.repeat(np.repeat(column[:, None], w, axis=1)[:, :, None], 3, axis=2)


class TestWhiteAndGrayOnRealSurfaces:
    """Found on real footage: a white car in partial shade is a gradient, so
    its pixels split between the white and gray buckets (e.g. 38% / 34%) and
    neither wins — it came out "mixed". White and gray are now judged as one
    family, split by the brightest pixels: white paint still reaches near-full
    brightness somewhere, gray paint doesn't."""

    def test_white_car_in_partial_shade_is_white(self):
        # Before the family rule this split ~50/50 and came out "mixed".
        assert dominant_color(shaded_panel(245, 200, 150)) == "white"

    def test_gray_car_with_modest_highlights_stays_gray(self):
        assert dominant_color(shaded_panel(205, 170, 110)) == "gray"

    def test_uniformly_mid_gray_is_gray(self):
        assert dominant_color(solid((140, 140, 140))) == "gray"

    def test_family_rule_does_not_swallow_a_real_colour(self):
        # A red car with bright white reflections is still red.
        crop = solid((20, 20, 200), 100, 100)
        crop[20:40, 20:80] = (250, 250, 250)
        assert dominant_color(crop) == "red"


class TestBlackInSunlight:
    """Found on real footage: black trousers in sun came out "gray" again and
    again. Sunlit black fabric is brighter than the V<55 black cut-off, so it
    fell in the gray bucket. Dark *and* unsaturated (V<110, S<45) now counts
    as black; saturated dark colours (navy, dark red) keep their hue.
    Measured on held-out people: precision 75% -> 90% (README)."""

    @pytest.mark.parametrize("v", [70, 85, 95])  # clearly inside V<110 even with noise
    def test_sunlit_black_fabric_is_black(self, v):
        assert dominant_color(noisy((v, v, v + 4), sigma=6)) == "black"

    def test_mid_gray_is_still_gray(self):
        assert dominant_color(solid((128, 128, 128))) == "gray"

    def test_dark_but_saturated_colours_keep_their_hue(self):
        assert dominant_color(solid((110, 30, 20))) == "blue"  # navy, V=110
        assert dominant_color(solid((30, 30, 100))) == "red"  # dark red, V=100

    def test_black_trousers_in_sun_on_a_person(self):
        assert clothing_colors(person((245, 245, 245), (85, 85, 88))) == ("white", "black")


class TestRegionAndAmbiguity:
    def test_background_around_the_object_is_ignored(self):
        # A detection box includes some background at its edges. A red car
        # with road-gray around it (a 15% border) is red, not gray.
        crop = solid((110, 110, 110), 100, 100)
        crop[15:85, 15:85] = (20, 20, 200)
        assert dominant_color(crop) == "red"

    def test_clear_majority_wins(self):
        crop = solid((20, 20, 200), 100, 100)  # red
        crop[:, 70:] = (200, 90, 20)  # 30% blue
        assert dominant_color(crop) == "red"

    @pytest.mark.parametrize(
        "layout",
        ["halves", "quarters"],
    )
    def test_genuinely_mixed_crop_says_mixed_instead_of_guessing(self, layout):
        crop = np.zeros((100, 100, 3), dtype=np.uint8)
        if layout == "halves":
            crop[:, :50] = (20, 20, 200)
            crop[:, 50:] = (200, 90, 20)
        else:
            crop[:50, :50] = (20, 20, 200)
            crop[:50, 50:] = (200, 90, 20)
            crop[50:, :50] = (40, 170, 40)
            crop[50:, 50:] = (0, 215, 230)
        assert dominant_color(crop) == "mixed"

    @pytest.mark.parametrize("shape", [(0, 0), (3, 3), (40, 0)])
    def test_too_small_to_judge_is_unknown(self, shape):
        crop = np.zeros((*shape, 3), dtype=np.uint8)
        assert dominant_color(crop) == "unknown"

    def test_a_few_pixels_are_too_few_to_judge(self):
        # 5x5 leaves 3x3 = 9 pixels after the edge inset: a colour from nine
        # pixels is noise. (Every tiny case above was already caught by an
        # earlier shape check, so a planted bug removing this minimum went
        # unnoticed until this test.)
        assert dominant_color(solid((20, 20, 200), 5, 5)) == "unknown"

    def test_result_does_not_depend_on_crop_size(self):
        small = solid((200, 90, 20), 30, 20)
        big = cv2.resize(small, (400, 600), interpolation=cv2.INTER_NEAREST)
        assert dominant_color(small) == dominant_color(big) == "blue"


def person(upper_bgr, lower_bgr, h=200, w=80):
    """A crude person crop: skin-tone head, shirt, trousers, road at the feet."""
    crop = np.full((h, w, 3), (120, 120, 120), dtype=np.uint8)  # background
    crop[: int(h * 0.15), 25:55] = (90, 140, 200)  # face / skin
    crop[int(h * 0.15) : int(h * 0.52), 10:70] = upper_bgr
    crop[int(h * 0.52) : int(h * 0.94), 15:65] = lower_bgr
    crop[int(h * 0.94) :, :] = (60, 60, 60)  # road / shoes
    return crop


class TestClothingColors:
    def test_upper_and_lower_are_read_separately(self):
        assert clothing_colors(person((200, 90, 20), (15, 15, 15))) == ("blue", "black")

    def test_order_is_upper_then_lower(self):
        # Guards a swapped split: red shirt, white trousers must not come out
        # as ("white", "red").
        assert clothing_colors(person((20, 20, 200), (245, 245, 245))) == ("red", "white")

    def test_face_does_not_become_the_shirt_colour(self):
        # Skin tone reads as orange/brown; the head region is skipped.
        upper, _ = clothing_colors(person((40, 170, 40), (15, 15, 15)))
        assert upper == "green"

    def test_road_at_the_feet_does_not_become_the_trousers_colour(self):
        _, lower = clothing_colors(person((245, 245, 245), (200, 90, 20)))
        assert lower == "blue"

    def test_same_colour_top_and_bottom(self):
        assert clothing_colors(person((15, 15, 15), (15, 15, 15))) == ("black", "black")

    def test_small_distant_person_is_unknown(self):
        # Found on real footage: people under 150 px tall were 19% of person
        # alerts but 48% of "blue top" results — shaded white shirts on a few
        # pixels read as blue. The evaluation only measured people >= 150 px,
        # so colours are only claimed in that range.
        small = person((200, 90, 20), (15, 15, 15), h=140, w=56)
        assert clothing_colors(small) == ("unknown", "unknown")

    def test_person_at_the_minimum_height_is_read(self):
        assert clothing_colors(person((200, 90, 20), (15, 15, 15), h=150, w=60)) == ("blue", "black")

    def test_tiny_person_crop_is_unknown_not_a_guess(self):
        assert clothing_colors(np.zeros((6, 3, 3), dtype=np.uint8)) == ("unknown", "unknown")


# --- real crops from the footage ---------------------------------------------------------

import json  # noqa: E402

_LABELS = json.loads((FIXTURES / "labels.json").read_text())


def _cases():
    for entry in _LABELS:
        marks = []
        if "known_wrong" in entry:
            marks.append(pytest.mark.xfail(strict=True, reason=entry["known_wrong"]["why"]))
        yield pytest.param(entry, id=entry["file"], marks=marks)


class TestRealCrops:
    """Crops cut from the 56 s source clip (CC0), from the *held-out* sets —
    not the crops the thresholds were tuned on — labelled by eye.

    The property tested is the one a colour filter needs: never name a
    wrong colour. An honest "mixed" is allowed; a confident wrong answer is
    not. Crops that do currently come out wrong are strict xfails with the
    reason, so they show up as a regression the moment they change either
    way. Accuracy over the whole labelled sets is in the README.
    """

    @pytest.mark.parametrize("entry", list(_cases()))
    def test_never_names_a_wrong_colour(self, entry):
        crop = cv2.imread(str(FIXTURES / entry["file"]))
        assert crop is not None, entry["file"]
        acceptable = lambda want: set(want) | {"mixed"}  # noqa: E731
        if entry["kind"] == "vehicle":
            assert dominant_color(crop) in acceptable(entry["color"])
        else:
            upper, lower = clothing_colors(crop)
            if entry["upper"]:
                assert upper in acceptable(entry["upper"]), f"upper: {upper}"
            if entry["lower"]:
                assert lower in acceptable(entry["lower"]), f"lower: {lower}"


# --- black-and-white (infrared) footage -------------------------------------------------

from app.services.colors import is_monochrome  # noqa: E402


class TestMonochromeFrames:
    """Found on real footage: a security camera's night (infrared) mode is
    black and white, so every person came out "gray/gray" — true of the
    pixels, useless for a filter ("gray top" matched everyone). A frame with
    no colour anywhere now gives "unknown" colours instead, which a colour
    filter never matches.

    Real frames: a public-domain Dahua CCTV night sample (Wikimedia Commons)
    and the sample clip's CC0 daylight footage."""

    def test_real_infrared_cctv_frame_is_monochrome(self):
        assert is_monochrome(cv2.imread(str(FIXTURES / "cctv_night_frame.jpg"))) is True

    def test_real_daylight_frame_is_not(self):
        assert is_monochrome(cv2.imread(str(FIXTURES / "daylight_frame.jpg"))) is False

    def test_grey_scene_with_one_coloured_object_is_not_monochrome(self):
        # An overcast concrete scene is mostly gray but still a colour image.
        frame = np.full((360, 640, 3), 128, dtype=np.uint8)
        frame[100:200, 300:420] = (20, 20, 200)
        assert is_monochrome(frame) is False

    def test_mostly_grey_scene_with_one_small_coloured_object_is_not_monochrome(self):
        # A small handbag (0.4% of the frame) in an otherwise grey scene.
        # A first version of the rule used the 99th-percentile saturation,
        # which ignores the top 1% of pixels — so this frame counted as
        # black and white and the handbag lost its colour.
        frame = np.full((720, 1280, 3), 120, dtype=np.uint8)
        frame[500:560, 900:960] = (0, 215, 230)
        assert is_monochrome(frame) is False

    def test_black_and_white_frame_with_jpeg_noise_is_monochrome(self):
        rng = np.random.default_rng(0)
        grey = rng.integers(0, 255, (360, 640), dtype=np.uint8)
        frame = np.dstack([grey, grey, grey])
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        assert is_monochrome(cv2.imdecode(buf, cv2.IMREAD_COLOR)) is True
