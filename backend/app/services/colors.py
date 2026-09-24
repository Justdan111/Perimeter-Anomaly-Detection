"""Dominant colour of a detection crop, as a small fixed set of names.

Pure functions: a BGR crop (as OpenCV delivers it) in, a colour name out.
No model — every pixel is put in one bucket by simple HSV thresholds, and the
crop's colour is the bucket most of its pixels fall into.

Why HSV: hue separates "which colour" from "how bright", so a red car is red
in sun and in shade (up to a point — see the README for where this breaks).
Achromatic colours are decided first, by brightness and saturation: very dark
is black whatever its hue, and washed-out pixels are white or gray.

Buckets (fixed, deliberately short): black, white, gray, red, orange, yellow,
green, blue, purple, pink, brown. Silver cars read as "gray". Two further
answers are not colours:
- "mixed":   no bucket clearly dominates (e.g. a two-tone crop). Better to say
             so than to name whichever bucket happened to win by a hair.
- "unknown": the crop is too small to judge.

Where the pixels come from matters as much as the thresholds: a detection box
includes background around its edges, so only the central part is sampled
(`_INSET`), and for people the head and feet are skipped (`clothing_colors`).
"""

from __future__ import annotations

import cv2
import numpy as np

from app.models.schemas import ColorName

COLOR_NAMES: tuple[str, ...] = (
    "black", "white", "gray", "red", "orange", "yellow",
    "green", "blue", "purple", "pink", "brown",
)

# --- thresholds (OpenCV HSV: H 0-179, S and V 0-255) ---------------------------
_BLACK_MAX_V = 55  # darker than this is black, whatever the hue
# Dark *and* unsaturated is black too: sunlit black fabric sits above V=55
# and was read as gray (measured on real footage; see the README).
_DARK_NEUTRAL_MAX_V = 110
_GRAY_MAX_S = 45  # less saturated than this is white/gray
_WHITE_MIN_V = 200  # ...and white only when also this bright
_BROWN_MAX_V = 150  # dark orange reads as brown
# Hue boundaries: red wraps around 0.
_HUE_BANDS = (
    (10, "red"),
    (22, "orange"),
    (35, "yellow"),
    (85, "green"),
    (130, "blue"),
    (150, "purple"),
    (170, "pink"),
    (180, "red"),
)

# --- decision rule ----------------------------------------------------------------
_INSET = 0.20  # sample the central 60% of the box in each direction
_MAX_SIDE = 64  # downsample first: the answer doesn't need every pixel
_MIN_PIXELS = 16
_MIN_SHARE = 0.35  # the winner must hold at least this share of pixels...
_MIN_LEAD = 1.5  # ...and at least this many times the runner-up's
# White and gray are judged together, then split by the brightest pixels
# (90th percentile of V): white paint in shade still reaches near-full
# brightness somewhere; gray paint doesn't. See colors_eval in the README.
_WHITE_FAMILY_P90_V = 230


def _pixel_buckets(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bucket index (into COLOR_NAMES) and brightness (V) for every pixel."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0].astype(np.int16)
    s = hsv[..., 1].astype(np.int16)
    v = hsv[..., 2].astype(np.int16)
    index = {name: i for i, name in enumerate(COLOR_NAMES)}

    buckets = np.empty(h.shape, dtype=np.int8)
    lower = 0
    for upper, name in _HUE_BANDS:
        buckets[(h >= lower) & (h < upper)] = index[name]
        lower = upper
    buckets[(buckets == index["orange"]) & (v < _BROWN_MAX_V)] = index["brown"]

    low_sat = s < _GRAY_MAX_S
    buckets[low_sat & (v >= _WHITE_MIN_V)] = index["white"]
    buckets[low_sat & (v < _WHITE_MIN_V)] = index["gray"]
    buckets[low_sat & (v < _DARK_NEUTRAL_MAX_V)] = index["black"]
    buckets[v < _BLACK_MAX_V] = index["black"]
    return buckets, v


def _color_of_region(region: np.ndarray) -> ColorName:
    if region.ndim != 3 or region.shape[0] < 2 or region.shape[1] < 2:
        return "unknown"
    height, width = region.shape[:2]
    if height * width < _MIN_PIXELS:
        return "unknown"
    scale = _MAX_SIDE / max(height, width)
    if scale < 1:
        region = cv2.resize(
            region,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    buckets, v = _pixel_buckets(region)
    buckets, v = buckets.ravel(), v.ravel()
    counts = np.bincount(buckets, minlength=len(COLOR_NAMES))

    # White + gray as one family (found on real footage: a white car in
    # partial shade split ~40/35 between the two buckets and came out
    # "mixed"). The family's size competes as one; its name comes from how
    # bright its brightest pixels get.
    white, gray = COLOR_NAMES.index("white"), COLOR_NAMES.index("gray")
    family = (buckets == white) | (buckets == gray)
    if family.any():
        family_size = counts[white] + counts[gray]
        counts[white] = counts[gray] = 0
        is_white = np.percentile(v[family], 90) >= _WHITE_FAMILY_P90_V
        counts[white if is_white else gray] = family_size

    order = np.argsort(counts)[::-1]
    total = counts.sum()
    first, second = counts[order[0]], counts[order[1]]
    if first < _MIN_SHARE * total or first < _MIN_LEAD * second:
        return "mixed"
    return COLOR_NAMES[order[0]]  # type: ignore[return-value]


# A frame is black and white if (almost) no pixel has clear colour.
# Measured on a real infrared (night mode) security-camera frame: every
# pixel's saturation was exactly 0 (R = G = B). A share-of-pixels test, not a
# percentile: a small coloured object in an otherwise grey scene (0.4% of the
# frame) must still count as colour — a 99th-percentile rule missed it.
_COLOURED_MIN_S = 40
_MONOCHROME_MAX_COLOURED_SHARE = 0.001


def is_monochrome(frame: np.ndarray) -> bool:
    """Whether a whole frame is black and white (e.g. infrared night mode).

    Colour names from such a frame would all be black/gray/white — true of
    the pixels, but meaningless as the object's colour.
    """
    height, width = frame.shape[:2]
    if height == 0 or width == 0:
        return False
    scale = min(1.0, 320 / width)
    small = cv2.resize(
        frame,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    saturation = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[..., 1]
    return bool(np.mean(saturation >= _COLOURED_MIN_S) < _MONOCHROME_MAX_COLOURED_SHARE)


def _slice(crop: np.ndarray, top: float, bottom: float, left: float, right: float) -> np.ndarray:
    h, w = crop.shape[:2]
    return crop[round(h * top) : round(h * bottom), round(w * left) : round(w * right)]


def dominant_color(crop: np.ndarray) -> ColorName:
    """The dominant colour of any detection crop (vehicle, bag, animal...).

    Only the central part of the box is sampled, to leave out the background
    a detection box includes around its edges.
    """
    if crop.ndim != 3 or crop.shape[0] == 0 or crop.shape[1] == 0:
        return "unknown"
    return _color_of_region(_slice(crop, _INSET, 1 - _INSET, _INSET, 1 - _INSET))


# Clothing colours are only claimed for people at least this tall (pixels in
# the original frame). Measured on real footage: people under 150 px were 19%
# of person alerts but 48% of "blue top" results — a shaded white shirt on a
# few pixels reads as blue. The accuracy evaluation covered people >= 150 px,
# so that is the range colours are claimed for.
_MIN_PERSON_HEIGHT = 150


def clothing_colors(person_crop: np.ndarray) -> tuple[ColorName, ColorName]:
    """(upper, lower) clothing colours of a person crop.

    Upper: 18–50% of the box height (below the head, down to about the
    waist). Lower: 56–90% (legs, stopping short of the feet and the ground).
    Both use the central columns, where the body is, not the background
    beside it. Proportions are for an upright, roughly full-length person;
    a person cut off by the frame edge gets a less reliable split. People
    under `_MIN_PERSON_HEIGHT` pixels tall get ("unknown", "unknown").
    """
    if (
        person_crop.ndim != 3
        or person_crop.shape[0] < _MIN_PERSON_HEIGHT
        or person_crop.shape[1] < 4
    ):
        return "unknown", "unknown"
    upper = _color_of_region(_slice(person_crop, 0.18, 0.50, 0.20, 0.80))
    lower = _color_of_region(_slice(person_crop, 0.56, 0.90, 0.25, 0.75))
    return upper, lower
