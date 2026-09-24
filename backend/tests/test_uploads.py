"""Upload validation — the pure parts, tested without a video file.

Two layers:
- `sniff_container` looks at the first bytes of the file. OpenCV alone can't
  be the test for "is this a video": it opens a JPEG as a one-frame "video"
  at a made-up 25 fps, and an animated GIF as a 20-frame one (both measured).
- `check_probe` applies the size/length/resolution limits to what OpenCV
  reported about the clip.
"""

import pytest

from app.services.uploads import (
    ClipProbe,
    UploadLimits,
    UploadRejected,
    check_probe,
    sniff_container,
)

LIMITS = UploadLimits(
    max_bytes=100 * 1024 * 1024,
    max_duration_s=60.0,
    max_long_side=1920,
    max_short_side=1080,
    min_short_side=120,
    max_fps=60.0,
    min_frames=2,
)


def probe(**overrides) -> ClipProbe:
    values = dict(width=1280, height=720, fps=30.0, frame_count=300)
    values.update(overrides)
    return ClipProbe(**values)


class TestSniffContainer:
    @pytest.mark.parametrize(
        "head,expected",
        [
            (b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 8, "mp4"),
            (b"\x00\x00\x00\x14ftypqt  " + b"\x00" * 8, "mp4"),  # QuickTime .mov
            (b"\x1a\x45\xdf\xa3" + b"\x00" * 12, "mkv"),  # Matroska / WebM
            (b"RIFF\x00\x00\x00\x00AVI " + b"\x00" * 4, "avi"),
            (b"\x47" + b"\x00" * 187 + b"\x47" + b"\x00" * 3, "ts"),  # MPEG-TS sync bytes 188 apart
        ],
    )
    def test_recognises_video_containers(self, head, expected):
        assert sniff_container(head) == expected

    @pytest.mark.parametrize(
        "head",
        [
            b"\xff\xd8\xff\xe0" + b"\x00" * 12,  # JPEG — OpenCV would open it
            b"GIF89a" + b"\x00" * 10,  # GIF — OpenCV would open it too
            b"\x89PNG\r\n\x1a\n" + b"\x00" * 8,
            b"%PDF-1.7\n" + b"\x00" * 8,
            b"hello, this is text",
            b"",
            b"RIFF\x00\x00\x00\x00WAVE" + b"\x00" * 4,  # RIFF, but audio not AVI
            b"\x47" + b"\x00" * 50,  # one stray 0x47 is not an MPEG-TS stream
        ],
    )
    def test_rejects_everything_else(self, head):
        assert sniff_container(head) is None

    def test_ftyp_must_be_at_offset_4(self):
        # "ftyp" appearing somewhere else in a text file isn't an MP4 header.
        assert sniff_container(b"just ftyp in some text") is None


class TestCheckProbe:
    def test_typical_phone_clip_passes(self):
        check_probe(probe(), LIMITS)  # no exception

    def test_portrait_1080p_passes(self):
        # Upright phone video is 1080 wide, 1920 tall: limits apply to the
        # long and short side, not to width and height.
        check_probe(probe(width=1080, height=1920), LIMITS)

    @pytest.mark.parametrize("w,h", [(3840, 2160), (2160, 3840), (2560, 1440)])
    def test_above_1080p_is_rejected(self, w, h):
        with pytest.raises(UploadRejected, match="resolution") as e:
            check_probe(probe(width=w, height=h), LIMITS)
        assert e.value.status_code == 422

    def test_tiny_resolution_is_rejected(self):
        with pytest.raises(UploadRejected, match="resolution"):
            check_probe(probe(width=160, height=90), LIMITS)

    def test_exactly_at_the_limits_passes(self):
        check_probe(probe(width=1920, height=1080, fps=60.0, frame_count=3600), LIMITS)

    def test_longer_than_the_limit_is_rejected_with_both_numbers(self):
        with pytest.raises(UploadRejected, match=r"61\.0 s.*60 s") as e:
            check_probe(probe(fps=30.0, frame_count=1830), LIMITS)
        assert e.value.status_code == 422

    def test_frame_rate_above_the_limit_is_rejected(self):
        with pytest.raises(UploadRejected, match="frame rate"):
            check_probe(probe(fps=120.0, frame_count=600), LIMITS)

    @pytest.mark.parametrize("fps", [0.0, -1.0])
    def test_no_usable_frame_rate_is_rejected(self, fps):
        with pytest.raises(UploadRejected, match="frame rate"):
            check_probe(probe(fps=fps), LIMITS)

    @pytest.mark.parametrize("frame_count", [0, -1])
    def test_unknown_length_is_rejected_not_waved_through(self, frame_count):
        # With no frame count there is no way to bound processing time.
        with pytest.raises(UploadRejected, match="length"):
            check_probe(probe(frame_count=frame_count), LIMITS)

    def test_single_frame_is_not_a_video(self):
        with pytest.raises(UploadRejected, match="single image"):
            check_probe(probe(frame_count=1), LIMITS)


class TestClipProbe:
    def test_duration_is_frames_over_fps(self):
        assert probe(fps=25.0, frame_count=250).duration_s == 10.0
