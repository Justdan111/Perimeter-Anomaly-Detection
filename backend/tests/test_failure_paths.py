"""Failure paths: what happens when the input or the model is bad.

Each test here was written before the fix it covers and watched failing
against the code as it stood (plant-the-bug-first, as every day so far):

- a truncated clip used to "succeed" on the frames it could decode (49 of
  120) with no sign anything was missing;
- an unreadable clip surfaced from the API as a bare 500;
- missing or corrupt weights stopped the service from starting at all, so
  /health could never report the problem.

No real model weights are needed: the detector is either a fake, or a real
`Detector` pointed at a weights file that is missing or corrupt on purpose.
"""

import os
import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main
from app.models.schemas import Detection
from app.services.clip_processor import ClipReadError, load_zone, process_clip
from app.services.detector import Detector

SAMPLE = main.CLIPS["sample"]
ZONE = load_zone(SAMPLE.zone_path)

# Bottom-centre (200, 500) is inside the committed zone.
IN_ZONE = Detection(class_name="person", confidence=0.9, bbox=(180.0, 380.0, 220.0, 500.0))


class InZoneDetector:
    """One person inside the zone on every frame: every frame writes a snapshot."""

    def detect(self, frame):
        return [IN_ZONE]


class NothingRelevantDetector:
    """A frame with things in it, none of them a person or vehicle."""

    def detect(self, frame):
        return [
            Detection(class_name="traffic light", confidence=0.9, bbox=IN_ZONE.bbox),
            Detection(class_name="handbag", confidence=0.9, bbox=IN_ZONE.bbox),
        ]


class EmptyDetector:
    def detect(self, frame):
        return []


# --- clip fixtures ---------------------------------------------------------------


@pytest.fixture
def truncated_clip(tmp_path):
    """The sample clip cut in half: opens, header says 120 frames, 49 decode."""
    data = SAMPLE.clip_path.read_bytes()
    path = tmp_path / "truncated.mp4"
    path.write_bytes(data[: len(data) // 2])
    return path


@pytest.fixture
def garbage_clip(tmp_path):
    path = tmp_path / "garbage.mp4"
    path.write_bytes(np.random.default_rng(0).bytes(5000))
    return path


@pytest.fixture
def empty_clip(tmp_path):
    path = tmp_path / "empty.mp4"
    path.write_bytes(b"")
    return path


def snapshots_in(directory):
    return sorted(p.name for p in directory.glob("*.jpg")) if directory.exists() else []


# --- clip_processor level ------------------------------------------------------------


class TestUnreadableClips:
    def test_truncated_clip_fails_instead_of_returning_a_partial_result(self, truncated_clip, tmp_path):
        with pytest.raises(ClipReadError, match=r"decoded 49 of 120 frames"):
            process_clip(truncated_clip, ZONE, InZoneDetector(), tmp_path / "snaps")

    def test_truncated_clip_leaves_no_snapshot_files_behind(self, truncated_clip, tmp_path):
        # The 49 frames that did decode each wrote a snapshot before the
        # truncation was discovered. None of them may survive the failure.
        snaps = tmp_path / "snaps"
        with pytest.raises(ClipReadError):
            process_clip(truncated_clip, ZONE, InZoneDetector(), snaps)
        assert snapshots_in(snaps) == []

    def test_failure_does_not_delete_unrelated_files_in_the_snapshot_dir(self, truncated_clip, tmp_path):
        # Cleanup removes what this run wrote, not whatever else is there.
        snaps = tmp_path / "snaps"
        snaps.mkdir()
        (snaps / "keep.txt").write_text("not ours")
        with pytest.raises(ClipReadError):
            process_clip(truncated_clip, ZONE, InZoneDetector(), snaps)
        assert (snaps / "keep.txt").exists()

    @pytest.mark.parametrize("clip", ["garbage_clip", "empty_clip"])
    def test_file_that_is_not_a_video_fails_clearly(self, clip, request, tmp_path):
        path = request.getfixturevalue(clip)
        with pytest.raises(ClipReadError, match="could not open"):
            process_clip(path, ZONE, InZoneDetector(), tmp_path / "snaps")
        assert snapshots_in(tmp_path / "snaps") == []

    def test_clip_read_error_is_a_value_error(self):
        assert issubclass(ClipReadError, ValueError)


class TestZeroDetectionClip:
    @pytest.mark.parametrize("detector", [EmptyDetector(), NothingRelevantDetector()])
    def test_nothing_relevant_in_any_frame_is_an_empty_result_not_an_error(self, detector, tmp_path):
        result = process_clip(SAMPLE.clip_path, ZONE, detector, tmp_path / "snaps")
        assert result.alerts == []
        assert result.frames_read == 120
        assert result.frames_processed == 120
        # No alert means no snapshot: nothing to review, nothing written.
        assert snapshots_in(tmp_path / "snaps") == []


# --- API level ---------------------------------------------------------------------------


@pytest.fixture
def api(tmp_path, monkeypatch):
    """A client whose detector each test chooses; lifespan not run."""
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(main, "_results", {})

    def use(detector):
        main.app.dependency_overrides[main.get_detector] = lambda: detector
        return TestClient(main.app)

    yield use
    main.app.dependency_overrides.clear()


def register_clip(monkeypatch, clip_id, clip_path):
    monkeypatch.setitem(
        main.CLIPS,
        clip_id,
        main.ClipSource(
            clip_path=clip_path,
            zone_path=SAMPLE.zone_path,
            reference_frame_path=SAMPLE.reference_frame_path,
        ),
    )


class TestZeroDetectionApi:
    def test_process_and_alerts_return_200_with_an_empty_list(self, api):
        client = api(NothingRelevantDetector())
        r = client.post("/clips/sample/process")
        assert r.status_code == 200
        assert r.json()["alert_count"] == 0

        r = client.get("/clips/sample/alerts")
        assert r.status_code == 200
        assert r.json()["alerts"] == []
        # "Processed, nothing found" must be distinguishable from "never
        # processed" (which is a 404).
        assert client.get("/clips/sample").json()["processed"] is True


class TestUnreadableClipApi:
    @pytest.mark.parametrize("clip", ["truncated_clip", "garbage_clip"])
    def test_unreadable_clip_is_a_clear_error_not_a_bare_500(self, clip, request, api, monkeypatch, tmp_path):
        register_clip(monkeypatch, "broken", request.getfixturevalue(clip))
        client = api(InZoneDetector())
        r = client.post("/clips/broken/process")
        assert r.status_code == 500
        detail = r.json()["detail"]
        assert detail.startswith("clip could not be read:")
        # No partial result is published and no snapshot files are left.
        assert client.get("/clips/broken/alerts").status_code == 404
        assert snapshots_in(tmp_path / "output" / "broken" / "snapshots") == []

    def test_a_broken_clip_does_not_take_the_service_down(self, api, monkeypatch, garbage_clip):
        register_clip(monkeypatch, "broken", garbage_clip)
        client = api(InZoneDetector())
        client.post("/clips/broken/process")
        # The lock was released and the good clip still processes.
        r = client.post("/clips/sample/process")
        assert r.status_code == 200
        assert r.json()["alert_count"] == 120


class TestConcurrentProcessing:
    def test_two_overlapping_requests_exactly_one_succeeds(self, tmp_path, monkeypatch):
        """Deliberately overlap two real requests, not a hand-held lock.

        The detector blocks on its first frame until released, so whichever
        request wins the lock is guaranteed to still be processing when the
        other one arrives. No sleeps, no timing luck.
        """
        monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path)
        monkeypatch.setattr(main, "_results", {})
        started, release = threading.Event(), threading.Event()

        class GatedDetector:
            def detect(self, frame):
                started.set()
                assert release.wait(timeout=5), "test never released the detector"
                return [IN_ZONE]

        main.app.dependency_overrides[main.get_detector] = lambda: GatedDetector()
        statuses: list[int] = []
        try:
            barrier = threading.Barrier(2)

            def post():
                client = TestClient(main.app)
                barrier.wait()
                statuses.append(client.post("/clips/sample/process").status_code)
                # The loser returns immediately; once it has, let the winner finish.
                if statuses[-1] == 409:
                    release.set()

            # Daemon threads: if a request hangs, the test fails at join()
            # instead of keeping the test process alive.
            threads = [threading.Thread(target=post, daemon=True) for _ in range(2)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=15)
            assert not any(t.is_alive() for t in threads), "a request hung"
        finally:
            release.set()
            main.app.dependency_overrides.clear()

        assert started.is_set()
        assert sorted(statuses) == [200, 409]
        # The winner's result is intact.
        assert len(main._results["sample"].alerts) == 120


# --- model load failure --------------------------------------------------------------


class StubDetector:
    """Stands in for `Detector` at startup without loading real weights."""

    def __init__(self, fail_with: Exception | None = None):
        self.fail_with = fail_with
        self.is_loaded = False
        self.weights_path = main.Path("yolo26n.pt")

    def load(self):
        if self.fail_with:
            raise self.fail_with
        self.is_loaded = True


@pytest.fixture
def boot(monkeypatch, tmp_path):
    """Start the app (lifespan included) with a given detector."""
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(main, "_results", {})
    monkeypatch.setattr(main, "model_load_error", None)

    def start(detector):
        monkeypatch.setattr(main, "detector", detector)
        return TestClient(main.app)  # used as a context manager by the test

    return start


class TestHealth:
    def test_health_reports_a_loaded_model(self, boot):
        with boot(StubDetector()) as client:
            body = client.get("/health").json()
        assert body == {
            "status": "ok",
            "model_loaded": True,
            "model_weights": "yolo26n.pt",
            "model_error": None,
        }

    def test_health_reports_a_model_that_failed_to_load(self, boot):
        # Guards against /health reporting True regardless of real state.
        with boot(StubDetector(fail_with=RuntimeError("boom"))) as client:
            body = client.get("/health").json()
        assert body["status"] == "degraded"
        assert body["model_loaded"] is False
        assert body["model_error"] == "RuntimeError: boom"


class TestModelLoadFailure:
    """Real `Detector`, real ultralytics, deliberately bad weights files."""

    @pytest.fixture(params=["missing", "corrupt"])
    def bad_weights(self, request, tmp_path):
        path = tmp_path / "yolo26n.pt"
        if request.param == "corrupt":
            path.write_bytes(b"not a model")
        return request.param, path

    def test_service_starts_and_health_reports_the_real_failure(self, boot, bad_weights):
        kind, path = bad_weights
        with boot(Detector(weights_path=path)) as client:  # must not raise
            r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["model_loaded"] is False
        assert body["status"] == "degraded"
        expected = "FileNotFoundError" if kind == "missing" else "TypeError"
        assert body["model_error"].startswith(expected)

    def test_processing_without_a_model_is_a_clear_503_not_a_500_or_a_hang(self, boot, bad_weights):
        _, path = bad_weights
        with boot(Detector(weights_path=path)) as client:
            r = client.post("/clips/sample/process")
        assert r.status_code == 503
        detail = r.json()["detail"]
        assert detail.startswith("model not loaded")
        assert "yolo26n.pt" in detail  # names the file to fix


# --- gaps found by measuring coverage, then confirmed with planted bugs -------------------


class TestProcessorGuards:
    def test_missing_clip_file(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="clip not found"):
            process_clip(tmp_path / "nope.mp4", ZONE, InZoneDetector(), tmp_path / "snaps")

    @pytest.mark.parametrize("bad", [0, -2.0])
    def test_non_positive_sample_fps_is_rejected_before_opening_the_clip(self, bad, tmp_path):
        with pytest.raises(ValueError, match="sample_fps"):
            process_clip(SAMPLE.clip_path, ZONE, InZoneDetector(), tmp_path / "s", sample_fps=bad)

    def test_clip_reporting_no_fps_fails_clearly(self, monkeypatch, tmp_path):
        # No real file reached this path: even a lone JPEG opens as a
        # one-frame "video" at 25 fps. So the capture is faked here.
        class NoFpsCapture:
            def __init__(self, path):
                pass

            def isOpened(self):
                return True

            def get(self, prop):
                return 0.0

            def release(self):
                pass

        monkeypatch.setattr("app.services.clip_processor.cv2.VideoCapture", NoFpsCapture)
        with pytest.raises(ClipReadError, match="no usable fps"):
            process_clip(SAMPLE.clip_path, ZONE, InZoneDetector(), tmp_path / "snaps")


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root ignores directory permissions, so they can't simulate a write failure",
)
class TestSnapshotWriteFailure:
    """The disk refusing writes partway through a run (full, or permissions).

    Simulated by making the snapshot directory read-only after a few frames.
    """

    def test_write_failure_reports_the_write_error_not_a_cleanup_error(self, tmp_path):
        # Cleanup can't delete the earlier snapshots from a read-only
        # directory either. That second failure must not replace the real
        # one in the error the caller sees.
        snaps = tmp_path / "snaps"

        class LockDirAfterFourFrames(InZoneDetector):
            calls = 0

            def detect(self, frame):
                self.calls += 1
                if self.calls == 5:
                    snaps.chmod(0o555)
                return super().detect(frame)

        try:
            with pytest.raises(OSError, match="failed to write snapshot"):
                process_clip(SAMPLE.clip_path, ZONE, LockDirAfterFourFrames(), snaps)
        finally:
            snaps.chmod(0o755)

    def test_api_reports_an_unwritable_output_dir_clearly(self, api, monkeypatch, tmp_path):
        output = tmp_path / "ro-output"
        output.mkdir()
        output.chmod(0o555)
        monkeypatch.setattr(main, "OUTPUT_DIR", output)
        try:
            r = api(InZoneDetector()).post("/clips/sample/process")
        finally:
            output.chmod(0o755)
        assert r.status_code == 500
        assert r.json()["detail"].startswith("could not write snapshots:")


class TestApiErrorMapping:
    def test_zone_drawn_for_another_resolution_is_a_clear_500(self, api, monkeypatch, tmp_path):
        zone_path = tmp_path / "zone.json"
        zone_path.write_text(ZONE.model_copy(update={"frame_width": 1920, "frame_height": 1080}).model_dump_json())
        monkeypatch.setitem(
            main.CLIPS,
            "hd-zone",
            main.ClipSource(
                clip_path=SAMPLE.clip_path,
                zone_path=zone_path,
                reference_frame_path=SAMPLE.reference_frame_path,
            ),
        )
        r = api(InZoneDetector()).post("/clips/hd-zone/process")
        assert r.status_code == 500
        assert "1280x720" in r.json()["detail"] and "1920x1080" in r.json()["detail"]


class TestLoadedModelPath:
    def test_a_loaded_model_is_used_for_processing_not_refused(self, boot):
        # Every other API test swaps the detector dependency out, so none of
        # them go through the real "model loaded? use it" path. Without this,
        # a guard that refused even a loaded model went unnoticed.
        class LoadedStub(StubDetector):
            def detect(self, frame):
                return [IN_ZONE]

        with boot(LoadedStub()) as client:
            r = client.post("/clips/sample/process")
        assert r.status_code == 200
        assert r.json()["alert_count"] == 120


def test_missing_weights_message_gives_a_fetch_command_for_the_real_path(tmp_path):
    # The Day 1 message suggested YOLO('yolo26n.pt'), which downloads into
    # the current directory — not where the service looks — so following the
    # error's own advice didn't fix the error. The command must name the
    # actual weights path (verified: ultralytics downloads to that path).
    path = tmp_path / "models" / "yolo26n.pt"
    with pytest.raises(FileNotFoundError) as excinfo:
        Detector(weights_path=path).load()
    assert f"YOLO('{path}')" in str(excinfo.value)
