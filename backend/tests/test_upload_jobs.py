"""Phase 1: upload a clip, process it as a background job, poll, get results.

Real files go through the real upload endpoint, the real OpenCV probe and
the real job runner thread. Only the detector is faked, so the expected
answers are exact and no model weights are needed.

Uploads are processed at UPLOAD_SAMPLE_FPS (2 fps). The committed sample
clip used as the "user's upload" here is 120 frames at 23.976 fps, so a
job processes exactly 10 frames.
"""

import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main
from app.models.schemas import Detection
from app.services import jobs as jobs_module
from app.services import uploads
from app.services.clip_processor import count_sampled_frames

SAMPLE_CLIP = main.CLIPS["sample"].clip_path
FRAMES_PER_JOB = count_sampled_frames(120, 24000 / 1001, main.UPLOAD_SAMPLE_FPS)


class PersonAndCarDetector:
    """A person in the top-left corner (outside the committed sample zone)
    and a car in the middle, on every frame."""

    def __init__(self):
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return [
            Detection(class_name="person", confidence=0.9, bbox=(10.0, 10.0, 40.0, 100.0)),
            Detection(class_name="car", confidence=0.8, bbox=(500.0, 300.0, 700.0, 400.0)),
            Detection(class_name="handbag", confidence=0.9, bbox=(600.0, 300.0, 620.0, 320.0)),
        ]


@pytest.fixture
def detector():
    return PersonAndCarDetector()


@pytest.fixture
def client(tmp_path, monkeypatch, detector):
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(main, "_results", {})
    runner = jobs_module.JobRunner(model_lock=main._processing_lock)
    monkeypatch.setattr(main, "jobs", runner)
    main.app.dependency_overrides[main.get_detector] = lambda: detector
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()
    runner.shutdown()


def upload(client, path=SAMPLE_CLIP, classes="both", filename="clip.mp4", data=None):
    content = path.read_bytes() if data is None else data
    return client.post(
        "/uploads",
        files={"file": (filename, content, "video/mp4")},
        data={"classes": classes},
    )


def wait_for(client, job_id, states=("complete", "failed"), timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/jobs/{job_id}").json()
        if body["status"] in states:
            return body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} never reached {states}; last: {body}")


def job_files(tmp_path):
    root = tmp_path / "output" / "jobs"
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()) if root.exists() else []


# --- the happy path ----------------------------------------------------------------------


class TestUploadToResults:
    def test_upload_returns_202_and_a_job_id_immediately(self, client):
        r = upload(client)
        assert r.status_code == 202
        body = r.json()
        assert len(body["job_id"]) == 32  # uuid4 hex
        assert body["status"] in ("queued", "processing", "complete")
        assert body["status_url"] == f"/jobs/{body['job_id']}"
        assert body["clip"] == {
            "width": 1280, "height": 720, "fps": pytest.approx(23.976, abs=1e-3),
            "frame_count": 120, "duration_s": pytest.approx(5.005, abs=1e-3),
        }
        assert body["frames_to_process"] == FRAMES_PER_JOB == 10

    def test_job_completes_and_results_use_the_whole_frame_zone(self, client):
        job_id = upload(client).json()["job_id"]
        status = wait_for(client, job_id)
        assert status["status"] == "complete", status
        assert status["frames_processed"] == 10

        r = client.get(f"/jobs/{job_id}/alerts")
        assert r.status_code == 200
        body = r.json()
        # The person is in the top-left corner — outside the committed sample
        # zone, but uploads watch the whole frame, so it alerts.
        classes = [a["class_name"] for a in body["alerts"]]
        assert classes.count("person") == 10 and classes.count("car") == 10
        assert "handbag" not in classes
        assert body["frame_width"] == 1280 and body["frame_height"] == 720
        assert status["zone"]["points"] == [[0, 0], [1280, 0], [1280, 720], [0, 720]]

    @pytest.mark.parametrize(
        "classes,expected", [("person", {"person"}), ("vehicle", {"car"}), ("both", {"person", "car"})]
    )
    def test_class_selection_is_applied(self, client, classes, expected):
        job_id = upload(client, classes=classes).json()["job_id"]
        wait_for(client, job_id)
        alerts = client.get(f"/jobs/{job_id}/alerts").json()["alerts"]
        assert {a["class_name"] for a in alerts} == expected

    def test_snapshots_and_reference_frame_are_served(self, client):
        job_id = upload(client).json()["job_id"]
        status = wait_for(client, job_id)
        ref = client.get(status["reference_frame_url"])
        assert ref.status_code == 200 and ref.headers["content-type"] == "image/jpeg"
        alerts = client.get(f"/jobs/{job_id}/alerts").json()["alerts"]
        snap = client.get(alerts[0]["snapshot_url"])
        assert snap.status_code == 200 and snap.headers["content-type"] == "image/jpeg"
        assert alerts[0]["snapshot_url"].startswith(f"/jobs/{job_id}/snapshots/")

    def test_uploaded_video_is_deleted_after_processing(self, client, tmp_path):
        job_id = upload(client).json()["job_id"]
        wait_for(client, job_id)
        files = job_files(tmp_path)
        assert not any(f.endswith("upload") for f in files), files
        assert f"{job_id}/reference.jpg" in files

    def test_original_filename_is_kept_for_display_but_never_used_as_a_path(self, client, tmp_path):
        body = upload(client, filename="../../etc/passwd.mp4").json()
        wait_for(client, body["job_id"])
        assert body["filename"] == "passwd.mp4"
        assert all(f.startswith(body["job_id"]) for f in job_files(tmp_path))


class TestJobStates:
    def test_results_before_completion_are_409_with_the_status(self, client, detector):
        started, release = threading.Event(), threading.Event()
        original = detector.detect

        def gated(frame):
            started.set()
            assert release.wait(timeout=10)
            return original(frame)

        detector.detect = gated
        try:
            job_id = upload(client).json()["job_id"]
            assert started.wait(timeout=10)
            assert client.get(f"/jobs/{job_id}").json()["status"] == "processing"
            r = client.get(f"/jobs/{job_id}/alerts")
            assert r.status_code == 409
            assert "processing" in r.json()["detail"]
        finally:
            release.set()
        assert wait_for(client, job_id)["status"] == "complete"

    def test_second_job_is_queued_behind_the_first(self, client, detector):
        started, release = threading.Event(), threading.Event()
        original = detector.detect

        def gated(frame):
            started.set()
            assert release.wait(timeout=10)
            return original(frame)

        detector.detect = gated
        try:
            first = upload(client).json()["job_id"]
            assert started.wait(timeout=10)
            second = upload(client).json()
            assert second["status"] == "queued"
            assert second["queue_position"] == 1
        finally:
            release.set()
        assert wait_for(client, first)["status"] == "complete"
        assert wait_for(client, second["job_id"])["status"] == "complete"

    def test_queue_limit_returns_429_instead_of_an_unbounded_backlog(self, client, detector, monkeypatch):
        main.jobs.max_active = 1
        started, release = threading.Event(), threading.Event()
        original = detector.detect

        def gated(frame):
            started.set()
            assert release.wait(timeout=10)
            return original(frame)

        detector.detect = gated
        try:
            first = upload(client).json()["job_id"]
            assert started.wait(timeout=10)
            r = upload(client)
            assert r.status_code == 429
            assert "busy" in r.json()["detail"]
        finally:
            release.set()
        wait_for(client, first)

    def test_unknown_job_says_it_may_be_from_before_a_restart(self, client):
        r = client.get("/jobs/" + "0" * 32)
        assert r.status_code == 404
        assert "restart" in r.json()["detail"]

    def test_old_finished_jobs_are_pruned(self, client, tmp_path):
        main.jobs.keep_finished = 2
        ids = []
        for _ in range(3):
            ids.append(upload(client).json()["job_id"])
            wait_for(client, ids[-1])
        assert client.get(f"/jobs/{ids[0]}").status_code == 404
        assert client.get(f"/jobs/{ids[2]}").status_code == 200
        assert not any(f.startswith(ids[0]) for f in job_files(tmp_path))


class TestJobRunnerQueueLimit:
    def test_runner_enforces_the_limit_itself(self, tmp_path):
        """Not just the endpoint's early check.

        Two uploads arriving together can both pass the endpoint's
        `active_count` check before either is submitted; the runner's own
        check under its lock is what holds the line then. Every API test
        was stopped by the endpoint's check first, so a planted bug that
        removed this one went unnoticed until this test existed.
        """
        started, release = threading.Event(), threading.Event()

        class Gated:
            def detect(self, frame):
                started.set()
                assert release.wait(timeout=10)
                return []

        runner = jobs_module.JobRunner(model_lock=threading.Lock(), max_active=1)
        probe, _ = uploads.probe_clip(SAMPLE_CLIP)

        def submit(n):
            d = tmp_path / str(n)
            d.mkdir()
            video = d / "upload"
            video.write_bytes(SAMPLE_CLIP.read_bytes())
            return runner.submit(
                job_id=f"{n:032x}", directory=d, video_path=video, filename="c.mp4",
                classes="both", probe=probe, zone=main.whole_frame_zone(1280, 720),
                sample_fps=2.0, detector=Gated(),
            )

        try:
            submit(1)
            assert started.wait(timeout=10)
            with pytest.raises(jobs_module.QueueFull):
                submit(2)
        finally:
            release.set()
            runner.shutdown()


# --- validation: each case is a way a real upload goes wrong ------------------------------


class TestUploadValidation:
    def test_text_file_is_415(self, client, tmp_path):
        r = upload(client, data=b"just some text, not a video", filename="notes.mp4")
        assert r.status_code == 415
        assert "not a supported video" in r.json()["detail"]
        assert job_files(tmp_path) == []  # nothing left behind

    def test_jpeg_is_415_even_though_opencv_would_open_it(self, client, tmp_path):
        jpeg = main.CLIPS["sample"].reference_frame_path.read_bytes()
        r = upload(client, data=jpeg, filename="photo.mp4")
        assert r.status_code == 415
        assert job_files(tmp_path) == []

    def test_video_header_with_garbage_content_is_422(self, client, tmp_path):
        fake_mp4 = b"\x00\x00\x00\x18ftypmp42" + np.random.default_rng(0).bytes(4000)
        r = upload(client, data=fake_mp4)
        assert r.status_code == 422
        assert "could not be opened" in r.json()["detail"] or "could not be decoded" in r.json()["detail"]
        assert job_files(tmp_path) == []

    def test_file_over_the_size_limit_is_413_from_the_real_byte_count(self, client, monkeypatch, tmp_path):
        # The limit sits between the file's real size and that size minus the
        # 1 MB multipart allowance: the declared-length check lets it through,
        # so only the endpoint's own byte count can stop it. (A first version
        # of this test used a 1000-byte limit, which the middleware caught
        # first — it never reached the byte count it was named for.)
        size = SAMPLE_CLIP.stat().st_size  # ~1.9 MB
        monkeypatch.setattr(main, "UPLOAD_LIMITS", main.UPLOAD_LIMITS.__class__(
            **{**main.UPLOAD_LIMITS.__dict__, "max_bytes": size - 1000}))
        r = upload(client)
        assert r.status_code == 413
        assert "size limit" in r.json()["detail"]
        assert job_files(tmp_path) == []

    def test_declared_content_length_over_the_limit_is_413_before_reading(self, client, monkeypatch):
        monkeypatch.setattr(main, "UPLOAD_LIMITS", main.UPLOAD_LIMITS.__class__(
            **{**main.UPLOAD_LIMITS.__dict__, "max_bytes": 1000}))
        r = client.post(
            "/uploads",
            content=b"x" * 10,
            headers={"content-type": "multipart/form-data; boundary=x", "content-length": str(50 * 1024 * 1024)},
        )
        assert r.status_code == 413

    def test_upload_without_a_declared_length_is_411(self, client):
        # A generator body makes the client send it chunked, with no
        # Content-Length: there'd be no way to bound its size up front.
        def body():
            yield b"--x\r\n"

        r = client.post(
            "/uploads", content=body(), headers={"content-type": "multipart/form-data; boundary=x"}
        )
        assert r.status_code == 411

    def test_clip_longer_than_the_limit_is_422_with_the_numbers(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr(main, "UPLOAD_LIMITS", main.UPLOAD_LIMITS.__class__(
            **{**main.UPLOAD_LIMITS.__dict__, "max_duration_s": 2.0}))
        r = upload(client)
        assert r.status_code == 422
        assert "5.0 s" in r.json()["detail"] and "2 s" in r.json()["detail"]
        assert job_files(tmp_path) == []

    def test_resolution_over_the_limit_is_422(self, client, monkeypatch):
        monkeypatch.setattr(main, "UPLOAD_LIMITS", main.UPLOAD_LIMITS.__class__(
            **{**main.UPLOAD_LIMITS.__dict__, "max_long_side": 640, "max_short_side": 360}))
        r = upload(client)
        assert r.status_code == 422
        assert "resolution" in r.json()["detail"]

    def test_invalid_class_choice_is_422(self, client):
        assert upload(client, classes="bicycle").status_code == 422

    def test_missing_file_is_422(self, client):
        assert client.post("/uploads", data={"classes": "both"}).status_code == 422

    def test_truncated_clip_is_accepted_then_the_job_fails_clearly(self, client, tmp_path):
        # Damage past the first frame can't be seen at upload time without
        # decoding the whole file; the job catches it (Day 4's check).
        data = SAMPLE_CLIP.read_bytes()
        r = upload(client, data=data[: len(data) // 2])
        assert r.status_code == 202
        status = wait_for(client, r.json()["job_id"])
        assert status["status"] == "failed"
        assert "truncated or corrupt" in status["error"]
        assert client.get(f"/jobs/{status['job_id']}/alerts").status_code == 409
        # The failed job's partial snapshots and the upload are gone.
        assert not any("snapshots/" in f or f.endswith("upload") for f in job_files(tmp_path))

    def test_upload_without_a_loaded_model_is_503(self, client, monkeypatch):
        main.app.dependency_overrides.clear()
        monkeypatch.setattr(main.detector, "_model", None)
        r = upload(client)
        assert r.status_code == 503


# --- the sample clip keeps working alongside -----------------------------------------------


class TestSamplePathAlongsideUploads:
    def test_sample_processing_still_works(self, client):
        r = client.post("/clips/sample/process")
        assert r.status_code == 200
        # Sample clip: every frame, committed zone. The fake's person is
        # outside that zone and the car's anchor (600, 400) is too, so this
        # also proves the sample path did NOT switch to the whole frame.
        assert r.json()["frames_processed"] == 120
        assert r.json()["alert_count"] == 0

    def test_sample_run_is_refused_while_an_upload_job_holds_the_model(self, client, detector):
        started, release = threading.Event(), threading.Event()
        original = detector.detect

        def gated(frame):
            started.set()
            assert release.wait(timeout=10)
            return original(frame)

        detector.detect = gated
        try:
            job_id = upload(client).json()["job_id"]
            assert started.wait(timeout=10)
            worker = threading.Thread(
                target=lambda: statuses.append(client.post("/clips/sample/process").status_code),
                daemon=True,
            )
            statuses: list[int] = []
            worker.start()
            worker.join(timeout=5)
            assert not worker.is_alive(), "sample request hung instead of returning 409"
            assert statuses == [409]
        finally:
            release.set()
        wait_for(client, job_id)


def test_startup_clears_job_files_left_from_before_a_restart(tmp_path, monkeypatch):
    # Jobs are in memory, so after a restart their files belong to jobs no
    # one can look up; left alone, they'd pile up on the host's disk.
    class StubDetector:  # starts without real weights
        is_loaded = False
        weights_path = main.Path("yolo26n.pt")

        def load(self):
            self.is_loaded = True

    stale = tmp_path / "output" / "jobs" / ("a" * 32)
    stale.mkdir(parents=True)
    (stale / "reference.jpg").write_bytes(b"old")
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(main, "detector", StubDetector())
    with TestClient(main.app):
        pass
    assert not stale.exists()


def test_limits_endpoint_reports_the_limits_actually_enforced(client, monkeypatch):
    # The dashboard checks size before uploading using these numbers, so
    # they must come from the same object the server enforces, not a copy.
    monkeypatch.setattr(main, "UPLOAD_LIMITS", main.UPLOAD_LIMITS.__class__(
        **{**main.UPLOAD_LIMITS.__dict__, "max_bytes": 5 * 1024 * 1024, "max_duration_s": 30.0}))
    body = client.get("/uploads/limits").json()
    assert body["max_bytes"] == 5 * 1024 * 1024
    assert body["max_duration_s"] == 30.0
    assert body["max_resolution"] == "1920x1080"
    assert body["classes"] == ["person", "vehicle", "both"]
    assert body["sample_fps"] == main.UPLOAD_SAMPLE_FPS
