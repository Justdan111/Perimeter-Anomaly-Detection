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

import boto3
import numpy as np
import pytest
from botocore.config import Config
from fastapi.testclient import TestClient
from moto import mock_aws

from app import main
from app.models.schemas import Detection
from app.services import jobs as jobs_module
from app.services import uploads
from app.services.clip_processor import count_sampled_frames
from app.services.storage import LocalObjectStore, S3ObjectStore

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


BUCKET = "perimeter-test"


@pytest.fixture(params=["local", "r2"])
def store(request, tmp_path):
    """Every test here runs against both stores: a local folder, and R2
    (via moto, an in-memory fake of the S3 API R2 implements)."""
    if request.param == "local":
        yield LocalObjectStore(tmp_path / "output" / "store")
        return
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1", config=Config(signature_version="s3v4"))
        s3.create_bucket(Bucket=BUCKET)
        yield S3ObjectStore(client=s3, bucket=BUCKET)


def make_runner(store, tmp_path):
    return jobs_module.JobRunner(
        model_lock=main._processing_lock, store=store, work_root=tmp_path / "output" / "work"
    )


@pytest.fixture
def client(tmp_path, monkeypatch, detector, store):
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(main, "_results", {})
    monkeypatch.setattr(main, "storage_error", None)
    runner = make_runner(store, tmp_path)
    monkeypatch.setattr(main, "jobs", runner)
    main.app.dependency_overrides[main.get_detector] = lambda: detector
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()
    # Wait for any job still running. Some tests return before their job
    # finishes (e.g. the one that only checks the immediate 202); without
    # this, that job outlived the test — and moto's fake S3 — and made REAL
    # requests to AWS, then held the shared model lock while retrying, which
    # stalled the next test's job. Found as an intermittent segfault in the
    # Docker test run (threads mid-TLS-handshake at interpreter exit).
    runner.shutdown(wait=True)


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


def work_files(tmp_path):
    """Files in the local work folder (uploaded videos while their job runs)."""
    root = tmp_path / "output" / "work"
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()) if root.exists() else []


def stored_keys(store):
    """Every object in the result store, as keys."""
    if isinstance(store, LocalObjectStore):
        root = store.root
        return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()) if root.exists() else []
    listing = store._client.list_objects_v2(Bucket=store.bucket)
    return sorted(o["Key"] for o in listing.get("Contents", []))


def fetch(client, url):
    """GET a snapshot/reference URL: an API route locally, a signed link on R2."""
    if url.startswith("https://"):
        # moto intercepts requests to the (fake) bucket host, so a signed
        # link can be fetched as the browser would.
        import requests

        return requests.get(url, timeout=10)
    return client.get(url)


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

    def test_snapshots_and_reference_frame_are_served(self, client, store):
        job_id = upload(client).json()["job_id"]
        status = wait_for(client, job_id)
        ref = fetch(client, status["reference_frame_url"])
        assert ref.status_code == 200 and ref.headers["content-type"] == "image/jpeg"
        alerts = client.get(f"/jobs/{job_id}/alerts").json()["alerts"]
        snap = fetch(client, alerts[0]["snapshot_url"])
        assert snap.status_code == 200 and snap.headers["content-type"] == "image/jpeg"
        if store.kind == "r2":
            # Signed, expiring links straight to the private bucket.
            assert "X-Amz-Signature=" in alerts[0]["snapshot_url"]
            assert "X-Amz-Signature=" in status["reference_frame_url"]
        else:
            assert alerts[0]["snapshot_url"].startswith(f"/jobs/{job_id}/snapshots/")
        # What's persisted is the object key, not a local path.
        assert alerts[0]["snapshot"].startswith(f"jobs/{job_id}/snapshots/frame_")

    def test_results_go_to_the_store_and_nothing_stays_on_local_disk(self, client, store, tmp_path):
        job_id = upload(client).json()["job_id"]
        wait_for(client, job_id)
        # The uploaded video and the local snapshots are gone...
        assert work_files(tmp_path) == []
        # ...and everything that must survive a restart is in the store.
        keys = stored_keys(store)
        assert f"jobs/{job_id}/job.json" in keys
        assert f"jobs/{job_id}/result.json" in keys
        assert f"jobs/{job_id}/reference.jpg" in keys
        assert sum(k.startswith(f"jobs/{job_id}/snapshots/frame_") for k in keys) == 10
        assert not any(k.endswith("upload") for k in keys)  # the video is never stored

    def test_original_filename_is_kept_for_display_but_never_used_as_a_path(self, client, store):
        body = upload(client, filename="../../etc/passwd.mp4").json()
        wait_for(client, body["job_id"])
        assert body["filename"] == "passwd.mp4"
        assert all(k.startswith(f"jobs/{body['job_id']}/") for k in stored_keys(store))


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

    def test_unknown_job_is_404_and_mentions_expiry(self, client):
        r = client.get("/jobs/" + "0" * 32)
        assert r.status_code == 404
        assert "expired" in r.json()["detail"]


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

        runner = jobs_module.JobRunner(
            model_lock=threading.Lock(),
            store=LocalObjectStore(tmp_path / "store"),
            work_root=tmp_path / "work",
            max_active=1,
        )
        probe, _ = uploads.probe_clip(SAMPLE_CLIP)
        reference = tmp_path / "ref.jpg"
        reference.write_bytes(main.CLIPS["sample"].reference_frame_path.read_bytes())

        def submit(n):
            d = tmp_path / "work" / str(n)
            d.mkdir(parents=True)
            (d / "upload").write_bytes(SAMPLE_CLIP.read_bytes())
            return runner.submit(
                job_id=f"{n:032x}", work_dir=d, reference_frame=reference, filename="c.mp4",
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
    def test_text_file_is_415(self, client, store, tmp_path):
        r = upload(client, data=b"just some text, not a video", filename="notes.mp4")
        assert r.status_code == 415
        assert "not a supported video" in r.json()["detail"]
        assert work_files(tmp_path) == [] and stored_keys(store) == []  # nothing left behind

    def test_jpeg_is_415_even_though_opencv_would_open_it(self, client, store, tmp_path):
        jpeg = main.CLIPS["sample"].reference_frame_path.read_bytes()
        r = upload(client, data=jpeg, filename="photo.mp4")
        assert r.status_code == 415
        assert work_files(tmp_path) == [] and stored_keys(store) == []

    def test_video_header_with_garbage_content_is_422(self, client, store, tmp_path):
        fake_mp4 = b"\x00\x00\x00\x18ftypmp42" + np.random.default_rng(0).bytes(4000)
        r = upload(client, data=fake_mp4)
        assert r.status_code == 422
        assert "could not be opened" in r.json()["detail"] or "could not be decoded" in r.json()["detail"]
        assert work_files(tmp_path) == [] and stored_keys(store) == []

    def test_file_over_the_size_limit_is_413_from_the_real_byte_count(self, client, store, monkeypatch, tmp_path):
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
        assert work_files(tmp_path) == [] and stored_keys(store) == []

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

    def test_clip_longer_than_the_limit_is_422_with_the_numbers(self, client, store, monkeypatch, tmp_path):
        monkeypatch.setattr(main, "UPLOAD_LIMITS", main.UPLOAD_LIMITS.__class__(
            **{**main.UPLOAD_LIMITS.__dict__, "max_duration_s": 2.0}))
        r = upload(client)
        assert r.status_code == 422
        assert "5.0 s" in r.json()["detail"] and "2 s" in r.json()["detail"]
        assert work_files(tmp_path) == [] and stored_keys(store) == []

    def test_resolution_over_the_limit_is_422(self, client, monkeypatch):
        monkeypatch.setattr(main, "UPLOAD_LIMITS", main.UPLOAD_LIMITS.__class__(
            **{**main.UPLOAD_LIMITS.__dict__, "max_long_side": 640, "max_short_side": 360}))
        r = upload(client)
        assert r.status_code == 422
        assert "resolution" in r.json()["detail"]

    @pytest.mark.parametrize("bad", ["giraffe", ["person", "carrot"]])
    def test_invalid_class_choice_is_422(self, client, bad):
        # ("bicycle" was the invalid example in Phase 1; it's selectable now.)
        assert upload(client, classes=bad).status_code == 422

    def test_missing_file_is_422(self, client):
        assert client.post("/uploads", data={"classes": "both"}).status_code == 422

    def test_truncated_clip_is_accepted_then_the_job_fails_clearly(self, client, store, tmp_path):
        # Damage past the first frame can't be seen at upload time without
        # decoding the whole file; the job catches it (Day 4's check).
        data = SAMPLE_CLIP.read_bytes()
        r = upload(client, data=data[: len(data) // 2])
        assert r.status_code == 202
        status = wait_for(client, r.json()["job_id"])
        assert status["status"] == "failed"
        assert "truncated or corrupt" in status["error"]
        assert client.get(f"/jobs/{status['job_id']}/alerts").status_code == 409
        # The failed job's partial snapshots and the upload are gone; its
        # record says why, so the failure survives a restart too.
        assert work_files(tmp_path) == []
        assert not any("snapshots/" in k for k in stored_keys(store))
        assert store.get_json(f"jobs/{status['job_id']}/job.json")["status"] == "failed"

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


def test_startup_clears_the_work_folder_left_from_before_a_restart(tmp_path, monkeypatch):
    # Uploaded videos live in the work folder only while their job runs. After
    # a restart nothing is running, so anything there is left over; left
    # alone, it would pile up on the host's disk.
    class StubDetector:  # starts without real weights
        is_loaded = False
        weights_path = main.Path("yolo26n.pt")

        def load(self):
            self.is_loaded = True

    stale = tmp_path / "output" / "work" / ("a" * 32)
    stale.mkdir(parents=True)
    (stale / "upload").write_bytes(b"an upload whose job died with the old process")
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
    assert body["classes"] == [
        "person", "vehicle", "bicycle", "dog", "cat", "backpack", "handbag", "suitcase",
    ]
    assert body["sample_fps"] == main.UPLOAD_SAMPLE_FPS


# --- persistence: what R2 is for (docs/PHASE1.md) -------------------------------------------


class TestSurvivesRestart:
    def test_finished_job_and_its_results_survive_a_restart(self, client, store, tmp_path, monkeypatch):
        job_id = upload(client).json()["job_id"]
        wait_for(client, job_id)
        before = client.get(f"/jobs/{job_id}/alerts").json()

        # A restart: a fresh process has an empty memory and an empty work
        # folder; only the store carries over.
        monkeypatch.setattr(main, "jobs", make_runner(store, tmp_path))
        status = client.get(f"/jobs/{job_id}").json()
        assert status["status"] == "complete"
        after = client.get(f"/jobs/{job_id}/alerts").json()
        assert [a["snapshot"] for a in after["alerts"]] == [a["snapshot"] for a in before["alerts"]]
        assert fetch(client, after["alerts"][0]["snapshot_url"]).status_code == 200
        assert fetch(client, status["reference_frame_url"]).status_code == 200

    def test_job_cut_off_by_a_restart_is_reported_as_interrupted(self, client, store, tmp_path, monkeypatch):
        # What a restart mid-job leaves behind: a record still saying
        # "processing", and no process that is running it.
        job_id = upload(client).json()["job_id"]
        wait_for(client, job_id)
        record = store.get_json(f"jobs/{job_id}/job.json")
        record["status"] = "processing"
        store.put_json(f"jobs/{job_id}/job.json", record)

        monkeypatch.setattr(main, "jobs", make_runner(store, tmp_path))
        status = client.get(f"/jobs/{job_id}").json()
        assert status["status"] == "failed"
        assert "interrupted by a server restart" in status["error"]
        # ...and that verdict is saved, not recomputed forever.
        assert store.get_json(f"jobs/{job_id}/job.json")["status"] == "failed"

    def test_complete_is_only_reported_once_the_stored_record_says_so(self, client, store):
        # Ordering bug found on the way: the status used to flip to
        # "complete" in memory before job.json was saved.
        job_id = upload(client).json()["job_id"]
        wait_for(client, job_id)
        assert store.get_json(f"jobs/{job_id}/job.json")["status"] == "complete"


class TestStorageFailures:
    def test_failure_saving_results_fails_the_job_clearly_without_partial_snapshots(
        self, client, store, tmp_path
    ):
        original = store.put_file
        lock = threading.Lock()
        calls = {"n": 0}

        def flaky_put_file(key, path, content_type):
            if "/snapshots/" in key:
                with lock:
                    calls["n"] += 1
                    n = calls["n"]
                if n == 1:  # the first snapshot upload fails, immediately
                    raise ConnectionError("simulated R2 outage")
                # ...while the others are in flight. Like a real upload, they
                # have already read the file (boto3 opens it before the
                # network wait), so deleting the work folder doesn't stop
                # them: cleanup must wait for them, or they land after it and
                # leave orphans. (A fake that slept *before* reading hid the
                # bug entirely: the file was gone by the time it woke.)
                held = tmp_path / f"in-flight-{n}.jpg"
                held.write_bytes(path.read_bytes())
                time.sleep(0.5)
                return original(key, held, content_type)
            return original(key, path, content_type)

        store.put_file = flaky_put_file
        job_id = upload(client).json()["job_id"]
        status = wait_for(client, job_id)
        assert status["status"] == "failed"
        assert "could not be saved to storage" in status["error"]
        # Give any upload that outlived the cleanup time to land, then check.
        # (A first version failed the 3rd upload; by the time it was noticed
        # the others had finished, so a planted "don't wait" bug was caught
        # only 1 run in 5.)
        time.sleep(1.0)
        assert not any("/snapshots/" in k for k in stored_keys(store))
        assert work_files(tmp_path) == []

    def test_upload_is_503_when_storage_is_unavailable(self, client, monkeypatch):
        monkeypatch.setattr(main, "storage_error", "StorageError: bucket unreachable")
        r = upload(client)
        assert r.status_code == 503
        assert "storage" in r.json()["detail"]

    def test_upload_is_503_if_the_store_refuses_the_new_job(self, client, store, tmp_path):
        def refuse(*args, **kwargs):
            raise ConnectionError("simulated R2 outage")

        store.put_file = refuse
        r = upload(client)
        assert r.status_code == 503
        assert work_files(tmp_path) == []
        assert main.jobs.active_count() == 0  # not left counting against the queue


def test_snapshots_are_uploaded_in_parallel(client, store, monkeypatch):
    # Measured on Render: one-at-a-time uploads to R2 took ~0.48 s per
    # snapshot — 55 s of a 56 s clip's job was spent saving, not processing.
    # With each upload taking 0.2 s here, 10 snapshots one after another
    # would take 2 s; in parallel they take a fraction of that.
    original = store.put_file

    def slow_put_file(key, path, content_type):
        if "/snapshots/" in key:
            time.sleep(0.2)
        return original(key, path, content_type)

    store.put_file = slow_put_file
    job_id = upload(client).json()["job_id"]
    status = wait_for(client, job_id)
    assert status["status"] == "complete"
    saving = status["finished_at"] - status["started_at"] - status["processing_time_s"]
    assert saving < 1.2, f"saving 10 snapshots took {saving:.2f}s — not parallel"



# --- Phase 2: more classes, colour on every alert ---------------------------------------------


class PersonCarDogDetector(PersonAndCarDetector):
    def detect(self, frame):
        return super().detect(frame) + [
            Detection(class_name="dog", confidence=0.7, bbox=(900.0, 500.0, 1000.0, 600.0)),
        ]


class TestPhase2:
    def test_several_classes_can_be_selected_at_once(self, client, monkeypatch):
        main.app.dependency_overrides[main.get_detector] = lambda: PersonCarDogDetector()
        body = upload(client, classes=["person", "dog"]).json()
        assert body["classes"] == ["person", "dog"]
        wait_for(client, body["job_id"])
        alerts = client.get(f"/jobs/{body['job_id']}/alerts").json()["alerts"]
        assert {a["class_name"] for a in alerts} == {"person", "dog"}  # no car, no handbag

    def test_every_alert_carries_its_colour(self, client):
        job_id = upload(client).json()["job_id"]
        wait_for(client, job_id)
        alerts = client.get(f"/jobs/{job_id}/alerts").json()["alerts"]
        people = [a for a in alerts if a["class_name"] == "person"]
        cars = [a for a in alerts if a["class_name"] == "car"]
        assert people and cars
        assert all(a["upper_color"] and a["lower_color"] and a["color"] is None for a in people)
        assert all(a["color"] and a["upper_color"] is None for a in cars)

    def test_a_phase1_job_record_without_colours_still_loads(self, client, store, tmp_path, monkeypatch):
        # Phase 1 stored classes as one string and alerts with no colour
        # fields. Such records are still in R2 (for 7 days) and must render.
        job_id = upload(client, classes="vehicle").json()["job_id"]
        wait_for(client, job_id)
        record = store.get_json(f"jobs/{job_id}/job.json")
        record["classes"] = "vehicle"
        store.put_json(f"jobs/{job_id}/job.json", record)
        result = store.get_json(f"jobs/{job_id}/result.json")
        for a in result["alerts"]:
            for k in ("color", "upper_color", "lower_color"):
                a.pop(k, None)
        store.put_json(f"jobs/{job_id}/result.json", result)

        monkeypatch.setattr(main, "jobs", make_runner(store, tmp_path))  # fresh process
        status = client.get(f"/jobs/{job_id}").json()
        assert status["status"] == "complete"
        assert status["classes"] == ["vehicle"]
        alerts = client.get(f"/jobs/{job_id}/alerts").json()["alerts"]
        assert alerts and all(a["color"] is None for a in alerts)
