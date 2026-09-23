"""API tests: the contract the dashboard depends on.

These run the real sample clip through the real OpenCV loop, but with a fake
detector in place of YOLO26-N, so they need no model weights and the expected
answers are exact. The fake reports, on every frame, one person standing
inside the committed zone and one handbag in the same spot: 120 frames means
exactly 120 alerts, all "person", if the pipeline and API are wired right.

The TestClient is used without a `with` block on purpose — that skips the
app's lifespan hook, so the real model is never loaded.
"""

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main
from app.models.schemas import Detection

# Bottom-centre (200, 500) is inside the committed zone (see zone.json).
IN_ZONE_BBOX = (180.0, 380.0, 220.0, 500.0)


class FakeDetector:
    def __init__(self):
        self.calls = 0

    def detect(self, frame: np.ndarray) -> list[Detection]:
        self.calls += 1
        return [
            Detection(class_name="person", confidence=0.9, bbox=IN_ZONE_BBOX),
            Detection(class_name="handbag", confidence=0.9, bbox=IN_ZONE_BBOX),
        ]


@pytest.fixture
def fake_detector():
    return FakeDetector()


@pytest.fixture
def client(tmp_path, monkeypatch, fake_detector):
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(main, "_results", {})
    main.app.dependency_overrides[main.get_detector] = lambda: fake_detector
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()


def test_clip_info_serves_the_configured_zone_not_a_copy(client):
    body = client.get("/clips/sample").json()
    zone = main.load_zone(main.CLIPS["sample"].zone_path)
    assert body["zone"]["points"] == [list(p) for p in zone.points]
    assert body["zone"]["frame_width"] == 1280
    assert body["zone"]["frame_height"] == 720
    assert body["processed"] is False


def test_reference_frame_is_a_jpeg_matching_the_zone_frame_size(client):
    r = client.get("/clips/sample/reference-frame")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    # The dashboard draws zone coordinates over this image, so its size must
    # be the size the zone was drawn against.
    assert img.shape[:2] == (720, 1280)


def test_unknown_clip_is_404_everywhere(client):
    for method, url in [
        ("get", "/clips/nope"),
        ("get", "/clips/nope/reference-frame"),
        ("post", "/clips/nope/process"),
        ("get", "/clips/nope/alerts"),
        ("get", "/clips/nope/snapshots/frame_00000.jpg"),
    ]:
        assert getattr(client, method)(url).status_code == 404, url


def test_alerts_before_processing_is_404(client):
    assert client.get("/clips/sample/alerts").status_code == 404


def test_process_then_alerts_returns_one_alert_per_in_zone_detection(client, fake_detector):
    r = client.post("/clips/sample/process")
    assert r.status_code == 200
    summary = r.json()
    assert summary["frames_read"] == 120
    assert summary["frames_processed"] == 120
    assert summary["alert_count"] == 120  # the handbag is filtered out
    assert fake_detector.calls == 120

    body = client.get("/clips/sample/alerts").json()
    assert body["frame_width"] == 1280 and body["frame_height"] == 720
    assert body["zone_name"] == "Roadway (west approach)"
    assert len(body["alerts"]) == 120
    assert {a["class_name"] for a in body["alerts"]} == {"person"}
    assert client.get("/clips/sample").json()["processed"] is True

    first, last = body["alerts"][0], body["alerts"][-1]
    assert first["timestamp_s"] == 0.0
    assert last["timestamp_s"] == pytest.approx(119 / (24000 / 1001), abs=1e-3)
    assert first["snapshot_url"] == "/clips/sample/snapshots/frame_00000.jpg"


def test_every_snapshot_url_serves_a_downscaled_jpeg(client):
    client.post("/clips/sample/process")
    alerts = client.get("/clips/sample/alerts").json()["alerts"]
    for url in {a["snapshot_url"] for a in alerts}:
        r = client.get(url)
        assert r.status_code == 200, url
        assert r.headers["content-type"] == "image/jpeg"
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[:2] == (360, 640)


def test_sample_fps_is_passed_through(client):
    summary = client.post("/clips/sample/process", params={"sample_fps": 2}).json()
    assert summary["frames_processed"] == 10
    assert client.get("/clips/sample/alerts").json()["sample_fps"] == 2


@pytest.mark.parametrize("bad", [0, -1, "fast"])
def test_invalid_sample_fps_is_rejected(client, bad):
    assert client.post("/clips/sample/process", params={"sample_fps": bad}).status_code == 422


def test_reprocessing_removes_the_previous_runs_snapshots(client):
    client.post("/clips/sample/process")
    assert client.get("/clips/sample/snapshots/frame_00001.jpg").status_code == 200
    # At 2 fps, frame 1 is not sampled, so its old snapshot must be gone —
    # not left behind for a stale alert list to point at.
    client.post("/clips/sample/process", params={"sample_fps": 2})
    assert client.get("/clips/sample/snapshots/frame_00001.jpg").status_code == 404


@pytest.mark.parametrize("name", ["frame_1.jpg", "frame_00001.png", "zone.json", "frame_00001.jpg.bak"])
def test_snapshot_route_rejects_names_clip_processor_never_generates(client, name):
    # 422 specifically: rejected by the filename pattern before any
    # filesystem lookup, not merely "not found". (A "/" can't reach the
    # handler at all — path params stop at slashes — so the pattern is what
    # limits requests to files clip_processor wrote.)
    client.post("/clips/sample/process")
    assert client.get(f"/clips/sample/snapshots/{name}").status_code == 422


def test_leftover_snapshot_files_are_not_served_without_a_result(client, tmp_path):
    # After a restart the in-memory result is gone but files from the last
    # run may still be on disk. They must not be served as if current.
    snapshots = tmp_path / "sample" / "snapshots"
    snapshots.mkdir(parents=True)
    (snapshots / "frame_00000.jpg").write_bytes(b"stale")
    assert client.get("/clips/sample/snapshots/frame_00000.jpg").status_code == 404


def test_concurrent_process_request_is_409(client):
    assert main._processing_lock.acquire(blocking=False)
    try:
        assert client.post("/clips/sample/process").status_code == 409
    finally:
        main._processing_lock.release()


def test_cors_allows_the_dashboard_origin(client):
    r = client.get("/clips/sample", headers={"Origin": "http://localhost:3000"})
    assert r.headers["access-control-allow-origin"] == "http://localhost:3000"
