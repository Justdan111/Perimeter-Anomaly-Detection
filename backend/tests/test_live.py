"""Live frame ingestion (Phase 4, Stage 4a): /live/frames, /live/stats, /live/ws.

A fake detector stands in for YOLO26-N, so the expected answers are exact.
The TestClient is used without a `with` block for HTTP (skips the lifespan,
so the real model never loads); WebSocket tests open their own connection.
"""

import threading

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main
from app.models.schemas import Detection
from app.services import live
from app.services.live import FrameRejected, decode_frame, live_alerts, whole_frame

W, H = 640, 480


def jpeg(width=W, height=H) -> bytes:
    frame = np.full((height, width, 3), 90, np.uint8)
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    return buf.tobytes()


class FakeDetector:
    def __init__(self, detections=None, fail=False):
        self.calls = 0
        self.fail = fail
        self.detections = detections if detections is not None else [
            Detection(class_name="person", confidence=0.9, bbox=(100, 100, 200, 400)),
            Detection(class_name="handbag", confidence=0.8, bbox=(120, 200, 160, 300)),
        ]

    def detect(self, frame):
        self.calls += 1
        if self.fail:
            raise RuntimeError("model exploded")
        return self.detections


@pytest.fixture
def fake():
    return FakeDetector()


@pytest.fixture
def client(monkeypatch, fake):
    monkeypatch.setattr(main, "live_stats", live.LiveStats())
    monkeypatch.setattr(main, "live_hub", live.LiveHub())
    main.app.dependency_overrides[main.get_detector] = lambda: fake
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()


def post_frame(client, body=None, **params):
    return client.post(
        "/live/frames",
        content=jpeg() if body is None else body,
        headers={"content-type": "image/jpeg"},
        params=params,
    )


# --- pure parts -----------------------------------------------------------------


def test_live_alerts_keeps_allowed_classes_whose_anchor_is_inside():
    zone = [(0, 0), (300, 0), (300, 300), (0, 300)]
    dets = [
        Detection(class_name="person", confidence=0.9, bbox=(10, 10, 50, 250)),   # anchor (30,250) in
        Detection(class_name="handbag", confidence=0.9, bbox=(10, 10, 50, 250)),  # not a live class
        Detection(class_name="car", confidence=0.9, bbox=(400, 10, 500, 100)),    # anchor (450,100) out
    ]
    out = live_alerts(dets, zone)
    assert [a["class_name"] for a in out] == ["person"]
    assert out[0]["anchor"] == [30.0, 250.0]


def test_live_alerts_tests_the_bottom_centre_not_the_box_centre():
    zone = [(0, 0), (300, 0), (300, 300), (0, 300)]
    # Centre (100, 250) is inside; bottom-centre (100, 350) is below the zone.
    feet_outside = Detection(class_name="person", confidence=0.9, bbox=(50, 150, 150, 350))
    assert live_alerts([feet_outside], zone) == []


def test_whole_frame_zone_includes_boxes_touching_the_bottom_edge():
    det = Detection(class_name="person", confidence=0.9, bbox=(10, 10, 50, H))
    assert len(live_alerts([det], whole_frame(W, H))) == 1


@pytest.mark.parametrize(
    "body, reason",
    [
        (b"", "empty"),
        (b"not an image at all", "not a decodable image"),
        (b"\xff" * (live.MAX_FRAME_BYTES + 1), "limit"),
    ],
)
def test_decode_frame_rejects_unusable_bodies(body, reason):
    with pytest.raises(FrameRejected, match=reason):
        decode_frame(body)


def test_decode_frame_rejects_frames_larger_than_1080p():
    with pytest.raises(FrameRejected, match="longest side"):
        decode_frame(jpeg(2560, 1440))


def test_decode_frame_returns_bgr_frame_of_the_sent_size():
    assert decode_frame(jpeg()).shape == (H, W, 3)


# --- endpoint ---------------------------------------------------------------------


def test_frame_returns_its_alerts_and_timings(client, fake):
    r = post_frame(client)
    assert r.status_code == 200
    body = r.json()
    assert [a["class_name"] for a in body["alerts"]] == ["person"]
    assert body["detections"] == 2
    assert (body["frame_width"], body["frame_height"]) == (W, H)
    assert body["detect_ms"] >= 0 and body["lock_wait_ms"] >= 0
    assert fake.calls == 1


def test_garbage_body_is_422_and_never_reaches_the_model(client, fake):
    r = post_frame(client, body=b"garbage")
    assert r.status_code == 422
    assert fake.calls == 0
    assert client.get("/live/stats").json()["frames_rejected"] == 1


def test_oversized_declared_body_is_413_before_reading(client, fake):
    r = post_frame(client, body=b"\xff" * (live.MAX_FRAME_BYTES + 10))
    assert r.status_code == 413
    assert fake.calls == 0


def test_frame_is_dropped_with_503_while_an_upload_holds_the_model(client, fake, monkeypatch):
    monkeypatch.setattr(main, "LIVE_LOCK_WAIT_S", 0.05)
    main._processing_lock.acquire()
    try:
        r = post_frame(client)
    finally:
        main._processing_lock.release()
    assert r.status_code == 503
    assert r.headers["retry-after"] == "1"
    assert fake.calls == 0
    stats = client.get("/live/stats").json()
    assert stats["frames_dropped_model_busy"] == 1
    assert stats["frames_processed"] == 0


def test_frame_waits_for_a_briefly_held_model_instead_of_dropping(client, fake, monkeypatch):
    monkeypatch.setattr(main, "LIVE_LOCK_WAIT_S", 2.0)
    main._processing_lock.acquire()
    threading.Timer(0.2, main._processing_lock.release).start()
    r = post_frame(client)
    assert r.status_code == 200
    assert r.json()["lock_wait_ms"] >= 150


def test_model_lock_is_released_when_detection_fails(client, monkeypatch):
    main.app.dependency_overrides[main.get_detector] = lambda: FakeDetector(fail=True)
    bad = TestClient(main.app, raise_server_exceptions=False)
    assert post_frame(bad).status_code == 500
    assert not main._processing_lock.locked()


def test_stats_counts_frames_and_alerts(client, fake):
    # Two people in one frame are two alerts, not one alerting frame.
    fake.detections = [
        Detection(class_name="person", confidence=0.9, bbox=(100, 100, 200, 400)),
        Detection(class_name="person", confidence=0.9, bbox=(300, 100, 400, 400)),
    ]
    post_frame(client)
    post_frame(client)
    stats = client.get("/live/stats").json()
    assert stats["frames_received"] == 2
    assert stats["frames_processed"] == 2
    assert stats["alerts"] == 4
    assert stats["model_busy"] is False
    assert "peak_rss_mb" in stats and stats["uptime_s"] >= 0


# --- WebSocket --------------------------------------------------------------------


def test_dashboard_receives_each_processed_frame_over_websocket(client):
    with client.websocket_connect("/live/ws") as ws:
        assert ws.receive_json()["type"] == "hello"
        assert post_frame(client, camera="door-1", sent_at=123.5).status_code == 200
        msg = ws.receive_json()
    assert msg["type"] == "frame"
    assert msg["camera"] == "door-1"
    assert msg["sent_at"] == 123.5
    assert [a["class_name"] for a in msg["alerts"]] == ["person"]
    preview = cv2.imdecode(
        np.frombuffer(__import__("base64").b64decode(msg["preview_jpeg"]), np.uint8), cv2.IMREAD_COLOR
    )
    assert preview.shape[1] == live.PREVIEW_WIDTH


def test_dashboard_is_told_when_a_frame_is_dropped(client, monkeypatch):
    monkeypatch.setattr(main, "LIVE_LOCK_WAIT_S", 0.05)
    with client.websocket_connect("/live/ws") as ws:
        ws.receive_json()
        main._processing_lock.acquire()
        try:
            post_frame(client)
        finally:
            main._processing_lock.release()
        assert ws.receive_json() == {"type": "dropped", "camera": "webcam", "reason": "model busy"}


def test_disconnected_dashboard_is_forgotten(client):
    with client.websocket_connect("/live/ws") as ws:
        ws.receive_json()
        assert client.get("/live/stats").json()["dashboards_connected"] == 1
    # The server notices the close on its next receive.
    for _ in range(50):
        if client.get("/live/stats").json()["dashboards_connected"] == 0:
            break
    assert client.get("/live/stats").json()["dashboards_connected"] == 0
    assert post_frame(client).status_code == 200
