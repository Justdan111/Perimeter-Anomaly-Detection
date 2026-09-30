// Run with: npm test
import assert from "node:assert/strict";
import { describe, test } from "node:test";

import { INITIAL_LIVE, RECENT_CAP, boxPercent, liveSocketUrl, reduceLive, summarise, type FrameMessage } from "./live.ts";

const person = { class_name: "person", confidence: 0.9, bbox: [0, 0, 10, 10] as [number, number, number, number], anchor: [5, 10] as [number, number] };

function frame(alerts = [person], at = 1): FrameMessage {
  return {
    type: "frame", camera: "webcam", received_at: at, sent_at: null, frame_width: 640, frame_height: 480,
    detections: alerts.length, alerts, detect_ms: 500, lock_wait_ms: 0, preview_jpeg: "",
  };
}

describe("liveSocketUrl", () => {
  test("https becomes wss", () => assert.equal(liveSocketUrl("https://x.onrender.com"), "wss://x.onrender.com/live/ws"));
  test("http becomes ws and a trailing slash is not doubled", () =>
    assert.equal(liveSocketUrl("http://localhost:8000/"), "ws://localhost:8000/live/ws"));
});

describe("reduceLive", () => {
  test("a frame becomes the latest and, with alerts, the head of recent", () => {
    const s = reduceLive(reduceLive(INITIAL_LIVE, frame([person], 1)), frame([person], 2));
    assert.equal(s.frames, 2);
    assert.equal(s.latest?.received_at, 2);
    assert.deepEqual(s.recent.map((r) => r.at), [2, 1]);
  });
  test("a frame without alerts updates the view but adds nothing to recent", () => {
    const s = reduceLive(INITIAL_LIVE, frame([], 3));
    assert.equal(s.latest?.received_at, 3);
    assert.equal(s.recent.length, 0);
  });
  test("recent is capped", () => {
    let s = INITIAL_LIVE;
    for (let i = 0; i < RECENT_CAP + 5; i++) s = reduceLive(s, frame([person], i));
    assert.equal(s.recent.length, RECENT_CAP);
    assert.equal(s.recent[0].at, RECENT_CAP + 4);
  });
  test("dropped frames are counted, the view is kept", () => {
    const s = reduceLive(reduceLive(INITIAL_LIVE, frame()), { type: "dropped", camera: "webcam", reason: "model busy" });
    assert.equal(s.dropped, 1);
    assert.equal(s.frames, 1);
    assert.notEqual(s.latest, null);
  });
});

test("boxPercent scales pixel boxes to the frame", () => {
  assert.deepEqual(boxPercent([64, 48, 320, 480], 640, 480), { left: 10, top: 10, width: 40, height: 90 });
});

test("summarise counts per class", () => {
  assert.equal(summarise([person, person, { ...person, class_name: "car" }]), "2 person, 1 car");
});
