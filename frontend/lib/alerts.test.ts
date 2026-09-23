// Run with: npm test   (Node's built-in runner; Node strips the TS types.)
import assert from "node:assert/strict";
import { describe, test } from "node:test";

import {
  type Alert,
  countByClass,
  formatCounts,
  groupAlerts,
  touchesBottomEdge,
  zoneBottomMargin,
} from "./alerts.ts";

const FPS = 24000 / 1001; // the sample clip's 23.976 fps

function alert(frame: number, cls = "person", fps = FPS): Alert {
  return {
    timestamp_s: frame / fps,
    frame_index: frame,
    class_name: cls,
    confidence: 0.8,
    bbox: [0, 0, 10, 10],
    anchor: [5, 10],
    zone_name: "z",
    snapshot: `frame_${String(frame).padStart(5, "0")}.jpg`,
    snapshot_url: `/clips/sample/snapshots/frame_${String(frame).padStart(5, "0")}.jpg`,
  };
}

describe("groupAlerts", () => {
  test("no alerts gives no buckets", () => {
    assert.deepEqual(groupAlerts([], 1), []);
  });

  test("buckets are half-open [start, end) in clip time", () => {
    // Frame 23 is 0.959 s, frame 24 is 1.001 s: the boundary sits between
    // them. A bucket-per-24-frames shortcut would agree here but drift later.
    const buckets = groupAlerts([alert(23), alert(24)], 1);
    assert.deepEqual(buckets.map((b) => [b.startS, b.endS, b.alerts.length]), [
      [0, 1, 1],
      [1, 2, 1],
    ]);
  });

  test("a timestamp exactly on a boundary starts the next bucket", () => {
    const buckets = groupAlerts([alert(30, "car", 30)], 1); // exactly 1.0 s
    assert.equal(buckets[0].startS, 1);
  });

  test("float error at a boundary does not push an alert into the previous bucket", () => {
    // 0.7 / 0.1 is 6.999999999999999 in floating point. Without the epsilon,
    // an alert at exactly 0.7 s lands in [0.6, 0.7) instead of [0.7, 0.8).
    const a = { ...alert(0), timestamp_s: 0.7 };
    assert.equal(groupAlerts([a], 0.1)[0].index, 7);
  });

  test("empty stretches of clip time produce no bucket", () => {
    const buckets = groupAlerts([alert(0), alert(80)], 1); // 0 s and 3.34 s
    assert.deepEqual(buckets.map((b) => b.startS), [0, 3]);
  });

  test("every alert lands in exactly one bucket and class totals are preserved", () => {
    // The dashboard's totals must reconcile with the API's: grouping may
    // regroup alerts, never lose or duplicate one.
    const alerts: Alert[] = [];
    for (let f = 0; f < 120; f++) {
      alerts.push(alert(f, "person"));
      if (f % 3 === 0) alerts.push(alert(f, "car"));
    }
    for (const size of [0.5, 1, 2]) {
      const buckets = groupAlerts(alerts, size);
      const flat = buckets.flatMap((b) => b.alerts);
      assert.equal(flat.length, alerts.length);
      assert.equal(new Set(flat).size, alerts.length);
      const summed = new Map<string, number>();
      for (const b of buckets)
        for (const [cls, n] of b.counts) summed.set(cls, (summed.get(cls) ?? 0) + n);
      assert.deepEqual(Object.fromEntries(summed), { person: 120, car: 40 });
    }
  });

  test("alerts from the same frame share one frame group and one snapshot", () => {
    const [bucket] = groupAlerts([alert(5, "person"), alert(5, "car"), alert(6)], 1);
    assert.deepEqual(
      bucket.frames.map((f) => [f.frameIndex, f.alerts.length, f.snapshotUrl.slice(-15)]),
      [
        [5, 2, "frame_00005.jpg"],
        [6, 1, "frame_00006.jpg"],
      ],
    );
  });

  test("out-of-order input is shown in frame order", () => {
    const [bucket] = groupAlerts([alert(9), alert(2), alert(5)], 1);
    assert.deepEqual(bucket.frames.map((f) => f.frameIndex), [2, 5, 9]);
  });

  test("a non-positive bucket size is an error, not an infinite loop or NaN buckets", () => {
    assert.throws(() => groupAlerts([alert(0)], 0));
    assert.throws(() => groupAlerts([alert(0)], -1));
  });
});

describe("countByClass / formatCounts", () => {
  test("most frequent class first, ties alphabetical", () => {
    const counts = countByClass([alert(0, "car"), alert(0), alert(1), alert(1), alert(2, "bus")]);
    assert.deepEqual(counts, [
      ["person", 3],
      ["bus", 1],
      ["car", 1],
    ]);
    assert.equal(formatCounts(counts), "3 person, 1 bus, 1 car");
  });
});

describe("touchesBottomEdge", () => {
  test("a box ending exactly at the frame bottom is cut off", () => {
    assert.equal(touchesBottomEdge([500, 300, 600, 720], 720), true);
  });

  test("boxes ending a few px short of the edge still count (seen on the real clip)", () => {
    // Day 2: people walking out of shot had y2 = 717-718, not 720.
    assert.equal(touchesBottomEdge([150, 400, 190, 717], 720), true);
  });

  test("a box clearly above the edge is not flagged", () => {
    assert.equal(touchesBottomEdge([100, 400, 140, 700], 720), false);
    assert.equal(touchesBottomEdge([100, 400, 140, 712], 720), false);
  });
});

describe("zoneBottomMargin", () => {
  test("the committed zone's lowest point is 92 px above the frame bottom", () => {
    const points: [number, number][] = [
      [0, 428],
      [340, 396],
      [372, 468],
      [252, 556],
      [0, 628],
    ];
    assert.equal(zoneBottomMargin(points, 720), 92);
  });
});
