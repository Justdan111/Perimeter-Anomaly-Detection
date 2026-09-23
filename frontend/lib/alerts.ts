// Display-layer grouping of alerts. Pure functions, no React, no fetching —
// tested directly with `node --test` (see alerts.test.ts).
//
// This is presentation only. It never merges, drops, or re-identifies an
// alert: every alert the API returns lands in exactly one bucket, and a
// bucket is just "alerts whose timestamp falls in this slice of clip time".
// Nothing here tries to decide that two alerts are the same object —
// cross-frame tracking is out of scope for this project.

export type BBox = [number, number, number, number];

export interface Alert {
  timestamp_s: number;
  frame_index: number;
  class_name: string;
  confidence: number;
  bbox: BBox;
  anchor: [number, number];
  zone_name: string;
  snapshot: string;
  snapshot_url: string;
}

export interface FrameGroup {
  frameIndex: number;
  timestampS: number;
  snapshotUrl: string;
  alerts: Alert[];
}

export interface Bucket {
  index: number;
  startS: number;
  endS: number;
  alerts: Alert[];
  /** Alerts per class, most frequent first (ties alphabetical). */
  counts: [string, number][];
  /** The bucket's alerts grouped by the frame (= snapshot) they came from. */
  frames: FrameGroup[];
}

// Absorbs float error when a timestamp is computed as exactly a bucket
// boundary (e.g. frame 30 of a 30 fps clip = 1.0 s).
const EPSILON = 1e-9;

/**
 * Fixed, clip-time buckets: [0, size), [size, 2*size), ...
 *
 * Boundaries depend only on `bucketSeconds`, never on the data, so the same
 * alert always lands in the same bucket. Empty buckets are omitted.
 */
export function groupAlerts(alerts: Alert[], bucketSeconds: number): Bucket[] {
  if (!(bucketSeconds > 0)) {
    throw new Error(`bucketSeconds must be positive, got ${bucketSeconds}`);
  }
  const byIndex = new Map<number, Alert[]>();
  for (const alert of alerts) {
    const index = Math.floor(alert.timestamp_s / bucketSeconds + EPSILON);
    const list = byIndex.get(index);
    if (list) list.push(alert);
    else byIndex.set(index, [alert]);
  }

  return [...byIndex.entries()]
    .sort(([a], [b]) => a - b)
    .map(([index, bucketAlerts]) => {
      // Stable sort: alerts in the same frame keep the API's order.
      const sorted = [...bucketAlerts].sort((a, b) => a.frame_index - b.frame_index);
      return {
        index,
        startS: index * bucketSeconds,
        endS: (index + 1) * bucketSeconds,
        alerts: sorted,
        counts: countByClass(sorted),
        frames: groupByFrame(sorted),
      };
    });
}

export function countByClass(alerts: Alert[]): [string, number][] {
  const counts = new Map<string, number>();
  for (const a of alerts) counts.set(a.class_name, (counts.get(a.class_name) ?? 0) + 1);
  return [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
}

/** "3 person, 1 car" */
export function formatCounts(counts: [string, number][]): string {
  return counts.map(([name, n]) => `${n} ${name}`).join(", ");
}

function groupByFrame(sortedAlerts: Alert[]): FrameGroup[] {
  const frames: FrameGroup[] = [];
  for (const a of sortedAlerts) {
    const last = frames.at(-1);
    if (last && last.frameIndex === a.frame_index) last.alerts.push(a);
    else
      frames.push({
        frameIndex: a.frame_index,
        timestampS: a.timestamp_s,
        snapshotUrl: a.snapshot_url,
        alerts: [a],
      });
  }
  return frames;
}

// A box whose bottom is this close to the bottom of the frame is treated as
// cut off by it. YOLO clips boxes to the frame, but a person whose feet are
// out of shot doesn't always get y2 of exactly the frame height: on the
// sample clip such boxes end at 717-720 of 720. 1% of the height (7 px at
// 720p) covers that without flagging people who are merely near the edge.
export const EDGE_TOLERANCE_FRACTION = 0.01;

/**
 * Whether a box is cut off by the bottom of the frame. For such a box the
 * anchor (bottom-centre) sits on the frame edge rather than at the object's
 * real, off-screen feet, so its position — and therefore its zone verdict —
 * can be off.
 */
export function touchesBottomEdge(bbox: BBox, frameHeight: number): boolean {
  return bbox[3] >= frameHeight * (1 - EDGE_TOLERANCE_FRACTION);
}

/** How far (px) the zone's lowest point is above the bottom of the frame. */
export function zoneBottomMargin(points: [number, number][], frameHeight: number): number {
  return frameHeight - Math.max(...points.map(([, y]) => y));
}
