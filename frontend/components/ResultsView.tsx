"use client";

import { useMemo, useState } from "react";

import { countByClass, formatCounts, groupAlerts } from "@/lib/alerts";
import type { AlertsResponse } from "@/lib/api";

import { AlertBuckets } from "./AlertBuckets";

// Display grouping only (see lib/alerts.ts). 1 s is the default: ~24 frames
// at the sample clip's 23.976 fps, short enough that everything in a bucket
// is "the same moment" to a reviewer, long enough to turn ~200 alerts into a
// handful of rows.
const BUCKET_OPTIONS = [0.5, 1, 2];
const DEFAULT_BUCKET_S = 1;

/** Totals, grouping control and the grouped alert list — shared by the
 *  sample clip and uploaded clips. */
export function ResultsView({
  result,
  zonePoints,
}: {
  result: AlertsResponse;
  zonePoints: [number, number][];
}) {
  const [bucketS, setBucketS] = useState(DEFAULT_BUCKET_S);
  const buckets = useMemo(() => groupAlerts(result.alerts, bucketS), [result, bucketS]);
  const totals = useMemo(() => countByClass(result.alerts), [result]);

  return (
    <>
      <div className="flex flex-wrap items-end justify-between gap-3 rounded-lg border border-line bg-panel px-4 py-3">
        <div>
          <div className="text-lg font-semibold" data-testid="totals">
            {result.alerts.length} alerts
            {totals.length > 0 && <span className="font-normal"> · {formatCounts(totals)}</span>}
          </div>
          <div className="text-sm text-muted">
            {result.frames_processed} of {result.frames_read} frames processed in{" "}
            {result.processing_time_s.toFixed(1)}s ·{" "}
            {new Set(result.alerts.map((a) => a.frame_index)).size} frames with an alert
          </div>
        </div>
        <label className="flex items-center gap-2 text-sm">
          Group by
          <select
            value={bucketS}
            onChange={(e) => setBucketS(Number(e.target.value))}
            className="rounded-md border border-line bg-background px-2 py-1"
          >
            {BUCKET_OPTIONS.map((s) => (
              <option key={s} value={s}>
                {s}s
              </option>
            ))}
          </select>
        </label>
      </div>
      <p className="text-xs text-muted">
        Every alert is one detection in one frame. Frames are checked independently, so an object
        that stays in the zone alerts on every frame it&apos;s seen in — the groups below are slices
        of clip time, not individual objects. Click a frame to enlarge it:{" "}
        <span className="text-sky-500">blue box</span> = person,{" "}
        <span className="text-amber-500">amber box</span> = vehicle,{" "}
        <span className="text-yellow-400">yellow dot</span> = the point tested against the zone.
      </p>
      <AlertBuckets
        buckets={buckets}
        frameWidth={result.frame_width}
        frameHeight={result.frame_height}
        zonePoints={zonePoints}
        clipDurationS={result.frames_read / result.clip_fps}
      />
    </>
  );
}
