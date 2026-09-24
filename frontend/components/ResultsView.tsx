"use client";

import { useMemo, useState } from "react";

import { countByClass, formatCounts, groupAlerts } from "@/lib/alerts";
import type { AlertsResponse } from "@/lib/api";
import { type PersonPart, colorsPresent, filterAlerts, hasColorData } from "@/lib/filters";

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
  const [shownClasses, setShownClasses] = useState<Set<string> | null>(null);
  const [color, setColor] = useState<string | null>(null);
  const [personPart, setPersonPart] = useState<PersonPart>("any");

  const totals = useMemo(() => countByClass(result.alerts), [result]);
  const colors = useMemo(() => colorsPresent(result.alerts), [result]);
  const colorData = useMemo(() => hasColorData(result.alerts), [result]);
  const hasPeople = totals.some(([name]) => name === "person");
  const filtered = useMemo(
    () => filterAlerts(result.alerts, { classes: shownClasses, color, personPart }),
    [result, shownClasses, color, personPart],
  );
  const filtering = shownClasses !== null || color !== null;
  const buckets = useMemo(() => groupAlerts(filtered, bucketS), [filtered, bucketS]);

  const toggleClass = (name: string) => {
    const next = new Set(shownClasses ?? totals.map(([n]) => n));
    if (next.has(name)) next.delete(name);
    else next.add(name);
    setShownClasses(next.size === totals.length ? null : next);
  };

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
      {result.alerts.length > 0 && (
        <div className="space-y-2 rounded-lg border border-line px-4 py-3 text-sm" data-testid="filters">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-muted">Show</span>
            {totals.map(([name, n]) => {
              const on = shownClasses === null || shownClasses.has(name);
              return (
                <button
                  key={name}
                  type="button"
                  aria-pressed={on}
                  onClick={() => toggleClass(name)}
                  className="rounded-full border border-line px-2.5 py-0.5 text-xs aria-pressed:border-foreground aria-pressed:bg-panel aria-[pressed=false]:text-muted aria-[pressed=false]:line-through"
                >
                  {name} ({n})
                </button>
              );
            })}
          </div>
          {colorData ? (
            <div className="flex flex-wrap items-center gap-3">
              <label className="flex items-center gap-2">
                Colour
                <select
                  value={color ?? ""}
                  onChange={(e) => setColor(e.target.value || null)}
                  className="rounded-md border border-line bg-background px-2 py-1"
                >
                  <option value="">any</option>
                  {colors.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
              </label>
              {hasPeople && color && (
                <label className="flex items-center gap-2">
                  on people
                  <select
                    value={personPart}
                    onChange={(e) => setPersonPart(e.target.value as PersonPart)}
                    className="rounded-md border border-line bg-background px-2 py-1"
                  >
                    <option value="any">top or bottom</option>
                    <option value="upper">top only</option>
                    <option value="lower">bottom only</option>
                  </select>
                </label>
              )}
              {filtering && (
                <span className="text-muted" data-testid="showing">
                  showing {filtered.length} of {result.alerts.length}
                </span>
              )}
            </div>
          ) : (
            <p className="text-xs text-muted">
              Colours weren&apos;t recorded for this clip (it was processed before colour extraction
              was added), so it can only be filtered by class.
            </p>
          )}
          {colorData && (
            <p className="text-xs text-muted">
              Colours are estimated from pixels and depend on lighting — white in shade can read as
              gray, sunlit black as gray, and two-tone objects read as &ldquo;mixed&rdquo; (never
              matched by a colour filter). Treat a colour as a hint, not a fact.
            </p>
          )}
        </div>
      )}
      <p className="text-xs text-muted">
        Every alert is one detection in one frame. Frames are checked independently, so an object
        that stays in the zone alerts on every frame it&apos;s seen in — the groups below are slices
        of clip time, not individual objects. Click a frame to enlarge it:{" "}
        <span className="text-sky-500">blue box</span> = person,{" "}
        <span className="text-amber-500">amber box</span> = vehicle,{" "}
        <span className="text-violet-400">violet box</span> = bicycle, animal or bag,{" "}
        <span className="text-yellow-400">yellow dot</span> = the point tested against the zone.
      </p>
      <AlertBuckets
        buckets={buckets}
        frameWidth={result.frame_width}
        frameHeight={result.frame_height}
        zonePoints={zonePoints}
        clipDurationS={result.frames_read / result.clip_fps}
        emptyMessage={
          filtering && result.alerts.length > 0
            ? "No alerts match these filters."
            : "No alerts: nothing entered the zone."
        }
      />
    </>
  );
}
