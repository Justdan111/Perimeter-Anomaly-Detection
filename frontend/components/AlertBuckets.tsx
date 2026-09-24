"use client";

import { useState } from "react";

import {
  type Bucket,
  type FrameGroup,
  formatCounts,
  isWholeFrameZone,
  touchesBottomEdge,
} from "@/lib/alerts";
import { apiUrl } from "@/lib/api";
import { colorLabel } from "@/lib/filters";

import { ColorSwatch } from "./ColorSwatch";

import { EdgeBadge } from "./EdgeNote";
import { ZoneOverlay } from "./ZoneOverlay";

interface Props {
  buckets: Bucket[];
  frameWidth: number;
  frameHeight: number;
  zonePoints: [number, number][];
  /** Clip length, so the last bucket's label doesn't run past the end. */
  clipDurationS: number;
  /** Shown when there are no buckets (e.g. because filters hide everything). */
  emptyMessage?: string;
}

const fmt = (s: number) => `${s.toFixed(1)}s`;

export function AlertBuckets({
  buckets,
  frameWidth,
  frameHeight,
  zonePoints,
  clipDurationS,
  emptyMessage = "No alerts: nothing entered the zone.",
}: Props) {
  // With a whole-frame zone a box cut off by the frame edge is inside either
  // way, so the frame-edge warning can't apply (and ZonePanel says so).
  const edgeMatters = !isWholeFrameZone(zonePoints, frameWidth, frameHeight);
  if (buckets.length === 0) {
    return <p className="text-sm text-muted">{emptyMessage}</p>;
  }
  return (
    <ol className="space-y-2" data-testid="buckets">
      {buckets.map((bucket) => {
        const atEdge = edgeMatters
          ? bucket.alerts.filter((a) => touchesBottomEdge(a.bbox, frameHeight)).length
          : 0;
        return (
          <li key={bucket.index}>
            <details className="group rounded-lg border border-line bg-panel">
              <summary className="flex cursor-pointer list-none flex-wrap items-baseline gap-x-3 gap-y-1 px-4 py-3 select-none">
                <span className="inline-block w-3 text-muted transition-transform group-open:rotate-90">
                  ›
                </span>
                <span className="font-mono text-sm tabular-nums">
                  {fmt(bucket.startS)}–{fmt(Math.min(bucket.endS, clipDurationS))}
                </span>
                <span className="font-medium" data-testid="bucket-counts">
                  {formatCounts(bucket.counts)}
                </span>
                <span className="text-sm text-muted">
                  {bucket.alerts.length} alerts in {bucket.frames.length} frames
                </span>
                {atEdge > 0 && <EdgeBadge label={`${atEdge} at frame edge`} />}
              </summary>

              <div className="grid gap-4 border-t border-line p-4 sm:grid-cols-2">
                {bucket.frames.map((frame) => (
                  <FrameCard
                    key={frame.frameIndex}
                    frame={frame}
                    edgeMatters={edgeMatters}
                    frameWidth={frameWidth}
                    frameHeight={frameHeight}
                    zonePoints={zonePoints}
                  />
                ))}
              </div>
            </details>
          </li>
        );
      })}
    </ol>
  );
}

function FrameCard({
  frame,
  edgeMatters,
  frameWidth,
  frameHeight,
  zonePoints,
}: {
  frame: FrameGroup;
  edgeMatters: boolean;
  frameWidth: number;
  frameHeight: number;
  zonePoints: [number, number][];
}) {
  // Thumbnails are small; click one to see its boxes at full panel width.
  const [enlarged, setEnlarged] = useState(false);
  return (
    <figure className={`space-y-2 ${enlarged ? "sm:col-span-2" : ""}`}>
      <button
        type="button"
        onClick={() => setEnlarged((e) => !e)}
        aria-expanded={enlarged}
        aria-label={`${enlarged ? "Shrink" : "Enlarge"} frame ${frame.frameIndex}`}
        className="block w-full cursor-zoom-in rounded-md focus-visible:outline-2 focus-visible:outline-offset-2 aria-expanded:cursor-zoom-out"
      >
        <ZoneOverlay
          compact={!enlarged}
          imageUrl={apiUrl(frame.snapshotUrl)}
          alt={`Frame ${frame.frameIndex} at ${fmt(frame.timestampS)}`}
          frameWidth={frameWidth}
          frameHeight={frameHeight}
          zonePoints={zonePoints}
          boxes={frame.alerts.map((a) => ({
            bbox: a.bbox,
            anchor: a.anchor,
            className: a.class_name,
          }))}
        />
      </button>
      <figcaption className="space-y-1 text-sm">
        <div className="font-mono tabular-nums text-muted">
          {frame.timestampS.toFixed(2)}s · frame {frame.frameIndex}
        </div>
        <ul className="space-y-0.5">
          {frame.alerts.map((a, i) => (
            <li key={i} className="flex flex-wrap items-center gap-2">
              <span className="font-medium">{a.class_name}</span>
              {colorLabel(a) && (
                <span className="flex items-center gap-1 text-xs" data-testid="alert-color">
                  {a.class_name === "person" ? (
                    <>
                      <ColorSwatch color={a.upper_color} />
                      <ColorSwatch color={a.lower_color} />
                    </>
                  ) : (
                    <ColorSwatch color={a.color} />
                  )}
                  {colorLabel(a)}
                </span>
              )}
              <span className="font-mono text-xs tabular-nums text-muted">
                {(a.confidence * 100).toFixed(0)}%
              </span>
              {edgeMatters && touchesBottomEdge(a.bbox, frameHeight) && (
                <EdgeBadge label="at frame edge" />
              )}
            </li>
          ))}
        </ul>
      </figcaption>
    </figure>
  );
}
