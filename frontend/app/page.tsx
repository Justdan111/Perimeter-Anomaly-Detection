"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import { AlertBuckets } from "@/components/AlertBuckets";
import { EdgeNote } from "@/components/EdgeNote";
import { ZoneOverlay } from "@/components/ZoneOverlay";
import { countByClass, formatCounts, groupAlerts } from "@/lib/alerts";
import {
  type AlertsResponse,
  type ClipInfo,
  ApiError,
  apiUrl,
  getAlerts,
  getClip,
  processClip,
} from "@/lib/api";

// The MVP works against the one committed clip.
const CLIP_ID = "sample";

// Display grouping only (see lib/alerts.ts). 1 s is the default: ~24 frames
// at the sample clip's 23.976 fps, short enough that everything in a bucket
// is "the same moment" to a reviewer, long enough to turn ~200 alerts into a
// handful of rows.
const BUCKET_OPTIONS = [0.5, 1, 2];
const DEFAULT_BUCKET_S = 1;

export default function Dashboard() {
  const [clip, setClip] = useState<ClipInfo | null>(null);
  const [result, setResult] = useState<AlertsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [processingSince, setProcessingSince] = useState<number | null>(null);
  const [bucketS, setBucketS] = useState(DEFAULT_BUCKET_S);

  // Initial load: the zone, plus the last result if the backend has one.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const info = await getClip(CLIP_ID);
        if (cancelled) return;
        setClip(info);
        if (info.processed) {
          const alerts = await getAlerts(CLIP_ID);
          if (!cancelled) setResult(alerts);
        }
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const runProcessing = useCallback(async () => {
    setError(null);
    setResult(null);
    setProcessingSince(Date.now());
    try {
      await processClip(CLIP_ID);
      setResult(await getAlerts(CLIP_ID));
    } catch (e) {
      setError(
        e instanceof ApiError && e.status === 409
          ? "The clip is already being processed (another tab?). Try again when it finishes."
          : e instanceof Error
            ? e.message
            : String(e),
      );
    } finally {
      setProcessingSince(null);
    }
  }, []);

  const buckets = useMemo(
    () => (result ? groupAlerts(result.alerts, bucketS) : []),
    [result, bucketS],
  );
  const totals = useMemo(() => (result ? countByClass(result.alerts) : []), [result]);
  const processing = processingSince !== null;

  return (
    <main className="mx-auto w-full max-w-6xl space-y-8 px-4 py-8 sm:px-6">
      <header className="space-y-1">
        <h1 className="text-2xl font-semibold tracking-tight">Perimeter alerts</h1>
        <p className="text-muted">
          People and vehicles detected inside the watched zone of the sample clip.
        </p>
      </header>

      {error && (
        <div role="alert" className="rounded-md border border-red-500/50 bg-red-500/10 px-4 py-3 text-sm">
          {error}
        </div>
      )}

      <div className="grid gap-8 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
        <section className="space-y-3" aria-labelledby="zone-heading">
          <h2 id="zone-heading" className="text-lg font-semibold">
            Watched zone
          </h2>
          {clip ? (
            <>
              <ZoneOverlay
                imageUrl={apiUrl(clip.reference_frame_url)}
                alt="Reference frame from the sample clip with the watched zone outlined"
                frameWidth={clip.zone.frame_width}
                frameHeight={clip.zone.frame_height}
                zonePoints={clip.zone.points}
              />
              <p className="text-sm">
                <span className="font-medium">{clip.zone.name}</span>{" "}
                <span className="text-muted">
                  · {clip.zone.points.length}-point polygon on a {clip.zone.frame_width}×
                  {clip.zone.frame_height} frame, from the backend zone config
                </span>
              </p>
              <EdgeNote points={clip.zone.points} frameHeight={clip.zone.frame_height} />
            </>
          ) : (
            !error && <div className="aspect-video animate-pulse rounded-md bg-panel" />
          )}
        </section>

        <section className="space-y-4" aria-labelledby="alerts-heading">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h2 id="alerts-heading" className="text-lg font-semibold">
              Alerts
            </h2>
            <button
              type="button"
              onClick={runProcessing}
              disabled={processing || !clip}
              className="rounded-md bg-foreground px-4 py-2 text-sm font-medium text-background transition-opacity hover:opacity-85 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {processing ? "Processing…" : result ? "Process again" : "Process clip"}
            </button>
          </div>

          {processing && <ProcessingState since={processingSince} />}

          {!processing && !result && clip && (
            <p className="text-sm text-muted">
              Not processed yet. Processing runs the detector on every frame of the clip.
            </p>
          )}

          {result && (
            <>
              <div className="flex flex-wrap items-end justify-between gap-3 rounded-lg border border-line bg-panel px-4 py-3">
                <div>
                  <div className="text-lg font-semibold" data-testid="totals">
                    {result.alerts.length} alerts
                    {totals.length > 0 && (
                      <span className="font-normal"> · {formatCounts(totals)}</span>
                    )}
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
                Every alert is one detection in one frame. Frames are checked independently, so an
                object that stays in the zone alerts on every frame it&apos;s seen in — the groups
                below are slices of clip time, not individual objects. Click a frame to enlarge it:{" "}
                <span className="text-sky-500">blue box</span> = person,{" "}
                <span className="text-amber-500">amber box</span> = vehicle,{" "}
                <span className="text-yellow-400">yellow dot</span> = the point tested against the
                zone.
              </p>
              <AlertBuckets
                buckets={buckets}
                frameWidth={result.frame_width}
                frameHeight={result.frame_height}
                zonePoints={clip?.zone.points ?? []}
                clipDurationS={result.frames_read / result.clip_fps}
              />
            </>
          )}
        </section>
      </div>
    </main>
  );
}

function ProcessingState({ since }: { since: number | null }) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 250);
    return () => clearInterval(id);
  }, []);
  const elapsed = since ? (now - since) / 1000 : 0;
  return (
    <div role="status" className="flex items-center gap-3 rounded-lg border border-line bg-panel px-4 py-3 text-sm">
      <span className="size-4 animate-spin rounded-full border-2 border-muted border-t-transparent" />
      <span>
        Running detection on the clip… <span className="font-mono tabular-nums">{elapsed.toFixed(1)}s</span>
        <span className="block text-muted">
          CPU-only inference on every frame: about 70 s on the free-tier server this demo runs on (measured), about 4 s on a laptop.
        </span>
      </span>
    </div>
  );
}
