"use client";

import { useCallback, useEffect, useState } from "react";

import {
  type AlertsResponse,
  type ClipInfo,
  ApiError,
  apiUrl,
  getAlerts,
  getClip,
  processClip,
} from "@/lib/api";

import { ProcessingState } from "./ProcessingState";
import { ResultsView } from "./ResultsView";
import { ZonePanel } from "./ZonePanel";

// The committed sample clip: a fixed demo and a known-good regression check.
const CLIP_ID = "sample";

export function SampleClip() {
  const [clip, setClip] = useState<ClipInfo | null>(null);
  const [result, setResult] = useState<AlertsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [processingSince, setProcessingSince] = useState<number | null>(null);

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
          ? "The server is busy processing another clip (an upload, or another tab). Try again when it finishes."
          : e instanceof Error
            ? e.message
            : String(e),
      );
    } finally {
      setProcessingSince(null);
    }
  }, []);

  const processing = processingSince !== null;

  return (
    <div className="space-y-6">
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
            <ZonePanel
              imageUrl={apiUrl(clip.reference_frame_url)}
              alt="Reference frame from the sample clip with the watched zone outlined"
              zone={clip.zone}
              source="from the backend zone config"
            />
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

          {processing && (
            <ProcessingState
              since={processingSince}
              label="Running detection on the clip…"
              note="CPU-only inference on every frame: about 70 s on the free-tier server this demo runs on (measured), about 4 s on a laptop."
            />
          )}

          {!processing && !result && clip && (
            <p className="text-sm text-muted">
              Not processed yet. Processing runs the detector on every frame of the clip.
            </p>
          )}

          {result && <ResultsView result={result} zonePoints={clip?.zone.points ?? []} />}
        </section>
      </div>
    </div>
  );
}
