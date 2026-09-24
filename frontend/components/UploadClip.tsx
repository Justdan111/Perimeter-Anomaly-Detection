"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import {
  type AlertsResponse,
  type ClassChoice,
  type Job,
  type UploadLimits,
  ApiError,
  apiUrl,
  getJob,
  getJobAlerts,
  getUploadLimits,
  uploadClip,
} from "@/lib/api";
import { checkFileBeforeUpload, describeJob } from "@/lib/uploads";

import { ProcessingState } from "./ProcessingState";
import { ResultsView } from "./ResultsView";
import { ZonePanel } from "./ZonePanel";

const POLL_MS = 1500;

// Display names only; the list of choices comes from the server
// (GET /uploads/limits), so the form can't offer a class it would refuse.
const CLASS_LABELS: Record<string, string> = {
  person: "People",
  vehicle: "Vehicles",
  bicycle: "Bicycles",
  dog: "Dogs",
  cat: "Cats",
  backpack: "Backpacks",
  handbag: "Handbags",
  suitcase: "Suitcases",
};
const classLabel = (c: string) => CLASS_LABELS[c] ?? c;
const DEFAULT_CLASSES: ClassChoice[] = ["person", "vehicle"];

function setJobInUrl(jobId: string | null) {
  const url = new URL(window.location.href);
  if (jobId) url.searchParams.set("job", jobId);
  else url.searchParams.delete("job");
  window.history.replaceState(null, "", url);
}

export function UploadClip({ initialJobId }: { initialJobId: string | null }) {
  const [limits, setLimits] = useState<UploadLimits | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [classes, setClasses] = useState<ClassChoice[]>(DEFAULT_CLASSES);
  const toggleClass = (choice: ClassChoice) =>
    setClasses((current) =>
      current.includes(choice) ? current.filter((c) => c !== choice) : [...current, choice],
    );
  const [uploadFraction, setUploadFraction] = useState<number | null>(null);
  const [uploadSince, setUploadSince] = useState<number | null>(null);
  // From ?job=... (read on the server): a reload resumes the same job.
  const [jobId, setJobId] = useState<string | null>(initialJobId);
  const [job, setJob] = useState<Job | null>(null);
  const [result, setResult] = useState<AlertsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    getUploadLimits().then(setLimits, (e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  // Poll the job until it finishes, then fetch its alerts.
  useEffect(() => {
    if (!jobId) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const poll = async () => {
      try {
        const current = await getJob(jobId);
        if (cancelled) return;
        setJob(current);
        if (current.status === "complete") {
          const alerts = await getJobAlerts(jobId);
          if (!cancelled) setResult(alerts);
          return;
        }
        if (current.status === "failed") return;
        timer = setTimeout(poll, POLL_MS);
      } catch (e) {
        if (cancelled) return;
        if (e instanceof ApiError && e.status === 404) {
          setError(
            "This job wasn't found — its results may have expired (they're kept for 7 days). " +
              "Upload the clip again.",
          );
          setJobId(null);
          setJobInUrl(null);
        } else {
          // A network blip shouldn't end the wait; keep polling.
          timer = setTimeout(poll, POLL_MS * 2);
        }
      }
    };
    poll();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [jobId]);

  const submit = useCallback(async () => {
    if (!file || !limits) return;
    setError(null);
    const problem = checkFileBeforeUpload(file, limits.max_bytes);
    if (problem) {
      setError(problem);
      return;
    }
    setUploadFraction(0);
    setUploadSince(Date.now());
    try {
      const created = await uploadClip(file, classes, setUploadFraction);
      setJob(created);
      setResult(null);
      setJobId(created.job_id);
      setJobInUrl(created.job_id);
    } catch (e) {
      setError(
        e instanceof ApiError && e.status === 429
          ? "The server is busy with other uploads right now. Try again in a few minutes."
          : e instanceof Error
            ? e.message
            : String(e),
      );
    } finally {
      setUploadFraction(null);
      setUploadSince(null);
    }
  }, [file, limits, classes]);

  const reset = useCallback(() => {
    setJobId(null);
    setJob(null);
    setResult(null);
    setError(null);
    setFile(null);
    setJobInUrl(null);
    if (fileInput.current) fileInput.current.value = "";
  }, []);

  const uploading = uploadFraction !== null;
  const tracking = jobId !== null;
  const status = job ? describeJob(job) : null;

  return (
    <div className="space-y-6">
      {error && (
        <div role="alert" className="rounded-md border border-red-500/50 bg-red-500/10 px-4 py-3 text-sm">
          {error}
        </div>
      )}

      <div className="grid gap-8 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
        <section className="space-y-4" aria-labelledby="upload-heading">
          <h2 id="upload-heading" className="text-lg font-semibold">
            {tracking ? "Your clip" : "Upload a clip"}
          </h2>

          {!tracking && (
            <form
              className="space-y-4 rounded-lg border border-line bg-panel p-4"
              onSubmit={(e) => {
                e.preventDefault();
                submit();
              }}
            >
              <label className="block space-y-1.5 text-sm">
                <span className="font-medium">Video file</span>
                <input
                  ref={fileInput}
                  type="file"
                  accept="video/*,.mp4,.mov,.m4v,.webm,.mkv,.avi,.ts"
                  disabled={uploading}
                  onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                  className="block w-full text-sm file:mr-3 file:rounded-md file:border file:border-line file:bg-background file:px-3 file:py-1.5 file:text-sm"
                />
              </label>

              <fieldset className="space-y-1.5 text-sm" disabled={uploading}>
                <legend className="font-medium">Alert on</legend>
                <div className="flex flex-wrap gap-2">
                  {(limits?.classes ?? DEFAULT_CLASSES).map((choice) => (
                    <label
                      key={choice}
                      className="flex cursor-pointer items-center gap-2 rounded-md border border-line bg-background px-3 py-1.5 has-checked:border-foreground"
                    >
                      <input
                        type="checkbox"
                        name="classes"
                        value={choice}
                        checked={classes.includes(choice)}
                        onChange={() => toggleClass(choice)}
                      />
                      {classLabel(choice)}
                    </label>
                  ))}
                </div>
                <p className="text-xs text-muted">
                  Vehicles = car, truck, bus, motorcycle. Each alert also records a colour
                  (vehicles and objects) or top and bottom clothing colours (people).
                </p>
              </fieldset>

              {limits && (
                <p className="text-xs text-muted">
                  Up to {Math.floor(limits.max_bytes / (1024 * 1024))} MB and{" "}
                  {limits.max_duration_s.toFixed(0)} s, at most {limits.max_resolution.replace("x", "×")}.{" "}
                  {limits.formats}. The whole frame is watched, and {limits.sample_fps} frames are
                  checked per second of video.
                </p>
              )}

              <button
                type="submit"
                disabled={!file || !limits || uploading || classes.length === 0}
                className="rounded-md bg-foreground px-4 py-2 text-sm font-medium text-background transition-opacity hover:opacity-85 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {uploading ? "Uploading…" : "Upload and process"}
              </button>
            </form>
          )}

          {job && (
            <>
              <ZonePanel
                imageUrl={apiUrl(job.reference_frame_url)}
                alt={`First frame of ${job.filename}`}
                zone={job.zone}
                source="the default for uploads"
              />
              <p className="text-sm text-muted">
                {job.filename} · {job.clip.duration_s.toFixed(1)} s · {job.clip.width}×{job.clip.height}{" "}
                · alerting on{" "}
                {(Array.isArray(job.classes) ? job.classes : [job.classes])
                  .map((c) => classLabel(c).toLowerCase())
                  .join(", ")}
              </p>
            </>
          )}
        </section>

        <section className="space-y-4" aria-labelledby="upload-alerts-heading">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h2 id="upload-alerts-heading" className="text-lg font-semibold">
              Alerts
            </h2>
            {tracking && (job?.status === "complete" || job?.status === "failed") && (
              <button
                type="button"
                onClick={reset}
                className="rounded-md border border-line px-4 py-2 text-sm font-medium hover:bg-panel"
              >
                Upload another clip
              </button>
            )}
          </div>

          {uploading && (
            <ProcessingState
              since={uploadSince}
              label={`Uploading… ${Math.round((uploadFraction ?? 0) * 100)}%`}
              fraction={uploadFraction ?? 0}
            />
          )}

          {job && status && (job.status === "queued" || job.status === "processing") && (
            <ProcessingState
              since={(job.started_at ?? job.created_at) * 1000}
              label={status.label}
              fraction={status.fraction}
              note="Runs in the background — you can leave this page open or come back to this URL. CPU-only on a free-tier server, so expect minutes, not seconds."
            />
          )}

          {job?.status === "failed" && (
            <div role="alert" className="rounded-md border border-red-500/50 bg-red-500/10 px-4 py-3 text-sm">
              Processing failed: {job.error}
            </div>
          )}

          {!tracking && !uploading && (
            <p className="text-sm text-muted">Choose a clip and what to alert on, then upload it.</p>
          )}

          {result && job && <ResultsView result={result} zonePoints={job.zone.points} />}
        </section>
      </div>
    </div>
  );
}
