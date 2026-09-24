// Pure helpers for the upload flow — no React, no network. Tested with
// `node --test` (see uploads.test.ts).

export type JobState = "queued" | "processing" | "complete" | "failed";

/** The fields of a job's status these helpers need. */
export interface JobStatus {
  status: JobState;
  frames_to_process: number;
  frames_processed: number;
  queue_position: number;
  error: string | null;
}

const VIDEO_EXTENSIONS = [".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi", ".ts", ".3gp"];
const MB = 1024 * 1024;

/**
 * A quick check before uploading, so an obviously wrong file fails in a
 * second instead of after a long upload. The server checks everything again
 * — this is for a fast answer, not the gatekeeper. Returns a message, or
 * null if the file looks fine.
 */
export function checkFileBeforeUpload(
  file: { name: string; size: number; type: string },
  maxBytes: number,
): string | null {
  if (file.size === 0) return "The file is empty.";
  if (file.size > maxBytes) {
    return `The file is ${Math.ceil(file.size / MB)} MB; the limit is ${Math.floor(maxBytes / MB)} MB.`;
  }
  const lower = file.name.toLowerCase();
  // Some browsers report an empty type for .mov/.mkv, so the extension counts too.
  const looksLikeVideo =
    file.type.startsWith("video/") || VIDEO_EXTENSIONS.some((ext) => lower.endsWith(ext));
  if (!looksLikeVideo) return "That doesn't look like a video file.";
  return null;
}

/** What to show for a job: a progress fraction (0–1) and a one-line status. */
export function describeJob(job: JobStatus): { fraction: number; label: string } {
  switch (job.status) {
    case "queued":
      return {
        fraction: 0,
        label:
          job.queue_position > 0
            ? `Waiting in the queue (position ${job.queue_position})`
            : "Waiting for the server to start processing",
      };
    case "processing": {
      const fraction =
        job.frames_to_process > 0 ? Math.min(1, job.frames_processed / job.frames_to_process) : 0;
      return {
        fraction,
        label: `Processing: ${job.frames_processed} of ${job.frames_to_process} frames`,
      };
    }
    case "complete":
      return { fraction: 1, label: "Done" };
    case "failed":
      return { fraction: 0, label: `Failed: ${job.error ?? "unknown error"}` };
  }
}
