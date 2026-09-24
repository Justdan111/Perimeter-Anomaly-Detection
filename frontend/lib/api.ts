// Typed client for the FastAPI backend (backend/app/main.py).
import type { Alert } from "./alerts";

// Inlined at build time (NEXT_PUBLIC_), so production builds must set it.
export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(
  /\/$/,
  "",
);

export interface Zone {
  name: string;
  points: [number, number][];
  frame_width: number;
  frame_height: number;
}

export interface ClipInfo {
  clip_id: string;
  zone: Zone;
  reference_frame_url: string;
  processed: boolean;
}

export interface ProcessSummary {
  clip_id: string;
  frames_read: number;
  frames_processed: number;
  processing_time_s: number;
  alert_count: number;
}

export interface AlertsResponse {
  clip_id: string;
  zone_name: string;
  frame_width: number;
  frame_height: number;
  clip_fps: number;
  sample_fps: number | null;
  frames_read: number;
  frames_processed: number;
  processing_time_s: number;
  alerts: Alert[];
}

export class ApiError extends Error {
  // A plain field rather than a constructor parameter property: Node's
  // type-stripping (used by `npm test`) doesn't support parameter properties.
  readonly status?: number;

  constructor(message: string, status?: number) {
    super(message);
    this.status = status;
  }
}

/**
 * Absolute URL for a path or URL the API returned (snapshot_url,
 * reference_frame_url). Upload results on R2 come back as complete, signed
 * links and are used untouched; everything else is a path on the API.
 */
export function apiUrl(path: string): string {
  return /^https?:\/\//.test(path) ? path : `${API_URL}${path}`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(apiUrl(path), { cache: "no-store", ...init });
  } catch {
    throw new ApiError(`Couldn't reach the API at ${API_URL}. Is the backend running?`);
  }
  if (!response.ok) {
    let detail = response.statusText;
    try {
      detail = (await response.json()).detail ?? detail;
    } catch {
      // not JSON; keep statusText
    }
    throw new ApiError(`${response.status}: ${detail}`, response.status);
  }
  return response.json() as Promise<T>;
}

export const getClip = (clipId: string) => request<ClipInfo>(`/clips/${clipId}`);

export const processClip = (clipId: string) =>
  request<ProcessSummary>(`/clips/${clipId}/process`, { method: "POST" });

export const getAlerts = (clipId: string) => request<AlertsResponse>(`/clips/${clipId}/alerts`);

// --- uploads and jobs (Phase 1) ----------------------------------------------------

export type ClassChoice = "person" | "vehicle" | "both";

export interface UploadLimits {
  max_bytes: number;
  max_duration_s: number;
  max_resolution: string;
  max_fps: number;
  classes: ClassChoice[];
  sample_fps: number;
  formats: string;
}

export interface Job {
  job_id: string;
  status: "queued" | "processing" | "complete" | "failed";
  filename: string;
  classes: ClassChoice;
  clip: { width: number; height: number; fps: number; frame_count: number; duration_s: number };
  zone: Zone;
  sample_fps: number | null;
  frames_to_process: number;
  frames_processed: number;
  queue_position: number;
  created_at: number;
  started_at: number | null;
  finished_at: number | null;
  processing_time_s: number | null;
  error: string | null;
  status_url: string;
  alerts_url: string;
  reference_frame_url: string;
}

export const getUploadLimits = () => request<UploadLimits>("/uploads/limits");

export const getJob = (jobId: string) => request<Job>(`/jobs/${jobId}`);

export const getJobAlerts = (jobId: string) => request<AlertsResponse>(`/jobs/${jobId}/alerts`);

/**
 * Upload a clip. Uses XMLHttpRequest rather than fetch because fetch can't
 * report upload progress, and a large file on a slow connection needs it.
 */
export function uploadClip(
  file: File,
  classes: ClassChoice,
  onProgress: (fraction: number) => void,
): Promise<Job> {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append("file", file);
    form.append("classes", classes);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", apiUrl("/uploads"));
    xhr.responseType = "json";
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress(e.loaded / e.total);
    };
    xhr.onload = () => {
      if (xhr.status === 202) {
        resolve(xhr.response as Job);
        return;
      }
      const detail = (xhr.response as { detail?: unknown } | null)?.detail;
      const message =
        typeof detail === "string"
          ? detail
          : Array.isArray(detail) // FastAPI validation errors
            ? detail.map((d: { msg?: string }) => d.msg).join("; ")
            : xhr.statusText || "upload failed";
      reject(new ApiError(`${xhr.status}: ${message}`, xhr.status));
    };
    xhr.onerror = () =>
      reject(new ApiError(`Couldn't reach the API at ${API_URL}. Is the backend running?`));
    xhr.send(form);
  });
}
