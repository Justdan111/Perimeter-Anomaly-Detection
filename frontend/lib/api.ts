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
  constructor(
    message: string,
    readonly status?: number,
  ) {
    super(message);
  }
}

/** Absolute URL for a path the API returned (snapshot_url, reference_frame_url). */
export function apiUrl(path: string): string {
  return `${API_URL}${path}`;
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
