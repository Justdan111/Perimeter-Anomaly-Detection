// Live feed (Phase 4, Stage 4a): messages pushed by the backend's /live/ws.

export interface LiveAlert {
  class_name: string;
  confidence: number;
  bbox: [number, number, number, number];
  anchor: [number, number];
}

export type LiveMessage =
  | { type: "hello"; frames_processed: number; uptime_s: number }
  | {
      type: "frame";
      camera: string;
      received_at: number;
      sent_at: number | null;
      frame_width: number;
      frame_height: number;
      detections: number;
      alerts: LiveAlert[];
      detect_ms: number;
      lock_wait_ms: number;
      preview_jpeg: string;
    }
  | { type: "dropped"; camera: string; reason: string };

export type FrameMessage = Extract<LiveMessage, { type: "frame" }>;

/** http(s)://host[/] -> ws(s)://host/live/ws */
export function liveSocketUrl(apiUrl: string): string {
  const u = new URL(apiUrl);
  u.protocol = u.protocol === "https:" ? "wss:" : "ws:";
  u.pathname = u.pathname.replace(/\/$/, "") + "/live/ws";
  return u.toString();
}

export interface LiveState {
  latest: FrameMessage | null;
  // Newest first; one entry per alert-bearing frame, capped.
  recent: { at: number; camera: string; alerts: LiveAlert[] }[];
  frames: number;
  dropped: number;
}

export const INITIAL_LIVE: LiveState = { latest: null, recent: [], frames: 0, dropped: 0 };
export const RECENT_CAP = 50;

export function reduceLive(state: LiveState, msg: LiveMessage): LiveState {
  switch (msg.type) {
    case "frame":
      return {
        latest: msg,
        frames: state.frames + 1,
        dropped: state.dropped,
        recent: msg.alerts.length
          ? [{ at: msg.received_at, camera: msg.camera, alerts: msg.alerts }, ...state.recent].slice(0, RECENT_CAP)
          : state.recent,
      };
    case "dropped":
      return { ...state, dropped: state.dropped + 1 };
    default:
      return state;
  }
}

/** A box as percentages of the frame, for drawing over a scaled preview. */
export function boxPercent(bbox: LiveAlert["bbox"], width: number, height: number) {
  const [x1, y1, x2, y2] = bbox;
  return {
    left: (100 * x1) / width,
    top: (100 * y1) / height,
    width: (100 * (x2 - x1)) / width,
    height: (100 * (y2 - y1)) / height,
  };
}

/** One line per class, e.g. "2 person, 1 car". */
export function summarise(alerts: LiveAlert[]): string {
  const counts = new Map<string, number>();
  for (const a of alerts) counts.set(a.class_name, (counts.get(a.class_name) ?? 0) + 1);
  return [...counts].map(([c, n]) => `${n} ${c}`).join(", ");
}
