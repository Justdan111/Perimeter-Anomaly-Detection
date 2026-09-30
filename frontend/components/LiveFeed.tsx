"use client";

// Live feed (Phase 4, Stage 4a validation): alerts pushed over a WebSocket as
// each frame from the local agent is processed. No history is kept on the
// server; this page shows what arrives while it's open.

import { useEffect, useReducer, useState } from "react";

import { API_URL } from "@/lib/api";
import { INITIAL_LIVE, boxPercent, liveSocketUrl, reduceLive, summarise, type LiveMessage } from "@/lib/live";

type Conn = "connecting" | "open" | "closed";

export function LiveFeed() {
  const [state, dispatch] = useReducer(reduceLive, INITIAL_LIVE);
  const [conn, setConn] = useState<Conn>("connecting");
  // A ticking clock, so "no new frame for N s" updates without new messages.
  const [now, setNow] = useState<number | null>(null);
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now() / 1000), 1000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    let ws: WebSocket | null = null;
    let retry: ReturnType<typeof setTimeout> | undefined;
    let delay = 1000;
    let stopped = false;

    const connect = () => {
      setConn("connecting");
      ws = new WebSocket(liveSocketUrl(API_URL));
      ws.onopen = () => {
        delay = 1000;
        setConn("open");
      };
      ws.onmessage = (e) => dispatch(JSON.parse(e.data) as LiveMessage);
      ws.onclose = () => {
        setConn("closed");
        if (!stopped) retry = setTimeout(connect, (delay = Math.min(delay * 2, 15000)));
      };
    };
    connect();
    return () => {
      stopped = true;
      clearTimeout(retry);
      ws?.close();
    };
  }, []);

  const f = state.latest;
  const ageS = f && now !== null ? Math.max(0, now - f.received_at) : null;

  return (
    <main className="mx-auto w-full max-w-6xl space-y-6 px-4 py-8 sm:px-6">
      <header className="space-y-1">
        <h1 className="text-2xl font-semibold tracking-tight">Live feed (experiment)</h1>
        <p className="text-muted">
          Frames from a local camera agent, checked one at a time. The whole frame is watched; every
          person or vehicle seen is shown, with no de-duplication.
        </p>
      </header>

      <div className="flex flex-wrap gap-4 text-sm">
        <span>
          Connection: <strong>{conn}</strong>
        </span>
        <span>Frames received here: {state.frames}</span>
        <span>Frames dropped (model busy with an upload): {state.dropped}</span>
        {f && <span>Detection: {Math.round(f.detect_ms)} ms</span>}
      </div>

      <div className="grid gap-6 lg:grid-cols-[2fr_1fr]">
        <div className="relative w-full overflow-hidden rounded border border-line bg-black">
          {f ? (
            <>
              {/* eslint-disable-next-line @next/next/no-img-element -- in-memory data URL, nothing to optimise */}
              <img src={`data:image/jpeg;base64,${f.preview_jpeg}`} alt={`Latest frame from ${f.camera}`} className="block w-full" />
              {f.alerts.map((a, i) => {
                const b = boxPercent(a.bbox, f.frame_width, f.frame_height);
                return (
                  <div
                    key={i}
                    className="absolute border-2 border-red-500"
                    style={{ left: `${b.left}%`, top: `${b.top}%`, width: `${b.width}%`, height: `${b.height}%` }}
                  >
                    <span className="bg-red-500 px-1 text-xs text-white">
                      {a.class_name} {Math.round(a.confidence * 100)}%
                    </span>
                  </div>
                );
              })}
            </>
          ) : (
            <p className="p-8 text-center text-muted">Waiting for frames from the agent…</p>
          )}
        </div>

        <section aria-label="Recent alerts" className="space-y-2">
          <h2 className="font-medium">Recent alerts</h2>
          {ageS !== null && ageS > 5 && <p className="text-sm text-muted">No new frame for {Math.round(ageS)} s.</p>}
          <ul className="max-h-[32rem] space-y-1 overflow-y-auto text-sm">
            {state.recent.map((r, i) => (
              <li key={`${r.at}-${i}`} className="flex justify-between gap-2 border-b border-line py-1">
                <span>{summarise(r.alerts)}</span>
                <time className="text-muted">{new Date(r.at * 1000).toLocaleTimeString()}</time>
              </li>
            ))}
          </ul>
        </section>
      </div>
    </main>
  );
}
