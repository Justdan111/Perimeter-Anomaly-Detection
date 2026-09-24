"use client";

import { useEffect, useState } from "react";

/** Spinner, live elapsed time, a status line and an optional progress bar. */
export function ProcessingState({
  since,
  label,
  note,
  fraction,
}: {
  since: number | null;
  label: string;
  note?: string;
  /** 0–1 when progress is known; omitted for an indeterminate wait. */
  fraction?: number;
}) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 250);
    return () => clearInterval(id);
  }, []);
  const elapsed = since ? Math.max(0, (now - since) / 1000) : 0;
  return (
    <div role="status" className="space-y-3 rounded-lg border border-line bg-panel px-4 py-3 text-sm">
      <div className="flex items-center gap-3">
        <span className="size-4 shrink-0 animate-spin rounded-full border-2 border-muted border-t-transparent" />
        <span>
          {label} <span className="font-mono tabular-nums">{elapsed.toFixed(1)}s</span>
          {note && <span className="block text-muted">{note}</span>}
        </span>
      </div>
      {fraction !== undefined && (
        <div
          className="h-1.5 overflow-hidden rounded-full bg-line"
          role="progressbar"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(fraction * 100)}
        >
          <div className="h-full bg-foreground transition-[width]" style={{ width: `${fraction * 100}%` }} />
        </div>
      )}
    </div>
  );
}
