"use client";

import { useState } from "react";

import { SampleClip } from "./SampleClip";
import { UploadClip } from "./UploadClip";

export type Tab = "upload" | "sample";

export function Dashboard({
  initialTab,
  initialJobId,
}: {
  initialTab: Tab;
  initialJobId: string | null;
}) {
  const [tab, setTab] = useState<Tab>(initialTab);

  const choose = (next: Tab) => {
    setTab(next);
    const url = new URL(window.location.href);
    if (next === "sample") url.searchParams.set("tab", "sample");
    else url.searchParams.delete("tab");
    window.history.replaceState(null, "", url);
  };

  return (
    <main className="mx-auto w-full max-w-6xl space-y-6 px-4 py-8 sm:px-6">
      <header className="space-y-1">
        <h1 className="text-2xl font-semibold tracking-tight">Perimeter alerts</h1>
        <p className="text-muted">
          Find the moments a person or vehicle appears in a video, with a timestamped snapshot of
          each.
        </p>
      </header>

      <div role="tablist" aria-label="Clip source" className="flex gap-1 border-b border-line">
        {(
          [
            ["upload", "Upload a clip"],
            ["sample", "Sample clip (demo)"],
          ] as const
        ).map(([id, label]) => (
          <button
            key={id}
            role="tab"
            type="button"
            aria-selected={tab === id}
            onClick={() => choose(id)}
            className="-mb-px border-b-2 border-transparent px-3 py-2 text-sm font-medium text-muted aria-selected:border-foreground aria-selected:text-foreground"
          >
            {label}
          </button>
        ))}
      </div>

      {/* Both stay mounted so switching tabs doesn't lose an upload in
          progress or a finished result. */}
      <div role="tabpanel" hidden={tab !== "upload"}>
        <UploadClip initialJobId={initialJobId} />
      </div>
      <div role="tabpanel" hidden={tab !== "sample"}>
        <SampleClip />
      </div>
    </main>
  );
}
