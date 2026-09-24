import { EDGE_TOLERANCE_FRACTION, isWholeFrameZone, zoneBottomMargin } from "@/lib/alerts";

export const EDGE_EXPLANATION =
  "An alert's position is the bottom-centre of its box — where feet or tyres meet the " +
  "ground. When someone is cut off by the bottom of the frame, their box stops at the " +
  "frame edge, so that point sits on the edge instead of at their real, off-screen feet. " +
  "Their position, and whether they count as inside the zone, can be off.";

/** Small inline marker on an alert whose box is cut off by the frame bottom. */
export function EdgeBadge({ label }: { label: string }) {
  return (
    <span
      title={EDGE_EXPLANATION}
      className="cursor-help rounded border border-amber-500/50 bg-amber-500/10 px-1.5 py-0.5 text-xs text-amber-700 dark:text-amber-300"
    >
      ⚠ {label}
    </span>
  );
}

/**
 * Always-visible note under the zone view. States the limitation, and says
 * whether this particular zone is exposed to it.
 */
export function EdgeNote({
  points,
  frameWidth,
  frameHeight,
}: {
  points: [number, number][];
  frameWidth: number;
  frameHeight: number;
}) {
  if (isWholeFrameZone(points, frameWidth, frameHeight)) {
    // Every point in the frame is inside, the bottom edge included, so a
    // box cut off by the edge can't change any verdict. The warning below
    // ("reaches the bottom edge — may be misplaced") would be wrong here.
    return (
      <p className="rounded-md border border-line bg-panel px-3 py-2 text-sm">
        The zone is the whole frame: every detected person or vehicle of the selected kind alerts,
        wherever it is. (Uploads can&apos;t have a drawn zone yet.)
      </p>
    );
  }
  const margin = zoneBottomMargin(points, frameHeight);
  const exposed = margin <= frameHeight * EDGE_TOLERANCE_FRACTION;
  return (
    <details
      className={`rounded-md border px-3 py-2 text-sm ${
        exposed
          ? "border-amber-500/50 bg-amber-500/10"
          : "border-line bg-panel"
      }`}
      open={exposed}
    >
      <summary className="cursor-pointer select-none">
        {exposed ? (
          <>⚠ This zone reaches the bottom edge of the frame — alerts there may be misplaced.</>
        ) : (
          <>
            Position is measured at the bottom-centre of each box. This zone&apos;s lowest point
            is {margin}px above the frame edge, so cut-off boxes can&apos;t fall inside it.
          </>
        )}
      </summary>
      <p className="mt-2 text-muted">
        {EDGE_EXPLANATION} Alerts affected by this are marked{" "}
        <EdgeBadge label="at frame edge" />. Estimating the real position is out of scope for this
        version.
      </p>
    </details>
  );
}
