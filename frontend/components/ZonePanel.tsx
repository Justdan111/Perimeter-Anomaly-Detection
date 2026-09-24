import type { Zone } from "@/lib/api";

import { EdgeNote } from "./EdgeNote";
import { ZoneOverlay } from "./ZoneOverlay";

/** The watched zone drawn over a frame, with its description and the
 *  frame-edge note. Shared by the sample clip and uploaded clips. */
export function ZonePanel({
  imageUrl,
  alt,
  zone,
  source,
}: {
  imageUrl: string;
  alt: string;
  zone: Zone;
  /** Where the zone comes from, e.g. "from the backend zone config". */
  source: string;
}) {
  return (
    <>
      <ZoneOverlay
        imageUrl={imageUrl}
        alt={alt}
        frameWidth={zone.frame_width}
        frameHeight={zone.frame_height}
        zonePoints={zone.points}
      />
      <p className="text-sm">
        <span className="font-medium">{zone.name}</span>{" "}
        <span className="text-muted">
          · {zone.points.length}-point polygon on a {zone.frame_width}×{zone.frame_height} frame,{" "}
          {source}
        </span>
      </p>
      <EdgeNote points={zone.points} frameWidth={zone.frame_width} frameHeight={zone.frame_height} />
    </>
  );
}
