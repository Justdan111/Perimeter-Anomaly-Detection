import type { BBox } from "@/lib/alerts";

export interface OverlayBox {
  bbox: BBox;
  anchor: [number, number];
  className: string;
}

interface Props {
  imageUrl: string;
  alt: string;
  /** Size of the frame the zone points and boxes are expressed in. */
  frameWidth: number;
  frameHeight: number;
  zonePoints: [number, number][];
  boxes?: OverlayBox[];
  /** Thinner strokes for small thumbnails. */
  compact?: boolean;
}

const VEHICLES = new Set(["car", "truck", "bus", "motorcycle"]);

/**
 * An image with the zone polygon (and optionally detection boxes) drawn over
 * it. The SVG's viewBox is the zone's own frame size, so coordinates are
 * used exactly as the backend reports them — no scaling maths here. That is
 * also what makes it correct on the half-size snapshots: the browser
 * stretches the 1280x720 coordinate space over the 640x360 image.
 */
export function ZoneOverlay({
  imageUrl,
  alt,
  frameWidth,
  frameHeight,
  zonePoints,
  boxes = [],
  compact = false,
}: Props) {
  const stroke = compact ? 4 : 3;
  return (
    <div className="relative w-full" style={{ aspectRatio: `${frameWidth} / ${frameHeight}` }}>
      {/* Plain <img>: the API serves fixed-size JPEGs from another origin, so
          next/image's resizing pipeline adds nothing here. */}
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={imageUrl}
        alt={alt}
        loading="lazy"
        className="absolute inset-0 h-full w-full rounded-md object-fill"
      />
      <svg
        viewBox={`0 0 ${frameWidth} ${frameHeight}`}
        preserveAspectRatio="none"
        className="absolute inset-0 h-full w-full"
        aria-hidden
      >
        <polygon
          points={zonePoints.map(([x, y]) => `${x},${y}`).join(" ")}
          fill="rgb(239 68 68 / 0.18)"
          stroke="rgb(239 68 68)"
          strokeWidth={stroke}
          strokeLinejoin="round"
        />
        {boxes.map(({ bbox: [x1, y1, x2, y2], anchor: [ax, ay], className }, i) => {
          const color = VEHICLES.has(className) ? "rgb(245 158 11)" : "rgb(56 189 248)";
          return (
            <g key={i}>
              <rect
                x={x1}
                y={y1}
                width={x2 - x1}
                height={y2 - y1}
                fill="none"
                stroke={color}
                strokeWidth={stroke}
              />
              <circle cx={ax} cy={ay} r={compact ? 9 : 7} fill="rgb(250 204 21)" stroke="black" strokeWidth={2} />
            </g>
          );
        })}
      </svg>
    </div>
  );
}
