// A small dot in the named colour, next to the colour's name in alert rows.
const SWATCH: Record<string, string> = {
  black: "#111",
  white: "#f5f5f5",
  gray: "#9ca3af",
  red: "#dc2626",
  orange: "#f97316",
  yellow: "#facc15",
  green: "#16a34a",
  blue: "#2563eb",
  purple: "#9333ea",
  pink: "#ec4899",
  brown: "#92400e",
};

export function ColorSwatch({ color }: { color: string | null | undefined }) {
  const fill = color ? SWATCH[color] : undefined;
  if (!fill) return null;
  return (
    <span
      aria-hidden
      className="inline-block size-2.5 shrink-0 rounded-full border border-line align-middle"
      style={{ background: fill }}
    />
  );
}
