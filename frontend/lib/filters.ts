// Filtering alerts in the results view by class and colour. Pure, tested
// with `node --test` (see filters.test.ts). Display only: never changes what
// was detected, only which alerts are shown.
import type { Alert } from "./alerts";

export type PersonPart = "any" | "upper" | "lower";

export interface AlertFilter {
  /** Detected classes to show (e.g. "car", "person"); null = all. */
  classes: Set<string> | null;
  /** A colour name; null = any colour. */
  color: string | null;
  /**
   * For people: match the colour on the top, the bottom, or either. Choosing
   * top or bottom is a question about people, so it also hides non-people.
   */
  personPart: PersonPart;
}

const NOT_A_COLOR = new Set(["mixed", "unknown"]);

function colorMatches(a: Alert, color: string, part: PersonPart): boolean {
  if (a.class_name === "person") {
    if (part === "upper") return a.upper_color === color;
    if (part === "lower") return a.lower_color === color;
    return a.upper_color === color || a.lower_color === color;
  }
  return part === "any" && a.color === color;
}

export function filterAlerts(alerts: Alert[], filter: AlertFilter): Alert[] {
  return alerts.filter(
    (a) =>
      (filter.classes === null || filter.classes.has(a.class_name)) &&
      (filter.color === null || colorMatches(a, filter.color, filter.personPart)),
  );
}

/** Whether any alert carries colour data (jobs from before Phase 2 don't). */
export function hasColorData(alerts: Alert[]): boolean {
  return alerts.some((a) => a.color != null || a.upper_color != null || a.lower_color != null);
}

/** Real colours present across the alerts, most common first. */
export function colorsPresent(alerts: Alert[]): string[] {
  const counts = new Map<string, number>();
  for (const a of alerts) {
    for (const c of [a.color, a.upper_color, a.lower_color]) {
      if (c && !NOT_A_COLOR.has(c)) counts.set(c, (counts.get(c) ?? 0) + 1);
    }
  }
  return [...counts.entries()].sort((x, y) => y[1] - x[1] || x[0].localeCompare(y[0])).map(([c]) => c);
}

function word(c: string | null | undefined): string {
  if (c === "unknown") return "?";
  return c ?? "?";
}

/** "red", "blue top, black bottom", "mixed colours" — or null if no data. */
export function colorLabel(a: Alert): string | null {
  if (a.class_name === "person") {
    if (a.upper_color == null && a.lower_color == null) return null;
    return `${word(a.upper_color)} top, ${word(a.lower_color)} bottom`;
  }
  if (a.color == null) return null;
  if (a.color === "mixed") return "mixed colours";
  if (a.color === "unknown") return "colour unclear";
  return a.color;
}
