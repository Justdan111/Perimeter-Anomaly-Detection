// Run with: npm test
import assert from "node:assert/strict";
import { describe, test } from "node:test";

import type { Alert } from "./alerts.ts";
import { colorLabel, colorsPresent, filterAlerts, hasColorData } from "./filters.ts";

function alert(cls: string, colors: Partial<Pick<Alert, "color" | "upper_color" | "lower_color">>, frame = 0): Alert {
  return {
    timestamp_s: frame / 24,
    frame_index: frame,
    class_name: cls,
    confidence: 0.8,
    bbox: [0, 0, 10, 10],
    anchor: [5, 10],
    zone_name: "z",
    snapshot: "s",
    snapshot_url: "/s",
    color: null,
    upper_color: null,
    lower_color: null,
    ...colors,
  };
}

const RED_CAR = alert("car", { color: "red" });
const BLUE_CAR = alert("car", { color: "blue" });
const BLUE_TOP = alert("person", { upper_color: "blue", lower_color: "black" });
const BLUE_BOTTOM = alert("person", { upper_color: "white", lower_color: "blue" });
const RED_BAG = alert("handbag", { color: "red" });
const ALL = [RED_CAR, BLUE_CAR, BLUE_TOP, BLUE_BOTTOM, RED_BAG];

describe("filterAlerts", () => {
  test("no filter shows everything", () => {
    assert.deepEqual(filterAlerts(ALL, { classes: null, color: null, personPart: "any" }), ALL);
  });

  test("'red vehicles': class + colour together", () => {
    const got = filterAlerts(ALL, { classes: new Set(["car"]), color: "red", personPart: "any" });
    assert.deepEqual(got, [RED_CAR]);
  });

  test("a colour on its own matches vehicles, objects, and people wearing it anywhere", () => {
    assert.deepEqual(filterAlerts(ALL, { classes: null, color: "blue", personPart: "any" }), [
      BLUE_CAR,
      BLUE_TOP,
      BLUE_BOTTOM,
    ]);
  });

  test("'blue upper-body clothing' matches only people with a blue top", () => {
    assert.deepEqual(filterAlerts(ALL, { classes: null, color: "blue", personPart: "upper" }), [BLUE_TOP]);
  });

  test("'blue lower-body clothing' matches only people with blue trousers", () => {
    assert.deepEqual(filterAlerts(ALL, { classes: null, color: "blue", personPart: "lower" }), [BLUE_BOTTOM]);
  });

  test("alerts without colour data never match a colour filter, and don't break anything", () => {
    const legacy = { ...RED_CAR, color: undefined } as unknown as Alert; // a Phase 1 record
    assert.deepEqual(filterAlerts([legacy], { classes: null, color: "red", personPart: "any" }), []);
    assert.deepEqual(filterAlerts([legacy], { classes: null, color: null, personPart: "any" }), [legacy]);
  });

  test("filters keep the original order", () => {
    const got = filterAlerts([BLUE_BOTTOM, BLUE_CAR, BLUE_TOP], { classes: null, color: "blue", personPart: "any" });
    assert.deepEqual(got, [BLUE_BOTTOM, BLUE_CAR, BLUE_TOP]);
  });
});

describe("colorsPresent / hasColorData", () => {
  test("lists each real colour once, most common first, without mixed/unknown", () => {
    const more = [...ALL, alert("car", { color: "mixed" }), alert("car", { color: "unknown" }), alert("car", { color: "blue" })];
    assert.deepEqual(colorsPresent(more), ["blue", "red", "black", "white"]);
  });

  test("a job processed before colour extraction has no colour data", () => {
    const legacy = [{ ...RED_CAR, color: undefined, upper_color: undefined, lower_color: undefined } as unknown as Alert];
    assert.equal(hasColorData(legacy), false);
    assert.equal(hasColorData(ALL), true);
  });
});

describe("colorLabel", () => {
  test("vehicles and objects show one colour; people show top and bottom", () => {
    assert.equal(colorLabel(RED_CAR), "red");
    assert.equal(colorLabel(BLUE_TOP), "blue top, black bottom");
  });

  test("mixed and unknown are said plainly; missing data shows nothing", () => {
    assert.equal(colorLabel(alert("car", { color: "mixed" })), "mixed colours");
    assert.equal(colorLabel(alert("person", { upper_color: "unknown", lower_color: "blue" })), "? top, blue bottom");
    assert.equal(colorLabel({ ...RED_CAR, color: undefined } as unknown as Alert), null);
  });
});
