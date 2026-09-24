# Phase 2 — Colors + extended object classes

Builds on Phase 1's upload + class-selection. Two additions, bundled together because the
second one is nearly free once the first exists.

## What this phase adds
1. **Dominant color extraction** — for vehicles and people, narrow results by color
2. **Extended class list** — bicycle, dog, cat, backpack, handbag, suitcase added to the
   selectable classes from Phase 1 (bundled here because it's just extending an existing list
   the detector already supports — YOLO26's 80 COCO classes already include all of these, so
   there's no new model work, only exposing more of what's already being detected)

## Why this order
Color is genuinely new logic (a dominant-color function + a way to filter/display results by
it). Extending the class list costs almost nothing on top of that — same reason it made sense
to bundle rather than spread across two separate efforts.

## Architecture note — written before Phase 1 existed, corrected here
This doc was drafted before Phase 1 was built. Phase 1's real architecture changes two things
about how color extraction has to work:

- **Snapshots and alert records live in R2 with signed URLs, not local paths.** Color
  extraction must happen during processing, on the raw detection crop, *before* that crop is
  saved as a snapshot to R2 — not by downloading a snapshot back afterward via its signed URL.
  It runs inside the same worker pass that already crops and uploads each snapshot.
- **Processing already costs ~2-2.75s per second of video** on the single background worker
  (shared model lock with the sample-clip endpoint). Color extraction adds real per-detection
  compute on top of that — re-measure processing time on the deployed site after adding it,
  same way Phase 1's timings were measured. If it meaningfully changes the per-second cost,
  that's real information for whether the 60s upload cap still makes sense, not something to
  quietly absorb.
- **Job records are informally versioned** — Phase 1 jobs won't have a color field. No need to
  migrate old test jobs; just make sure the dashboard doesn't break rendering an older record
  that's missing it.

## Design: one reusable primitive, one specialization
- **`dominant_color(crop) -> ColorName`**: a generic, pure function — given any detection's
  cropped region (vehicle, bag, whatever), extract the dominant color via HSV thresholding and
  map it to a small set of named buckets (red, blue, black, white, silver, green, yellow, etc.
  — pick a reasonable fixed list, don't try to be exhaustive). This works on *any* detection
  crop, not just vehicles — same function, no special-casing per class.
- **`clothing_colors(person_crop) -> (upper, lower)`**: person-specific, built on top of the
  same `dominant_color` primitive — split the person's bounding box roughly into upper and
  lower halves, run `dominant_color` on each independently.

Both are pure functions (crop in, color name out) — same testing discipline as
`zone_check.py` from Day 1: write these with fixtures (a red crop, a blue crop, a genuinely
ambiguous/mixed crop) and confirm they're actually being discriminated correctly, not just
returning plausible-looking answers.

## Scope for this phase
- [ ] `dominant_color()`, tested with real fixture crops (solid colors and at least one
      ambiguous case — a crop that's genuinely a mix, confirm the function does something
      sensible, not undefined)
- [ ] `clothing_colors()`, built on top of the above, tested similarly
- [ ] Extend `clip_processor.py`'s class list: add bicycle, dog, cat, backpack, handbag,
      suitcase alongside the existing person/vehicle filter
- [ ] Extend the Phase 1 upload UI's class-selection to include the new classes
- [ ] Add color as a field on `Alert` (vehicle color, or upper/lower clothing color for
      people) — extend the schema, don't create a parallel data structure
- [ ] Add color as a filter in the dashboard's alert view — "show me only red vehicles," "show
      me only alerts with blue upper-body clothing"
- [ ] Document the color-name bucket list and its real limitations plainly: HSV-based
      extraction is lighting-sensitive (a white car under orange sunset light may not read as
      "white") — say this in the README rather than implying the color is ground truth

## What's explicitly OUT of scope for this phase
- Car make/model (Phase 3, flagged as lower-confidence/optional — see the earlier research:
  ~60% top-1 accuracy even on the better pretrained options, and those are trained on
  US-market vehicles, so real-world accuracy on other markets' vehicles needs testing before
  trusting it)
- Live camera input (Phase 4)
- License plate recognition (not planned — real accuracy and privacy/legal concerns, as
  discussed)

## What "done" looks like
Upload a clip, select "vehicle" plus "red" as filters (or "person" plus "blue" for clothing),
and get back only the alerts that actually match both the class and the color — verified by
eye against the real clip, same discipline as every phase before this one.

## Notes
If `dominant_color()` produces obviously wrong results on real footage (e.g. shadows or motion
blur consistently throwing off a car's color), don't just accept it — try a slightly larger or
differently-positioned sample region before assuming the whole HSV approach is inadequate.
Document whichever result actually holds up against real footage, not the first thing that
compiles.