# Phase 3 — Car make/model (validate first, integrate only if it holds up)

Flagged from the start as lower-confidence and optional. This phase is split into two stages
on purpose: don't build UI/integration around a model that hasn't been proven to work on real
footage first.

## Candidate model — researched, not assumed
**`Jordo23/vehicle-classifier`** (Hugging Face) — EfficientNet-B4, fine-tuned on VMMRdb
(Vehicle Make and Model Recognition dataset — real-world marketplace photos, not studio catalog
images), ~8,949 classes covering make/model/year combinations. **MIT licensed** — cleaner than
the Stanford Cars-based alternatives (which carry per-model, often restrictive licensing).

**What's not yet known and must be tested, not assumed:**
- Real accuracy on footage that isn't the training distribution — no verified top-1/top-5
  number for this specific model going in
- Whether VMMRdb's coverage generalizes to vehicles common in your actual test footage, given
  it's still a primarily US-sourced dataset
- Real inference cost on the free-tier CPU, stacked on top of YOLO26 + color extraction already
  running there

## Stage 3a — Validation only (do this first, build nothing else until it's done)
- [x] Load `Jordo23/vehicle-classifier` via `timm`/Hugging Face Hub, confirm it runs on a
      single cropped vehicle image from your real footage
- [x] Run it against every vehicle detection in your existing sample clip and any uploaded
      test clips — by hand, check whether the predicted make/model is even plausible (you don't
      need ground truth for every prediction, just enough to sanity-check it isn't producing
      nonsense)
- [x] **Measure real added inference time per detection**, and the resulting total processing
      time increase on the deployed Render service specifically — not estimated, measured,
      same as every timing claim in this project so far
- [x] Decide, with real numbers in hand: is this accurate enough and fast enough to be worth
      shipping? Write the decision down, either way, with the numbers that drove it

## Stage 3b — Integration (only if Stage 3a's answer is yes)
- [ ] Add make/model as a field on vehicle `Alert`s, extending the existing schema
- [ ] Add it to the dashboard's filter options, same pattern as color
- [ ] Document real, measured accuracy expectations in the README plainly — a make/model guess
      that's right 6 times out of 10 (or whatever the real number turns out to be) needs to be
      presented as a confidence-qualified estimate, not a fact, same honesty standard as every
      other stated limitation in this project

## What "done" looks like for Stage 3a specifically
A written, numbers-backed answer to "should we build this" — not a working feature yet. If the
answer is no, that's a complete, successful outcome for this stage: a real go/no-go decision
based on actual measurement is exactly the kind of engineering judgment worth having on record,
not a failure to route around.

## Notes
If Stage 3a's accuracy looks weak specifically on vehicle types/markets underrepresented in
VMMRdb, say so plainly rather than either overselling the feature or quietly shelving it
without explanation — the reasoning is worth as much as the result here.

---

# Stage 3a — results (2026-09-29/30)

## Decision: **no-go.** Don't build Stage 3b on this model.

It is **not accurate enough** and **doesn't fit on the free tier**. Either one
alone would be enough to stop.

| | Bar for "enough" | Measured |
|---|---|---|
| Right make **and** model, per vehicle, best frame | most of the time (≥70%) | **25%** overall; **41%** on its best footage (2011 Toronto) |
| Obviously wrong (a sedan called an SUV or a pickup, a van called a sedan) | rare (under 5%) | **29%** |
| Non-US-market vehicles (India, Kenya, Mexico) | usable | **0 of 31** right |
| Deployed processing, 20 s clip | similar to colour (~4%, within run-to-run noise) | **38 s → 153 s (4.0×)** |
| Survives the 512 MB free instance | always | **killed on 6 of 7 runs** |

**Why those bars:** a make/model tag would be a *filter*, like colour ("show me
the Ford Escapes"). A filter that's wrong three times in four returns mostly the
wrong cars and misses most of the right ones. A label that turns a sedan into an
SUV destroys trust faster than no label. Colour was held to about 90% precision,
reporting "mixed"/"unknown" instead of guessing, and it cost ~4% of processing
time. This classifier is nowhere near either number, and no confidence threshold
gets it there (see below).

## 1. It loads and runs

- `vehicle_classifier.pth` loads into `timm.create_model("efficientnet_b4",
  num_classes=8949)`, all keys matched. It loads with `weights_only=True`, and it
  should always be loaded that way: the checkpoint is a third-party pickle.
- The model card is wrong on details worth knowing:
  - It has **33.6 M parameters**, not "~19 M": the 8,949-way head alone is 16 M.
  - The **~1 MB `vehicle_classifier.onnx` is only the graph.** Its external
    weights file isn't in the repo, so it can't run.
  - The card's "~50% top-1" comes with no stated test set.
- **Class list** (read from the checkpoint): 91 makes, heavily US-weighted.
  - Chevrolet has 929 classes and Ford 808. Peugeot, Renault and Citroën have
    one each.
  - There is no Maruti, Tata or Mahindra, no Toyota Avanza or Renault Kwid, and
    one Toyota HiAce class.
  - There are **no 2018 model years and nothing after 2020**.
  - Some labels are duplicated under two spellings: "BMW"/"Bmw",
    "Tesla S"/"Tesla Model S", "Toyota Solara"/"Toyota Camry Solara". The
    probability gets split between them.
- **Preprocessing:** the card's squash-to-380×380 was used. I compared it with
  the checkpoint's own resize-400/centre-crop-380 and with an aspect-preserving
  pad, on the 115 reviewed vehicles whose make I could identify by eye:

  | Preprocessing | Makes right |
  |---|---|
  | Squash | **62%** |
  | Pad | 59% |
  | Centre-crop | 50% |

  So the weak results below aren't a preprocessing artifact.

## 2. Accuracy on real footage

**Method.** Every car, truck, bus and motorcycle detection that the project's
own YOLO26 detector finds was classified (1,973 detections). Clips were checked
at 2 per second, like uploads; the sample clip was checked on every frame.

- Detections were linked into per-vehicle tracks (IoU ≥ 0.3 between
  consecutive checked frames). This linking was for analysis only; the product
  has no tracking.
- For each track, the **largest crop** (its best frame, so this is
  best-case accuracy) was judged by eye on contact sheets: 451 vehicle
  appearances wider than 100 px.
- Verdicts:
  - **C**: make and model right (badge or unmistakable shape; year not judged)
  - **M**: make right only
  - **P**: plausible, can't verify
  - **U**: unverifiable
  - **W**: wrong, same body type
  - **B**: obviously wrong body type or kind of vehicle
- Excluded from the percentages: crops that weren't one vehicle (**X**:
  partial, two cars, blocked by people), motorcycles and buses. None of these
  can get a meaningful label from this class list, and a product would skip
  them.

| Clip | Judged | Right (C) | Make only | Wrong (W) | Obviously wrong (B) | C % |
|---|---|---|---|---|---|---|
| Sample clip (720p, every frame) | 14 | 1 | 0 | 4 | 7 | 7% |
| Toronto 2011, full 1080p source | 100 | 41 | 1 | 17 | 25 | **41%** |
| Mexico City street, close-up daylight | 37 | 9 | 7 | 7 | 8 | 24% |
| New York, 2020s traffic, elevated | 31 | 8 | 1 | 16 | 1 | 26% |
| Nairobi 2025, 4K, from inside traffic | 36 | 6 | 1 | 12 | 13 | 17% |
| Hyderabad 2025, 1080p | 23 | 1 | 1 | 6 | 12 | 4% |
| Cuttack (Odisha) 2013, elevated | 27 | 0 | 0 | 13 | 13 | 0% |
| **All** | **268** | **66** | **11** | **75** | **79** | **25%** |

(P and U make up the rest of each row. All verdicts, each with what the vehicle
actually was, are in `docs/phase3a/verdicts/`.)

**Where it holds up:** 2000s–2010s North American cars, seen side-on at a decent
size. In Toronto it got right:
- a BMW X5, on most frames, at 0.85–0.95 confidence
- a Cadillac SRX (0.92), a Pontiac Grand Am, a Chevrolet Silverado, a Cobalt
  and a Malibu
- 1990s and 2000s Honda Accords and Civics, a VW Rabbit and Jetta, a Mazda3, a
  Mercedes ML and a Toyota Highlander

The 41% is also flattered: those 41 appearances are about 16 distinct vehicles
(the X5 alone is 11 of them).

**Where it visibly doesn't:**
- **The sample clip itself: 1 of 14.** Cars there are 40–140 px wide and
  usually half-hidden by pedestrians.
  - Its most visible car, a dark-blue sedan, was called a Ford Escape, Jeep
    Liberty, Honda CR-V and Dodge Durango (all SUVs) on different frames.
  - A 2011 Toronto taxi was called a **Tesla Model S** (0.55), a car that
    went on sale in 2012.
- **Non-US-market vehicles: 0 of 31 right.** 15 were wrong and 11 obviously
  wrong. Examples:
  - Mahindra jeeps (Cuttack) came out as "Kia Soul", consistently.
  - Toyota HiAce matatus (Nairobi) came out as Tesla Model S or Toyota Tacoma.
  - A Renault Kwid and a Toyota Avanza (Mexico) came out as Jeep Renegade and
    Chevrolet Equinox/Silverado, with the Renault and Toyota badges plainly
    visible.
  - Indian goods trucks came out as "VW Atlas".
  - Most of these vehicles aren't in the class list at all, so the model *can't*
    be right. It always answers with the nearest US label, and never says "I
    don't know".
  - Where the model *is* in the class list (Hyundai Santa Fe, Honda Jazz = Fit,
    Mercedes SUVs), it was still usually wrong. A Mercedes with the star
    visible came out as a Ram 1500.
- **Newer cars:** a 2021+ Chevrolet Trailblazer came out as a Highlander,
  Pathfinder, Tucson or Tundra. Its model years aren't in the dataset.
- **Other detections that reach it:**
  - YOLO labels auto-rickshaws (Hyderabad, Cuttack) as car/truck/bus. They got
    Tesla Model S, FIAT 500L, Kia Soul and McLaren 720S.
  - Motorcycles got Nissan Leaf (0.74) and Tesla Model S (0.70).
  - An integration would have to filter all of these out upstream.
- **It isn't stable:** a vehicle it reads correctly on its best frame gets a
  *different* make/model on 17–29% of its other frames. Every frame is a
  separate alert, so the same car would carry several labels.

**Confidence doesn't rescue it.** Among the judged vehicles:

| Top-1 confidence | Vehicles | Right | Wrong or obviously wrong |
|---|---|---|---|
| any | 268 | 25% | 57% |
| ≥ 0.3 | 40 | 68% | 28% |
| ≥ 0.5 | 14 | 79% | 21% |

- The median top-1 probability is **0.04–0.09** on every clip.
- Only **1–18% of all car/truck detections** reach 0.3.
- Confident nonsense exists: motorcycle → Nissan Leaf 0.74, sedan → Tesla
  Model S 0.55, sedan → MINI Countryman 0.55.

A threshold high enough to be trustworthy labels almost nothing.

## 3. Speed and memory — measured on the deployed free tier

Measured on a **separate throwaway Render free web service** built from the
`phase3a-experiment` branch (never merged; production untouched):
- same Dockerfile and instance type as production
- local-disk results instead of R2
- the classifier runs on an upload only when it's asked for, so the same
  instance was timed with it off and on

It classifies every car/truck alert's crop inside the same pass as YOLO and
colour, so its cost lands in `processing_time_s`.

**Per crop:**

| Where | EfficientNet-B4 per crop (median) |
|---|---|
| Laptop (M2), native, 1 thread | 216 ms (more threads were *slower*: 374 ms at 4, 646 ms at 8) |
| Docker, 1 CPU, 512 MB | 249 ms |
| **Render free tier (deployed)** | **2,093 ms** (p90 2,294, max 2,797) |

For scale: on Render, the whole existing pipeline (decode, YOLO26-N, colour)
costs about **0.95 s per checked frame** (38 s for 40 frames). **Classifying one
vehicle crop costs over twice what the entire existing pipeline spends on a
frame**, and a busy frame has several.

**Whole clip, deployed** (20 s 720p Toronto clip, 40 checked frames, 54
car/truck crops):

| | Processing | Runs |
|---|---|---|
| Classifier off | 36.0 / 38.1 / 40.0 s (Phase 1 production: 38.8 s) | 3 of 3 completed |
| Classifier on | **153.2 s** (112.8 s of it classifying) | **1 of 6 completed; 5 of 6 killed the instance** (and 1 of 1 on the 56 s clip) |

Of the kills:
- **3 of 3 with the first loader.**
- **2 of 3 with a leaner one** (weights memory-mapped and loaded once instead
  of twice).
- Each kill took the instance down (502s, then a restart that wiped every job
  on it, including finished ones).
- The one run that completed had its container memory pinned at the **512 MB**
  ceiling. Without the classifier the peak was **406–433 MB**.
- The cause is consistent with running out of memory. I can't read the
  service's event log from here, so Render's own "out of memory" message is
  still to be checked.

**56 s 1080p clip:** with the classifier off: **163.2 s** (Phase 1 production: 147.2 s). With it on,
the instance was **killed again**, so there is no deployed number for it.
- Locally, in Docker at 1 CPU and 512 MB, the same clip went **21.4 s → 111.8 s
  (5.2×)**: 366 crops, 92 s of classifying.
- At the deployed 2.09 s per crop, those 366 crops would add about
  **12–13 minutes** to the 2.7-minute baseline. That's an estimate from measured
  parts, not a measurement.

**What it would mean for the product:**
- Processing time scales with the number of cars, not the length of the clip.
  The Toronto clip averages about 6.5 car/truck alerts per second of video,
  about 13 s of classifying per second of video on the free tier.
- Phase 1's budget is 2–2.75 s per second of video. The classifier would
  multiply it by roughly 4–6×, and a 60 s upload would take around a quarter of
  an hour.
- It would also crash the single shared instance most of the time, taking
  everyone's queued and running jobs with it.

## What would change the answer (not tried; each is its own project)

- **Classify once per tracked vehicle, not per alert.** This needs tracking
  across frames, which is out of scope today.
- **A different model or dataset:**
  - smaller, and quantised to fit in memory
  - trained on vehicles from the market the cameras are actually in
  - able to say "unknown"
- **Paid compute:** more memory and a real CPU. That would fix speed and memory
  but not accuracy.

None of these fixes accuracy on the footage this project uses except the
second.

## Reproducing

- `docs/phase3a/` has:
  - `run_clip.py`: classify every vehicle detection in a clip
  - `review.py`: tracks and contact sheets
  - `tally.py`: the tables above
  - `prep_fair.py`: the preprocessing comparison
  - `verdicts/`: every by-eye verdict, with what the vehicle actually was
- The scripts expect the weights, the clips and a `runs/` folder next to them.
- Clips:
  - the Toronto source and the sample clip: CC0
  - Nairobi and Hyderabad: Wikimedia Commons, CC BY-SA 4.0
  - Cuttack: Wikimedia Commons, CC BY-SA 3.0
  - Mexico City and New York: Pexels stock clips
  None are committed.
- Timing code: the `phase3a-experiment` branch (`app/services/make_model.py`).
