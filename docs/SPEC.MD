# Perimeter Anomaly Detection — Product Specification

**Status:** MVP complete and deployed (Days 1–5); Phase 1 (upload + class selection, results in R2) built and deployed. Live camera input is Phase 3.
**Owner:** Dan
**Stack:** Python/FastAPI (uv-managed), Pydantic, pytest, Docker, Next.js

---

## 1. What this is
A tool that watches a recorded video clip, detects people and vehicles per frame using a
pretrained model, and raises a timestamped alert with a snapshot whenever a detection falls
inside a defined zone. A human reviews the alerts instead of watching raw footage.

**Explicitly not built this sprint:** live camera support. That's a deliberate, planned Day 6
addition after the 5-day MVP, not an oversight — see section 9.

## 2. The problem it solves
Someone watching a camera feed for hours will miss things — attention doesn't scale. This
automates the "was anything worth looking at" filter: run detection on every frame, check
whether anything landed inside a zone that matters, and only surface those moments.

## 3. Who it's for (in the fiction of the product)
Someone reviewing footage from a fixed camera over a protected area (a yard, a restricted zone,
a perimeter) who wants a list of "here's what happened and when," not hours of empty footage.

## 4. Core design principle
**Detection and zone-checking are pure, independently testable functions.** Given a frame,
what did the model see — that's one function. Given a detection's position and a polygon, is it
inside — that's another, with zero dependency on the model, video files, or anything else. This
is what makes the zone-check logic testable from Day 1 with no video file, no model weights,
and no I/O at all — just coordinates in, boolean out.

---

## 5. Architecture
```
                   ┌─────────────────────┐
                   │   Video clip (file)   │
                   └──────────┬──────────────┘
                              ▼
                   ┌─────────────────────┐
                   │  clip_processor.py    │
                   │  reads frames one       │
                   │  by one                  │
                   └──────────┬──────────────┘
                              ▼
                   ┌─────────────────────┐
                   │   detector.py           │
                   │   YOLO26-N inference     │
                   │   → boxes, classes,       │
                   │     confidence             │
                   └──────────┬──────────────┘
                              ▼
                   ┌─────────────────────┐
                   │   zone_check.py         │
                   │   point-in-polygon,       │
                   │   pure function            │
                   └──────────┬──────────────┘
                              ▼
                   ┌─────────────────────┐
                   │   Alert (if inside)      │
                   │   timestamp + class +      │
                   │   snapshot frame            │
                   └──────────┬──────────────┘
                              ▼
                   ┌─────────────────────┐
                   │   Next.js dashboard      │
                   │   alert list + zone view   │
                   └─────────────────────┘
```

## 6. Components

### 6.1 `detector.py`
Wraps YOLO26-N. Input: a single frame (image array). Output: a list of detections, each with
class, confidence, and bounding box. This is the only component that touches the model —
everything downstream works with plain detection data, not model internals.

**Model choice, researched (see PROJECT.md for full reasoning):** YOLO26-N, Ultralytics,
January 2026 — CPU-optimized, NMS-free, meaningfully faster CPU inference than the prior
generation. Ships under AGPL-3.0; acceptable for this open-source project, stated explicitly
in the README.

### 6.2 `zone_check.py`
Pure function: given a detection's position (e.g. bounding box center or base point) and a
polygon, returns whether it's inside. No model, no video, no I/O — this is the easiest
component in the whole project to get right and to test, so there's no excuse for it not
having tests from the day it's written.

### 6.3 `clip_processor.py`
Orchestrates the pipeline: reads a clip frame by frame (sampling, not necessarily every single
frame — decide a reasonable rate and document why), runs each sampled frame through
`detector.py`, filters to person/vehicle classes, runs surviving detections through
`zone_check.py`, and produces an `Alert` for each one that lands inside the zone.

### 6.4 FastAPI layer
- `POST /clips/{clip_id}/process` (or similar) — runs `clip_processor.py` against a committed
  sample clip (or an uploaded one, if that's in scope for the MVP — keep this simple, a fixed
  sample clip is enough to prove the pipeline)
- `GET /clips/{clip_id}/alerts` — returns the alert list for a processed clip
- `GET /health`

### 6.5 Next.js dashboard
Upload/select a clip, trigger processing, show the resulting alert list with snapshots, and
render the configured zone over a reference frame so it's clear what area is being watched.

---

## 7. Data model

**Zone**
| Field | Type | Notes |
|---|---|---|
| `points` | list of (x, y) | polygon vertices, in frame pixel coordinates |

**Detection**
| Field | Type | Notes |
|---|---|---|
| `class_name` | string | "person", "car", etc. — filtered to relevant classes only |
| `confidence` | float | |
| `bbox` | (x1, y1, x2, y2) | |

**Alert**
| Field | Type | Notes |
|---|---|---|
| `timestamp_s` | float | position in the clip |
| `class_name` | string | |
| `snapshot` | image/bytes or path | the frame the detection came from |

---

## 8. Non-functional requirements
- **Explainability over cleverness** — a zone check that's a plain, readable point-in-polygon
  function beats a more "sophisticated" approach that's harder to verify by hand.
- **Realistic performance expectations, stated plainly.** This will not run in real-time on a
  free-tier CPU host. Document actual processing time for the sample clip in the README rather
  than implying real-time capability it doesn't have.
- **License disclosure** — AGPL-3.0 (YOLO26) stated explicitly in the README, not left for
  someone else to discover.
- **Pure functions tested without I/O** — `zone_check.py` specifically should never need a real
  video file or model weights to test.

## 9. Explicit scope
**In scope:** recorded clip processing, zone-based alerting, a dashboard to review results.
**Out of scope (this sprint):** live camera feeds, multi-camera correlation, cross-frame object
tracking, face/identity recognition.
**Planned, explicitly deferred to Day 6 (post-MVP):** live camera support via a `FrameSource`
abstraction, frame sampling for live feeds, alert debouncing, WebSocket-pushed alerts. See
PROJECT.md's Day 6 section for the concrete plan — deferred on purpose, not forgotten.

## 10. Tech stack
- Backend: Python, FastAPI, managed with `uv` (same pattern as the F1 project:
  `uv init` + `uv add`, `uv.lock` committed)
- Data models: Pydantic throughout
- Tests: `pytest`, starting Day 1 for `zone_check.py` specifically since it needs no fixtures
- Model: YOLO26-N via the `ultralytics` package
- Video I/O: OpenCV (`opencv-python-headless` — no GUI dependencies needed on a server)
- Containerization: Docker, same `uv`-in-Docker pattern as the F1 project
- Frontend: Next.js
- Hosting: Render (backend container) + Vercel (frontend) — same as F1, for the same
  now-verified free-tier reasons

## 11. Success criteria
- Running the pipeline against the committed sample clip produces a correct, hand-verified
  alert list — no false positives/negatives you can't explain
- `zone_check.py` has real unit test coverage, added the day it's written, not deferred
- The dashboard shows the alert list with snapshots and the zone overlay clearly
- Deployed container actually runs inference successfully on Render's free-tier CPU within a
  documented, reasonable time — verified, not assumed
- README states plainly: the AGPL-3.0 license implication, that this isn't benchmarked against
  an academic dataset, and that live camera support is a stated future direction (Day 6), not
  a missing feature nobody thought about

## 12. Current status
**MVP complete and deployed** (as of 2026-09-24, end of Day 5).

- Frontend: https://perimeter-anomaly-detection.vercel.app
- Backend: https://perimeter-backend-ax3a.onrender.com (Docker on Render free tier)

Against the section 11 success criteria:
- Pipeline on the sample clip produces a hand-verified alert list (Day 2:
  entry/exit timestamps checked frame by frame). Platform differences are
  borderline cases only — see README, "Results differ slightly between
  platforms".
- `zone_check.py` has had unit tests since Day 1; the backend suite is 139
  tests, mutation-checked, and also runs inside the Docker image.
- The dashboard shows grouped alerts with snapshots and the zone overlay,
  verified against the deployed URLs.
- Deployed inference measured on Render's free tier: 64–70 s for the sample
  clip, synchronous, well inside Render's documented 100-minute response limit.
- README states the AGPL-3.0 implication, the lack of academic-dataset
  benchmarking, and live camera support as a future direction.

Not built, by design: live camera input (Day 6 plan in PROJECT.md), clip
upload, cross-frame tracking.