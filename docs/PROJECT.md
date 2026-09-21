# Perimeter Anomaly Detection

## The real problem
A camera watches a defined area. Someone (or some vehicle) enters who shouldn't be there.
Nobody can watch a feed 24/7 without missing things — this flags entries into a defined zone
automatically, with a timestamped snapshot, so a human reviews only the moments that matter
instead of hours of empty footage.

## Scope (MVP — do not exceed)
- Process a recorded video clip (not a live camera feed — that's a real complexity jump this
  sprint doesn't need)
- Detect people and vehicles per frame using a pretrained model (no training your own —
  productionizing an existing model is the actual skill this project tests)
- A configurable zone (polygon) — entries into the zone generate an alert; detections outside
  it don't
- Each alert: timestamp, detected class, a snapshot frame
- A dashboard to review alerts for a processed clip

## What's explicitly OUT of scope
- Live camera integration
- Multi-camera correlation
- Object tracking across frames (each frame's detections are independent — no "this is the
  same person as 3 frames ago" logic; that's a real feature, just not this sprint's)
- Face recognition / identification of *who* someone is — this project only ever answers
  "something entered the zone," never "who"

## Model choice — researched, not assumed
**YOLO26-N** (Ultralytics, Jan 2026) — current generation, optimized specifically for CPU/edge
inference, which matters since this runs on a free-tier host with no GPU. NMS-free design,
meaningfully faster CPU inference than the previous generation (YOLO11n).

**License note, stated explicitly rather than glossed over:** Ultralytics YOLO ships under
AGPL-3.0. That's fine for this project (open-source, public GitHub repo) but would require an
Ultralytics Enterprise License for a closed-source commercial product — say this plainly in
the README, it's a real thing a technical interviewer might actually ask about.

## Test data
Not using an academic surveillance dataset (i-LIDS, VIRAT) — those require registration and
add friction this sprint doesn't need. Use either your own short recorded clips, or clearly
royalty-free stock footage (Pexels/Pixabay have real outdoor people/vehicle clips). State this
plainly in the README: this wasn't benchmarked against an academic dataset, that's a stated
limitation, not something to imply otherwise.

## Tech stack
Python/FastAPI backend (managed with `uv`, same as the F1 project), Pydantic for all data
models, `pytest` for automated tests, Docker for the deployable artifact, Next.js for the
dashboard. Same toolchain across projects on purpose — the point of the sprint is depth on a
consistent stack, not a new toolchain every project.

## Architecture
```
backend/               FastAPI service
  app/
    main.py             upload/process endpoint + alert retrieval
    models/schemas.py    Detection, Alert, Zone models
    services/
      detector.py         wraps YOLO26-N inference on a single frame
      zone_check.py        point-in-polygon check (pure function, easy to unit test)
      clip_processor.py    runs a clip frame-by-frame through detector + zone_check,
                            produces alerts
    data/
      sample_clip/          a short test clip + zone config committed to the repo
  tests/
    test_zone_check.py     unit tests for the polygon logic — this one has no external
                            dependencies (no model, no video), so there's no excuse for it
                            not being tested from Day 1 onward, not deferred to Day 4
  Dockerfile
```

## Day-by-day (this project's slice)
- **Day 1:** `uv init` + `uv add`, FastAPI skeleton, load YOLO26-N, confirm inference works on
  a single test frame, zone config as data (not hardcoded), first `pytest` tests for the zone
  polygon logic (pure function, no reason to wait on this)
- **Day 2:** `clip_processor.py` — run a full clip frame-by-frame, filter to person/vehicle
  classes, point-in-polygon check, produce a correct, timestamped alert list — verify by hand
  against the clip, and add tests for whatever of this logic is pure enough to test without a
  real video file
- **Day 3:** Next.js dashboard — upload/select a clip, alert list with snapshots, zone drawn
  over a reference frame
- **Day 4:** hardening — zero-detection clips, missing model weights, realistic
  processing-time expectations documented (this will not be real-time on free-tier CPU — say
  so, don't imply otherwise), fill out the `pytest` suite for anything still untested
- **Day 5:** Dockerize (same `uv`-in-Docker pattern as the F1 project — install `uv`,
  `uv sync --frozen` against the lockfile) and deploy the container to Render — confirm the
  free-tier CPU can actually run inference in reasonable time, this is a real risk worth
  checking early, not assuming — plus README with the license note and a demo clip

## Day 6 — future improvement (explicitly post-MVP, not part of the 5-day scope)
Support a live camera feed alongside recorded clips, instead of clips only. This is the same
shape of change as the F1 project's live/replay split — reuse that pattern deliberately:

- Introduce a `FrameSource` interface (`ClipFrameSource` wraps the existing file-based logic;
  `LiveFrameSource` reads from an RTSP URL or webcam via the same OpenCV `VideoCapture` API)
- Sample frames rather than processing every one (1-2 fps is plenty for perimeter detection,
  and matters a lot more here than in F1 — CPU inference on every frame of a live 30fps feed
  isn't viable on free-tier hosting)
- Alert debouncing — a clip runs once and stops, so re-alerting was never an issue; a live feed
  needs "alert once per intrusion, not once per frame it's still present" state
- Push alerts live via WebSocket instead of returning a finished list after processing — same
  pattern as the F1 dashboard, different payload
- **Known architectural fork worth flagging now:** a real camera is usually on a private local
  network, not reachable from a public host like Render. A cloud-hosted version of this needs
  either a local forwarding agent or running on-site — that's a genuine fork, not a pure code
  change, and shouldn't be underestimated when this day actually comes up

Don't build any of this now. It's here so Day 6 has a real plan instead of starting from
scratch, and so `clip_processor.py`'s design isn't accidentally hostile to this later split —
but no `FrameSource` abstraction needed on Day 1, unlike `TickSource` in the F1 project, since
live support here is a deliberate future addition, not this project's primary aim from the
start.