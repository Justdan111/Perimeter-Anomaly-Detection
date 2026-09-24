# Perimeter Anomaly Detection

Detects people and vehicles in a recorded video clip and raises a timestamped
alert, with a snapshot, whenever one is inside a configured zone — so a person
reviews the moments that matter instead of hours of footage.

**Live demo:** https://perimeter-anomaly-detection.vercel.app — press
*Process clip* and expect to wait about **70 seconds** (see
[Processing time](#processing-time--measured-not-assumed)).
Backend: https://perimeter-backend-ax3a.onrender.com/health

![Demo: processing the sample clip on the deployed site, then reviewing alerts](docs/demo/deployed-demo.gif)

*The demo is a sequence of screenshots taken during one real run on the
deployed site (2026-09-24), stitched into a GIF — not a screen recording, so
the ~70 s wait is compressed; the live timer in the frames shows the real
elapsed time.*

- **Backend** (`backend/`): FastAPI, YOLO26-N (Ultralytics) for detection,
  OpenCV for video, managed with `uv`, deployed as a Docker image on Render.
- **Dashboard** (`frontend/`): Next.js on Vercel — process the clip, review
  alerts grouped by time, see the zone drawn over the frame.

Scope, design and the day-by-day plan: [`docs/PROJECT.md`](docs/PROJECT.md),
[`docs/SPEC.MD`](docs/SPEC.MD).

## What this project deliberately does not do

- **No live camera input.** It processes one recorded clip. Live camera
  support is a planned next step — the design is written up in
  [`docs/PROJECT.md`](docs/PROJECT.md) ("Day 6") — but it is **not built**.
- **No evaluation against an academic dataset.** The test footage is a
  royalty-free (CC0) street clip, chosen to avoid the registration friction of
  datasets like i-LIDS or VIRAT. That is a deliberate scope decision, not an
  oversight — and it means this project makes **no accuracy claims**.
- **No clip upload.** The MVP works against the one committed sample clip.
- **Only people and vehicles.** The model can see 80 object types; alerts are
  deliberately limited to `person`, `car`, `truck`, `bus` and `motorcycle`
  (everything else — traffic lights, handbags — is dropped before the zone
  check).
- **No tracking across frames**, and **no identification** of who someone is:
  it only ever answers "something entered the zone".

## License — AGPL-3.0, and what it means

Detection uses **Ultralytics YOLO26**, and both the `ultralytics` package and
the YOLO26 weights are licensed under **AGPL-3.0**. For this project that is
fine: it is open source and public.

It would **not** be fine for a closed-source commercial product. AGPL-3.0
requires that anyone who uses the software over a network can get the
complete source of the whole application, under the same license. A company
wanting to ship this in a proprietary product or service would need an
**Ultralytics Enterprise License** instead (or a detection model under a
permissive license).

The sample clip is CC0 (public domain) — provenance in
[`backend/app/data/sample_clip/SOURCE.md`](backend/app/data/sample_clip/SOURCE.md).

## Processing time — measured, not assumed

This is **not real-time**. Detection runs on the CPU, one frame at a time.
All numbers are for the committed sample clip: 5 s of video, 120 frames at
1280×720, every frame processed.

### Deployed (Render free tier) — measured 2026-09-24

| | |
|---|---|
| Server-side processing, 8 runs | **64.2 – 70.2 s** |
| What a user waits (button press → results shown), 4 runs | 67.2 – 71.1 s |
| Render's documented maximum HTTP response time | 100 minutes ([source](https://render.com/docs/render-vs-vercel-comparison)) |

About 70 s is far inside Render's documented limit, so the synchronous
`POST /clips/{id}/process` design holds up for this clip — but it is a real
wait, and it scales linearly with clip length: a clip a few minutes long would
take tens of minutes, at which point a process-then-poll design would be
needed. The requests pass through Cloudflare in front of Render; the longest
request actually exercised is ~71 s, so the 100-minute figure is Render's
documentation, not something tested here.

Not measured: cold start after the free instance spins down from inactivity,
and memory use on Render (locally, the container peaks at ~434 MiB while
processing; the free tier allows 512 MB).

### Local

| Where | Processing |
|---|---|
| Native, laptop (Apple M2, 8 cores), server warm | **3.2 – 3.3 s** (range across all runs: 3.2 – 4.7 s) |
| Docker image on the same laptop, no CPU limit | ~10 s |
| Docker image, limited to 1 CPU | ~11 s |
| Docker image, limited to 0.5 CPU | ~28 s |

The container is slower than native for identifiable reasons: macOS decodes
the video in hardware (0.14 s vs 1.4 s in the container) and its PyTorch build
uses different math libraries (inference 34 vs 65 ms/frame).

**A CPU-limit bug found and fixed on the way (Day 5).** Ultralytics sets
PyTorch's thread count from the machine's core count — but inside a container
that is the *host's* core count, not the container's CPU allowance. Limited to
1 CPU, the container ran 7 inference threads fighting over it: **173 s** for
the clip. The detector now reads the container's CPU limit and uses that many
threads (and idle threads sleep instead of spinning): **10.7 s** under the same
limit. `/health` reports the thread count in use (`inference_threads`).

### Results differ slightly between platforms

Same code, clip and weights; the PyTorch builds differ (macOS vs Linux CPU-only), and so do their floating-point results:

| Where | Alerts |
|---|---|
| Native macOS | 202 (112 person, 90 car) |
| Deployed on Render (Linux, CPU-only PyTorch) | 198 (111 person, 87 car) |

195 alerts match (same frame and class, boxes within a few pixels); every difference is a borderline case — a
detection within a hair of the 0.35 confidence cut-off, a person standing on
the zone boundary, or which of two duplicate boxes on one car survives. The
alert list is reproducible to within those cases, not bit-for-bit across
platforms.

## Running locally

```sh
# Backend — http://localhost:8000
cd backend
uv sync
uv run python -c "import os; os.environ['YOLO_CONFIG_DIR']='models'; from ultralytics import YOLO; YOLO('models/yolo26n.pt')"  # fetch weights once
uv run uvicorn app.main:app --port 8000

# Dashboard — http://localhost:3000
cd frontend
npm install
npm run dev
```

Or run the backend exactly as deployed:

```sh
cd backend
docker build -t perimeter-backend .          # weights are baked into the image
docker run --rm -p 8000:8000 perimeter-backend
```

The dashboard calls the API at `NEXT_PUBLIC_API_URL` (default
`http://localhost:8000`; see `frontend/.env.example`). It is inlined at
**build** time, so a deployed frontend must have it set before building. The
API only accepts browser requests from the origins in `PERIMETER_CORS_ORIGINS`
(default `http://localhost:3000`).

If the model weights are missing or corrupt, the backend still starts:
`GET /health` reports `"model_loaded": false` with the reason, and processing
requests return `503` with the same message.

## Deployment

- **Backend — Render**, Docker runtime, Root Directory `backend`, free
  instance, health check `/health`, env `PERIMETER_CORS_ORIGINS` set to the
  Vercel origin. The image runs as a non-root user and has the YOLO26-N
  weights baked in at build time, so a cold start never downloads them.
- **Frontend — Vercel**, Root Directory `frontend`, env `NEXT_PUBLIC_API_URL`
  set to the Render URL before the build.
- Results are kept in memory: a restart or redeploy (or the free instance
  spinning down) forgets the last run, and the dashboard shows "not processed
  yet" until someone presses the button again.

## Tests

```sh
cd backend && uv run pytest        # 139 tests; no model weights needed
cd frontend && npm test            # alert grouping logic

# Inside the Docker image, as its non-root user:
cd backend && docker build --target test -t perimeter-test . && docker run --rm perimeter-test
```

Deprecation warnings fail the backend suite on purpose, so new ones get fixed
instead of piling up in the output. Two tests simulate a disk refusing writes
using directory permissions; they skip when run as root, which is one reason
the image runs as a non-root user (inside it, all tests run).

## Known limitations

- **No tracking across frames.** Every alert is one detection in one frame, so
  a car waiting in the zone alerts on every frame it is visible in. The
  dashboard groups alerts by time to keep this reviewable.
- **Objects cut off by the bottom of the frame.** Position is measured at the
  bottom-centre of each box; for a box cut off by the frame edge, that point
  sits on the edge rather than at the object's real feet. The dashboard flags
  affected alerts and warns when a zone reaches the bottom edge.
- **~70 s per run on the free tier**, synchronous — fine for this clip, not
  for long footage (see above).
- **One committed sample clip**; not benchmarked against an academic dataset.
