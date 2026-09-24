# Perimeter Anomaly Detection

Detects people and vehicles in a recorded video clip and raises a timestamped
alert, with a snapshot, whenever one is inside a configured zone — so a person
reviews the moments that matter instead of hours of footage.

- **Backend** (`backend/`): FastAPI, YOLO26-N (Ultralytics) for detection,
  OpenCV for video, managed with `uv`.
- **Dashboard** (`frontend/`): Next.js — process the clip, review alerts
  grouped by time, see the zone drawn over the frame.

Scope, design and the day-by-day plan: [`docs/PROJECT.md`](docs/PROJECT.md),
[`docs/SPEC.MD`](docs/SPEC.MD).

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

The dashboard calls the API at `NEXT_PUBLIC_API_URL` (default
`http://localhost:8000`; see `frontend/.env.example`). The API only accepts
browser requests from the origins in `PERIMETER_CORS_ORIGINS` (default
`http://localhost:3000`).

If the model weights are missing or corrupt, the backend still starts:
`GET /health` reports `"model_loaded": false` with the reason, and processing
requests return `503` with the same message.

## Tests

```sh
cd backend && uv run pytest        # 114 tests; no model weights needed
cd frontend && npm test            # alert grouping logic
```

Deprecation warnings fail the backend suite on purpose, so new ones get fixed
instead of piling up in the output.

## Processing time — measured, not assumed

This is **not real-time**. Detection runs on the CPU, one frame at a time.

Measured on a laptop (Apple M2, 8 cores) with the committed sample clip
(5 s, 120 frames at 1280×720, every frame processed):

| | Time |
|---|---|
| Processing the clip, warm server | **3.2–3.3 s** |
| First processing after a fresh start | 3.9–4.1 s |
| Range seen across all runs so far | 3.2–4.7 s |
| Server startup (import + model load) | 1.1–1.7 s |

That is roughly 0.7–0.9 s of processing per second of video on this machine.
Processing time grows linearly with the number of frames processed; the
`sample_fps` query parameter on `POST /clips/{id}/process` trades timestamp
precision for speed (e.g. `sample_fps=2` processes 10 frames instead of 120).

### Not yet measured — deliberately left for Day 5

- **Render free-tier CPU performance.** A free-tier instance has a small
  fraction of a laptop's CPU, so processing will be slower — how much slower
  will be measured on the deployed container, not guessed here.
- **Request-timeout risk.** `POST /clips/{id}/process` is a single blocking
  request that returns when processing finishes. If free-tier processing
  takes long enough to hit the host's or the browser's request timeout, the
  endpoint will need to change (e.g. start processing and poll for the
  result). Whether that's necessary will be decided from the Day 5
  measurement.

## Known limitations

- **No tracking across frames.** Every alert is one detection in one frame, so
  a car waiting in the zone alerts on every frame it is visible in. The
  dashboard groups alerts by time to keep this reviewable.
- **Objects cut off by the bottom of the frame.** Position is measured at the
  bottom-centre of each box; for a box cut off by the frame edge, that point
  sits on the edge rather than at the object's real feet. The dashboard flags
  affected alerts and warns when a zone reaches the bottom edge.
- **One committed sample clip** (CC0, see
  `backend/app/data/sample_clip/SOURCE.md`); not benchmarked against an
  academic surveillance dataset.
