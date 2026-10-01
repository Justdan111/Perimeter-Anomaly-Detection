# Architecture

How Perimeter Anomaly Detection works, why it's built the way it is, what was
tried and rejected, and where it falls short. The [README](README.md) covers
setup, deployment and the measured numbers in full; this document explains the
system.

## What this is, and the problem it solves

A fixed camera watches an area: a yard, a driveway, a restricted zone. Nobody
can watch hours of that footage and stay alert, so the moments that matter (a
person or vehicle entering the area) get missed.

This project does the watching:
1. You give it a recorded clip and say what to look for: people, vehicles,
   bicycles, animals, bags.
2. It finds every moment one appears inside the watched zone.
3. It gives you a timestamped list of alerts, each with a snapshot and the
   object's colour.

A person reviews a few minutes of alerts instead of hours of video.

It answers "did something enter the zone, and when?" — never "who was it?".
There is no face recognition and no identification.

## System architecture

```
                     ┌──────────────────────── Next.js dashboard (Vercel) ────────────────────────┐
                     │  upload form · class choice · job progress · alerts grouped by time ·      │
                     │  zone + box overlay on snapshots · class and colour filters                │
                     └──────┬───────────────────────────▲──────────────────────────▲──────────────┘
          POST /uploads     │      GET /jobs/{id}       │ (polled every 1.5 s)     │ <img src=signed URL>
          (video + classes) │      GET /jobs/{id}/alerts│                          │ (expires after 6 h)
                            ▼                           │                          │
┌──────────────────────────────── FastAPI backend (Docker on Render) ──────────────┼──────────────┐
│                                                                                   │             │
│  validate upload (type from first bytes, size, length, resolution, fps)          │              │
│        │  202 + job id, straight away                                             │             │
│        ▼                                                                          │             │
│  job queue (FIFO, ≤ 3 unfinished) ──► ONE worker thread ◄── model lock ──► sample-clip          │
│                                              │                               endpoint (sync)    │
│                                              ▼                                                  │
│                          clip_processor: OpenCV reads the video, samples frames (2/s)           │
│                                              │  each sampled frame                              │
│                                              ▼                                                  │
│                          detector: YOLO26-N ──► list of Detection (class, confidence, box)      │
│                                              ▼                                                  │
│                          class filter ──► zone_check: is the box's bottom-centre in the zone?   │
│                                              ▼                                                  │
│                          colours: HSV on the box's pixels (vehicle colour; person top/bottom)   │
│                                              ▼                                                  │
│                          Alert (timestamp, class, box, anchor, colour) + snapshot JPEG          │
│                                              │                                                  │
└──────────────────────────────────────────────┼──────────────────────────────────────────────────┘
                                               ▼
                       Cloudflare R2 (private bucket, objects deleted after 7 days)
                       jobs/<id>/job.json · result.json · reference.jpg · snapshots/*.jpg
                       — read back for status and alerts; images served as signed links

 Rejected experiment, not in the product:
   camera ──► local agent (1 frame/s) ──► POST /live/frames ──► detector + zone_check
                                          ──► WebSocket push to a /live dashboard page
   It worked on its own, but couldn't share the model and memory with uploads.
   See "What was tried and rejected".
```

**The path of one upload:**
1. **Upload and validate.** The dashboard sends the video and the chosen
   classes. The backend checks it is a real video within the limits.
2. **Queue.** It saves the video to a temporary folder, writes a job record to
   R2, and answers at once with a job id. Processing a minute of video takes
   minutes, far longer than an HTTP request should wait.
3. **Process.** A single background worker runs the clip. For each sampled
   frame: detect objects, keep the wanted classes, test each object's position
   against the zone, read its colour, and save a downscaled snapshot of any
   frame that alerted.
4. **Store.** Snapshots and the alert list go to R2, the job record is marked
   complete, and the temporary video is deleted.
5. **Review.** The dashboard has been polling the job's status. When the job
   is complete it fetches the alerts. Each snapshot's URL is a signed,
   expiring link straight to the private bucket.

A committed 5-second sample clip, with a hand-drawn zone, takes a shorter path.
It is processed synchronously on request, through the same `clip_processor`,
checking every frame. It is the demo and the regression check.

## Components

**`detector.py`: the model, behind one function.**
- Wraps YOLO26-N: a frame goes in, a list of plain `Detection` objects (class,
  confidence, box) comes out. Nothing downstream ever sees an Ultralytics
  object, so swapping the model touches this one file.
- Loads the weights once, at startup, so no request pays for it.
- Detections below 0.35 confidence are dropped.
- Sets PyTorch's thread count from the container's CPU limit, not the host's
  core count. Without that, a container limited to one CPU ran seven threads
  fighting over it, about 13× slower per frame.

**`zone_check.py`: is this point inside this polygon?**
- A pure function: coordinates in, true or false out. No model, no video, no
  imports from the rest of the project.
- It uses ray casting, with an explicit on-the-edge check first. Ray casting
  handles concave zones, like a yard minus a building corner.
- It has a half-open rule for vertices: a ray that passes exactly through a
  vertex would otherwise be counted twice.
- Every alert depends on it, so it is tested against hand-drawn shapes,
  including the configurations that break naive versions.

**`clip_processor.py`: from a video to alerts.**
- A thin OpenCV loop around small pure functions that hold every decision:
  which classes count, which point of a box is tested, which frames are
  sampled, what time a frame is, whether the video matches the zone's
  resolution.
- **The tested point is the bottom-centre of the box**: where feet or tyres
  meet the ground the zone is drawn on. The box centre of a person on the
  pavement *behind* a zone can project inside it.
- **Sampling is a rate in video time** ("twice a second"), not a frame stride,
  so it means the same at 24 or 60 fps. Uploads are checked twice a second. At
  walking pace a person moves about 0.7 m between checks, so nobody crosses a
  zone unseen.
- Frames are independent: with no state between frames, sampling can only
  remove alerts, never change one.
- If a video stops decoding before the length its header declares, the run
  fails loudly. A truncated upload doesn't quietly return results for half the
  clip.

**The job and worker system (`jobs.py`).**
- **One worker thread and a first-in, first-out queue.** The free host has less
  than one CPU and there is one model instance, which isn't safe to share
  between threads. Two jobs at once would each run at half speed; one at a
  time finishes the first sooner.
- **One lock shared by the worker and the sample-clip endpoint**, so the model
  only ever runs one thing.
- **At most three unfinished jobs**; a fourth upload is told the server is
  busy.
- **R2 holds the job record and memory is only a cache**, because the free
  host wipes its disk on every restart and idle spin-down. A job that was
  mid-run when the server restarted can't be resumed (its video is gone), so
  it is reported, and saved, as failed with that reason.

**The R2 integration (`storage.py`).**
- **Results live in Cloudflare R2:** the job record, the alerts, the first
  frame (for drawing the zone) and the snapshots. R2 is reached through
  `boto3`, R2 being S3-compatible.
- **The bucket is private.** The dashboard gets each image as a signed link
  that expires after 6 hours.
- **A lifecycle rule on the bucket deletes everything after 7 days.**
- **Snapshots upload in parallel.** One at a time, the per-request latency
  made saving take almost as long as processing.
- **Without R2 credentials, the same interface writes to a local folder.** That
  is for development; tests run against a fake S3.

**The Next.js dashboard (`frontend/`).**
- Upload form with class checkboxes.
- Live progress while a job runs.
- The results view:
  - **groups alerts into time slices**, since a car parked in the zone produces
    an alert on every checked frame;
  - draws the zone and each detection's box over the snapshot;
  - **flags alerts whose box is cut off by the bottom of the frame** (see
    limitations);
  - filters by class and colour ("red cars", "people in a blue top").
- The job id is in the URL, so a reload or a shared link reopens the results.
- The grouping, filtering and edge-detection logic are pure functions with
  their own tests.

## Tools, and why each was chosen

The constraints were: run entirely on free tiers, on a CPU, with nothing paid
beyond what the job truly needs, and stay checkable at every step.

- **`uv`.** Fast, reproducible Python dependency management with a committed
  lockfile. The Docker build uses `uv sync --frozen`, so it fails instead of
  silently re-resolving versions.
  - It pins the CPU-only PyTorch build on Linux; the GPU build would add
    gigabytes for nothing.
  - It keeps the GUI build of OpenCV out of the server image.
- **FastAPI.** Typed request handling and validation with little ceremony, and
  automatic API docs. Heavy work (video probing, processing) runs in worker
  threads, so the server stays responsive while a clip is processed.
- **Pydantic.** Every piece of data that crosses a boundary is a validated
  model: detections, zones, alerts, results.
  - A zone stores the frame size it was drawn against, so a zone drawn on 1080p
    applied to a 720p video fails loudly instead of being silently 1.5× off.
  - New fields are optional, so older stored results still load.
- **pytest.** About 360 tests, none of which need model weights, a bucket or
  the network. A fake detector and a fake S3 make the expected answers exact.
  - Tests that guard against a specific mistake were checked by planting that
    mistake and confirming the test fails.
  - Deprecation warnings fail the suite, so they get fixed rather than pile up.
- **YOLO26-N (Ultralytics).** A current-generation detector, small and built
  for CPU and edge inference. That matters on a host with no GPU and a
  fraction of a CPU. Pretrained on the 80 COCO classes, which already include
  everything this project looks for.
  - **License: the `ultralytics` package and the YOLO26 weights are
    AGPL-3.0.** That's fine for an open-source public project.
  - A closed-source commercial product or service would have to publish its
    full source under the same license, or buy an Ultralytics Enterprise
    License, or switch to a permissively licensed detector.
- **Docker.** The backend ships as one image, identical locally and in
  production.
  - It runs as a **non-root user**: one fewer privilege to misuse, and the only
    way the tests for refused file writes can run inside it.
  - The **model weights are downloaded at build time and baked into the
    image**, so a cold start never depends on reaching the internet.
  - A test stage runs the whole suite inside the image.
- **Render (backend).** Free container hosting from a Dockerfile.
  - Its limits shaped the design: 512 MB of memory, a small CPU share, and a
    disk wiped on restart.
  - The disk is why results go to R2, and the CPU share is why uploads are
    capped at 60 s and sampled at 2 frames per second.
- **Vercel (frontend).** Free, zero-configuration Next.js hosting.
- **Cloudflare R2 (storage).** S3-compatible object storage with 10 GB free, no
  time limit and no download (egress) fees. It is the one thing the free
  backend host can't provide: storage that outlives a restart.

## What was tried and rejected

Two features were built far enough to measure. Both were rejected on the
evidence, not on assumptions.

**Car make and model.**
- **What was tried:** a pretrained classifier (EfficientNet-B4 trained on the
  US-centric VMMRdb dataset, MIT-licensed) on every vehicle detection across
  seven real clips, including footage from India, Kenya and Mexico.
- **What happened:** judged by eye, it named the make and model correctly for
  **25%** of vehicles, against a 70% bar for something people would filter by.
  - 29% of answers were obviously wrong, such as a sedan called an SUV.
  - It got none of 31 vehicles from outside the US market right.
  - On the deployed host it took **2 s per vehicle**, turned a 38 s job into
    153 s, and pushed memory past the limit: the instance was killed on 6 of 7
    runs.
- **Verdict:** not accurate enough and not affordable. Rejected.

**Live camera feed.**
- **What was tried:** a small local agent next to a webcam sent one frame a
  second to a per-frame endpoint. The endpoint ran the same detector and zone
  check, and a WebSocket pushed the alerts to a live dashboard page. It ran
  for 13.5 minutes against the deployed host, with uploads sent partway
  through.
- **The mechanism worked:** no crashes, and flat memory with the live feed
  alone.
- **It couldn't share the host with uploads.** There is one model and one
  lock, so every upload blocked the live feed for the whole upload: the camera
  went blind for up to 4 minutes. With both running, memory sat at the 512 MB
  limit, and the server stopped answering for about 80 s.
- Giving live frames their own worker would mean a second copy of the model,
  which doesn't fit.
- **Verdict:** a resource-contention problem specific to this single free
  instance, not a flaw in the live-feed design. Rejected here. A separate
  instance, a paid host, or running fully on a local machine would change the
  answer.

## Known limitations

- **No tracking across frames.** Each checked frame is judged on its own, so a
  car parked in the zone raises an alert on every checked frame. The dashboard
  groups alerts by time to keep that reviewable, but there is no "this is the
  same car as before".
- **Colours are estimates, and lighting changes them.** They come from simple
  colour thresholds, not a model.
  - White cars in shade read blue, white cars under street lights read gray,
    and a white truck at sunset read pink.
  - Infrared night footage has no colour at all, so every colour is reported as
    unknown.
  - When no colour clearly dominates, the answer is "mixed" or "unknown", and
    those never match a colour filter.
  - On labelled real footage, a named colour was right about 90% of the time.
- **Objects cut off by the bottom of the frame.** The tested point is the
  bottom of the box, and the detector stops a box at the frame edge. A person
  walking out of the bottom of the shot is placed at the edge, not where their
  feet really are. That matters only for a zone reaching the bottom edge; the
  dashboard flags affected alerts and warns about such zones.
- **AGPL-3.0.** The detector's license means this can't become part of a
  closed-source commercial product or hosted service without publishing the
  whole source, or buying an Ultralytics Enterprise License, or replacing the
  detector.
- **Not real-time, and slow on the free tier.**
  - About 70 s for the 5 s sample clip at every frame.
  - About 2–2.75 s of processing per second of uploaded video.
  - One upload processed at a time.
- **No accuracy benchmark.** The detector wasn't evaluated against an academic
  surveillance dataset; test footage is royalty-free street video. The project
  makes no detection-accuracy claims. The colour accuracy above was measured
  on hand-labelled real footage.

## Running it locally

```sh
cd backend && uv sync && uv run uvicorn app.main:app --port 8000   # API on :8000
cd frontend && npm install && npm run dev                           # dashboard on :3000
```

- The first run needs the model weights fetched once.
- Without R2 credentials, results are kept in a local folder.
- The [README](README.md) has the exact commands, the Docker setup, the
  environment variables for R2 and CORS, the tests, and the full measured
  timings.
