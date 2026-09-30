# Phase 4 — Live camera feed (validate first, same discipline as Phase 3)

The original "Day 6" sketch from `PROJECT.md` named the right components but was written
before Phase 1-3 existed. Two things learned since then change how this needs to be approached,
not just what it needs to contain:

1. **Phase 3 already showed this pipeline is close to the free tier's resource ceiling** — a
   second model plus continuous processing got OOM-killed 6 of 7 runs. A live feed adds
   continuous, indefinite processing load, not a one-shot job that finishes — that's a
   different and likely harder resource problem, not a smaller version of the same one.
2. **The current architecture is one background worker processing one job at a time**, sharing
   a model lock with the sample-clip endpoint. A "job" today starts and finishes. A live stream
   doesn't finish — it would occupy that worker indefinitely, blocking every upload while it
   runs. This is a real architectural conflict, not a detail to patch later.

Given two independent real risks stacked here (resource ceiling under continuous load, and
worker-model conflict) — same as Phase 3, this is validate-first, not build-first.

## The architectural fork, stated plainly
A real camera (RTSP or a webcam) usually isn't reachable from a public host like Render —
it's on a private local network. The natural fix, and the one that reuses everything already
built: a **local agent** — a small program running on the same network as the camera — captures
and samples frames, then forwards them to a new ingestion endpoint on the existing backend. The
detector, zone-check, and alerting pipeline stay exactly where they are; only frame capture
moves to the edge. This preserves the whole existing investment (R2, Render, the dashboard)
instead of forking into a separate on-site deployment model.

## Stage 4a — Prove the core mechanism, at small scale, before building anything durable
- [x] Local agent: a simple script using OpenCV's `VideoCapture` against a webcam (not an RTSP
      camera yet — a webcam is the simplest possible test case and proves the mechanism without
      needing real camera hardware/network setup)
- [x] Sample at 1-2 fps, not every frame — pick one, document why
- [x] New lightweight ingestion endpoint on the existing backend: accepts a single frame, runs
      it through the existing `detector.py` + `zone_check.py` (no clip_processor involved — a
      live frame isn't a clip)
- [x] Push resulting alerts live via WebSocket to the dashboard, instead of the batch
      list-after-processing pattern the rest of the project uses
- [x] **Run this for a sustained period (at least 10-15 minutes continuous) against the
      deployed Render service specifically, and watch memory and worker behavior the whole
      time** — this is the actual test. A single successful frame proves nothing; Phase 3's
      failure only showed up under sustained load, not on the first run
- [x] Confirm whether a live stream running blocks upload processing (per the shared-worker
      concern above) — test this explicitly by trying to upload a clip while the live stream
      is active, don't assume the answer

## Stage 4a's real question
Does continuous, sustained frame ingestion survive on the free tier without getting killed, and
does it conflict with the existing upload/job system? Write the answer down with real numbers
(memory over time, whether anything got killed, whether uploads still worked), the same way
Phase 3's answer was a numbers-backed no-go, not a guess.

## Stage 4b — Only if Stage 4a says yes
- [ ] Solve the worker conflict properly — likely a separate process/worker for live ingestion
      so it doesn't block uploads, not a shared queue
- [ ] Alert debouncing — a live feed doesn't stop like a clip does, so "alert once per
      intrusion" needs real state (cooldown per zone, or entry/exit tracking), not the
      per-frame-independent model the rest of this project deliberately uses
- [ ] Move from a webcam test to a real RTSP camera, if available
- [ ] Harden the local agent: reconnect logic if the camera drops, clear failure states if the
      backend is unreachable

## What "done" looks like for Stage 4a
A written go/no-go decision, backed by a real sustained test against the deployed service, not
a guess extrapolated from Phase 3's numbers. Either answer is a complete, valuable outcome for
this stage.

## Notes
If Stage 4a's answer is no — likely, given what Phase 3 already showed about this free tier's
ceiling — that's a legitimate place to stop and document rather than force. Live camera support
running entirely on a local machine (no cloud component at all) is a real fallback worth naming
in that case, even if it's a bigger step than this project's scope currently goes.

---

# Stage 4a — results (2026-09-30)

## Decision: **no-go for Stage 4b on the current free-tier architecture.**

The mechanism works: a webcam agent, a per-frame endpoint and a WebSocket
dashboard, running for 13.5 minutes against the deployed service with no crash
and flat memory. But **live frames and uploads can't share this instance**:
- Every upload made the camera go **blind for its whole processing time**, and
  up to **4 minutes** in total (see the gap table below).
- Together they pushed memory to the **512 MB ceiling**.
- During the heavier upload, the whole server **stopped answering for about
  80 s**.

Stage 4b's planned fix, a separate worker, needs a second copy of the model,
which this instance has no memory for (details below).

| Question | Answer | Measured |
|---|---|---|
| Does sustained live ingestion survive on the free tier? | **Yes** | 13.5 min; **0 restarts** in 154 polls; memory flat at **452–454 MB** with only the live feed running |
| Does it conflict with uploads? | **Yes, badly** | An upload holds the model for its whole run. Live frames got `503` for all of it: **40 of 181 frames dropped**, longest blind gap **245 s** |
| Headroom with both running | **None** | cgroup memory **510–512 MB** (peak 513 MiB) against the 512 MB limit |
| Server responsive throughout? | **No** | **~80 s with no response on any endpoint** during the 56 s upload |
| Did uploads still work? | **Yes** | Both completed; processing time unchanged (see below) |

## How it was tested

- **Code:** on the `phase4a-live` branch. It stays pushed and is never merged,
  like `phase3a-experiment`. Deployed to the same throwaway Render free service
  used in Phase 3 (Docker, 512 MB, local-disk results), not production.
  - `agent/live_agent.py` captures the webcam with OpenCV. A reader thread
    keeps only the newest frame. It sends one JPEG at a time (1280 px wide,
    quality 80) at 1 fps and polls `/live/stats` every 15 s.
  - `POST /live/frames` decodes the frame and runs `detector.py` plus the zone
    check directly, with no `clip_processor`. The whole frame is the zone.
    - It uses the **same model lock** as uploads and the sample clip: the model
      isn't thread-safe, and a second copy is the extra memory Phase 3 showed
      the free tier can't take.
    - A frame waits at most 1 s for the model, then is dropped with `503`. A
      late frame is worthless, and a queue of them is a memory leak.
  - `WS /live/ws` pushes each frame's detections, with a 480 px preview, to the
    `/live` dashboard page. The page stayed connected for the whole run.
  - Nothing is stored.
- **Tests:** 18 backend tests plus 8 frontend tests. Mutation-checked: 11
  planted bugs, all caught (one survivor was found and closed by a stronger
  test).
- **Run:** your real webcam, from your Terminal, 13 min 30 s. Memory was
  recorded two ways:
  - by the agent (`/live/stats` every 15 s);
  - independently, by a 5 s poller on a separate machine connection.
- **Uploads during the run** (the Phase 1 upload flow):
  - the 20 s 720p clip at ~5 min;
  - the 56 s 1080p clip at ~7.5 min, the largest upload the site accepts.
- Raw logs: `docs/phase4a/` (`agent_frames.csv`, `agent_stats.csv`,
  `server_stats_5s.jsonl`).

**Why 1 fps.** On this host detection costs about 0.4 s per frame, measured
below. At 2 fps the single CPU share would be about 80% busy with the camera
alone, with no headroom for anything else. A person walking at about 1.4 m/s
is still seen several times while crossing a room-sized view at 1 fps.

## Memory over time (container memory, `memory.current`, deployed)

| Period | Samples | MB (min / median / max) |
|---|---|---|
| Live feed only, first 5 min | 22 | 452.5 / 453.9 / 454.2 |
| Live + 20 s upload processing | 8 | 491.3 / 504.4 / 510.8, **peak 512.0** |
| Live only, between uploads | 5 | 482.7 / 483.7 / 498.4 |
| Live + 56 s upload processing | 18 | 489.6 / 510.4 / **512.0**, peak 513 MiB |
| After the run | 16 | 469.2 / 473.9 / 474.2 |

- **The live feed alone doesn't grow.** Memory was 452.5 → 454.2 MB over its
  first 5 minutes, and flat in each later live-only stretch. This is not the
  Phase 3 failure: one YOLO frame at a time is far lighter than a second model.
- **The live feed alone leaves little room.** At ~454 MB it sits about
  35–50 MB above this same instance idling after an upload in Phase 3
  (405–421 MB), and an upload's own working memory stacks on top. With both,
  the container sat at its limit for the whole of each upload's processing.
- It wasn't killed this time. In Phase 3 the same "pinned at 512" state ended
  in a kill on 6 of 7 runs; the likely difference is that the extra load
  here is mostly cache the kernel can reclaim, not a second resident model
  (not verified). That's one run
  at the edge, not a safety margin.

## What happened to uploads, and to the live feed during them

| | 20 s 720p upload | 56 s 1080p upload |
|---|---|---|
| Upload queued and completed? | Yes | Yes |
| Processing time with the live feed running | 42.4 s | 162.2 s |
| Same clip, same instance, no live feed | 36.0–40.0 s (Phase 3) | 163.2 s (Phase 3) |
| Live frames dropped (`503 model busy`) | 11 | 29 |
| Longest gap with no processed live frame | 45 s | **245 s** |
| Whole server unresponsive | No | **~80 s** (6 consecutive polls timed out; 2 agent frames timed out after 18–23 s) |

- **Uploads are not hurt:** they queue normally and take the same time,
  within run-to-run noise.
- **The live feed is fully pre-empted:** an upload takes the model lock for
  its entire run. That's how the job system is designed: one job, start to
  finish.
- Live frames never block uploads. Each frame holds the lock for about 0.4 s,
  and the job acquires it between frames.
- Receiving the 72 MB upload over the network did **not** stop the live feed:
  frames kept being processed during the 221 s transfer. Only *processing* did.
- **The ~80 s unresponsive window** came during the 56 s upload's processing,
  with memory pinned at the limit:
  - `/live/stats`, `/jobs/{id}` and `/live/frames` all stopped answering.
  - The process did **not** restart: the same process start time and
    continuous uptime before and after.
  - The cause wasn't isolated. It's consistent with the container thrashing
    at its memory limit (the kernel evicting and re-reading pages), on top of
    a CPU share that's already fully used.

For the live feed this is worse than a clean failure: for up to 4 minutes, a
camera watching a perimeter sees nothing, and nothing alerts on it.

## Latency and frame rate

| | Median | p90 | Max |
|---|---|---|---|
| Detection on the server, per frame | **409 ms** | 495 ms | 904 ms |
| Agent round trip (send JPEG → response) | **2.7 s** | 5.8 s | 10.0 s |

- **Effective rate: 0.22 frames/s** (181 in 810 s), not the 1 fps configured.
- About 2.3 s of each round trip is network: the frame going up from your
  connection to Render, plus the proxy. The agent keeps one request in flight
  by design, so the rate is capped at 1 / round trip.
- Sending two or more at once would raise it. The server's 0.4 s per frame
  would then cap it at about 2 fps, with nothing left for uploads.
- In practice, an alert reaches the dashboard **~3 s** after the moment it
  shows, and a person in view for under ~5 s can be missed.

## Why Stage 4b's fix doesn't fit here

Stage 4b's first item is "a separate process/worker for live ingestion, so it
doesn't block uploads". On this instance, that needs a **second copy of the
model**:
- A second process loads its own YOLO26-N and PyTorch runtime.
- The live feed alone already sits at ~454 MB of the 512 MB.

This is the Phase 3 failure again: the second model is what got killed 6 times
in 7. The one-worker lock *is* the memory budget. You can't remove the
conflict without paying for memory this tier doesn't have.

## What would change the answer

- **A second free Render service just for live.** It would have its own
  512 MB and no lock to share, so the conflict disappears.
  - Cost: Render's free tier is 750 instance-hours a month per workspace. A
    live feed running 24/7 uses ~720 of them, leaving the upload service
    almost none.
  - Viable for a demo that runs a few hours; not for always-on.
- **A paid instance:** more memory and a real CPU. That would make one process
  with a live thread plus the job worker realistic.
- **Fully local** (the fallback named above): the agent runs the detector
  itself, next to the camera, with no cloud round trip. That removes the 2.7 s
  latency and the conflict. It gives up the shared dashboard and storage, or
  needs them rebuilt locally.

## Security note, for whoever picks this up

The Stage 4a endpoints have **no authentication**:
- Anyone who knows the URL can post frames.
- Every `/live` dashboard shows whatever it's sent.

That's acceptable on a throwaway service for a test, and is one more reason
`phase4a-live` must not be merged as-is.
