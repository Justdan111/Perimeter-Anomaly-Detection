# Day 4 — Hardening

## Goal
By the end of today: the pipeline behaves correctly on the inputs it *hasn't* been tested
against yet — a clip with nothing in it, a model that fails to load, a corrupted file — plus
paying off the small cleanup items flagged across the last three days instead of letting them
quietly accumulate.

## Context — what's already solid, so today doesn't re-litigate it
74 backend tests, 14 frontend tests, lint, and the production build all pass. The concurrent-
processing 409, the snapshot resize, the endpoints, the zone-overlay UI, and the edge-case
warning system are all built and verified. Today is specifically about the paths nobody has
pointed a test at yet — not a rebuild of anything above.

## Checklist

### Untested failure paths
- [ ] **Zero-detection clip**: process a clip (or a frame sequence) where nothing matches the
      allowed classes. Confirm this returns a clean, empty alert list — 200 with `[]`, not an
      error, not a crash. This is a real, expected case (most footage most of the time has no
      alerts), not an edge case to shrug off.
- [ ] **Missing/failed model weights**: simulate `yolo26n.pt` failing to load (rename/remove
      the cached weights, or mock the load call). The service should start and clearly report
      `model_loaded: False` via `/health` (you already have this field — make sure it's
      actually accurate, not hardcoded `True`), and any processing request should fail with a
      clear, specific error — not an opaque 500 or a silent hang.
- [ ] **Corrupted/invalid clip file**: point the processor at a file that isn't a valid video.
      Fails clearly, doesn't crash the process, doesn't leave partial snapshot files behind.
- [ ] Add a real test for the 409 concurrent-processing path specifically, if one doesn't
      already exist from Day 3 — plant the race condition deliberately (two requests, confirm
      exactly one succeeds) rather than trusting the manual check that found it

### Cleanup flagged across Days 1-3 — pay these off now, don't defer further
- [ ] Exclude the transitive `opencv-python` (non-headless) pull-in from `ultralytics`, keep
      only `opencv-python-headless` — this was flagged Day 1 specifically because it'll bloat
      the Day 5 Docker image otherwise
- [ ] Fix the `starlette.testclient` deprecation warning (wants `httpx` v2) — flagged Day 1,
      harmless but worth clearing before it's buried under new warnings
- [ ] Add a real automated test for `/health` — it was verified manually on Day 1 but never
      got a test; one line, no excuse to keep deferring it
- [ ] Confirm test coverage on the image-serving routes (snapshot and reference-frame) and the
      zone endpoint (`GET /clips/sample`) — these existed since Day 3 but weren't specifically
      named as tested in your reports; verify, don't assume

### Documentation
- [ ] Document actual measured processing time (you have real numbers: ~3.5-4.6s locally for
      the sample clip) as a stated expectation in the README, not an implied "fast enough"
- [ ] Note explicitly that Render's free-tier CPU performance and any request-timeout risk from
      the blocking process endpoint will be *measured*, not assumed, on Day 5 — this was
      already flagged correctly by you, just make sure it survives into written docs, not just
      chat history

## What "done" looks like today
Every failure path above has a test that fails first (plant the bug, confirm red, fix, confirm
green) — same discipline you've used since Day 1. The dependency and warning cleanup items are
closed, not carried forward again. `/health` accurately reflects real model state, verified by
a test, not by memory of having checked it once.

## Explicitly not today
- No live camera / `FrameSource` work — that's Day 6, deliberately deferred, not today's job
- No Docker (Day 5)
- No new features — today only hardens what exists

## Notes
If you plant a bug for one of the failure-path tests and it *doesn't* turn red, don't just
delete the test — that's the same trap from Day 1's decorative vertex test. Figure out why it
didn't catch the bug before deciding whether the test or the code needs to change.