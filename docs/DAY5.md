# Day 5 — Docker, deploy, and close out the deferred items

## Goal
By the end of today: the whole thing containerized and deployed, reachable by a stranger with
no local setup — plus resolving the two things Day 4 correctly left open rather than guessing
at, and closing out the license/demo/docs items that were deliberately deferred to today.

## Decisions this project owes from Day 4

**1. Docker runs as a non-root user.** Day 4 found that the write-failure tests skip under
root, and flagged that this matters if the Day 5 container runs as root — which most
Dockerfiles do by default unless told otherwise. Resolve it properly: create and run as a
non-root user in the Dockerfile. This is also just better practice independent of the test
issue (a compromised container process shouldn't have root inside the container). Once this is
in place, run the test suite *inside the built container* specifically — not just locally — to
confirm the previously-skipped write-failure tests now actually execute and pass, not just that
they no longer skip.

**2. Model weights get baked into the image at build time, not downloaded at runtime.** The
weights are correctly gitignored, but a container that downloads `yolo26n.pt` from Ultralytics'
servers on first request adds a real runtime dependency on external network access at exactly
the moment (cold start on a fresh deploy) it's least convenient to debug a failure. Add a
`RUN` step in the Dockerfile that triggers the download during the build (loading the model
once is enough to cache the weights file into that layer), so the deployed image is
self-contained and a cold start never needs outbound internet access to Ultralytics.

## Checklist

### Docker
- [ ] Dockerfile: `uv`-based (same pattern as the F1 project — install `uv`, `uv sync --frozen`
      against the committed lockfile), non-root user, weights baked in at build time per above
- [ ] Build and run the container **locally** first — hit `/health`, run the process endpoint
      against the container specifically, confirm timings are in the same ballpark as your
      native runs (3.2-4.7s), not wildly different — a container that's much slower than native
      for no clear reason is worth investigating before deploying it, not after
- [ ] Run the test suite inside the container, confirm the write-failure tests execute (not
      skip) now that the container isn't running as root

### Deploy
- [ ] Backend deployed to Render **via the Dockerfile**, `Root Directory` set to `backend` (or
      wherever this project's backend lives in the repo — confirm the monorepo setup matches
      what you used for the F1 project)
- [ ] `/health` reachable at the public URL, correctly reporting `model_loaded: true` (if it
      reports `false` on a fresh deploy, that's a real bug to chase, not something to route
      around)
- [ ] Frontend deployed to Vercel with `NEXT_PUBLIC_API_URL` set **before the build runs** —
      already flagged from Day 3 as a build-time requirement, not a runtime env var
- [ ] `PERIMETER_CORS_ORIGINS` on the backend includes the actual deployed Vercel origin
- [ ] **Measure, don't assume, the actual processing time and any timeout behavior on Render's
      free-tier CPU** — this was correctly flagged as a Day 5 task since Day 2. Time the
      blocking process endpoint against the deployed URL specifically. If it's close to
      whatever Render's request timeout turns out to be, that's a real finding to document, not
      a risk to silently absorb
- [ ] Full smoke test against the deployed URLs: trigger processing, confirm the alert list,
      snapshots, and zone overlay all load correctly from the live frontend

### Close out deferred items
- [ ] License note in the README: AGPL-3.0 (YOLO26), stated plainly, with what that implies
      for a closed-source commercial use case (would need an Ultralytics Enterprise License)
- [ ] Demo clip recorded against the deployed version (or local, if deploy timing makes a clean
      recording hard — say which in the README)
- [ ] Fix `SPEC.md`'s stale "Not yet started" status — it's been wrong since Day 1
- [ ] Note in the README, plainly, the two things this project deliberately doesn't do:
      real-time/live camera input (Day 6 plan exists but isn't built) and evaluation against an
      academic surveillance dataset (used royalty-free/self-recorded footage instead, by
      design, not by oversight)

## What "done" looks like
A stranger opens the deployed frontend, triggers processing on the sample clip, and sees a
correct, grouped alert list with snapshots and the zone overlay — with no local setup, running
against a container that was verified locally before it was trusted to a host. The README
honestly states what was measured (timings, on which hardware) versus what's still unverified
about the deployed environment specifically.

## Explicitly not today
- No Day 6 (`FrameSource`/live camera) work — stays a documented future plan, not built
- No new features beyond what's listed above

## Notes
If the deployed processing time turns out to be uncomfortably close to a timeout, resist
patching it with something hacky (fire-and-forget with polling, arbitrarily raising a timeout
setting) without first understanding whether it's a real architectural problem — a synchronous
process-and-respond design was a reasonable MVP choice for a single fixed clip; if it doesn't
hold up on free-tier hardware, that's worth naming as a real limitation in the README rather
than quietly working around it.