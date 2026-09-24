# Phase 1 — Upload + Class Selection

Supersedes "Day 6" as the next priority. Live camera input (previously planned as Day 6) and
attribute narrowing (car/clothing color, car type) both move later — this phase makes the
project actually usable on someone's own footage first, which is the real gap right now.

## The real problem (restated)
The MVP proves the pipeline works, but only against one fixed clip nobody but you can change.
This phase makes it a tool: upload a clip, choose whether you're looking for people or
vehicles, get back alerts for that clip specifically.

## Storage — a real gap in the original scope, fixed here
Render's free tier has **no persistent disk option at all** (paid-plan-only feature). The
filesystem is ephemeral and gets wiped on every redeploy, restart, *and* spin-down/wake cycle
— which happens routinely on the free tier's 15-minute idle sleep. This was never a problem for
the original MVP, since the one committed sample clip and its data ship inside the Docker image
itself. It's a real problem now: an uploaded clip's alerts and snapshots are per-user, dynamic
data that must survive between requests, and possibly across a restart if someone checks back
on a job later.

**Use Cloudflare R2** (S3-compatible API, 10GB free with no time limit, no egress fees — fits
the project's existing no-paid-services pattern):
- Uploaded clips and generated snapshots get pushed to R2, not written to local disk
- `Alert` stores an R2 object key/URL, not a local file path
- The original uploaded clip itself doesn't need to persist long-term once processed — only
  the results (alerts + snapshots) need to survive; the raw clip can be temporary/local for
  the duration of one processing job and cleaned up after
- Use an S3-compatible client library (e.g. `boto3` pointed at R2's endpoint) rather than a
  bespoke integration — this is a well-understood pattern, no need to reinvent it

## Scope (MVP for this phase — do not exceed)
- Upload a video clip (reasonable length/size limit — document a real number, don't guess one
  and hope)
- Choose class filter at upload time: person, vehicle, or both — this exposes the filter list
  that already exists in `clip_processor.py`, it's not new detection logic
- **Zone handling for arbitrary uploads**: don't build a full zone-drawing UI yet — default the
  zone to the entire frame for an uploaded clip. This unblocks upload without requiring the
  bigger zone-editor feature; "alert on anything of the selected class anywhere in frame" is a
  legitimate, simpler mode, not a cop-out
- **Process-then-poll, not a single blocking request** — an uploaded clip won't be the 5-second
  sample anymore, and you already measured ~14s of processing per second of footage. A clip
  longer than a few seconds will exceed any reasonable request timeout. Processing kicks off,
  returns a job ID immediately, and the frontend polls (or the existing dashboard pattern is
  adapted) for status/completion
- Validate upload: real video file, within size/length limits, resolution sane — fail clearly,
  not with an opaque error, consistent with every failure-path decision made so far

## What's explicitly OUT of scope for this phase
- Car color, car type as a filter, clothing color — Phase 2
- License plate recognition — not planned by default; see the research notes on why
- Zone drawing/editing UI — deferred; whole-frame default is this phase's answer
- Live camera input — now Phase 3, further out than before

## Architecture changes from the existing MVP
- New upload endpoint, validated, stores the clip temporarily (local disk for the duration of
  processing is fine — it doesn't need to survive a restart, only long enough to run the job)
- Processing becomes async: kick off a background job, return a job ID, add a status/result
  endpoint (this is a real architecture change, not a tweak — the existing synchronous
  `clip_processor.py` call gets wrapped in a job runner, not rewritten)
- Generated alerts and snapshots get pushed to Cloudflare R2, not written to local disk — this
  is what actually needs to survive between requests and across a possible restart
- Zone becomes per-upload (default: whole frame) instead of the single committed `zone.json`
- Frontend: an upload form + class-selection UI, polling for job status, same alert-list/
  snapshot display as before once results are ready — snapshot URLs now point at R2, not a
  local backend route

## What "done" looks like
Someone uploads their own short clip, picks "vehicle," and gets back a correct alert list for
that specific clip — with the existing zone-overlay, snapshot, and grouping UI reused, not
rebuilt.

## Notes
Keep the existing fixed sample clip working too — it's still useful for demos and as a known-
good regression check when you add upload on top.