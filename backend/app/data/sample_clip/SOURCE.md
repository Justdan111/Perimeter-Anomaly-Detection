# Sample clip — source and license

## `crosswalk.mp4`

| | |
|---|---|
| **Source file** | `DiagonalCrosswalkYongeDundas.webm` |
| **Source page** | https://commons.wikimedia.org/wiki/File:DiagonalCrosswalkYongeDundas.webm |
| **Author** | Raysonho @ Open Grid Scheduler / Grid Engine ("own work") |
| **License** | **CC0 1.0 — Creative Commons Zero, Public Domain Dedication** |
| **Recorded** | 2011-08-18, Yonge–Dundas Square, Toronto |
| **Retrieved** | 2026-09-20 |

CC0 places the work in the public domain, so there is no attribution
obligation. The credit above is recorded anyway, because knowing where test
data came from matters more than the minimum the license demands.

### What was changed from the original

The original is 1920x1080, 56.6 s, ~69 MB — too large to commit and longer
than anything Day 1 needs. Derived with `ffmpeg`:

```sh
ffmpeg -ss 33 -i DiagonalCrosswalkYongeDundas.webm -t 5 \
       -vf scale=1280:720 -c:v libx264 -preset slow -crf 25 \
       -pix_fmt yuv420p -movflags +faststart -an crosswalk.mp4
```

- **Trimmed** to 5.005 s starting at t=33 s — a segment where a pedestrian
  crossing is in full flow, so there are both people and vehicles in frame.
- **Scaled** to 1280x720, the resolution the pipeline will actually run at.
- **Re-encoded** to H.264/MP4 for reliable OpenCV `VideoCapture` support.
- **Audio stripped** (`-an`) — irrelevant here, and it's dead weight in the repo.

Result: 120 frames at 23.976 fps, 1.9 MB.

### Why this clip

- **The camera is fixed.** Framing is identical at t=20 s and t=35 s, which
  matters: a zone polygon in fixed pixel coordinates is only meaningful if the
  camera doesn't move. A handheld clip would invalidate the whole zone model.
- It contains **both target classes** (people and vehicles) simultaneously.
- It has **a natural zone** — the crosswalk — to draw a polygon around, with
  real traffic both inside and outside it, so Day 2 can be hand-verified.

### Stated limitation

This is a busy public street, not a secured perimeter. It exercises the
pipeline honestly (real people, real vehicles, real occlusion, a fixed camera)
but it is **not** a benchmark, and this project makes no accuracy claims
measured against an academic surveillance dataset such as i-LIDS or VIRAT.
See `docs/PROJECT.md` → "Test data".

## `reference_frame.jpg`

The first frame of `crosswalk.mp4`, extracted for two purposes: the Day 1
single-image inference check, and (Day 3) the still the configured zone is
drawn over in the dashboard. Same source and license as above.
