# Stage 3a working files

The scripts behind the make/model validation in [`../PHASE3.md`](../PHASE3.md).
They were run from a scratch folder holding the model weights
(`vehicle_classifier.pth` from `Jordo23/vehicle-classifier`), the clips and a
`runs/` output folder. They're kept for the record, not as project code.

- `run_clip.py CLIP NAME [--sample-fps N]`: runs the project's YOLO26 detector
  over a clip. It classifies every vehicle detection (top-5) and saves each
  crop and its timing. `sys.path` points at `backend/`. Run it with
  `uv run --with timm` from `backend/`.
- `review.py NAME --min-w 100`: links detections into per-vehicle tracks (IoU,
  for analysis only), measures how stable the label is within a track, and draws
  contact sheets of each track's largest crop.
- `verdicts/*.txt`: the by-eye verdict for every reviewed track, and what the
  vehicle actually was when it could be told. `C` right, `M` make only,
  `P` plausible, `U` unverifiable, `W` wrong, `B` obviously wrong body type,
  `X` not one vehicle; `bus`/`moto` are excluded. `nonUS` marks vehicles from
  outside the US market.
- `tally.py`: the accuracy and confidence tables.
- `prep_fair.py`: compares preprocessing variants by make-level accuracy.
