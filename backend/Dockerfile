# syntax=docker/dockerfile:1
#
# Backend image for the perimeter anomaly detection API.
#
#   docker build -t perimeter-backend .              # runtime image (default)
#   docker build --target test -t perimeter-test .   # same image + dev deps + tests
#   docker run --rm perimeter-test                   # runs pytest inside it
#
# Decisions (see docs/DAY5.md):
# - Runs as a non-root user. Also the only way the write-failure tests can
#   run in here: root ignores directory permissions, so they skip as root.
# - The YOLO26-N weights are downloaded while building and baked into the
#   image, so a cold start never needs outbound internet access to fetch them.
# - PyTorch is the CPU-only build on Linux (see [tool.uv.sources] in
#   pyproject.toml); the CUDA build would add several GB for nothing.

# ---------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS runtime

# Pinned to the uv version the lockfile was produced with.
COPY --from=ghcr.io/astral-sh/uv:0.12.13 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    YOLO_CONFIG_DIR=/app/models \
    # Idle OpenMP threads sleep instead of spinning. On a CPU-limited host,
    # spinning threads burn the CPU allowance inference needs: measured at
    # --cpus=1, 168 -> 79 ms/frame with 1 inference thread.
    OMP_WAIT_POLICY=PASSIVE

RUN groupadd --system app \
 && useradd --system --gid app --no-create-home --home-dir /app app \
 && mkdir -p /app/models /app/output \
 && chown app:app /app /app/models /app/output

WORKDIR /app
USER app

# Dependencies first, in their own layer: rebuilt only when the lockfile
# changes, not on every code edit. --frozen fails the build if uv.lock is out
# of date with pyproject.toml instead of silently re-resolving.
COPY --chown=app:app pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-cache

# Bake the weights in. Loading the model once through the same Detector class
# the service uses downloads yolo26n.pt to exactly the path it loads from at
# runtime (app/services/detector.py sets YOLO_CONFIG_DIR to models/ too).
COPY --chown=app:app app ./app
RUN python -c "\
from ultralytics import YOLO; \
from app.services.detector import DEFAULT_WEIGHTS, Detector; \
YOLO(str(DEFAULT_WEIGHTS)); \
d = Detector(); d.load(); \
print('baked', DEFAULT_WEIGHTS, DEFAULT_WEIGHTS.stat().st_size, 'bytes')"

# Render sets $PORT; default to 8000 elsewhere. One worker on purpose: the
# processing lock and the last result live in process memory, so a second
# worker would have its own lock and its own (empty) results.
EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]

# ---------------------------------------------------------------------------
# Test stage: the runtime image above plus dev dependencies and tests, still
# running as the non-root user.
FROM runtime AS test
RUN uv sync --frozen --no-cache
COPY --chown=app:app tests ./tests
CMD ["pytest", "-q", "-rs"]

# ---------------------------------------------------------------------------
# Last stage = what `docker build` (and Render) produce by default.
FROM runtime
