"""YOLO26-N inference on a single frame.

This is the ONLY module that touches the model. Everything downstream consumes
`Detection` objects (plain Pydantic data), never an Ultralytics `Results`
object — so replacing or upgrading the model touches this file and nothing
else.

Model: **YOLO26-N** (Ultralytics, January 2026). Chosen for CPU inference —
this is intended to run on a free-tier host with no GPU. See docs/PROJECT.md
for the full reasoning.

LICENSE: the `ultralytics` package and the YOLO26 weights are **AGPL-3.0**.
That is fine for this project, which is open source and public. It would
require an Ultralytics Enterprise License to use inside a closed-source
commercial product. Stated here rather than left to be discovered.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from app.models.schemas import Detection

# Keep Ultralytics' settings/config inside the project instead of the user's
# home directory, so behaviour is the same in Docker (Day 5) as locally.
# Must be set before `ultralytics` is imported.
_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
os.environ.setdefault("YOLO_CONFIG_DIR", str(_MODELS_DIR))
os.environ.setdefault("YOLO_VERBOSE", "false")

import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402  (must follow the env vars above)

DEFAULT_WEIGHTS = _MODELS_DIR / "yolo26n.pt"

# Only surface detections the model is reasonably sure about. 0.35 is a
# starting point, not a tuned value — Day 4 revisits it against the sample
# clip. Recorded here so the number is a decision, not an accident.
DEFAULT_CONFIDENCE_THRESHOLD = 0.35

_CGROUP_V2_CPU_MAX = Path("/sys/fs/cgroup/cpu.max")
_CGROUP_V1_QUOTA = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
_CGROUP_V1_PERIOD = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")


# --- CPU threads for inference ------------------------------------------------
#
# ultralytics sets PyTorch's thread count to os.cpu_count() - 1 during the
# first prediction. In a container, os.cpu_count() is the HOST's core count,
# not the container's CPU allowance, so a container limited to 1 CPU ran 7
# inference threads fighting over it — ~13x slower per frame than 1 thread
# (measured Day 5; see tests/test_inference_threads.py for the numbers).
# The fix: read the container's actual CPU limit and use that many threads.
# With no limit (native runs), ultralytics' own default is left alone.


def cpu_limit_from_cgroup(
    cpu_max_v2: str | None = None,
    cfs_quota_v1: str | None = None,
    cfs_period_v1: str | None = None,
) -> float | None:
    """The container's CPU allowance in CPUs (e.g. 0.5), or None if unlimited.

    Takes the cgroup files' contents rather than reading them, so it can be
    tested with plain strings. Anything unparseable counts as "no limit":
    this only tunes performance, so it must never stop the service starting.
    """
    try:
        if cpu_max_v2 is not None:
            quota, period = cpu_max_v2.split()
            if quota == "max":
                return None
            limit = int(quota) / int(period)
        elif cfs_quota_v1 is not None and cfs_period_v1 is not None:
            quota_us = int(cfs_quota_v1)
            if quota_us < 0:
                return None
            limit = quota_us / int(cfs_period_v1)
        else:
            return None
    except (ValueError, ZeroDivisionError):
        return None
    return limit if limit > 0 else None


def read_cgroup_cpu_limit() -> float | None:
    """`cpu_limit_from_cgroup` applied to this machine's cgroup files."""

    def read(path: Path) -> str | None:
        try:
            return path.read_text()
        except OSError:
            return None

    return cpu_limit_from_cgroup(
        cpu_max_v2=read(_CGROUP_V2_CPU_MAX),
        cfs_quota_v1=read(_CGROUP_V1_QUOTA),
        cfs_period_v1=read(_CGROUP_V1_PERIOD),
    )


def choose_inference_threads(cpu_limit: float | None, override: str | None) -> int | None:
    """Thread count to set, or None to keep the library default.

    `override` is the PERIMETER_TORCH_THREADS environment variable. Otherwise
    a CPU limit is rounded DOWN to whole CPUs (minimum 1): a thread beyond
    the allowance gets throttled, and throttling is what made 7 threads slow.
    """
    if override is not None:
        try:
            threads = int(override)
        except ValueError:
            threads = 0
        if threads < 1:
            raise ValueError(
                f"PERIMETER_TORCH_THREADS must be a positive integer, got {override!r}"
            )
        return threads
    if cpu_limit is None:
        return None
    return max(1, int(cpu_limit))


class Detector:
    """Wraps YOLO26-N. Frame in, list of `Detection` out."""

    def __init__(
        self,
        weights_path: Path | str = DEFAULT_WEIGHTS,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        self.weights_path = Path(weights_path)
        self.confidence_threshold = confidence_threshold
        self._model: YOLO | None = None

    def load(self) -> None:
        """Load the weights into memory.

        Separate from `__init__` and from `detect` so the cost is paid at a
        moment of our choosing (service startup) rather than inside the first
        request. Idempotent.
        """
        if self._model is not None:
            return
        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"YOLO26-N weights not found at {self.weights_path}. "
                # Ultralytics downloads a missing named checkpoint to the
                # exact path given, so name the path the service loads from.
                # (A bare 'yolo26n.pt' would land in the current directory.)
                f"Fetch them with: "
                f'uv run python -c "from ultralytics import YOLO; '
                f"YOLO('{self.weights_path}')\""
            )
        model = YOLO(str(self.weights_path))

        # Run one prediction now: ultralytics finishes its setup (including
        # resetting the thread count, see above) on the first prediction, so
        # the thread count can only be set reliably after it. It also means
        # the first real request doesn't pay that setup cost.
        model.predict(
            np.zeros((64, 64, 3), dtype=np.uint8),
            conf=self.confidence_threshold,
            verbose=False,
        )
        threads = choose_inference_threads(
            read_cgroup_cpu_limit(), os.environ.get("PERIMETER_TORCH_THREADS")
        )
        if threads is not None:
            torch.set_num_threads(threads)
        self._model = model

    @property
    def inference_threads(self) -> int:
        """PyTorch CPU threads inference is actually using."""
        return torch.get_num_threads()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def class_names(self) -> dict[int, str]:
        """The model's class id -> label map (80 COCO classes)."""
        self.load()
        assert self._model is not None
        return self._model.names

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run detection on one frame.

        Args:
            frame: An image as a numpy array in BGR channel order — i.e.
                exactly what `cv2.imread` and `cv2.VideoCapture.read` return.

        Returns:
            Detections above the confidence threshold, in the model's own
            order (descending confidence). An empty list is a normal,
            expected result: a frame with nothing in it.
        """
        self.load()
        assert self._model is not None

        results = self._model.predict(
            frame,
            conf=self.confidence_threshold,
            verbose=False,
        )

        # predict() accepts batches, so it always returns a list. One frame in
        # means exactly one Results out.
        result = results[0]
        names = result.names

        detections: list[Detection] = []
        # .boxes holds parallel torch tensors, not per-object records:
        #   boxes.xyxy -> (N, 4) float32   boxes.conf -> (N,) float32
        #   boxes.cls  -> (N,) float32 (class ids as floats, hence int())
        # Moving to CPU numpy once is cheaper than touching tensors per row.
        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        for (x1, y1, x2, y2), conf, class_id in zip(xyxy, confs, classes):
            detections.append(
                Detection(
                    class_name=names[int(class_id)],
                    confidence=float(conf),
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                )
            )
        return detections
