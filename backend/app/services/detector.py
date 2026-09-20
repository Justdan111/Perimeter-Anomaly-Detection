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

from ultralytics import YOLO  # noqa: E402  (must follow the env vars above)

DEFAULT_WEIGHTS = _MODELS_DIR / "yolo26n.pt"

# Only surface detections the model is reasonably sure about. 0.35 is a
# starting point, not a tuned value — Day 4 revisits it against the sample
# clip. Recorded here so the number is a decision, not an accident.
DEFAULT_CONFIDENCE_THRESHOLD = 0.35


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
                f"Fetch them with: "
                f'uv run python -c "from ultralytics import YOLO; '
                f"YOLO('yolo26n.pt')\""
            )
        self._model = YOLO(str(self.weights_path))

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
