"""EXPERIMENT (Phase 3, Stage 3a) — make/model classifier, for timing only.

This branch exists to measure what `Jordo23/vehicle-classifier` (EfficientNet-B4
fine-tuned on VMMRdb, MIT) costs on the deployed free tier when stacked on
YOLO26 + colour extraction. It is NOT the Stage 3b integration: nothing here
touches `Alert`, the dashboard or the default processing path. An upload runs
it only when it asks to (`experiment_make_model=true`), so the same instance
can be timed with and without it.

- Weights are baked into the image at build time (see Dockerfile), pinned to
  a Hugging Face revision, and loaded with `weights_only=True`: the checkpoint
  is a third-party pickle, and a full unpickle would execute whatever it holds.
- The model is loaded lazily on first use, so an instance that never runs the
  experiment never pays its memory.
- Cars and trucks only: the class list is make/model/year of cars and light
  trucks, so a bus or motorcycle can only get a meaningless label.
"""

from __future__ import annotations

import resource
import threading
import time
from pathlib import Path

import numpy as np

WEIGHTS = Path(__file__).resolve().parents[2] / "models" / "vehicle_classifier.pth"
CLASSIFIED_CLASSES = frozenset({"car", "truck"})
_INPUT = 380
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def peak_rss_mb() -> float:
    """Peak resident memory of this process so far (Linux reports KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def cgroup_memory_mb() -> dict:
    """The container's own memory accounting (what an OOM kill is based on).

    Process RSS overstates it here (it counts shared, reclaimable file
    pages), so both are reported. Missing files (not in a container) -> {}.
    """
    out = {}
    for key, name in (("cgroup_peak_mb", "memory.peak"), ("cgroup_current_mb", "memory.current")):
        try:
            out[key] = round(int(Path("/sys/fs/cgroup", name).read_text()) / 2**20, 1)
        except (OSError, ValueError):
            pass
    return out


class MakeModelClassifier:
    def __init__(self, weights: Path = WEIGHTS) -> None:
        self.weights = weights
        self._model = None
        self._names: dict[int, str] = {}
        self._lock = threading.Lock()
        self.load_s: float | None = None

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            import timm
            import torch

            started = time.perf_counter()
            # Lean load: memory-map the checkpoint and build the model on the
            # meta device, then adopt the mapped tensors (assign=True), so
            # the ~134 MB of weights exist once, not twice (checkpoint + a
            # randomly initialised model) — the first version of this was
            # killed for memory on the free tier.
            ck = torch.load(self.weights, map_location="cpu", weights_only=True, mmap=True)
            with torch.device("meta"):
                model = timm.create_model(
                    "efficientnet_b4", pretrained=False, num_classes=len(ck["class_mapping"])
                )
            model.load_state_dict(ck["model_state"], assign=True)
            model.eval()
            self._names = ck["class_mapping"]
            self._model = model
            self.load_s = time.perf_counter() - started

    def classify(self, crop_bgr: np.ndarray) -> tuple[str, float]:
        """Top-1 "Make Model Year" and its probability for one BGR crop.

        Preprocessing is the model card's: squash to 380x380, ImageNet
        normalisation.
        """
        import cv2
        import torch

        self.load()
        rgb = cv2.cvtColor(cv2.resize(crop_bgr, (_INPUT, _INPUT), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) / 255.0 - _MEAN) / _STD
        tensor = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)
        with torch.inference_mode():
            probs = torch.softmax(self._model(tensor), dim=1)[0]
        p, i = probs.max(0)
        return self._names[i.item()], p.item()


class ExperimentStats:
    """What one job's experiment measured; stored on the job record."""

    def __init__(self) -> None:
        self.crops = 0
        self.classify_s = 0.0
        self.per_crop_ms: list[float] = []
        self.samples: list[dict] = []

    def record(self, ms: float, frame_index: int, bbox, label: str, prob: float) -> None:
        self.crops += 1
        self.classify_s += ms / 1000
        self.per_crop_ms.append(ms)
        if len(self.samples) < 30:
            self.samples.append(
                {"frame": frame_index, "bbox": [round(v) for v in bbox], "label": label, "p": round(prob, 3)}
            )

    def summary(self, load_s: float | None) -> dict:
        ms = sorted(self.per_crop_ms)
        pct = lambda q: round(ms[min(len(ms) - 1, int(q * len(ms)))], 1) if ms else None
        return {
            "crops_classified": self.crops,
            "classify_total_s": round(self.classify_s, 2),
            "per_crop_ms_median": pct(0.5),
            "per_crop_ms_p90": pct(0.9),
            "per_crop_ms_max": round(ms[-1], 1) if ms else None,
            "model_load_s": round(load_s, 2) if load_s is not None else None,
            "peak_rss_mb": round(peak_rss_mb(), 1),
            **cgroup_memory_mb(),
            "samples": self.samples,
        }
