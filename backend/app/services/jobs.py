"""Background processing for uploaded clips (Phase 1).

Design — why it looks like this:

- **One worker thread, a FIFO queue.** The host has well under one CPU and
  there is one model instance, which isn't safe to share across threads.
  Two jobs at once would each run at half speed and risk the model; one at a
  time finishes the first job sooner and keeps the model single-threaded.
- **State in memory** (a dict keyed by job id). The service runs as one
  process on one instance with no persistent disk. A database or Redis only
  pays for itself with several workers or instances. The cost, stated: a
  restart or redeploy forgets every job, and its files are cleared at
  startup. The API says so ("unknown job — the server may have restarted")
  rather than returning a bare 404.
- **Shares the model lock with the sample-clip endpoint**, so the model runs
  one clip at a time whichever path started it. A job waiting for that lock
  reports `queued`.
- **Bounded**: at most `max_active` unfinished jobs (the upload endpoint
  answers 429 past that), and only the newest `keep_finished` finished jobs
  keep their files. Each uploaded video is deleted as soon as its job ends;
  only the snapshots and the reference frame are kept.

`clip_processor.process_clip` is unchanged in what it does — the job just
calls it, with the uploaded clip, a whole-frame zone, the chosen classes and
a progress callback.
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from app.models.schemas import ClipResult, Zone
from app.services.clip_processor import (
    CLASS_GROUPS,
    ClipReadError,
    FrameDetector,
    FrameSizeMismatchError,
    count_sampled_frames,
    process_clip,
)
from app.services.uploads import ClipProbe

logger = logging.getLogger(__name__)


class JobState(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"


FINISHED = (JobState.COMPLETE, JobState.FAILED)


class QueueFull(Exception):
    """Too many unfinished jobs; the caller should try again later."""


@dataclass
class Job:
    job_id: str
    directory: Path
    video_path: Path
    filename: str
    classes: str
    probe: ClipProbe
    zone: Zone
    sample_fps: float | None
    frames_to_process: int
    created_at: float = field(default_factory=time.time)
    status: JobState = JobState.QUEUED
    frames_processed: int = 0
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    result: ClipResult | None = None

    @property
    def snapshot_dir(self) -> Path:
        return self.directory / "snapshots"

    @property
    def reference_frame_path(self) -> Path:
        return self.directory / "reference.jpg"


def new_job_id() -> str:
    # 128 random bits: there are no accounts, so a job's id is what keeps
    # one uploader's results from being guessed by another.
    return uuid.uuid4().hex


def describe_failure(error: BaseException) -> str:
    """A message for the uploader. Specific where the cause is known."""
    if isinstance(error, ClipReadError):
        return f"the video could not be read: {error}"
    if isinstance(error, FrameSizeMismatchError):
        return "the video changes resolution partway through, which isn't supported"
    return f"processing failed unexpectedly ({type(error).__name__}); see the server log"


class JobRunner:
    def __init__(
        self,
        model_lock: threading.Lock,
        max_active: int = 3,
        keep_finished: int = 10,
    ) -> None:
        self.max_active = max_active
        self.keep_finished = keep_finished
        self._model_lock = model_lock
        self._state_lock = threading.Lock()
        self._jobs: dict[str, Job] = {}  # insertion order = submission order
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="clip-job")

    # --- queries ----------------------------------------------------------------------

    def get(self, job_id: str) -> Job | None:
        with self._state_lock:
            return self._jobs.get(job_id)

    def active_count(self) -> int:
        with self._state_lock:
            return sum(j.status not in FINISHED for j in self._jobs.values())

    def queue_position(self, job: Job) -> int:
        """1 for the next job to run; 0 if it isn't waiting."""
        if job.status is not JobState.QUEUED:
            return 0
        with self._state_lock:
            queued = [j for j in self._jobs.values() if j.status is JobState.QUEUED]
        return queued.index(job) + 1 if job in queued else 0

    # --- submission ------------------------------------------------------------------

    def submit(
        self,
        *,
        job_id: str,
        directory: Path,
        video_path: Path,
        filename: str,
        classes: str,
        probe: ClipProbe,
        zone: Zone,
        sample_fps: float | None,
        detector: FrameDetector,
    ) -> Job:
        job = Job(
            job_id=job_id,
            directory=directory,
            video_path=video_path,
            filename=filename,
            classes=classes,
            probe=probe,
            zone=zone,
            sample_fps=sample_fps,
            frames_to_process=count_sampled_frames(probe.frame_count, probe.fps, sample_fps),
        )
        with self._state_lock:
            active = sum(j.status not in FINISHED for j in self._jobs.values())
            if active >= self.max_active:
                raise QueueFull
            self._jobs[job_id] = job
        self._executor.submit(self._run, job, detector)
        return job

    # --- the worker --------------------------------------------------------------------

    def _run(self, job: Job, detector: FrameDetector) -> None:
        try:
            # Blocks while the sample-clip endpoint (or anything else) has
            # the model; the job stays "queued" until then.
            with self._model_lock:
                job.status = JobState.PROCESSING
                job.started_at = time.time()

                def progress(frames_done: int) -> None:
                    job.frames_processed = frames_done

                try:
                    job.result = process_clip(
                        job.video_path,
                        job.zone,
                        detector,
                        job.snapshot_dir,
                        sample_fps=job.sample_fps,
                        allowed_classes=CLASS_GROUPS[job.classes],
                        on_progress=progress,
                    )
                    job.status = JobState.COMPLETE
                except Exception as e:
                    logger.exception("job %s failed", job.job_id)
                    job.error = describe_failure(e)
                    job.status = JobState.FAILED
        except Exception:  # never let the worker thread die silently
            logger.exception("job %s crashed outside processing", job.job_id)
            job.error = job.error or "processing failed unexpectedly; see the server log"
            job.status = JobState.FAILED
        finally:
            job.finished_at = time.time()
            job.video_path.unlink(missing_ok=True)
            self._prune()

    def _prune(self) -> None:
        with self._state_lock:
            finished = [j for j in self._jobs.values() if j.status in FINISHED]
            stale = finished[: max(0, len(finished) - self.keep_finished)]
            for job in stale:
                del self._jobs[job.job_id]
        for job in stale:
            shutil.rmtree(job.directory, ignore_errors=True)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
