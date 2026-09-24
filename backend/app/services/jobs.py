"""Background processing for uploaded clips (Phase 1).

Design — why it looks like this:

- **One worker thread, a FIFO queue.** The host has well under one CPU and
  there is one model instance, which isn't safe to share across threads.
  Two jobs at once would each run at half speed and risk the model; one at a
  time finishes the first job sooner and keeps the model single-threaded.
- **The result store (Cloudflare R2) is the record of a job; memory is a
  cache.** Render's free tier wipes the local filesystem on every restart,
  redeploy and idle spin-down (docs/PHASE1.md), so everything that must
  outlive the process goes to R2 under `jobs/<id>/`: `job.json` (status,
  rewritten at each state change), `result.json` (alerts), `reference.jpg`
  and `snapshots/`. Memory holds only live progress for the job running now.
- **The uploaded video is local and temporary.** It exists in a work folder
  only while its job runs, then is deleted — only results persist.
- **A restart can't resume a job** (its local video is gone). A job whose
  stored status is still queued/processing but which this process doesn't
  know about was interrupted by a restart: it is reported, and saved, as
  failed with that reason.
- **Shares the model lock with the sample-clip endpoint**, so the model runs
  one clip at a time whichever path started it. A job waiting for that lock
  reports `queued`.
- **Bounded**: at most `max_active` unfinished jobs (the upload endpoint
  answers 429 past that). Old results are removed by a lifecycle rule on the
  bucket (README), not by this code.

`clip_processor.process_clip` is unchanged in what it does — the job calls
it with the uploaded clip, a whole-frame zone, the chosen classes and a
progress callback, then uploads what it produced.
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
    ClipReadError,
    FrameDetector,
    FrameSizeMismatchError,
    classes_for,
    count_sampled_frames,
    process_clip,
)
from app.services.storage import ObjectStore
from app.services.uploads import ClipProbe

logger = logging.getLogger(__name__)


class JobState(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"


FINISHED = (JobState.COMPLETE, JobState.FAILED)

INTERRUPTED_MESSAGE = (
    "processing was interrupted by a server restart (the free-tier host restarts on "
    "redeploys and after idling); please upload the clip again"
)


class QueueFull(Exception):
    """Too many unfinished jobs; the caller should try again later."""


class SaveError(RuntimeError):
    """Results were produced but couldn't be saved to the store."""


def job_prefix(job_id: str) -> str:
    return f"jobs/{job_id}/"


def reference_key(job_id: str) -> str:
    return f"jobs/{job_id}/reference.jpg"


def snapshot_key(job_id: str, filename: str) -> str:
    return f"jobs/{job_id}/snapshots/{filename}"


@dataclass
class Job:
    job_id: str
    filename: str
    classes: list[str]  # the uploader's selection, e.g. ["person", "dog"]
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
    processing_time_s: float | None = None
    # Local, temporary: only set while this process is running the job.
    work_dir: Path | None = None

    @property
    def video_path(self) -> Path:
        assert self.work_dir is not None
        return self.work_dir / "upload"

    def to_record(self) -> dict:
        return {
            "job_id": self.job_id,
            "filename": self.filename,
            "classes": self.classes,
            "probe": {
                "width": self.probe.width,
                "height": self.probe.height,
                "fps": self.probe.fps,
                "frame_count": self.probe.frame_count,
            },
            "zone": self.zone.model_dump(mode="json"),
            "sample_fps": self.sample_fps,
            "frames_to_process": self.frames_to_process,
            "frames_processed": self.frames_processed,
            "created_at": self.created_at,
            "status": self.status.value,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "processing_time_s": self.processing_time_s,
        }

    @classmethod
    def from_record(cls, record: dict) -> Job:
        return cls(
            job_id=record["job_id"],
            filename=record["filename"],
            # Phase 1 stored one string ("person" | "vehicle" | "both").
            classes=_normalise_classes(record["classes"]),
            probe=ClipProbe(**record["probe"]),
            zone=Zone.model_validate(record["zone"]),
            sample_fps=record["sample_fps"],
            frames_to_process=record["frames_to_process"],
            frames_processed=record["frames_processed"],
            created_at=record["created_at"],
            status=JobState(record["status"]),
            started_at=record["started_at"],
            finished_at=record["finished_at"],
            error=record["error"],
            processing_time_s=record["processing_time_s"],
        )


def _normalise_classes(stored: str | list[str]) -> list[str]:
    if isinstance(stored, str):
        return ["person", "vehicle"] if stored == "both" else [stored]
    return list(stored)


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
    if isinstance(error, SaveError):
        return f"the results could not be saved to storage: {error}"
    return f"processing failed unexpectedly ({type(error).__name__}); see the server log"


class JobRunner:
    def __init__(
        self,
        model_lock: threading.Lock,
        store: ObjectStore,
        work_root: Path,
        max_active: int = 3,
    ) -> None:
        self.max_active = max_active
        self.store = store
        self.work_root = Path(work_root)
        self._model_lock = model_lock
        self._state_lock = threading.Lock()
        # Jobs this process knows about: unfinished ones (the source of live
        # progress) plus finished ones it ran or has looked up (a cache).
        self._jobs: dict[str, Job] = {}
        self._results: dict[str, ClipResult] = {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="clip-job")

    # --- queries ----------------------------------------------------------------------

    def get(self, job_id: str) -> Job | None:
        """The job, from memory or (after a restart) from the store."""
        with self._state_lock:
            job = self._jobs.get(job_id)
        if job is not None:
            return job
        record = self.store.get_json(f"{job_prefix(job_id)}job.json")
        if record is None:
            return None
        job = Job.from_record(record)
        if job.status not in FINISHED:
            # The store says it was still running, but this process has never
            # seen it: whichever process was running it is gone.
            job.status = JobState.FAILED
            job.error = INTERRUPTED_MESSAGE
            job.finished_at = job.finished_at or time.time()
            self._save_record(job)
        with self._state_lock:
            self._jobs.setdefault(job_id, job)
        return job

    def result(self, job: Job) -> ClipResult | None:
        """The finished job's alerts (snapshot fields are store keys)."""
        if job.status is not JobState.COMPLETE:
            return None
        with self._state_lock:
            cached = self._results.get(job.job_id)
        if cached is not None:
            return cached
        record = self.store.get_json(f"{job_prefix(job.job_id)}result.json")
        if record is None:
            return None
        result = ClipResult.model_validate(record)
        with self._state_lock:
            self._results[job.job_id] = result
        return result

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
        work_dir: Path,
        reference_frame: Path,
        filename: str,
        classes: list[str],
        probe: ClipProbe,
        zone: Zone,
        sample_fps: float | None,
        detector: FrameDetector,
    ) -> Job:
        """Register a validated upload (its video already at work_dir/upload)."""
        job = Job(
            job_id=job_id,
            filename=filename,
            classes=classes,
            probe=probe,
            zone=zone,
            sample_fps=sample_fps,
            frames_to_process=count_sampled_frames(probe.frame_count, probe.fps, sample_fps),
            work_dir=work_dir,
        )
        with self._state_lock:
            active = sum(j.status not in FINISHED for j in self._jobs.values())
            if active >= self.max_active:
                raise QueueFull
            self._jobs[job_id] = job
        try:
            self.store.put_file(reference_key(job_id), reference_frame, "image/jpeg")
            self._save_record(job)
        except Exception:
            with self._state_lock:
                self._jobs.pop(job_id, None)
            raise
        self._executor.submit(self._run, job, detector)
        return job

    # --- the worker --------------------------------------------------------------------

    def _run(self, job: Job, detector: FrameDetector) -> None:
        # The final state is decided first and *published last*: pollers see
        # `complete` only once the results, the final record and the local
        # cleanup are all done. (Publishing first let a client see
        # "complete" before job.json said so — caught by a flaky test.)
        final_status = JobState.FAILED
        error: str | None = None
        try:
            # Blocks while the sample-clip endpoint (or anything else) has
            # the model; the job stays "queued" until then.
            with self._model_lock:
                job.status = JobState.PROCESSING
                job.started_at = time.time()
                self._save_record(job)

                def progress(frames_done: int) -> None:
                    job.frames_processed = frames_done

                try:
                    result = process_clip(
                        job.video_path,
                        job.zone,
                        detector,
                        job.work_dir / "snapshots",
                        sample_fps=job.sample_fps,
                        allowed_classes=classes_for(job.classes),
                        on_progress=progress,
                    )
                    stored = self._save_results(job, result)
                    with self._state_lock:
                        self._results[job.job_id] = stored
                    job.processing_time_s = result.processing_time_s
                    final_status = JobState.COMPLETE
                except Exception as e:
                    logger.exception("job %s failed", job.job_id)
                    error = describe_failure(e)
        except Exception:  # never let the worker thread die silently
            logger.exception("job %s crashed outside processing", job.job_id)
            error = error or "processing failed unexpectedly; see the server log"
        finally:
            job.finished_at = time.time()
            job.error = error
            try:
                record = job.to_record()
                record["status"] = final_status.value
                self.store.put_json(f"{job_prefix(job.job_id)}job.json", record)
            except Exception:
                logger.exception("could not save final status of job %s", job.job_id)
            if job.work_dir is not None:
                shutil.rmtree(job.work_dir, ignore_errors=True)  # video + local snapshots
            job.status = final_status  # published last
            with self._state_lock:
                self._prune_memory()

    def _save_results(self, job: Job, result: ClipResult) -> ClipResult:
        """Upload snapshots, then the alerts with store keys in place of filenames."""
        try:
            # In parallel: one at a time, each upload to R2 took ~0.48 s on
            # Render (latency, not bandwidth) — 55 s of saving for a 56 s
            # clip. boto3 clients are thread-safe; 8 workers stay within its
            # default pool of 10 connections.
            filenames = sorted({a.snapshot for a in result.alerts})
            pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="snapshot-upload")
            try:
                futures = [
                    pool.submit(
                        self.store.put_file,
                        snapshot_key(job.job_id, filename),
                        job.work_dir / "snapshots" / filename,
                        "image/jpeg",
                    )
                    for filename in filenames
                ]
                for future in futures:
                    future.result()  # re-raises the first upload failure
            finally:
                # On failure: cancel what hasn't started and wait for what
                # has, so the cleanup below can't race an in-flight upload.
                pool.shutdown(wait=True, cancel_futures=True)
            stored = result.model_copy(
                update={
                    "alerts": [
                        a.model_copy(update={"snapshot": snapshot_key(job.job_id, a.snapshot)})
                        for a in result.alerts
                    ]
                }
            )
            self.store.put_json(f"{job_prefix(job.job_id)}result.json", stored.model_dump(mode="json"))
            return stored
        except Exception as e:
            # Don't leave half a set of snapshots behind for a failed job.
            try:
                self.store.delete_prefix(f"{job_prefix(job.job_id)}snapshots/")
            except Exception:
                logger.exception("could not clean up snapshots of job %s", job.job_id)
            raise SaveError(str(e)) from e

    def _save_record(self, job: Job) -> None:
        self.store.put_json(f"{job_prefix(job.job_id)}job.json", job.to_record())

    def _prune_memory(self, keep: int = 50) -> None:
        """Forget old finished jobs from memory; they stay in the store."""
        finished = [j.job_id for j in self._jobs.values() if j.status in FINISHED]
        for job_id in finished[: max(0, len(finished) - keep)]:
            self._jobs.pop(job_id, None)
            self._results.pop(job_id, None)

    def shutdown(self, wait: bool = False) -> None:
        """Stop taking work: queued jobs are cancelled; the running one is
        left to finish (or waited for, with wait=True — tests use that so no
        job outlives the fake storage it was started against)."""
        self._executor.shutdown(wait=wait, cancel_futures=True)
