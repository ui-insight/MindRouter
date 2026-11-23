############################################################
#
# mindrouter mesh-worker - async image-to-3D mesh service
#
# worker.py: JobManager — in-memory job store, bounded FIFO queue, ONE GPU
#     executor thread, per-job timeout, cooperative cancel and result TTL.
#
#   Generation runs on a dedicated single-thread executor, never on the event
#   loop, so GET /health answers in <5 s while a job is in flight. One job at a
#   time: a mesh job saturates the GPU, and max_concurrent=1 on the gateway's
#   backend row is the real backpressure.
#
#   A timed-out job is reported failed at once; the GPU thread cannot be
#   killed, so it is told to cancel and the NEXT job waits until it has
#   stopped at a stage boundary (the GPU stays serialized). If it never
#   stops, /health turns 503 ("stuck") after the grace period.
#
#   State is in memory only. After a restart every id is unknown (404) and
#   the gateway requeues; leftover files from the previous process are
#   deleted at startup.
#
# Luke Sheneman — University of Idaho RCDS — sheneman@uidaho.edu
#
############################################################

"""Async job manager for the mesh worker."""

import asyncio
import collections
import hashlib
import logging
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from config import WorkerConfig
from engines.base import Cancelled, EngineInputError, MeshEngine, StageTracker
from glb import GLBError, inspect_glb
from inputs import MeshSpec, open_image

log = logging.getLogger(__name__)

JOB_ID_RE = re.compile(r"^mjob-[0-9a-f]{24}$")
TERMINAL = ("completed", "failed", "cancelled")


class QueueFull(Exception):
    pass


@dataclass
class Job:
    id: str
    spec: Optional[MeshSpec]
    status: str = "queued"          # queued|in_progress|completed|failed|cancelled
    stage: str = "queued"           # queued|preprocess|shape|texture|export|done
    progress: float = 0.0           # 0-100, never decreases
    error: Optional[Dict[str, str]] = None
    outputs: Optional[Dict[str, Dict[str, Any]]] = None
    faces: Optional[int] = None
    vertices: Optional[int] = None
    cancel: bool = False
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    duration_ms: Optional[int] = None


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_png(path: str) -> bool:
    with open(path, "rb") as fh:
        return fh.read(8) == b"\x89PNG\r\n\x1a\n"


class JobManager:
    def __init__(self, engine: MeshEngine, config: WorkerConfig):
        self.engine = engine
        self.config = config
        self.jobs: Dict[str, Job] = {}
        self._pending: "collections.deque[str]" = collections.deque()
        self._wakeup: Optional[asyncio.Event] = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mesh-gpu")
        self._consumer: Optional[asyncio.Task] = None
        self._sweeper: Optional[asyncio.Task] = None
        self._current: Optional[str] = None
        os.makedirs(config.output_dir, exist_ok=True)

    # -- paths -------------------------------------------------------------
    def glb_path(self, job_id: str) -> str:
        return os.path.join(self.config.output_dir, f"{os.path.basename(job_id)}.glb")

    def preview_path(self, job_id: str) -> str:
        return os.path.join(self.config.output_dir, f"{os.path.basename(job_id)}.preview.png")

    def _remove_outputs(self, job_id: str) -> None:
        for path in (self.glb_path(job_id), self.preview_path(job_id)):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            except OSError:
                log.warning("could not remove %s", path, exc_info=True)

    def _remove_orphans(self) -> None:
        """Files left by a previous process belong to ids nobody can poll."""
        for name in os.listdir(self.config.output_dir):
            stem = name.split(".", 1)[0]
            if JOB_ID_RE.match(stem) and (name.endswith(".glb") or name.endswith(".preview.png")):
                try:
                    os.remove(os.path.join(self.config.output_dir, name))
                except OSError:
                    pass

    # -- lifecycle ---------------------------------------------------------
    async def start(self) -> None:
        self._wakeup = asyncio.Event()
        self._remove_orphans()
        if self._consumer is None:
            self._consumer = asyncio.create_task(self._consume())
        if self._sweeper is None:
            self._sweeper = asyncio.create_task(self._sweep_forever())

    async def stop(self) -> None:
        for job in self.jobs.values():
            if job.status in ("queued", "in_progress"):
                job.cancel = True
        for task in (self._consumer, self._sweeper):
            if task:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        self._consumer = self._sweeper = None
        self._executor.shutdown(wait=False, cancel_futures=True)

    # -- public API --------------------------------------------------------
    def queue_depth(self) -> int:
        return len(self._pending)

    def has_capacity(self) -> bool:
        return len(self._pending) < self.config.max_queue

    def submit(self, spec: MeshSpec) -> Job:
        if not self.has_capacity():
            raise QueueFull()
        job = Job(id=f"mjob-{uuid.uuid4().hex[:24]}", spec=spec)
        self.jobs[job.id] = job
        self._pending.append(job.id)
        if self._wakeup is not None:
            self._wakeup.set()
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    def request_cancel(self, job_id: str) -> Optional[Job]:
        job = self.jobs.get(job_id)
        if job is None:
            return None
        if job.status in TERMINAL:
            return job          # idempotent no-op
        job.cancel = True
        if job.status == "queued":
            # Never started: cancel now and free its queue slot.
            try:
                self._pending.remove(job.id)
            except ValueError:
                pass
            self._finish(job, "cancelled")
        # in_progress: the engine stops at its next stage boundary.
        return job

    def is_stuck(self) -> bool:
        job = self.jobs.get(self._current) if self._current else None
        if job is None or job.started_at is None:
            return False
        limit = self.config.job_timeout_seconds + self.config.stuck_grace_seconds
        return time.time() - job.started_at > limit

    def health(self) -> Dict[str, Any]:
        return {
            "status": "stuck" if self.is_stuck() else "ok",
            "mode": self.config.mode,
            "model": self.config.model_id,
            "busy": self._current is not None,
            "current_job": self._current,
            "queue_depth": len(self._pending),
        }

    # -- internals ---------------------------------------------------------
    def _finish(self, job: Job, status: str, error: Optional[Dict[str, str]] = None) -> None:
        job.status = status
        job.error = error
        job.completed_at = time.time()
        if job.started_at is not None:
            job.duration_ms = int((job.completed_at - job.started_at) * 1000)
        if status == "completed":
            job.stage = "done"
            job.progress = 100.0
        job.spec = None   # drop the image bytes
        if status != "completed":
            self._remove_outputs(job.id)

    async def _consume(self) -> None:
        while True:
            while not self._pending:
                self._wakeup.clear()
                await self._wakeup.wait()
            job = self.jobs.get(self._pending.popleft())
            if job is None or job.status != "queued":
                continue
            try:
                await self._run(job)
            except asyncio.CancelledError:
                raise
            except Exception:  # never let one job kill the consumer
                log.exception("job %s crashed the runner", job.id)
                if job.status not in TERMINAL:
                    self._finish(job, "failed", {"code": "internal_error", "message": "worker error"})

    def _execute(self, job: Job) -> Dict[str, Any]:
        """Runs on the GPU executor thread."""
        spec = job.spec

        def report(stage: str, progress: float) -> None:
            if job.status in TERMINAL:   # e.g. already failed by the timeout
                return
            job.stage = stage
            job.progress = max(job.progress, round(min(float(progress), 99.0), 1))

        tracker = StageTracker(report, lambda: job.cancel)
        image = open_image(spec)
        glb_path, preview_path = self.glb_path(job.id), self.preview_path(job.id)
        result = self.engine.generate(spec, image, glb_path, preview_path, tracker) or {}
        # Engine-agnostic output checks before anything is served.
        with open(glb_path, "rb") as fh:
            stats = inspect_glb(fh.read())
        if not os.path.exists(preview_path) or not _is_png(preview_path):
            raise RuntimeError("engine did not write a PNG preview")
        outputs = {
            "glb": {"size_bytes": os.path.getsize(glb_path), "sha256": _sha256(glb_path)},
            "preview": {"size_bytes": os.path.getsize(preview_path), "sha256": _sha256(preview_path)},
        }
        tracker.check()   # a cancel that arrived during the last stage still wins
        return {"outputs": outputs, "faces": stats["faces"], "vertices": stats["vertices"],
                "engine": result}

    async def _run(self, job: Job) -> None:
        self._current = job.id
        job.status = "in_progress"
        job.stage = "preprocess"
        job.started_at = time.time()
        loop = asyncio.get_running_loop()
        fut = loop.run_in_executor(self._executor, self._execute, job)
        try:
            try:
                result = await asyncio.wait_for(asyncio.shield(fut), timeout=self.config.job_timeout_seconds)
            except asyncio.TimeoutError:
                job.cancel = True
                self._finish(job, "failed", {
                    "code": "timeout",
                    "message": f"job exceeded {int(self.config.job_timeout_seconds)} s",
                })
                log.warning("job %s timed out; waiting for the engine to stop", job.id)
                try:
                    await fut    # keep the GPU serialized
                except BaseException:
                    pass
                self._remove_outputs(job.id)
                return
            except asyncio.CancelledError:
                job.cancel = True
                raise
            except Cancelled:
                self._finish(job, "cancelled")
                return
            except EngineInputError as exc:
                self._finish(job, "failed", {"code": "invalid_input", "message": str(exc)[:500]})
                return
            except GLBError as exc:
                self._finish(job, "failed", {"code": "invalid_output", "message": f"engine wrote an invalid GLB: {exc}"[:500]})
                return
            except Exception as exc:
                log.exception("job %s failed", job.id)
                self._finish(job, "failed", {"code": "generation_failed", "message": str(exc)[:500]})
                return
            job.outputs = result["outputs"]
            job.faces = result["faces"]
            job.vertices = result["vertices"]
            self._finish(job, "completed")
        finally:
            self._current = None

    def sweep(self, now: Optional[float] = None) -> int:
        """Forget terminal jobs (and delete their files) older than the TTL."""
        now = time.time() if now is None else now
        ttl = self.config.result_ttl_seconds
        expired = [j.id for j in self.jobs.values()
                   if j.status in TERMINAL and j.completed_at is not None and now - j.completed_at >= ttl]
        for job_id in expired:
            self._remove_outputs(job_id)
            self.jobs.pop(job_id, None)
        return len(expired)

    async def _sweep_forever(self) -> None:
        while True:
            await asyncio.sleep(self.config.sweep_interval)
            try:
                self.sweep()
            except Exception:
                log.exception("result sweep failed")
