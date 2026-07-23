############################################################
#
# mindrouter video-worker - LTX-2.3 async video generation service
#
# worker.py: JobManager — in-memory job store + single-slot serialized
#     execution that runs the blocking generation OFF the event loop, so
#     GET /health answers in <5s while a render is in flight.
#
# max_concurrent=1 on the gateway backend row is the real backpressure; the
# worker serializes anyway because one diffusion request saturates the GPU.
#
# Luke Sheneman — University of Idaho RCDS — sheneman@uidaho.edu
#
############################################################

"""Async job manager for the video worker."""

import asyncio
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from engine import Cancelled, VideoEngine


@dataclass
class Job:
    id: str
    spec: Dict[str, Any]
    status: str = "queued"          # queued|in_progress|completed|failed|cancelled
    progress: float = 0.0           # 0-100
    step: int = 0
    total_steps: int = 0
    error: Optional[Dict[str, str]] = None
    output_path: Optional[str] = None
    duration_ms: Optional[int] = None
    cancel: bool = False
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


class JobManager:
    """Serialized, off-event-loop job execution."""

    def __init__(self, engine: VideoEngine, output_dir: str):
        self.engine = engine
        self.output_dir = output_dir
        self.jobs: Dict[str, Job] = {}
        self._queue: "asyncio.Queue[str]" = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None
        self._current: Optional[str] = None
        os.makedirs(output_dir, exist_ok=True)

    # -- lifecycle ---------------------------------------------------------
    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._consume())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    # -- public API --------------------------------------------------------
    def submit(self, spec: Dict[str, Any]) -> Job:
        job = Job(id=f"wjob-{uuid.uuid4().hex[:20]}", spec=spec)
        job.output_path = os.path.join(self.output_dir, f"{job.id}.mp4")
        self.jobs[job.id] = job
        self._queue.put_nowait(job.id)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    def request_cancel(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if not job:
            return False
        job.cancel = True
        # A still-queued job can be cancelled immediately; an in-flight one is
        # cancelled cooperatively by the engine between steps.
        if job.status == "queued":
            job.status = "cancelled"
            job.completed_at = time.time()
        return True

    def health(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "current_job": self._current,
            "queue_depth": self._queue.qsize(),
        }

    # -- internals ---------------------------------------------------------
    async def _consume(self) -> None:
        while True:
            job_id = await self._queue.get()
            job = self.jobs.get(job_id)
            if job is None or job.cancel:
                if job and job.status == "queued":
                    job.status = "cancelled"
                continue
            await self._run(job)

    async def _run(self, job: Job) -> None:
        self._current = job.id
        job.status = "in_progress"
        job.started_at = time.time()
        loop = asyncio.get_running_loop()

        def progress_cb(step: int, total: int) -> None:
            job.step = step
            job.total_steps = total
            job.progress = round(step / total * 100.0, 1) if total else 0.0

        def should_cancel() -> bool:
            return job.cancel

        try:
            # Run the blocking generation in a thread so the event loop (and
            # GET /health) stays responsive. torch releases the GIL during CUDA.
            result = await loop.run_in_executor(
                None, self.engine.generate, job.spec, job.output_path, progress_cb, should_cancel
            )
            job.duration_ms = (result or {}).get("duration_ms")
            job.status = "completed"
            job.progress = 100.0
        except Cancelled:
            job.status = "cancelled"
            self._safe_unlink(job.output_path)
        except Exception as exc:  # surface render failures
            job.status = "failed"
            job.error = {"code": "render_failed", "message": str(exc)[:500]}
            self._safe_unlink(job.output_path)
        finally:
            job.completed_at = time.time()
            self._current = None

    @staticmethod
    def _safe_unlink(path: Optional[str]) -> None:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except OSError:
            pass
