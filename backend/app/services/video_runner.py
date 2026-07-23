############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# video_runner.py: Gateway-side async runner for video jobs.
#
# Claims persisted queued jobs, routes each through the existing registry,
# drives the async worker (submit/poll/fetch), and never enters
# _proxy_with_retry — the LLM hot path is untouched. v1 handles single-shot,
# text-to-video jobs. See docs/video-generation-plan.md.
#
# The DB lives behind an injected repository (VideoJobRepo) so the state
# machine is testable without the db/pymysql chain (test_website_publisher.py
# injected-client pattern). The default CrudVideoJobRepo wraps crud.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Async video job runner (claim -> submit -> poll -> fetch -> complete)."""

import asyncio
import os
import uuid
from typing import Any, Dict, List, Optional, Protocol

from backend.app.logging_config import get_logger
from backend.app.services.video_store import job_output_path
from backend.app.services.video_worker_client import (
    FetchResult,
    HttpVideoWorkerClient,
    VideoWorkerClient,
    WorkerSubmitError,
)

logger = get_logger(__name__)


# ── Repository contract ───────────────────────────────────────────────────
# All returns are plain dicts (detached snapshots), never session-bound ORM
# objects — a job is processed across many awaits/polls, so nothing may hold a
# session open. Each method opens and closes its own short transaction.
class VideoJobRepo(Protocol):
    async def readopt_stale(self, threshold_seconds: int) -> int: ...

    async def claim_next(self, worker_id: str) -> Optional[Dict[str, Any]]: ...

    async def is_cancelled(self, job_id: int) -> bool: ...

    async def update_progress(self, job_id: int, progress: float) -> None: ...

    async def mark_shot(self, shot_id: int, **fields) -> None: ...

    async def requeue(self, job_id: int) -> None: ...

    async def mark_cancelled(self, job_id: int) -> None: ...

    async def store_output(
        self, job: Dict[str, Any], src_path: str, fetch: FetchResult
    ) -> int: ...

    async def complete(
        self,
        job_id: int,
        output_asset_id: int,
        *,
        duration_seconds: Optional[float],
        gpu_seconds: int,
        token_equivalent: Optional[int],
    ) -> None: ...

    async def fail(self, job_id: int, *, error_code: str, error_message: str) -> None: ...

    async def select_backend(self, model: str) -> Optional[Dict[str, Any]]: ...


# ── The runner ────────────────────────────────────────────────────────────
class VideoRunner:
    """Single-process video job runner. One instance per gateway; the backend
    row's max_concurrent=1 is the real GPU backpressure."""

    def __init__(
        self,
        repo: VideoJobRepo,
        worker: VideoWorkerClient,
        *,
        storage_root: str,
        worker_id: Optional[str] = None,
        poll_interval: float = 5.0,
        stale_threshold_seconds: int = 120,
        max_retries_per_shot: int = 2,
        token_cost_per_second: int = 2000,
    ):
        self.repo = repo
        self.worker = worker
        self.storage_root = storage_root
        self.worker_id = worker_id or f"runner-{uuid.uuid4().hex[:8]}"
        self.poll_interval = poll_interval
        self.stale_threshold_seconds = stale_threshold_seconds
        self.max_retries_per_shot = max_retries_per_shot
        self.token_cost_per_second = token_cost_per_second

    # -- lifecycle ---------------------------------------------------------
    async def run_forever(self) -> None:
        """Re-adopt crashed renders, then loop claiming and processing jobs."""
        try:
            n = await self.repo.readopt_stale(self.stale_threshold_seconds)
            if n:
                logger.info("video_runner_readopted", count=n)
        except Exception:
            logger.exception("video_runner_readopt_error")

        while True:
            try:
                did_work = await self.tick()
                if not did_work:
                    await asyncio.sleep(self.poll_interval)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("video_runner_tick_error")
                await asyncio.sleep(self.poll_interval)

    async def tick(self) -> bool:
        """Claim and process one job. Returns False if the queue was empty."""
        job = await self.repo.claim_next(self.worker_id)
        if job is None:
            return False
        await self.process_job(job)
        return True

    # -- one job -----------------------------------------------------------
    async def process_job(self, job: Dict[str, Any]) -> None:
        """Drive a single claimed job to a terminal state. v1: one shot."""
        job_id = job["id"]
        job_uuid = job["job_uuid"]

        try:
            if await self.repo.is_cancelled(job_id):
                await self.repo.mark_cancelled(job_id)
                return

            backend = await self.repo.select_backend(job["model"])
            if backend is None:
                # No healthy video backend right now — leave it queued and let a
                # later tick retry (bounded by the job's wall deadline elsewhere).
                await self.repo.requeue(job_id)
                logger.warning("video_runner_no_backend", job=job_uuid, model=job["model"])
                return

            # Submit (retryable transient failures re-queue the whole job).
            payload = self._build_payload(job)
            try:
                worker_job_id = await self.worker.submit(backend["url"], payload)
            except WorkerSubmitError as exc:
                await self._handle_submit_failure(job, exc)
                return

            await self.repo.mark_shot(
                job["shot_id"],
                status="rendering",
                backend_id=backend["id"],
                backend_job_id=worker_job_id,
            )

            # Poll to completion.
            await self._poll_to_completion(job, backend, worker_job_id)

        except Exception as exc:  # unexpected — surface, do not retry
            logger.exception("video_runner_job_error", job=job_uuid)
            await self.repo.fail(job_id, error_code="runner_error", error_message=str(exc))

    async def _poll_to_completion(
        self, job: Dict[str, Any], backend: Dict[str, Any], worker_job_id: str
    ) -> None:
        job_id = job["id"]
        while True:
            if await self.repo.is_cancelled(job_id):
                await self.worker.cancel(backend["url"], worker_job_id)
                await self.repo.mark_shot(job["shot_id"], status="skipped")
                await self.repo.mark_cancelled(job_id)
                return

            status = await self.worker.poll(backend["url"], worker_job_id)
            state = status.get("status")
            await self.repo.update_progress(job_id, float(status.get("progress", 0.0)))

            if state == "completed":
                await self._finalize(job, backend, worker_job_id)
                return
            if state == "failed":
                # A render that started then failed is surfaced, never retried.
                err = status.get("error") or {}
                await self.repo.mark_shot(job["shot_id"], status="failed")
                await self.repo.fail(
                    job_id,
                    error_code=err.get("code", "render_failed"),
                    error_message=err.get("message", "worker reported failure"),
                )
                return

            await asyncio.sleep(self.poll_interval)

    async def _finalize(
        self, job: Dict[str, Any], backend: Dict[str, Any], worker_job_id: str
    ) -> None:
        """Fetch the artifact, store it, and complete the job."""
        dest = job_output_path(self.storage_root, job["user_id"], job["job_uuid"])
        fetch = await self.worker.fetch(backend["url"], worker_job_id, dest)
        asset_id = await self.repo.store_output(job, dest, fetch)

        duration = float(job.get("seconds") or 0)
        gpu_seconds = int(round(fetch.duration_ms / 1000)) if fetch.duration_ms else 0
        # Keep the amount reserved at submit (charged with quality/resolution
        # multipliers); only fall back to a flat estimate if none was reserved.
        token_equivalent = job.get("token_equivalent") or int(duration * self.token_cost_per_second)

        await self.repo.mark_shot(
            job["shot_id"], status="rendered", output_asset_id=asset_id
        )
        await self.repo.complete(
            job["id"],
            asset_id,
            duration_seconds=duration,
            gpu_seconds=gpu_seconds,
            token_equivalent=token_equivalent,
        )
        logger.info("video_runner_completed", job=job["job_uuid"], size=fetch.size_bytes)

    async def _handle_submit_failure(self, job: Dict[str, Any], exc: WorkerSubmitError) -> None:
        """Transient submit failure re-queues the shot up to the cap; otherwise
        the job fails. A started-then-failed render never reaches here."""
        attempts = int(job.get("attempts", 0)) + 1
        await self.repo.mark_shot(job["shot_id"], attempts=attempts)
        if exc.retryable and attempts <= self.max_retries_per_shot:
            await self.repo.requeue(job["id"])
            logger.warning(
                "video_runner_submit_retry", job=job["job_uuid"], attempt=attempts, error=str(exc)
            )
        else:
            await self.repo.mark_shot(job["shot_id"], status="failed")
            await self.repo.fail(
                job["id"], error_code="submit_failed", error_message=str(exc)
            )

    def _build_payload(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """Worker request body for a single text-to-video shot."""
        payload: Dict[str, Any] = {
            "model": job["model"],
            "prompt": job["prompt"],
            "size": job["size"],
            "seconds": str(job["seconds"]),
            "fps": job["fps"],
            "quality": job["quality"],
        }
        if job.get("seed") is not None:
            payload["seed"] = job["seed"]
        return payload


# ── Default crud-backed repository ────────────────────────────────────────
class CrudVideoJobRepo:
    """Production repo: opens a short session per operation via crud."""

    async def readopt_stale(self, threshold_seconds: int) -> int:
        from backend.app.db import crud
        from backend.app.db.session import get_async_db_context

        async with get_async_db_context() as db:
            return await crud.readopt_stale_video_jobs(db, threshold_seconds)

    async def claim_next(self, worker_id: str) -> Optional[Dict[str, Any]]:
        from backend.app.db import crud
        from backend.app.db.session import get_async_db_context

        async with get_async_db_context() as db:
            job = await crud.claim_next_video_job(db, worker_id)
            if job is None:
                return None
            # Load project + single shot for the worker payload (v1 = one shot).
            proj = await crud.get_video_project(db, job.project_id)
            shots = await crud.get_video_shots(db, job.id)
            shot = shots[0] if shots else None
            quality = proj.quality if proj else None
            return {
                "id": job.id,
                "job_uuid": job.job_uuid,
                "user_id": job.user_id,
                "project_id": job.project_id,
                "model": proj.model if proj else None,
                "size": proj.size if proj else None,
                "fps": proj.fps if proj else 24,
                "quality": getattr(quality, "value", quality) or "standard",
                "shot_id": shot.id if shot else None,
                "prompt": shot.prompt if shot else None,
                "seconds": shot.seconds if shot else None,
                "seed": shot.seed if shot else None,
                "attempts": shot.attempts if shot else 0,
                "token_equivalent": job.token_equivalent,
            }

    async def is_cancelled(self, job_id: int) -> bool:
        from backend.app.db import crud
        from backend.app.db.session import get_async_db_context

        async with get_async_db_context() as db:
            job = await crud.get_video_job_by_id(db, job_id) if hasattr(crud, "get_video_job_by_id") else None
            return bool(job and job.cancel_requested)

    async def update_progress(self, job_id: int, progress: float) -> None:
        from backend.app.db import crud
        from backend.app.db.session import get_async_db_context

        async with get_async_db_context() as db:
            job = await crud.get_video_job_by_id(db, job_id)
            if job:
                await crud.update_video_job_progress(db, job, progress)

    async def mark_shot(self, shot_id: int, **fields) -> None:
        from backend.app.db import crud
        from backend.app.db.session import get_async_db_context

        async with get_async_db_context() as db:
            shot = await crud.get_video_shot_by_id(db, shot_id)
            if shot:
                await crud.update_video_shot(db, shot, **fields)

    async def requeue(self, job_id: int) -> None:
        from backend.app.db import crud
        from backend.app.db.models import VideoJobStatus
        from backend.app.db.session import get_async_db_context

        async with get_async_db_context() as db:
            job = await crud.get_video_job_by_id(db, job_id)
            if job:
                job.status = VideoJobStatus.QUEUED
                job.claimed_by = None
                await db.commit()

    async def mark_cancelled(self, job_id: int) -> None:
        from datetime import datetime, timezone

        from backend.app.db import crud
        from backend.app.db.models import VideoJobStatus
        from backend.app.db.session import get_async_db_context

        async with get_async_db_context() as db:
            job = await crud.get_video_job_by_id(db, job_id)
            if job:
                job.status = VideoJobStatus.CANCELLED
                job.completed_at = datetime.now(timezone.utc)
                if job.token_equivalent:
                    await crud.refund_video_tokens(db, job.user_id, job.token_equivalent)
                await db.commit()

    async def store_output(self, job: Dict[str, Any], src_path: str, fetch: FetchResult) -> int:
        from backend.app.db import crud
        from backend.app.db.models import VideoAssetKind
        from backend.app.db.session import get_async_db_context

        async with get_async_db_context() as db:
            asset = await crud.create_video_asset(
                db,
                user_id=job["user_id"],
                project_id=job["project_id"],
                kind=VideoAssetKind.FINAL,
                storage_path=src_path,
                content_type="video/mp4",
                sha256=fetch.sha256,
                size_bytes=fetch.size_bytes,
                duration_ms=fetch.duration_ms,
            )
            await db.commit()
            return asset.id

    async def complete(
        self,
        job_id: int,
        output_asset_id: int,
        *,
        duration_seconds: Optional[float],
        gpu_seconds: int,
        token_equivalent: Optional[int],
    ) -> None:
        from backend.app.db import crud
        from backend.app.db.session import get_async_db_context

        async with get_async_db_context() as db:
            job = await crud.get_video_job_by_id(db, job_id)
            if job:
                await crud.complete_video_job(
                    db, job, output_asset_id=output_asset_id,
                    duration_seconds=duration_seconds, gpu_seconds=gpu_seconds,
                    token_equivalent=token_equivalent,
                )

    async def fail(self, job_id: int, *, error_code: str, error_message: str) -> None:
        from backend.app.db import crud
        from backend.app.db.session import get_async_db_context

        async with get_async_db_context() as db:
            job = await crud.get_video_job_by_id(db, job_id)
            if job:
                await crud.fail_video_job(db, job, error_code=error_code, error_message=error_message)

    async def select_backend(self, model: str) -> Optional[Dict[str, Any]]:
        from backend.app.core.telemetry.registry import get_registry
        from backend.app.db.models import Modality

        registry = get_registry()
        backends = await registry.get_backends_with_model(model, Modality.VIDEO_GENERATION)
        for b in backends:
            return {"id": b.id, "url": b.url}
        return None


# ── Lifespan entry point ──────────────────────────────────────────────────
async def run_video_runner_loop() -> None:
    """Background task started from main.py lifespan when video_runner_enabled."""
    from backend.app.settings import get_settings

    settings = get_settings()
    if not settings.video_runner_enabled:
        logger.info("video_runner_disabled")
        return

    runner = VideoRunner(
        repo=CrudVideoJobRepo(),
        worker=HttpVideoWorkerClient(
            control_timeout=settings.video_worker_timeout_seconds,
            fetch_timeout=settings.video_worker_fetch_timeout_seconds,
        ),
        storage_root=settings.video_storage_path,
        poll_interval=settings.video_runner_poll_interval_seconds,
        stale_threshold_seconds=settings.video_job_stale_heartbeat_seconds,
    )
    logger.info("video_runner_started", worker_id=runner.worker_id)
    await runner.run_forever()
