############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# mesh_runner.py: Gateway-side async runner for image-to-mesh
#     jobs. Claims persisted QUEUED jobs, picks a healthy mesh
#     backend through the registry, optionally cuts the subject
#     out with the matting server, drives the mesh-worker by
#     submit/poll/fetch/cancel, and stores the GLB + preview.
#
# The same design as video_runner (whose comments explain the
# why): a Redis-leased single leader, an injected repository so
# the state machine is testable without the DB, re-adoption of
# jobs whose runner died, and never entering _proxy_with_retry.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
#
############################################################

"""Async mesh job runner (claim -> cut out -> submit -> poll -> fetch -> complete)."""

import asyncio
import base64
import io
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Coroutine, Dict, List, Optional, Protocol, Tuple

import httpx

from backend.app.logging_config import get_logger
from backend.app.services import mesh_store
from backend.app.services.mesh_worker_client import (
    VARIANT_GLB,
    VARIANT_PREVIEW,
    HttpMeshWorkerClient,
    MeshWorkerClient,
    WorkerFetchError,
    WorkerSubmitError,
)

logger = get_logger(__name__)

# Optional background-removal step: image bytes in, RGBA PNG bytes out, or
# None to send the original unchanged (fail open).
Cutout = Callable[[bytes], Awaitable[Optional[bytes]]]

# Consecutive poll failures (other than 404) tolerated before the job is
# treated as lost by its worker.
MAX_CONSECUTIVE_POLL_ERRORS = 5

# Pictures are shrunk to this many pixels on the long side before the cut-out
# and the submit: the models read far smaller inputs (TRELLIS.2 518-1024 px,
# Hunyuan3D 512 px), and a full-size PNG of a 24 MP photo would exceed the
# inference nodes' 64 MB request limit once base64-encoded.
MAX_INPUT_SIDE = 2048


class MeshJobRepo(Protocol):
    """Everything the runner needs from the database. Every return is a plain
    dict (a detached snapshot); every method opens its own short session.
    State changes are conditional on the job still being active, so a job
    reaches a terminal state, and is refunded, at most once."""

    async def list_stale_rendering(self, threshold_seconds: int) -> List[Dict[str, Any]]: ...
    async def readopt(self, job_id: int, worker_id: str, threshold_seconds: int) -> bool: ...
    async def get_backend(self, backend_id: int) -> Optional[Dict[str, Any]]: ...
    async def servable_models(self) -> List[str]: ...
    async def claim_next(self, worker_id: str, models: Optional[List[str]] = None) -> Optional[Dict[str, Any]]: ...
    async def is_cancelled(self, job_id: int) -> bool: ...
    async def update(self, job_id: int, **fields) -> None: ...
    async def requeue(self, job_id: int, *, attempts: Optional[int] = None, clear_backend: bool = False) -> None: ...
    async def mark_cancelled(self, job_id: int) -> None: ...
    async def complete(self, job_id: int, **fields) -> bool: ...
    async def fail(self, job_id: int, *, error_code: str, error_message: str) -> None: ...
    async def select_backend(self, model: str) -> Optional[Dict[str, Any]]: ...


def _alpha_is_used(img) -> bool:
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        low, _high = img.convert("RGBA").getchannel("A").getextrema()
        return low < 255
    return False


def has_transparency(image_bytes: bytes) -> bool:
    """True when the picture already carries a real alpha channel (some pixel
    is not fully opaque), i.e. the subject is already cut out. CPU-bound:
    call it off the event loop."""
    from PIL import Image

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            return _alpha_is_used(img)
    except Exception:
        return False


def prepare_image(image_bytes: bytes, max_side: int = MAX_INPUT_SIDE) -> Tuple[bytes, bool]:
    """(picture to send, whether it already has transparency). A picture
    larger than ``max_side`` is shrunk and re-encoded as PNG; a smaller one is
    sent unchanged. CPU-bound: call it off the event loop."""
    from PIL import Image

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            transparent = _alpha_is_used(img)
            if max(img.size) <= max_side:
                return image_bytes, transparent
            out = img.convert("RGBA" if transparent else "RGB")
            out.thumbnail((max_side, max_side), Image.LANCZOS)
            buf = io.BytesIO()
            out.save(buf, format="PNG")
            return buf.getvalue(), transparent
    except Exception:
        return image_bytes, False


class MeshRunner:
    """One runner per gateway (elected by lease). Each model is served by its
    own single-GPU worker, so the runner drives at most one job per model at a
    time: jobs for different models run side by side, while a model's own
    jobs wait their turn in the queue (its worker would only queue them)."""

    def __init__(
        self,
        repo: MeshJobRepo,
        worker: MeshWorkerClient,
        *,
        storage_root: str,
        cutout: Optional[Cutout] = None,
        worker_id: Optional[str] = None,
        poll_interval: float = 3.0,
        stale_threshold_seconds: int = 120,
        max_retries: int = 2,
        reconcile_interval: float = 20.0,
        max_wall_seconds: int = 1800,
        retry_backoff_seconds: float = 15.0,
        max_input_side: int = MAX_INPUT_SIDE,
    ):
        self.repo = repo
        self.worker = worker
        self.storage_root = storage_root
        self.cutout = cutout
        self.worker_id = worker_id or f"mesh-runner-{uuid.uuid4().hex[:8]}"
        self.poll_interval = poll_interval
        self.stale_threshold_seconds = stale_threshold_seconds
        self.max_retries = max_retries
        self.reconcile_interval = reconcile_interval
        self.max_wall_seconds = max_wall_seconds
        self.retry_backoff_seconds = retry_backoff_seconds
        self.max_input_side = max_input_side
        # Jobs this runner is driving right now: id -> model. Reconcile never
        # touches them (their heartbeat may lapse during a slow call), and
        # their models are not claimed again until they finish.
        self._active: Dict[int, str] = {}
        self._tasks: set = set()

    @property
    def busy_models(self) -> set:
        return set(self._active.values())

    # -- lifecycle -----------------------------------------------------------
    async def run_forever(self) -> None:
        import time

        await self._safe_reconcile()
        last_reconcile = time.monotonic()
        try:
            while True:
                try:
                    if time.monotonic() - last_reconcile >= self.reconcile_interval:
                        await self._safe_reconcile()
                        last_reconcile = time.monotonic()
                    job = await self.claim()
                    if job is None:
                        await asyncio.sleep(self.poll_interval)
                        continue
                    self.start(job)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("mesh_runner_tick_error")
                    await asyncio.sleep(self.poll_interval)
        except asyncio.CancelledError:
            pass
        finally:
            for task in list(self._tasks):
                task.cancel()

    async def claim(self) -> Optional[Dict[str, Any]]:
        """Claim the next job for a servable model that is not already busy."""
        busy = self.busy_models
        models = [m for m in await self.repo.servable_models() if m not in busy]
        if not models:
            return None
        return await self.repo.claim_next(self.worker_id, models=models)

    def start(self, job: Dict[str, Any]) -> "asyncio.Task":
        """Drive a claimed job in its own task."""
        return self._spawn(job, self.process_job(job))

    def _spawn(self, job: Dict[str, Any], work: Coroutine) -> "asyncio.Task":
        self._active[job["id"]] = job["model"]
        task = asyncio.create_task(self._drive(job, work))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def _drive(self, job: Dict[str, Any], work: Coroutine) -> None:
        try:
            pause = await work
            if pause:
                # The job went back to the queue: hold its model for a while so
                # the loop does not claim it again at once.
                await asyncio.sleep(pause)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("mesh_runner_job_error", job=job.get("job_uuid"))
            await self.repo.fail(job["id"], error_code="runner_error", error_message=type(exc).__name__)
        finally:
            self._active.pop(job["id"], None)

    async def _safe_reconcile(self) -> None:
        try:
            await self.reconcile()
        except Exception:
            logger.exception("mesh_runner_reconcile_error")

    # -- reconciliation --------------------------------------------------------
    async def reconcile(self) -> int:
        """Adopt every stale RENDERING job whose runner is gone and drive it to
        a correct state from the worker's ground truth. Jobs this runner is
        driving are skipped. Returns how many were looked at."""
        stale = [
            j for j in await self.repo.list_stale_rendering(self.stale_threshold_seconds)
            if j["id"] not in self._active
        ]
        for job in stale:
            try:
                await self.reconcile_job(job)
            except Exception:
                logger.exception("mesh_runner_reconcile_job_error", job=job.get("job_uuid"))
        return len(stale)

    async def reconcile_job(self, job: Dict[str, Any]) -> None:
        job_id = job["id"]
        backend_id, backend_job_id = job.get("backend_id"), job.get("backend_job_id")
        backend = await self.repo.get_backend(backend_id) if backend_id else None
        if self._over_wall(job):
            if await self.repo.readopt(job_id, self.worker_id, self.stale_threshold_seconds):
                if backend and backend_job_id:
                    await self.worker.cancel(backend["url"], backend_job_id)
                await self.repo.fail(
                    job_id, error_code="wall_timeout",
                    error_message=f"job exceeded the {self.max_wall_seconds}s time limit",
                )
            return
        if not backend_job_id or backend is None:
            # Died before or during submit, or the backend is gone: requeue.
            if await self.repo.readopt(job_id, self.worker_id, self.stale_threshold_seconds):
                await self.repo.requeue(job_id, clear_backend=True)
            return
        status: Optional[Dict[str, Any]] = None
        try:
            status = await self.worker.poll(backend["url"], backend_job_id)
        except httpx.HTTPStatusError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                if await self.repo.readopt(job_id, self.worker_id, self.stale_threshold_seconds):
                    await self._lost(job, "the worker no longer knows the job")
                return
        except Exception:
            pass  # unreachable for now: resume polling, which tolerates errors
        if not await self.repo.readopt(job_id, self.worker_id, self.stale_threshold_seconds):
            return  # another runner took it
        logger.info("mesh_reconcile_adopted", job=job.get("job_uuid"),
                    worker_status=(status or {}).get("status"))
        self._spawn(job, self._continue(job, backend, backend_job_id, status))

    async def _continue(
        self, job: Dict[str, Any], backend: Dict[str, Any], worker_job_id: str,
        status: Optional[Dict[str, Any]],
    ) -> Optional[float]:
        if status is not None and await self._handle_status(job, backend, worker_job_id, status):
            return None
        return await self._poll_to_completion(job, backend, worker_job_id)

    def _over_wall(self, job: Dict[str, Any]) -> bool:
        started = job.get("started_at")
        if started is None:
            return False
        try:
            if started.tzinfo is None:
                started = started.replace(tzinfo=timezone.utc)
            return (datetime.now(timezone.utc) - started).total_seconds() > self.max_wall_seconds
        except Exception:
            return False

    def _backoff(self, attempts: int) -> float:
        return min(120.0, self.retry_backoff_seconds * max(1, attempts))

    async def _lost(
        self, job: Dict[str, Any], reason: str,
        backend: Optional[Dict[str, Any]] = None, worker_job_id: Optional[str] = None,
    ) -> Optional[float]:
        """The worker lost the job (or stopped answering). Cancel any copy it may
        still run, then requeue under the retry cap (it re-runs from scratch),
        otherwise fail and refund. Returns the pause before the next claim."""
        if backend and worker_job_id:
            await self.worker.cancel(backend["url"], worker_job_id)
        attempts = int(job.get("attempts") or 0) + 1
        if attempts <= self.max_retries:
            await self.repo.requeue(job["id"], attempts=attempts, clear_backend=True)
            logger.warning("mesh_job_requeued_lost", job=job.get("job_uuid"), attempt=attempts)
            return self._backoff(attempts)
        await self.repo.fail(
            job["id"], error_code="worker_lost",
            error_message=f"the worker lost the job after {attempts} attempt(s): {reason}",
        )
        return None

    # -- one job ---------------------------------------------------------------
    async def tick(self) -> bool:
        """Claim and process one job to the end, sequentially (run_forever uses
        claim() + start()). False when nothing was claimed or the job went
        back to the queue."""
        job = await self.claim()
        if job is None:
            return False
        return await self.process_job(job) is None

    async def process_job(self, job: Dict[str, Any]) -> Optional[float]:
        """Drive one claimed job to a terminal state. Returns None when done, or
        the seconds to hold the model back when the job went back to the queue."""
        job_id, job_uuid = job["id"], job["job_uuid"]
        try:
            if await self.repo.is_cancelled(job_id):
                await self.repo.mark_cancelled(job_id)
                return None
            backend = await self.repo.select_backend(job["model"])
            if backend is None:
                await self.repo.requeue(job_id)
                logger.warning("mesh_runner_no_backend", job=job_uuid, model=job["model"])
                return self.poll_interval

            await self.repo.update(job_id, stage="preprocess", progress=1.0)
            try:
                payload = await self.build_payload(job)
            except FileNotFoundError:
                await self.repo.fail(
                    job_id, error_code="input_missing",
                    error_message="the uploaded image is no longer on disk",
                )
                return None

            try:
                worker_job_id = await self.worker.submit(backend["url"], payload)
            except WorkerSubmitError as exc:
                return await self._submit_failed(job, exc)

            await self.repo.update(
                job_id, backend_id=backend["id"], backend_job_id=worker_job_id, stage="queued"
            )
            return await self._poll_to_completion(job, backend, worker_job_id)
        except Exception as exc:
            logger.exception("mesh_runner_job_error", job=job_uuid)
            await self.repo.fail(job_id, error_code="runner_error", error_message=type(exc).__name__)
        return None

    async def _submit_failed(self, job: Dict[str, Any], exc: WorkerSubmitError) -> Optional[float]:
        attempts = int(job.get("attempts") or 0) + 1
        if exc.retryable and attempts <= self.max_retries:
            await self.repo.requeue(job["id"], attempts=attempts)
            logger.warning("mesh_runner_submit_retry", job=job["job_uuid"], attempt=attempts, error=str(exc))
            return self._backoff(attempts)
        await self.repo.fail(job["id"], error_code="submit_failed", error_message=str(exc))
        return None

    async def _poll_to_completion(
        self, job: Dict[str, Any], backend: Dict[str, Any], worker_job_id: str
    ) -> Optional[float]:
        errors = 0
        while True:
            if await self.repo.is_cancelled(job["id"]):
                await self.worker.cancel(backend["url"], worker_job_id)
                await self.repo.mark_cancelled(job["id"])
                return None
            if self._over_wall(job):
                await self.worker.cancel(backend["url"], worker_job_id)
                await self.repo.fail(
                    job["id"], error_code="wall_timeout",
                    error_message=f"job exceeded the {self.max_wall_seconds}s time limit",
                )
                return None
            try:
                status = await self.worker.poll(backend["url"], worker_job_id)
                errors = 0
            except httpx.HTTPStatusError as exc:
                if exc.response is not None and exc.response.status_code == 404:
                    return await self._lost(job, "worker returned 404")
                errors += 1
            except Exception:
                errors += 1
            else:
                if await self._handle_status(job, backend, worker_job_id, status):
                    return None
            if errors >= MAX_CONSECUTIVE_POLL_ERRORS:
                return await self._lost(
                    job, f"{errors} consecutive poll failures", backend=backend, worker_job_id=worker_job_id
                )
            # Heartbeat every round, even when the worker did not answer, so a
            # slow worker is never mistaken for a dead runner.
            await self.repo.update(job["id"])
            await asyncio.sleep(self.poll_interval)

    async def _handle_status(
        self, job: Dict[str, Any], backend: Dict[str, Any], worker_job_id: str,
        status: Dict[str, Any],
    ) -> bool:
        """Apply one worker status. True when the job reached a terminal state."""
        state = status.get("status")
        if state == "completed":
            await self._finalize(job, backend, worker_job_id, status)
            return True
        if state == "failed":
            err = status.get("error") or {}
            await self.repo.fail(
                job["id"], error_code=str(err.get("code") or "generation_failed"),
                error_message=str(err.get("message") or "the worker reported a failure"),
            )
            return True
        if state == "cancelled":
            await self.repo.mark_cancelled(job["id"])
            return True
        try:
            progress = float(status.get("progress") or 0.0)
        except (TypeError, ValueError):
            progress = 0.0
        await self.repo.update(job["id"], progress=max(1.0, min(progress, 99.0)), stage=status.get("stage"))
        return False

    async def _heartbeat(self, job_id: int) -> None:
        """Keep a job's heartbeat fresh during a long download."""
        interval = max(1.0, min(30.0, self.stale_threshold_seconds / 4))
        while True:
            await asyncio.sleep(interval)
            try:
                await self.repo.update(job_id)
            except Exception:
                logger.warning("mesh_runner_heartbeat_failed", job_id=job_id)

    async def _finalize(
        self, job: Dict[str, Any], backend: Dict[str, Any], worker_job_id: str, status: Dict[str, Any]
    ) -> None:
        """Fetch the GLB (required) and preview (best effort), then complete."""
        root, user_id, job_uuid = self.storage_root, job["user_id"], job["job_uuid"]
        await self.repo.update(job["id"], progress=99.0, stage="export")
        glb_path = mesh_store.output_path(root, user_id, job_uuid, create=True)
        preview_path = mesh_store.preview_path(root, user_id, job_uuid)
        beat = asyncio.create_task(self._heartbeat(job["id"]))
        try:
            try:
                glb = await self.worker.fetch(backend["url"], worker_job_id, VARIANT_GLB, glb_path)
            except (WorkerFetchError, httpx.HTTPError, OSError) as exc:
                await self.repo.fail(job["id"], error_code="fetch_failed", error_message=str(exc)[:300])
                return
            preview_size = 0
            try:
                preview = await self.worker.fetch(backend["url"], worker_job_id, VARIANT_PREVIEW, preview_path)
                preview_size = preview.size_bytes
            except Exception as exc:  # a missing preview never fails the job
                logger.warning("mesh_runner_preview_unavailable", job=job_uuid, error=type(exc).__name__)
        finally:
            beat.cancel()

        def _int(key: str) -> Optional[int]:
            try:
                value = status.get(key)
                return int(value) if value is not None else None
            except (TypeError, ValueError):
                return None

        duration_ms = _int("duration_ms") or 0
        completed = await self.repo.complete(
            job["id"],
            output_size_bytes=glb.size_bytes,
            output_sha256=glb.sha256,
            preview_size_bytes=preview_size,
            faces=_int("faces"),
            vertices=_int("vertices"),
            gpu_seconds=int(round(duration_ms / 1000)),
        )
        if completed is False:
            # Cancelled, failed or deleted meanwhile: the outputs belong to no
            # job (nothing counts them against storage), so remove them.
            for path in (glb_path, preview_path):
                try:
                    os.remove(path)
                except OSError:
                    pass
            logger.info("mesh_job_outputs_discarded", job=job_uuid)
            return
        logger.info("mesh_job_completed", job=job_uuid, size=glb.size_bytes)

    # -- payload ---------------------------------------------------------------
    async def build_payload(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """The worker request. Reads the stored input, shrinks it to
        ``max_input_side``, and, when asked (or on "auto" for an opaque
        picture), cuts the subject out first. Decoding runs off the loop."""
        path = mesh_store.input_path(
            self.storage_root, job["user_id"], job["job_uuid"], job.get("input_content_type")
        )
        with open(path, "rb") as fh:
            image = fh.read()
        image, transparent = await asyncio.to_thread(prepare_image, image, self.max_input_side)
        mode = (job.get("remove_background") or "auto").lower()
        want_cutout = mode == "on" or (mode == "auto" and not transparent)
        if want_cutout and self.cutout is not None:
            try:
                cut = await self.cutout(image)
            except Exception as exc:  # fail open: the worker can still run
                logger.warning("mesh_runner_cutout_failed", job=job["job_uuid"], error=type(exc).__name__)
                cut = None
            if cut:
                image = cut
        payload: Dict[str, Any] = {
            "model": job["model"],
            "image": base64.b64encode(image).decode("ascii"),
            "texture": bool(job.get("texture", True)),
            "texture_size": int(job.get("texture_size") or 2048),
            "target_faces": int(job.get("target_faces") or 100000),
        }
        if job.get("seed") is not None:
            payload["seed"] = int(job["seed"])
        return payload


# ── Production repository ─────────────────────────────────────────────────────
class CrudMeshJobRepo:
    """Opens a short session per operation via crud."""

    @staticmethod
    def _ctx():
        from backend.app.db.session import get_async_db_context

        return get_async_db_context()

    async def list_stale_rendering(self, threshold_seconds: int) -> List[Dict[str, Any]]:
        from backend.app.db import crud

        async with self._ctx() as db:
            return await crud.list_stale_rendering_mesh_jobs(db, threshold_seconds)

    async def readopt(self, job_id: int, worker_id: str, threshold_seconds: int) -> bool:
        from backend.app.db import crud

        async with self._ctx() as db:
            return await crud.readopt_mesh_job(db, job_id, worker_id, threshold_seconds)

    async def get_backend(self, backend_id: int) -> Optional[Dict[str, Any]]:
        from backend.app.db import crud

        async with self._ctx() as db:
            return await crud.get_backend_snapshot(db, backend_id)

    async def servable_models(self) -> List[str]:
        from backend.app.db import crud

        async with self._ctx() as db:
            return await crud.get_servable_mesh_models(db)

    async def claim_next(self, worker_id: str, models: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        from backend.app.db import crud

        async with self._ctx() as db:
            job = await crud.claim_next_mesh_job(db, worker_id, models=models)
            return crud.mesh_job_snapshot(job) if job else None

    async def is_cancelled(self, job_id: int) -> bool:
        from backend.app.db import crud

        async with self._ctx() as db:
            return await crud.is_mesh_job_cancel_requested(db, job_id)

    async def update(self, job_id: int, **fields) -> None:
        from backend.app.db import crud

        async with self._ctx() as db:
            await crud.update_mesh_job(db, job_id, **fields)

    async def requeue(self, job_id: int, *, attempts: Optional[int] = None, clear_backend: bool = False) -> None:
        from backend.app.db import crud

        async with self._ctx() as db:
            await crud.requeue_mesh_job(db, job_id, attempts=attempts, clear_backend=clear_backend)

    async def mark_cancelled(self, job_id: int) -> None:
        from backend.app.db import crud

        async with self._ctx() as db:
            await crud.mark_mesh_job_cancelled(db, job_id)

    async def complete(self, job_id: int, **fields) -> bool:
        from backend.app.db import crud

        async with self._ctx() as db:
            return await crud.complete_mesh_job(db, job_id, **fields)

    async def fail(self, job_id: int, *, error_code: str, error_message: str) -> None:
        from backend.app.db import crud

        async with self._ctx() as db:
            await crud.fail_mesh_job(db, job_id, error_code=error_code, error_message=error_message)

    async def select_backend(self, model: str) -> Optional[Dict[str, Any]]:
        from backend.app.core.telemetry.registry import get_registry
        from backend.app.db.models import Modality

        backends = await get_registry().get_backends_with_model(model, Modality.MESH_GENERATION)
        for b in backends:
            return {"id": b.id, "url": b.url}
        return None


async def matting_cutout(image: bytes) -> Optional[bytes]:
    """Cut the subject out with the fleet's matting server (img.matting_url),
    gated by mesh.background_removal rather than the images feature's own
    switch. Returns RGBA PNG bytes, or None to send the original."""
    from backend.app.db import crud
    from backend.app.db.session import get_async_db_context
    from backend.app.services import image_matting

    async with get_async_db_context() as db:
        if not await crud.get_config_json(db, "mesh.background_removal", True):
            return None
        config = await image_matting.load_config(db)
    import dataclasses

    # mesh's own switch was checked above; img.transparent_enabled governs only
    # the images feature.
    config = dataclasses.replace(config, enabled=True)
    if not config.usable:
        return None
    b64 = base64.b64encode(image).decode("ascii")
    out_b64, outcome = await image_matting.make_transparent(b64, config)
    if outcome != image_matting.OUTCOME_TRANSPARENT:
        return None
    return base64.b64decode(out_b64)


# ── Lifespan entry point ──────────────────────────────────────────────────────
_RUNNER_LEASE_KEY = "mesh:runner:leader"


def _build_runner(settings) -> MeshRunner:
    return MeshRunner(
        repo=CrudMeshJobRepo(),
        worker=HttpMeshWorkerClient(
            control_timeout=settings.mesh_worker_timeout_seconds,
            fetch_timeout=settings.mesh_worker_fetch_timeout_seconds,
            api_key=settings.mesh_worker_api_key,
        ),
        storage_root=settings.mesh_storage_path,
        cutout=matting_cutout,
        poll_interval=settings.mesh_runner_poll_interval_seconds,
        stale_threshold_seconds=settings.mesh_job_stale_heartbeat_seconds,
        max_retries=settings.mesh_max_retries,
        reconcile_interval=settings.mesh_reconcile_interval_seconds,
        max_wall_seconds=settings.mesh_job_max_wall_seconds,
    )


async def run_mesh_runner_loop() -> None:
    """Started from main.py's lifespan when mesh_runner_enabled. Elects one
    leader across uvicorn workers with a Redis lease (as the video runner
    does); without Redis it runs unguarded."""
    import contextlib

    from backend.app.core import redis_client
    from backend.app.settings import get_settings

    settings = get_settings()
    if not settings.mesh_runner_enabled:
        logger.info("mesh_runner_disabled")
        return
    if not redis_client.is_available():
        logger.warning("mesh_runner_no_redis_running_unguarded")
        await _build_runner(settings).run_forever()
        return

    ttl = max(10, settings.mesh_runner_lease_ttl_seconds)
    renew_every = max(3, ttl // 3)
    token = uuid.uuid4().hex
    while True:
        try:
            if not await redis_client.acquire_lease(_RUNNER_LEASE_KEY, token, ttl):
                await asyncio.sleep(renew_every)
                continue
            runner = _build_runner(settings)
            logger.info("mesh_runner_leader_acquired", worker_id=runner.worker_id)
            loop_task = asyncio.create_task(runner.run_forever())
            try:
                while True:
                    done, _ = await asyncio.wait({loop_task}, timeout=renew_every)
                    if loop_task in done:
                        break
                    if not await redis_client.renew_lease(_RUNNER_LEASE_KEY, token, ttl):
                        logger.warning("mesh_runner_leader_lost")
                        break
            finally:
                loop_task.cancel()
                with contextlib.suppress(Exception):
                    await loop_task
                with contextlib.suppress(Exception):
                    await redis_client.release_lease(_RUNNER_LEASE_KEY, token)
        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("mesh_runner_leader_error")
            await asyncio.sleep(renew_every)
