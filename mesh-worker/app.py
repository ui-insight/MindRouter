############################################################
#
# mindrouter mesh-worker - async image-to-3D mesh service
#
# app.py: FastAPI app exposing the async contract the gateway drives:
#     submit / poll / fetch / cancel, plus health, version, models and
#     capabilities. The 3D analogue of video-worker/.
#
# Run: uvicorn app:app --host 127.0.0.1 --port 18401
# Mode: MESH_WORKER_MODE=mock (default, no GPU) | trellis2 | pixal3d | hunyuan3d21
#
# Luke Sheneman — University of Idaho RCDS — sheneman@uidaho.edu
#
############################################################

"""FastAPI application for the mesh worker."""

import asyncio
import hmac
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse

from config import WorkerConfig
from engines import build_engine
from inputs import RequestError, validate_request
from worker import JOB_ID_RE, Job, JobManager, QueueFull

VERSION = "0.1.0"

log = logging.getLogger(__name__)

MEDIA_TYPES = {"glb": "model/gltf-binary", "preview": "image/png"}


def _status(job: Job) -> dict:
    completed = job.status == "completed"
    return {
        "id": job.id,
        "status": job.status,
        "progress": job.progress,
        "stage": job.stage,
        "error": job.error,
        "duration_ms": job.duration_ms,
        "outputs": job.outputs if completed else None,
        "faces": job.faces if completed else None,
        "vertices": job.vertices if completed else None,
        # Additive: DELETE on an in-flight job is accepted but only takes
        # effect at the next stage boundary.
        "cancel_requested": job.cancel and job.status not in ("completed", "failed", "cancelled"),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    config: WorkerConfig = app.state.config
    engine = build_engine(config)
    if config.mode != "mock":  # pragma: no cover - GPU node only
        # Weights load before the port opens: until then the gateway sees a
        # refused connection (= unhealthy), never a half-loaded worker.
        engine.load()
    app.state.engine = engine
    app.state.manager = JobManager(engine, config)
    await app.state.manager.start()
    try:
        yield
    finally:
        await app.state.manager.stop()


def create_app(config: Optional[WorkerConfig] = None) -> FastAPI:
    app = FastAPI(title="mindrouter mesh-worker", version=VERSION, lifespan=lifespan)
    app.state.config = config or WorkerConfig()

    async def require_worker_key(
        x_worker_key: Optional[str] = Header(default=None, alias="X-Worker-Key"),
    ) -> None:
        """Guard every /v1/* route with the shared secret when one is set.
        Empty api_key = auth disabled (rollout window). Constant-time compare."""
        expected = app.state.config.api_key
        if not expected:
            return
        if not x_worker_key or not hmac.compare_digest(x_worker_key.encode(), expected.encode()):
            raise HTTPException(status_code=401, detail="invalid or missing worker key")

    guarded = [Depends(require_worker_key)]

    def manager() -> JobManager:
        return app.state.manager

    def capabilities_body() -> dict:
        cfg = app.state.config
        engine_caps = {}
        caps_fn = getattr(app.state.engine, "capabilities", None)
        if callable(caps_fn):
            engine_caps = caps_fn() or {}
        return {
            "model": cfg.model_id,
            "inputs": ["image"],
            "formats": ["glb"],
            "supports_texture": bool(engine_caps.get("supports_texture", True)),
            "texture_sizes": list(cfg.texture_sizes),
            "default_texture_size": cfg.default_texture_size,
            "min_faces": cfg.min_faces,
            "max_faces": cfg.max_faces,
            "default_faces": cfg.default_faces,
            "max_image_bytes": cfg.max_image_bytes,
            "max_queue": cfg.max_queue,
        }

    @app.get("/health")
    async def health():
        # Open (no key) and always fast: generation runs on its own executor
        # thread, so this never waits behind a job. 503 only when the GPU
        # thread has overrun its timeout + grace and is presumed hung.
        body = manager().health()
        return JSONResponse(body, status_code=503 if body["status"] != "ok" else 200)

    @app.get("/version")
    async def version():
        cfg = app.state.config
        return {"version": VERSION, "mode": cfg.mode, "model": cfg.model_id}

    # /v1/models and /v1/capabilities stay OPEN, like /health: the gateway's
    # registry discovers models with VLLMAdapter, which sends no X-Worker-Key,
    # so a guarded /v1/models would make discovery find zero models once
    # MESH_WORKER_API_KEY is set. They expose only the served model name and
    # limits. Every /v1/meshes route (the work and the outputs) stays guarded.
    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": app.state.config.model_id, "object": "model"}]}

    @app.get("/v1/capabilities")
    async def capabilities():
        return capabilities_body()

    @app.post("/v1/meshes", status_code=202, dependencies=guarded)
    async def submit(request: Request):
        cfg = app.state.config
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > cfg.max_body_bytes:
            raise HTTPException(status_code=413, detail="request body too large")
        # Cheap reject before decoding a large image.
        if not manager().has_capacity():
            raise HTTPException(status_code=503, detail="queue full", headers={"Retry-After": "30"})
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="invalid JSON body")
        try:
            # Image decode/verify off the event loop (keeps /health snappy).
            spec = await asyncio.to_thread(validate_request, body, cfg)
        except RequestError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        try:
            job = manager().submit(spec)
        except QueueFull:
            raise HTTPException(status_code=503, detail="queue full", headers={"Retry-After": "30"})
        return {"id": job.id, "status": job.status}

    def _job_or_404(job_id: str) -> Job:
        job = manager().get(job_id) if JOB_ID_RE.match(job_id or "") else None
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return job

    @app.get("/v1/meshes/{job_id}", dependencies=guarded)
    async def poll(job_id: str):
        return _status(_job_or_404(job_id))

    @app.get("/v1/meshes/{job_id}/content", dependencies=guarded)
    async def content(job_id: str, variant: str = Query(default="glb")):
        if variant not in MEDIA_TYPES:
            raise HTTPException(status_code=400, detail="variant must be 'glb' or 'preview'")
        job = _job_or_404(job_id)
        if job.status != "completed":
            raise HTTPException(status_code=409, detail="job not completed")
        # Built from clean components only (output_dir + the validated id).
        path = manager().glb_path(job.id) if variant == "glb" else manager().preview_path(job.id)
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="job not found")
        ext = "glb" if variant == "glb" else "png"
        return FileResponse(path, media_type=MEDIA_TYPES[variant], filename=f"{job.id}.{ext}")

    @app.delete("/v1/meshes/{job_id}", dependencies=guarded)
    async def cancel(job_id: str):
        if not JOB_ID_RE.match(job_id or ""):
            raise HTTPException(status_code=404, detail="job not found")
        job = manager().request_cancel(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return _status(job)

    return app


app = create_app()
