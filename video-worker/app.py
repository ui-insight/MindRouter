############################################################
#
# mindrouter video-worker - LTX-2.3 async video generation service
#
# app.py: FastAPI app exposing the async worker contract the gateway runner
#     drives: submit / poll / fetch / cancel, plus health + capabilities.
#
# Run: uvicorn app:app --host 0.0.0.0 --port 18300
# Mode: VIDEO_WORKER_MODE=mock (default, no GPU) | ltx (GPU node).
#
# Luke Sheneman — University of Idaho RCDS — sheneman@uidaho.edu
#
############################################################

"""FastAPI application for the video worker."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from config import SUPPORTED_SIZES, WorkerConfig, frames_for
from engine import build_engine
from worker import JobManager

VERSION = "0.1.0"


def _external_status(job) -> dict:
    return {
        "id": job.id,
        "status": job.status,
        "progress": job.progress,
        "step": job.step,
        "total_steps": job.total_steps,
        "duration_ms": job.duration_ms,
        "error": job.error,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    config: WorkerConfig = app.state.config
    engine = build_engine(config)
    if config.mode == "ltx":  # pragma: no cover - GPU node only
        engine.load()
    app.state.manager = JobManager(engine, config.output_dir)
    await app.state.manager.start()
    try:
        yield
    finally:
        await app.state.manager.stop()


def create_app(config: WorkerConfig | None = None) -> FastAPI:
    app = FastAPI(title="mindrouter video-worker", version=VERSION, lifespan=lifespan)
    app.state.config = config or WorkerConfig()

    def manager() -> JobManager:
        return app.state.manager

    @app.get("/health")
    async def health():
        # Must answer <5s even while a render is in flight — generation runs off
        # the event loop, so this never blocks.
        return manager().health()

    @app.get("/version")
    async def version():
        return {"version": VERSION, "mode": app.state.config.mode, "model": app.state.config.model_id}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": m} for m in manager().engine.model_ids()]}

    @app.get("/v1/capabilities")
    async def capabilities():
        return manager().engine.capabilities()

    @app.post("/v1/videos", status_code=202)
    async def submit(request: Request):
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="invalid JSON body")

        prompt = body.get("prompt")
        if not prompt or not str(prompt).strip():
            raise HTTPException(status_code=400, detail="'prompt' is required")

        size = str(body.get("size") or "")
        if size not in SUPPORTED_SIZES:
            raise HTTPException(status_code=400, detail=f"size '{size}' not supported")

        seconds = str(body.get("seconds") or "")
        try:
            num_frames = frames_for(seconds)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        spec = {
            "model": body.get("model"),
            "prompt": prompt,
            "size": size,
            "seconds": seconds,
            "num_frames": num_frames,
            "fps": int(body.get("fps") or app.state.config.default_fps),
            "quality": body.get("quality") or "standard",
            "seed": body.get("seed"),
        }
        job = manager().submit(spec)
        return {"id": job.id, "status": job.status}

    @app.get("/v1/videos/{job_id}")
    async def poll(job_id: str):
        job = manager().get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        return _external_status(job)

    @app.get("/v1/videos/{job_id}/content")
    async def content(job_id: str):
        job = manager().get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        if job.status != "completed" or not job.output_path:
            raise HTTPException(status_code=409, detail="job not completed")
        # FileResponse (starlette) serves Accept-Ranges + 206 partial content.
        return FileResponse(job.output_path, media_type="video/mp4", filename=f"{job_id}.mp4")

    @app.delete("/v1/videos/{job_id}")
    async def cancel(job_id: str):
        if not manager().request_cancel(job_id):
            raise HTTPException(status_code=404, detail="job not found")
        return JSONResponse({"id": job_id, "status": "cancelling"})

    return app


app = create_app()
