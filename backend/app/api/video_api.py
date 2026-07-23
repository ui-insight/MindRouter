############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# video_api.py: OpenAI-compatible video generation endpoints (/v1/videos).
#
# v1 scope: text-to-video, single clip. The endpoint persists a job and
# returns 202 immediately; a gateway-side runner (video_runner, built next)
# claims queued jobs and drives the async worker. Nothing here enters
# InferenceService._proxy_with_retry — video is fully async by design
# (see docs/video-generation-plan.md).
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Public video generation API (/v1/videos)."""

import time
import uuid
from typing import Any, Dict, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.auth import authenticate_request
from backend.app.core.telemetry.registry import get_registry
from backend.app.db import crud
from backend.app.db.models import (
    ApiKey,
    BackendEngine,
    Modality,
    User,
    VideoJob,
    VideoJobStatus,
)
from backend.app.db.session import get_async_db
from backend.app.logging_config import bind_request_context, get_logger

router = APIRouter(prefix="/v1", tags=["video"])
logger = get_logger(__name__)

# Map the gateway's fine-grained job status onto the OpenAI-shaped external
# status (stock SDK polling loops expect in_progress, not planning/rendering).
_EXTERNAL_STATUS = {
    VideoJobStatus.QUEUED: "queued",
    VideoJobStatus.PLANNING: "in_progress",
    VideoJobStatus.RENDERING: "in_progress",
    VideoJobStatus.ASSEMBLING: "in_progress",
    VideoJobStatus.COMPLETED: "completed",
    VideoJobStatus.FAILED: "failed",
    VideoJobStatus.CANCELLED: "cancelled",
}


def _unix(dt) -> Optional[int]:
    return int(dt.timestamp()) if dt else None


def _job_to_dict(job: VideoJob, project=None) -> Dict[str, Any]:
    """Render a job as the OpenAI-shaped video object echoed by every endpoint."""
    obj: Dict[str, Any] = {
        "id": job.job_uuid,
        "object": "video",
        "status": _EXTERNAL_STATUS.get(job.status, "queued"),
        "progress": round(job.progress or 0.0, 1),
        "created_at": _unix(job.created_at),
        "started_at": _unix(job.started_at),
        "completed_at": _unix(job.completed_at),
        "expires_at": _unix(job.expires_at),
        "content_url": (
            f"/v1/videos/{job.job_uuid}/content" if job.output_asset_id else None
        ),
        "error": (
            {"code": job.error_code, "message": job.error_message}
            if job.error_code
            else None
        ),
        "usage": {
            "duration_seconds": job.duration_seconds,
            "gpu_seconds": job.gpu_seconds,
            "token_equivalent": job.token_equivalent,
        },
    }
    if project is not None:
        obj.update(
            {
                "model": project.model,
                "size": project.size,
                "fps": project.fps,
                "quality": (
                    project.quality.value
                    if hasattr(project.quality, "value")
                    else project.quality
                ),
            }
        )
    return obj


async def _client_ip(request: Request) -> Optional[str]:
    return (
        request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        or request.headers.get("x-real-ip")
        or (request.client.host if request.client else None)
    )


# NOTE: /videos/models MUST be declared before /videos/{video_id} or the path
# param would swallow "models".
@router.get("/videos/models")
async def list_video_models(
    db: AsyncSession = Depends(get_async_db),
    auth: Tuple[User, ApiKey] = Depends(authenticate_request),
):
    """Capability discovery. The UI renders its controls from this so the
    preset matrix has exactly one source of truth."""
    allowed_sizes = await crud.get_config_json(
        db, "vid.allowed_sizes", "1280x704,704x1280,960x544,768x448"
    )
    allowed_durations = await crud.get_config_json(db, "vid.allowed_durations", "4,5,8,10")
    registry = get_registry()
    models = []
    for name in await _video_model_names(registry):
        models.append(
            {
                "id": name,
                "supported_sizes": [s.strip() for s in allowed_sizes.split(",") if s.strip()],
                "supported_durations": [s.strip() for s in allowed_durations.split(",") if s.strip()],
                "supported_fps": [24],
                "supported_qualities": ["draft", "standard", "final"],
                "supports_text_to_video": True,
                # v1 is t2v only — these arrive in later phases.
                "supports_image_to_video": False,
                "supports_keyframes": False,
                "max_shots": 1,
                "license_notice": "AI-generated",
            }
        )
    return {"object": "list", "data": models}


async def _video_model_names(registry) -> list:
    """Names of models served by any video-engine backend (deduped)."""
    names: list = []
    seen = set()
    for backend in await registry.get_all_backends():
        if getattr(backend, "engine", None) != BackendEngine.VIDEO:
            continue
        for model in await registry.get_backend_models(backend.id):
            name = getattr(model, "name", None)
            if name and name not in seen:
                seen.add(name)
                names.append(name)
    return names


@router.post("/videos", status_code=status.HTTP_202_ACCEPTED)
async def create_video(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
    auth: Tuple[User, ApiKey] = Depends(authenticate_request),
):
    """Create a text-to-video job. Returns 202 with the job object immediately —
    never the video. The runner renders it asynchronously."""
    user, api_key = auth

    # ── Access control ───────────────────────────────────────────
    if not await crud.get_config_json(db, "vid.enabled", True):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Video generation is currently disabled",
        )
    if not user.video_generation_enabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Video generation is not enabled for your account. Contact an administrator.",
        )

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON body")

    job_uuid = f"vid-{uuid.uuid4().hex[:24]}"
    bind_request_context(request_id=job_uuid, user_id=user.id)

    prompt = body.get("prompt")
    if not prompt or not str(prompt).strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="'prompt' is required")

    # ── Config defaults + guardrails ─────────────────────────────
    default_model = await crud.get_config_json(db, "vid.default_model", "lightricks/ltx-2.3-distilled")
    default_size = await crud.get_config_json(db, "vid.default_size", "1280x704")
    default_seconds = str(await crud.get_config_json(db, "vid.default_seconds", 5))
    default_fps = await crud.get_config_json(db, "vid.default_fps", 24)
    default_quality = await crud.get_config_json(db, "vid.default_quality", "standard")
    allowed_sizes_str = await crud.get_config_json(
        db, "vid.allowed_sizes", "1280x704,704x1280,960x544,768x448"
    )
    allowed_durations_str = await crud.get_config_json(db, "vid.allowed_durations", "4,5,8,10")
    allowed_sizes = [s.strip() for s in allowed_sizes_str.split(",") if s.strip()]
    allowed_durations = [s.strip() for s in allowed_durations_str.split(",") if s.strip()]
    max_per_user = await crud.get_config_json(db, "vid.max_concurrent_jobs_per_user", 1)

    model = body.get("model") or default_model
    size = str(body.get("size") or default_size)
    seconds = str(body.get("seconds") or default_seconds)
    fps = int(body.get("fps") or default_fps)
    quality = body.get("quality") or default_quality

    # Off-menu size/duration are rejected, not silently recompiled (torch.compile
    # is warmed per-shape on the worker).
    if allowed_sizes and size not in allowed_sizes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Size '{size}' is not allowed. Allowed: {', '.join(allowed_sizes)}",
        )
    if allowed_durations and seconds not in allowed_durations:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Duration '{seconds}' is not allowed. Allowed: {', '.join(allowed_durations)}",
        )
    if quality not in ("draft", "standard", "final"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Quality '{quality}' is not valid (draft|standard|final)",
        )

    # ── Model validation ─────────────────────────────────────────
    registry = get_registry()
    model, _ = registry.resolve_alias(model)
    if not await registry.model_exists(model):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "message": f"The model '{model}' does not exist",
                    "type": "invalid_request_error",
                    "code": "model_not_found",
                }
            },
        )

    # ── Per-user concurrency (fairness on a one-render-at-a-time GPU) ──
    active = await crud.count_active_video_jobs_for_user(db, user.id)
    if active >= max_per_user:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"You already have {active} video job(s) in progress "
                f"(limit {max_per_user}). Wait for one to finish."
            ),
        )

    # ── Persist: audit request → one-shot project → job → shot ───
    audit = await crud.create_request(
        db=db,
        user_id=user.id,
        api_key_id=api_key.id,
        endpoint="/v1/videos",
        model=model,
        modality=Modality.VIDEO_GENERATION,
        prompt=prompt,
        parameters={"size": size, "seconds": seconds, "fps": fps, "quality": quality},
        client_ip=await _client_ip(request),
        user_agent=request.headers.get("user-agent"),
    )
    project = await crud.create_video_project(
        db=db,
        user_id=user.id,
        model=model,
        size=size,
        fps=fps,
        quality=quality,
        style_prompt=None,
    )
    job = await crud.create_video_job(
        db=db,
        job_uuid=job_uuid,
        project_id=project.id,
        user_id=user.id,
        api_key_id=api_key.id,
        request_id=audit.id,
        shots_total=1,
        callback_url=body.get("callback_url"),
    )
    await crud.create_video_shot(
        db=db,
        job_id=job.id,
        shot_index=0,
        seconds=float(seconds),
        prompt=prompt,
        seed=body.get("seed"),
    )
    await db.commit()
    # expire_on_commit may have expired these; reload before serializing.
    await db.refresh(job)
    await db.refresh(project)

    logger.info("video_job_created", job_uuid=job_uuid, model=model, size=size, seconds=seconds)
    return _job_to_dict(job, project)


@router.get("/videos")
async def list_videos(
    status_filter: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_async_db),
    auth: Tuple[User, ApiKey] = Depends(authenticate_request),
):
    """List the caller's video jobs, newest first."""
    user, _ = auth
    jobs, total = await crud.list_video_jobs(
        db, user.id, status=status_filter, limit=limit, offset=offset
    )
    return {"object": "list", "data": [_job_to_dict(j) for j in jobs], "total": total}


@router.get("/videos/{video_id}")
async def get_video(
    video_id: str,
    db: AsyncSession = Depends(get_async_db),
    auth: Tuple[User, ApiKey] = Depends(authenticate_request),
):
    """Poll a single job. 404 (never 403) for another user's id — no existence leak."""
    user, _ = auth
    job = await crud.get_video_job_by_uuid(db, video_id, user_id=user.id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")
    return _job_to_dict(job)


@router.delete("/videos/{video_id}")
async def cancel_video(
    video_id: str,
    db: AsyncSession = Depends(get_async_db),
    auth: Tuple[User, ApiKey] = Depends(authenticate_request),
):
    """Cancel a running job (propagated to the worker by the runner) or delete a
    terminal one's record."""
    user, _ = auth
    job = await crud.get_video_job_by_uuid(db, video_id, user_id=user.id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")
    await crud.request_cancel_video_job(db, job)
    await db.commit()
    await db.refresh(job)
    return _job_to_dict(job)


@router.get("/videos/{video_id}/content")
async def get_video_content(
    video_id: str,
    db: AsyncSession = Depends(get_async_db),
    auth: Tuple[User, ApiKey] = Depends(authenticate_request),
):
    """Stream the rendered MP4 (Range-capable). Until the runner + worker land,
    a job never has an output asset, so this reports not-ready."""
    user, _ = auth
    job = await crud.get_video_job_by_uuid(db, video_id, user_id=user.id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")
    if not job.output_asset_id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Video is not ready yet",
        )
    # Streaming delivery (video_store, Range/206) is built with the runner.
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Video content delivery is not yet available",
    )
