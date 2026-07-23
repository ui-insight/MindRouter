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

import os

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import FileResponse
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
        db, "vid.allowed_sizes", "1280x704,704x1280,1024x576,768x448"
    )
    min_seconds = int(await crud.get_config_json(db, "vid.min_seconds", 4))
    max_seconds = int(await crud.get_config_json(db, "vid.max_total_seconds", 90))
    registry = get_registry()
    models = []
    for name in await _video_model_names(registry):
        models.append(
            {
                "id": name,
                "supported_sizes": [s.strip() for s in allowed_sizes.split(",") if s.strip()],
                # Duration is a continuous whole-second range, not a preset list.
                "min_seconds": min_seconds,
                "max_seconds": max_seconds,
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


async def submit_video_job(
    db: AsyncSession,
    user: User,
    api_key: ApiKey,
    body: Dict[str, Any],
    *,
    endpoint: str = "/v1/videos",
    client_ip: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate a video request and enqueue a job. THE single create path —
    both the public API (create_video) and the dashboard route call this, so
    identical bodies produce identical results (no images-style split-brain).
    Raises HTTPException on any gate. Returns the job object."""
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

    # Surface any field MindRouter would silently drop (typos, wrong-API fields,
    # unsupported-in-v1 params). Runs at the global field_validation setting
    # ('log' by default — dark launch; flips to reject fleet-wide later).
    from backend.app.core.translators.field_validation import (
        VIDEO_ACCEPTED,
        VIDEO_DIALECT_HINTS,
        VIDEO_IGNORED,
        validate_request_fields,
    )

    validate_request_fields(
        body, dialect="video",
        accepted=VIDEO_ACCEPTED, ignored=VIDEO_IGNORED, hints=VIDEO_DIALECT_HINTS,
    )

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
        db, "vid.allowed_sizes", "1280x704,704x1280,1024x576,768x448"
    )
    allowed_sizes = [s.strip() for s in allowed_sizes_str.split(",") if s.strip()]
    min_seconds = int(await crud.get_config_json(db, "vid.min_seconds", 4))
    max_seconds = int(await crud.get_config_json(db, "vid.max_total_seconds", 90))
    max_per_user = await crud.get_config_json(db, "vid.max_concurrent_jobs_per_user", 1)

    model = body.get("model") or default_model
    size = str(body.get("size") or default_size)
    seconds = str(body.get("seconds") or default_seconds)
    fps = int(body.get("fps") or default_fps)
    quality = body.get("quality") or default_quality

    # Sizes are a fixed preset menu (torch.compile shape set). Duration is a
    # continuous whole-second slider: frames = 24*seconds + 1 (always 8k+1).
    if allowed_sizes and size not in allowed_sizes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Size '{size}' is not allowed. Allowed: {', '.join(allowed_sizes)}",
        )
    try:
        sec_val = float(seconds)
        sec_int = int(sec_val)
    except (ValueError, TypeError):
        sec_val, sec_int = -1.0, -1
    if sec_val != sec_int or not (min_seconds <= sec_int <= max_seconds):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Duration must be a whole number of seconds between {min_seconds} and {max_seconds}.",
        )
    seconds = str(sec_int)
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

    # ── Quota reservation (charge up front; refunded if the render fails/cancels).
    # Video bypasses InferenceService, so it enforces quota here, mirroring
    # _check_quota's DB-tokens model. Reserving (not just checking) is what
    # actually bounds spend across concurrent submissions.
    cost = await crud.compute_video_token_cost(db, seconds=float(seconds), quality=quality, size=size)
    if not await crud.reserve_video_tokens(db, user, cost):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Insufficient token quota for this video (needs ~{cost:,} tokens).",
        )

    # ── Persist: audit request → one-shot project → job → shot ───
    audit = await crud.create_request(
        db=db,
        user_id=user.id,
        api_key_id=api_key.id,
        endpoint=endpoint,
        model=model,
        modality=Modality.VIDEO_GENERATION,
        prompt=prompt,
        parameters={"size": size, "seconds": seconds, "fps": fps, "quality": quality},
        client_ip=client_ip,
        user_agent=user_agent,
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
    job.token_equivalent = cost  # the reserved amount (refunded on fail/cancel)
    await crud.create_video_shot(
        db=db,
        job_id=job.id,
        shot_index=0,
        seconds=float(seconds),
        prompt=prompt,
        seed=body.get("seed"),
    )
    await db.commit()
    # Sync the reservation to Redis post-commit (the documented quota pattern).
    await crud.incr_quota_redis(user.id, cost)
    # expire_on_commit may have expired these; reload before serializing.
    await db.refresh(job)
    await db.refresh(project)

    logger.info("video_job_created", job_uuid=job_uuid, model=model, size=size, seconds=seconds)
    return _job_to_dict(job, project)


@router.post("/videos", status_code=status.HTTP_202_ACCEPTED)
async def create_video(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
    auth: Tuple[User, ApiKey] = Depends(authenticate_request),
):
    """Create a text-to-video job. Returns 202 with the job object immediately —
    never the video. The runner renders it asynchronously."""
    user, api_key = auth
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON body")
    return await submit_video_job(
        db, user, api_key, body,
        client_ip=await _client_ip(request),
        user_agent=request.headers.get("user-agent"),
    )


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
    """Stream the rendered MP4 from disk. FileResponse (starlette) serves
    Accept-Ranges + 206 partial content so browser <video> scrubbing works —
    it streams the file rather than loading it into memory (unlike the images
    path, which does not scale past a few MB)."""
    user, _ = auth
    job = await crud.get_video_job_by_uuid(db, video_id, user_id=user.id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")
    if job.output_asset_id is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Video is not ready yet")

    asset = await crud.get_video_asset(db, job.output_asset_id)
    if not asset or not asset.storage_path or not os.path.exists(asset.storage_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Video file not found on disk"
        )
    return FileResponse(
        asset.storage_path,
        media_type=asset.content_type or "video/mp4",
        filename=f"{video_id}.mp4",
        headers={"Cache-Control": "private, max-age=86400"},
    )
