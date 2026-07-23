############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# video.py: Dashboard Video tab (session-cookie routes).
#
# General-purpose video generation UI (research/teaching/comms/science/ads —
# NOT ad-centric). Shares the ONE create path (video_api.submit_video_job) with
# the public API, so both produce identical results (no images split-brain).
# See docs/video-generation-plan.md.
#
# Luke Sheneman — University of Idaho RCDS — sheneman@uidaho.edu
#
############################################################

"""Dashboard Video tab routes."""

import os
from typing import Tuple

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.video_api import submit_video_job, _job_to_dict
from backend.app.dashboard.routes import get_masquerade_user_id, get_session_user_id
from backend.app.db import crud
from backend.app.db.models import ApiKey, User
from backend.app.db.session import get_async_db
from backend.app.settings import get_settings

video_router = APIRouter(tags=["video"])

templates_path = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=templates_path)
templates.env.globals["version"] = get_settings().app_version


async def _get_video_user(request: Request, db: AsyncSession) -> Tuple[User, ApiKey]:
    """Session user + first active API key, enforcing video access."""
    user_id = get_session_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    user = await crud.get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    if not user.video_generation_enabled:
        raise HTTPException(status_code=403, detail="Video generation not enabled for your account")
    api_keys = await crud.get_user_api_keys(db, user_id, include_revoked=False)
    if not api_keys:
        raise HTTPException(
            status_code=403,
            detail="No active API key. Create one in your dashboard first.",
        )
    return user, api_keys[0]


@video_router.get("/video", response_class=HTMLResponse)
async def video_page(request: Request, db: AsyncSession = Depends(get_async_db)):
    """Serve the Video tab."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)
    user = await crud.get_user_by_id(db, user_id)
    if not user or not user.video_generation_enabled:
        return RedirectResponse(url="/dashboard", status_code=302)

    allowed_sizes = await crud.get_config_json(db, "vid.allowed_sizes", "1280x704,704x1280,1024x576,768x448")
    allowed_durations = await crud.get_config_json(db, "vid.allowed_durations", "4,5,8,10")
    default_size = await crud.get_config_json(db, "vid.default_size", "1280x704")
    default_seconds = str(await crud.get_config_json(db, "vid.default_seconds", 5))
    default_quality = await crud.get_config_json(db, "vid.default_quality", "standard")
    max_total_seconds = await crud.get_config_json(db, "vid.max_total_seconds", 90)

    api_keys = await crud.get_user_api_keys(db, user_id, include_revoked=False)

    masquerade_user = None
    masq_id = get_masquerade_user_id(request)
    if masq_id:
        masquerade_user = await crud.get_user_by_id(db, masq_id)

    return templates.TemplateResponse(
        "user/video.html",
        {
            "request": request,
            "user": user,
            "masquerade_user": masquerade_user,
            "allowed_sizes": [s.strip() for s in allowed_sizes.split(",") if s.strip()],
            "allowed_durations": [s.strip() for s in allowed_durations.split(",") if s.strip()],
            "default_size": default_size,
            "default_seconds": default_seconds,
            "default_quality": default_quality,
            "max_total_seconds": max_total_seconds,
            "has_api_key": len(api_keys) > 0,
        },
    )


@video_router.post("/video/api/videos")
async def video_create(request: Request, db: AsyncSession = Depends(get_async_db)):
    """Create a job via the SHARED create path (same as POST /v1/videos)."""
    user, api_key = await _get_video_user(request, db)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": {"message": "invalid JSON body"}})
    try:
        job = await submit_video_job(
            db, user, api_key, body,
            endpoint="/video/api/videos",
            client_ip=(request.client.host if request.client else None),
            user_agent=request.headers.get("user-agent"),
        )
    except HTTPException as exc:
        return JSONResponse(status_code=exc.status_code, content={"error": {"message": exc.detail}})
    return job


@video_router.get("/video/api/jobs/{video_id}")
async def video_poll(video_id: str, request: Request, db: AsyncSession = Depends(get_async_db)):
    user, _ = await _get_video_user(request, db)
    job = await crud.get_video_job_by_uuid(db, video_id, user_id=user.id)
    if not job:
        raise HTTPException(status_code=404, detail="Video not found")
    return _job_to_dict(job)


@video_router.get("/video/api/jobs")
async def video_library(
    request: Request, limit: int = 12, offset: int = 0, db: AsyncSession = Depends(get_async_db)
):
    """Paginated library of the user's jobs (newest first)."""
    user, _ = await _get_video_user(request, db)
    jobs, total = await crud.list_video_jobs(db, user.id, limit=limit, offset=offset)
    return {"data": [_job_to_dict(j) for j in jobs], "total": total}


@video_router.delete("/video/api/jobs/{video_id}")
async def video_cancel(video_id: str, request: Request, db: AsyncSession = Depends(get_async_db)):
    user, _ = await _get_video_user(request, db)
    job = await crud.get_video_job_by_uuid(db, video_id, user_id=user.id)
    if not job:
        raise HTTPException(status_code=404, detail="Video not found")
    await crud.request_cancel_video_job(db, job)
    await db.commit()
    await db.refresh(job)
    return _job_to_dict(job)


@video_router.get("/video/serve/{video_id}")
async def video_serve(video_id: str, request: Request, db: AsyncSession = Depends(get_async_db)):
    """Stream a rendered MP4 (Range/206 via FileResponse, ownership-checked)."""
    user, _ = await _get_video_user(request, db)
    job = await crud.get_video_job_by_uuid(db, video_id, user_id=user.id)
    if not job or job.output_asset_id is None:
        raise HTTPException(status_code=404, detail="Video not found")
    asset = await crud.get_video_asset(db, job.output_asset_id)
    if not asset or not asset.storage_path or not os.path.exists(asset.storage_path):
        raise HTTPException(status_code=404, detail="Video file not found on disk")
    return FileResponse(
        asset.storage_path,
        media_type=asset.content_type or "video/mp4",
        headers={"Cache-Control": "private, max-age=86400"},
    )
