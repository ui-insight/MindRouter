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

from backend.app.api.video_api import submit_video_job, _job_to_dict, _video_model_names
from backend.app.dashboard.routes import get_masquerade_user_id, get_session_user_id
from backend.app.db import crud
from backend.app.security.api_keys import first_live_api_key
import os as _os

from backend.app.db.models import ApiKey, User, VideoJobStatus
from backend.app.db.session import get_async_db
from backend.app.settings import get_settings

video_router = APIRouter(tags=["video"])

templates_path = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=templates_path)
templates.env.globals["version"] = get_settings().app_version
# base.html calls branding() on every page — this env needs the global too.
from backend.app.services import branding as _branding  # noqa: E402
templates.env.globals["branding"] = _branding.get_branding
# base.html gates the Images nav link on image_access(user): the flag is
# tri-state, so a raw truthiness test hides the link from every inheriting
# user. Nav visibility only — the routes re-resolve against the database.
from backend.app.services import feature_access as _feature_access  # noqa: E402
templates.env.globals["image_access"] = _feature_access.image_access


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
    live_key = first_live_api_key(api_keys)
    if live_key is None:
        raise HTTPException(
            status_code=403,
            detail="No active API key. Create or renew one in your dashboard first.",
        )
    return user, live_key


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
    allowed_durations = await crud.get_config_json(db, "vid.allowed_durations", "4,5,8,10,12,15,20")
    default_size = await crud.get_config_json(db, "vid.default_size", "1280x704")
    default_seconds = str(await crud.get_config_json(db, "vid.default_seconds", 5))
    default_quality = await crud.get_config_json(db, "vid.default_quality", "standard")
    max_total_seconds = await crud.get_config_json(db, "vid.max_total_seconds", 30)
    default_model = await crud.get_config_json(db, "vid.default_model", "lightricks/ltx-2.3-distilled")
    from backend.app.core.telemetry.registry import get_registry

    video_models = await _video_model_names(get_registry())
    if default_model not in video_models:
        video_models.insert(0, default_model)

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
            "video_models": video_models,
            "default_model": default_model,
            "min_seconds": await crud.get_config_json(db, "vid.min_seconds", 4),
            "max_seconds": max_total_seconds,
            "has_api_key": first_live_api_key(api_keys) is not None,
            "storage_cap_gb": int(await crud.get_config_json(db, "vid.user_storage_cap_gb", 50)),
            "storage_used_gb": round(await crud.get_user_video_storage_bytes(db, user_id) / 1024**3, 2),
        },
    )


async def _storage_cap_error(db: AsyncSession, user_id: int, incoming_bytes: int):
    """Return a 507 JSONResponse if adding `incoming_bytes` would exceed the
    per-user video storage cap (vid.user_storage_cap_gb), else None."""
    cap_gb = int(await crud.get_config_json(db, "vid.user_storage_cap_gb", 50))
    if cap_gb <= 0:
        return None
    used = await crud.get_user_video_storage_bytes(db, user_id)
    if used + incoming_bytes > cap_gb * 1024 * 1024 * 1024:
        return JSONResponse(
            status_code=507,
            content={"error": {"message": (
                f"Video storage cap reached ({used / 1024**3:.1f} GB of {cap_gb} GB used). "
                "Delete some videos to free space."
            )}},
        )
    return None


async def _store_reference_asset(db: AsyncSession, user_id: int, data: bytes, content_type: str):
    """Write reference-image bytes into the user's video refs dir (sha-named, so
    re-imports dedup on disk) and create a REFERENCE VideoAsset. Caller commits."""
    import hashlib

    from backend.app.db.models import VideoAssetKind

    ct = (content_type or "").lower() or "image/png"
    sha = hashlib.sha256(data).hexdigest()
    ext = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}.get(ct, "png")
    ref_dir = _os.path.join(get_settings().video_storage_path, str(user_id), "refs")
    _os.makedirs(ref_dir, exist_ok=True)
    path = _os.path.join(ref_dir, f"{sha}.{ext}")
    with open(path, "wb") as fh:
        fh.write(data)
    return await crud.create_video_asset(
        db, user_id=user_id, kind=VideoAssetKind.REFERENCE,
        storage_path=path, content_type=ct, sha256=sha, size_bytes=len(data),
    )


@video_router.post("/video/api/assets")
async def video_upload_asset(request: Request, db: AsyncSession = Depends(get_async_db)):
    """Upload an optional start/end conditioning image. Returns its asset id,
    which the create call references as start_image_asset_id / end_image_asset_id."""
    user, _ = await _get_video_user(request, db)
    form = await request.form()
    upload = form.get("file")
    if upload is None or not hasattr(upload, "read"):
        return JSONResponse(status_code=400, content={"error": {"message": "no file provided"}})
    ct = (getattr(upload, "content_type", "") or "").lower()
    if not ct.startswith("image/"):
        return JSONResponse(status_code=400, content={"error": {"message": "file must be an image"}})
    data = await upload.read()
    max_bytes = int(await crud.get_config_json(db, "vid.max_image_upload_mb", 10)) * 1024 * 1024
    if len(data) > max_bytes:
        return JSONResponse(status_code=400, content={"error": {"message": f"image exceeds {max_bytes // 1024 // 1024}MB"}})

    cap_err = await _storage_cap_error(db, user.id, len(data))
    if cap_err:
        return cap_err
    asset = await _store_reference_asset(db, user.id, data, ct)
    await db.commit()
    return {"asset_id": asset.id}


@video_router.post("/video/api/assets/from-gallery")
async def video_import_gallery_asset(request: Request, db: AsyncSession = Depends(get_async_db)):
    """Import a gallery image (UserImage) as a video conditioning reference.

    COPIES the bytes into the video asset domain — never a shared reference — so
    deleting a video (which deletes its asset files) can't touch the gallery
    original. Returns an asset id usable as start_image_asset_id/end_image_asset_id."""
    from backend.app.storage.artifacts import get_artifact_storage

    user, _ = await _get_video_user(request, db)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": {"message": "invalid JSON body"}})
    raw_id = body.get("image_id")
    try:
        image_id = int(raw_id)
    except (TypeError, ValueError):
        return JSONResponse(status_code=400, content={"error": {"message": "image_id is required"}})

    img = await crud.get_user_image(db, image_id, user_id=user.id)
    if not img:
        return JSONResponse(status_code=404, content={"error": {"message": "image not found"}})
    data = await get_artifact_storage().retrieve(img.storage_path)
    if data is None:
        return JSONResponse(status_code=404, content={"error": {"message": "image file not found on disk"}})

    cap_err = await _storage_cap_error(db, user.id, len(data))
    if cap_err:
        return cap_err
    asset = await _store_reference_asset(db, user.id, data, img.content_type or "image/png")
    await db.commit()
    return {"asset_id": asset.id}


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


@video_router.get("/video/api/queue")
async def video_queue(request: Request, db: AsyncSession = Depends(get_async_db)):
    """The shared render queue with per-job elapsed time and ETA, for the live
    queue panel. Rows are in processing order (rendering first, then the waiting
    line). Only the caller's own rows carry an id; others are anonymous."""
    import time

    user, _ = await _get_video_user(request, db)
    rows = await crud.get_active_video_queue(db)
    ratios = {}          # model -> render seconds per output second
    for row in rows:
        if row.get("model") not in ratios:
            ratios[row.get("model")] = await crud.get_recent_render_ratio(db, model=row.get("model"))
    now = time.time()

    def _est(row):
        secs = row.get("seconds") or 0
        base = ratios[row.get("model")] * secs
        q = (row.get("quality") or "standard")
        mult = {"draft": 0.6, "standard": 1.0, "final": 1.6}.get(q, 1.0)
        return max(5.0, base * mult)

    out = []
    ahead = 0.0          # estimated seconds until the next queued job starts
    slot = 0
    for row in rows:
        slot += 1
        est = _est(row)
        rendering = row["status"] == "in_progress"
        started = row.get("started_at")
        elapsed = (now - started) if started else None
        if rendering:
            remaining = max(0.0, est - (elapsed or 0.0))
            eta = remaining
            ahead = remaining        # this job's remaining gates the queue
        else:
            eta = ahead              # time until THIS job begins
            ahead += est
        item = {
            "slot": slot,
            "mine": row["user_id"] == user.id,
            "status": row["status"],
            "progress": round(row.get("progress") or 0.0, 1),
            "size": row.get("size"),
            "seconds": row.get("seconds"),
            "elapsed_seconds": round(elapsed) if elapsed is not None else None,
            "est_seconds": round(est),
            "eta_seconds": round(eta),
            "created_at": row.get("created_at"),
        }
        if row["user_id"] == user.id:
            item["id"] = row["job_uuid"]
        out.append(item)

    depth = sum(1 for r in rows if r["status"] == "queued")
    return {"queue": out, "depth": depth, "total": len(rows)}


@video_router.get("/video/api/jobs/{video_id}")
async def video_poll(video_id: str, request: Request, db: AsyncSession = Depends(get_async_db)):
    user, _ = await _get_video_user(request, db)
    job = await crud.get_video_job_by_uuid(db, video_id, user_id=user.id)
    if not job:
        raise HTTPException(status_code=404, detail="Video not found")
    data = _job_to_dict(job)
    # Surface the shared render queue + this job's place in it (fairness on a
    # one-render-at-a-time GPU). position is 0 unless the job is still queued.
    position, depth = await crud.get_video_queue_position(db, job)
    data["queue_position"] = position
    data["queue_depth"] = depth
    return data


@video_router.get("/video/api/jobs")
async def video_library(
    request: Request, limit: int = 12, offset: int = 0, status: str = None,
    db: AsyncSession = Depends(get_async_db),
):
    """Paginated library of the user's jobs (newest first), each enriched with
    its project (size/quality) so the gallery can label clips. Pass
    status=completed to paginate only finished videos (the gallery view)."""
    user, _ = await _get_video_user(request, db)
    st = None
    if status:
        try:
            st = VideoJobStatus(status)
        except ValueError:
            st = None
    jobs, total = await crud.list_video_jobs(
        db, user.id, status=st, limit=limit, offset=offset
    )
    data = []
    for j in jobs:
        proj = await crud.get_video_project(db, j.project_id)
        data.append(_job_to_dict(j, proj))
    return {"data": data, "total": total}


@video_router.delete("/video/api/jobs/{video_id}")
async def video_delete(video_id: str, request: Request, db: AsyncSession = Depends(get_async_db)):
    """Cancel a running job, or DELETE a terminal one (removes its artifact +
    rows) — this is what the gallery delete button calls."""
    user, _ = await _get_video_user(request, db)
    job = await crud.get_video_job_by_uuid(db, video_id, user_id=user.id)
    if not job:
        raise HTTPException(status_code=404, detail="Video not found")

    terminal = {VideoJobStatus.COMPLETED, VideoJobStatus.FAILED, VideoJobStatus.CANCELLED}
    if job.status in terminal:
        paths = await crud.delete_video_job(db, job)
        for p in paths:
            try:
                if p and _os.path.exists(p):
                    _os.remove(p)
            except OSError:
                pass
        return {"id": video_id, "deleted": True}

    # still running/queued -> cancel (runner propagates + refunds quota)
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
    if not asset:
        raise HTTPException(status_code=404, detail="Video file not found on disk")
    # Reconstruct the served path from Snyk-clean components only (config root +
    # int user id + os.path.basename(id)) so no stored path string reaches
    # FileResponse. Layout: <root>/<user_id>/<vid-uuid>.mp4.
    safe_path = os.path.join(
        get_settings().video_storage_path, str(user.id), f"{os.path.basename(video_id)}.mp4"
    )
    if not os.path.exists(safe_path):
        raise HTTPException(status_code=404, detail="Video file not found on disk")
    return FileResponse(
        safe_path,
        media_type=asset.content_type or "video/mp4",
        headers={"Cache-Control": "private, max-age=86400"},
    )
