############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# mesh.py: The dashboard's 3D tab (image to mesh). Session-
#     authenticated wrappers around the SAME create path as
#     POST /v1/meshes (submit_mesh_job), attributed to the
#     user's first live API key, as the Video tab does.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
#
############################################################

"""Dashboard routes for the 3D (image-to-mesh) tab."""

import os
from typing import Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.mesh_api import content_response, job_to_dict, submit_mesh_job
from backend.app.dashboard.routes import get_masquerade_user_id, get_session_user_id
from backend.app.db import crud
from backend.app.db.models import ApiKey, MeshJobStatus, User
from backend.app.db.session import get_async_db
from backend.app.security.api_keys import first_live_api_key
from backend.app.settings import get_settings

mesh_router = APIRouter(tags=["mesh"])

templates_path = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=templates_path)
templates.env.globals["version"] = get_settings().app_version
from backend.app.services import branding as _branding  # noqa: E402
templates.env.globals["branding"] = _branding.get_branding
from backend.app.services import feature_access as _feature_access  # noqa: E402
templates.env.globals["image_access"] = _feature_access.image_access


def _error(status_code: int, message) -> JSONResponse:
    if isinstance(message, dict):
        message = (message.get("error") or {}).get("message") or str(message)
    return JSONResponse(status_code=status_code, content={"error": {"message": str(message)}})


async def _get_mesh_user(request: Request, db: AsyncSession) -> Tuple[User, ApiKey]:
    """Session user + first live API key, enforcing 3D access."""
    user_id = get_session_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    user = await crud.get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    if not user.mesh_generation_enabled:
        raise HTTPException(status_code=403, detail="3D generation is not enabled for your account")
    live_key = first_live_api_key(await crud.get_user_api_keys(db, user_id, include_revoked=False))
    if live_key is None:
        raise HTTPException(
            status_code=403, detail="No active API key. Create or renew one in your dashboard first."
        )
    return user, live_key


@mesh_router.get("/3d", response_class=HTMLResponse)
async def mesh_page(request: Request, db: AsyncSession = Depends(get_async_db)):
    """Serve the 3D tab."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)
    user = await crud.get_user_by_id(db, user_id)
    if not user or not user.mesh_generation_enabled:
        return RedirectResponse(url="/dashboard", status_code=302)

    from backend.app.api.mesh_api import _limits, license_ready, license_summary, model_settings
    from backend.app.core.telemetry.registry import get_registry
    from backend.app.services import mesh_models as mm

    default_model = await crud.get_config_json(db, "mesh.default_model", "default-mesh")
    default_model, _ = get_registry().resolve_alias(default_model)
    models = await crud.get_servable_mesh_models(db)
    if default_model in models:
        models.remove(default_model)
        models.insert(0, default_model)
    # Per-model limits (Admin -> Meshes caps) and license notices, for the
    # form to follow the selected model; models whose license an admin has
    # not accepted are not offered.
    base = await _limits(db)
    settings = await model_settings(db)
    per_model = {}
    for name in list(models):
        if not await license_ready(db, name):
            models.remove(name)
            continue
        lim = mm.model_limits(base, settings.get(name))
        lic = license_summary(name) or {}
        per_model[name] = {
            "max_faces": lim["max_faces"], "min_faces": lim["min_faces"], "default_faces": lim["default_faces"],
            "texture_sizes": lim["texture_sizes"], "default_texture_size": lim["default_texture_size"],
            "notice": lic.get("notice", ""), "license_name": lic.get("name", ""), "license_url": lic.get("url", ""),
        }
    first = per_model.get(models[0]) if models else None

    masquerade_user = None
    masq_id = get_masquerade_user_id(request)
    if masq_id:
        masquerade_user = await crud.get_user_by_id(db, masq_id)

    api_keys = await crud.get_user_api_keys(db, user_id, include_revoked=False)
    return templates.TemplateResponse(
        "user/mesh.html",
        {
            "request": request,
            "user": user,
            "masquerade_user": masquerade_user,
            "mesh_models": models,
            "default_model": default_model,
            "service_enabled": bool(await crud.get_config_json(db, "mesh.enabled", False)),
            # The service-wide sizes; the form narrows them per selected model.
            "texture_sizes": base["texture_sizes"],
            "default_texture_size": (first or base)["default_texture_size"],
            "default_faces": (first or base)["default_faces"],
            "min_faces": base["min_faces"],
            "max_faces": base["max_faces"],
            "max_image_mb": base["max_image_mb"],
            "per_model": per_model,
            "first_notice": (first or {}).get("notice", ""),
            "first_license": first or {},
            "has_api_key": first_live_api_key(api_keys) is not None,
            "storage_cap_gb": int(await crud.get_config_json(db, "mesh.user_storage_cap_gb", 20)),
            "storage_used_gb": round(await crud.get_user_mesh_storage_bytes(db, user_id) / 1024**3, 2),
        },
    )


@mesh_router.post("/3d/api/meshes")
async def mesh_create(request: Request, db: AsyncSession = Depends(get_async_db)):
    """Create a job through the shared create path (as POST /v1/meshes)."""
    try:
        user, api_key = await _get_mesh_user(request, db)
    except HTTPException as exc:
        return _error(exc.status_code, exc.detail)
    form = await request.form()
    upload = form.get("image")
    if upload is None or not hasattr(upload, "read"):
        return _error(400, "Choose a picture to turn into a 3D model.")
    image = await upload.read()
    params = {k: v for k, v in form.items() if k != "image"}
    params["image"] = True
    try:
        return await submit_mesh_job(
            db, user, api_key, image, params,
            endpoint="/3d/api/meshes",
            client_ip=(request.client.host if request.client else None),
            user_agent=request.headers.get("user-agent"),
        )
    except HTTPException as exc:
        return _error(exc.status_code, exc.detail)


@mesh_router.get("/3d/api/jobs")
async def mesh_list(
    request: Request, limit: int = 60, offset: int = 0, db: AsyncSession = Depends(get_async_db)
):
    try:
        user, _ = await _get_mesh_user(request, db)
    except HTTPException as exc:
        return _error(exc.status_code, exc.detail)
    jobs, total = await crud.list_mesh_jobs(db, user.id, limit=limit, offset=offset)
    used = await crud.get_user_mesh_storage_bytes(db, user.id)
    return {
        "data": [job_to_dict(j) for j in jobs],
        "total": total,
        "storage_used_gb": round(used / 1024**3, 2),
    }


@mesh_router.get("/3d/api/jobs/{mesh_id}")
async def mesh_get(mesh_id: str, request: Request, db: AsyncSession = Depends(get_async_db)):
    try:
        user, _ = await _get_mesh_user(request, db)
    except HTTPException as exc:
        return _error(exc.status_code, exc.detail)
    job = await crud.get_mesh_job_by_uuid(db, mesh_id, user_id=user.id)
    if not job:
        return _error(404, "Mesh not found")
    return job_to_dict(job, queue=await crud.get_mesh_queue_position(db, job))


@mesh_router.delete("/3d/api/jobs/{mesh_id}")
async def mesh_delete(mesh_id: str, request: Request, db: AsyncSession = Depends(get_async_db)):
    """Cancel a running job, or delete a finished one with its files."""
    try:
        user, _ = await _get_mesh_user(request, db)
    except HTTPException as exc:
        return _error(exc.status_code, exc.detail)
    job = await crud.get_mesh_job_by_uuid(db, mesh_id, user_id=user.id)
    if not job:
        return _error(404, "Mesh not found")
    if job.status in crud.MESH_TERMINAL_STATUSES:
        await crud.delete_mesh_job(db, job)
        return {"id": mesh_id, "deleted": True}
    await crud.request_cancel_mesh_job(db, job)
    await db.commit()
    await db.refresh(job)
    return job_to_dict(job)


@mesh_router.get("/3d/serve/{mesh_id}")
async def mesh_serve(
    mesh_id: str, request: Request, variant: Optional[str] = "glb", db: AsyncSession = Depends(get_async_db)
):
    """The GLB (for the viewer and downloads) or the preview PNG."""
    user_id = get_session_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    if variant not in ("glb", "preview"):
        raise HTTPException(status_code=400, detail="variant must be glb or preview")
    job = await crud.get_mesh_job_by_uuid(db, mesh_id, user_id=user_id)
    if not job or job.status != MeshJobStatus.COMPLETED:
        raise HTTPException(status_code=404, detail="Mesh not found")
    return content_response(user_id, job.job_uuid, variant)
