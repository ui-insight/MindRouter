############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# images.py: User-facing image generation gallery routes
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Image generation gallery routes for MindRouter."""

import base64
import os
from typing import Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db import crud
from backend.app.db.models import ApiKey, User, UserImage
from backend.app.db.session import get_async_db
from backend.app.dashboard.routes import get_session_user_id, get_effective_user_id, get_masquerade_user_id
from backend.app.logging_config import get_logger
from backend.app.services.inference import InferenceService
from backend.app.settings import get_settings
from backend.app.storage.artifacts import get_artifact_storage

logger = get_logger(__name__)

images_router = APIRouter(tags=["images"])

# Setup templates — share globals with main dashboard templates
templates_path = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=templates_path)
templates.env.globals["version"] = get_settings().app_version


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _get_image_user(
    request: Request, db: AsyncSession
) -> Tuple[User, ApiKey]:
    """Get user and first active API key from session, enforce image access."""
    user_id = get_session_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")

    user = await crud.get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    if not user.image_generation_enabled:
        raise HTTPException(status_code=403, detail="Image generation not enabled for your account")

    api_keys = await crud.get_user_api_keys(db, user_id, include_revoked=False)
    if not api_keys:
        raise HTTPException(
            status_code=403,
            detail="No active API key. Create one in your dashboard first.",
        )

    return user, api_keys[0]


# ---------------------------------------------------------------------------
# Page route
# ---------------------------------------------------------------------------

@images_router.get("/images", response_class=HTMLResponse)
async def images_page(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """Serve the image generation UI."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or not user.image_generation_enabled:
        return RedirectResponse(url="/dashboard", status_code=302)

    # Load config
    default_model = await crud.get_config_json(db, "img.default_model", "")
    default_size = await crud.get_config_json(db, "img.default_size", "1024x1024")
    default_steps = await crud.get_config_json(db, "img.default_steps", 30)
    max_steps = await crud.get_config_json(db, "img.max_steps", 50)
    default_guidance = await crud.get_config_json(db, "img.default_guidance_scale", 3.5)
    allowed_sizes_str = await crud.get_config_json(db, "img.allowed_sizes", "")
    max_width = await crud.get_config_json(db, "img.max_width", 2048)
    max_height = await crud.get_config_json(db, "img.max_height", 2048)

    allowed_sizes = [s.strip() for s in allowed_sizes_str.split(",") if s.strip()] if allowed_sizes_str else []

    # Check if user has an API key
    api_keys = await crud.get_user_api_keys(db, user_id, include_revoked=False)
    has_api_key = len(api_keys) > 0

    # Masquerade context
    masquerade_user = None
    masq_id = get_masquerade_user_id(request)
    if masq_id:
        masquerade_user = await crud.get_user_by_id(db, masq_id)

    return templates.TemplateResponse(
        "user/images.html",
        {
            "request": request,
            "user": user,
            "masquerade_user": masquerade_user,
            "default_model": default_model,
            "default_size": default_size,
            "default_steps": default_steps,
            "max_steps": max_steps,
            "default_guidance": default_guidance,
            "allowed_sizes": allowed_sizes,
            "max_width": max_width,
            "max_height": max_height,
            "has_api_key": has_api_key,
        },
    )


# ---------------------------------------------------------------------------
# Proxy: policy check + image generation
# ---------------------------------------------------------------------------

@images_router.post("/images/api/check-policy")
async def images_api_check_policy(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """Check prompt against content policy (fast, separate from generation)."""
    user, _ = await _get_image_user(request, db)
    body = await request.json()
    prompt = body.get("prompt", "").strip()
    if not prompt:
        return JSONResponse(status_code=400, content={"error": {"message": "prompt is required"}})

    policy = await crud.get_config_json(db, "img.policy", "")
    if not policy:
        return JSONResponse(content={"passed": True, "reason": "No policy configured"})

    judge_model = await crud.get_config_json(db, "img.judge_model", "")
    judge_secondary = await crud.get_config_json(db, "img.judge_model_secondary", "")
    if not judge_model:
        return JSONResponse(content={"passed": True, "reason": "No judge model configured"})

    from backend.app.services.image_policy import evaluate_prompt
    verdict = await evaluate_prompt(prompt, policy, judge_model, judge_secondary or None)

    if not verdict.passed:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": f"Your prompt was not approved: {verdict.reason}",
                    "type": "content_policy_violation",
                    "policy_passed": False,
                }
            },
        )

    return JSONResponse(content={
        "passed": True,
        "reason": verdict.reason,
        "model": verdict.judge_model,
    })


@images_router.post("/images/api/generate")
async def images_api_generate(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """Proxy image generation through session auth (like chat does for completions)."""
    user, api_key = await _get_image_user(request, db)

    body = await request.json()

    from backend.app.core.canonical_schemas import CanonicalImageRequest

    # Load config for defaults/guardrails
    enabled = await crud.get_config_json(db, "img.enabled", True)
    if not enabled:
        return JSONResponse(
            status_code=503,
            content={"error": {"message": "Image generation is currently disabled.", "type": "service_unavailable"}},
        )

    default_model = await crud.get_config_json(db, "img.default_model", "")
    default_size = await crud.get_config_json(db, "img.default_size", "1024x1024")
    default_steps = await crud.get_config_json(db, "img.default_steps", 30)
    max_steps = await crud.get_config_json(db, "img.max_steps", 50)
    default_guidance = await crud.get_config_json(db, "img.default_guidance_scale", 3.5)
    max_n = await crud.get_config_json(db, "img.max_n", 1)
    allowed_sizes_str = await crud.get_config_json(db, "img.allowed_sizes", "")
    max_width = await crud.get_config_json(db, "img.max_width", 2048)
    max_height = await crud.get_config_json(db, "img.max_height", 2048)

    # Apply defaults
    if not body.get("model"):
        body["model"] = default_model
    if not body.get("size"):
        body["size"] = default_size
    if body.get("num_inference_steps") is None:
        body["num_inference_steps"] = default_steps
    if body.get("guidance_scale") is None:
        body["guidance_scale"] = default_guidance

    # Guardrails
    n = body.get("n", 1)
    if n > max_n:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": f"n must be <= {max_n}", "type": "invalid_request_error"}},
        )

    steps = body.get("num_inference_steps", default_steps)
    if steps > max_steps:
        body["num_inference_steps"] = max_steps

    # Size validation
    size_str = body.get("size", default_size)
    if allowed_sizes_str:
        allowed = [s.strip() for s in allowed_sizes_str.split(",") if s.strip()]
        if size_str not in allowed:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": f"Size must be one of: {', '.join(allowed)}", "type": "invalid_request_error"}},
            )

    try:
        w, h = map(int, size_str.split("x"))
        if w > max_width or h > max_height:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": f"Dimensions exceed maximum ({max_width}x{max_height})", "type": "invalid_request_error"}},
            )
    except (ValueError, AttributeError):
        pass

    # Policy check
    policy = await crud.get_config_json(db, "img.policy", "")
    if policy:
        judge_model = await crud.get_config_json(db, "img.judge_model", "")
        judge_secondary = await crud.get_config_json(db, "img.judge_model_secondary", "")
        if judge_model:
            from backend.app.services.image_policy import evaluate_prompt
            verdict = await evaluate_prompt(body.get("prompt", ""), policy, judge_model, judge_secondary or None)
            if not verdict.passed:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": {
                            "message": f"Your prompt was not approved: {verdict.reason}",
                            "type": "content_policy_violation",
                            "policy_passed": False,
                        }
                    },
                )
            # Attach verdict for audit trail
            body["_policy_verdict"] = {"passed": True, "reason": verdict.reason, "model": verdict.judge_model}

    # Translate to canonical request
    prompt = body.get("prompt", "")
    if not prompt:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "prompt is required", "type": "invalid_request_error"}},
        )

    canonical = CanonicalImageRequest(
        model=body["model"],
        prompt=prompt,
        n=body.get("n", 1),
        size=body.get("size", default_size),
        response_format=body.get("response_format", "b64_json"),
        num_inference_steps=body.get("num_inference_steps"),
        guidance_scale=body.get("guidance_scale"),
        seed=body.get("seed"),
    )

    if body.get("_policy_verdict"):
        canonical.policy_verdict = body["_policy_verdict"]

    # Call inference service
    svc = InferenceService(db)
    try:
        result = await svc.image_generation(canonical, user, api_key, request)
    except HTTPException as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"message": exc.detail, "type": "api_error"}},
        )
    except Exception as exc:
        logger.exception("image_generation_error")
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(exc), "type": "server_error"}},
        )

    return JSONResponse(content=result)


# ---------------------------------------------------------------------------
# Gallery: save, history, serve, delete
# ---------------------------------------------------------------------------

@images_router.post("/images/save")
async def images_save(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """Save a generated image to the user's gallery."""
    user, _ = await _get_image_user(request, db)
    body = await request.json()

    image_b64 = body.get("image_b64", "")
    prompt = body.get("prompt", "")[:2000]
    model = body.get("model", "")[:200] or "unknown"
    size = body.get("size", "")[:20] or "unknown"
    steps = body.get("steps")
    guidance = body.get("guidance")
    seed = body.get("seed")

    if not image_b64:
        return JSONResponse(status_code=400, content={"error": "No image data"})

    # Decode base64
    try:
        image_data = base64.b64decode(image_b64)
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid base64 data"})

    # Store via ArtifactStorage
    storage = get_artifact_storage()
    storage_path, sha256, size_bytes = await storage.store(
        image_data, f"generated.png", "image/png"
    )

    # Create DB record
    img = UserImage(
        user_id=user.id,
        prompt=prompt,
        model=model,
        size=size,
        steps=int(steps) if steps is not None else None,
        guidance_scale=float(guidance) if guidance is not None else None,
        seed=int(seed) if seed is not None else None,
        storage_path=storage_path,
        content_type="image/png",
        size_bytes=size_bytes,
    )
    db.add(img)
    await db.commit()
    await db.refresh(img)

    return JSONResponse(content={"id": img.id, "ok": True})


@images_router.get("/images/history")
async def images_history(
    request: Request,
    offset: int = 0,
    limit: int = 12,
    db: AsyncSession = Depends(get_async_db),
):
    """Return paginated list of user's generated images."""
    user, _ = await _get_image_user(request, db)

    limit = min(limit, 50)

    # Total count
    count_q = select(func.count()).select_from(UserImage).where(UserImage.user_id == user.id)
    total = (await db.execute(count_q)).scalar() or 0

    # Fetch images
    q = (
        select(UserImage)
        .where(UserImage.user_id == user.id)
        .order_by(desc(UserImage.created_at))
        .offset(offset)
        .limit(limit)
    )
    result = await db.execute(q)
    images = result.scalars().all()

    return JSONResponse(content={
        "images": [
            {
                "id": img.id,
                "prompt": img.prompt[:200],
                "model": img.model,
                "size": img.size,
                "steps": img.steps,
                "guidance": img.guidance_scale,
                "created": img.created_at.strftime("%Y-%m-%d %H:%M") if img.created_at else "",
            }
            for img in images
        ],
        "total": total,
        "offset": offset,
    })


@images_router.get("/images/serve/{image_id}")
async def images_serve(
    image_id: int,
    request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """Serve a stored image file (with ownership check)."""
    user_id = get_session_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")

    q = select(UserImage).where(UserImage.id == image_id, UserImage.user_id == user_id)
    result = await db.execute(q)
    img = result.scalar_one_or_none()

    if not img:
        raise HTTPException(status_code=404, detail="Image not found")

    storage = get_artifact_storage()
    data = await storage.retrieve(img.storage_path)
    if data is None:
        raise HTTPException(status_code=404, detail="Image file not found on disk")

    return Response(
        content=data,
        media_type=img.content_type,
        headers={
            "Cache-Control": "private, max-age=86400",
            "Content-Disposition": f'inline; filename="mindrouter-{img.id}.png"',
        },
    )


@images_router.delete("/images/delete/{image_id}")
async def images_delete(
    image_id: int,
    request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """Delete a user's image (with ownership check)."""
    user_id = get_session_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")

    q = select(UserImage).where(UserImage.id == image_id, UserImage.user_id == user_id)
    result = await db.execute(q)
    img = result.scalar_one_or_none()

    if not img:
        raise HTTPException(status_code=404, detail="Image not found")

    # Delete from storage
    storage = get_artifact_storage()
    await storage.delete(img.storage_path)

    # Delete DB record
    await db.delete(img)
    await db.commit()

    return JSONResponse(content={"ok": True})
