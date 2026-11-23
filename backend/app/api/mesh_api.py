############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# mesh_api.py: Image-to-3D mesh generation (/v1/meshes).
#
# A job is persisted and 202 returned at once; mesh_runner claims
# it and drives a mesh-worker backend (submit/poll/fetch). Nothing
# here enters InferenceService._proxy_with_retry. The shape of the
# job object follows /v1/videos. See docs/mesh-api.md.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
#
############################################################

"""Public image-to-mesh API (/v1/meshes)."""

import base64
import binascii
import io
import os
import secrets
import uuid
from typing import Any, Dict, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.auth import authenticate_request
from backend.app.api.model_availability import AVAILABLE, model_availability, openai_error
from backend.app.core.telemetry.registry import get_registry
from backend.app.db import crud
from backend.app.db.models import ApiKey, BackendEngine, MeshJob, MeshJobStatus, Modality, User
from backend.app.db.session import get_async_db
from backend.app.logging_config import bind_request_context, get_logger
from backend.app.services import mesh_store
from backend.app.settings import get_settings

router = APIRouter(prefix="/v1", tags=["mesh"])
logger = get_logger(__name__)

# PIL format name -> the content type the file is stored and served as.
_FORMATS = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}
_MAX_IMAGE_SIDE = 8192
_REMOVE_BACKGROUND = {"auto": "auto", "on": "on", "off": "off", "true": "on", "false": "off"}


def _unix(dt) -> Optional[int]:
    return int(dt.timestamp()) if dt else None


def _bad_request(param: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={"error": {"type": "invalid_request_error", "param": param, "message": message}},
    )


def job_to_dict(job: MeshJob, *, queue: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
    """The mesh job object every endpoint returns."""
    done = job.status == MeshJobStatus.COMPLETED
    obj: Dict[str, Any] = {
        "id": job.job_uuid,
        "object": "mesh",
        "status": crud.EXTERNAL_MESH_STATUS.get(job.status, "queued"),
        "stage": job.stage,
        "progress": round(float(job.progress or 0.0), 1),
        "model": job.model,
        "created_at": _unix(job.created_at),
        "started_at": _unix(job.started_at),
        "completed_at": _unix(job.completed_at),
        "seed": job.seed,
        "texture": bool(job.texture),
        "texture_size": job.texture_size,
        "target_faces": job.target_faces,
        "remove_background": job.remove_background,
        "content_url": f"/v1/meshes/{job.job_uuid}/content" if done else None,
        "preview_url": (
            f"/v1/meshes/{job.job_uuid}/content?variant=preview"
            if done and job.preview_size_bytes else None
        ),
        "faces": job.faces,
        "vertices": job.vertices,
        "size_bytes": job.output_size_bytes or None,
        "error": (
            {"code": job.error_code, "message": job.error_message} if job.error_code else None
        ),
        "usage": {"gpu_seconds": job.gpu_seconds, "token_equivalent": job.token_equivalent},
    }
    if queue is not None and job.status == MeshJobStatus.QUEUED:
        obj["queue_position"], obj["queue_depth"] = queue
    return obj


async def _client_ip(request: Request) -> Optional[str]:
    return (
        request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        or request.headers.get("x-real-ip")
        or (request.client.host if request.client else None)
    )


async def mesh_model_names(registry) -> list:
    """Names of models served by any mesh-engine backend (deduped)."""
    names: list = []
    seen = set()
    for backend in await registry.get_all_backends():
        if getattr(backend, "engine", None) != BackendEngine.MESH:
            continue
        for model in await registry.get_backend_models(backend.id):
            name = getattr(model, "name", None)
            if name and name not in seen:
                seen.add(name)
                names.append(name)
    return names


def _csv_ints(value: Any, default: list) -> list:
    try:
        out = [int(v) for v in str(value).split(",") if str(v).strip()]
        return out or default
    except (TypeError, ValueError):
        return default


async def _limits(db: AsyncSession) -> Dict[str, Any]:
    return {
        "texture_sizes": _csv_ints(
            await crud.get_config_json(db, "mesh.allowed_texture_sizes", "1024,2048,4096"),
            [1024, 2048, 4096],
        ),
        "default_texture_size": int(await crud.get_config_json(db, "mesh.default_texture_size", 2048)),
        "min_faces": int(await crud.get_config_json(db, "mesh.min_target_faces", 5000)),
        "max_faces": int(await crud.get_config_json(db, "mesh.max_target_faces", 1000000)),
        "default_faces": int(await crud.get_config_json(db, "mesh.default_target_faces", 100000)),
        "max_image_mb": int(await crud.get_config_json(db, "mesh.max_image_upload_mb", 10)),
    }


def _parse_bool(param: str, value: Any, default: bool) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    word = str(value).strip().lower()
    if word in ("true", "1", "yes", "on"):
        return True
    if word in ("false", "0", "no", "off"):
        return False
    raise _bad_request(param, f"'{param}' must be true or false")


def _parse_int(param: str, value: Any, default: int) -> int:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        raise _bad_request(param, f"'{param}' must be an integer")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise _bad_request(param, f"'{param}' must be an integer") from None
    if not number.is_integer():
        raise _bad_request(param, f"'{param}' must be an integer")
    return int(number)


def decode_image_field(value: Any) -> bytes:
    """A JSON 'image': base64, optionally as a data: URL."""
    if not isinstance(value, str) or not value.strip():
        raise _bad_request("image", "'image' is required (a file upload, or base64 / a data: URL)")
    text = value.strip()
    if text.startswith("data:"):
        _, _, text = text.partition(",")
    try:
        return base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError):
        raise _bad_request("image", "'image' is not valid base64") from None


def sniff_image(data: bytes) -> str:
    """Validate the upload is a PNG, JPEG or WebP picture of sane size, from
    its bytes (never the client's content type). Returns the content type."""
    from PIL import Image

    try:
        with Image.open(io.BytesIO(data)) as img:
            fmt = img.format
            width, height = img.size
    except Exception:
        raise _bad_request("image", "'image' could not be read as a picture") from None
    if fmt not in _FORMATS:
        raise _bad_request("image", "'image' must be a PNG, JPEG or WebP picture")
    if width < 16 or height < 16 or width > _MAX_IMAGE_SIDE or height > _MAX_IMAGE_SIDE:
        raise _bad_request(
            "image", f"'image' must be between 16 and {_MAX_IMAGE_SIDE} pixels on each side"
        )
    return _FORMATS[fmt]


async def submit_mesh_job(
    db: AsyncSession,
    user: User,
    api_key: ApiKey,
    image: bytes,
    params: Dict[str, Any],
    *,
    endpoint: str = "/v1/meshes",
    client_ip: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate a request and enqueue a job. THE single create path: the API
    and the dashboard both call it. Raises HTTPException on any gate."""
    if not await crud.get_config_json(db, "mesh.enabled", False):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="3D mesh generation is currently disabled",
        )
    if not user.mesh_generation_enabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="3D mesh generation is not enabled for your account. Contact an administrator.",
        )

    from backend.app.core.translators.field_validation import (
        MESH_ACCEPTED,
        MESH_DIALECT_HINTS,
        MESH_IGNORED,
        validate_request_fields,
    )

    validate_request_fields(
        params, dialect="mesh", accepted=MESH_ACCEPTED, ignored=MESH_IGNORED, hints=MESH_DIALECT_HINTS,
    )

    limits = await _limits(db)
    max_bytes = limits["max_image_mb"] * 1024 * 1024
    if not image:
        raise _bad_request("image", "'image' is required")
    if len(image) > max_bytes:
        raise _bad_request("image", f"'image' exceeds {limits['max_image_mb']} MB")
    content_type = sniff_image(image)

    texture = _parse_bool("texture", params.get("texture"), True)
    texture_size = _parse_int("texture_size", params.get("texture_size"), limits["default_texture_size"])
    if texture_size not in limits["texture_sizes"]:
        raise _bad_request(
            "texture_size",
            f"'texture_size' must be one of {', '.join(str(s) for s in limits['texture_sizes'])}",
        )
    target_faces = _parse_int("target_faces", params.get("target_faces"), limits["default_faces"])
    if not limits["min_faces"] <= target_faces <= limits["max_faces"]:
        raise _bad_request(
            "target_faces",
            f"'target_faces' must be between {limits['min_faces']:,} and {limits['max_faces']:,}",
        )
    seed = params.get("seed")
    seed = secrets.randbelow(2**31) if seed in (None, "") else _parse_int("seed", seed, 0)
    if not 0 <= seed < 2**31:
        raise _bad_request("seed", "'seed' must be between 0 and 2147483647")
    rb_raw = params.get("remove_background")
    if isinstance(rb_raw, bool):
        rb_raw = "on" if rb_raw else "off"
    remove_background = _REMOVE_BACKGROUND.get(str(rb_raw or "auto").strip().lower())
    if remove_background is None:
        raise _bad_request("remove_background", "'remove_background' must be auto, on or off")

    # ── Model ─────────────────────────────────────────────────────────────
    registry = get_registry()
    requested = params.get("model") or await crud.get_config_json(db, "mesh.default_model", "default-mesh")
    model, _ = registry.resolve_alias(str(requested))
    availability = await model_availability(registry, model)
    if availability != AVAILABLE:
        code, detail, headers = openai_error(model, availability)
        raise HTTPException(status_code=code, detail=detail, headers=headers)
    if model not in await mesh_model_names(registry):
        raise _bad_request("model", f"'{requested}' is not a 3D mesh model (see GET /v1/meshes/models)")

    # ── Fairness, storage, quota ──────────────────────────────────────────
    max_active = int(await crud.get_config_json(db, "mesh.max_concurrent_jobs_per_user", 2))
    active = await crud.count_active_mesh_jobs_for_user(db, user.id)
    if active >= max_active:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"You already have {active} mesh job(s) in progress (limit {max_active}). Wait for one to finish.",
        )
    cap_gb = int(await crud.get_config_json(db, "mesh.user_storage_cap_gb", 20))
    if cap_gb > 0:
        used = await crud.get_user_mesh_storage_bytes(db, user.id)
        if used + len(image) > cap_gb * 1024**3:
            raise HTTPException(
                status_code=status.HTTP_507_INSUFFICIENT_STORAGE,
                detail=(
                    f"3D storage cap reached ({used / 1024**3:.1f} GB of {cap_gb} GB used). "
                    "Delete some meshes to free space, then try again."
                ),
            )
    cost = await crud.compute_mesh_token_cost(db, texture=texture)
    if not await crud.reserve_mesh_tokens(db, user, cost):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Insufficient token quota for this mesh (needs {cost:,} tokens).",
        )

    # ── Persist: audit row, job row, input file ──────────────────────────
    job_uuid = f"mesh-{uuid.uuid4().hex[:24]}"
    bind_request_context(request_id=job_uuid, user_id=user.id)
    options = {
        "texture": texture, "texture_size": texture_size, "target_faces": target_faces,
        "seed": seed, "remove_background": remove_background,
    }
    audit = await crud.create_request(
        db=db, user_id=user.id, api_key_id=api_key.id, endpoint=endpoint, model=model,
        modality=Modality.MESH_GENERATION, prompt=None, parameters=options,
        client_ip=client_ip, user_agent=user_agent,
    )
    job = await crud.create_mesh_job(
        db, job_uuid=job_uuid, user_id=user.id, api_key_id=api_key.id, request_id=audit.id,
        model=model, input_content_type=content_type, input_size_bytes=len(image),
        token_equivalent=cost, **options,
    )
    root = get_settings().mesh_storage_path
    path = mesh_store.input_path(root, user.id, job_uuid, content_type, create=True)
    try:
        with open(path, "wb") as fh:
            fh.write(image)
        await db.commit()
    except Exception:
        await db.rollback()
        try:
            os.remove(path)
        except OSError:
            pass
        raise
    await crud.incr_quota_redis(user.id, cost)
    await db.refresh(job)
    logger.info("mesh_job_created", job_uuid=job_uuid, model=model, texture=texture)
    return job_to_dict(job)


# NOTE: /meshes/models is declared before /meshes/{mesh_id} so the path
# parameter cannot swallow "models".
@router.get("/meshes/models")
async def list_mesh_models(
    db: AsyncSession = Depends(get_async_db),
    auth: Tuple[User, ApiKey] = Depends(authenticate_request),
):
    """Capability discovery for clients and the 3D tab."""
    limits = await _limits(db)
    default = await crud.get_config_json(db, "mesh.default_model", "default-mesh")
    data = [
        {
            "id": name,
            "object": "model",
            "inputs": ["image"],
            "formats": ["glb"],
            "supports_texture": True,
            "texture_sizes": limits["texture_sizes"],
            "default_texture_size": limits["default_texture_size"],
            "min_faces": limits["min_faces"],
            "max_faces": limits["max_faces"],
            "default_faces": limits["default_faces"],
            "max_image_mb": limits["max_image_mb"],
        }
        for name in await mesh_model_names(get_registry())
    ]
    return {"object": "list", "default_model": default, "data": data}


@router.post("/meshes", status_code=status.HTTP_202_ACCEPTED)
async def create_mesh(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
    auth: Tuple[User, ApiKey] = Depends(authenticate_request),
):
    """Create an image-to-mesh job from a multipart upload (field 'image') or
    a JSON body ('image' as base64 or a data: URL). Returns 202 + the job."""
    user, api_key = auth
    ctype = (request.headers.get("content-type") or "").lower()
    if ctype.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("image")
        if upload is None or not hasattr(upload, "read"):
            raise _bad_request("image", "'image' must be a file upload in a multipart request")
        image = await upload.read()
        params = {k: v for k, v in form.items() if k != "image"}
        params["image"] = True  # present (for field validation)
    else:
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON body") from None
        if not isinstance(body, dict):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="JSON body must be an object")
        image = decode_image_field(body.get("image"))
        params = body
    return await submit_mesh_job(
        db, user, api_key, image, params,
        client_ip=await _client_ip(request), user_agent=request.headers.get("user-agent"),
    )


@router.get("/meshes")
async def list_meshes(
    status_filter: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_async_db),
    auth: Tuple[User, ApiKey] = Depends(authenticate_request),
):
    """The caller's mesh jobs, newest first."""
    user, _ = auth
    jobs, total = await crud.list_mesh_jobs(db, user.id, status=status_filter, limit=limit, offset=offset)
    return {"object": "list", "data": [job_to_dict(j) for j in jobs], "total": total}


@router.get("/meshes/{mesh_id}")
async def get_mesh(
    mesh_id: str,
    db: AsyncSession = Depends(get_async_db),
    auth: Tuple[User, ApiKey] = Depends(authenticate_request),
):
    """Poll a job. Another user's id is a 404 (no existence leak)."""
    user, _ = auth
    job = await crud.get_mesh_job_by_uuid(db, mesh_id, user_id=user.id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Mesh not found")
    return job_to_dict(job, queue=await crud.get_mesh_queue_position(db, job))


@router.delete("/meshes/{mesh_id}")
async def delete_mesh(
    mesh_id: str,
    db: AsyncSession = Depends(get_async_db),
    auth: Tuple[User, ApiKey] = Depends(authenticate_request),
):
    """Cancel a running job (refunded), or delete a finished one with its files."""
    user, _ = auth
    job = await crud.get_mesh_job_by_uuid(db, mesh_id, user_id=user.id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Mesh not found")
    if job.status in crud.MESH_TERMINAL_STATUSES:
        await crud.delete_mesh_job(db, job)
        return {"id": mesh_id, "object": "mesh", "deleted": True}
    await crud.request_cancel_mesh_job(db, job)
    await db.commit()
    await db.refresh(job)
    return job_to_dict(job)


def content_response(user_id: int, mesh_id: str, variant: str) -> FileResponse:
    """Serve a finished job's GLB or preview from paths rebuilt from
    (root, user_id, job id) only. Raises 404 when absent."""
    root = get_settings().mesh_storage_path
    try:
        if variant == "preview":
            path = mesh_store.preview_path(root, user_id, mesh_id)
            media_type, filename = "image/png", f"{mesh_id}.png"
        else:
            path = mesh_store.output_path(root, user_id, mesh_id)
            media_type, filename = "model/gltf-binary", f"{mesh_id}.glb"
    except ValueError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Mesh not found") from None
    if not os.path.exists(path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Mesh file not found on disk")
    return FileResponse(
        path, media_type=media_type, filename=filename,
        headers={"Cache-Control": "private, max-age=86400"},
    )


@router.get("/meshes/{mesh_id}/content")
async def get_mesh_content(
    mesh_id: str,
    variant: str = "glb",
    db: AsyncSession = Depends(get_async_db),
    auth: Tuple[User, ApiKey] = Depends(authenticate_request),
):
    """Download the GLB (default) or the PNG preview (?variant=preview)."""
    user, _ = auth
    if variant not in ("glb", "preview"):
        raise _bad_request("variant", "'variant' must be glb or preview")
    job = await crud.get_mesh_job_by_uuid(db, mesh_id, user_id=user.id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Mesh not found")
    if job.status != MeshJobStatus.COMPLETED:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Mesh is not ready yet")
    return content_response(user.id, job.job_uuid, variant)
