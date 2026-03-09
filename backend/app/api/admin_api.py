############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# admin_api.py: Administrative API endpoints for management
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Admin API endpoints."""

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.auth import require_admin, require_admin_or_session
from backend.app.core.scheduler.policy import get_scheduler
from backend.app.core.telemetry.registry import get_registry
from backend.app.db import crud
from backend.app.db.models import BackendEngine, BackendStatus, Group, RequestStatus, User, UserRole
from backend.app.db.session import get_async_db
from backend.app.logging_config import get_logger
from backend.app.security.api_keys import generate_api_key
from backend.app.security.password_hash import hash_password
from backend.app.settings import get_settings

logger = get_logger(__name__)
router = APIRouter()

# In-memory pull job tracking (replaces sidecar-based tracking)
_pull_jobs: Dict[str, dict] = {}


async def _refresh_node_ollama_backends(node_id: Optional[int]) -> None:
    """Refresh all Ollama backends on a node (shared models folder)."""
    if node_id is None:
        return
    registry = get_registry()
    from backend.app.db.session import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        from backend.app.db.models import Backend as BackendModel
        from sqlalchemy import select
        result = await db.execute(
            select(BackendModel.id).where(
                BackendModel.node_id == node_id,
                BackendModel.engine == BackendEngine.OLLAMA,
            )
        )
        backend_ids = [row[0] for row in result.all()]
    for bid in backend_ids:
        await registry.refresh_backend(bid)
    logger.info("refreshed_node_backends", node_id=node_id, backend_ids=backend_ids)


async def _run_ollama_pull(job_id: str, ollama_url: str, model: str, node_id: Optional[int] = None) -> None:
    """Stream an Ollama pull and update job progress in-memory.

    Retries up to 3 times with exponential backoff on transient failures
    (network errors, DNS resolution, HTTP errors).
    """
    import asyncio as _asyncio

    job = _pull_jobs[job_id]
    max_attempts = 3
    last_error = ""

    for attempt in range(1, max_attempts + 1):
        try:
            # Reset status for retry
            job["status"] = "pulling"
            if attempt > 1:
                job["progress"] = {"status": f"Retry {attempt}/{max_attempts}..."}
                logger.info(
                    "ollama_pull_retry",
                    job_id=job_id, model=model, attempt=attempt,
                )

            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=600.0)) as client:
                async with client.stream(
                    "POST",
                    f"{ollama_url}/api/pull",
                    json={"name": model, "stream": True},
                ) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        last_error = f"Ollama returned {resp.status_code}: {body.decode(errors='replace')}"
                        raise _PullRetryable(last_error)

                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        job["progress"] = {
                            "status": data.get("status", ""),
                            "digest": data.get("digest", ""),
                            "total": data.get("total", 0),
                            "completed": data.get("completed", 0),
                        }

                        # Ollama may return an error in the stream
                        if data.get("error"):
                            last_error = data["error"]
                            raise _PullRetryable(last_error)

                        if data.get("status") == "success":
                            job["status"] = "success"
                            job["completed_at"] = datetime.now(timezone.utc).isoformat()
                            await _refresh_node_ollama_backends(node_id)
                            return

            # Stream ended without explicit success — treat as success
            if job["status"] == "pulling":
                job["status"] = "success"
                job["completed_at"] = datetime.now(timezone.utc).isoformat()
                await _refresh_node_ollama_backends(node_id)
            return

        except Exception as exc:
            last_error = last_error or str(exc) or f"{type(exc).__name__}: {repr(exc)}"
            if attempt < max_attempts:
                delay = 2 ** attempt  # 2s, 4s
                logger.warning(
                    "ollama_pull_attempt_failed",
                    job_id=job_id, model=model, attempt=attempt,
                    error=last_error, retry_in=delay,
                )
                job["progress"] = {"status": f"Failed (attempt {attempt}/{max_attempts}), retrying in {delay}s..."}
                await _asyncio.sleep(delay)
                last_error = ""
            else:
                job["status"] = "error"
                job["error"] = f"{last_error} (after {max_attempts} attempts)"
                job["completed_at"] = datetime.now(timezone.utc).isoformat()
                logger.error(
                    "ollama_pull_failed",
                    job_id=job_id, model=model,
                    error=last_error, attempts=max_attempts,
                )


class _PullRetryable(Exception):
    """Raised to trigger retry logic in _run_ollama_pull."""
    pass


# Request/Response models
class NodeRegisterRequest(BaseModel):
    """Request to register a new node."""
    name: str = Field(..., min_length=1, max_length=100)
    hostname: Optional[str] = None
    sidecar_url: Optional[str] = None
    sidecar_key: Optional[str] = None


class NodeResponse(BaseModel):
    """Node information response."""
    id: int
    name: str
    hostname: Optional[str]
    sidecar_url: Optional[str]
    sidecar_key_set: bool = False
    status: str
    gpu_count: Optional[int]
    driver_version: Optional[str]
    cuda_version: Optional[str]
    sidecar_version: Optional[str] = None

    class Config:
        from_attributes = True


class BackendRegisterRequest(BaseModel):
    """Request to register a new backend."""
    name: str = Field(..., min_length=1, max_length=100)
    url: str = Field(..., min_length=1)
    engine: BackendEngine
    max_concurrent: int = Field(default=4, ge=1)
    gpu_memory_gb: Optional[float] = None
    gpu_type: Optional[str] = None
    node_id: Optional[int] = None
    gpu_indices: Optional[List[int]] = None


class BackendResponse(BaseModel):
    """Backend information response."""
    id: int
    name: str
    url: str
    engine: str
    status: str
    max_concurrent: int
    current_concurrent: int
    gpu_memory_gb: Optional[float]
    gpu_type: Optional[str]
    node_id: Optional[int]
    node_name: Optional[str]
    gpu_indices: Optional[List[int]]
    version: Optional[str]
    last_health_check: Optional[datetime]

    class Config:
        from_attributes = True


class BackendUpdateRequest(BaseModel):
    """Request to update an existing backend. Only provided fields are changed."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    url: Optional[str] = Field(None, min_length=1)
    engine: Optional[BackendEngine] = None
    max_concurrent: Optional[int] = Field(None, ge=1)
    gpu_memory_gb: Optional[float] = None
    gpu_type: Optional[str] = None
    priority: Optional[int] = None
    node_id: Optional[int] = None
    gpu_indices: Optional[List[int]] = None


class NodeUpdateRequest(BaseModel):
    """Request to update an existing node. Only provided fields are changed."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    hostname: Optional[str] = None
    sidecar_url: Optional[str] = None
    sidecar_key: Optional[str] = None


class QueueStats(BaseModel):
    """Queue statistics."""
    total: int
    by_user: Dict[int, int]
    by_model: Dict[str, int]
    average_wait_seconds: float


class AuditSearchRequest(BaseModel):
    """Audit log search parameters."""
    user_id: Optional[int] = None
    model: Optional[str] = None
    status: Optional[RequestStatus] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    search_text: Optional[str] = None
    skip: int = 0
    limit: int = 100


class AuditRecord(BaseModel):
    """Audit record response."""
    id: int
    request_uuid: str
    user_id: int
    endpoint: str
    model: str
    status: str
    is_streaming: bool
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_time_ms: Optional[int]
    created_at: datetime
    backend_id: Optional[int]

    class Config:
        from_attributes = True


# Backend Management
@router.post("/backends/register", response_model=BackendResponse)
async def register_backend(
    request: BackendRegisterRequest,
    admin: User = Depends(require_admin()),
    db: AsyncSession = Depends(get_async_db),
):
    """
    Register a new backend server.

    Requires admin role.
    """
    # Check for duplicate name
    existing = await crud.get_backend_by_name(db, request.name)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Backend with name '{request.name}' already exists",
        )

    # Check for duplicate URL
    existing_url = await crud.get_backend_by_url(db, request.url)
    if existing_url:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Backend with URL '{request.url}' already exists (registered as '{existing_url.name}')",
        )

    registry = get_registry()

    backend = await registry.register_backend(
        name=request.name,
        url=request.url,
        engine=request.engine,
        max_concurrent=request.max_concurrent,
        gpu_memory_gb=request.gpu_memory_gb,
        gpu_type=request.gpu_type,
        node_id=request.node_id,
        gpu_indices=request.gpu_indices,
        db=db,
    )

    # Resolve node name
    node_name = None
    if request.node_id:
        node = await crud.get_node_by_id(db, request.node_id)
        if node:
            node_name = node.name

    logger.info(
        "backend_registered_by_admin",
        admin_id=admin.id,
        backend_id=backend.id,
        name=backend.name,
    )

    return BackendResponse(
        id=backend.id,
        name=backend.name,
        url=backend.url,
        engine=backend.engine.value,
        status=backend.status.value,
        max_concurrent=backend.max_concurrent,
        current_concurrent=backend.current_concurrent,
        gpu_memory_gb=backend.gpu_memory_gb,
        gpu_type=backend.gpu_type,
        node_id=backend.node_id,
        node_name=node_name,
        gpu_indices=backend.gpu_indices,
        version=backend.version,
        last_health_check=backend.last_health_check,
    )


@router.post("/backends/{backend_id}/disable")
async def disable_backend(
    backend_id: int,
    admin: User = Depends(require_admin()),
    db: AsyncSession = Depends(get_async_db),
):
    """Disable a backend."""
    backend = await crud.get_backend_by_id(db, backend_id)
    if not backend:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Backend not found",
        )

    registry = get_registry()
    await registry.disable_backend(backend_id)

    logger.info(
        "backend_disabled",
        admin_id=admin.id,
        backend_id=backend_id,
    )

    return {"status": "disabled", "backend_id": backend_id}


@router.post("/backends/{backend_id}/enable")
async def enable_backend(
    backend_id: int,
    admin: User = Depends(require_admin()),
    db: AsyncSession = Depends(get_async_db),
):
    """Enable a previously disabled backend."""
    backend = await crud.get_backend_by_id(db, backend_id)
    if not backend:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Backend not found",
        )

    registry = get_registry()
    await registry.enable_backend(backend_id)

    logger.info(
        "backend_enabled",
        admin_id=admin.id,
        backend_id=backend_id,
    )

    return {"status": "enabled", "backend_id": backend_id}


@router.post("/backends/{backend_id}/refresh")
async def refresh_backend(
    backend_id: int,
    admin: User = Depends(require_admin_or_session()),
    db: AsyncSession = Depends(get_async_db),
):
    """Force refresh capabilities for a backend."""
    backend = await crud.get_backend_by_id(db, backend_id)
    if not backend:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Backend not found",
        )

    registry = get_registry()
    success = await registry.refresh_backend(backend_id)

    if success:
        return {"status": "refreshed", "backend_id": backend_id}
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to refresh backend",
        )


# Ollama Model Management (direct pull/delete — bypasses sidecar)
class OllamaPullRequest(BaseModel):
    """Request to pull a model to an Ollama backend."""
    model: str = Field(..., min_length=1)


class OllamaDeleteRequest(BaseModel):
    """Request to delete a model from an Ollama backend."""
    model: str = Field(..., min_length=1)


@router.post("/backends/{backend_id}/ollama/pull")
async def ollama_pull(
    backend_id: int,
    request: OllamaPullRequest,
    admin: User = Depends(require_admin_or_session()),
    db: AsyncSession = Depends(get_async_db),
):
    """Start pulling a model to an Ollama backend (direct call, no sidecar)."""
    backend = await crud.get_backend_by_id(db, backend_id)
    if not backend:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Backend not found")
    if backend.engine != BackendEngine.OLLAMA:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Backend is not an Ollama engine")
    if not backend.url:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Backend has no URL configured")

    job_id = str(uuid.uuid4())
    _pull_jobs[job_id] = {
        "job_id": job_id,
        "model": request.model,
        "ollama_url": backend.url,
        "status": "pulling",
        "progress": {},
        "error": None,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
    }

    asyncio.create_task(_run_ollama_pull(job_id, backend.url, request.model, backend.node_id))

    logger.info(
        "ollama_pull_started",
        admin_id=admin.id,
        backend_id=backend_id,
        model=request.model,
        job_id=job_id,
    )

    return {"job_id": job_id, "status": "pulling"}


@router.get("/backends/{backend_id}/ollama/pull/{job_id}")
async def ollama_pull_status(
    backend_id: int,
    job_id: str,
    admin: User = Depends(require_admin_or_session()),
    db: AsyncSession = Depends(get_async_db),
):
    """Poll progress of an Ollama model pull (in-memory lookup)."""
    job = _pull_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Pull job not found")

    return job


@router.post("/backends/{backend_id}/ollama/delete")
async def ollama_delete(
    backend_id: int,
    request: OllamaDeleteRequest,
    admin: User = Depends(require_admin_or_session()),
    db: AsyncSession = Depends(get_async_db),
):
    """Delete a model from an Ollama backend (direct call, no sidecar)."""
    backend = await crud.get_backend_by_id(db, backend_id)
    if not backend:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Backend not found")
    if backend.engine != BackendEngine.OLLAMA:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Backend is not an Ollama engine")
    if not backend.url:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Backend has no URL configured")

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
            resp = await client.request(
                "DELETE",
                f"{backend.url}/api/delete",
                json={"name": request.model},
            )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to reach Ollama at {backend.url}: {exc}",
        )

    if resp.status_code == 404:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found on this backend")
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=f"Ollama delete failed: {resp.text}")

    # Refresh all Ollama backends on this node (they share a models folder)
    await _refresh_node_ollama_backends(backend.node_id)

    logger.info(
        "ollama_model_deleted",
        admin_id=admin.id,
        backend_id=backend_id,
        model=request.model,
    )

    return {"status": "success", "model": request.model}


@router.get("/backends", response_model=List[BackendResponse])
async def list_backends(
    admin: User = Depends(require_admin()),
    db: AsyncSession = Depends(get_async_db),
):
    """List all backends."""
    backends = await crud.get_all_backends(db)

    return [
        BackendResponse(
            id=b.id,
            name=b.name,
            url=b.url,
            engine=b.engine.value,
            status=b.status.value,
            max_concurrent=b.max_concurrent,
            current_concurrent=b.current_concurrent,
            gpu_memory_gb=b.gpu_memory_gb,
            gpu_type=b.gpu_type,
            node_id=b.node_id,
            node_name=b.node.name if b.node else None,
            gpu_indices=b.gpu_indices,
            version=b.version,
            last_health_check=b.last_health_check,
        )
        for b in backends
    ]


@router.patch("/backends/{backend_id}", response_model=BackendResponse)
async def update_backend(
    backend_id: int,
    request: BackendUpdateRequest,
    admin: User = Depends(require_admin()),
    db: AsyncSession = Depends(get_async_db),
):
    """Update an existing backend. Only provided fields are changed."""
    backend = await crud.get_backend_by_id(db, backend_id)
    if not backend:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Backend not found",
        )

    # Build kwargs from the request, only including explicitly set fields
    raw = request.model_dump(exclude_unset=True)
    if not raw:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No fields to update",
        )

    # Determine which fields to explicitly clear (set to null in JSON)
    all_fields = request.model_dump()
    clear_fields = [k for k in raw if all_fields[k] is None]
    kwargs = {k: v for k, v in raw.items() if v is not None}
    if clear_fields:
        kwargs["_clear_fields"] = clear_fields

    registry = get_registry()
    try:
        updated = await registry.update_backend(backend_id, **kwargs)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        error_msg = str(e).lower()
        if "duplicate" in error_msg or "unique" in error_msg:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Unique constraint violation: {e}",
            )
        raise

    logger.info(
        "backend_updated_by_admin",
        admin_id=admin.id,
        backend_id=backend_id,
        fields=list(raw.keys()),
    )

    return BackendResponse(
        id=updated.id,
        name=updated.name,
        url=updated.url,
        engine=updated.engine.value,
        status=updated.status.value,
        max_concurrent=updated.max_concurrent,
        current_concurrent=updated.current_concurrent,
        gpu_memory_gb=updated.gpu_memory_gb,
        gpu_type=updated.gpu_type,
        node_id=updated.node_id,
        node_name=updated.node.name if updated.node else None,
        gpu_indices=updated.gpu_indices,
        version=updated.version,
        last_health_check=updated.last_health_check,
    )


@router.patch("/nodes/{node_id}", response_model=NodeResponse)
async def update_node(
    node_id: int,
    request: NodeUpdateRequest,
    admin: User = Depends(require_admin()),
    db: AsyncSession = Depends(get_async_db),
):
    """Update an existing node. Only provided fields are changed."""
    node = await crud.get_node_by_id(db, node_id)
    if not node:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Node not found",
        )

    raw = request.model_dump(exclude_unset=True)
    if not raw:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No fields to update",
        )

    all_fields = request.model_dump()
    clear_fields = [k for k in raw if all_fields[k] is None]
    kwargs = {k: v for k, v in raw.items() if v is not None}
    if clear_fields:
        kwargs["_clear_fields"] = clear_fields

    registry = get_registry()
    try:
        updated = await registry.update_node(node_id, **kwargs)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        error_msg = str(e).lower()
        if "duplicate" in error_msg or "unique" in error_msg:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Unique constraint violation: {e}",
            )
        raise

    logger.info(
        "node_updated_by_admin",
        admin_id=admin.id,
        node_id=node_id,
        fields=list(raw.keys()),
    )

    return NodeResponse(
        id=updated.id,
        name=updated.name,
        hostname=updated.hostname,
        sidecar_url=updated.sidecar_url,
        sidecar_key_set=bool(updated.sidecar_key),
        status=updated.status.value,
        gpu_count=updated.gpu_count,
        driver_version=updated.driver_version,
        cuda_version=updated.cuda_version,
        sidecar_version=updated.sidecar_version,
    )


# Node Management
@router.post("/nodes/register", response_model=NodeResponse)
async def register_node(
    request: NodeRegisterRequest,
    admin: User = Depends(require_admin()),
    db: AsyncSession = Depends(get_async_db),
):
    """Register a new physical node."""
    existing = await crud.get_node_by_name(db, request.name)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Node with name '{request.name}' already exists",
        )

    registry = get_registry()
    node = await registry.register_node(
        name=request.name,
        hostname=request.hostname,
        sidecar_url=request.sidecar_url,
        sidecar_key=request.sidecar_key,
    )

    logger.info(
        "node_registered_by_admin",
        admin_id=admin.id,
        node_id=node.id,
        name=node.name,
    )

    return NodeResponse(
        id=node.id,
        name=node.name,
        hostname=node.hostname,
        sidecar_url=node.sidecar_url,
        sidecar_key_set=bool(node.sidecar_key),
        status=node.status.value,
        gpu_count=node.gpu_count,
        driver_version=node.driver_version,
        cuda_version=node.cuda_version,
        sidecar_version=node.sidecar_version,
    )


@router.get("/nodes", response_model=List[NodeResponse])
async def list_nodes(
    admin: User = Depends(require_admin()),
    db: AsyncSession = Depends(get_async_db),
):
    """List all nodes."""
    nodes = await crud.get_all_nodes(db)
    return [
        NodeResponse(
            id=n.id,
            name=n.name,
            hostname=n.hostname,
            sidecar_url=n.sidecar_url,
            sidecar_key_set=bool(n.sidecar_key),
            status=n.status.value,
            gpu_count=n.gpu_count,
            driver_version=n.driver_version,
            cuda_version=n.cuda_version,
            sidecar_version=n.sidecar_version,
        )
        for n in nodes
    ]


@router.delete("/nodes/{node_id}")
async def delete_node(
    node_id: int,
    admin: User = Depends(require_admin()),
    db: AsyncSession = Depends(get_async_db),
):
    """Delete a node (fails if backends still reference it)."""
    node = await crud.get_node_by_id(db, node_id)
    if not node:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Node not found",
        )

    registry = get_registry()
    removed = await registry.remove_node(node_id)
    if not removed:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Cannot delete node with active backends. Remove backends first.",
        )

    return {"status": "deleted", "node_id": node_id}


@router.post("/nodes/{node_id}/refresh")
async def refresh_node(
    node_id: int,
    admin: User = Depends(require_admin()),
    db: AsyncSession = Depends(get_async_db),
):
    """Force refresh sidecar data for a node."""
    node = await crud.get_node_by_id(db, node_id)
    if not node:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Node not found",
        )

    registry = get_registry()
    success = await registry.refresh_node(node_id)
    if success:
        return {"status": "refreshed", "node_id": node_id}
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Node has no sidecar configured",
        )


# Queue Management
@router.get("/queue")
async def get_queue(
    admin: User = Depends(require_admin()),
):
    """Get scheduler queue statistics."""
    scheduler = get_scheduler()
    stats = await scheduler.get_stats()

    return {
        "queue": stats["queue"],
        "fair_share": stats["fair_share"],
        "backend_queues": stats.get("backend_queues", {}),
    }


# Audit Search
@router.get("/audit/search")
async def search_audit(
    user_id: Optional[int] = Query(None),
    model: Optional[str] = Query(None),
    status: Optional[RequestStatus] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    search_text: Optional[str] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    admin: User = Depends(require_admin()),
    db: AsyncSession = Depends(get_async_db),
):
    """
    Search audit logs.

    Supports filtering by user, model, status, date range, and text search.
    """
    requests, total = await crud.search_requests(
        db=db,
        user_id=user_id,
        model=model,
        status=status,
        start_date=start_date,
        end_date=end_date,
        search_text=search_text,
        skip=skip,
        limit=limit,
    )

    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "results": [
            {
                "id": r.id,
                "request_uuid": r.request_uuid,
                "user_id": r.user_id,
                "endpoint": r.endpoint,
                "model": r.model,
                "status": r.status.value,
                "is_streaming": r.is_streaming,
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "total_time_ms": r.total_time_ms,
                "created_at": r.created_at.isoformat(),
                "backend_id": r.backend_id,
            }
            for r in requests
        ],
    }


@router.get("/audit/{request_id}")
async def get_audit_detail(
    request_id: int,
    admin: User = Depends(require_admin()),
    db: AsyncSession = Depends(get_async_db),
):
    """Get full details for an audit record including prompt and response."""
    from sqlalchemy import select
    from backend.app.db.models import Request, Response

    result = await db.execute(
        select(Request).where(Request.id == request_id)
    )
    request = result.scalar_one_or_none()

    if not request:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Request not found",
        )

    # Get response
    result = await db.execute(
        select(Response).where(Response.request_id == request_id)
    )
    response = result.scalar_one_or_none()

    return {
        "request": {
            "id": request.id,
            "request_uuid": request.request_uuid,
            "user_id": request.user_id,
            "api_key_id": request.api_key_id,
            "endpoint": request.endpoint,
            "model": request.model,
            "modality": request.modality.value,
            "is_streaming": request.is_streaming,
            "messages": request.messages,
            "prompt": request.prompt,
            "parameters": request.parameters,
            "response_format": request.response_format,
            "status": request.status.value,
            "backend_id": request.backend_id,
            "queued_at": request.queued_at.isoformat() if request.queued_at else None,
            "started_at": request.started_at.isoformat() if request.started_at else None,
            "completed_at": request.completed_at.isoformat() if request.completed_at else None,
            "queue_delay_ms": request.queue_delay_ms,
            "processing_time_ms": request.processing_time_ms,
            "total_time_ms": request.total_time_ms,
            "prompt_tokens": request.prompt_tokens,
            "completion_tokens": request.completion_tokens,
            "error_message": request.error_message,
            "client_ip": request.client_ip,
        },
        "response": {
            "content": response.content if response else None,
            "finish_reason": response.finish_reason if response else None,
            "chunk_count": response.chunk_count if response else 0,
            "structured_output_valid": response.structured_output_valid if response else None,
            "validation_errors": response.validation_errors if response else None,
        } if response else None,
    }


# User Management
@router.get("/users")
async def list_users(
    group_id: Optional[int] = Query(None),
    search: Optional[str] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    admin: User = Depends(require_admin()),
    db: AsyncSession = Depends(get_async_db),
):
    """List all users with optional group filter and search."""
    users, total = await crud.get_users(db, skip=skip, limit=limit, group_id=group_id, search=search)

    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "users": [
            {
                "id": u.id,
                "uuid": u.uuid,
                "username": u.username,
                "email": u.email,
                "full_name": u.full_name,
                "role": u.role.value,
                "group_id": u.group_id,
                "group_name": u.group.display_name if u.group else None,
                "is_active": u.is_active,
                "created_at": u.created_at.isoformat(),
                "last_login_at": u.last_login_at.isoformat() if u.last_login_at else None,
            }
            for u in users
        ],
    }


# Quota Request Management
@router.get("/quota-requests")
async def list_quota_requests(
    status: Optional[str] = Query(None),
    admin: User = Depends(require_admin()),
    db: AsyncSession = Depends(get_async_db),
):
    """List pending quota requests."""
    requests = await crud.get_pending_quota_requests(db)

    return {
        "requests": [
            {
                "id": r.id,
                "user_id": r.user_id,
                "requester_name": r.requester_name,
                "requester_email": r.requester_email,
                "affiliation": r.affiliation,
                "request_type": r.request_type,
                "justification": r.justification,
                "requested_tokens": r.requested_tokens,
                "status": r.status.value,
                "created_at": r.created_at.isoformat(),
            }
            for r in requests
        ]
    }


class QuotaReviewRequest(BaseModel):
    """Request to review a quota request."""
    approved: bool
    notes: Optional[str] = None
    granted_tokens: Optional[int] = None


@router.post("/quota-requests/{request_id}/review")
async def review_quota_request(
    request_id: int,
    review: QuotaReviewRequest,
    admin: User = Depends(require_admin()),
    db: AsyncSession = Depends(get_async_db),
):
    """Approve or deny a quota request."""
    from backend.app.db.models import QuotaRequestStatus

    status = QuotaRequestStatus.APPROVED if review.approved else QuotaRequestStatus.DENIED

    quota_request = await crud.review_quota_request(
        db=db,
        request_id=request_id,
        reviewer_id=admin.id,
        status=status,
        review_notes=review.notes,
    )

    if not quota_request:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Quota request not found",
        )

    logger.info(
        "quota_request_reviewed",
        admin_id=admin.id,
        request_id=request_id,
        approved=review.approved,
    )

    return {"status": "reviewed", "approved": review.approved}


# User & API Key Provisioning
class CreateUserRequest(BaseModel):
    """Request to create a new user."""
    username: str = Field(..., min_length=1, max_length=100)
    email: str = Field(..., min_length=3, max_length=255)
    password: str = Field(..., min_length=8)
    group_id: int
    role: UserRole = UserRole.STUDENT  # kept for backward compat during migration
    full_name: Optional[str] = None
    college: Optional[str] = None
    department: Optional[str] = None
    intended_use: Optional[str] = None


class CreateUserResponse(BaseModel):
    """Response after creating a user."""
    id: int
    uuid: str
    username: str
    email: str
    role: str
    group_id: int
    group_name: Optional[str] = None
    full_name: Optional[str]
    is_active: bool


class CreateApiKeyRequest(BaseModel):
    """Request to create an API key for a user."""
    name: str = Field(..., min_length=1, max_length=100)
    expires_at: Optional[datetime] = None


class CreateApiKeyResponse(BaseModel):
    """Response after creating an API key (full_key only available at creation)."""
    id: int
    key_prefix: str
    full_key: str
    name: str


@router.post("/users", response_model=CreateUserResponse)
async def create_user(
    request: CreateUserRequest,
    admin: User = Depends(require_admin()),
    db: AsyncSession = Depends(get_async_db),
):
    """Create a new user with quota defaults from their group."""
    # Validate group exists
    group = await crud.get_group_by_id(db, request.group_id)
    if not group:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Group with id {request.group_id} not found",
        )

    # Check for duplicate username
    existing = await crud.get_user_by_username(db, request.username)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Username '{request.username}' already exists",
        )

    # Check for duplicate email
    existing = await crud.get_user_by_email(db, request.email)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Email '{request.email}' already exists",
        )

    # Hash password and create user
    pw_hash = hash_password(request.password)
    user = await crud.create_user(
        db,
        username=request.username,
        email=request.email,
        password_hash=pw_hash,
        role=request.role,
        full_name=request.full_name,
        group_id=request.group_id,
        college=request.college,
        department=request.department,
        intended_use=request.intended_use,
    )

    # Create quota with group defaults
    await crud.create_quota(
        db,
        user_id=user.id,
        rpm_limit=group.rpm_limit,
        max_concurrent=group.max_concurrent,
    )

    await db.commit()

    logger.info(
        "user_created_by_admin",
        admin_id=admin.id,
        user_id=user.id,
        username=user.username,
        group=group.name,
    )

    return CreateUserResponse(
        id=user.id,
        uuid=user.uuid,
        username=user.username,
        email=user.email,
        role=user.role.value,
        group_id=group.id,
        group_name=group.display_name,
        full_name=user.full_name,
        is_active=user.is_active,
    )


@router.post("/users/{user_id}/api-keys", response_model=CreateApiKeyResponse)
async def create_user_api_key(
    user_id: int,
    request: CreateApiKeyRequest,
    admin: User = Depends(require_admin()),
    db: AsyncSession = Depends(get_async_db),
):
    """Create an API key for a user. The full key is only returned once."""
    # Verify user exists
    user = await crud.get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    # Generate key
    full_key, key_hash, key_prefix = generate_api_key()

    # Create record
    api_key = await crud.create_api_key(
        db,
        user_id=user_id,
        key_hash=key_hash,
        key_prefix=key_prefix,
        name=request.name,
        expires_at=request.expires_at,
    )

    await db.commit()

    logger.info(
        "api_key_created_by_admin",
        admin_id=admin.id,
        user_id=user_id,
        api_key_id=api_key.id,
        key_prefix=key_prefix,
    )

    return CreateApiKeyResponse(
        id=api_key.id,
        key_prefix=key_prefix,
        full_key=full_key,
        name=api_key.name,
    )


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    admin: User = Depends(require_admin()),
    db: AsyncSession = Depends(get_async_db),
):
    """Hard-delete a user and all associated data. Cannot delete yourself."""
    # Prevent self-deletion
    if user_id == admin.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account",
        )

    # Verify user exists
    user = await crud.get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    deleted = await crud.delete_user(db, user_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete user",
        )

    await db.commit()

    logger.info(
        "user_deleted_by_admin",
        admin_id=admin.id,
        deleted_user_id=user_id,
        deleted_username=user.username,
    )

    return {"status": "deleted", "user_id": user_id}


# User Detail
@router.get("/users/{user_id}")
async def get_user_detail(
    user_id: int,
    admin: User = Depends(require_admin()),
    db: AsyncSession = Depends(get_async_db),
):
    """Get full user profile with usage stats, API keys, and monthly usage."""
    stats = await crud.get_user_with_stats(db, user_id)
    if not stats:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    user = stats["user"]
    monthly_usage = await crud.get_user_monthly_usage(db, user_id)

    # Per-key token totals
    api_key_ids = [k.id for k in stats["api_keys"]]
    key_token_totals = await crud.get_api_key_token_totals(db, api_key_ids)

    return {
        "user": {
            "id": user.id,
            "uuid": user.uuid,
            "username": user.username,
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role.value,
            "group_id": user.group_id,
            "group_name": user.group.display_name if user.group else None,
            "college": user.college,
            "department": user.department,
            "intended_use": user.intended_use,
            "is_active": user.is_active,
            "created_at": user.created_at.isoformat(),
            "last_login_at": user.last_login_at.isoformat() if user.last_login_at else None,
        },
        "stats": {
            "total_tokens": stats["total_tokens"],
            "prompt_tokens": stats["prompt_tokens"],
            "completion_tokens": stats["completion_tokens"],
            "request_count": stats["request_count"],
            "favorite_models": [{"model": m, "count": c} for m, c in stats["favorite_models"]],
            "api_key_count": stats["api_key_count"],
        },
        "quota": {
            "token_budget": stats["user"].group.token_budget if stats["user"].group else 0,
            "tokens_used": stats["quota"].tokens_used,
            "rpm_limit": stats["quota"].rpm_limit,
            "max_concurrent": stats["quota"].max_concurrent,
            "weight_override": stats["quota"].weight_override,
        } if stats["quota"] else None,
        "api_keys": [
            {
                "id": k.id,
                "key_prefix": k.key_prefix,
                "name": k.name,
                "status": k.status.value,
                "last_used_at": k.last_used_at.isoformat() if k.last_used_at else None,
                "usage_count": k.usage_count,
                "created_at": k.created_at.isoformat(),
                "token_totals": key_token_totals.get(k.id, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
            }
            for k in stats["api_keys"]
        ],
        "monthly_usage": monthly_usage,
    }


class UpdateUserRequest(BaseModel):
    """Request to update a user's profile."""
    group_id: Optional[int] = None
    full_name: Optional[str] = None
    college: Optional[str] = None
    department: Optional[str] = None
    intended_use: Optional[str] = None
    is_active: Optional[bool] = None


@router.patch("/users/{user_id}")
async def update_user(
    user_id: int,
    request: UpdateUserRequest,
    admin: User = Depends(require_admin()),
    db: AsyncSession = Depends(get_async_db),
):
    """Update user profile, group, or status."""
    user = await crud.get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    updates = request.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No fields to update",
        )

    # Validate group if changing
    if "group_id" in updates and updates["group_id"] is not None:
        group = await crud.get_group_by_id(db, updates["group_id"])
        if not group:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Group with id {updates['group_id']} not found",
            )

    updated = await crud.update_user(db, user_id, **updates)
    await db.commit()

    return {"status": "updated", "user_id": user_id}


# Group Management
class CreateGroupRequest(BaseModel):
    """Request to create a group."""
    name: str = Field(..., min_length=1, max_length=50)
    display_name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    token_budget: int = Field(default=100000, ge=0)
    rpm_limit: int = Field(default=30, ge=1)
    max_concurrent: int = Field(default=2, ge=1)
    scheduler_weight: int = Field(default=1, ge=1)
    is_admin: bool = False


class UpdateGroupRequest(BaseModel):
    """Request to update a group."""
    display_name: Optional[str] = None
    description: Optional[str] = None
    token_budget: Optional[int] = Field(None, ge=0)
    rpm_limit: Optional[int] = Field(None, ge=1)
    max_concurrent: Optional[int] = Field(None, ge=1)
    scheduler_weight: Optional[int] = Field(None, ge=1)
    is_admin: Optional[bool] = None


@router.get("/groups")
async def list_groups(
    admin: User = Depends(require_admin()),
    db: AsyncSession = Depends(get_async_db),
):
    """List all groups with user counts."""
    groups_with_counts = await crud.get_all_groups_with_counts(db)
    return {
        "groups": [
            {
                "id": g.id,
                "name": g.name,
                "display_name": g.display_name,
                "description": g.description,
                "token_budget": g.token_budget,
                "rpm_limit": g.rpm_limit,
                "max_concurrent": g.max_concurrent,
                "scheduler_weight": g.scheduler_weight,
                "is_admin": g.is_admin,
                "user_count": count,
            }
            for g, count in groups_with_counts
        ],
    }


@router.post("/groups")
async def create_group(
    request: CreateGroupRequest,
    admin: User = Depends(require_admin()),
    db: AsyncSession = Depends(get_async_db),
):
    """Create a new group."""
    existing = await crud.get_group_by_name(db, request.name)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Group with name '{request.name}' already exists",
        )

    group = await crud.create_group(
        db,
        name=request.name,
        display_name=request.display_name,
        description=request.description,
        token_budget=request.token_budget,
        rpm_limit=request.rpm_limit,
        max_concurrent=request.max_concurrent,
        scheduler_weight=request.scheduler_weight,
        is_admin=request.is_admin,
    )
    await db.commit()

    logger.info("group_created", admin_id=admin.id, group_id=group.id, name=group.name)

    return {
        "id": group.id,
        "name": group.name,
        "display_name": group.display_name,
    }


@router.patch("/groups/{group_id}")
async def update_group(
    group_id: int,
    request: UpdateGroupRequest,
    admin: User = Depends(require_admin()),
    db: AsyncSession = Depends(get_async_db),
):
    """Update group defaults."""
    updates = request.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No fields to update",
        )

    group = await crud.update_group(db, group_id, **updates)
    if not group:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Group not found",
        )
    await db.commit()

    logger.info("group_updated", admin_id=admin.id, group_id=group_id, fields=list(updates.keys()))

    return {"status": "updated", "group_id": group_id}


@router.delete("/groups/{group_id}")
async def delete_group(
    group_id: int,
    admin: User = Depends(require_admin()),
    db: AsyncSession = Depends(get_async_db),
):
    """Delete a group (fails if users are assigned)."""
    deleted = await crud.delete_group(db, group_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Cannot delete group with active users. Reassign users first.",
        )
    await db.commit()

    logger.info("group_deleted", admin_id=admin.id, group_id=group_id)

    return {"status": "deleted", "group_id": group_id}


# API Keys listing
@router.get("/api-keys")
async def list_api_keys(
    search: Optional[str] = Query(None),
    key_status: Optional[str] = Query(None, alias="status"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    admin: User = Depends(require_admin()),
    db: AsyncSession = Depends(get_async_db),
):
    """List all API keys with user info."""
    keys, total = await crud.get_all_api_keys(
        db, skip=skip, limit=limit, search=search, status_filter=key_status
    )

    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "api_keys": [
            {
                "id": k.id,
                "key_prefix": k.key_prefix,
                "name": k.name,
                "status": k.status.value,
                "user_id": k.user_id,
                "username": k.user.username if k.user else None,
                "last_used_at": k.last_used_at.isoformat() if k.last_used_at else None,
                "usage_count": k.usage_count,
                "created_at": k.created_at.isoformat(),
            }
            for k in keys
        ],
    }
