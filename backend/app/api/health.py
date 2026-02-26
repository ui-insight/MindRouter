############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# health.py: Health check and Prometheus metrics endpoints
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Health check and metrics endpoints."""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict

from fastapi import APIRouter, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from sqlalchemy import and_, func, select

from backend.app.core import redis_client
from backend.app.core.scheduler.policy import get_scheduler
from backend.app.core.telemetry.registry import get_registry
from backend.app.db.models import Request as DBRequest, RequestStatus
from backend.app.db.session import AsyncSessionLocal
from backend.app.settings import get_settings

router = APIRouter(tags=["health"])

# Prometheus metrics
REQUEST_COUNT = Counter(
    "mindrouter_requests_total",
    "Total number of requests",
    ["endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "mindrouter_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
)
QUEUE_SIZE = Gauge(
    "mindrouter_queue_size",
    "Current queue size",
)
ACTIVE_BACKENDS = Gauge(
    "mindrouter_active_backends",
    "Number of healthy backends",
)
TOKENS_PROCESSED = Counter(
    "mindrouter_tokens_total",
    "Total tokens processed",
    ["type"],  # prompt, completion
)


@router.get("/healthz")
async def liveness_probe() -> Dict[str, str]:
    """
    Liveness probe - checks if the application is running.

    Returns 200 if the application is alive.
    """
    return {"status": "alive", "timestamp": datetime.now(timezone.utc).isoformat()}


@router.get("/readyz")
async def readiness_probe() -> Dict[str, Any]:
    """
    Readiness probe - checks if the application is ready to serve traffic.

    Checks:
    - Database connectivity
    - At least one healthy backend available
    """
    checks = {
        "database": False,
        "backends": False,
    }

    # Check database
    try:
        async with AsyncSessionLocal() as db:
            await db.execute("SELECT 1")
            checks["database"] = True
    except Exception:
        pass

    # Check backends
    try:
        registry = get_registry()
        backends = await registry.get_healthy_backends()
        checks["backends"] = len(backends) > 0
    except Exception:
        pass

    all_ready = all(checks.values())

    return {
        "status": "ready" if all_ready else "not_ready",
        "checks": checks,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/metrics")
async def prometheus_metrics() -> Response:
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus text format.
    """
    try:
        # Update dynamic metrics
        scheduler = get_scheduler()
        stats = await scheduler.get_stats()
        QUEUE_SIZE.set(stats["queue"]["total"])

        registry = get_registry()
        backends = await registry.get_healthy_backends()
        ACTIVE_BACKENDS.set(len(backends))

    except Exception:
        pass

    # Generate metrics
    metrics = generate_latest()
    return Response(content=metrics, media_type=CONTENT_TYPE_LATEST)


@router.get("/status")
async def cluster_status() -> Dict[str, Any]:
    """
    Get cluster status summary.

    Returns high-level status information about the cluster.
    """
    settings = get_settings()

    # Get scheduler stats
    try:
        scheduler = get_scheduler()
        scheduler_stats = await scheduler.get_stats()
    except Exception:
        scheduler_stats = {"error": "unavailable"}

    # Get backend info
    try:
        registry = get_registry()
        all_backends = await registry.get_all_backends()
        healthy_backends = await registry.get_healthy_backends()
    except Exception:
        all_backends = []
        healthy_backends = []

    # Collect model names from healthy backends
    models = set()
    for backend in healthy_backends:
        try:
            backend_models = await registry.get_backend_models(backend.id)
            for m in backend_models:
                models.add(m.name)
        except Exception:
            pass

    # Active users in last 24 hours
    active_users = 0
    try:
        from backend.app.db.session import get_async_db
        from backend.app.db import crud
        async for db in get_async_db():
            active_users = await crud.get_active_user_count(db)
            break
    except Exception:
        pass

    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "backends": {
            "total": len(all_backends),
            "healthy": len(healthy_backends),
        },
        "models": sorted(models),
        "queue": scheduler_stats.get("queue", {}),
        "fair_share": {
            "total_users": scheduler_stats.get("fair_share", {}).get("total_users", 0),
        },
        "active_users": active_users,
    }


@router.get("/api/cluster/throughput")
async def cluster_throughput() -> Dict[str, Any]:
    """
    Public endpoint: cluster-wide token throughput.

    Returns tokens/second computed from completed requests in the last 10 seconds.
    """
    window_seconds = 10
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=window_seconds)

    try:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(
                    func.coalesce(func.sum(DBRequest.total_tokens), 0),
                    func.count(DBRequest.id),
                ).where(
                    and_(
                        DBRequest.status == RequestStatus.COMPLETED,
                        DBRequest.completed_at >= cutoff,
                    )
                )
            )
            row = result.one()
            total_tokens = int(row[0])
            request_count = int(row[1])
    except Exception:
        total_tokens = 0
        request_count = 0

    # Add inflight streaming tokens to the total
    inflight_tokens = await redis_client.get_inflight_tokens()
    combined_tokens = total_tokens + inflight_tokens
    tokens_per_second = round(combined_tokens / window_seconds, 1)

    # Get active request count from scheduler
    try:
        scheduler = get_scheduler()
        stats = await scheduler.get_stats()
        active_requests = stats.get("queue", {}).get("total", 0)
    except Exception:
        active_requests = 0

    return {
        "tokens_per_second": tokens_per_second,
        "requests_per_minute": request_count,
        "active_requests": active_requests,
        "total_tokens_last_5s": total_tokens,
        "inflight_tokens": inflight_tokens,
    }
