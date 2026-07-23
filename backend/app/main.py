############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# main.py: FastAPI application entry point and configuration
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""FastAPI application entry point."""

import asyncio
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.app.api import api_router
from backend.app.core.redis_client import (
    close_redis,
    get_all_token_keys,
    init_redis,
    is_available as redis_is_available,
    reset_tokens as redis_reset_tokens,
    set_tokens as redis_set_tokens,
)
from backend.app.core.scheduler.policy import init_scheduler, shutdown_scheduler
from backend.app.core.telemetry.registry import init_registry, shutdown_registry
from backend.app.dashboard.blog import blog_router
from backend.app.dashboard.dlp_routes import dlp_router
from backend.app.dashboard.email_routes import email_router
from backend.app.dashboard.chat import chat_router
from backend.app.dashboard.images import images_router
from backend.app.dashboard.video import video_router
from backend.app.dashboard.routes import dashboard_router
from backend.app.logging_config import (
    bind_request_context,
    clear_request_context,
    get_logger,
    setup_logging,
)
from backend.app.db.session import get_async_db_context
from backend.app.settings import get_settings
from backend.app.storage.artifacts import get_artifact_storage
from backend.app.core.otel import setup_telemetry, shutdown_telemetry, instrument_app

# Setup logging first
setup_logging()

# Initialize OpenTelemetry (before app creation so httpx/redis/sqlalchemy get instrumented)
setup_telemetry()
logger = get_logger(__name__)


async def _retention_loop() -> None:
    """Background loop for tiered data retention (archive + cleanup).

    Cross-worker serialization and lock management live in
    ``try_run_retention_with_lock`` so this loop stays simple: sleep,
    attempt cycle, repeat.  If another worker (or a manual trigger)
    is already running a cycle, the helper returns a skipped summary
    and we try again after the next interval.
    """
    from backend.app.services.retention import (
        get_retention_config,
        try_run_retention_with_lock,
    )

    while True:
        try:
            async with get_async_db_context() as db:
                config = await get_retention_config(db)
            interval = config.get("retention.cleanup_interval", 3600)

            await asyncio.sleep(interval)

            await try_run_retention_with_lock("scheduled")

        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("retention_cycle_error")


async def _cleanup_orphaned_requests() -> None:
    """Mark stale 'queued' requests as failed on startup.

    Any request still in 'queued' status from before this app instance
    started is orphaned — the in-memory scheduler queue was lost on
    restart, so these will never be routed.
    """
    try:
        from backend.app.db.models import Request, RequestStatus
        from sqlalchemy import update

        # Requests older than 5 minutes in queued status are definitely orphaned
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=5)
        async with get_async_db_context() as db:
            result = await db.execute(
                update(Request)
                .where(Request.status == RequestStatus.QUEUED)
                .where(Request.created_at < cutoff)
                .values(
                    status=RequestStatus.FAILED,
                    error_message="Orphaned: request was still queued after app restart",
                )
            )
            await db.commit()
            if result.rowcount > 0:
                logger.info("orphaned_requests_cleaned", count=result.rowcount)
    except Exception:
        logger.exception("orphaned_request_cleanup_failed")


async def _seed_redis_from_db() -> None:
    """Seed per-user Redis token counters from actual request data on startup.

    Computes each user's current-period token usage directly from the
    requests table rather than trusting ``quotas.tokens_used``, which can
    carry stale values from a previous period due to the Redis↔DB sync
    loop.  Also resets expired budget periods and cleans up orphan Redis
    keys.
    """
    if not redis_is_available():
        return
    try:
        from backend.app.db.models import Quota, Request
        from sqlalchemy import select, func

        now = datetime.now(timezone.utc)

        async with get_async_db_context() as db:
            result = await db.execute(select(Quota))
            quotas = list(result.scalars().all())

            valid_user_ids = set()
            seeded = 0
            reset_count = 0

            for quota in quotas:
                valid_user_ids.add(quota.user_id)
                period_end = quota.budget_period_start
                if period_end.tzinfo is None:
                    period_end = period_end.replace(tzinfo=timezone.utc)
                period_end = period_end + timedelta(days=quota.budget_period_days)

                if now >= period_end:
                    quota.budget_period_start = now
                    quota.tokens_used = 0
                    await redis_set_tokens(quota.user_id, 0)
                    reset_count += 1
                else:
                    row = await db.execute(
                        select(func.coalesce(func.sum(Request.total_tokens), 0))
                        .where(
                            Request.user_id == quota.user_id,
                            Request.created_at >= quota.budget_period_start,
                            Request.total_tokens.isnot(None),
                        )
                    )
                    period_tokens = int(row.scalar())
                    await redis_set_tokens(quota.user_id, period_tokens)
                    quota.tokens_used = period_tokens

                seeded += 1

            await db.commit()

        all_redis = await get_all_token_keys()
        orphans = set(all_redis.keys()) - valid_user_ids
        for orphan_id in orphans:
            await redis_reset_tokens(orphan_id)
        if orphans:
            logger.info("redis_orphan_keys_cleaned", count=len(orphans))

        logger.info(
            "redis_seeded_from_requests",
            users=seeded,
            periods_reset=reset_count,
        )
    except Exception:
        logger.exception("redis_seed_failed")


async def _warm_page_caches(force_seed: bool = False) -> None:
    """Pre-warm Redis caches for expensive dashboard queries.

    Called on startup (with *force_seed=True*) and periodically by
    _cache_warm_loop to ensure page loads never trigger full table scans.

    When *force_seed* is True the cluster token counter is seeded from
    the larger of the live DB formula and the persisted high-water mark
    (``stats.cluster_hwm``).  This ensures the counter never drops on
    restart even if the DB formula has archival gaps.
    """
    if not redis_is_available():
        return
    try:
        import json as _json
        from backend.app.core import redis_client as _rc
        from backend.app.db import crud

        async with get_async_db_context() as db:
            # Model token totals (used by /models popularity chart)
            token_totals = await crud.get_model_token_totals(db, limit=15)
            if _rc._redis:
                await _rc._redis.set(
                    "cache:model_token_totals",
                    _json.dumps(token_totals),
                    ex=3600,
                )

            # Global token total
            existing = await _rc.get_cluster_tokens()
            if existing is None or force_seed:
                totals = await crud.get_global_token_total(db, include_offset=False)
                seed_total = totals["total_tokens"]

                hwm = await crud.get_config_json(db, "stats.cluster_hwm", 0)
                if hwm and int(hwm) > seed_total:
                    logger.info(
                        "cluster_seed_using_hwm",
                        db_total=seed_total,
                        hwm=int(hwm),
                    )
                    seed_total = int(hwm)

                await _rc.seed_cluster_tokens(
                    totals["prompt_tokens"],
                    totals["completion_tokens"],
                    seed_total,
                )

        logger.info("page_caches_warmed")
    except Exception:
        logger.exception("page_cache_warm_failed")


async def _cache_warm_loop() -> None:
    """Background loop: warm caches immediately on start, then every 30 min."""
    try:
        await _warm_page_caches(force_seed=True)
    except Exception:
        logger.exception("cache_warm_initial_error")
    while True:
        try:
            await asyncio.sleep(1800)
            await _warm_page_caches()
        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("cache_warm_loop_error")


async def _redis_sync_loop() -> None:
    """Background loop: sync Redis token counters to DB every 60s.

    Writes current Redis values into ``quotas.tokens_used`` for durability.
    Also persists the cluster-wide token counter as a high-water mark in
    ``app_config`` so it survives process restarts without losing state.
    Only syncs keys that correspond to existing quota rows (ignores orphans).
    Orphan cleanup happens on startup in ``_seed_redis_from_db``.
    """
    from backend.app.core import redis_client as _rc

    while True:
        try:
            await asyncio.sleep(60)
            if not redis_is_available():
                continue
            token_map = await get_all_token_keys()
            if not token_map:
                continue
            from backend.app.db.models import Quota
            from sqlalchemy import select, update

            async with get_async_db_context() as db:
                # Get valid user_ids that have quota rows
                result = await db.execute(select(Quota.user_id))
                valid_ids = {row[0] for row in result.all()}

                for user_id, tokens in token_map.items():
                    if user_id in valid_ids:
                        await db.execute(
                            update(Quota)
                            .where(Quota.user_id == user_id)
                            .values(tokens_used=tokens)
                        )

                # Persist cluster counter high-water mark
                cluster = await _rc.get_cluster_tokens()
                if cluster and cluster["total_tokens"] > 0:
                    from backend.app.db import crud
                    await crud.set_config(
                        db, "stats.cluster_hwm",
                        cluster["total_tokens"],
                        description="Cluster token counter high-water mark",
                    )

                await db.commit()
            logger.debug("redis_sync_to_db", users=len(token_map))
        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("redis_sync_error")


async def _warm_trend_caches() -> None:
    """Pre-warm in-memory trend caches for expensive time ranges.

    The week/month/year trend queries scan millions of rows, so we run
    them once at startup (in the background) so the first user request
    hits the cache instead of waiting several minutes.  A random delay
    staggers the work across uvicorn workers to avoid saturating the
    DB connection pool.
    """
    import random
    await asyncio.sleep(random.uniform(5, 60))
    try:
        from backend.app.db import crud

        for range_name in ("week", "month", "year"):
            async with get_async_db_context() as db:
                await crud.get_token_trend(db, range_name)
                await crud.get_active_users_trend(db, range_name)
            logger.info("trend_cache_warmed", range=range_name)
    except Exception:
        logger.exception("trend_cache_warm_failed")


_cleanup_task: Optional[asyncio.Task] = None
_redis_sync_task: Optional[asyncio.Task] = None
_cache_warm_task: Optional[asyncio.Task] = None
_trend_warm_task: Optional[asyncio.Task] = None
_dlp_task: Optional[asyncio.Task] = None


@asynccontextmanager
async def _run_migrations() -> None:
    """Run 'alembic upgrade head' at startup (opt-in via RUN_MIGRATIONS=1).

    Alembic is synchronous, so run it in a worker thread. Fail fast on error —
    a clear migration failure beats a downstream "Table 'backends' doesn't
    exist" crash-loop. NOTE: with multiple workers, run single-worker on first
    boot (or migrate out-of-band) to avoid a concurrent-DDL race.
    """
    def _upgrade() -> None:
        from alembic import command
        from alembic.config import Config

        command.upgrade(Config("alembic.ini"), "head")

    logger.info("run_migrations_start")
    await asyncio.to_thread(_upgrade)
    logger.info("run_migrations_done")


async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan manager."""
    global _cleanup_task, _redis_sync_task, _dlp_task, _cache_warm_task, _trend_warm_task
    logger.info("Starting MindRouter...")

    # Opt-in: bring the schema to head before anything reads it (init_registry
    # queries the backends table). Prevents a crash-loop on a fresh DB.
    if get_settings().run_migrations:
        await _run_migrations()

    # Initialize components
    await init_registry()
    await init_scheduler()

    # Initialize storage
    storage = get_artifact_storage()
    await storage.initialize()

    # Clean up orphaned queued requests from previous runs
    await _cleanup_orphaned_requests()

    # Initialize Redis and seed counters from DB
    await init_redis()
    await _seed_redis_from_db()

    # Initialize timezone cache from DB
    from backend.app.db.session import AsyncSessionLocal
    from backend.app.dashboard.routes import _init_tz_cache
    async with AsyncSessionLocal() as db:
        await _init_tz_cache(db)

    # Initialize archive database if configured
    settings_ref = get_settings()
    if settings_ref.archive_database_url:
        try:
            from backend.app.db.archive_models import ArchiveBase
            from backend.app.db.session import get_archive_engine
            archive_engine = get_archive_engine()
            if archive_engine:
                async with archive_engine.begin() as conn:
                    await conn.run_sync(ArchiveBase.metadata.create_all)
                logger.info("archive_db_initialized")
        except Exception:
            logger.exception("archive_db_init_failed")

    # Start background loops
    _cleanup_task = asyncio.create_task(_retention_loop())
    if redis_is_available():
        _redis_sync_task = asyncio.create_task(_redis_sync_loop())
        _cache_warm_task = asyncio.create_task(_cache_warm_loop())

    # Pre-warm in-memory trend caches in background (takes several minutes)
    _trend_warm_task = asyncio.create_task(_warm_trend_caches())

    # Start DLP background worker
    from backend.app.services.dlp_worker import dlp_worker_loop
    _dlp_task = asyncio.create_task(dlp_worker_loop())

    # Start video generation runner (claims queued video jobs, drives the worker)
    _video_task = None
    if get_settings().video_runner_enabled:
        from backend.app.services.video_runner import run_video_runner_loop
        _video_task = asyncio.create_task(run_video_runner_loop())

    logger.info("MindRouter started successfully")

    yield

    # Shutdown
    logger.info("Shutting down MindRouter...")
    if _cleanup_task:
        _cleanup_task.cancel()
        try:
            await _cleanup_task
        except asyncio.CancelledError:
            pass
    if _redis_sync_task:
        _redis_sync_task.cancel()
        try:
            await _redis_sync_task
        except asyncio.CancelledError:
            pass
    if _cache_warm_task:
        _cache_warm_task.cancel()
        try:
            await _cache_warm_task
        except asyncio.CancelledError:
            pass
    if _trend_warm_task:
        _trend_warm_task.cancel()
        try:
            await _trend_warm_task
        except asyncio.CancelledError:
            pass
    if _dlp_task:
        _dlp_task.cancel()
        try:
            await _dlp_task
        except asyncio.CancelledError:
            pass
    if _video_task:
        _video_task.cancel()
        try:
            await _video_task
        except asyncio.CancelledError:
            pass
    # Final flush of Redis counters to DB before shutdown
    if redis_is_available():
        try:
            from backend.app.db.models import Quota
            from sqlalchemy import update as sa_update

            token_map = await get_all_token_keys()
            if token_map:
                async with get_async_db_context() as db:
                    for user_id, tokens in token_map.items():
                        await db.execute(
                            sa_update(Quota)
                            .where(Quota.user_id == user_id)
                            .values(tokens_used=tokens)
                        )
                    await db.commit()
                logger.info("redis_final_sync", users=len(token_map))
        except Exception:
            logger.exception("redis_final_sync_failed")
    await close_redis()
    # Close archive DB engine if initialized
    try:
        from backend.app.db.session import close_archive_engine
        await close_archive_engine()
    except Exception:
        pass
    await shutdown_scheduler()
    await shutdown_registry()
    shutdown_telemetry()
    logger.info("MindRouter shutdown complete")


class RequestIDMiddleware:
    """Raw ASGI middleware for request ID injection.

    Unlike @app.middleware("http") which wraps in BaseHTTPMiddleware,
    this does NOT run the handler in a separate task, so client disconnects
    won't cancel in-flight DB operations and leak connections.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Extract request ID from headers
        headers = dict(scope.get("headers", []))
        request_id = (
            headers.get(b"x-request-id", b"").decode()
            or str(uuid.uuid4())
        )

        bind_request_context(request_id=request_id)

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                # Inject X-Request-ID into response headers
                response_headers = list(message.get("headers", []))
                response_headers.append(
                    (b"x-request-id", request_id.encode())
                )
                message = {**message, "headers": response_headers}
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            clear_request_context()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="LLM Inference Load Balancer for Ollama and vLLM backends",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request ID middleware — raw ASGI to avoid BaseHTTPMiddleware's task
    # cancellation behavior which corrupts DB sessions on client disconnect.
    app.add_middleware(RequestIDMiddleware)

    # OpenTelemetry FastAPI auto-instrumentation
    instrument_app(app)

    # Exception handlers
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Global exception handler."""
        logger.exception("unhandled_exception", error=str(exc))
        return JSONResponse(
            status_code=500,
            content={"error": {"message": "Internal server error", "type": "server_error"}},
        )

    # Include routers
    app.include_router(api_router)
    app.include_router(dashboard_router)
    app.include_router(chat_router)
    app.include_router(images_router)
    app.include_router(video_router)
    app.include_router(blog_router)
    app.include_router(email_router)
    app.include_router(dlp_router)
    # Proxy /mcp/* to the standalone MCP service (single-worker process
    # that avoids SseServerTransport session-affinity issues).
    try:
        from backend.app.api.mcp_proxy import mcp_proxy_app
        app.mount("/mcp", mcp_proxy_app)
    except ImportError:
        logger.warning("mcp_proxy_disabled", reason="httpx not available")

    # Mount static files for dashboard
    import os
    static_path = os.path.join(os.path.dirname(__file__), "dashboard", "static")
    if os.path.exists(static_path):
        app.mount("/static", StaticFiles(directory=static_path), name="static")

    return app


# Create application instance
app = create_app()


def main():
    """Run the application using uvicorn."""
    import uvicorn

    settings = get_settings()

    uvicorn.run(
        "backend.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
