############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
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
    set_tokens as redis_set_tokens,
)
from backend.app.core.scheduler.policy import init_scheduler, shutdown_scheduler
from backend.app.core.telemetry.registry import init_registry, shutdown_registry
from backend.app.dashboard.blog import blog_router
from backend.app.dashboard.chat import chat_router
from backend.app.dashboard.routes import dashboard_router
from backend.app.logging_config import (
    bind_request_context,
    clear_request_context,
    get_logger,
    setup_logging,
)
from backend.app.db import chat_crud
from backend.app.db.session import get_async_db_context
from backend.app.settings import get_settings
from backend.app.storage.artifacts import get_artifact_storage

# Setup logging first
setup_logging()
logger = get_logger(__name__)


async def _conversation_cleanup_loop() -> None:
    """Background loop that deletes expired conversations and orphan attachments."""
    settings = get_settings()
    while True:
        try:
            await asyncio.sleep(settings.conversation_cleanup_interval)
            async with get_async_db_context() as db:
                deleted = await chat_crud.delete_expired_conversations(
                    db, settings.conversation_retention_days
                )
                orphans = await chat_crud.delete_all_orphan_attachments(db)
                await db.commit()
                if deleted or orphans:
                    logger.info(
                        "conversation_cleanup",
                        conversations_deleted=deleted,
                        orphan_attachments_deleted=orphans,
                    )
        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("conversation_cleanup_error")


async def _seed_redis_from_db() -> None:
    """Load current token counts from DB into Redis on startup."""
    if not redis_is_available():
        return
    try:
        from backend.app.db.models import Quota
        from sqlalchemy import select

        async with get_async_db_context() as db:
            result = await db.execute(select(Quota.user_id, Quota.tokens_used))
            rows = result.all()
            seeded = 0
            for user_id, tokens_used in rows:
                await redis_set_tokens(user_id, tokens_used or 0)
                seeded += 1
            logger.info("redis_seeded_from_db", users=seeded)
    except Exception:
        logger.exception("redis_seed_failed")


async def _redis_sync_loop() -> None:
    """Background loop: flush Redis token counters to DB every 60s for durability."""
    while True:
        try:
            await asyncio.sleep(60)
            if not redis_is_available():
                continue
            token_map = await get_all_token_keys()
            if not token_map:
                continue
            from backend.app.db.models import Quota
            from sqlalchemy import update

            async with get_async_db_context() as db:
                for user_id, tokens in token_map.items():
                    await db.execute(
                        update(Quota)
                        .where(Quota.user_id == user_id)
                        .values(tokens_used=tokens)
                    )
                await db.commit()
            logger.debug("redis_sync_to_db", users=len(token_map))
        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("redis_sync_error")


_cleanup_task: Optional[asyncio.Task] = None
_redis_sync_task: Optional[asyncio.Task] = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan manager."""
    global _cleanup_task, _redis_sync_task
    logger.info("Starting MindRouter2...")

    # Initialize components
    await init_registry()
    await init_scheduler()

    # Initialize storage
    storage = get_artifact_storage()
    await storage.initialize()

    # Initialize Redis and seed counters from DB
    await init_redis()
    await _seed_redis_from_db()

    # Initialize timezone cache from DB
    from backend.app.db.session import AsyncSessionLocal
    from backend.app.dashboard.routes import _init_tz_cache
    async with AsyncSessionLocal() as db:
        await _init_tz_cache(db)

    # Start background loops
    _cleanup_task = asyncio.create_task(_conversation_cleanup_loop())
    if redis_is_available():
        _redis_sync_task = asyncio.create_task(_redis_sync_loop())

    logger.info("MindRouter2 started successfully")

    yield

    # Shutdown
    logger.info("Shutting down MindRouter2...")
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
    await shutdown_scheduler()
    await shutdown_registry()
    logger.info("MindRouter2 shutdown complete")


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
    app.include_router(blog_router)

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
