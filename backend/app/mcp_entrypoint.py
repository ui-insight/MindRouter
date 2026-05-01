############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# mcp_entrypoint.py: Standalone entrypoint for the MCP server
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Standalone entrypoint for the MCP SSE server.

Runs the MCP search server as a dedicated single-worker process,
avoiding the session-affinity problem with SseServerTransport's
in-memory session store when the main app runs multiple workers.

Usage:
    uvicorn backend.app.mcp_entrypoint:app --host 0.0.0.0 --port 8001 --workers 1
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from backend.app.logging_config import get_logger, setup_logging
from backend.app.core.redis_client import close_redis, init_redis

setup_logging()
logger = get_logger(__name__)


async def _healthz(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "service": "mcp"})


@asynccontextmanager
async def _lifespan(app: Starlette) -> AsyncGenerator:
    logger.info("Starting MCP server...")
    await init_redis()
    logger.info("MCP server started")
    yield
    await close_redis()
    logger.info("MCP server stopped")


from backend.app.api.mcp_server import mcp_app  # noqa: E402

app = Starlette(
    routes=[
        Route("/healthz", _healthz),
        Mount("/mcp", app=mcp_app),
    ],
    lifespan=_lifespan,
)
