############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# mcp_server.py: Server-side SSE MCP server for web search
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Server-side SSE MCP server for MindRouter web search.

Exposes the ``web_search`` tool over SSE transport so MCP-compatible
agents can connect directly to the server without local dependencies.

Mount path: ``/mcp/search`` (configured in main.py)
  - GET  /mcp/search/sse          — SSE stream (requires API key)
  - POST /mcp/search/messages/    — client messages (session-bound)

Client configuration::

    {
      "mcpServers": {
        "mindrouter": {
          "type": "sse",
          "url": "https://mindrouter.uidaho.edu/mcp/search/sse",
          "headers": {
            "Authorization": "Bearer mr2_your_key_here"
          }
        }
      }
    }
"""

import contextvars
import time
from typing import Optional

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route

from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport

from backend.app.db.session import get_async_db_context
from backend.app.logging_config import get_logger

logger = get_logger(__name__)

_auth_info: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar(
    "mcp_auth", default=None
)

mcp = FastMCP(
    "mindrouter",
    instructions="MindRouter web search. Use the web_search tool to find current information on the web.",
)

# Endpoint is relative — connect_sse prepends scope["root_path"] automatically
sse_transport = SseServerTransport("/messages/")


@mcp.tool()
async def web_search(query: str, max_results: Optional[int] = 5) -> str:
    """Search the web using MindRouter's search API.

    Returns titles, URLs, and snippets for each result. Use this when
    you need current information, documentation, or facts that may not
    be in your training data.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return (1-50, default 5).
    """
    auth = _auth_info.get()
    if not auth:
        return "Error: Not authenticated. Connect with an API key."

    from backend.app.db import crud
    from backend.app.db.models import Modality, User
    from backend.app.services.search.registry import PROVIDERS, get_search_config
    from sqlalchemy import select
    from sqlalchemy.orm import joinedload

    async with get_async_db_context() as db:
        result = await db.execute(
            select(User)
            .where(User.id == auth["user_id"])
            .options(joinedload(User.group))
        )
        user = result.scalar_one_or_none()
        if not user:
            return "Error: User not found."

        config = await get_search_config(db)
        if not config.get("search.enabled", True):
            return "Error: Web search is not enabled on this server."

        await crud.reset_quota_if_needed(db, user.id)
        quota = await crud.get_user_quota(db, user.id)
        group_budget = user.group.token_budget if user.group else 0
        if quota and group_budget > 0 and quota.tokens_used >= group_budget:
            return "Error: Token quota exceeded."

        provider_key = config.get("search.provider", "brave")
        provider = PROVIDERS.get(provider_key)
        if not provider:
            return f"Error: Search provider '{provider_key}' is not available."

        count = min(max_results or 5, 50)
        t0 = time.monotonic()
        try:
            results = await provider.search(query, max_results=count, config=config)
        except Exception:
            logger.exception("mcp_search_error", provider=provider_key)
            return "Error: Search provider returned an error."
        latency_ms = int((time.monotonic() - t0) * 1000)

        token_cost = int(config.get("search.quota_tokens_per_request", 50))
        if token_cost > 0:
            req_record = await crud.create_request(
                db,
                user_id=user.id,
                api_key_id=auth["api_key_id"],
                model="web_search",
                endpoint="/mcp/search",
                modality=Modality.CHAT,
            )
            await crud.update_request_completed(
                db, req_record.id, prompt_tokens=token_cost, completion_tokens=0
            )
            await crud.update_quota_usage(db, user.id, token_cost)
            await db.commit()
            await crud.incr_quota_redis(user.id, token_cost)

        logger.info(
            "mcp_search",
            user_id=user.id,
            provider=provider_key,
            query_len=len(query),
            results=len(results),
            latency_ms=latency_ms,
        )

        if not results:
            return f"No results found for: {query}"

        lines = [f"Search results for: {query}\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r.title}")
            lines.append(f"   {r.url}")
            lines.append(f"   {r.snippet}")
            if r.published:
                lines.append(f"   Published: {r.published}")
            lines.append("")
        return "\n".join(lines)


def _extract_api_key(request: Request) -> Optional[str]:
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:]
    return request.headers.get("x-api-key")


async def _handle_sse(request: Request) -> Response:
    """SSE connection endpoint with MindRouter API key authentication."""
    from backend.app.security.api_keys import verify_api_key

    api_key_str = _extract_api_key(request)
    if not api_key_str:
        return JSONResponse(
            {
                "error": (
                    "Missing API key. Provide via "
                    "'Authorization: Bearer <key>' or 'X-API-Key: <key>' header."
                )
            },
            status_code=401,
        )

    async with get_async_db_context() as db:
        api_key = await verify_api_key(db, api_key_str)
        if not api_key:
            return JSONResponse({"error": "Invalid API key"}, status_code=401)
        user = api_key.user
        if not user or not user.is_active:
            return JSONResponse(
                {"error": "User account is inactive"}, status_code=401
            )
        _auth_info.set({"user_id": user.id, "api_key_id": api_key.id})
        logger.info("mcp_sse_connect", user_id=user.id)

    async with sse_transport.connect_sse(
        request.scope, request.receive, request._send
    ) as streams:
        await mcp._mcp_server.run(
            streams[0],
            streams[1],
            mcp._mcp_server.create_initialization_options(),
        )
    return Response()


mcp_app = Starlette(
    routes=[
        Route("/sse", endpoint=_handle_sse, methods=["GET"]),
        Mount("/messages", app=sse_transport.handle_post_message),
    ],
)
