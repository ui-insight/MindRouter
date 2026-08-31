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

"""MCP server for MindRouter, exposing tools such as ``web_search``.

Two transports are served in parallel during the deprecation window:

**Streamable HTTP (2025-03-26, current)** — a single endpoint, ``POST /mcp``.
The JSON-RPC response comes back inline in that same POST response, either as
one ``application/json`` body or as an SSE stream. This is what modern MCP
clients speak. It runs ``stateless=True``, so there is no server-side session
affinity and it is served directly from the main multi-worker app.

**HTTP+SSE (2024-11-05, legacy, deprecated)** — ``GET /mcp/sse`` opens a
long-lived stream and ``POST /mcp/messages/`` posts into it. Session state
lives in memory, so this transport requires the dedicated single-worker
process on :8001 (see mcp_entrypoint.py) fronted by mcp_proxy.py.

Client configuration (preferred)::

    {
      "mcpServers": {
        "mindrouter": {
          "type": "http",
          "url": "https://mindrouter.uidaho.edu/mcp",
          "headers": {
            "Authorization": "Bearer mr2_your_key_here"
          }
        }
      }
    }

Legacy clients that cannot speak Streamable HTTP may still use
``"type": "sse"`` against ``https://mindrouter.uidaho.edu/mcp/sse``.
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
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

from backend.app.db.session import get_async_db_context
from backend.app.logging_config import get_logger
from backend.app.settings import get_settings

logger = get_logger(__name__)

_auth_info: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar(
    "mcp_auth", default=None
)

mcp = FastMCP(
    "mindrouter",
    instructions="MindRouter tools. Use web_search to find current information on the web.",
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
    from backend.app.db.models import Modality, User, WebSearchSource
    from backend.app.services.search.registry import PROVIDERS, get_search_config
    from backend.app.services.search.audit import run_logged_search
    from backend.app.services.search.dlp_gate import WebSearchBlockedError
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
            # Audited drop-in for provider.search() — same result, same errors.
            results = await run_logged_search(
                db,
                query,
                source=WebSearchSource.MCP.value,
                max_results=count,
                config=config,
                provider=provider,
                user_id=user.id,
                api_key_id=auth.get("api_key_id"),
            )
        except WebSearchBlockedError as e:
            logger.warning("mcp_search_blocked_by_dlp", user_id=user.id, reason=str(e))
            return f"Search blocked by data-loss prevention: {e}"
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
                endpoint="/mcp",
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


_MISSING_KEY_ERROR = (
    "Missing API key. Provide via "
    "'Authorization: Bearer <key>' or 'X-API-Key: <key>' header."
)


def _extract_api_key(request: Request) -> Optional[str]:
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:]
    return request.headers.get("x-api-key")


async def _resolve_auth(api_key_str: Optional[str]) -> tuple[Optional[dict], Optional[str]]:
    """Verify a MindRouter API key.

    Returns ``(auth, None)`` on success or ``(None, error_message)`` on
    rejection. Shared by both transports so they cannot drift apart.
    """
    from backend.app.security.api_keys import api_key_rejection_reason, verify_api_key

    if not api_key_str:
        return None, _MISSING_KEY_ERROR

    async with get_async_db_context() as db:
        api_key = await verify_api_key(db, api_key_str)
        if not api_key:
            return None, "Invalid API key"
        # Shared post-verify gate: rejects revoked, expired, and
        # inactive/deleted-user keys — same checks as authenticate_request
        reason = api_key_rejection_reason(api_key)
        if reason:
            return None, reason
        # Read ids inside the session — the instance is detached on exit.
        return {"user_id": api_key.user.id, "api_key_id": api_key.id}, None


class StreamableHTTPEndpoint:
    """ASGI endpoint for the Streamable HTTP transport at ``POST /mcp``.

    ``StreamableHTTPSessionManager`` performs no authentication of its own, so
    every request is authenticated here before the scope is handed to the SDK
    — unlike the legacy SSE path, which can only authenticate at connect time.

    This is an ASGI app rather than a Starlette endpoint function on purpose:
    Starlette wraps plain functions in ``request_response()``, which would
    consume the body and defeat the streaming response. Registering it as
    ``Route("/mcp", endpoint=<instance>)`` also serves the canonical path with
    no trailing slash — a ``Mount`` would 307-redirect ``POST /mcp`` to
    ``/mcp/``, which not every client follows on a POST.
    """

    def __init__(self, session_manager: StreamableHTTPSessionManager) -> None:
        self._session_manager = session_manager

    async def __call__(self, scope, receive, send) -> None:
        # Header-only access; the body is left for the session manager.
        request = Request(scope, receive)
        auth, error = await _resolve_auth(_extract_api_key(request))
        if error:
            await JSONResponse({"error": error}, status_code=401)(scope, receive, send)
            return

        # Set per-request so the tool can read it. Verified to propagate into
        # the tool under stateless=True (the SDK spawns the server task from
        # this request's context) with no bleed between concurrent requests.
        _auth_info.set(auth)
        logger.info(
            "mcp_streamable_request",
            user_id=auth["user_id"],
            method=scope.get("method"),
        )
        await self._session_manager.handle_request(scope, receive, send)


async def _handle_sse(request: Request) -> Response:
    """Legacy SSE connection endpoint with MindRouter API key authentication."""
    auth, error = await _resolve_auth(_extract_api_key(request))
    if error:
        return JSONResponse({"error": error}, status_code=401)

    _auth_info.set(auth)
    logger.info("mcp_sse_connect", user_id=auth["user_id"])

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


# --- Streamable HTTP (current transport) ------------------------------------
#
# stateless=True gives every request its own transport, so there is no
# server-side session to pin a client to. That is what lets this transport be
# served from the main multi-worker app instead of the dedicated :8001 process
# the legacy SSE transport needs. Trading it for stateful sessions would buy
# resumable streams (with an event_store) at the cost of bringing session
# affinity back — and would also break per-request auth, because the server
# task is then spawned once per session and the auth contextvar would freeze
# at whoever opened it.
streamable_session_manager = StreamableHTTPSessionManager(
    app=mcp._mcp_server,
    event_store=None,  # no resumability in stateless mode
    json_response=get_settings().mcp_streamable_json_response,
    stateless=True,
)

streamable_endpoint = StreamableHTTPEndpoint(streamable_session_manager)


def streamable_http_route(path: str = "/mcp") -> Route:
    """Build the route to register on the main app.

    DELETE is part of the spec (session termination); it is a no-op under
    stateless=True but is accepted so clients do not see a 405.
    """
    return Route(
        path,
        endpoint=streamable_endpoint,
        methods=["POST", "GET", "DELETE"],
    )
