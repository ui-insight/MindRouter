############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# mcp_proxy.py: Reverse proxy for the standalone MCP server
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Reverse proxy that forwards legacy /mcp/* requests to the standalone MCP service.

The deprecated HTTP+SSE transport keeps session state in memory, so it runs as
a separate single-worker process (port 8001) to avoid session-affinity issues
with multi-worker uvicorn. This proxy preserves the public URL at /mcp/sse
while routing to that service.

The current Streamable HTTP transport does NOT come through here: it is
stateless and is served directly by the main app at POST /mcp (registered
ahead of this mount in main.py). Anything that reaches this proxy is legacy
traffic.
"""

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route

from backend.app.logging_config import get_logger
from backend.app.settings import get_settings

logger = get_logger(__name__)

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=None, write=30.0, pool=30.0),
            follow_redirects=False,
        )
    return _client


async def _proxy(request: Request) -> Response:
    settings = get_settings()
    target_url = f"{settings.mcp_server_url}{request.url.path}"
    if request.url.query:
        target_url += f"?{request.url.query}"

    headers = dict(request.headers)
    headers.pop("host", None)

    client = _get_client()

    if request.method == "GET" and "/sse" in request.url.path:
        req = client.build_request("GET", target_url, headers=headers)
        upstream = await client.send(req, stream=True)
        resp_headers = dict(upstream.headers)
        resp_headers["Cache-Control"] = "no-cache, no-store"
        resp_headers["X-Accel-Buffering"] = "no"
        resp_headers.pop("content-length", None)
        resp_headers.pop("Content-Length", None)
        return StreamingResponse(
            upstream.aiter_raw(),
            status_code=upstream.status_code,
            headers=resp_headers,
            media_type="text/event-stream",
            background=upstream.aclose,
        )

    body = await request.body()
    req = client.build_request(
        request.method, target_url, headers=headers, content=body
    )
    upstream = await client.send(req, stream=True)

    # A POST whose response is an SSE stream must not be buffered: reading
    # .content here would hold the whole stream until the upstream closed it.
    # The legacy transport does not answer POSTs this way today, but streaming
    # the response costs nothing and removes the trap if it ever does.
    if "text/event-stream" in upstream.headers.get("content-type", ""):
        resp_headers = dict(upstream.headers)
        resp_headers["Cache-Control"] = "no-cache, no-store"
        resp_headers["X-Accel-Buffering"] = "no"
        resp_headers.pop("content-length", None)
        resp_headers.pop("Content-Length", None)
        return StreamingResponse(
            upstream.aiter_raw(),
            status_code=upstream.status_code,
            headers=resp_headers,
            media_type="text/event-stream",
            background=upstream.aclose,
        )

    try:
        content = await upstream.aread()
    finally:
        await upstream.aclose()
    return Response(
        content=content,
        status_code=upstream.status_code,
        headers=dict(upstream.headers),
    )


mcp_proxy_app = Starlette(
    routes=[
        Route("/{path:path}", _proxy, methods=["GET", "POST"]),
    ],
)
