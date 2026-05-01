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

"""Reverse proxy that forwards /mcp/* requests to the standalone MCP service.

The MCP SSE server runs as a separate single-worker process (port 8001)
to avoid session-affinity issues with multi-worker uvicorn. This proxy
preserves the public URL at /mcp/sse while routing to that service.
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
        return StreamingResponse(
            upstream.aiter_bytes(),
            status_code=upstream.status_code,
            headers=dict(upstream.headers),
            background=upstream.aclose,
        )

    body = await request.body()
    resp = await client.request(
        method=request.method,
        url=target_url,
        headers=headers,
        content=body,
    )
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=dict(resp.headers),
    )


mcp_proxy_app = Starlette(
    routes=[
        Route("/{path:path}", _proxy, methods=["GET", "POST"]),
    ],
)
