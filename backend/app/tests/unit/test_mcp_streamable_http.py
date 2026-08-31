############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_mcp_streamable_http.py: MCP Streamable HTTP transport
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Tests for the MCP Streamable HTTP transport (POST /mcp).

Covers the three things that can silently break it: the canonical path must
not redirect, every request must be authenticated (not just the first), and
the auth contextvar must reach the tool without leaking between concurrent
requests.
"""

import contextlib
from typing import Optional

import anyio
import httpx
import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette

from backend.app.api import mcp_server

PROTOCOL_VERSION = "2025-06-18"
HEADERS = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
    "MCP-Protocol-Version": PROTOCOL_VERSION,
}


def _build_app(server=None, json_response: bool = True) -> Starlette:
    """Mirror the production wiring with a fresh session manager.

    A manager's run() may only be called once per instance, so each test gets
    its own rather than sharing the module-level singleton.
    """
    manager = StreamableHTTPSessionManager(
        app=server if server is not None else mcp_server.mcp._mcp_server,
        event_store=None,
        json_response=json_response,
        stateless=True,
    )
    endpoint = mcp_server.StreamableHTTPEndpoint(manager)

    @contextlib.asynccontextmanager
    async def lifespan(app):
        async with manager.run():
            yield

    # Build the route through the production factory (with the fresh endpoint
    # swapped in) so that a regression there — e.g. switching to a Mount, which
    # 307-redirects POST /mcp — fails these tests rather than passing silently.
    import unittest.mock

    with unittest.mock.patch.object(mcp_server, "streamable_endpoint", endpoint):
        route = mcp_server.streamable_http_route("/mcp")

    return Starlette(routes=[route], lifespan=lifespan)


@contextlib.asynccontextmanager
async def _client(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        async with app.router.lifespan_context(app):
            yield client


def _auth_headers(key: str = "mr2_valid") -> dict:
    return {**HEADERS, "Authorization": f"Bearer {key}"}


@pytest.fixture
def allow_all(monkeypatch):
    """Accept any key as user 1 without touching the database."""

    async def _resolve(api_key_str: Optional[str]):
        if not api_key_str:
            return None, mcp_server._MISSING_KEY_ERROR
        return {"user_id": 1, "api_key_id": 7}, None

    monkeypatch.setattr(mcp_server, "_resolve_auth", _resolve)


# --------------------------------------------------------------------------
# Canonical path
# --------------------------------------------------------------------------


async def test_post_mcp_does_not_redirect(allow_all):
    """POST /mcp must answer directly.

    A Mount would 307 to /mcp/, and clients that drop the body on a POST
    redirect fail to connect.
    """
    app = _build_app()
    async with _client(app) as client:
        resp = await client.post(
            "/mcp",
            headers=_auth_headers(),
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        )
    assert resp.status_code == 200, resp.text
    assert resp.history == []


def test_production_route_is_exact_path_with_spec_methods():
    route = mcp_server.streamable_http_route("/mcp")
    assert route.path == "/mcp"
    # DELETE is session termination in the spec; a 405 confuses clients.
    assert {"POST", "GET", "DELETE"}.issubset(route.methods)


def test_session_manager_is_stateless():
    """Stateless is what removes session affinity and keeps auth per-request."""
    assert mcp_server.streamable_session_manager.stateless is True


# --------------------------------------------------------------------------
# Round trip
# --------------------------------------------------------------------------


async def test_initialize_then_tools_list_round_trip(allow_all):
    app = _build_app()
    async with _client(app) as client:
        init = await client.post(
            "/mcp",
            headers=_auth_headers(),
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1"},
                },
            },
        )
        assert init.status_code == 200, init.text

        listed = await client.post(
            "/mcp",
            headers=_auth_headers(),
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        )
    assert listed.status_code == 200, listed.text
    names = [t["name"] for t in listed.json()["result"]["tools"]]
    assert "web_search" in names


async def test_sse_framing_when_json_response_disabled(allow_all):
    """The spec default returns the response as an SSE stream in the POST."""
    app = _build_app(json_response=False)
    async with _client(app) as client:
        resp = await client.post(
            "/mcp",
            headers=_auth_headers(),
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    assert "web_search" in resp.text


# --------------------------------------------------------------------------
# Authentication — on every request, not just the first
# --------------------------------------------------------------------------


async def test_missing_api_key_is_rejected():
    app = _build_app()
    async with _client(app) as client:
        resp = await client.post(
            "/mcp",
            headers=HEADERS,
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        )
    assert resp.status_code == 401
    assert "Missing API key" in resp.json()["error"]


@pytest.mark.parametrize(
    "verify_result,rejection,expected",
    [
        (None, None, "Invalid API key"),
        ("key", "Key has been revoked", "Key has been revoked"),
        ("key", "Key has expired", "Key has expired"),
    ],
)
async def test_bad_keys_are_rejected(monkeypatch, verify_result, rejection, expected):
    """Revoked/expired keys go through the same gate as the REST API."""
    from backend.app.security import api_keys

    class _Key:
        def __init__(self):
            self.id = 7
            self.user = type("U", (), {"id": 1})()

    async def _verify(db, key_str):
        return _Key() if verify_result else None

    @contextlib.asynccontextmanager
    async def _fake_db():
        yield None

    monkeypatch.setattr(api_keys, "verify_api_key", _verify)
    monkeypatch.setattr(api_keys, "api_key_rejection_reason", lambda k: rejection)
    monkeypatch.setattr(mcp_server, "get_async_db_context", _fake_db)

    app = _build_app()
    async with _client(app) as client:
        resp = await client.post(
            "/mcp",
            headers=_auth_headers("mr2_bad"),
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        )
    assert resp.status_code == 401
    assert expected in resp.json()["error"]


async def test_every_request_is_authenticated(monkeypatch):
    """Auth must not be connect-time-only the way the legacy SSE path is."""
    calls = []

    async def _resolve(api_key_str: Optional[str]):
        calls.append(api_key_str)
        return {"user_id": 1, "api_key_id": 7}, None

    monkeypatch.setattr(mcp_server, "_resolve_auth", _resolve)

    app = _build_app()
    async with _client(app) as client:
        for _ in range(3):
            await client.post(
                "/mcp",
                headers=_auth_headers(),
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            )
    assert len(calls) == 3


# --------------------------------------------------------------------------
# Auth context reaches the tool, and does not leak between requests
# --------------------------------------------------------------------------


def _probe_server():
    """A FastMCP whose tool reports the auth contextvar it observed."""
    probe = FastMCP("probe")

    @probe.tool()
    async def whoami(tag: str) -> str:
        await anyio.sleep(0.05)  # force concurrent requests to interleave
        auth = mcp_server._auth_info.get()
        return f"{tag}:{(auth or {}).get('api_key_id')}"

    return probe._mcp_server


async def _call_whoami(client, tag: str, results: dict):
    resp = await client.post(
        "/mcp",
        headers=_auth_headers(f"mr2_{tag}"),
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "whoami", "arguments": {"tag": tag}},
        },
    )
    results[tag] = resp.json()["result"]["content"][0]["text"]


async def test_auth_context_reaches_the_tool(monkeypatch):
    """stateless=True spawns the server task from the request's context."""

    async def _resolve(api_key_str: Optional[str]):
        return {"user_id": 1, "api_key_id": 99}, None

    monkeypatch.setattr(mcp_server, "_resolve_auth", _resolve)

    app = _build_app(server=_probe_server())
    results: dict = {}
    async with _client(app) as client:
        await _call_whoami(client, "solo", results)
    assert results["solo"] == "solo:99"


async def test_concurrent_requests_do_not_leak_auth(monkeypatch):
    """The regression that would bill or authorize one user as another."""

    async def _resolve(api_key_str: Optional[str]):
        tag = (api_key_str or "").replace("mr2_", "")
        # api_key_id is derived from the caller's own key
        return {"user_id": 1, "api_key_id": tag}, None

    monkeypatch.setattr(mcp_server, "_resolve_auth", _resolve)

    app = _build_app(server=_probe_server())
    results: dict = {}
    tags = ["alice", "bob", "carol", "dave"]
    async with _client(app) as client:
        async with anyio.create_task_group() as tg:
            for tag in tags:
                tg.start_soon(_call_whoami, client, tag, results)

    for tag in tags:
        assert results[tag] == f"{tag}:{tag}", results


# --------------------------------------------------------------------------
# The legacy transport stays mounted for the deprecation window
# --------------------------------------------------------------------------


def test_legacy_sse_routes_still_exist():
    paths = {getattr(r, "path", None) for r in mcp_server.mcp_app.routes}
    assert "/sse" in paths
    assert any(
        getattr(r, "path", "").startswith("/messages") for r in mcp_server.mcp_app.routes
    )


def test_streamable_route_is_registered_before_the_legacy_proxy_mount():
    """Order decides which transport answers POST /mcp.

    Starlette matches routes in registration order, and the legacy proxy is a
    Mount on "/mcp" that would swallow the path and forward it to :8001 (where
    only the SSE transport lives). Asserted on the source because building the
    whole app here would pull in the database and logging stack.
    """
    import pathlib

    source = (
        pathlib.Path(__file__).resolve().parents[2] / "main.py"
    ).read_text()

    route_at = source.find("streamable_http_route")
    mount_at = source.find('app.mount("/mcp"')
    assert route_at != -1, "streamable route registration missing from main.py"
    assert mount_at != -1, "legacy proxy mount missing from main.py"
    assert route_at < mount_at, (
        "the /mcp Route must be registered before the legacy proxy Mount, "
        "or the Mount swallows POST /mcp"
    )
