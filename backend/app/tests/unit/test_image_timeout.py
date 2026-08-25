"""Image-generation timeout semantics in InferenceService._proxy_with_retry.

Diffusion workers (serve_klein.py) keep the GPU busy for the whole job even
after the gateway hangs up, so a per-attempt timeout on an image must NOT be
retried on another backend (that would burn a second and third GPU on the
same oversized job) and must NOT count against the backend's health. Images
also get their own attempt budget (``backend_image_request_timeout``) instead
of the chat ``backend_request_timeout_per_attempt``.

Loads inference.py via importlib to bypass the db package chain (see the
"Import Chain Gotcha" in MEMORY.md / test_hotpath_trims.py).
"""

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import HTTPException

# Reuse the exact module-loading recipe from test_hotpath_trims so both tests
# see the same InferenceService object.
from backend.app.tests.unit import test_hotpath_trims as hp

inf = hp.inf


def _make_service(image_timeout=7, chat_timeout=5, attempts=3):
    svc = inf.InferenceService.__new__(inf.InferenceService)
    svc.db = MagicMock()
    svc._settings = SimpleNamespace(
        backend_retry_max_attempts=attempts,
        backend_request_timeout_per_attempt=chat_timeout,
        backend_image_request_timeout=image_timeout,
        thinking_off_by_default=False,
    )
    svc._scheduler = AsyncMock()
    svc._registry = AsyncMock()
    svc._latency_tracker = AsyncMock()
    svc._http_client = None
    svc.cap_max_tokens = AsyncMock()
    backends = [SimpleNamespace(engine=hp._VLLM_SENTINEL, url=f"http://b{i}", id=i) for i in (1, 2, 3)]
    calls = {"n": 0}

    async def _route(job, user, modality, exclude_backend_ids=None, max_wait=None):
        excl = exclude_backend_ids or set()
        for b in backends:
            if b.id not in excl:
                calls["n"] += 1
                return b, []
        raise HTTPException(status_code=503, detail="none")

    svc._route_request = _route
    return svc, backends, calls


def _job():
    return SimpleNamespace(model="black-forest-labs/FLUX.2-klein-9B", request_id="img-1")


class TestImageTimeout:
    @pytest.mark.asyncio
    async def test_image_timeout_is_not_retried_and_returns_504(self):
        svc, backends, routed = _make_service()
        svc._proxy_image_request = AsyncMock(side_effect=asyncio.TimeoutError())
        with pytest.raises(HTTPException) as exc:
            await svc._proxy_with_retry(object(), _job(), MagicMock(), proxy_fn="_proxy_image_request")
        assert exc.value.status_code == 504
        assert "Image generation exceeded 7s" in exc.value.detail
        assert svc._proxy_image_request.await_count == 1        # no second attempt
        assert routed["n"] == 1                                  # no second backend
        svc._scheduler.on_job_failed.assert_awaited_once()      # slot released
        svc._registry.report_live_failure.assert_not_awaited()  # backend not blamed

    @pytest.mark.asyncio
    async def test_image_attempt_uses_image_budget(self, monkeypatch):
        svc, _, _ = _make_service(image_timeout=123, chat_timeout=5)
        seen = {}

        async def fake_wait_for(coro, timeout):
            seen["timeout"] = timeout
            return await coro

        monkeypatch.setattr(inf.asyncio, "wait_for", fake_wait_for)
        svc._proxy_image_request = AsyncMock(return_value={"data": []})
        resp, backend = await svc._proxy_with_retry(object(), _job(), MagicMock(), proxy_fn="_proxy_image_request")
        assert resp == {"data": []}
        assert seen["timeout"] == 123.0

    @pytest.mark.asyncio
    async def test_chat_timeout_still_retries_on_other_backends(self):
        svc, backends, routed = _make_service(chat_timeout=5)
        svc._proxy_chat_request = AsyncMock(side_effect=asyncio.TimeoutError())
        request = hp._make_request(max_tokens=10)
        with pytest.raises(HTTPException) as exc:
            await svc._proxy_with_retry(request, SimpleNamespace(model="test-model", request_id="r"), MagicMock())
        assert exc.value.status_code == 502
        assert svc._proxy_chat_request.await_count == 3
        assert routed["n"] == 3
        assert svc._registry.report_live_failure.await_count == 3
        # The empty str() of a timeout no longer produces "Last error: "
        assert exc.value.detail.endswith("Last error: TimeoutError")

    @pytest.mark.asyncio
    async def test_chat_attempt_uses_chat_budget(self, monkeypatch):
        svc, _, _ = _make_service(image_timeout=123, chat_timeout=5)
        seen = {}

        async def fake_wait_for(coro, timeout):
            seen["timeout"] = timeout
            return await coro

        monkeypatch.setattr(inf.asyncio, "wait_for", fake_wait_for)
        svc._proxy_chat_request = AsyncMock(return_value={"ok": True})
        await svc._proxy_with_retry(hp._make_request(max_tokens=10), SimpleNamespace(model="test-model", request_id="r"), MagicMock())
        assert seen["timeout"] == 5.0
