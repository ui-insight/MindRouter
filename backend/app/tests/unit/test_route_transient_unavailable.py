############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_route_transient_unavailable.py: routing when a configured
#     model has no healthy backend for a moment
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Routing's answer when no HEALTHY backend serves the requested model.

It used to be an immediate 404 "Model 'x' not found", for a model the fleet
has as much as for a typo.  The API surfaces check availability first and
answer 503 for a configured-but-down model (api/model_availability.py), but
internal callers (OCR chunks, dashboard chat) never pass that check, and a
backend can drop between the check and routing.  OCR does not retry a 4xx,
so one such 404 failed a whole document.

Now: an unknown name is still a 404 at once; a configured model is looked up
again a few times (``model_lookup_retry_*``), and only then answered with the
same 503 + Retry-After the API surfaces give.  Failover attempts
(``max_wait=0``) do not wait.

inference.py is spec-loaded with its heavy imports stubbed, as in
test_streaming_session_isolation.py; nothing is written to sys.modules
outside the load.
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

_APP_DIR = Path(__file__).resolve().parents[2]
_MISSING = object()


def _load(name, path, stubs=None, seeded=None):
    stubs, seeded = stubs or {}, seeded or {}
    touched = set(stubs) | set(seeded)
    saved = {k: sys.modules.get(k, _MISSING) for k in touched}
    try:
        sys.modules.update(stubs)
        for key, seed_path in seeded.items():
            spec = importlib.util.spec_from_file_location(key, seed_path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[key] = mod
            spec.loader.exec_module(mod)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for key, original in saved.items():
            if original is _MISSING:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = original


@pytest.fixture(scope="module")
def inf():
    stubbed = [
        "backend.app.db", "backend.app.db.session", "backend.app.db.crud",
        "backend.app.db.models", "backend.app.settings", "backend.app.logging_config",
        "backend.app.core.redis_client", "backend.app.core.quota_budget",
        "backend.app.core.telemetry", "backend.app.core.telemetry.registry",
        "backend.app.core.scheduler.policy", "backend.app.core.scheduler.queue",
        "backend.app.core.translators", "backend.app.core.translators.vllm_out",
    ]
    return _load(
        "mr2_inference_route_transient", _APP_DIR / "services" / "inference.py",
        stubs={k: MagicMock() for k in stubbed},
        seeded={
            "backend.app.core.canonical_schemas": _APP_DIR / "core" / "canonical_schemas.py",
            "backend.app.core.stream_coalesce": _APP_DIR / "core" / "stream_coalesce.py",
        },
    )


@pytest.fixture(scope="module")
def availability():
    return _load("mr2_model_availability", _APP_DIR / "api" / "model_availability.py")


class _Proceeded(Exception):
    """Routing got past the lookup with backends in hand."""


class _Registry:
    """Healthy backends per lookup, in order (the last one repeats), and the
    statuses of every backend carrying the model (empty: unknown model)."""

    def __init__(self, lookups, statuses=("UNKNOWN",), forced_offline=False):
        self.is_force_offline = forced_offline
        self.lookups = list(lookups)
        self.statuses = statuses
        self.lookup_calls = []
        self.status_calls = []

    async def get_backends_with_model(self, model, modality=None):
        self.lookup_calls.append((model, modality))
        return self.lookups.pop(0) if len(self.lookups) > 1 else self.lookups[0]

    async def get_model_backend_statuses(self, model, modality=None):
        self.status_calls.append((model, modality))
        # Names -> the module's own BackendStatus members (set by `make`).
        return {getattr(self.backend_status, s) for s in self.statuses}

    async def is_backend_available(self, backend_id):
        raise _Proceeded(backend_id)


@pytest.fixture
def make(inf, monkeypatch):
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    # Replace the module's asyncio reference, not asyncio.sleep itself (which
    # pytest-asyncio uses too).
    monkeypatch.setattr(inf, "asyncio", SimpleNamespace(sleep=fake_sleep))
    inf.logger.reset_mock()

    def _make(registry, attempts=2, pause=2.0):
        registry.backend_status = inf.BackendStatus
        service = object.__new__(inf.InferenceService)
        service._registry = registry
        service._settings = SimpleNamespace(
            model_lookup_retry_attempts=attempts,
            model_lookup_retry_pause_seconds=pause,
        )
        return service, sleeps

    return _make


BACKEND = SimpleNamespace(id=7)
CHAT = "chat-modality"


async def test_an_unknown_model_is_a_404_at_once(make):
    registry = _Registry([[]], statuses=())
    service, sleeps = make(registry)
    with pytest.raises(HTTPException) as info:
        await service._await_routable_backends("qwen-typo", CHAT, wait=True)
    assert info.value.status_code == 404
    assert info.value.detail == "Model 'qwen-typo' not found"
    assert sleeps == [] and registry.lookup_calls == []


async def test_a_short_gap_is_ridden_out(make, inf):
    registry = _Registry([[], [BACKEND]])
    service, sleeps = make(registry)
    backends = await service._await_routable_backends("dots.MOCR", CHAT, wait=True)
    assert backends == [BACKEND]
    assert sleeps == [2.0, 2.0]
    # The modality is kept on every look: a chat request to an
    # embedding-only model must not wait and then be called "unavailable".
    assert registry.status_calls == [("dots.MOCR", CHAT)]
    assert registry.lookup_calls == [("dots.MOCR", CHAT)] * 2
    inf.logger.info.assert_any_call(
        "model_lookup_transient_recovered", model="dots.MOCR", attempt=2
    )


async def test_a_configured_model_still_down_is_a_503_like_the_api_surfaces(make, availability):
    registry = _Registry([[]])
    service, sleeps = make(registry, attempts=3, pause=1.5)
    with pytest.raises(HTTPException) as info:
        await service._await_routable_backends("dots.MOCR", CHAT, wait=True)
    assert info.value.status_code == 503
    # Word for word what /v1/chat/completions etc. answer for the same state.
    assert info.value.detail == availability.unavailable_message(
        "dots.MOCR", availability.UNAVAILABLE
    )
    assert info.value.headers == {"Retry-After": str(availability.RETRY_AFTER_SECONDS)}
    # The marker the Responses API labels `model_unavailable` by.
    assert info.value.model_unavailable is True
    assert sleeps == [1.5] * 3 and len(registry.lookup_calls) == 3


@pytest.mark.parametrize("statuses, waits", [
    (("UNHEALTHY",), True),
    (("UNKNOWN",), True),          # re-enabled, waiting for its first check
    (("DISABLED", "UNHEALTHY"), True),
    (("DISABLED",), False),        # an admin took it out: no point waiting
    (("DRAINING",), False),
    (("DISABLED", "DRAINING"), False),
])
async def test_waits_only_while_a_backend_could_come_back(make, statuses, waits):
    registry = _Registry([[]], statuses=statuses)
    service, sleeps = make(registry)
    with pytest.raises(HTTPException) as info:
        await service._await_routable_backends("glm-5.3", CHAT, wait=True)
    assert info.value.status_code == 503
    assert sleeps == ([2.0, 2.0] if waits else [])


async def test_a_backend_that_just_turned_healthy_is_used_without_sleeping(make):
    # Both queries share their filters, so HEALTHY among the statuses means
    # one came back between them.
    registry = _Registry([[BACKEND]], statuses=("HEALTHY", "UNHEALTHY"))
    service, sleeps = make(registry)
    assert await service._await_routable_backends("dots.MOCR", CHAT, wait=True) == [BACKEND]
    assert sleeps == []


async def test_no_wait_while_the_fleet_is_forced_offline(make):
    # force_offline stops the pollers and marks every backend UNHEALTHY:
    # nothing will come back by itself.
    registry = _Registry([[]], statuses=("UNHEALTHY",), forced_offline=True)
    service, sleeps = make(registry)
    with pytest.raises(HTTPException) as info:
        await service._await_routable_backends("dots.MOCR", CHAT, wait=True)
    assert info.value.status_code == 503 and sleeps == []


def test_failover_attempts_route_with_max_wait_zero():
    # Only `max_wait=0` skips the lookup wait; pin that every retry loop
    # passes it on attempts after the first (both calls in each loop, the
    # cleared-exclusions retry included).
    import ast

    tree = ast.parse((_APP_DIR / "services" / "inference.py").read_text())
    loops = {
        n.name: n for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef)
        and n.name in ("_proxy_with_retry", "_proxy_stream_with_retry")
    }
    assert set(loops) == {"_proxy_with_retry", "_proxy_stream_with_retry"}
    for name, fn in loops.items():
        src = ast.unparse(fn)
        assert "retry_wait = None if attempt == 0 else 0" in src, name
        calls = [
            c for c in ast.walk(fn)
            if isinstance(c, ast.Call) and ast.unparse(c.func) == "self._route_request"
        ]
        assert len(calls) == 2, name
        for call in calls:
            kw = {k.arg: ast.unparse(k.value) for k in call.keywords}
            assert kw.get("max_wait") == "retry_wait", name


@pytest.mark.parametrize("wait, attempts", [(False, 2), (True, 0)])
async def test_no_wait_on_failover_or_when_disabled(make, wait, attempts):
    registry = _Registry([[]])
    service, sleeps = make(registry, attempts=attempts)
    with pytest.raises(HTTPException) as info:
        await service._await_routable_backends("dots.MOCR", CHAT, wait=wait)
    assert info.value.status_code == 503
    assert sleeps == [] and registry.lookup_calls == []


@pytest.mark.parametrize("max_wait, waits", [(None, True), (30.0, True), (0, False)])
async def test_route_request_inner_waits_except_on_failover(make, max_wait, waits):
    # A failover attempt routes with max_wait=0 ("retries fail fast"): it
    # must not sit out the lookup retries either.
    registry = _Registry([[], [BACKEND]])
    service, sleeps = make(registry)
    job = SimpleNamespace(model="dots.MOCR")
    if waits:
        with pytest.raises(_Proceeded):
            await service._route_request_inner(job, user=None, modality=CHAT, max_wait=max_wait)
        assert sleeps == [2.0]
    else:
        with pytest.raises(HTTPException) as info:
            await service._route_request_inner(job, user=None, modality=CHAT, max_wait=max_wait)
        assert info.value.status_code == 503 and sleeps == []


async def test_a_healthy_model_never_checks_the_catalog(make):
    registry = _Registry([[BACKEND]])
    service, sleeps = make(registry)
    with pytest.raises(_Proceeded):
        await service._route_request_inner(
            SimpleNamespace(model="dots.MOCR"), user=None, modality=CHAT
        )
    assert registry.status_calls == [] and sleeps == []


async def test_backend_statuses_query_mirrors_routing_without_the_health_filter():
    from sqlalchemy.dialects import mysql

    from backend.app.db import crud
    from backend.app.db.models import Modality

    statements = []

    class _Db:
        async def execute(self, stmt):
            statements.append(str(stmt.compile(dialect=mysql.dialect(),
                                               compile_kwargs={"literal_binds": True})))
            return SimpleNamespace(all=lambda: [("unknown",), ("disabled",)])

    assert await crud.get_model_backend_statuses(_Db(), "bge-m3", Modality.CHAT) == {
        "unknown", "disabled",
    }
    await crud.get_model_backend_statuses(_Db(), "bge-m3")
    with_modality, without = statements
    assert "models.modality = 'chat'" in with_modality
    assert "modality" not in without
    for sql in statements:
        assert "SELECT DISTINCT backends.status" in sql
        assert "backends.engine NOT IN" in sql
        # Every status, not just HEALTHY: that is the point.
        assert "backends.status =" not in sql
