"""A local fault is not the backend's fault.

On 2026-09-18 campus DNS failed for ~11 minutes. Every backend health check
raised "[Errno -3] Temporary failure in name resolution"; `_check_backend_health`
charged each failure as a strike, all 59 backends hit
`backend_unhealthy_threshold` (in ~30s, because live-request failures arm
adaptive fast-poll at 10s), and `get_healthy_backends()` returned nothing — so
every request got `model_unavailable` while all 14 nodes were healthy and
serving the whole time.

These tests pin the rule that a failure THIS host caused (DNS, local network)
must not be charged against a remote backend, and — just as important — that
anything unrecognised still IS charged, because misclassifying a genuinely sick
backend as a local fault would keep routing traffic to it.

registry.py imports the db package chain at module top, so per the project's
import-chain rules imports happen inside module-scoped fixtures that skip
cleanly when those deps are unavailable.
"""

import socket

import pytest


@pytest.fixture(scope="module")
def tmodels():
    try:
        from backend.app.core.telemetry import models
    except Exception as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"telemetry models unavailable: {exc}")
    return models


@pytest.fixture(scope="module")
def reg_mod():
    try:
        from backend.app.core.telemetry import registry
    except Exception as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"registry module unavailable: {exc}")
    return registry


@pytest.fixture(scope="module")
def sidecar_mod():
    try:
        from backend.app.core.telemetry.adapters import sidecar_client
    except Exception as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"sidecar client unavailable: {exc}")
    return sidecar_client


# --------------------------------------------------------------------------
# Test doubles
# --------------------------------------------------------------------------

class _SettingsShim:
    """Real settings with a few fields overridden.

    Proxies everything it does not override, so the code under test still sees
    genuine values for the dozen other settings it reads.
    """

    def __init__(self, real, **overrides):
        self._real = real
        self._overrides = overrides

    def __getattr__(self, name):
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._real, name)


class _StubAdapter:
    def __init__(self, health):
        self._health = health

    async def health_check(self):
        return self._health


class _DbRecorder:
    """Stands in for get_async_db_context, recording whether it was entered."""

    def __init__(self):
        self.entered = 0

    def __call__(self):
        return self

    async def __aenter__(self):
        self.entered += 1
        return self

    async def __aexit__(self, *exc):
        return False

    async def flush(self):
        return None


class _CrudStub:
    def __init__(self):
        self.node_status_calls = []

    async def get_backend_by_id(self, db, backend_id):
        return None

    async def update_backend_status(self, **kw):
        return None

    async def update_node_status(self, db, node_id, status):
        self.node_status_calls.append((node_id, status))


def _registry_with(reg_mod, monkeypatch, adapters=None, **setting_overrides):
    r = reg_mod.BackendRegistry()
    r._settings = _SettingsShim(r._settings, **setting_overrides)
    if adapters:
        r._adapters.update(adapters)
    db = _DbRecorder()
    crud = _CrudStub()
    monkeypatch.setattr(reg_mod, "get_async_db_context", db)
    monkeypatch.setattr(reg_mod, "crud", crud)
    return r, db, crud


# --------------------------------------------------------------------------
# classify_transport_error
# --------------------------------------------------------------------------

def test_classify_gaierror_is_dns(tmodels):
    """The canonical case: resolution failed, attached as the cause."""
    exc = ConnectionError("connect failed")
    exc.__cause__ = socket.gaierror(-3, "Temporary failure in name resolution")
    assert tmodels.classify_transport_error(exc) == tmodels.ERROR_KIND_DNS


def test_classify_flattened_dns_message_is_dns(tmodels):
    """httpx sometimes presents the resolver failure with no gaierror attached."""

    class ConnectError(Exception):
        pass

    exc = ConnectError("[Errno -3] Temporary failure in name resolution")
    assert tmodels.classify_transport_error(exc) == tmodels.ERROR_KIND_DNS


@pytest.mark.parametrize(
    "name", ["TimeoutException", "ConnectTimeout", "ReadTimeout", "PoolTimeout"]
)
def test_classify_timeout_types(tmodels, name):
    exc = type(name, (Exception,), {})("slow")
    assert tmodels.classify_transport_error(exc) == tmodels.ERROR_KIND_TIMEOUT


def test_classify_connect_error_without_dns_text_is_connect(tmodels):
    """Connection refused is NOT a local fault — the host resolved fine."""

    class ConnectError(Exception):
        pass

    exc = ConnectError("[Errno 111] Connection refused")
    assert tmodels.classify_transport_error(exc) == tmodels.ERROR_KIND_CONNECT
    assert tmodels.ERROR_KIND_CONNECT not in tmodels.LOCAL_FAULT_KINDS


def test_classify_unknown_returns_none(tmodels):
    """Unrecognised failures must return None so callers blame the backend.

    Failing the other way would keep routing traffic to a sick backend.
    """
    assert tmodels.classify_transport_error(ValueError("something else")) is None


# --------------------------------------------------------------------------
# _check_backend_health
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_local_fault_does_not_charge_a_strike(reg_mod, tmodels, monkeypatch):
    """A DNS-class failure must not touch the backend's status at all."""
    health = tmodels.BackendHealth(
        is_healthy=False,
        error_message="[Errno -3] Temporary failure in name resolution",
        error_kind=tmodels.ERROR_KIND_DNS,
    )
    r, db, _ = _registry_with(
        reg_mod, monkeypatch, adapters={7: _StubAdapter(health)}
    )

    await r._check_backend_health(7)

    assert db.entered == 0, "a local fault must not open a status-writing session"
    assert 7 in r._sweep_local_faults


@pytest.mark.asyncio
async def test_unclassified_failure_still_charges_a_strike(reg_mod, tmodels, monkeypatch):
    """An HTTP 500 or unknown error is the backend's problem, as before."""
    health = tmodels.BackendHealth(
        is_healthy=False, status_code=500, error_message="HTTP 500"
    )
    r, db, _ = _registry_with(
        reg_mod, monkeypatch, adapters={8: _StubAdapter(health)}
    )

    await r._check_backend_health(8)

    assert db.entered == 1, "a backend-side failure must still be recorded"
    assert 8 not in r._sweep_local_faults


@pytest.mark.asyncio
async def test_setting_restores_old_behaviour(reg_mod, tmodels, monkeypatch):
    """backend_local_fault_trips_health=True brings back the pre-fix path."""
    health = tmodels.BackendHealth(
        is_healthy=False, error_kind=tmodels.ERROR_KIND_DNS, error_message="dns"
    )
    r, db, _ = _registry_with(
        reg_mod,
        monkeypatch,
        adapters={9: _StubAdapter(health)},
        backend_local_fault_trips_health=True,
    )

    await r._check_backend_health(9)

    assert db.entered == 1


@pytest.mark.asyncio
async def test_recovery_clears_the_local_fault_marker(reg_mod, tmodels, monkeypatch):
    """Once a check succeeds the backend leaves the local-fault set."""
    r, _, _ = _registry_with(reg_mod, monkeypatch)
    r._sweep_local_faults.add(11)
    r._adapters[11] = _StubAdapter(tmodels.BackendHealth(is_healthy=True))

    await r._check_backend_health(11)

    assert 11 not in r._sweep_local_faults


@pytest.mark.asyncio
async def test_fleet_local_fault_logged_once(reg_mod, tmodels, monkeypatch):
    """Most of the fleet failing locally is ONE event, not N.

    During the outage the logs reported 59 backend failures and 14 node
    failures, which pointed the investigation at the cluster rather than at the
    gateway's own resolver.
    """
    health = tmodels.BackendHealth(
        is_healthy=False, error_kind=tmodels.ERROR_KIND_DNS, error_message="dns"
    )
    adapters = {i: _StubAdapter(health) for i in range(1, 5)}
    r, _, _ = _registry_with(reg_mod, monkeypatch, adapters=adapters)

    events = []

    class _Log:
        def error(self, event, **kw):
            events.append((event, kw))

        def __getattr__(self, _n):
            return lambda *a, **k: None

    monkeypatch.setattr(reg_mod, "logger", _Log())

    await r._poll_all_backends()

    fleet = [e for e in events if e[0] == "fleet_local_fault"]
    assert len(fleet) == 1, "expected exactly one fleet-level event"
    assert fleet[0][1]["affected"] == 4


# --------------------------------------------------------------------------
# node hysteresis
# --------------------------------------------------------------------------

class _FailingSidecar:
    def __init__(self, exc=None, value=None):
        self._exc = exc
        self._value = value

    async def get_gpu_info(self):
        if self._exc:
            raise self._exc
        return self._value


@pytest.mark.asyncio
async def test_node_offline_only_after_threshold(reg_mod, monkeypatch):
    """One failed poll used to flip a node OFFLINE; now it takes three."""
    r, _, crud = _registry_with(reg_mod, monkeypatch, node_unhealthy_threshold=3)
    r._sidecar_clients[4] = _FailingSidecar(exc=OSError("boom"))

    await r._collect_node_telemetry(4)
    assert crud.node_status_calls == [], "first failure must not mark it offline"

    await r._collect_node_telemetry(4)
    assert crud.node_status_calls == [], "second failure must not either"

    await r._collect_node_telemetry(4)
    assert len(crud.node_status_calls) == 1
    assert crud.node_status_calls[0][0] == 4


@pytest.mark.asyncio
async def test_successful_poll_clears_the_streak(reg_mod, monkeypatch):
    """Two failures then a success must not leave the node one strike away."""
    r, _, crud = _registry_with(reg_mod, monkeypatch, node_unhealthy_threshold=3)
    r._sidecar_clients[5] = _FailingSidecar(exc=OSError("boom"))
    await r._collect_node_telemetry(5)
    await r._collect_node_telemetry(5)
    assert r._node_failures.get(5) == 2

    class _Data:
        gpus = []
        gpu_count = 0

    r._sidecar_clients[5] = _FailingSidecar(value=_Data())
    await r._collect_node_telemetry(5)

    assert 5 not in r._node_failures
    assert crud.node_status_calls == []


@pytest.mark.asyncio
async def test_no_response_counts_toward_threshold(reg_mod, monkeypatch):
    """A None response is a failed poll too, and uses the same counter."""
    r, _, crud = _registry_with(reg_mod, monkeypatch, node_unhealthy_threshold=2)
    r._sidecar_clients[6] = _FailingSidecar(value=None)

    await r._collect_node_telemetry(6)
    assert crud.node_status_calls == []
    await r._collect_node_telemetry(6)
    assert len(crud.node_status_calls) == 1


# --------------------------------------------------------------------------
# sidecar retry
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sidecar_retries_transport_failure(sidecar_mod, monkeypatch):
    """A single dropped packet should not count as a failed poll."""
    calls = {"n": 0}

    class _Client:
        async def get(self, _path):
            calls["n"] += 1
            raise OSError("transient")

    c = sidecar_mod.SidecarClient("https://node.example:8007")
    monkeypatch.setattr(c, "_get_client", lambda: _async(_Client()))
    monkeypatch.setattr(sidecar_mod.asyncio, "sleep", _noop_sleep)

    assert await c.get_gpu_info() is None
    assert calls["n"] == 2, "expected one retry"


@pytest.mark.asyncio
async def test_sidecar_does_not_retry_non_200(sidecar_mod, monkeypatch):
    """The sidecar answered — asking again just loads a struggling agent."""
    calls = {"n": 0}

    class _Resp:
        status_code = 503
        text = "busy"

    class _Client:
        async def get(self, _path):
            calls["n"] += 1
            return _Resp()

    c = sidecar_mod.SidecarClient("https://node.example:8007")
    monkeypatch.setattr(c, "_get_client", lambda: _async(_Client()))

    assert await c.get_gpu_info() is None
    assert calls["n"] == 1, "a non-200 must not be retried"


async def _noop_sleep(_seconds):
    return None


def _async(value):
    async def _coro():
        return value

    return _coro()
