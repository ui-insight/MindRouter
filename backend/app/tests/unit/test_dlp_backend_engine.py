############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_dlp_backend_engine.py: Unit tests for the DLP backend
#     engine type — the DlpAdapter (health / discovery / telemetry)
#     and migration 077.
#
############################################################

"""Unit tests for the DLP (GLiNER) backend engine.

The DLP backend is a fleet member for status + (via the per-node sidecar)
GPU/power telemetry, but must NEVER be routable or catalog-visible. The
mechanism is DlpAdapter.discover_capabilities() returning models == [] in
ALL cases, which these tests pin down.

Import style mirrors test_sidecar_client.py: heavy package __init__ chains are
pre-mocked in sys.modules and the telemetry models + adapter are loaded from
their .py files directly, bypassing the db/settings import chain.
"""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

# ---- isolation: bypass heavy package __init__ imports ----
for mod_name in [
    "backend.app.db",
    "backend.app.db.session",
    "backend.app.db.models",
    "backend.app.db.crud",
    "backend.app.settings",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

# Load telemetry models directly to get real dataclasses.
_models_path = Path(__file__).resolve().parents[2] / "core" / "telemetry" / "models.py"
_spec = importlib.util.spec_from_file_location("telemetry_models", _models_path)
telemetry_models = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(telemetry_models)

BackendCapabilities = telemetry_models.BackendCapabilities
BackendHealth = telemetry_models.BackendHealth
TelemetrySnapshot = telemetry_models.TelemetrySnapshot

# Make the real telemetry models importable by the adapter module.
sys.modules["backend.app.core.telemetry.models"] = telemetry_models

# Mock logging (structlog-style get_logger).
mock_logging = MagicMock()
mock_logging.get_logger = MagicMock(return_value=MagicMock())
sys.modules["backend.app.logging_config"] = mock_logging

# Now load the DLP adapter module directly.
_dlp_path = (
    Path(__file__).resolve().parents[2]
    / "core"
    / "telemetry"
    / "adapters"
    / "dlp.py"
)
_dlp_spec = importlib.util.spec_from_file_location("dlp_adapter", _dlp_path)
dlp_adapter_mod = importlib.util.module_from_spec(_dlp_spec)
_dlp_spec.loader.exec_module(dlp_adapter_mod)

DlpAdapter = dlp_adapter_mod.DlpAdapter


# ---- Sample /healthz bodies (see dlp_service/server.py) ----
WARM_BODY = {
    "status": "ok",
    "model": "urchade/gliner_multi_pii-v1",
    "device": "cuda:0",
    "replicas": 2,
    "queue_depth": 0,
    "max_queue": 512,
    "warm": True,
}
COLD_BODY = {
    "status": "ok",
    "model": "urchade/gliner_multi_pii-v1",
    "device": "cuda:0",
    "replicas": 2,
    "queue_depth": 0,
    "max_queue": 512,
    "warm": False,
}


def _mock_client(*, status_code=200, json_body=None, exc=None):
    """Build an AsyncMock httpx client whose .get returns/raises as configured."""
    client = AsyncMock()
    client.is_closed = False
    if exc is not None:
        client.get = AsyncMock(side_effect=exc)
    else:
        resp = MagicMock()
        resp.status_code = status_code
        resp.json = MagicMock(return_value=json_body if json_body is not None else {})
        client.get = AsyncMock(return_value=resp)
    return client


def _attach(adapter, client):
    """Inject a pre-built mock client so _get_client returns it unchanged."""
    adapter._client = client


# ---- Init ----
class TestDlpAdapterInit:
    def test_strips_trailing_slash(self):
        a = DlpAdapter("https://dlp-node:8001/")
        assert a.base_url == "https://dlp-node:8001"

    def test_default_timeout(self):
        a = DlpAdapter("https://dlp-node:8001")
        assert a.timeout == 10.0

    def test_custom_timeout(self):
        a = DlpAdapter("https://dlp-node:8001", timeout=3.0)
        assert a.timeout == 3.0


# ---- health_check ----
class TestDlpHealthCheck:
    @pytest.mark.asyncio
    async def test_200_warm_is_healthy(self):
        a = DlpAdapter("https://dlp-node:8001")
        _attach(a, _mock_client(status_code=200, json_body=WARM_BODY))

        health = await a.health_check()
        assert isinstance(health, BackendHealth)
        assert health.is_healthy is True
        assert health.status_code == 200
        # /healthz, not /health.
        a._client.get.assert_awaited_once_with("/healthz")

    @pytest.mark.asyncio
    async def test_200_not_warm_is_unhealthy(self):
        a = DlpAdapter("https://dlp-node:8001")
        _attach(a, _mock_client(status_code=200, json_body=COLD_BODY))

        health = await a.health_check()
        assert health.is_healthy is False
        assert health.status_code == 200

    @pytest.mark.asyncio
    async def test_200_missing_warm_field_is_unhealthy(self):
        a = DlpAdapter("https://dlp-node:8001")
        _attach(a, _mock_client(status_code=200, json_body={"status": "ok"}))

        health = await a.health_check()
        assert health.is_healthy is False

    @pytest.mark.asyncio
    async def test_503_is_unhealthy(self):
        a = DlpAdapter("https://dlp-node:8001")
        _attach(a, _mock_client(status_code=503, json_body={}))

        health = await a.health_check()
        assert health.is_healthy is False
        assert health.status_code == 503

    @pytest.mark.asyncio
    async def test_connect_error_is_unhealthy(self):
        a = DlpAdapter("https://dlp-node:8001")
        _attach(a, _mock_client(exc=httpx.ConnectError("refused")))

        health = await a.health_check()
        assert health.is_healthy is False

    @pytest.mark.asyncio
    async def test_timeout_is_unhealthy(self):
        a = DlpAdapter("https://dlp-node:8001")
        _attach(a, _mock_client(exc=httpx.TimeoutException("slow")))

        health = await a.health_check()
        assert health.is_healthy is False


# ---- discover_capabilities: models == [] in ALL cases ----
class TestDlpDiscoverCapabilities:
    @pytest.mark.asyncio
    async def test_warm_discovers_zero_models(self):
        a = DlpAdapter("https://dlp-node:8001")
        _attach(a, _mock_client(status_code=200, json_body=WARM_BODY))

        caps = await a.discover_capabilities()
        assert isinstance(caps, BackendCapabilities)
        assert caps.is_healthy is True
        assert caps.models == []
        assert caps.loaded_models == []
        # engine_version surfaces the loaded GLiNER model name.
        assert caps.engine_version == "urchade/gliner_multi_pii-v1"

    @pytest.mark.asyncio
    async def test_cold_discovers_zero_models(self):
        a = DlpAdapter("https://dlp-node:8001")
        _attach(a, _mock_client(status_code=200, json_body=COLD_BODY))

        caps = await a.discover_capabilities()
        assert caps.is_healthy is False
        assert caps.models == []
        assert caps.loaded_models == []

    @pytest.mark.asyncio
    async def test_503_discovers_zero_models(self):
        a = DlpAdapter("https://dlp-node:8001")
        _attach(a, _mock_client(status_code=503, json_body={}))

        caps = await a.discover_capabilities()
        assert caps.is_healthy is False
        assert caps.models == []
        assert caps.loaded_models == []

    @pytest.mark.asyncio
    async def test_connect_error_discovers_zero_models(self):
        a = DlpAdapter("https://dlp-node:8001")
        _attach(a, _mock_client(exc=httpx.ConnectError("refused")))

        caps = await a.discover_capabilities()
        assert caps.is_healthy is False
        assert caps.models == []
        assert caps.loaded_models == []


# ---- get_telemetry: never raises, no GPU fields, no /stats ----
class TestDlpGetTelemetry:
    @pytest.mark.asyncio
    async def test_warm_snapshot_healthy(self):
        a = DlpAdapter("https://dlp-node:8001")
        _attach(a, _mock_client(status_code=200, json_body=WARM_BODY))

        snap = await a.get_telemetry(backend_id=42)
        assert isinstance(snap, TelemetrySnapshot)
        assert snap.backend_id == 42
        assert snap.is_healthy is True
        # No GPU fields — the per-node sidecar supplies those.
        assert snap.gpu_utilization is None
        assert snap.gpu_memory_used_gb is None
        # Never probes the key-protected /stats endpoint.
        for call in a._client.get.await_args_list:
            assert call.args[0] != "/stats"

    @pytest.mark.asyncio
    async def test_cold_snapshot_unhealthy(self):
        a = DlpAdapter("https://dlp-node:8001")
        _attach(a, _mock_client(status_code=200, json_body=COLD_BODY))

        snap = await a.get_telemetry(backend_id=7)
        assert snap.is_healthy is False

    @pytest.mark.asyncio
    async def test_connect_error_never_raises(self):
        a = DlpAdapter("https://dlp-node:8001")
        _attach(a, _mock_client(exc=httpx.ConnectError("refused")))

        snap = await a.get_telemetry(backend_id=1)
        assert snap.is_healthy is False

    @pytest.mark.asyncio
    async def test_arbitrary_exception_never_raises(self):
        a = DlpAdapter("https://dlp-node:8001")
        _attach(a, _mock_client(exc=RuntimeError("boom")))

        snap = await a.get_telemetry(backend_id=1)
        assert snap.is_healthy is False


# ---- close ----
class TestDlpClose:
    @pytest.mark.asyncio
    async def test_close_open_client(self):
        a = DlpAdapter("https://dlp-node:8001")
        client = AsyncMock()
        client.is_closed = False
        a._client = client
        await a.close()
        client.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_no_client(self):
        a = DlpAdapter("https://dlp-node:8001")
        await a.close()  # must not raise


# ---- Migration 077 loads and is wired correctly ----
class TestMigration077:
    def test_migration_module_imports(self):
        mig_path = (
            Path(__file__).resolve().parents[2]
            / "db"
            / "migrations"
            / "versions"
            / "20260819_000001_077_add_dlp_backend_engine.py"
        )
        spec = importlib.util.spec_from_file_location("migration_077", mig_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        assert module.revision == "077"
        assert module.down_revision == "076"
        assert callable(module.upgrade)
        assert callable(module.downgrade)
        # The appended value must be present in the new enum spelling.
        assert "'dlp'" in module.NEW_ENGINE
        assert "'dlp'" not in module.OLD_ENGINE


class TestDlpRoutingExclusionStructural:
    """A backend converted to engine=dlp must be non-routable and catalog-
    invisible even if stale Model rows linger (defense-in-depth beyond the
    discovery-time prune). Read source by path — importing these modules pulls
    the db/telemetry chain, and the invariant is textual/structural anyway."""

    _ROOT = Path(__file__).resolve().parents[4]

    def _src(self, rel):
        return (self._ROOT / rel).read_text()

    def test_get_backends_with_model_excludes_dlp_structurally(self):
        crud = self._src("backend/app/db/crud.py")
        block = crud.split("async def get_backends_with_model", 1)[1].split("async def ", 1)[0]
        # Every model-less engine (DLP, decision servers) is excluded, by the shared tuple.
        assert "Backend.engine.notin_(MODELLESS_ENGINES)" in block
        models = self._src("backend/app/db/models.py")
        assert "MODELLESS_ENGINES = (BackendEngine.DLP, BackendEngine.DECISION)" in models

    def test_update_backend_prunes_models_on_engine_change(self):
        crud = self._src("backend/app/db/crud.py")
        block = crud.split("async def update_backend(", 1)[1].split("\nasync def ", 1)[0]
        assert "delete(Model)" in block and "Model.backend_id == backend_id" in block

    def test_catalog_loops_skip_dlp(self):
        models_api = self._src("backend/app/api/models_api.py")
        health = self._src("backend/app/api/health.py")
        assert "backend.engine in MODELLESS_ENGINES" in models_api
        assert "backend.engine in MODELLESS_ENGINES" in health
