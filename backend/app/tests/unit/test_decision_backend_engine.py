############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_decision_backend_engine.py: Unit tests for the
# 'decision' backend engine (migration 087): the adapter that
# health-checks a System One decision server (Clef behind
# clef_service, Laya) so it is a monitored fleet member.
#
# Covers: /health semantics (ok / loading / no status field /
# non-200 / unreachable / timeout), zero models discovered (the
# invariant that keeps a decision server out of chat routing
# and the catalog), telemetry that never raises, the registry
# choosing this adapter, the migration, and the admin form.
#
############################################################

"""Unit tests for the decision-server backend engine."""

import importlib.util
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

_APP = Path(__file__).resolve().parents[2]


def _adapter(url="https://aspen4.example.edu:8001/"):
    # Imported here, not at module level: other test files swap telemetry modules in sys.modules.
    from backend.app.core.telemetry.adapters.decision import DecisionAdapter
    return DecisionAdapter(url)


def _client(*, status_code=200, body=None, exc=None, not_json=False):
    client = AsyncMock()
    client.is_closed = False
    if exc is not None:
        client.get = AsyncMock(side_effect=exc)
        return client
    resp = MagicMock()
    resp.status_code = status_code
    resp.json = MagicMock(side_effect=ValueError("not json")) if not_json else MagicMock(return_value=body)
    client.get = AsyncMock(return_value=resp)
    return client


class TestHealth:
    async def test_ok_is_healthy_and_probes_health_without_a_key(self):
        a = _adapter()
        a._client = _client(body={"status": "ok"})
        health = await a.health_check()
        assert health.is_healthy is True and health.status_code == 200
        a._client.get.assert_awaited_once_with("/health")
        assert a.base_url == "https://aspen4.example.edu:8001"          # trailing slash stripped

    async def test_loading_is_not_healthy(self):
        a = _adapter()
        a._client = _client(body={"status": "loading"})
        health = await a.health_check()
        assert health.is_healthy is False and "loading" in health.error_message

    @pytest.mark.parametrize("word", ["loading", "Loading", "starting", "initializing", "warming", "error", "unhealthy"])
    async def test_every_not_ready_word_is_unhealthy(self, word):
        a = _adapter()
        a._client = _client(body={"status": word})
        assert (await a.health_check()).is_healthy is False

    @pytest.mark.parametrize("kwargs", [{"body": {}}, {"body": {"model": "x"}}, {"body": ["x"]}, {"not_json": True},
                                        {"body": {"status": "healthy"}}, {"body": {"status": "OK"}},
                                        {"body": {"status": "ready"}}, {"body": {"status": 1}}])
    async def test_a_200_that_does_not_say_not_ready_is_healthy(self, kwargs):
        a = _adapter()
        a._client = _client(**kwargs)
        assert (await a.health_check()).is_healthy is True

    @pytest.mark.parametrize("status_code", [401, 500, 502, 503])
    async def test_non_200_is_unhealthy(self, status_code):
        a = _adapter()
        a._client = _client(status_code=status_code, body={"status": "ok"})
        health = await a.health_check()
        assert health.is_healthy is False and health.error_message == f"HTTP {status_code}"

    async def test_unreachable_and_timeout_are_unhealthy_not_exceptions(self):
        a = _adapter()
        a._client = _client(exc=httpx.ConnectError("refused"))
        assert (await a.health_check()).is_healthy is False
        a._client = _client(exc=httpx.ReadTimeout("slow"))
        health = await a.health_check()
        assert health.is_healthy is False and health.error_message == "Connection timeout"


class TestTls:
    @pytest.mark.parametrize("setting", [True, False])
    async def test_verification_follows_the_internal_tls_setting(self, setting, monkeypatch):
        # The decisions API dials the server with this setting; the health check must agree,
        # or registering a server with an internal certificate would mark a working server down.
        import backend.app.core.telemetry.adapters.decision as mod
        import backend.app.settings as settings_mod

        seen = {}
        monkeypatch.setattr(settings_mod, "get_settings", lambda: MagicMock(internal_tls_verify=setting))
        monkeypatch.setattr(mod.httpx, "AsyncClient", lambda **kw: seen.update(kw) or MagicMock(is_closed=False))
        await _adapter()._get_client()
        assert seen["verify"] is setting and seen["base_url"] == "https://aspen4.example.edu:8001"


class TestNoModels:
    @pytest.mark.parametrize("kwargs", [{"body": {"status": "ok"}}, {"body": {"status": "loading"}},
                                        {"status_code": 503, "body": {}}, {"exc": httpx.ConnectError("x")}])
    async def test_discovers_zero_models_whatever_the_server_says(self, kwargs):
        a = _adapter()
        a._client = _client(**kwargs)
        caps = await a.discover_capabilities()
        assert caps.models == [] and caps.loaded_models == []

    async def test_capabilities_carry_the_health(self):
        a = _adapter()
        a._client = _client(body={"status": "ok"})
        assert (await a.discover_capabilities()).is_healthy is True
        a._client = _client(body={"status": "loading"})
        caps = await a.discover_capabilities()
        assert caps.is_healthy is False and "loading" in caps.error_message


class TestTelemetry:
    async def test_snapshot_reports_liveness_and_never_raises(self):
        a = _adapter()
        a._client = _client(body={"status": "ok"})
        snap = await a.get_telemetry(42)
        assert snap.backend_id == 42 and snap.is_healthy is True
        a._client = _client(exc=RuntimeError("anything"))
        assert (await a.get_telemetry(42)).is_healthy is False


class TestWiring:
    def test_registry_uses_this_adapter_for_the_decision_engine(self):
        from backend.app.core.telemetry.adapters.decision import DecisionAdapter
        from backend.app.core.telemetry.registry import BackendRegistry
        from backend.app.db.models import BackendEngine

        reg = MagicMock()
        reg._settings.backend_health_timeout = 5
        backend = MagicMock(url="https://h:8001", engine=BackendEngine.DECISION)
        assert isinstance(BackendRegistry._create_adapter(reg, backend), DecisionAdapter)
        backend.engine = BackendEngine.VLLM
        assert not isinstance(BackendRegistry._create_adapter(reg, backend), DecisionAdapter)

    def test_decision_servers_are_model_less_like_dlp(self):
        from backend.app.db.models import MODELLESS_ENGINES, BackendEngine

        assert BackendEngine.DECISION.value == "decision"
        assert set(MODELLESS_ENGINES) == {BackendEngine.DLP, BackendEngine.DECISION, BackendEngine.MATTING}

    def test_the_orm_enum_and_the_migration_agree(self):
        from backend.app.db.models import BackendEngine

        path = _APP / "db" / "migrations" / "versions" / "20261004_000000_087_add_decision_backend_engine.py"
        spec = importlib.util.spec_from_file_location("migration_087", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module.revision == "087" and module.down_revision == "086"
        assert callable(module.upgrade) and callable(module.downgrade)
        assert "'decision'" in module.NEW_ENGINE and "'decision'" not in module.OLD_ENGINE
        # Same values, same order: a mismatch would make the ORM write a value the column rejects.
        # Later migrations (089, matting) only append, so 087's list is the ORM enum's beginning.
        orm = ",".join(f"'{e.value}'" for e in BackendEngine)
        assert orm == module.NEW_ENGINE or orm.startswith(module.NEW_ENGINE + ",")

    def test_admin_can_register_and_edit_one(self):
        html = (_APP / "dashboard" / "templates" / "admin" / "backends.html").read_text()
        assert html.count('<option value="decision"') == 2
