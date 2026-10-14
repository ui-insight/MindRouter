############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_queue_monitor_decisions.py: Decision models on the
#     admin queue monitor. A decision server (e.g. Clef) has
#     no model rows, so its capacity is found through
#     decisions.upstreams, matched to its backend by URL.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Queue monitor capacity for model-less decision servers."""

import json

import pytest

pytest.importorskip("aiosqlite")

CLEF_URL = "https://aspen4.hpc.uidaho.edu:8001"


@pytest.fixture
async def db():
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from backend.app.db.models import Base

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    names = ("groups", "users", "backends", "models", "app_config", "requests")
    tables = [Base.metadata.tables[t] for t in names]
    async with engine.begin() as conn:
        await conn.run_sync(lambda s: Base.metadata.create_all(s, tables=tables))
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as session:
        yield session
    await engine.dispose()


async def _backend(db, bid, name, url, engine, status, max_concurrent=1):
    from backend.app.db.models import Backend

    db.add(Backend(id=bid, name=name, url=url, engine=engine, status=status, max_concurrent=max_concurrent))
    await db.flush()


async def _upstreams(db, value):
    from backend.app.db.models import AppConfig

    db.add(AppConfig(key="decisions.upstreams", value=json.dumps(value)))
    await db.flush()


async def _capacity(db):
    from backend.app.api.admin_api import _add_decision_capacity

    capacity = {}
    await _add_decision_capacity(db, capacity)
    return capacity


class TestDecisionCapacity:
    async def test_clef_counts_its_backend(self, db):
        from backend.app.db.models import BackendEngine, BackendStatus

        await _backend(db, 78, "aspen4-gpu1-clef", CLEF_URL, BackendEngine.DECISION, BackendStatus.HEALTHY)
        # The upstream spells the same server differently; URLs are compared normalized.
        await _upstreams(db, {"clef": {"url": "https://ASPEN4.hpc.uidaho.edu:8001/", "model": "clef"}})
        await db.commit()

        cap = await _capacity(db)
        assert cap["clef"]["healthy_backends"] == 1
        assert cap["clef"]["total_max_concurrent"] == 1
        assert cap["clef"]["backends"] == [
            {"name": "aspen4-gpu1-clef", "status": "healthy", "max_concurrent": 1, "current_concurrent": 0}
        ]

    async def test_an_unhealthy_server_is_listed_but_not_healthy(self, db):
        from backend.app.db.models import BackendEngine, BackendStatus

        await _backend(db, 78, "aspen4-gpu1-clef", CLEF_URL, BackendEngine.DECISION, BackendStatus.UNHEALTHY)
        await _upstreams(db, {"clef": {"url": CLEF_URL}})
        await db.commit()

        cap = await _capacity(db)
        assert cap["clef"]["healthy_backends"] == 0
        assert cap["clef"]["total_max_concurrent"] == 1
        assert cap["clef"]["backends"][0]["status"] == "unhealthy"

    async def test_a_disabled_server_is_not_capacity(self, db):
        # Same rule as model backends: disabled ones are left out.
        from backend.app.db.models import BackendEngine, BackendStatus

        await _backend(db, 78, "aspen4-gpu1-clef", CLEF_URL, BackendEngine.DECISION, BackendStatus.DISABLED)
        await _upstreams(db, {"clef": {"url": CLEF_URL}})
        await db.commit()
        assert await _capacity(db) == {}

    async def test_only_decision_backends_match(self, db):
        # A chat backend that once had the same URL (qwen3.5-35b on aspen4 gpu1) is not Clef.
        from backend.app.db.models import BackendEngine, BackendStatus

        await _backend(db, 20, "aspen4-gpu1-qwen3.5-35b", CLEF_URL, BackendEngine.VLLM, BackendStatus.HEALTHY, 96)
        await _upstreams(db, {"clef": {"url": CLEF_URL}})
        await db.commit()
        assert await _capacity(db) == {}

    async def test_unregistered_upstream_and_other_servers_are_left_alone(self, db):
        from backend.app.db.models import BackendEngine, BackendStatus

        await _backend(db, 78, "aspen4-gpu1-clef", CLEF_URL, BackendEngine.DECISION, BackendStatus.HEALTHY)
        await _upstreams(db, {"clef": {"url": CLEF_URL}, "laya": {"url": "https://wintermute:8010"}})
        await db.commit()

        cap = await _capacity(db)
        assert set(cap) == {"clef"}

    async def test_one_server_registered_twice_counts_once(self, db):
        # An upstream is one URL, so one machine. The router takes the first
        # match; capacity must not double.
        from backend.app.db.models import BackendEngine, BackendStatus

        await _backend(db, 78, "a-clef", CLEF_URL, BackendEngine.DECISION, BackendStatus.UNHEALTHY, 1)
        await _backend(db, 81, "b-clef", "https://aspen4.hpc.uidaho.edu:8001/", BackendEngine.DECISION,
                       BackendStatus.HEALTHY, 2)
        await _upstreams(db, {"clef": {"url": CLEF_URL}})
        await db.commit()

        cap = await _capacity(db)
        assert [b["name"] for b in cap["clef"]["backends"]] == ["b-clef"]     # the healthy one
        assert cap["clef"]["healthy_backends"] == 1
        assert cap["clef"]["total_max_concurrent"] == 2

    @pytest.mark.parametrize("status", ["DRAINING", "UNKNOWN"])
    async def test_draining_or_unknown_is_listed_but_not_healthy(self, db, status):
        # Same rule as model backends: only `healthy` counts as healthy.
        from backend.app.db.models import BackendEngine, BackendStatus

        await _backend(db, 78, "aspen4-gpu1-clef", CLEF_URL, BackendEngine.DECISION, BackendStatus[status])
        await _upstreams(db, {"clef": {"url": CLEF_URL}})
        await db.commit()

        cap = await _capacity(db)
        assert cap["clef"]["healthy_backends"] == 0
        assert cap["clef"]["backends"][0]["status"] == BackendStatus[status].value

    async def test_two_names_for_one_server_each_show_it(self, db):
        from backend.app.db.models import BackendEngine, BackendStatus

        await _backend(db, 78, "aspen4-gpu1-clef", CLEF_URL, BackendEngine.DECISION, BackendStatus.HEALTHY)
        await _upstreams(db, {"clef": {"url": CLEF_URL}, "clef-alt": {"url": CLEF_URL, "model": "clef"}})
        await db.commit()

        cap = await _capacity(db)
        assert cap["clef"]["healthy_backends"] == cap["clef-alt"]["healthy_backends"] == 1

    async def test_no_setting_or_a_broken_one_changes_nothing(self, db):
        from backend.app.db.models import AppConfig, BackendEngine, BackendStatus

        await _backend(db, 78, "aspen4-gpu1-clef", CLEF_URL, BackendEngine.DECISION, BackendStatus.HEALTHY)
        await db.commit()
        assert await _capacity(db) == {}

        db.add(AppConfig(key="decisions.upstreams", value="not json"))
        await db.commit()
        assert await _capacity(db) == {}


class TestQueueMonitorEndpoint:
    async def test_clef_row_has_its_backend(self, db):
        from backend.app.api.admin_api import get_queue_monitor
        from backend.app.db.models import BackendEngine, BackendStatus, Modality, Model

        await _backend(db, 78, "aspen4-gpu1-clef", CLEF_URL, BackendEngine.DECISION, BackendStatus.HEALTHY)
        await _backend(db, 5, "aspen2-gpu2-qwen", "https://aspen2:8002", BackendEngine.VLLM, BackendStatus.HEALTHY, 72)
        db.add(Model(id=1, backend_id=5, name="qwen/qwen3.8-27b", modality=Modality.CHAT))
        await _upstreams(db, {"clef": {"url": CLEF_URL, "model": "clef"}})
        await db.commit()

        out = await get_queue_monitor(window=5, admin=None, db=db)
        assert out["capacity"]["clef"]["healthy_backends"] == 1
        assert out["capacity"]["clef"]["total_max_concurrent"] == 1
        assert out["capacity"]["qwen/qwen3.8-27b"]["total_max_concurrent"] == 72

    async def test_the_capacity_key_is_the_name_requests_are_recorded_under(self, db):
        # /v1/systemone records an upstream request with model = the upstream's
        # name (the default model and jev aliases are resolved to it first), so
        # the request counts and the capacity land on the same row of the page.
        from backend.app.api.admin_api import get_queue_monitor
        from backend.app.db.models import BackendEngine, BackendStatus, Modality, Request, RequestStatus

        await _backend(db, 78, "aspen4-gpu1-clef", CLEF_URL, BackendEngine.DECISION, BackendStatus.HEALTHY)
        await _upstreams(db, {"clef": {"url": CLEF_URL, "model": "clef"}})
        db.add(Request(id=1, user_id=1, api_key_id=1, endpoint="/v1/systemone", model="clef",
                       modality=Modality.CHAT, status=RequestStatus.COMPLETED, total_time_ms=400))
        await db.commit()

        out = await get_queue_monitor(window=60 * 24 * 365 * 100, admin=None, db=db)
        assert [m["model"] for m in out["by_model"]] == ["clef"]
        assert set(out["capacity"]) == {"clef"}

    async def test_a_failed_decision_lookup_does_not_break_the_monitor(self, db):
        from unittest.mock import AsyncMock, patch

        from backend.app.api import admin_api

        with patch.object(admin_api, "_add_decision_capacity", AsyncMock(side_effect=RuntimeError("db"))):
            out = await admin_api.get_queue_monitor(window=5, admin=None, db=db)
        assert out["capacity"] == {}
