############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_telemetry_rollup_mariadb.py: the 5-minute telemetry
#     rollups against a real MariaDB (app + archive DBs).
#
############################################################

"""Real-MariaDB integration test for the 5-minute telemetry rollups.

Skipped unless ``MINDROUTER_TEST_MARIADB_URL`` names a server where the user
may create and drop databases, for example::

    MINDROUTER_TEST_MARIADB_URL=mysql+aiomysql://root:test@127.0.0.1:3399 \\
        pytest backend/app/tests/integration/test_telemetry_rollup_mariadb.py

Every test creates two scratch databases (app + archive) with random names:
the app side gets nodes, gpu_devices, backends and the three telemetry
tables from the real ORM models; the archive side gets every archive table
via ArchiveBase metadata, as app startup does.  The test drops both
databases when it ends.  It then runs the real retention code against them
with readings whose aggregates are known.

The modules are loaded from their files (models.py with base.py pinned under
its package name) so the db/settings package ``__init__`` chain never runs,
and retention's call-time import of the archive models is pinned to the same
loaded module for the duration of each test.
"""

import contextlib
import functools
import importlib.util
import os
import sys
import types
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import insert, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.sql.dml import Insert

_URL = os.environ.get("MINDROUTER_TEST_MARIADB_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _URL, reason="MINDROUTER_TEST_MARIADB_URL not set (needs a real MariaDB)"
    ),
]

_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_MISSING = object()

T0 = datetime(2026, 10, 1, 0, 0)
# horizon = floor(NOW - 10 min) = T0 + 50 min: buckets T0 .. T0+45 are
# complete, the bucket at T0+50 is not.
NOW = T0 + timedelta(minutes=63)


def m(minutes, seconds=0):
    return T0 + timedelta(minutes=minutes, seconds=seconds)


@contextlib.contextmanager
def _pinned_modules(mapping):
    saved = {key: sys.modules.get(key, _MISSING) for key in mapping}
    sys.modules.update(mapping)
    try:
        yield
    finally:
        for key, original in saved.items():
            if original is _MISSING:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = original


def _load(module_name, rel_path, stubs=None):
    with _pinned_modules(stubs or {}):
        spec = importlib.util.spec_from_file_location(
            module_name, os.path.join(_APP_DIR, rel_path)
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module


class _RecordingLogger:
    def __init__(self):
        self.records = []

    def __getattr__(self, level):
        def _log(event, *args, **kwargs):
            self.records.append((level, event, kwargs))

        return _log


@functools.lru_cache(maxsize=1)
def _modules():
    base = _load("rollup_it_db_base", "db/base.py")
    models = _load(
        "rollup_it_db_models", "db/models.py", {"backend.app.db.base": base}
    )
    archive_models = _load("rollup_it_archive_models", "db/archive_models.py")
    log = _RecordingLogger()
    logging_stub = types.ModuleType("backend.app.logging_config")
    logging_stub.get_logger = lambda *a, **k: log
    retention = _load(
        "rollup_it_retention", "services/retention.py",
        {"backend.app.logging_config": logging_stub},
    )
    return types.SimpleNamespace(
        models=models, archive=archive_models, retention=retention, log=log
    )


@contextlib.asynccontextmanager
async def _scratch(app_time_zone=None):
    """Two scratch databases with the tables the rollup reads and writes."""
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    mods = _modules()
    suffix = uuid.uuid4().hex[:12]
    app_name, archive_name = f"mr_rollup_app_{suffix}", f"mr_rollup_arch_{suffix}"
    base_url = make_url(_URL)
    admin = create_async_engine(base_url, poolclass=NullPool)
    async with admin.begin() as conn:
        await conn.execute(text(f"CREATE DATABASE `{app_name}`"))
        await conn.execute(text(f"CREATE DATABASE `{archive_name}`"))

    connect_args = {}
    if app_time_zone:
        # Per connection, so it holds for every pooled connection.
        connect_args["init_command"] = f"SET time_zone = '{app_time_zone}'"
    app_engine = create_async_engine(
        base_url.set(database=app_name), connect_args=connect_args
    )
    archive_engine = create_async_engine(base_url.set(database=archive_name))
    try:
        m_ = mods.models
        async with app_engine.begin() as conn:
            await conn.run_sync(
                m_.Base.metadata.create_all,
                tables=[
                    m_.Node.__table__, m_.GPUDevice.__table__, m_.Backend.__table__,
                    m_.NodeTelemetry.__table__, m_.GPUDeviceTelemetry.__table__,
                    m_.BackendTelemetry.__table__,
                ],
            )
        async with archive_engine.begin() as conn:
            await conn.run_sync(mods.archive.ArchiveBase.metadata.create_all)

        env = types.SimpleNamespace(
            mods=mods,
            app_engine=app_engine,
            archive_engine=archive_engine,
            app_session=async_sessionmaker(app_engine, expire_on_commit=False),
            archive_session=async_sessionmaker(archive_engine, expire_on_commit=False),
        )
        mods.log.records.clear()
        with _pinned_modules({"backend.app.db.archive_models": mods.archive}):
            yield env
    finally:
        await app_engine.dispose()
        await archive_engine.dispose()
        async with admin.begin() as conn:
            await conn.execute(text(f"DROP DATABASE IF EXISTS `{app_name}`"))
            await conn.execute(text(f"DROP DATABASE IF EXISTS `{archive_name}`"))
        await admin.dispose()


# ------------------------------------------------------------------
# Fixture data
# ------------------------------------------------------------------


async def _seed_dimensions(env):
    m_ = env.mods.models
    async with env.app_engine.begin() as conn:
        await conn.execute(insert(m_.Node.__table__), [
            {"id": 1, "name": "aspen1", "status": m_.NodeStatus.ONLINE},
            {"id": 2, "name": "lynx", "status": m_.NodeStatus.ONLINE},
        ])
        await conn.execute(insert(m_.GPUDevice.__table__), [
            {"id": 10, "node_id": 1, "gpu_index": 0, "name": "NVIDIA H200"},
            {"id": 11, "node_id": 1, "gpu_index": 1, "name": "NVIDIA H200 NVL"},
        ])
        await conn.execute(insert(m_.Backend.__table__), [
            {"id": 5, "name": "qwen-aspen1", "url": "http://aspen1:8000",
             "engine": m_.BackendEngine.VLLM, "status": m_.BackendStatus.HEALTHY,
             "node_id": 1},
        ])


def _node(node_id, ts, server, gpu):
    return {"node_id": node_id, "timestamp": ts,
            "server_power_watts": server, "gpu_power_watts": gpu}


NODE_READINGS = [
    # Bucket T0, node 1: server has a NULL reading, GPU power has none.
    _node(1, m(0), 1000, 400.0),
    _node(1, m(2, 30), 1100, 500.0),
    _node(1, m(4, 59), None, 600.0),
    # Bucket T0+5, node 1: exactly on the boundary; GPU power missing.
    _node(1, m(5), 1200, None),
    # Node 2: bucket T0, and the last complete bucket (T0+45).
    _node(2, m(1), 500, 100.0),
    _node(2, m(49, 59), 700, 200.0),
    # Bucket T0+50 is inside the 10-minute margin at NOW: not complete.
    _node(1, m(50), 1300, 650.0),
    _node(1, m(58), 1400, 700.0),
]


def _gpu(device_id, ts, power, util, util_mem, mem, temp, clock):
    return {"gpu_device_id": device_id, "timestamp": ts, "power_draw_watts": power,
            "utilization_gpu": util, "utilization_memory": util_mem,
            "memory_used_gb": mem, "temperature_gpu": temp, "clock_sm_mhz": clock}


GPU_READINGS = [
    _gpu(10, m(0, 10), 300.0, 50.0, 20.0, 40.0, 60.0, 1500),
    _gpu(10, m(3), 350.0, 70.0, 30.0, 42.0, 64.0, 1700),
    # A reading where every sensor failed still counts as a sample.
    _gpu(11, m(0, 20), None, None, None, None, None, None),
    _gpu(10, m(51), 999.0, 99.0, 99.0, 99.0, 99.0, 1999),  # incomplete bucket
]


def _backend(backend_id, ts, power, util, mem, active, queued, rps):
    return {"backend_id": backend_id, "timestamp": ts, "gpu_power_draw_watts": power,
            "gpu_utilization": util, "gpu_memory_used_gb": mem,
            "active_requests": active, "queued_requests": queued,
            "requests_per_second": rps}


BACKEND_READINGS = [
    _backend(5, m(0, 5), 250.0, 40.0, 30.0, 2, 0, 1.5),
    _backend(5, m(4), 350.0, 60.0, 32.0, 4, 3, 2.5),
]


async def _insert_readings(env, node=(), gpu=(), backend=()):
    m_ = env.mods.models
    async with env.app_engine.begin() as conn:
        for table, rows in (
            (m_.NodeTelemetry.__table__, node),
            (m_.GPUDeviceTelemetry.__table__, gpu),
            (m_.BackendTelemetry.__table__, backend),
        ):
            if rows:
                await conn.execute(insert(table), list(rows))


async def _seed_all(env):
    await _seed_dimensions(env)
    await _insert_readings(env, NODE_READINGS, GPU_READINGS, BACKEND_READINGS)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


async def _run(env, cutoff=None, now=NOW, batch=500, archive_wrapper=None):
    retention = env.mods.retention
    async with env.app_session() as app_db, env.archive_session() as archive_db:
        archive = archive_wrapper(archive_db) if archive_wrapper else archive_db
        return await retention.archive_expired_telemetry(
            app_db, archive, cutoff, batch, now=now
        )


async def _rollups(env, model, key):
    async with env.archive_session() as s:
        rows = (await s.execute(select(model.__table__))).mappings().all()
    return {(r[key], r["bucket_start"]): dict(r) for r in rows}


async def _raw_timestamps(env, model):
    column = model.__table__.c.timestamp
    async with env.app_session() as s:
        rows = (await s.execute(select(column).order_by(column))).scalars().all()
    return list(rows)


def _models(env):
    a = env.mods.archive
    return (a.ArchivedNodeTelemetry5m, a.ArchivedGPUDeviceTelemetry5m,
            a.ArchivedBackendTelemetry5m)


class _FailingArchive:
    """Archive session proxy that raises on the Nth INSERT statement."""

    def __init__(self, session, fail_on_insert, table=None):
        self._session = session
        self._fail_on = fail_on_insert
        self._table = table
        self.inserts = 0

    async def execute(self, stmt, *args, **kwargs):
        if isinstance(stmt, Insert) and self._table in (None, stmt.table.name):
            self.inserts += 1
            if self.inserts == self._fail_on:
                raise RuntimeError("simulated archive failure")
        return await self._session.execute(stmt, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._session, name)


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


async def test_aggregates_per_bucket_and_skips_incomplete_buckets():
    async with _scratch() as env:
        await _seed_all(env)
        summary = await _run(env)
        node_m, gpu_m, backend_m = _models(env)

        nodes = await _rollups(env, node_m, "node_id")
        assert set(nodes) == {(1, m(0)), (1, m(5)), (2, m(0)), (2, m(45))}

        b = nodes[(1, m(0))]
        assert b["node_name"] == "aspen1" and b["sample_count"] == 3
        assert b["server_power_watts_avg"] == pytest.approx(1050.0)
        assert (b["server_power_watts_min"], b["server_power_watts_max"]) == (1000, 1100)
        assert b["server_power_samples"] == 2  # the NULL reading is not a sample
        assert b["gpu_power_watts_avg"] == pytest.approx(500.0)
        assert (b["gpu_power_watts_min"], b["gpu_power_watts_max"]) == (400, 600)
        assert b["gpu_power_samples"] == 3
        assert b["archived_at"] is not None

        b = nodes[(1, m(5))]
        assert b["sample_count"] == 1 and b["server_power_watts_avg"] == 1200
        assert b["gpu_power_watts_avg"] is None and b["gpu_power_samples"] == 0

        assert nodes[(2, m(0))]["node_name"] == "lynx"
        assert nodes[(2, m(45))]["server_power_watts_max"] == 700

        gpus = await _rollups(env, gpu_m, "gpu_device_id")
        assert set(gpus) == {(10, m(0)), (11, m(0))}
        g = gpus[(10, m(0))]
        assert (g["node_id"], g["gpu_index"], g["gpu_name"]) == (1, 0, "NVIDIA H200")
        assert g["sample_count"] == 2 and g["power_samples"] == 2
        assert g["power_draw_watts_avg"] == pytest.approx(325.0)
        assert (g["power_draw_watts_min"], g["power_draw_watts_max"]) == (300, 350)
        assert g["utilization_gpu_avg"] == pytest.approx(60.0)
        assert g["utilization_gpu_max"] == pytest.approx(70.0)
        assert g["utilization_memory_avg"] == pytest.approx(25.0)
        assert g["memory_used_gb_avg"] == pytest.approx(41.0)
        assert g["memory_used_gb_max"] == pytest.approx(42.0)
        assert g["temperature_gpu_avg"] == pytest.approx(62.0)
        assert g["temperature_gpu_max"] == pytest.approx(64.0)
        assert g["clock_sm_mhz_avg"] == pytest.approx(1600.0)
        g = gpus[(11, m(0))]
        assert g["gpu_name"] == "NVIDIA H200 NVL" and g["gpu_index"] == 1
        assert g["sample_count"] == 1 and g["power_samples"] == 0
        assert g["power_draw_watts_avg"] is None

        backends = await _rollups(env, backend_m, "backend_id")
        assert set(backends) == {(5, m(0))}
        k = backends[(5, m(0))]
        assert (k["backend_name"], k["node_id"]) == ("qwen-aspen1", 1)
        assert k["sample_count"] == 2 and k["power_samples"] == 2
        assert k["gpu_power_draw_watts_avg"] == pytest.approx(300.0)
        assert k["gpu_power_draw_watts_max"] == pytest.approx(350.0)
        assert k["gpu_utilization_avg"] == pytest.approx(50.0)
        assert k["gpu_memory_used_gb_avg"] == pytest.approx(31.0)
        assert k["active_requests_avg"] == pytest.approx(3.0)
        assert k["active_requests_max"] == 4
        assert k["queued_requests_avg"] == pytest.approx(1.5)
        assert k["queued_requests_max"] == 3
        assert k["requests_per_second_avg"] == pytest.approx(2.0)

        for table in ("node_telemetry", "gpu_device_telemetry", "backend_telemetry"):
            assert summary[table]["rolled_through"] == m(50).isoformat()
            assert summary[table]["deleted"] == 0
            assert "error" not in summary[table]
        assert summary["node_telemetry"]["rolled_up"] == 4


async def test_second_run_is_idempotent_and_resumes_from_the_watermark():
    async with _scratch() as env:
        await _seed_all(env)
        await _run(env)
        node_m, gpu_m, backend_m = _models(env)
        before = await _rollups(env, node_m, "node_id")

        again = await _run(env)
        assert all(again[t]["rolled_up"] == 0 for t in again)
        assert await _rollups(env, node_m, "node_id") == before

        # New readings arrive; the clock moves on by 10 minutes.
        await _insert_readings(env, node=[
            _node(1, m(52), 1500, 750.0),   # bucket T0+50, with 1300 and 1400
            _node(2, m(61), 900, 300.0),    # bucket T0+60: incomplete at NOW+10
        ])
        later = await _run(env, now=NOW + timedelta(minutes=10))
        nodes = await _rollups(env, node_m, "node_id")
        assert set(nodes) - set(before) == {(1, m(50)), (1, m(55))}
        assert nodes[(1, m(50))]["sample_count"] == 2
        assert nodes[(1, m(50))]["server_power_watts_avg"] == pytest.approx(1400.0)
        assert nodes[(1, m(55))]["server_power_watts_avg"] == pytest.approx(1400.0)
        # Rows already archived were not touched.
        assert {k: nodes[k] for k in before} == before
        assert later["node_telemetry"]["rolled_up"] == 2
        assert later["node_telemetry"]["rolled_through"] == m(60).isoformat()


async def test_rerunning_a_committed_window_inserts_nothing(monkeypatch):
    async with _scratch() as env:
        await _seed_all(env)
        await _run(env)
        node_m = _models(env)[0]
        before = await _rollups(env, node_m, "node_id")
        retention = env.mods.retention

        # Force the planner to hand back the window that is already archived:
        # INSERT IGNORE on (node_id, bucket_start) must make it a no-op.
        real_plan = retention._plan_rollup_windows
        monkeypatch.setattr(
            retention, "_plan_rollup_windows",
            lambda wm, nxt, now: real_plan(None, m(0), now),
        )
        summary = await _run(env)
        assert summary["node_telemetry"]["windows"] == 1
        assert summary["node_telemetry"]["rolled_up"] == 0
        assert await _rollups(env, node_m, "node_id") == before


async def test_delete_never_passes_the_rollup_frontier():
    async with _scratch() as env:
        await _seed_all(env)
        node_m = env.mods.models.NodeTelemetry
        gpu_m = env.mods.models.GPUDeviceTelemetry

        # A cutoff in the future: everything is "expired", yet only readings
        # in archived buckets (before T0+50) may be deleted.
        summary = await _run(env, cutoff=NOW + timedelta(days=1))
        assert await _raw_timestamps(env, node_m) == [m(50), m(58)]
        assert await _raw_timestamps(env, gpu_m) == [m(51)]
        assert summary["node_telemetry"]["deleted"] == 6
        assert summary["gpu_device_telemetry"]["deleted"] == 3
        assert summary["backend_telemetry"]["deleted"] == 2

        # A cutoff older than the frontier wins.
        await _insert_readings(env, node=[_node(2, m(55), 1, 1.0)])
        await _run(env, cutoff=m(55), now=NOW + timedelta(minutes=10))
        assert await _raw_timestamps(env, node_m) == [m(55), m(58)]


async def test_backfill_is_capped_and_delete_waits_for_it():
    async with _scratch() as env:
        await _seed_dimensions(env)
        now = datetime(2026, 10, 30, 12, 0)
        days = range(30, 0, -1)
        await _insert_readings(env, node=[
            _node(1, now - timedelta(days=d), 1000 + d, 10.0) for d in days
        ])
        node_m = _models(env)[0]
        raw = env.mods.models.NodeTelemetry

        summary = await _run(env, cutoff=now - timedelta(days=1), now=now)
        frontier = now - timedelta(days=30) + timedelta(days=14)
        assert summary["node_telemetry"]["rolled_through"] == frontier.isoformat()
        assert summary["node_telemetry"]["windows"] == 14 * 4
        archived = await _rollups(env, node_m, "node_id")
        assert sorted(b for _, b in archived) == [
            now - timedelta(days=d) for d in range(30, 16, -1)
        ]
        # Readings past the frontier survive although older than the cutoff.
        remaining = await _raw_timestamps(env, raw)
        assert remaining == [now - timedelta(days=d) for d in range(16, 0, -1)]

        # The next cycle picks up where the cap stopped: 14 more days, to
        # now - 2 days.  The reading AT the new frontier is not rolled up
        # yet (windows are half-open), so it survives the delete too.
        summary = await _run(env, cutoff=now - timedelta(days=1), now=now)
        frontier = now - timedelta(days=2)
        assert summary["node_telemetry"]["rolled_through"] == frontier.isoformat()
        archived = await _rollups(env, node_m, "node_id")
        assert len(archived) == 28
        assert await _raw_timestamps(env, raw) == [
            now - timedelta(days=2), now - timedelta(days=1),
        ]


async def test_failure_mid_window_leaves_no_partial_window():
    async with _scratch() as env:
        await _seed_dimensions(env)
        retention = env.mods.retention
        node_m = _models(env)[0]
        # Readings at T0+5 and T0+6h05 for two nodes.  With now = T0+12h15
        # the horizon is T0+12h05, so the pass runs two windows,
        # [T0+5, T0+6h05) and [T0+6h05, T0+12h05), each holding two buckets.
        readings = [_node(n, m(mins), 100 * n, 1.0) for n in (1, 2) for mins in (5, 365)]
        await _insert_readings(env, node=readings)
        now = T0 + timedelta(hours=12, minutes=15)

        async with env.app_session() as app_db, env.archive_session() as archive_db:
            # One row per INSERT: window 1 is inserts 1-2, window 2 is 3-4.
            # Fail on insert 4, after window 2's first row was already sent.
            failing = _FailingArchive(archive_db, fail_on_insert=4)
            with pytest.raises(RuntimeError, match="simulated archive failure"):
                await retention._rollup_telemetry_table(
                    app_db, failing, retention._NODE_ROLLUP, node_m, now,
                    insert_batch=1,
                )
            assert failing.inserts == 4
        # Window 1 committed; none of window 2 may be visible, not even the
        # row its first INSERT wrote, so the watermark cannot skip node 2.
        assert set(await _rollups(env, node_m, "node_id")) == {(1, m(5)), (2, m(5))}

        async with env.app_session() as app_db, env.archive_session() as archive_db:
            progress = await retention._rollup_telemetry_table(
                app_db, archive_db, retention._NODE_ROLLUP, node_m, now,
                insert_batch=1,
            )
        assert set(await _rollups(env, node_m, "node_id")) == {
            (1, m(5)), (2, m(5)), (1, m(365)), (2, m(365)),
        }
        assert progress["rolled_through"] == T0 + timedelta(hours=12, minutes=5)


async def test_failed_table_keeps_its_raw_rows_others_proceed():
    async with _scratch() as env:
        await _seed_all(env)

        def fail_node_inserts(session):
            return _FailingArchive(session, 1, table="archived_node_telemetry_5m")

        summary = await _run(
            env, cutoff=NOW + timedelta(days=1), archive_wrapper=fail_node_inserts
        )
        assert "simulated archive failure" in summary["node_telemetry"]["error"]
        assert summary["node_telemetry"]["deleted"] == 0
        node_m, gpu_m, _ = _models(env)
        assert await _rollups(env, node_m, "node_id") == {}
        assert len(await _raw_timestamps(env, env.mods.models.NodeTelemetry)) == len(
            NODE_READINGS
        )
        # The other tables rolled up and deleted up to their frontier.
        assert set(await _rollups(env, gpu_m, "gpu_device_id")) == {(10, m(0)), (11, m(0))}
        assert await _raw_timestamps(env, env.mods.models.GPUDeviceTelemetry) == [m(51)]
        events = [(lvl, e, k.get("table"), k.get("reason"))
                  for (lvl, e, k) in env.mods.log.records]
        assert ("warning", "retention_telemetry_delete_skipped", "node_telemetry",
                "rollup_failed") in events


async def test_buckets_ignore_the_session_time_zone():
    # A session zone whose offset is not a multiple of 5 minutes: bucketing
    # through UNIX_TIMESTAMP/FROM_UNIXTIME would land on :03, :08, ...
    async with _scratch(app_time_zone="+00:03") as env:
        await _seed_all(env)
        await _run(env)
        nodes = await _rollups(env, _models(env)[0], "node_id")
        assert set(nodes) == {(1, m(0)), (1, m(5)), (2, m(0)), (2, m(45))}


async def test_default_clock_is_the_databases():
    # The app DB's session clock runs 7 hours behind UTC.  Readings stamped
    # by its NOW() must be judged by that clock, not the host's.
    async with _scratch(app_time_zone="-07:00") as env:
        await _seed_dimensions(env)
        async with env.app_session() as s:
            db_now = (await s.execute(text("SELECT NOW()"))).scalar()
        old = db_now - timedelta(minutes=30)
        await _insert_readings(env, node=[_node(1, old, 800, 80.0)])
        # Stamped by the server default, exactly as the registry writes it.
        async with env.app_engine.begin() as conn:
            await conn.execute(text(
                "INSERT INTO node_telemetry (node_id, server_power_watts) VALUES (2, 900)"
            ))

        await _run(env, now=None)
        nodes = await _rollups(env, _models(env)[0], "node_id")
        floor = env.mods.retention._floor_to_bucket
        assert set(nodes) == {(1, floor(old))}


async def test_purge_drops_rollups_older_than_tier2():
    async with _scratch() as env:
        retention = env.mods.retention
        node_m, gpu_m, backend_m = _models(env)
        legacy = env.mods.archive.ArchivedNodeTelemetry
        # The purge measures tier 2 from the real clock (naive UTC here).
        # DATETIME keeps whole seconds.
        now = datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)
        old, recent = now - timedelta(days=400), now - timedelta(days=300)
        async with env.archive_engine.begin() as conn:
            await conn.execute(insert(node_m.__table__), [
                {"node_id": 1, "bucket_start": old, "sample_count": 1},
                {"node_id": 1, "bucket_start": recent, "sample_count": 1},
            ])
            await conn.execute(insert(gpu_m.__table__), [
                {"gpu_device_id": 10, "bucket_start": old, "sample_count": 1},
            ])
            await conn.execute(insert(backend_m.__table__), [
                {"backend_id": 5, "bucket_start": recent, "sample_count": 1},
            ])
            await conn.execute(insert(legacy.__table__), [
                {"id": 1, "node_id": 1, "timestamp": old, "archived_at": old},
                {"id": 2, "node_id": 1, "timestamp": old, "archived_at": recent},
            ])

        config = dict(retention._DEFAULTS)
        config["retention.telemetry.tier2_days"] = 365
        async with env.archive_session() as archive_db:
            counts = await retention.purge_expired_archives(archive_db, config)

        assert counts["node_telemetry_5m"] == 1
        assert counts["gpu_device_telemetry_5m"] == 1
        assert counts["backend_telemetry_5m"] == 0
        assert counts["node_telemetry"] == 1  # legacy row, by archived_at
        assert set(await _rollups(env, node_m, "node_id")) == {(1, recent)}
        assert await _rollups(env, gpu_m, "gpu_device_id") == {}
        assert set(await _rollups(env, backend_m, "backend_id")) == {(5, recent)}


async def test_rollup_purge_loops_over_batches():
    async with _scratch() as env:
        retention = env.mods.retention
        node_m = _models(env)[0]
        async with env.archive_engine.begin() as conn:
            await conn.execute(insert(node_m.__table__), [
                {"node_id": n, "bucket_start": m(5 * i), "sample_count": 1}
                for n in (1, 2) for i in range(4)
            ])
        async with env.archive_session() as archive_db:
            deleted = await retention._purge_rollups_before(
                archive_db, node_m, m(15), batch=2
            )
        assert deleted == 6
        assert set(await _rollups(env, node_m, "node_id")) == {(1, m(15)), (2, m(15))}
