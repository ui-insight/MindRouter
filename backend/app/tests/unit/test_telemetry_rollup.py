############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_telemetry_rollup.py: 5-minute telemetry rollups in the
#     archive DB: planning helpers, transaction boundaries,
#     retention wiring, archive models, and the Retention page.
#
############################################################

"""Unit tests for the continuous 5-minute telemetry rollups.

The SQL itself is exercised against a real MariaDB in
``tests/integration/test_telemetry_rollup_mariadb.py``.  These tests cover
everything that needs no database: the pure planning helpers, the order of
commits and rollbacks around each window, which telemetry path each
tier/archive combination takes, the purge and stats wiring, the archive
models, and the rendered Retention page.

Import hygiene: ``retention.py`` and ``archive_models.py`` are loaded from
their files under private module names, and retention's only heavy
top-level import, ``logging_config``, is stubbed for the load and restored.
retention imports the session, settings, chat_crud and archive-model modules
at call time, so the tests that reach those imports pin fresh stub modules
for the duration of the call (``_pinned_modules``) and restore every key
afterwards.  No pre-existing sys.modules entry is ever mutated.
"""

import ast
import contextlib
import importlib.util
import re
import sys
import types
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path

import pytest
from jinja2 import ChoiceLoader, DictLoader, Environment, FileSystemLoader
from sqlalchemy.dialects import mysql
from sqlalchemy.sql.dml import Delete, Insert
from sqlalchemy.sql.selectable import Select

_APP = Path(__file__).resolve().parents[2]
_RETENTION_PATH = _APP / "services" / "retention.py"
_ARCHIVE_MODELS_PATH = _APP / "db" / "archive_models.py"
_TEMPLATES = _APP / "dashboard" / "templates"
_REPO = _APP.parents[1]

_MISSING = object()


@contextlib.contextmanager
def _pinned_modules(mapping):
    """Force-set sys.modules keys for the block, then restore each one."""
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


class _RecordingLogger:
    """structlog-shaped logger that records ``(level, event, kwargs)``."""

    def __init__(self):
        self.records = []

    def __getattr__(self, level):
        def _log(event, *args, **kwargs):
            self.records.append((level, event, kwargs))

        return _log

    def events(self, level=None):
        return [e for (lvl, e, _) in self.records if level in (None, lvl)]


def _load_file(module_name, path, stubs=None):
    """Load one source file as a private module, with ``stubs`` pinned."""
    with _pinned_modules(stubs or {}):
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        # Registered under its private name only (as test_ws8 does), so
        # dataclass/NamedTuple machinery can resolve the module.
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module


_LOG = _RecordingLogger()
_logging_stub = types.ModuleType("backend.app.logging_config")
_logging_stub.get_logger = lambda *a, **k: _LOG

retention = _load_file(
    "telemetry_rollup_test_retention",
    _RETENTION_PATH,
    {"backend.app.logging_config": _logging_stub},
)
archive_models = _load_file(
    "telemetry_rollup_test_archive_models", _ARCHIVE_MODELS_PATH
)

ROLLUP_MODELS = {
    "node_telemetry": archive_models.ArchivedNodeTelemetry5m,
    "gpu_device_telemetry": archive_models.ArchivedGPUDeviceTelemetry5m,
    "backend_telemetry": archive_models.ArchivedBackendTelemetry5m,
}


@pytest.fixture(autouse=True)
def _clear_log():
    _LOG.records.clear()
    yield


def dt(*args):
    return datetime(*args)


# ===================================================================
# Pure helpers
# ===================================================================


class TestFloorToBucket:
    def test_boundary_is_unchanged(self):
        assert retention._floor_to_bucket(dt(2026, 10, 30, 12, 5)) == dt(2026, 10, 30, 12, 5)

    def test_floors_minutes_seconds_and_microseconds(self):
        assert retention._floor_to_bucket(
            dt(2026, 10, 30, 12, 9, 59, 999999)
        ) == dt(2026, 10, 30, 12, 5)
        assert retention._floor_to_bucket(dt(2026, 10, 30, 12, 4, 59)) == dt(2026, 10, 30, 12, 0)

    def test_last_bucket_of_the_hour(self):
        assert retention._floor_to_bucket(dt(2026, 10, 30, 23, 59, 59)) == dt(2026, 10, 30, 23, 55)

    def test_aware_input_is_converted_to_naive_utc(self):
        pacific = timezone(timedelta(hours=-7))
        got = retention._floor_to_bucket(datetime(2026, 10, 30, 5, 7, tzinfo=pacific))
        assert got == dt(2026, 10, 30, 12, 5)
        assert got.tzinfo is None

    def test_half_hour_offset_zone_lands_on_utc_buckets(self):
        india = timezone(timedelta(hours=5, minutes=30))
        assert retention._floor_to_bucket(
            datetime(2026, 10, 30, 17, 38, tzinfo=india)
        ) == dt(2026, 10, 30, 12, 5)


class TestPlanRollupWindows:
    NOW = dt(2026, 10, 30, 12, 7, 30)

    def plan(self, watermark, next_reading, now=None, **kw):
        return retention._plan_rollup_windows(watermark, next_reading, now or self.NOW, **kw)

    def test_horizon_is_now_minus_margin_floored(self):
        # 12:07:30 - 10 min = 11:57:30, floored: the bucket [11:55, 12:00)
        # is not complete yet.
        assert self.plan(None, None).horizon == dt(2026, 10, 30, 11, 55)

    def test_both_empty_has_no_anchor(self):
        plan = self.plan(None, None)
        assert plan.start is None and plan.windows == []

    def test_watermark_without_new_readings_resumes_after_it(self):
        plan = self.plan(dt(2026, 10, 30, 9, 0), None)
        assert plan.start == dt(2026, 10, 30, 9, 5)
        assert plan.windows == []

    def test_empty_archive_starts_at_oldest_reading_bucket(self):
        plan = self.plan(None, dt(2026, 10, 30, 8, 3, 17))
        assert plan.start == dt(2026, 10, 30, 8, 0)
        assert plan.windows == [
            (dt(2026, 10, 30, 8, 0), dt(2026, 10, 30, 11, 55)),
        ]

    def test_resumes_at_bucket_after_watermark(self):
        plan = self.plan(dt(2026, 10, 30, 9, 0), dt(2026, 10, 30, 9, 5, 2))
        assert plan.start == dt(2026, 10, 30, 9, 5)
        assert plan.windows[0][0] == dt(2026, 10, 30, 9, 5)

    def test_skips_ahead_over_a_gap_with_no_readings(self):
        # Nothing between the watermark and the next reading exists, so the
        # pass must not walk the gap window by window, cycle after cycle.
        plan = self.plan(dt(2026, 9, 1, 0, 0), dt(2026, 10, 30, 6, 1))
        assert plan.start == dt(2026, 10, 30, 6, 0)

    def test_never_replans_buckets_already_archived(self):
        # Defensive: a next_reading before the watermark's successor bucket
        # (the caller should never pass one) must not move the start back.
        plan = self.plan(dt(2026, 10, 30, 9, 0), dt(2026, 10, 30, 8, 0))
        assert plan.start == dt(2026, 10, 30, 9, 5)

    def test_windows_are_contiguous_six_hours_and_end_at_horizon(self):
        now = dt(2026, 10, 30, 20, 13)
        plan = self.plan(None, dt(2026, 10, 30, 0, 0), now)
        assert plan.horizon == dt(2026, 10, 30, 20, 0)
        assert plan.windows == [
            (dt(2026, 10, 30, 0, 0), dt(2026, 10, 30, 6, 0)),
            (dt(2026, 10, 30, 6, 0), dt(2026, 10, 30, 12, 0)),
            (dt(2026, 10, 30, 12, 0), dt(2026, 10, 30, 18, 0)),
            (dt(2026, 10, 30, 18, 0), dt(2026, 10, 30, 20, 0)),
        ]

    def test_backfill_is_capped_at_fourteen_days_per_pass(self):
        now = dt(2026, 10, 30, 12, 0)
        plan = self.plan(None, now - timedelta(days=90), now)
        assert len(plan.windows) == 14 * 4
        assert plan.windows[0][0] == dt(2026, 8, 1, 12, 0)
        assert plan.windows[-1][1] == dt(2026, 8, 15, 12, 0)
        assert all(b - a == timedelta(hours=6) for a, b in plan.windows)

    def test_next_pass_continues_where_the_capped_one_stopped(self):
        now = dt(2026, 10, 30, 12, 0)
        first = self.plan(None, now - timedelta(days=90), now)
        last_bucket = first.windows[-1][1] - timedelta(minutes=5)
        second = self.plan(last_bucket, first.windows[-1][1], now)
        assert second.start == first.windows[-1][1]

    def test_nothing_complete_yet(self):
        plan = self.plan(None, dt(2026, 10, 30, 12, 1))
        assert plan.start == dt(2026, 10, 30, 12, 0)
        assert plan.windows == []

    def test_caught_up_watermark_plans_nothing(self):
        plan = self.plan(dt(2026, 10, 30, 11, 50), dt(2026, 10, 30, 11, 56))
        assert plan.windows == []

    def test_every_boundary_is_bucket_aligned(self):
        plan = self.plan(None, dt(2026, 10, 1, 3, 17, 41), dt(2026, 10, 9, 4, 44, 12))
        for a, b in plan.windows:
            for edge in (a, b):
                assert edge.minute % 5 == 0 and edge.second == 0 and edge.microsecond == 0

    def test_custom_caps(self):
        plan = self.plan(
            None, dt(2026, 10, 30, 0, 0), dt(2026, 10, 30, 12, 0),
            window=timedelta(hours=1), max_span=timedelta(hours=3),
            margin=timedelta(0),
        )
        assert [w[0].hour for w in plan.windows] == [0, 1, 2]
        assert plan.windows[-1][1] == dt(2026, 10, 30, 3, 0)

    def test_aware_now_is_read_as_utc(self):
        now = datetime(2026, 10, 30, 12, 7, 30, tzinfo=timezone.utc)
        assert self.plan(None, None, now).horizon == dt(2026, 10, 30, 11, 55)


class TestEffectiveDeleteCutoff:
    def test_no_cutoff_deletes_nothing(self):
        assert retention._effective_delete_cutoff(None, dt(2026, 10, 30)) is None

    def test_no_frontier_deletes_nothing(self):
        assert retention._effective_delete_cutoff(dt(2026, 10, 30), None) is None

    def test_cutoff_older_than_frontier_wins(self):
        assert retention._effective_delete_cutoff(
            dt(2026, 8, 1), dt(2026, 10, 30, 11, 55)
        ) == dt(2026, 8, 1)

    def test_capped_at_an_unrolled_frontier(self):
        # The backfill has only reached Aug 15: rows after it must survive
        # even though they are past the tier-1 cutoff.
        assert retention._effective_delete_cutoff(
            dt(2026, 9, 1), dt(2026, 8, 15, 12, 0)
        ) == dt(2026, 8, 15, 12, 0)

    def test_aware_cutoff_compares_on_utc(self):
        cutoff = datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)
        got = retention._effective_delete_cutoff(cutoff, dt(2026, 10, 1))
        assert got == dt(2026, 8, 1) and got.tzinfo is None


# ===================================================================
# Static SQL contracts
# ===================================================================

_BUCKET_EXPR = (
    "TIMESTAMPADD(MINUTE, FLOOR(TIMESTAMPDIFF(MINUTE, TIMESTAMP'2000-01-01 00:00:00', "
    "t.timestamp) / 5) * 5, TIMESTAMP'2000-01-01 00:00:00') AS bucket_start"
)

_ENTITY = {
    "node_telemetry": ("node_id", "nodes", "LEFT JOIN nodes n ON n.id = a.node_id"),
    "gpu_device_telemetry": (
        "gpu_device_id", "gpu_devices",
        "LEFT JOIN gpu_devices g ON g.id = a.gpu_device_id",
    ),
    "backend_telemetry": (
        "backend_id", "backends", "LEFT JOIN backends b ON b.id = a.backend_id",
    ),
}


def _squash(sql):
    sql = re.sub(r"\s+", " ", sql)
    return re.sub(r"\(\s+", "(", re.sub(r"\s+\)", ")", sql)).strip()


def _outer_aliases(sql):
    outer = sql.split("FROM (", 1)[0]
    return re.findall(r"\bAS (\w+)", outer)


SPECS = {spec.table: spec for spec in retention._ROLLUP_SPECS}


class TestRollupSql:
    def test_one_spec_per_telemetry_table(self):
        assert set(SPECS) == set(retention._TELEMETRY_TABLES)
        assert len(retention._ROLLUP_SPECS) == 3

    @pytest.mark.parametrize("table", sorted(_ENTITY))
    def test_bucket_expression_is_time_zone_free_datetime_arithmetic(self, table):
        sql = _squash(SPECS[table].rollup_sql)
        assert _BUCKET_EXPR in sql
        for tz_dependent in ("UNIX_TIMESTAMP", "FROM_UNIXTIME", "CONVERT_TZ", "NOW()"):
            assert tz_dependent not in sql.upper()

    @pytest.mark.parametrize("table", sorted(_ENTITY))
    def test_reads_one_half_open_window_of_its_own_table(self, table):
        sql = _squash(SPECS[table].rollup_sql)
        entity, _, join = _ENTITY[table]
        assert f"FROM {table} t WHERE t.timestamp >= :start AND t.timestamp < :end" in sql
        assert f"GROUP BY t.{entity}, bucket_start" in sql
        assert join in sql

    @pytest.mark.parametrize("table", sorted(_ENTITY))
    def test_aliases_are_exactly_the_archive_columns(self, table):
        model = ROLLUP_MODELS[table]
        columns = {c.name for c in model.__table__.columns} - {"id", "archived_at"}
        aliases = _outer_aliases(SPECS[table].rollup_sql)
        assert len(aliases) == len(set(aliases))
        assert set(aliases) == columns

    @pytest.mark.parametrize("table", sorted(_ENTITY))
    def test_counts_non_null_samples_beside_each_average(self, table):
        sql = _squash(SPECS[table].rollup_sql)
        assert "COUNT(*) AS sample_count" in sql
        assert re.search(r"COUNT\(t\.\w+\) AS \w*samples", sql)

    @pytest.mark.parametrize("table", sorted(_ENTITY))
    def test_next_reading_and_delete_statements(self, table):
        spec = SPECS[table]
        assert _squash(spec.next_reading_sql) == (
            f"SELECT MIN(timestamp) FROM {table} WHERE timestamp >= :after"
        )
        assert _squash(spec.delete_sql) == (
            f"DELETE FROM {table} WHERE timestamp < :cutoff LIMIT :batch"
        )

    def test_specs_are_literal_strings_not_interpolated(self):
        """Every SQL argument of every _RollupSpec(...) is a plain string
        literal: no f-string, name, call or concatenation reaches it."""
        tree = ast.parse(_RETENTION_PATH.read_text())
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "_RollupSpec"
        ]
        assert len(calls) == 3
        for call in calls:
            assert not call.args
            for kw in call.keywords:
                assert isinstance(kw.value, ast.Constant), kw.arg
                assert isinstance(kw.value.value, str), kw.arg

    def test_rollup_models_cover_the_same_tables(self):
        with _pinned_modules({"backend.app.db.archive_models": archive_models}):
            models = retention._rollup_models()
        assert models == ROLLUP_MODELS

    def test_defaults_still_leave_telemetry_unarchived(self):
        # tier 2 stays 0 by default (the migration-029 seed): no install
        # starts archiving telemetry without an admin opting in.
        assert retention._DEFAULTS["retention.telemetry.tier2_days"] == 0


# ===================================================================
# Archive models
# ===================================================================


class TestArchiveModels:
    @pytest.mark.parametrize("table,entity", [
        ("archived_node_telemetry_5m", "node_id"),
        ("archived_gpu_device_telemetry_5m", "gpu_device_id"),
        ("archived_backend_telemetry_5m", "backend_id"),
    ])
    def test_shape(self, table, entity):
        t = archive_models.ArchiveBase.metadata.tables[table]
        assert t.c.id.primary_key and t.c.id.autoincrement is True
        assert not t.foreign_keys
        assert not t.c[entity].nullable and not t.c.bucket_start.nullable
        assert t.c.archived_at.server_default is not None
        uniques = [
            tuple(c.name for c in con.columns)
            for con in t.constraints
            if con.__class__.__name__ == "UniqueConstraint"
        ]
        assert (entity, "bucket_start") in uniques
        assert any(
            tuple(c.name for c in ix.columns) == ("bucket_start",) for ix in t.indexes
        )

    def test_denormalized_dimension_fields(self):
        md = archive_models.ArchiveBase.metadata.tables
        node = md["archived_node_telemetry_5m"]
        gpu = md["archived_gpu_device_telemetry_5m"]
        backend = md["archived_backend_telemetry_5m"]
        assert node.c.node_name.type.length == 100 and node.c.node_name.nullable
        assert gpu.c.gpu_name.type.length == 200
        for col in ("node_id", "gpu_index", "gpu_name"):
            assert gpu.c[col].nullable  # LEFT JOIN may find the device gone
        assert backend.c.backend_name.type.length == 100
        assert backend.c.backend_name.nullable and backend.c.node_id.nullable

    def test_legacy_raw_archive_tables_are_untouched(self):
        md = archive_models.ArchiveBase.metadata.tables
        for table in ("archived_node_telemetry", "archived_gpu_device_telemetry",
                      "archived_backend_telemetry"):
            assert md[table].c.timestamp is not None
            assert md[table].c.id.autoincrement is False


# ===================================================================
# _rollup_telemetry_table: transaction boundaries (fake sessions)
# ===================================================================


class _Result:
    def __init__(self, scalar=None, rows=(), rowcount=0):
        self._scalar = scalar
        self._rows = list(rows)
        self.rowcount = rowcount

    def scalar(self):
        return self._scalar

    def mappings(self):
        return types.SimpleNamespace(all=lambda: list(self._rows))


class _FakeAppDb:
    def __init__(self, next_reading, rows_per_window):
        self.next_reading = next_reading
        self.rows_per_window = rows_per_window
        self.events = []

    async def execute(self, stmt, params=None):
        sql = str(stmt)
        if "SELECT MIN(timestamp)" in sql:
            self.events.append(("next_reading", params["after"]))
            return _Result(scalar=self.next_reading)
        if "GROUP BY" in sql:
            self.events.append(("rollup", params["start"], params["end"]))
            return _Result(rows=[
                {"node_id": i, "bucket_start": params["start"], "sample_count": 1}
                for i in range(self.rows_per_window)
            ])
        raise AssertionError(f"unexpected app statement: {sql[:80]}")

    async def commit(self):
        self.events.append(("commit",))

    async def rollback(self):
        self.events.append(("rollback",))


class _FakeArchiveDb:
    def __init__(self, watermark=None, fail_on_insert=None):
        self.watermark = watermark
        self.fail_on_insert = fail_on_insert
        self.inserts = 0
        self.events = []

    async def execute(self, stmt, params=None):
        if isinstance(stmt, Select):
            self.events.append(("max",))
            return _Result(scalar=self.watermark)
        if isinstance(stmt, Insert):
            self.inserts += 1
            assert "INSERT IGNORE INTO archived_node_telemetry_5m" in str(
                stmt.compile(dialect=mysql.dialect())
            )
            if self.inserts == self.fail_on_insert:
                self.events.append(("insert_failed",))
                raise RuntimeError("archive went away")
            self.events.append(("insert",))
            return _Result(rowcount=2)
        raise AssertionError(f"unexpected archive statement: {stmt}")

    async def commit(self):
        self.events.append(("commit",))

    async def rollback(self):
        self.events.append(("rollback",))


class TestRollupTransactions:
    NOW = dt(2026, 10, 30, 12, 13)  # horizon 12:00

    async def run(self, app, archive, **kw):
        return await retention._rollup_telemetry_table(
            app, archive, SPECS["node_telemetry"],
            archive_models.ArchivedNodeTelemetry5m, self.NOW, **kw,
        )

    async def test_one_archive_commit_per_window_after_all_its_batches(self):
        app = _FakeAppDb(dt(2026, 10, 29, 23, 58), rows_per_window=5)
        archive = _FakeArchiveDb()
        progress = await self.run(app, archive, insert_batch=2)

        # Windows: [23:55, 05:55), [05:55, 11:55), [11:55, 12:00).
        window_events = [("insert",)] * 3 + [("commit",)]
        assert archive.events == [("max",), ("commit",)] + window_events * 3
        assert progress["windows"] == 3
        assert progress["rows_written"] == 3 * 3 * 2
        assert progress["rolled_through"] == dt(2026, 10, 30, 12, 0)
        assert progress["more_pending"] is False

    async def test_app_read_transaction_ends_after_every_window(self):
        app = _FakeAppDb(dt(2026, 10, 30, 0, 0), rows_per_window=1)
        await self.run(app, _FakeArchiveDb())
        kinds = [e[0] for e in app.events]
        assert kinds == ["next_reading", "commit"] + ["rollup", "commit"] * 2
        # Each rollup read is committed before the next statement runs.
        for i, kind in enumerate(kinds):
            if kind == "rollup":
                assert kinds[i + 1] == "commit"

    async def test_seeks_from_the_bucket_after_the_watermark(self):
        app = _FakeAppDb(None, rows_per_window=0)
        progress = await self.run(app, _FakeArchiveDb(watermark=dt(2026, 10, 30, 9, 40)))
        assert app.events[0] == ("next_reading", dt(2026, 10, 30, 9, 45))
        assert progress["windows"] == 0
        assert progress["rolled_through"] == dt(2026, 10, 30, 9, 45)

    async def test_empty_archive_seeks_from_the_epoch(self):
        app = _FakeAppDb(None, rows_per_window=0)
        progress = await self.run(app, _FakeArchiveDb())
        assert app.events[0] == ("next_reading", retention._ROLLUP_EPOCH)
        assert progress["rolled_through"] is None

    async def test_empty_window_still_advances_the_frontier(self):
        app = _FakeAppDb(dt(2026, 10, 30, 6, 0), rows_per_window=0)
        archive = _FakeArchiveDb()
        progress = await self.run(app, archive)
        assert ("insert",) not in archive.events
        assert progress["rolled_through"] == dt(2026, 10, 30, 12, 0)

    async def test_failure_mid_window_rolls_the_window_back_and_raises(self):
        app = _FakeAppDb(dt(2026, 10, 29, 23, 58), rows_per_window=5)
        # Batches of 2 → 3 inserts per window; fail on the 2nd insert of
        # the 2nd window (insert #5).
        archive = _FakeArchiveDb(fail_on_insert=5)
        with pytest.raises(RuntimeError, match="archive went away"):
            await self.run(app, archive, insert_batch=2)
        assert archive.events == [
            ("max",), ("commit",),
            ("insert",), ("insert",), ("insert",), ("commit",),
            ("insert",), ("insert_failed",), ("rollback",),
        ]

    async def test_capped_backfill_reports_more_pending(self):
        app = _FakeAppDb(self.NOW - timedelta(days=60), rows_per_window=0)
        progress = await self.run(app, _FakeArchiveDb())
        assert progress["windows"] == 56
        assert progress["more_pending"] is True


# ===================================================================
# archive_expired_telemetry: rollup, then capped delete
# ===================================================================


class _NullDb:
    def __init__(self):
        self.rollbacks = 0

    async def rollback(self):
        self.rollbacks += 1

    async def commit(self):
        pass


@pytest.fixture
def orchestration(monkeypatch):
    """Stub the per-table rollup and delete; record what was asked."""
    state = types.SimpleNamespace(
        frontiers={
            "node_telemetry": dt(2026, 10, 30, 11, 55),
            "gpu_device_telemetry": dt(2026, 8, 15, 12, 0),
            "backend_telemetry": None,
        },
        failing=set(),
        rollup_now=[],
        deletes=[],
    )

    async def fake_rollup(app_db, archive_db, spec, model, now, **kw):
        state.rollup_now.append(now)
        assert model is ROLLUP_MODELS[spec.table]
        if spec.table in state.failing:
            raise RuntimeError(f"{spec.table} boom")
        frontier = state.frontiers[spec.table]
        return {"rows_written": 7, "windows": 2, "rolled_through": frontier,
                "more_pending": False}

    async def fake_delete(app_db, spec, before, batch):
        state.deletes.append((spec.table, before, batch))
        return 11

    async def fake_db_now(app_db):
        return dt(2026, 10, 30, 12, 13)

    monkeypatch.setattr(retention, "_rollup_telemetry_table", fake_rollup)
    monkeypatch.setattr(retention, "_delete_telemetry_before", fake_delete)
    monkeypatch.setattr(retention, "_telemetry_db_now", fake_db_now)
    monkeypatch.setattr(retention, "_rollup_models", lambda: dict(ROLLUP_MODELS))
    return state


class TestArchiveExpiredTelemetry:
    CUTOFF = datetime(2026, 9, 1, tzinfo=timezone.utc)

    async def test_delete_is_capped_at_each_tables_frontier(self, orchestration):
        summary = await retention.archive_expired_telemetry(
            _NullDb(), _NullDb(), self.CUTOFF, 500
        )
        assert orchestration.deletes == [
            # Rolled through 11:55 today: the tier-1 cutoff applies.
            ("node_telemetry", dt(2026, 9, 1), 10000),
            # Backfill only reached Aug 15: nothing newer may go.
            ("gpu_device_telemetry", dt(2026, 8, 15, 12, 0), 10000),
            # backend_telemetry has no frontier (never held a reading).
        ]
        assert summary["node_telemetry"] == {
            "rolled_up": 7, "windows": 2,
            "rolled_through": "2026-10-30T11:55:00", "deleted": 11,
        }
        assert summary["backend_telemetry"]["deleted"] == 0
        assert "retention_telemetry_delete_skipped" not in _LOG.events()

    async def test_failed_rollup_skips_that_tables_delete_and_warns(self, orchestration):
        orchestration.failing = {"gpu_device_telemetry"}
        app, archive = _NullDb(), _NullDb()
        summary = await retention.archive_expired_telemetry(
            app, archive, self.CUTOFF, 500
        )
        assert [d[0] for d in orchestration.deletes] == ["node_telemetry"]
        assert "boom" in summary["gpu_device_telemetry"]["error"]
        assert summary["gpu_device_telemetry"]["deleted"] == 0
        warnings = [(e, k) for (lvl, e, k) in _LOG.records if lvl == "warning"]
        assert ("retention_telemetry_rollup_failed", "gpu_device_telemetry") in [
            (e, k.get("table")) for e, k in warnings
        ]
        assert ("retention_telemetry_delete_skipped", "rollup_failed") in [
            (e, k.get("reason")) for e, k in warnings
        ]
        # Both sessions were reset so the next table starts clean.
        assert app.rollbacks == 1 and archive.rollbacks == 1

    async def test_rollup_only_call_never_deletes(self, orchestration):
        summary = await retention.archive_expired_telemetry(
            _NullDb(), _NullDb(), None, 500
        )
        assert orchestration.deletes == []
        assert summary["node_telemetry"]["rolled_up"] == 7

    async def test_defaults_to_the_db_clock_and_honors_an_injected_now(self, orchestration):
        await retention.archive_expired_telemetry(_NullDb(), _NullDb(), None, 500)
        assert set(orchestration.rollup_now) == {dt(2026, 10, 30, 12, 13)}
        orchestration.rollup_now.clear()
        injected = dt(2026, 1, 2, 3, 4)
        await retention.archive_expired_telemetry(
            _NullDb(), _NullDb(), None, 500, now=injected
        )
        assert set(orchestration.rollup_now) == {injected}

    async def test_large_configured_batch_is_kept(self, orchestration):
        await retention.archive_expired_telemetry(
            _NullDb(), _NullDb(), self.CUTOFF, 25000
        )
        assert {d[2] for d in orchestration.deletes} == {25000}


# ===================================================================
# run_retention_cycle / run_purge wiring
# ===================================================================


class _FakeSession:
    def __init__(self, kind):
        self.kind = kind

    async def commit(self):
        pass

    async def rollback(self):
        pass

    async def execute(self, *a, **k):
        raise AssertionError("wiring tests must not reach the database")


def _session_stub():
    mod = types.ModuleType("backend.app.db.session")

    @contextlib.asynccontextmanager
    async def get_async_db_context():
        yield _FakeSession("app")

    @contextlib.asynccontextmanager
    async def get_archive_db_context():
        yield _FakeSession("archive")

    mod.get_async_db_context = get_async_db_context
    mod.get_archive_db_context = get_archive_db_context
    mod.get_archive_engine = lambda: None
    return mod


def _wiring_modules(archive_configured):
    settings = types.ModuleType("backend.app.settings")
    settings.get_settings = lambda: types.SimpleNamespace(
        archive_database_url="mysql+pymysql://a/b" if archive_configured else None,
        artifact_storage_path="/nonexistent",
    )
    chat_crud = types.ModuleType("backend.app.db.chat_crud")

    async def delete_all_orphan_attachments(db):
        return 0

    async def delete_expired_conversations(db, days):
        return 0

    chat_crud.delete_all_orphan_attachments = delete_all_orphan_attachments
    chat_crud.delete_expired_conversations = delete_expired_conversations
    db_pkg = types.ModuleType("backend.app.db")
    db_pkg.__path__ = []
    db_pkg.chat_crud = chat_crud
    return {
        "backend.app.db": db_pkg,
        "backend.app.db.session": _session_stub(),
        "backend.app.db.chat_crud": chat_crud,
        "backend.app.settings": settings,
        "backend.app.db.archive_models": archive_models,
    }


def _config(**overrides):
    config = dict(retention._DEFAULTS)
    # Only telemetry is under test: switch every other sweep off.
    config.update({
        "retention.requests.tier1_days": 0,
        "retention.chat.tier1_days": 0,
        "retention.responses_store_days": 0,
        "retention.conversations_days": 0,
        "retention.dlp_alerts_days": 0,
        "retention.web_search_logs_days": 0,
    })
    config.update({f"retention.telemetry.{k}": v for k, v in overrides.items()})
    return config


@pytest.fixture
def wiring(monkeypatch):
    calls = []
    state = types.SimpleNamespace(calls=calls, config=_config(), rollup_error=None)

    async def get_retention_config(db):
        return state.config

    async def archive_expired_telemetry(app_db, archive_db, cutoff, batch_size, **kw):
        assert app_db.kind == "app" and archive_db.kind == "archive"
        calls.append(("archive_expired_telemetry", cutoff))
        if state.rollup_error:
            raise state.rollup_error
        return {"rolled": True}

    async def cleanup_expired_telemetry(app_db, cutoff):
        assert app_db.kind == "app"
        calls.append(("cleanup_expired_telemetry", cutoff))
        return {"deleted": True}

    async def purge_expired_archives(archive_db, config):
        calls.append(("purge_expired_archives", None))
        return {}

    async def cleanup_expired_request_images(*a, **k):
        return {}

    monkeypatch.setattr(retention, "get_retention_config", get_retention_config)
    monkeypatch.setattr(retention, "archive_expired_telemetry", archive_expired_telemetry)
    monkeypatch.setattr(retention, "cleanup_expired_telemetry", cleanup_expired_telemetry)
    monkeypatch.setattr(retention, "purge_expired_archives", purge_expired_archives)
    monkeypatch.setattr(
        retention, "cleanup_expired_request_images", cleanup_expired_request_images
    )
    return state


def _telemetry_calls(calls):
    return [c for c in calls if "telemetry" in c[0]]


def _assert_days_ago(cutoff, days):
    expected = datetime.now(timezone.utc) - timedelta(days=days)
    assert abs((cutoff - expected).total_seconds()) < 60


class TestRetentionCycleWiring:
    @pytest.mark.parametrize("archive,tier1,tier2,expected", [
        # Archive on, tier 2 on: rollup + capped delete.
        (True, 90, 730, ("archive_expired_telemetry", 90)),
        # Tier 1 = 0 keeps raw rows forever, but rollups still run.
        (True, 0, 730, ("archive_expired_telemetry", None)),
        # Tier 2 = 0: no averages kept, expired rows simply deleted.
        (True, 90, 0, ("cleanup_expired_telemetry", 90)),
        # No archive DB: delete-only fallback.
        (False, 90, 730, ("cleanup_expired_telemetry", 90)),
        (False, 0, 730, None),
        (True, 0, 0, None),
    ])
    async def test_telemetry_path(self, wiring, archive, tier1, tier2, expected):
        wiring.config = _config(tier1_days=tier1, tier2_days=tier2)
        with _pinned_modules(_wiring_modules(archive)):
            summary = await retention.run_retention_cycle()

        calls = _telemetry_calls(wiring.calls)
        if expected is None:
            assert calls == []
            assert "telemetry" not in summary
            return
        assert len(calls) == 1
        name, cutoff = calls[0]
        assert name == expected[0]
        if expected[1] is None:
            assert cutoff is None
        else:
            _assert_days_ago(cutoff, expected[1])
        assert summary["telemetry"]

    async def test_archive_purge_still_runs_after_telemetry(self, wiring):
        wiring.config = _config(tier1_days=90, tier2_days=365)
        with _pinned_modules(_wiring_modules(True)):
            await retention.run_retention_cycle()
        names = [c[0] for c in wiring.calls]
        assert names.index("archive_expired_telemetry") < names.index(
            "purge_expired_archives"
        )


class TestManualPurgeWiring:
    async def test_rollup_runs_first_then_the_requested_purge(self, wiring):
        wiring.config = _config(tier1_days=90, tier2_days=365)
        with _pinned_modules(_wiring_modules(True)):
            summary = await retention.run_purge("telemetry", 7)
        calls = _telemetry_calls(wiring.calls)
        assert [c[0] for c in calls] == [
            "archive_expired_telemetry", "cleanup_expired_telemetry",
        ]
        assert calls[0][1] is None  # rollup only: the purge does the deleting
        _assert_days_ago(calls[1][1], 7)
        assert summary["rollup"] == {"rolled": True}

    async def test_purge_proceeds_when_the_rollup_fails(self, wiring):
        wiring.config = _config(tier1_days=90, tier2_days=365)
        wiring.rollup_error = RuntimeError("archive down")
        with _pinned_modules(_wiring_modules(True)):
            summary = await retention.run_purge("telemetry", 0)
        assert [c[0] for c in _telemetry_calls(wiring.calls)] == [
            "archive_expired_telemetry", "cleanup_expired_telemetry",
        ]
        assert "archive down" in summary["rollup"]["error"]
        assert summary["result"] == {"deleted": True}
        assert "purge_telemetry_rollup_failed" in _LOG.events("warning")

    @pytest.mark.parametrize("archive,tier2", [(False, 365), (True, 0)])
    async def test_no_rollup_without_archival(self, wiring, archive, tier2):
        wiring.config = _config(tier1_days=90, tier2_days=tier2)
        with _pinned_modules(_wiring_modules(archive)):
            summary = await retention.run_purge("telemetry", 7)
        assert [c[0] for c in _telemetry_calls(wiring.calls)] == [
            "cleanup_expired_telemetry",
        ]
        assert "rollup" not in summary


# ===================================================================
# purge_expired_archives / get_archive_stats
# ===================================================================


class _PurgeArchiveDb:
    """Records compiled DELETEs; rollup deletes report one full batch first."""

    def __init__(self):
        self.deletes = []
        self.commits = 0
        self._full_batch_sent = set()

    async def execute(self, stmt, params=None):
        if isinstance(stmt, Select):
            return types.SimpleNamespace(all=lambda: [])
        assert isinstance(stmt, Delete)
        compiled = stmt.compile(dialect=mysql.dialect())
        table = stmt.table.name
        self.deletes.append((table, str(compiled), dict(compiled.params)))
        rowcount = 0
        if table.endswith("_5m") and table not in self._full_batch_sent:
            self._full_batch_sent.add(table)
            rowcount = retention._ROLLUP_PURGE_BATCH
        return types.SimpleNamespace(rowcount=rowcount)

    async def flush(self):
        pass

    async def commit(self):
        self.commits += 1


class TestPurgeExpiredArchives:
    async def purge(self, tier2):
        db = _PurgeArchiveDb()
        config = dict(retention._DEFAULTS)
        config["retention.telemetry.tier2_days"] = tier2
        with _pinned_modules({"backend.app.db.archive_models": archive_models}):
            counts = await retention.purge_expired_archives(db, config)
        return db, counts

    async def test_rollups_expire_by_bucket_start_in_batches(self):
        db, counts = await self.purge(365)
        expected_cutoff = (
            datetime.now(timezone.utc) - timedelta(days=365)
        ).replace(tzinfo=None)
        for table in ("archived_node_telemetry_5m",
                      "archived_gpu_device_telemetry_5m",
                      "archived_backend_telemetry_5m"):
            mine = [d for d in db.deletes if d[0] == table]
            # One full batch, then a short one ends the loop.
            assert len(mine) == 2
            for _, sql, params in mine:
                assert f"{table}.bucket_start < %s" in sql
                assert sql.endswith(f"LIMIT {retention._ROLLUP_PURGE_BATCH}")
                (cutoff,) = params.values()
                assert cutoff.tzinfo is None
                assert abs((cutoff - expected_cutoff).total_seconds()) < 60
        assert counts["node_telemetry_5m"] == retention._ROLLUP_PURGE_BATCH
        assert counts["backend_telemetry_5m"] == retention._ROLLUP_PURGE_BATCH

    async def test_legacy_raw_tables_still_expire_by_archived_at(self):
        db, counts = await self.purge(365)
        for table in ("archived_node_telemetry", "archived_gpu_device_telemetry",
                      "archived_backend_telemetry"):
            (entry,) = [d for d in db.deletes if d[0] == table]
            assert f"{table}.archived_at < %s" in entry[1]

    async def test_tier2_zero_purges_no_telemetry(self):
        db, counts = await self.purge(0)
        assert not [d for d in db.deletes if "telemetry" in d[0]]
        assert not any(key.endswith("_5m") for key in counts)


class _StatsDb:
    def __init__(self):
        self.tables = []

    async def execute(self, stmt, params):
        self.tables.append(params["tbl"])
        return types.SimpleNamespace(first=lambda: (10, 1.5))


class TestArchiveStats:
    async def test_rollup_tables_are_counted(self):
        db = _StatsDb()
        stats = await retention.get_archive_stats(db)
        for table in ("archived_node_telemetry_5m",
                      "archived_gpu_device_telemetry_5m",
                      "archived_backend_telemetry_5m"):
            assert table in db.tables
            assert stats["tables"][table] == {"rows": 10, "size_mb": 1.5}
        assert stats["total_rows"] == 10 * len(db.tables)


# ===================================================================
# Retention page (rendered)
# ===================================================================


class _Inputs(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))

    handle_startendtag = handle_starttag


def _render_policies(archive_configured):
    env = Environment(
        loader=ChoiceLoader([
            DictLoader({
                "base.html": "{% block content %}{% endblock %}{% block extra_js %}{% endblock %}",
                "admin/_sidebar.html": "",
            }),
            FileSystemLoader(str(_TEMPLATES)),
        ]),
        autoescape=True,
    )
    html = env.get_template("admin/retention.html").render(
        tab="policies", is_read_only=False, retention_running=False,
        success=None, error=None, archive_configured=archive_configured,
        config=dict(retention._DEFAULTS, **{"retention.telemetry.tier2_days": 365}),
        app_counts={},
    )
    parser = _Inputs()
    parser.feed(html)
    return html, parser.tags


def _by_name(tags, name):
    (attrs,) = [a for t, a in tags if t == "input" and a.get("name") == name]
    return attrs


class TestRetentionPage:
    def test_tier2_input_is_live_when_the_archive_is_configured(self):
        html, tags = _render_policies(True)
        tier2 = _by_name(tags, "retention.telemetry.tier2_days")
        assert "disabled" not in tier2
        assert tier2["min"] == "0" and tier2["value"] == "365"

    def test_tier2_input_disabled_only_without_an_archive(self):
        html, tags = _render_policies(False)
        assert "disabled" in _by_name(tags, "retention.telemetry.tier2_days")
        # Tier 1 still works without an archive (delete-only).
        assert "disabled" not in _by_name(tags, "retention.telemetry.tier1_days")

    def test_copy_describes_rollups_not_deletion_only(self):
        html, _ = _render_policies(True)
        assert "Not archived" not in html
        assert "always 0" not in html
        assert "5-minute averages" in html
        assert "TELEMETRY_RETENTION_DAYS" in html

    def test_telemetry_inputs_are_labelled_and_described(self):
        html, tags = _render_policies(True)
        labels = {a.get("for") for t, a in tags if t == "label"}
        ids = [a["id"] for _, a in tags if "id" in a]
        assert len(ids) == len(set(ids))
        for name in ("retention.telemetry.tier1_days", "retention.telemetry.tier2_days"):
            attrs = _by_name(tags, name)
            assert attrs["id"] in labels
            assert attrs["aria-describedby"] in ids

    def test_post_handler_saves_tier2(self):
        routes = (_APP / "dashboard" / "routes.py").read_text()
        assert '"retention.telemetry.tier2_days",' in routes


class TestDocs:
    def test_docs_describe_rollups(self):
        docs = (_REPO / "docs" / "index.md").read_text()
        for table in ("archived_node_telemetry_5m", "archived_gpu_device_telemetry_5m",
                      "archived_backend_telemetry_5m"):
            assert table in docs
        assert "5-minute averages" in docs
