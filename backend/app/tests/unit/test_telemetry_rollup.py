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
tier/archive combination takes, the purge and stats wiring, the backend
registry's own raw-telemetry delete, the archive models, and the rendered
Retention page.

Import hygiene: ``retention.py`` and ``archive_models.py`` are loaded from
their files under private module names, and retention's only heavy
top-level import, ``logging_config``, is stubbed for the load and restored.
retention imports the session, settings, chat_crud and archive-model modules
at call time, so the tests that reach those imports pin fresh stub modules
for the duration of the call (``_pinned_modules``) and restore every key
afterwards.  No pre-existing sys.modules entry is ever mutated.  registry.py
is never imported (it pulls in the db, adapter and session chain): the one
method under test is compiled from its source on its own.
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
_REGISTRY_PATH = _APP / "core" / "telemetry" / "registry.py"
_TEMPLATES = _APP / "dashboard" / "templates"
_REPO = _APP.parents[1]
_TABLES = ("node_telemetry", "gpu_device_telemetry", "backend_telemetry")

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

    # --- tier 2 bounds where a pass may start -----------------------------

    def test_empty_archive_starts_where_tier2_begins_not_at_the_oldest_reading(self):
        # 60 days of live history, tier 2 = 7 days.  Starting at the oldest
        # reading wrote 14 days that the same cycle's purge deleted again,
        # emptying the archive, so every cycle started over there.
        keep_since = self.NOW - timedelta(days=7)  # Oct 23 12:07:30
        plan = self.plan(None, self.NOW - timedelta(days=60), keep_since=keep_since)
        assert plan.start == dt(2026, 10, 23, 12, 5)
        assert plan.windows[0][0] == dt(2026, 10, 23, 12, 5)
        # Seven days fit in one pass, so it reaches the horizon.
        assert plan.windows[-1][1] == plan.horizon

    def test_watermark_older_than_tier2_resumes_where_tier2_begins(self):
        # Rollups stalled for longer than tier 2: the gap is past keeping.
        plan = self.plan(
            dt(2026, 9, 1), dt(2026, 9, 1, 0, 5), keep_since=dt(2026, 10, 23, 12, 7)
        )
        assert plan.start == dt(2026, 10, 23, 12, 5)

    def test_newer_watermark_is_untouched_by_tier2(self):
        plan = self.plan(
            dt(2026, 10, 30, 9, 0), dt(2026, 10, 30, 9, 5, 2), keep_since=dt(2026, 10, 23)
        )
        assert plan.start == dt(2026, 10, 30, 9, 5)

    def test_no_readings_within_tier2_still_gives_a_frontier(self):
        # Readings older than tier 2 will never be averaged, so the frontier
        # sits where tier 2 begins and the tier-1 delete may remove them.
        plan = self.plan(None, None, keep_since=dt(2026, 10, 23, 12, 7))
        assert plan.start == dt(2026, 10, 23, 12, 5)
        assert plan.windows == []

    def test_uncapped_pass_runs_to_the_horizon(self):
        now = dt(2026, 10, 30, 12, 0)
        plan = self.plan(None, now - timedelta(days=90), now, max_span=None)
        assert plan.windows[0][0] == dt(2026, 8, 1, 12, 0)
        assert plan.windows[-1][1] == plan.horizon == dt(2026, 10, 30, 11, 50)
        # 90 days of 6-hour windows, the last one 10 minutes short.
        assert len(plan.windows) == 90 * 4
        assert all(a == b for (_, a), (b, _) in zip(plan.windows, plan.windows[1:]))


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
        self.insert_calls = []
        self.events = []

    async def execute(self, stmt, params=None):
        if isinstance(stmt, Select):
            self.events.append(("max",))
            return _Result(scalar=self.watermark)
        if isinstance(stmt, Insert):
            self.inserts += 1
            self.insert_calls.append((stmt, params))
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
        # The frontier is the end of the 56th window, not the horizon.  The
        # delete is capped at it: reporting the horizon after a capped pass
        # would let tier 1 delete the 46 days this pass never averaged.
        start = retention._floor_to_bucket(self.NOW - timedelta(days=60))
        assert progress["rolled_through"] == start + timedelta(days=14)
        assert progress["rolled_through"] < dt(2026, 10, 30, 12, 0)

    async def test_uncapped_pass_reaches_the_horizon(self):
        app = _FakeAppDb(self.NOW - timedelta(days=60), rows_per_window=0)
        progress = await self.run(app, _FakeArchiveDb(), max_span=None)
        assert progress["windows"] == 60 * 4
        assert progress["rolled_through"] == dt(2026, 10, 30, 12, 0)
        assert progress["more_pending"] is False

    async def test_rows_go_in_as_executemany_on_one_cached_statement(self):
        app = _FakeAppDb(dt(2026, 10, 29, 23, 58), rows_per_window=5)
        archive = _FakeArchiveDb()
        await self.run(app, archive, insert_batch=2)

        cache_keys = set()
        for stmt, params in archive.insert_calls:
            # The rows travel as executemany parameters, in batches...
            assert isinstance(params, list) and 1 <= len(params) <= 2
            assert all(isinstance(row, dict) for row in params)
            # ...of the single-row statement.  A multi-VALUES statement
            # (.values(rows)) has no cache key, so every 1000-row batch was
            # compiled from scratch (~100 ms for GPU rows) on the event loop.
            sql = str(stmt.compile(dialect=mysql.dialect()))
            assert sql.count("VALUES (") == 1 and "), (" not in sql
            key = stmt._generate_cache_key()
            assert key is not None
            cache_keys.add(key.key)
        # One statement shape for every batch: compiled once, then cached.
        assert len(cache_keys) == 1
        # 3 windows x 5 rows, each sent exactly once.
        sent = [row for _, params in archive.insert_calls for row in params]
        assert len(sent) == 15
        assert [row["node_id"] for row in sent] == list(range(5)) * 3

    async def test_seek_and_plan_start_where_tier2_begins(self):
        # Empty archive, live history far older than tier 2 (7 days).
        keep_since = self.NOW - timedelta(days=7)  # Oct 23 12:13 -> 12:10
        app = _FakeAppDb(dt(2026, 10, 23, 12, 11), rows_per_window=0)
        progress = await self.run(app, _FakeArchiveDb(), keep_since=keep_since)
        assert app.events[0] == ("next_reading", dt(2026, 10, 23, 12, 10))
        # 7 days less 10 minutes, to the horizon: 28 windows in one pass.
        assert progress["windows"] == 28
        assert progress["rolled_through"] == dt(2026, 10, 30, 12, 0)
        assert progress["more_pending"] is False

    async def test_a_watermark_older_than_tier2_resumes_where_tier2_begins(self):
        keep_since = self.NOW - timedelta(days=7)
        app = _FakeAppDb(None, rows_per_window=0)
        progress = await self.run(
            app, _FakeArchiveDb(watermark=dt(2026, 9, 1)), keep_since=keep_since
        )
        assert app.events[0] == ("next_reading", dt(2026, 10, 23, 12, 10))
        # No readings since: the frontier is where tier 2 begins.
        assert progress["rolled_through"] == dt(2026, 10, 23, 12, 10)


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
        rollup_kw=[],
        deletes=[],
    )

    async def fake_rollup(app_db, archive_db, spec, model, now, **kw):
        state.rollup_now.append(now)
        state.rollup_kw.append(kw)
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

    async def test_tier2_bounds_every_tables_pass_on_the_db_clock(self, orchestration):
        await retention.archive_expired_telemetry(
            _NullDb(), _NullDb(), self.CUTOFF, 500, tier2_days=7
        )
        # Measured on the clock that stamped the readings (12:13 here).
        assert orchestration.rollup_kw == [{
            "keep_since": dt(2026, 10, 23, 12, 13),
            "max_span": retention._ROLLUP_MAX_SPAN_PER_CYCLE,
        }] * 3

    async def test_no_tier2_limit_and_an_uncapped_catch_up_pass_through(self, orchestration):
        await retention.archive_expired_telemetry(
            _NullDb(), _NullDb(), None, 500, max_span=None
        )
        assert orchestration.rollup_kw == [{"keep_since": None, "max_span": None}] * 3


# ===================================================================
# _delete_telemetry_before: batched delete, one commit per batch
# ===================================================================


class _DeletingAppDb:
    """App session whose DELETEs report the given rowcounts, in order."""

    def __init__(self, rowcounts):
        self.rowcounts = list(rowcounts)
        self.events = []

    async def execute(self, stmt, params=None):
        self.events.append(("delete", str(stmt), dict(params)))
        return _Result(rowcount=self.rowcounts.pop(0))

    async def commit(self):
        self.events.append(("commit",))


class TestDeleteTelemetryBefore:
    BEFORE = dt(2026, 9, 1)

    async def test_loops_until_a_short_batch_committing_after_each(self):
        db = _DeletingAppDb([500, 500, 3])
        spec = SPECS["gpu_device_telemetry"]
        total = await retention._delete_telemetry_before(db, spec, self.BEFORE, 500)
        assert total == 1003
        # A commit after every batch keeps InnoDB lock time bounded while
        # the poller keeps inserting; one commit at the end would hold row
        # locks on every expired reading at once.
        assert [e[0] for e in db.events] == ["delete", "commit"] * 3
        for event in db.events[::2]:
            assert event[1] == spec.delete_sql
            assert event[2] == {"cutoff": self.BEFORE, "batch": 500}

    async def test_a_full_last_batch_takes_one_more_round(self):
        db = _DeletingAppDb([500, 0])
        total = await retention._delete_telemetry_before(
            db, SPECS["node_telemetry"], self.BEFORE, 500
        )
        assert total == 500
        assert [e[0] for e in db.events] == ["delete", "commit"] * 2


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
    state = types.SimpleNamespace(
        calls=calls, config=_config(), rollup_error=None, rollup_kw=[],
        rollup_result={t: {"rolled_up": 1, "deleted": 0} for t in _TABLES},
    )

    async def get_retention_config(db):
        return state.config

    async def archive_expired_telemetry(app_db, archive_db, cutoff, batch_size, **kw):
        assert app_db.kind == "app" and archive_db.kind == "archive"
        calls.append(("archive_expired_telemetry", cutoff))
        state.rollup_kw.append(kw)
        if state.rollup_error:
            raise state.rollup_error
        return state.rollup_result

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
        if name == "archive_expired_telemetry":
            # Tier 2 bounds the rollup (so the purge below cannot undo it);
            # the scheduled cycle keeps the 14-day cap.
            assert wiring.rollup_kw == [{"tier2_days": tier2}]

    async def test_archive_purge_still_runs_after_telemetry(self, wiring):
        wiring.config = _config(tier1_days=90, tier2_days=365)
        with _pinned_modules(_wiring_modules(True)):
            await retention.run_retention_cycle()
        names = [c[0] for c in wiring.calls]
        assert names.index("archive_expired_telemetry") < names.index(
            "purge_expired_archives"
        )

    @pytest.mark.parametrize("archive,tier2,expected", [
        (True, 730, "archive_expired_telemetry"),
        (True, 0, "cleanup_expired_telemetry"),
        (False, 730, "cleanup_expired_telemetry"),
    ])
    async def test_failing_earlier_steps_do_not_stop_telemetry_pruning(
        self, wiring, monkeypatch, archive, tier2, expected
    ):
        # The retention cycle is the only pruner of raw telemetry, so one
        # broken category (archive DB down, /tmp full for the request dump)
        # must not skip it: before, an exception in the requests step ended
        # the cycle and raw telemetry was never pruned again.
        wiring.config = _config(tier1_days=90, tier2_days=tier2)
        wiring.config["retention.requests.tier1_days"] = 30
        wiring.config["retention.chat.tier1_days"] = 30

        async def boom(*a, **k):
            wiring.calls.append(("boom", None))
            raise ConnectionRefusedError("archive unreachable")

        for fn in ("archive_expired_requests", "delete_expired_requests_no_archive",
                   "archive_expired_chats"):
            monkeypatch.setattr(retention, fn, boom)
        chat_mods = _wiring_modules(archive)
        chat_mods["backend.app.db.chat_crud"].delete_expired_conversations = boom
        with _pinned_modules(chat_mods):
            summary = await retention.run_retention_cycle()

        assert [c[0] for c in _telemetry_calls(wiring.calls)] == [expected]
        assert summary["telemetry"]
        assert summary["failed_steps"] == ["requests", "chats"]
        assert summary["requests"]["error"].startswith("ConnectionRefusedError")
        assert summary["chats"]["error"].startswith("ConnectionRefusedError")
        failed = [k["step"] for (lvl, e, k) in _LOG.records if e == "retention_step_failed"]
        assert failed == ["requests", "chats"]
        assert ("warning", "retention_cycle_partial") in [(lvl, e) for (lvl, e, _) in _LOG.records]

    async def test_a_failing_telemetry_step_does_not_stop_the_rest(self, wiring):
        wiring.config = _config(tier1_days=90, tier2_days=730)
        wiring.rollup_error = RuntimeError("rollup exploded")
        with _pinned_modules(_wiring_modules(True)):
            summary = await retention.run_retention_cycle()
        assert summary["failed_steps"] == ["telemetry"]
        assert "purge_expired_archives" in [c[0] for c in wiring.calls]
        assert "orphan_attachments" in summary and "request_images" in summary

    async def test_a_clean_cycle_reports_no_failed_steps(self, wiring):
        wiring.config = _config(tier1_days=90, tier2_days=730)
        with _pinned_modules(_wiring_modules(True)):
            summary = await retention.run_retention_cycle()
        assert "failed_steps" not in summary
        assert [e for (_, e, _) in _LOG.records if e == "retention_step_failed"] == []


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
        assert summary["rollup"] == wiring.rollup_result
        assert "rollup_incomplete" not in summary

    async def test_rollup_catches_up_in_one_uncapped_pass(self, wiring):
        # A capped pass covers 14 days; the purge then deleted everything
        # past that frontier unaveraged, with the archive healthy.
        wiring.config = _config(tier1_days=90, tier2_days=365)
        with _pinned_modules(_wiring_modules(True)):
            await retention.run_purge("telemetry", 7)
        assert wiring.rollup_kw == [{"tier2_days": 365, "max_span": None}]

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
        assert summary["rollup_incomplete"] == sorted(_TABLES)
        assert "purge_telemetry_rollup_failed" in _LOG.events("warning")
        assert "purge_telemetry_rollup_incomplete" in _LOG.events("warning")

    async def test_tables_whose_rollup_failed_are_named(self, wiring):
        wiring.config = _config(tier1_days=90, tier2_days=365)
        wiring.rollup_result = dict(wiring.rollup_result)
        wiring.rollup_result["gpu_device_telemetry"] = {
            "rolled_up": 0, "deleted": 0, "error": "OperationalError: gone",
        }
        with _pinned_modules(_wiring_modules(True)):
            summary = await retention.run_purge("telemetry", 30)
        # The purge still runs; the summary and a warning say which table's
        # readings went without averages.
        assert summary["result"] == {"deleted": True}
        assert summary["rollup_incomplete"] == ["gpu_device_telemetry"]
        (record,) = [
            k for (lvl, e, k) in _LOG.records
            if e == "purge_telemetry_rollup_incomplete"
        ]
        assert record["tables"] == ["gpu_device_telemetry"]

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
# Raw telemetry has exactly one deleter: the retention cycle
# ===================================================================

# Age-based deletes of the raw telemetry tables (filtered on ``timestamp``),
# matched as whole identifiers: the archive's archived_*_telemetry_5m tables
# may be purged, and delete_backend may drop a removed backend's own rows.
_RAW_DELETE_RE = re.compile(
    r"DELETE\s+FROM\s+`?(node_telemetry|gpu_device_telemetry|backend_telemetry)`?\s+WHERE\s+`?timestamp`?"
    r"|\bdelete\(\s*(NodeTelemetry|GPUDeviceTelemetry|BackendTelemetry)\s*\)\s*\.where\(\s*\2\.timestamp\b",
    re.IGNORECASE,
)


class TestSingleTelemetryDeleter:
    """The backend registry used to run its own hourly loop deleting raw
    telemetry after TELEMETRY_RETENTION_DAYS (default 30, never passed through
    docker-compose), which silently overrode Tier 1: setting Tier 1 to 90 days
    on Admin -> Retention kept nothing past 30.  Tier 1 is now the only control."""

    def test_registry_runs_no_telemetry_cleanup(self):
        tree = ast.parse(_REGISTRY_PATH.read_text())
        (cls,) = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BackendRegistry"]
        methods = {n.name for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        assert "_telemetry_cleanup_loop" not in methods
        assert "_cleanup_old_telemetry" not in methods
        src = _REGISTRY_PATH.read_text()
        assert "telemetry_retention_days" not in src
        assert "telemetry_cleanup_interval" not in src

    def test_settings_no_longer_define_the_registry_knobs(self):
        tree = ast.parse((_APP / "settings.py").read_text())
        fields = {
            t.target.id
            for n in ast.walk(tree) if isinstance(n, ast.ClassDef)
            for t in n.body if isinstance(t, ast.AnnAssign) and isinstance(t.target, ast.Name)
        }
        assert "telemetry_retention_days" not in fields
        assert "telemetry_cleanup_interval" not in fields

    def test_only_the_retention_service_deletes_raw_telemetry(self):
        offenders = []
        for path in sorted(_APP.rglob("*.py")):
            rel = path.relative_to(_APP).as_posix()
            if rel.startswith("tests/") or rel == "services/retention.py":
                continue
            for m in _RAW_DELETE_RE.finditer(path.read_text()):
                offenders.append(f"{rel}: {m.group(0)}")
        assert offenders == []

    def test_the_guard_pattern_catches_both_delete_styles(self):
        # Mutation check for the scan above: it must see raw and ORM deletes,
        # and must not trip on the archive's rollup tables.
        assert _RAW_DELETE_RE.search("DELETE FROM node_telemetry WHERE timestamp < :c")
        assert _RAW_DELETE_RE.search(
            "await db.execute(delete(GPUDeviceTelemetry).where(GPUDeviceTelemetry.timestamp < cutoff))"
        )
        assert not _RAW_DELETE_RE.search("DELETE FROM archived_node_telemetry_5m WHERE bucket_start < :c")
        assert not _RAW_DELETE_RE.search("delete(ArchivedNodeTelemetry5m).where(x)")
        # delete_backend's cascade of the removed backend's own rows is not retention
        assert not _RAW_DELETE_RE.search(
            "delete(BackendTelemetry).where(BackendTelemetry.backend_id == backend_id)"
        )

    def test_raw_telemetry_deleters_have_pinned_callers(self):
        # Pins the callers, not the SQL text: a new deleter that went through
        # the retention service's own functions (as the removed registry loop
        # last did) would pass the text scan above but fails here.
        names = {"cleanup_expired_telemetry", "archive_expired_telemetry", "_delete_telemetry_before"}
        for path in sorted(_APP.rglob("*.py")):
            rel = path.relative_to(_APP).as_posix()
            if rel.startswith("tests/") or rel == "services/retention.py":
                continue
            src = path.read_text()
            assert not [n for n in names if n in src], rel

        def direct_refs(fn):
            out, stack = set(), list(fn.body)
            while stack:
                n = stack.pop()
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                    continue
                if isinstance(n, ast.Name) and n.id in names:
                    out.add(n.id)
                stack.extend(ast.iter_child_nodes(n))
            return out

        tree = ast.parse(_RETENTION_PATH.read_text())
        callers = {
            fn.name: direct_refs(fn)
            for fn in ast.walk(tree)
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and direct_refs(fn)
        }
        assert callers == {
            # the scheduled cycle's telemetry step (nested in run_retention_cycle)
            "_telemetry": {"archive_expired_telemetry", "cleanup_expired_telemetry"},
            # Admin -> Retention -> Purge
            "run_purge": {"archive_expired_telemetry", "cleanup_expired_telemetry"},
            "archive_expired_telemetry": {"_delete_telemetry_before"},
        }

    def test_delete_node_bulk_deletes_telemetry_before_the_orm_cascade(self):
        # The ORM cascade loaded every reading of the node's GPUs into memory
        # just to delete it; with a 90-day Tier 1 that is millions of rows.
        crud_src = (_APP / "db" / "crud.py").read_text()
        fn = crud_src[crud_src.index("async def delete_node("):]
        fn = fn[: fn.index("\nasync def ", 1)]
        bulk = fn.index("delete(model).where(condition).with_dialect_options(mysql_limit=batch)")
        assert "GPUDeviceTelemetry.gpu_device_id.in_(device_ids)" in fn
        assert "(NodeTelemetry, NodeTelemetry.node_id == node_id)" in fn
        assert bulk < fn.index("await db.delete(device)") < fn.index("await db.delete(node)")
        loop = fn[bulk: fn.index("await db.delete(device)")]
        assert "await db.commit()" in loop and "if deleted < batch:" in loop

    def test_retention_service_keeps_both_raw_delete_paths(self):
        # With averages kept: rollup, then a delete capped at the frontier.
        # Without: the plain batched delete.  Nothing else.
        assert callable(retention.archive_expired_telemetry)
        assert callable(retention.cleanup_expired_telemetry)
        assert not hasattr(retention, "delete_raw_telemetry_older_than")
        assert not hasattr(retention, "_delete_rolled_up_telemetry")


# ===================================================================
# purge_expired_archives / get_archive_stats
# ===================================================================


class _PurgeArchiveDb:
    """Records compiled DELETEs; rollup deletes report one full batch first."""

    def __init__(self):
        self.deletes = []
        self.events = []
        self._full_batch_sent = set()

    async def execute(self, stmt, params=None):
        if isinstance(stmt, Select):
            return types.SimpleNamespace(all=lambda: [])
        assert isinstance(stmt, Delete)
        compiled = stmt.compile(dialect=mysql.dialect())
        table = stmt.table.name
        self.deletes.append((table, str(compiled), dict(compiled.params)))
        self.events.append(("delete", table))
        rowcount = 0
        if table.endswith("_5m") and table not in self._full_batch_sent:
            self._full_batch_sent.add(table)
            rowcount = retention._ROLLUP_PURGE_BATCH
        return types.SimpleNamespace(rowcount=rowcount)

    async def flush(self):
        pass

    async def commit(self):
        self.events.append(("commit",))


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

    async def test_a_missing_rollup_table_does_not_stop_the_others(self):
        # A *_5m table that create_all never made (or that errors) used to
        # abort every purge - and, before steps were isolated, every cycle.
        class _MissingGpuTable(_PurgeArchiveDb):
            async def execute(self, stmt, params=None):
                if isinstance(stmt, Delete) and stmt.table.name == "archived_gpu_device_telemetry_5m":
                    raise RuntimeError("Table 'archived_gpu_device_telemetry_5m' doesn't exist")
                return await super().execute(stmt, params)

            async def rollback(self):
                self.events.append(("rollback",))

        db = _MissingGpuTable()
        config = dict(retention._DEFAULTS)
        config["retention.telemetry.tier2_days"] = 365
        with _pinned_modules({"backend.app.db.archive_models": archive_models}):
            counts = await retention.purge_expired_archives(db, config)
        assert counts["gpu_device_telemetry_5m"]["error"].startswith("RuntimeError")
        assert counts["node_telemetry_5m"] == retention._ROLLUP_PURGE_BATCH
        assert counts["backend_telemetry_5m"] == retention._ROLLUP_PURGE_BATCH
        assert ("rollback",) in db.events
        (record,) = [k for (lvl, e, k) in _LOG.records if e == "retention_rollup_purge_failed"]
        assert record["table"] == "gpu_device_telemetry_5m"

    async def test_every_rollup_batch_is_committed_before_the_next(self):
        db, _ = await self.purge(365)
        rollup_deletes = [
            i for i, e in enumerate(db.events)
            if e[0] == "delete" and e[1].endswith("_5m")
        ]
        assert len(rollup_deletes) == 6
        for i in rollup_deletes:
            assert db.events[i + 1] == ("commit",)

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


def _render_policies(archive_configured, tab="policies"):
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
        tab=tab, is_read_only=False, retention_running=False,
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


def _text(html_fragment):
    """Visible text of an HTML fragment, whitespace collapsed."""
    return " ".join(re.sub(r"<[^>]+>", " ", html_fragment).split())


def _element_text(html, element_id):
    match = re.search(rf'id="{element_id}">(.*?)</div>', html, re.S)
    assert match, element_id
    return _text(match.group(1))


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
        assert "TELEMETRY_RETENTION_DAYS" not in html

    def test_tier1_help_says_it_is_the_only_raw_delete(self):
        # Tier 1 used to be overridden by the registry's own 30-day loop; it
        # is now the only setting that deletes raw readings, and says so.
        help_text = _element_text(_render_policies(True)[0], "retention_telemetry_tier1_help")
        assert "deleted only after its average is in the archive" in help_text
        assert "This is the only setting that deletes raw readings." in help_text
        assert "TELEMETRY_RETENTION_DAYS" not in help_text

    def test_tier2_zero_is_described_as_keeping_archived_averages(self):
        # Tier 2 = 0 skips the rollup purge, so averages already archived
        # stay.  The help used to promise "0 = keep no averages".
        help_text = _element_text(_render_policies(True)[0], "retention_telemetry_tier2_help")
        assert "keep no averages" not in help_text
        assert "0 = write no new averages" in help_text
        assert "averages already in the archive are kept, not purged" in help_text

    def test_purge_tab_says_the_telemetry_rollup_is_best_effort(self):
        html, _ = _render_policies(True, tab="purge")
        intro = _text(html.split("Manual Purge", 1)[1].split("This cannot be undone", 1)[0])
        # The rollup before a manual purge is not a guarantee: the purge
        # deletes the readings either way.
        assert "brought up to date first, as far as the archive allows" in intro
        assert "the readings are deleted either way" in intro
        assert "are deleted without averages" in intro

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
    DOCS = (_REPO / "docs" / "index.md").read_text()

    def test_docs_describe_rollups(self):
        for table in ("archived_node_telemetry_5m", "archived_gpu_device_telemetry_5m",
                      "archived_backend_telemetry_5m"):
            assert table in self.DOCS
        assert "5-minute averages" in self.DOCS

    def test_public_documentation_page_matches(self):
        page = " ".join((_TEMPLATES / "public" / "documentation.html").read_text().split())
        assert "TELEMETRY_RETENTION_DAYS" not in page
        assert "TELEMETRY_CLEANUP_INTERVAL" not in page
        assert "Not archived (deleted only)" not in page
        assert "0 = telemetry is not archived" not in page
        assert "5-minute averages in <code>archived_*_telemetry_5m</code>" in page
        assert "The only setting that deletes raw telemetry" in page
        assert "Each category runs on its own" in page

    def test_docs_describe_isolation_and_entity_deletes(self):
        docs = " ".join(self.DOCS.split())
        assert "listed in the cycle summary's `failed_steps`" in docs
        assert "readings not yet averaged (up to about the last hour) go with it" in docs

    def test_docs_match_the_review_fixes(self):
        docs = " ".join(self.DOCS.split())
        # The retention cycle is the only deleter of raw telemetry; the
        # registry's TELEMETRY_RETENTION_DAYS loop and its settings are gone.
        assert "by the retention cycle (the only deleter of raw telemetry)" in docs
        assert "were removed in this release and are ignored if set" in docs
        assert "| `TELEMETRY_RETENTION_DAYS`" not in docs
        # Backfill never reaches past tier 2.
        assert "starting no further back than Tier 2" in docs
        # Tier 2 = 0 keeps what is archived; the purge rollup is best effort.
        assert "no averages are kept" not in docs
        assert "does not purge the averages already in the archive" in docs
        assert "That step is best effort" in docs
