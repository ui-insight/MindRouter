"""Tests for the /healthz and /readyz probes.

Regression cover for a silent readiness failure: `readiness_probe` checked the
database with `await db.execute("SELECT 1")`. SQLAlchemy 2.0 rejects a raw
string with ArgumentError, and the bare `except Exception: pass` swallowed it,
so `{"database": false}` was reported on EVERY request from the 2.0 migration
onward — outage or not. Anything monitoring /readyz was therefore useless, and
an 11-minute production outage on 2026-09-18 went unnoticed.

The stub session below enforces the same rule SQLAlchemy does (reject `str`,
accept a TextClause), so these tests FAIL on the pre-fix line and pass on the
fixed one.

backend/app/api/health.py imports the db package chain at module top, so per
the project's import-chain rules the import happens inside a fixture and skips
cleanly when those deps are unavailable in the test env.
"""

import pytest


@pytest.fixture(scope="module")
def health_mod():
    try:
        from backend.app.api import health
    except Exception as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"health module import unavailable: {exc}")
    return health


@pytest.fixture(scope="module")
def sqlalchemy_bits():
    try:
        from sqlalchemy.exc import ArgumentError
        from sqlalchemy.sql.elements import TextClause
    except Exception as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"sqlalchemy unavailable: {exc}")
    return ArgumentError, TextClause


def _session_factory(recorder, ArgumentError, TextClause, fail_with=None):
    """Build a stand-in for AsyncSessionLocal enforcing SQLAlchemy 2.0's rule.

    SQLAlchemy 2.0 raises ArgumentError when execute() is handed a plain
    string; only an Executable (e.g. text("SELECT 1")) is accepted. Mirroring
    that here is what makes these tests catch the regression instead of
    passing against a permissive mock.
    """

    class _StubSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def execute(self, statement, *args, **kwargs):
            recorder.append(statement)
            if fail_with is not None:
                raise fail_with
            if isinstance(statement, str):
                raise ArgumentError(
                    "Textual SQL expression should be explicitly declared as text()"
                )
            assert isinstance(statement, TextClause)
            return object()

    return lambda: _StubSession()


class _StubRegistry:
    def __init__(self, backends):
        self._backends = backends

    async def get_healthy_backends(self):
        return self._backends


@pytest.mark.asyncio
async def test_readyz_database_check_passes_with_healthy_db(
    health_mod, sqlalchemy_bits, monkeypatch
):
    """The database check must report True when the database answers.

    Fails on the pre-fix code: the raw string raised ArgumentError, which the
    bare except swallowed into database=False.
    """
    ArgumentError, TextClause = sqlalchemy_bits
    seen = []
    monkeypatch.setattr(
        health_mod, "AsyncSessionLocal", _session_factory(seen, ArgumentError, TextClause)
    )
    monkeypatch.setattr(health_mod, "get_registry", lambda: _StubRegistry([object()]))

    result = await health_mod.readiness_probe()

    assert result["checks"]["database"] is True
    assert result["checks"]["backends"] is True
    assert result["status"] == "ready"
    assert len(seen) == 1
    assert isinstance(seen[0], TextClause), "statement must be wrapped in text()"


@pytest.mark.asyncio
async def test_readyz_reports_not_ready_when_database_fails(
    health_mod, sqlalchemy_bits, monkeypatch
):
    """A genuinely broken database must still report False.

    Guards against 'fixing' the check by hardcoding True.
    """
    ArgumentError, TextClause = sqlalchemy_bits
    seen = []
    monkeypatch.setattr(
        health_mod,
        "AsyncSessionLocal",
        _session_factory(seen, ArgumentError, TextClause, fail_with=OSError("db down")),
    )
    monkeypatch.setattr(health_mod, "get_registry", lambda: _StubRegistry([object()]))

    result = await health_mod.readiness_probe()

    assert result["checks"]["database"] is False
    assert result["status"] == "not_ready"


@pytest.mark.asyncio
async def test_readyz_logs_why_the_database_check_failed(
    health_mod, sqlalchemy_bits, monkeypatch
):
    """A failing check must say why.

    The original bare `except Exception: pass` is exactly what hid this bug;
    a silent failure mode on a readiness probe is not acceptable.
    """
    ArgumentError, TextClause = sqlalchemy_bits
    logged = []

    class _StubLogger:
        def warning(self, event, **kw):
            logged.append((event, kw))

        def __getattr__(self, _name):  # info/debug/error are no-ops here
            return lambda *a, **k: None

    monkeypatch.setattr(
        health_mod,
        "AsyncSessionLocal",
        _session_factory([], ArgumentError, TextClause, fail_with=OSError("db down")),
    )
    monkeypatch.setattr(health_mod, "get_registry", lambda: _StubRegistry([object()]))
    monkeypatch.setattr(health_mod, "logger", _StubLogger())

    await health_mod.readiness_probe()

    assert logged, "a failed database check must be logged, not swallowed"
    assert any("db down" in str(kw) for _event, kw in logged)


@pytest.mark.asyncio
async def test_readyz_not_ready_when_no_healthy_backends(
    health_mod, sqlalchemy_bits, monkeypatch
):
    """Database up but zero healthy backends is still not ready.

    This is the state the 2026-09-18 DNS outage produced fleet-wide.
    """
    ArgumentError, TextClause = sqlalchemy_bits
    monkeypatch.setattr(
        health_mod, "AsyncSessionLocal", _session_factory([], ArgumentError, TextClause)
    )
    monkeypatch.setattr(health_mod, "get_registry", lambda: _StubRegistry([]))

    result = await health_mod.readiness_probe()

    assert result["checks"]["database"] is True
    assert result["checks"]["backends"] is False
    assert result["status"] == "not_ready"


@pytest.mark.asyncio
async def test_healthz_is_independent_of_database_and_backends(health_mod):
    """Liveness must not depend on the database or the registry.

    A liveness probe that fails during a dependency outage gets the container
    killed exactly when it is still able to serve.
    """
    result = await health_mod.liveness_probe()

    assert result["status"] == "alive"
    assert "timestamp" in result
