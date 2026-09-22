"""Tests for GET /v1/me/limits (backend/app/api/me_api.py).

The endpoint exists so a long-running client (VandalChat's Deep Research: a
few hundred requests over half an hour on the launching user's own key) can
pace itself from the real numbers instead of discovering them through a 429.
Its one promise is PARITY: it must report exactly what
`InferenceService._check_quota` enforces, resolved in the same order — the
key's own `rpm_limit` override, else the quota row; the quota's
`token_budget_override`, else the group budget, with 0 meaning unlimited in
both. A number that disagrees with enforcement is worse than no endpoint,
because the pacer would trust it.

me_api.py imports the db package chain at module top, so per the project's
import-chain rules the import happens inside a module-scoped fixture and the
file skips cleanly when those deps are unavailable.

Only the two crud reads the endpoint makes are stubbed; the budget resolution
itself runs for real through core/quota_budget.py, so a drift there shows up
here too.
"""

import inspect
from datetime import datetime
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def me_mod():
    try:
        from backend.app.api import me_api
    except Exception as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"me_api import unavailable: {exc}")
    return me_api


# --------------------------------------------------------------------------
# Minimal stand-ins: only the attributes the endpoint reads.
# --------------------------------------------------------------------------


def _user(group_budget=100_000, has_group=True, uid=7):
    group = SimpleNamespace(name="faculty", token_budget=group_budget) if has_group else None
    return SimpleNamespace(id=uid, username="alice", group=group)


# budget_period_start is naive on purpose: that is how MariaDB hands it back.
def _quota(rpm=30, used=0, override=None, days=30, start=datetime(2026, 9, 1, 12, 0, 0)):
    return SimpleNamespace(
        rpm_limit=rpm,
        tokens_used=used,
        token_budget_override=override,
        budget_period_days=days,
        budget_period_start=start,
    )


def _key(rpm=None, prefix="mr2_abcdefgh"):
    return SimpleNamespace(rpm_limit=rpm, key_prefix=prefix)


def _wire(monkeypatch, me_mod, quota, calls=None):
    """Replace the endpoint's two crud reads with stubs that record order."""

    async def _reset(db, user_id):
        if calls is not None:
            calls.append(("reset", user_id))
        return quota

    async def _get(db, user_id):
        if calls is not None:
            calls.append(("get", user_id))
        return quota

    monkeypatch.setattr(me_mod.crud, "reset_quota_if_needed", _reset)
    monkeypatch.setattr(me_mod.crud, "get_user_quota", _get)


async def _call(me_mod, user, key):
    return await me_mod.my_limits(db=object(), auth=(user, key))


# --------------------------------------------------------------------------
# RPM: the key's own override wins, else the quota row — as _check_quota does.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_key_rpm_override_beats_the_quota_row(me_mod, monkeypatch):
    _wire(monkeypatch, me_mod, _quota(rpm=30))
    out = await _call(me_mod, _user(), _key(rpm=120))
    assert out.rpm_limit == 120


@pytest.mark.asyncio
@pytest.mark.parametrize("key_rpm", [None, 0])
async def test_quota_rpm_applies_when_the_key_has_no_override(me_mod, monkeypatch, key_rpm):
    """_check_quota tests the key's rpm_limit for truthiness, so 0 falls through too."""
    _wire(monkeypatch, me_mod, _quota(rpm=30))
    out = await _call(me_mod, _user(), _key(rpm=key_rpm))
    assert out.rpm_limit == 30


@pytest.mark.asyncio
async def test_no_quota_row_means_no_rpm_limit_and_the_group_budget(me_mod, monkeypatch):
    _wire(monkeypatch, me_mod, None)
    out = await _call(me_mod, _user(group_budget=100_000), _key())
    assert out.rpm_limit == 0
    assert out.tokens_used == 0
    assert out.token_budget == 100_000
    assert out.tokens_remaining == 100_000
    assert out.budget_period_days == 30
    assert out.period_start is None and out.period_end is None


# --------------------------------------------------------------------------
# Token budget: per-user override, else group; 0 = unlimited in both.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_user_override_beats_the_group_budget(me_mod, monkeypatch):
    """A user at 99,999/100,000 who was granted 250,000 must see the grant."""
    _wire(monkeypatch, me_mod, _quota(override=250_000, used=99_999))
    out = await _call(me_mod, _user(group_budget=100_000), _key())
    assert out.token_budget == 250_000
    assert out.tokens_remaining == 150_001


@pytest.mark.asyncio
async def test_zero_budget_is_reported_as_unlimited(me_mod, monkeypatch):
    """An override of 0 and a group budget of 0 both mean 'no limit'."""
    _wire(monkeypatch, me_mod, _quota(override=0, used=10**9))
    out = await _call(me_mod, _user(group_budget=100_000), _key())
    assert out.token_budget == 0 and out.tokens_remaining is None

    _wire(monkeypatch, me_mod, _quota(used=10**9))
    out = await _call(me_mod, _user(group_budget=0), _key())
    assert out.token_budget == 0 and out.tokens_remaining is None


@pytest.mark.asyncio
async def test_groupless_user_has_no_budget(me_mod, monkeypatch):
    _wire(monkeypatch, me_mod, _quota())
    out = await _call(me_mod, _user(has_group=False), _key())
    assert out.group is None
    assert out.token_budget == 0 and out.tokens_remaining is None


@pytest.mark.asyncio
async def test_tokens_remaining_never_goes_negative(me_mod, monkeypatch):
    _wire(monkeypatch, me_mod, _quota(used=150))
    out = await _call(me_mod, _user(group_budget=100), _key())
    assert out.tokens_used == 150
    assert out.tokens_remaining == 0


# --------------------------------------------------------------------------
# Period handling and the ordering that makes the numbers current.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_period_reset_runs_before_the_quota_is_read(me_mod, monkeypatch):
    """Same order as _check_quota: a lapsed period is rolled, then reported.

    Reading first would report last period's usage against a period that the
    very next inference call resets to zero.
    """
    calls = []
    _wire(monkeypatch, me_mod, _quota(), calls=calls)
    await _call(me_mod, _user(uid=42), _key())
    assert calls == [("reset", 42), ("get", 42)]


@pytest.mark.asyncio
async def test_naive_mariadb_datetimes_are_reported_as_utc(me_mod, monkeypatch):
    """A client parsing an offset-less timestamp as local time would believe
    the period ends hours later than it does."""
    _wire(monkeypatch, me_mod, _quota(days=30, start=datetime(2026, 9, 1, 12, 0, 0)))
    out = await _call(me_mod, _user(), _key())
    assert out.period_start == "2026-09-01T12:00:00+00:00"
    assert out.period_end == "2026-10-01T12:00:00+00:00"


@pytest.mark.asyncio
async def test_identity_fields_describe_the_calling_key(me_mod, monkeypatch):
    _wire(monkeypatch, me_mod, _quota())
    out = await _call(me_mod, _user(uid=7), _key(prefix="mr2_zzzzzzzz"))
    assert (out.user_id, out.username, out.group) == (7, "alice", "faculty")
    assert out.key_prefix == "mr2_zzzzzzzz"


# --------------------------------------------------------------------------
# Wiring guards.
# --------------------------------------------------------------------------


def test_route_is_get_and_gated_by_the_inference_scope(me_mod):
    """authenticate_request is where the `inference` scope is enforced, so an
    app's provisioning credential cannot read a user's limits through this
    door; a per-user app key (inference scope) can read only its own."""
    routes = [r for r in me_mod.router.routes if getattr(r, "path", None) == "/v1/me/limits"]
    assert len(routes) == 1
    assert routes[0].methods == {"GET"}
    dep = inspect.signature(me_mod.my_limits).parameters["auth"].default
    assert dep.dependency is me_mod.authenticate_request


def test_reading_limits_costs_nothing_and_resolves_like_enforcement(me_mod):
    """A pacer polling its own limit must not consume the window it measures,
    and the budget must come from the shared helper (see test_quota_budget's
    anti-recurrence guard, which also lists this module)."""
    src = inspect.getsource(me_mod)
    assert "check_rpm" not in src
    assert "effective_token_budget(user, quota)" in src
