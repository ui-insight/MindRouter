############################################################
#
# mindrouter - unit tests for backend status ownership and removal
#
# 2026-09-30: backend 76 was DISABLED through the admin API, its server
# restarted, and the gateway routed 65 requests to it minutes later. The
# health poller had written UNHEALTHY over DISABLED while the server was
# down, so its healthy-branch guard (which only spares DISABLED/DRAINING)
# no longer applied and it wrote HEALTHY on return. Every status writer in
# the health/circuit machinery now consults health_status_transition /
# ADMIN_SET_STATUSES. Also: delete_backend left video_shots.backend_id
# pointing at the row, which would make removing a video backend fail.
#
############################################################

"""Unit tests for backend status transitions and delete_backend."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.app.core.telemetry import registry as reg
from backend.app.db.models import BackendStatus

D, R, H, U, K = (BackendStatus.DISABLED, BackendStatus.DRAINING, BackendStatus.HEALTHY,
                 BackendStatus.UNHEALTHY, BackendStatus.UNKNOWN)


class TestTransitionRule:
    @pytest.mark.parametrize("current", [D, R])
    def test_admin_set_statuses_are_never_touched(self, current):
        assert reg.health_status_transition(current, True, 0, 3) is None
        assert reg.health_status_transition(current, False, 99, 3) is None

    @pytest.mark.parametrize("current", [H, U, K, None])
    def test_healthy_check_marks_healthy(self, current):
        assert reg.health_status_transition(current, True, 0, 3) is H

    def test_failures_below_threshold_leave_status(self):
        assert reg.health_status_transition(H, False, 2, 3) is None

    def test_failures_at_threshold_mark_unhealthy(self):
        assert reg.health_status_transition(H, False, 3, 3) is U
        assert reg.health_status_transition(K, False, 5, 3) is U

    def test_admin_set_constant(self):
        assert set(reg.ADMIN_SET_STATUSES) == {D, R}


def _registry_with(status):
    from types import SimpleNamespace
    r = reg.BackendRegistry.__new__(reg.BackendRegistry)
    r._circuit_breakers = {}
    r._fast_poll_backends = {}
    # Real numbers: the breaker does datetime arithmetic with these.
    r._settings = SimpleNamespace(
        backend_circuit_breaker_threshold=1, backend_circuit_breaker_recovery_seconds=30,
        backend_adaptive_poll_fast_duration=60, backend_unhealthy_threshold=3,
    )
    backend = MagicMock(status=status)
    crud = MagicMock(get_backend_by_id=AsyncMock(return_value=backend), update_backend_status=AsyncMock(), update_backend_circuit_breaker=AsyncMock())
    class _Ctx:
        async def __aenter__(self): return MagicMock()
        async def __aexit__(self, *a): return False
    return r, crud, _Ctx


class TestLiveOutcomeWriters:
    @pytest.mark.parametrize("status, expect_healthy_write", [(D, False), (R, False), (U, True)])
    def test_half_open_recovery_respects_admin_status(self, status, expect_healthy_write):
        r, crud, ctx = _registry_with(status)
        cb = SimpleNamespace(is_half_open=True, is_open=False, live_failure_count=3, circuit_open_until=None, last_failure_time=None)
        r._circuit_breakers[7] = cb
        with patch.object(reg, "crud", crud), patch.object(reg, "get_async_db_context", ctx):
            asyncio.run(r.report_live_success(7))
        writes = [c.kwargs.get("status") for c in crud.update_backend_status.await_args_list]
        assert (H in writes) is expect_healthy_write
        assert cb.live_failure_count == 0  # the breaker itself always resets

    @pytest.mark.parametrize("status, expect_unhealthy_write", [(D, False), (R, False), (H, True)])
    def test_circuit_open_respects_admin_status(self, status, expect_unhealthy_write):
        r, crud, ctx = _registry_with(status)
        cb = SimpleNamespace(live_failure_count=0, is_open=False, is_half_open=False, circuit_open_until=None, last_failure_time=None)
        r._circuit_breakers[7] = cb
        with patch.object(reg, "crud", crud), patch.object(reg, "get_async_db_context", ctx):
            asyncio.run(r.report_live_failure(7))
        writes = [c.kwargs.get("status") for c in crud.update_backend_status.await_args_list]
        assert (U in writes) is expect_unhealthy_write


class TestDeleteBackend:
    def test_nulls_request_and_video_shot_references_before_deleting(self):
        from backend.app.db import crud as crud_mod
        executed = []
        backend = MagicMock()
        async def execute(stmt):
            executed.append(str(stmt.compile(compile_kwargs={"literal_binds": True})).lower())
            return MagicMock(scalar_one_or_none=lambda: backend)
        db = MagicMock(execute=execute, delete=AsyncMock(), flush=AsyncMock())
        assert asyncio.run(crud_mod.delete_backend(db, 76)) is True
        updates = [s for s in executed if s.startswith("update")]
        assert any("update requests set backend_id=null" in s for s in updates)
        assert any("update video_shots set backend_id=null" in s for s in updates)
        deletes = [s for s in executed if s.startswith("delete")]
        assert {t for t in ("backend_telemetry", "models", "scheduler_decisions") if any(t in s for s in deletes)} == {"backend_telemetry", "models", "scheduler_decisions"}
        db.delete.assert_awaited_once_with(backend)
