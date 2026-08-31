############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_api_key_sha256.py: SHA-256 fast-path API-key verification
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Unit tests for SHA-256 API-key verification with verify-and-upgrade.

Covers:
- generate_api_key 4-tuple: digest matches key, Argon2 hash kept, prefix format
- Fast path: sha256 lookup hit returns row, no prefix lookup, no Argon2 work
- Belt-and-braces digest mismatch on fast path rejected
- Fallback: prefix + Argon2 verify, key_sha256 backfilled via a dedicated
  session (survives a subsequent 401 rollback of the request session)
- Already-upgraded rows short-circuit the fallback without Argon2 work
- Argon2 semaphore queue is bounded (timeout sheds load instead of
  pinning pooled DB connections)
- Wrong key on fallback: rejected, no backfill
- Garbage keys rejected without any DB lookup / Argon2 work
- api_key_rejection_reason post-verify gate (status/expiry/user checks)
  and the source contract that every verify_api_key caller applies it
- Argon2 fallback bounded by Semaphore(4) via asyncio.to_thread
- Migration 069 / crud / models / caller source contracts

api_keys.py is spec-loaded with backend.app.db* pre-mocked to avoid the
package import chain — see MEMORY.md "Import Chain Gotcha".
"""

import asyncio
import hashlib
import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

_APP_DIR = Path(__file__).resolve().parents[2]
_SECURITY_DIR = _APP_DIR / "security"
_DB_DIR = _APP_DIR / "db"
_MIGRATION = _DB_DIR / "migrations" / "versions" / "20260801_000000_069_add_api_key_sha256.py"


def _load_api_keys():
    """Spec-load api_keys.py, pre-mocking the backend.app.db package chain."""
    saved = {}
    for name in [
        "backend",
        "backend.app",
        "backend.app.db",
        "backend.app.db.crud",
        "backend.app.db.models",
        "backend.app.db.session",
    ]:
        saved[name] = sys.modules.get(name)
        sys.modules[name] = MagicMock()

    spec = importlib.util.spec_from_file_location(
        "mr2_api_keys_under_test", _SECURITY_DIR / "api_keys.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, saved


@pytest.fixture(scope="module")
def api_keys():
    mod, saved = _load_api_keys()
    yield mod
    for name, original in saved.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


@pytest.fixture
def crud_mock(api_keys):
    """Fresh crud mock per test so call assertions don't leak between tests."""
    crud = MagicMock()
    crud.get_api_key_by_sha256 = AsyncMock(return_value=None)
    crud.get_api_key_by_prefix = AsyncMock(return_value=None)
    original = api_keys.crud
    api_keys.crud = crud
    yield crud
    api_keys.crud = original


@pytest.fixture
def backfill_env(api_keys, monkeypatch):
    """Stub the dedicated-session backfill machinery.

    The real path opens its own get_async_db_context session, executes a
    single-column UPDATE, and reflects the value via set_committed_value
    (an ORM-only call that would reject our SimpleNamespace rows).
    """
    session = SimpleNamespace(execute=AsyncMock())

    class _CM:
        async def __aenter__(self):
            return session

        async def __aexit__(self, *args):
            return False

    monkeypatch.setattr(api_keys, "get_async_db_context", lambda: _CM())
    stmt = MagicMock()
    stmt.where.return_value = stmt
    stmt.values.return_value = stmt
    update_mock = MagicMock(return_value=stmt)
    monkeypatch.setattr(api_keys, "update", update_mock)
    committed = []

    def _fake_set_committed(obj, key, value):
        committed.append((obj, key, value))
        setattr(obj, key, value)

    monkeypatch.setattr(api_keys, "set_committed_value", _fake_set_committed)
    return SimpleNamespace(
        session=session, stmt=stmt, update=update_mock, committed=committed
    )


def _digest(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def _db_key(**kwargs) -> SimpleNamespace:
    defaults = dict(id=1, key_hash="", key_sha256=None, status="active")
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


# ===================================================================
# generate_api_key
# ===================================================================

class TestGenerate:
    def test_returns_four_tuple_with_matching_digests(self, api_keys):
        full_key, key_hash, key_prefix, key_sha256 = api_keys.generate_api_key()
        assert full_key.startswith("mr2_")
        assert key_sha256 == _digest(full_key)
        assert len(key_sha256) == 64
        # Argon2 hash still written (rollback safety)
        assert key_hash.startswith("$argon2")
        assert api_keys._verify_key_hash(full_key, key_hash)
        assert key_prefix == full_key[:12]

    def test_entropy_invariant_pinned(self, api_keys):
        # The plain-SHA-256 column is only safe with token_urlsafe(32) entropy
        src = (_SECURITY_DIR / "api_keys.py").read_text()
        assert "secrets.token_urlsafe(32)" in src
        assert "SECURITY INVARIANT" in src


# ===================================================================
# verify_api_key — fast path
# ===================================================================

class TestFastPath:
    async def test_hit_skips_prefix_lookup_and_argon2(self, api_keys, crud_mock, monkeypatch):
        full_key, key_hash, _, key_sha256 = api_keys.generate_api_key()
        row = _db_key(key_hash=key_hash, key_sha256=key_sha256)
        crud_mock.get_api_key_by_sha256 = AsyncMock(return_value=row)
        # Any Argon2 verify on the fast path is a failure
        monkeypatch.setattr(
            api_keys, "_verify_key_hash",
            MagicMock(side_effect=AssertionError("Argon2 ran on fast path")),
        )

        result = await api_keys.verify_api_key(MagicMock(), full_key)

        assert result is row
        crud_mock.get_api_key_by_sha256.assert_awaited_once_with(
            crud_mock.get_api_key_by_sha256.await_args.args[0], key_sha256
        )
        crud_mock.get_api_key_by_prefix.assert_not_awaited()

    async def test_stored_digest_mismatch_rejected(self, api_keys, crud_mock):
        full_key, key_hash, _, _ = api_keys.generate_api_key()
        # Belt-and-braces: a row whose stored digest doesn't match is rejected
        row = _db_key(key_hash=key_hash, key_sha256="0" * 64)
        crud_mock.get_api_key_by_sha256 = AsyncMock(return_value=row)

        assert await api_keys.verify_api_key(MagicMock(), full_key) is None


# ===================================================================
# verify_api_key — Argon2 fallback + backfill upgrade
# ===================================================================

class TestFallback:
    async def test_argon2_verify_backfills_sha256(self, api_keys, crud_mock, backfill_env):
        full_key, key_hash, key_prefix, key_sha256 = api_keys.generate_api_key()
        row = _db_key(key_hash=key_hash, key_sha256=None)
        crud_mock.get_api_key_by_prefix = AsyncMock(return_value=row)

        result = await api_keys.verify_api_key(MagicMock(), full_key)

        assert result is row
        # Verify-and-upgrade: single-column UPDATE issued in the dedicated
        # session, and the persisted value reflected onto the loaded row
        backfill_env.session.execute.assert_awaited_once_with(backfill_env.stmt)
        backfill_env.stmt.values.assert_called_once_with(key_sha256=key_sha256)
        assert backfill_env.committed == [(row, "key_sha256", key_sha256)]
        assert row.key_sha256 == key_sha256
        crud_mock.get_api_key_by_prefix.assert_awaited_once()
        assert crud_mock.get_api_key_by_prefix.await_args.args[1] == key_prefix

    async def test_backfill_skips_request_session(self, api_keys, crud_mock, backfill_env):
        # The backfill must survive a subsequent 401 rollback of the request
        # session (revoked/expired key), so it must never stage state there.
        full_key, key_hash, _, _ = api_keys.generate_api_key()
        row = _db_key(key_hash=key_hash, key_sha256=None)
        crud_mock.get_api_key_by_prefix = AsyncMock(return_value=row)
        request_db = MagicMock()

        assert await api_keys.verify_api_key(request_db, full_key) is row

        request_db.commit.assert_not_called()
        request_db.execute.assert_not_called()
        backfill_env.session.execute.assert_awaited_once()

    async def test_backfill_failure_never_fails_verify(self, api_keys, crud_mock, backfill_env):
        # The backfill is an optimization — a failed write must not 401 a
        # key that just passed Argon2 verification.
        full_key, key_hash, _, _ = api_keys.generate_api_key()
        row = _db_key(key_hash=key_hash, key_sha256=None)
        crud_mock.get_api_key_by_prefix = AsyncMock(return_value=row)
        backfill_env.session.execute = AsyncMock(side_effect=RuntimeError("db down"))

        assert await api_keys.verify_api_key(MagicMock(), full_key) is row
        assert row.key_sha256 is None  # not reflected — next request retries

    async def test_already_upgraded_row_short_circuits(self, api_keys, crud_mock, monkeypatch):
        # The unique sha256 lookup already missed, so a row with a populated
        # key_sha256 provably doesn't match — no Argon2 spend allowed.
        full_key, key_hash, _, _ = api_keys.generate_api_key()
        row = _db_key(key_hash=key_hash, key_sha256="f" * 64)
        crud_mock.get_api_key_by_prefix = AsyncMock(return_value=row)
        monkeypatch.setattr(
            api_keys, "_verify_key_hash",
            MagicMock(side_effect=AssertionError("Argon2 ran for an upgraded row")),
        )

        assert await api_keys.verify_api_key(MagicMock(), full_key) is None

    async def test_semaphore_queue_timeout_sheds_load(self, api_keys, crud_mock, monkeypatch):
        # A saturated Argon2 queue must fast-reject instead of pinning the
        # request's pooled DB connection behind an unbounded wait.
        full_key, key_hash, _, _ = api_keys.generate_api_key()
        row = _db_key(key_hash=key_hash, key_sha256=None)
        crud_mock.get_api_key_by_prefix = AsyncMock(return_value=row)
        monkeypatch.setattr(api_keys, "_ARGON2_QUEUE_TIMEOUT_SECONDS", 0.01)
        sem = api_keys._argon2_verify_semaphore
        acquired = 0
        while not sem.locked():
            await sem.acquire()
            acquired += 1
        try:
            assert await api_keys.verify_api_key(MagicMock(), full_key) is None
        finally:
            for _ in range(acquired):
                sem.release()

    async def test_wrong_key_rejected_no_backfill(self, api_keys, crud_mock, backfill_env):
        _, other_hash, _, _ = api_keys.generate_api_key()
        row = _db_key(key_hash=other_hash, key_sha256=None)
        crud_mock.get_api_key_by_prefix = AsyncMock(return_value=row)

        wrong_key = "mr2_" + "A" * 43
        assert await api_keys.verify_api_key(MagicMock(), wrong_key) is None
        assert row.key_sha256 is None
        backfill_env.session.execute.assert_not_awaited()

    def test_argon2_bounded_by_semaphore_and_to_thread(self, api_keys):
        # 64 MiB per verify — the cap prevents RSS blowup under key floods
        assert isinstance(api_keys._argon2_verify_semaphore, asyncio.Semaphore)
        assert api_keys._argon2_verify_semaphore._value == 4
        src = (_SECURITY_DIR / "api_keys.py").read_text()
        assert "asyncio.to_thread(_verify_key_hash" in src


# ===================================================================
# verify_api_key — garbage keys
# ===================================================================

class TestGarbageKeys:
    async def test_wrong_prefix_rejected_without_db(self, api_keys, crud_mock):
        assert await api_keys.verify_api_key(MagicMock(), "sk-not-ours-at-all") is None
        crud_mock.get_api_key_by_sha256.assert_not_awaited()
        crud_mock.get_api_key_by_prefix.assert_not_awaited()

    async def test_unknown_mr2_key_rejected_without_argon2(self, api_keys, crud_mock, monkeypatch):
        monkeypatch.setattr(
            api_keys, "_verify_key_hash",
            MagicMock(side_effect=AssertionError("Argon2 ran with no candidate row")),
        )
        assert await api_keys.verify_api_key(MagicMock(), "mr2_" + "x" * 43) is None
        crud_mock.get_api_key_by_sha256.assert_awaited_once()
        crud_mock.get_api_key_by_prefix.assert_awaited_once()


# ===================================================================
# Revocation/expiry/user-active enforcement: the shared post-verify gate
# ===================================================================

class TestCallerEnforcement:
    async def test_verify_returns_revoked_row_for_caller_to_reject(self, api_keys, crud_mock):
        # verify_api_key proves possession only — status is the caller's job
        full_key, key_hash, _, key_sha256 = api_keys.generate_api_key()
        row = _db_key(key_hash=key_hash, key_sha256=key_sha256, status="revoked")
        crud_mock.get_api_key_by_sha256 = AsyncMock(return_value=row)
        assert await api_keys.verify_api_key(MagicMock(), full_key) is row

    def test_auth_checks_run_after_verify(self):
        # The API-key authenticator must apply the shared post-verify gate
        # AFTER verify_api_key — fast path included.  This lives in
        # authenticate_credential, which authenticate_request wraps to add the
        # inference-scope check; the gate itself did not move.
        src = (_APP_DIR / "api" / "auth.py").read_text()
        body = src[src.index("async def authenticate_credential"):]
        verify_pos = body.index("await verify_api_key(db, api_key_str)")
        assert body.index("api_key_rejection_reason(api_key)") > verify_pos
        # The gate itself carries the status/expiry/user checks
        gate = (_SECURITY_DIR / "api_keys.py").read_text()
        gate = gate[gate.index("def api_key_rejection_reason"):]
        gate = gate[: gate.index("\nasync def")]
        assert "status != ApiKeyStatus.ACTIVE" in gate
        assert "API key has expired" in gate
        assert "not user or not user.is_active or user.deleted_at" in gate

    def test_all_gap_sites_invoke_rejection_predicate(self):
        # Every verify_api_key caller outside authenticate_request must apply
        # the same gate — the fast path's safety argument depends on it.
        # Both MCP transports share one authenticator, _resolve_auth, so the
        # gate is asserted there — and then that neither transport can reach
        # the tools without going through it.
        mcp = (_APP_DIR / "api" / "mcp_server.py").read_text()
        body = mcp[mcp.index("async def _resolve_auth"):]
        body = body[: body.index("\nclass StreamableHTTPEndpoint")]
        assert (
            body.index("api_key_rejection_reason(api_key)")
            > body.index("await verify_api_key(db, api_key_str)")
        )
        # Legacy SSE transport
        sse = mcp[mcp.index("async def _handle_sse"):]
        assert "_resolve_auth(" in sse[: sse.index("connect_sse")]
        # Streamable HTTP transport: authenticated before the SDK sees the scope
        streamable = mcp[mcp.index("class StreamableHTTPEndpoint"):]
        streamable = streamable[: streamable.index("async def _handle_sse")]
        assert (
            streamable.index("_resolve_auth(")
            < streamable.index("handle_request(scope, receive, send)")
        )
        auth_src = (_APP_DIR / "api" / "auth.py").read_text()
        for fn in ("def require_admin_or_session", "def require_admin_read_or_session"):
            body = auth_src[auth_src.index(fn):]
            assert (
                body.index("api_key_rejection_reason(api_key)")
                > body.index("await _verify(db, api_key_str)")
            )
        dash = (_APP_DIR / "dashboard" / "routes.py").read_text()
        body = dash[dash.index("async def api_tts_voices"):]
        assert (
            body.index("api_key_rejection_reason(api_key)")
            > body.index("await verify_api_key(db, api_key_str)")
        )


class TestRejectionPredicate:
    """Unit tests for the shared api_key_rejection_reason gate."""

    def _row(self, api_keys, **kwargs):
        defaults = dict(
            id=1,
            status=api_keys.ApiKeyStatus.ACTIVE,  # mocked sentinel; != is identity
            is_service=False,
            expires_at=None,
            user=SimpleNamespace(id=2, is_active=True, deleted_at=None),
        )
        defaults.update(kwargs)
        return SimpleNamespace(**defaults)

    def test_active_key_accepted(self, api_keys):
        assert api_keys.api_key_rejection_reason(self._row(api_keys)) is None

    def test_revoked_key_rejected(self, api_keys):
        row = self._row(api_keys, status=SimpleNamespace(value="revoked"))
        assert api_keys.api_key_rejection_reason(row) == "API key is revoked"

    def test_expired_naive_datetime_rejected(self, api_keys):
        # MariaDB returns naive datetimes — the gate must normalize to UTC
        row = self._row(api_keys, expires_at=datetime(2020, 1, 1))
        assert api_keys.api_key_rejection_reason(row) == "API key has expired"

    def test_expired_aware_datetime_rejected(self, api_keys):
        row = self._row(api_keys, expires_at=datetime(2020, 1, 1, tzinfo=timezone.utc))
        assert api_keys.api_key_rejection_reason(row) == "API key has expired"

    def test_service_key_never_expires(self, api_keys):
        row = self._row(api_keys, is_service=True, expires_at=datetime(2020, 1, 1))
        assert api_keys.api_key_rejection_reason(row) is None

    def test_future_expiry_accepted(self, api_keys):
        row = self._row(
            api_keys, expires_at=datetime.now(timezone.utc) + timedelta(days=1)
        )
        assert api_keys.api_key_rejection_reason(row) is None

    def test_inactive_user_rejected(self, api_keys):
        row = self._row(
            api_keys, user=SimpleNamespace(id=2, is_active=False, deleted_at=None)
        )
        assert api_keys.api_key_rejection_reason(row) == "User account is inactive"

    def test_deleted_user_rejected(self, api_keys):
        row = self._row(
            api_keys,
            user=SimpleNamespace(id=2, is_active=True, deleted_at=datetime(2025, 1, 1)),
        )
        assert api_keys.api_key_rejection_reason(row) == "User account is inactive"

    def test_missing_user_rejected(self, api_keys):
        row = self._row(api_keys, user=None)
        assert api_keys.api_key_rejection_reason(row) == "User account is inactive"


# ===================================================================
# Source contracts: migration, crud, models, generate_api_key callers
# ===================================================================

class TestSourceContracts:
    def test_migration_069(self):
        src = _MIGRATION.read_text()
        assert 'revision = "069"' in src
        assert 'down_revision = "068"' in src
        # Type and index name must match what Base.metadata renders for the
        # model (String(64) + unique=True/index=True → ix_ name), so future
        # autogenerate runs don't churn the uniqueness guarantee
        assert '"api_keys", sa.Column("key_sha256", sa.String(64), nullable=True)' in src
        assert '"ix_api_keys_key_sha256"' in src
        assert "uq_api_keys_key_sha256" not in src
        assert "unique=True" in src
        # Working downgrade: index first, then column
        downgrade = src[src.index("def downgrade"):]
        assert 'op.drop_index("ix_api_keys_key_sha256"' in downgrade
        assert 'op.drop_column("api_keys", "key_sha256")' in downgrade

    def test_crud_sha256_lookup_and_create(self):
        src = (_DB_DIR / "crud.py").read_text()
        fn = src[src.index("async def get_api_key_by_sha256"):]
        fn = fn[: fn.index("\n\n\nasync def")]
        assert "selectinload(ApiKey.user).selectinload(User.group)" in fn
        # Unique column: scalar_one_or_none is safe (no MultipleResultsFound)
        assert "scalar_one_or_none" in fn
        assert "ApiKey.key_sha256 == key_sha256" in fn
        create = src[src.index("async def create_api_key"):]
        create = create[: create.index("\n\n\nasync def")]
        assert "key_sha256: Optional[str] = None" in create
        assert "key_sha256=key_sha256" in create

    def test_model_column(self):
        src = (_DB_DIR / "models.py").read_text()
        assert (
            "key_sha256: Mapped[Optional[str]] = mapped_column"
            "(String(64), unique=True, nullable=True, index=True)" in src
        )

    # services/dlp_worker.py was here until 2.9.9, when the DLP scanner stopped
    # minting a key entirely (it dispatches straight to a backend now).  The
    # module must therefore NOT appear in this list — see
    # test_dlp_worker.TestCredentialRemoved.
    @pytest.mark.parametrize(
        "rel_path",
        [
            "dashboard/routes.py",
            "api/admin_api.py",
            "api/apps_api.py",
            "dashboard/apps_routes.py",
        ],
    )
    def test_callers_store_both_columns(self, rel_path):
        src = (_APP_DIR / rel_path).read_text()
        assert "full_key, key_hash, key_prefix, key_sha256 = generate_api_key()" in src
        assert "key_sha256=key_sha256" in src

    def test_every_generate_api_key_caller_is_covered(self):
        """The parametrize list above must not silently drift out of date."""
        callers = {
            str(p.relative_to(_APP_DIR))
            for p in _APP_DIR.rglob("*.py")
            if "tests" not in p.parts and "generate_api_key()" in p.read_text()
        }
        callers.discard("security/api_keys.py")  # the definition itself
        callers.discard("scripts/seed_dev_data.py")
        assert callers == {
            "dashboard/routes.py", "api/admin_api.py", "api/apps_api.py",
            "dashboard/apps_routes.py",
        }, (
            f"generate_api_key() callers changed: {sorted(callers)} — update the "
            "parametrize list above so each new caller is checked."
        )
