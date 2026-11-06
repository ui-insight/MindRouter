"""Tests for admin user-management hardening (2.9.4).

Covers:
- IDOR fix: self-service key revocation is ownership-scoped and excludes
  service keys (crud.revoke_api_key scoping + dashboard route contract)
- Admin API key revoke/delete endpoints (+ delete refuses on references)
- Admin password reset (API + dashboard route + local-account gating)
- delete_user cascade completeness: EVERY FK to users.id in the schema
  must be handled (deleted or detached) by crud.delete_user
- Manual purge: category allowlist excludes admin_audit_log, server-side
  PURGE confirmation, request-reference detach wired into both request
  deletion paths
- Migration 069 + nullable model columns
- Template affordances (danger zone, reset modal, revoke buttons)

models.py is spec-loaded directly (with backend.app.db.base pre-loaded the
same way) to avoid the backend.app.db package import chain — see MEMORY.md
"Import Chain Gotcha".  Route/crud behavior is checked by source inspection,
matching test_local_user_accounts.py.
"""

import importlib.util
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_APP_DIR = Path(__file__).resolve().parents[2]
_DB_DIR = _APP_DIR / "db"
_API_DIR = _APP_DIR / "api"
_DASHBOARD_DIR = _APP_DIR / "dashboard"
_SERVICES_DIR = _APP_DIR / "services"
_TEMPLATES_DIR = _DASHBOARD_DIR / "templates"
_MIGRATIONS_DIR = _DB_DIR / "migrations" / "versions"

CRUD_SRC = (_DB_DIR / "crud.py").read_text()
ROUTES_SRC = (_DASHBOARD_DIR / "routes.py").read_text()
ADMIN_API_SRC = (_API_DIR / "admin_api.py").read_text()
RETENTION_SRC = (_SERVICES_DIR / "retention.py").read_text()


def _extract_function(source: str, name: str) -> str:
    """Return the source of one top-level (async) def, brace-agnostic:
    from its def line to the next top-level def/class/decorator."""
    pattern = re.compile(
        rf"^(?:async )?def {name}\(.*?(?=^@|^(?:async )?def |^class |\Z)",
        re.DOTALL | re.MULTILINE,
    )
    match = pattern.search(source)
    assert match, f"function {name} not found"
    return match.group(0)


def _load_models():
    """Spec-load base.py then models.py, bypassing backend.app.db.__init__."""
    saved = {}
    for name in ["backend", "backend.app", "backend.app.db"]:
        if name not in sys.modules:
            saved[name] = None
            sys.modules[name] = MagicMock()

    base_spec = importlib.util.spec_from_file_location(
        "backend.app.db.base", _DB_DIR / "base.py"
    )
    base_mod = importlib.util.module_from_spec(base_spec)
    saved_base = sys.modules.get("backend.app.db.base")
    sys.modules["backend.app.db.base"] = base_mod
    base_spec.loader.exec_module(base_mod)

    models_spec = importlib.util.spec_from_file_location(
        "_test_admin_mgmt_models", _DB_DIR / "models.py"
    )
    models_mod = importlib.util.module_from_spec(models_spec)
    try:
        models_spec.loader.exec_module(models_mod)
    finally:
        if saved_base is not None:
            sys.modules["backend.app.db.base"] = saved_base
        else:
            sys.modules.pop("backend.app.db.base", None)
        for name, val in saved.items():
            if val is None:
                sys.modules.pop(name, None)
    return models_mod


@pytest.fixture(scope="module")
def models():
    return _load_models()


# ---------------------------------------------------------------------------
# IDOR fix: self-service revocation is scoped
# ---------------------------------------------------------------------------

class TestRevokeKeyScoping:
    def test_crud_revoke_supports_owner_scoping(self):
        fn = _extract_function(CRUD_SRC, "revoke_api_key")
        assert "owner_user_id" in fn
        assert "ApiKey.user_id == owner_user_id" in fn
        assert "allow_service" in fn
        assert "ApiKey.is_service.is_(False)" in fn

    def test_dashboard_self_service_route_is_scoped(self):
        fn = _extract_function(ROUTES_SRC, "revoke_key")
        assert "owner_user_id=user_id" in fn
        assert "allow_service=False" in fn
        # Failure to match the scope must NOT fall through to a commit
        assert "if not revoked" in fn

    def test_admin_dashboard_revoke_requires_admin(self):
        fn = _extract_function(ROUTES_SRC, "admin_revoke_api_key")
        assert "is_admin" in fn
        assert "log_admin_action" in fn
        # open-redirect guard: the target is allowlisted AND reconstructed —
        # no request-supplied bytes reach the Location header
        assert "_admin_redirect_target(redirect_to)" in fn
        helper = _extract_function(ROUTES_SRC, "_admin_redirect_target")
        assert "re.fullmatch" in helper
        assert r"/admin/users/(\d+)" in helper
        assert "int(m.group(1))" in helper
        assert '"/admin/api-keys"' in helper

    def test_dashboard_admin_helper_rejects_deactivated_admins(self):
        """A deactivated admin's surviving session cookie must not keep
        working the new privileged endpoints."""
        fn = _extract_function(ROUTES_SRC, "_require_dashboard_admin")
        assert "not user.is_active" in fn

    def test_no_raw_username_in_js_string_context(self):
        """Usernames rendered into onclick handlers must go through
        |tojson — raw interpolation into a JS string is an injection
        vector and breaks on apostrophes."""
        for name in ("user_detail.html", "api_keys.html"):
            tpl = (_TEMPLATES_DIR / "admin" / name).read_text()
            for line in tpl.splitlines():
                if "onclick" in line and "username" in line:
                    assert "|tojson" in line, f"raw username in JS: {name}: {line.strip()}"


# ---------------------------------------------------------------------------
# Admin API: key revoke/delete
# ---------------------------------------------------------------------------

class TestAdminKeyEndpoints:
    def test_revoke_endpoint_exists_and_admin_gated(self):
        assert '@router.post("/api-keys/{key_id}/revoke")' in ADMIN_API_SRC
        fn = _extract_function(ADMIN_API_SRC, "revoke_api_key_admin")
        assert "require_admin()" in fn
        assert "log_admin_action" in fn

    def test_delete_endpoint_refuses_referenced_keys(self):
        assert '@router.delete("/api-keys/{key_id}")' in ADMIN_API_SRC
        fn = _extract_function(ADMIN_API_SRC, "delete_api_key_admin")
        assert "count_api_key_references" in fn
        assert "HTTP_409_CONFLICT" in fn
        # TOCTOU: a row can appear between the check and the delete —
        # the IntegrityError must land as a 409, not a 500
        assert "IntegrityError" in fn
        assert "rollback" in fn

    def test_reference_counter_covers_audit_tables(self):
        fn = _extract_function(CRUD_SRC, "count_api_key_references")
        for table in ("Request", "StoredResponse", "Conversation", "VideoJob"):
            assert table in fn, f"{table} missing from key reference check"


# ---------------------------------------------------------------------------
# Admin password reset
# ---------------------------------------------------------------------------

class TestPasswordReset:
    def test_api_endpoint_local_only(self):
        assert '@router.post("/users/{user_id}/reset-password")' in ADMIN_API_SRC
        fn = _extract_function(ADMIN_API_SRC, "reset_user_password")
        assert "require_admin()" in fn
        # Credential-based gate: account_type is 'admin' for local users
        # in admin groups, so it must NOT be the discriminator
        assert "password_hash is None" in fn
        assert 'account_type != "local"' not in fn
        assert "hash_password" in fn
        # The new password must never reach the audit log
        assert "new_password" not in fn.split("log_admin_action", 1)[1].split(")")[0]

    def test_api_password_minimum_length(self):
        assert "min_length=8" in _extract_function(
            ADMIN_API_SRC.replace("class ResetPasswordRequest", "def _rpr_marker():\n    pass\nclass ResetPasswordRequest"),
            "_rpr_marker",
        ) or "new_password: str = Field(..., min_length=8" in ADMIN_API_SRC

    def test_dashboard_route_validates(self):
        fn = _extract_function(ROUTES_SRC, "admin_reset_user_password")
        assert "password_hash is None" in fn
        assert "new_password != confirm_password" in fn
        assert "len(new_password) < 8" in fn
        assert "log_admin_action" in fn

    def test_reset_modal_gated_on_credential(self):
        tpl = (_TEMPLATES_DIR / "admin" / "user_detail.html").read_text()
        assert "resetPasswordModal" in tpl
        assert "detail_user.password_hash" in tpl
        assert "account_type == 'local'" not in tpl.split("resetPasswordModal")[0].rsplit("card border-danger", 1)[-1]


# ---------------------------------------------------------------------------
# delete_user cascade completeness (schema-driven)
# ---------------------------------------------------------------------------

class TestDeleteUserCascade:
    def test_every_user_fk_column_is_handled(self, models):
        """Walk the live schema: EVERY FK COLUMN referencing users.id must
        appear as Class.column in delete_user (deleted or detached).
        Per-column, not per-table — tables carry second FKs like
        api_keys.promoted_by and quota_requests.reviewed_by that a
        user_id-scoped delete never touches (found by adversarial
        review after a weaker per-table version of this test passed)."""
        fn = _extract_function(CRUD_SRC, "delete_user")
        base = models.Base

        # table name -> mapped class name
        table_to_class = {
            mapper.persist_selectable.name: mapper.class_.__name__
            for mapper in base.registry.mappers
        }

        unhandled = []
        for table in base.metadata.tables.values():
            if table.name == "users":
                continue
            for fk in table.foreign_keys:
                if fk.column.table.name != "users":
                    continue
                cls = table_to_class.get(table.name, table.name)
                col = fk.parent.name
                if f"{cls}.{col}" not in fn:
                    unhandled.append(f"{table.name}.{col} ({cls})")
        assert not unhandled, (
            "delete_user does not handle users.id FK columns: "
            f"{unhandled} — add delete/detach handling for each."
        )

    def test_reviewer_and_actor_fks_are_detached(self):
        """Rows owned by OTHER users where the deleted user was the
        approving admin must be detached, not deleted."""
        fn = _extract_function(CRUD_SRC, "delete_user")
        for detach in (
            "ApiKey.promoted_by",
            "QuotaRequest.reviewed_by",
            "ServiceKeyRequest.reviewed_by",
            "DlpAlert.acknowledged_by",
        ):
            assert detach in fn, f"missing detach of {detach}"

    def test_dlp_request_reference_detached_before_request_delete(self):
        fn = _extract_function(CRUD_SRC, "delete_user")
        assert "DlpAlert.request_id" in fn
        # The detach must precede the request delete
        assert fn.index("DlpAlert.request_id") < fn.index(
            "delete(Request).where(Request.user_id == user_id)"
        )

    def test_children_of_requests_handled(self):
        fn = _extract_function(CRUD_SRC, "delete_user")
        for cls in ("SchedulerDecision", "Response", "Artifact"):
            assert cls in fn

    def test_preserved_tables_are_detached_not_deleted(self):
        fn = _extract_function(CRUD_SRC, "delete_user")
        for cls in ("BlogPost", "EmailLog", "AdminAuditLog"):
            assert f"update({cls})" in fn, f"{cls} must be detached via UPDATE"
            assert f"delete({cls})" not in fn, f"{cls} rows must never be deleted"

    def test_detached_columns_are_nullable(self, models):
        blog = models.Base.metadata.tables["blog_posts"]
        email = models.Base.metadata.tables["email_log"]
        audit = models.Base.metadata.tables["admin_audit_log"]
        assert blog.c.author_id.nullable
        assert email.c.sent_by.nullable
        assert audit.c.user_id.nullable

    def test_api_delete_returns_conflict_not_500(self):
        fn = _extract_function(ADMIN_API_SRC, "delete_user")
        assert "IntegrityError" in fn
        assert "HTTP_409_CONFLICT" in fn
        # Never echo DBAPIError text (leaks statement parameters)
        assert "str(e)" not in fn

    def test_dashboard_delete_requires_username_confirmation(self):
        fn = _extract_function(ROUTES_SRC, "admin_delete_user")
        assert "confirm_username != target.username" in fn
        assert "user_id == admin.id" in fn  # self-delete block
        assert "IntegrityError" in fn
        assert "str(e)" not in fn

    def test_migration_070_exists(self):
        migration = next(_MIGRATIONS_DIR.glob("*070_nullable_user_refs.py"))
        src = migration.read_text()
        for table, col in (
            ("blog_posts", "author_id"),
            ("email_log", "sent_by"),
            ("admin_audit_log", "user_id"),
        ):
            assert table in src and col in src


# ---------------------------------------------------------------------------
# Manual purge
# ---------------------------------------------------------------------------

class TestManualPurge:
    def test_purge_categories_exclude_admin_audit(self):
        match = re.search(r"PURGE_CATEGORIES = \((.*?)\)", RETENTION_SRC, re.DOTALL)
        assert match
        cats = match.group(1)
        assert "admin_audit" not in cats
        for cat in ("requests", "chat", "telemetry", "request_images",
                    "responses_store", "conversations"):
            assert f'"{cat}"' in cats

    def test_purge_route_verifies_confirmation_server_side(self):
        fn = _extract_function(ROUTES_SRC, "admin_retention_post")
        assert 'form.get("confirm_text") != "PURGE"' in fn
        assert "try_run_purge_with_lock" in fn
        assert '"retention.purge"' in fn  # audit-logged

    def test_purge_runs_under_retention_lock(self):
        fn = _extract_function(RETENTION_SRC, "try_run_purge_with_lock")
        assert "GET_LOCK" in fn
        assert "_RETENTION_LOCK_NAME" in fn

    def test_request_reference_detach_in_both_paths(self):
        """user_images/video_jobs/stored_responses back-references must be
        nulled before requests are deleted — archive and no-archive paths."""
        detach = _extract_function(RETENTION_SRC, "_detach_request_references")
        for cls in ("UserImage", "VideoJob", "StoredResponse", "DlpAlert"):
            assert cls in detach
        archive_fn = _extract_function(RETENTION_SRC, "archive_expired_requests")
        assert "_detach_request_references" in archive_fn
        no_archive_fn = _extract_function(
            RETENTION_SRC, "delete_expired_requests_no_archive"
        )
        assert "_detach_request_references" in no_archive_fn

    def test_purge_template_has_verification_modal(self):
        tpl = (_TEMPLATES_DIR / "admin" / "retention.html").read_text()
        assert "purgeConfirmModal" in tpl
        assert "PURGE" in tpl
        assert 'name="action" value="purge"' in tpl


# ---------------------------------------------------------------------------
# Template affordances
# ---------------------------------------------------------------------------

class TestTemplates:
    def test_user_detail_danger_zone(self):
        tpl = (_TEMPLATES_DIR / "admin" / "user_detail.html").read_text()
        assert "/set-active" in tpl
        assert "/delete" in tpl
        assert "deleteUserModal" in tpl
        assert "confirm_username" in tpl
        # Both danger actions hidden from yourself
        assert "detail_user.id != user.id" in tpl

    def test_admin_api_keys_personal_revoke(self):
        tpl = (_TEMPLATES_DIR / "admin" / "api_keys.html").read_text()
        assert "/admin/api-keys/" in tpl and "/revoke" in tpl
        assert "not k.is_service and k.status.value == 'active'" in tpl

    def test_user_detail_key_actions(self):
        tpl = (_TEMPLATES_DIR / "admin" / "user_detail.html").read_text()
        assert "/admin/api-keys/{{ k.id }}/revoke" in tpl


# ===================================================================
# 2.9.49 — API key renewal + expiry lifecycle
# ===================================================================

REGISTRY_SRC = (_APP_DIR / "core" / "telemetry" / "registry.py").read_text()


def _extract_method(source: str, name: str) -> str:
    """Like _extract_function but for an indented (class) method."""
    pattern = re.compile(
        rf"^[ \t]+(?:async )?def {name}\(.*?(?=^[ \t]+@|^[ \t]+(?:async )?def |^class |\Z)",
        re.DOTALL | re.MULTILINE,
    )
    match = pattern.search(source)
    assert match, f"method {name} not found"
    return match.group(0)
SECURITY_SRC = (_APP_DIR / "security" / "api_keys.py").read_text()
USER_DASHBOARD_HTML = (_TEMPLATES_DIR / "user" / "dashboard.html").read_text()


class TestKeyRenewal:
    """Renew = same secret, new period.  Owner-scoped, non-service, never a
    revoked key; an expired key is reactivated but counts against the cap."""

    def test_crud_renew_is_owner_scoped_and_excludes_service_and_revoked(self):
        fn = _extract_function(CRUD_SRC, "renew_api_key")
        assert "ApiKey.user_id == owner_user_id" in fn
        assert "ApiKey.is_service.is_(False)" in fn
        assert "ApiKey.status.in_([ApiKeyStatus.ACTIVE, ApiKeyStatus.EXPIRED])" in fn
        assert "ApiKeyStatus.REVOKED" not in fn
        # same key: only expires_at and status change, never key_hash/sha
        assert "key_hash" not in fn and "key_sha256" not in fn
        assert "api_key.status = ApiKeyStatus.ACTIVE" in fn

    def test_crud_renew_enforces_the_key_cap_when_reactivating(self):
        fn = _extract_function(CRUD_SRC, "renew_api_key")
        assert "api_key.status == ApiKeyStatus.EXPIRED" in fn
        assert "count_user_active_api_keys(db, owner_user_id) >= max_active" in fn
        assert 'raise ValueError("limit")' in fn

    def test_dashboard_renew_route_is_scoped_and_audited(self):
        fn = _extract_function(ROUTES_SRC, "renew_key")
        assert "owner_user_id=user_id" in fn
        assert "api_key_expiry_days" in fn, "period comes from the group lifetime"
        assert "max_active=max_keys" in fn
        assert 'action="apikey.renew"' in fn
        assert "key_error=limit" in fn and "key_error=not_renewable" in fn
        assert "key_success=renewed" in fn
        assert '@dashboard_router.post("/dashboard/renew-key/{key_id}")' in ROUTES_SRC

    def test_dashboard_passes_success_and_period_to_template(self):
        fn = _extract_function(ROUTES_SRC, "user_dashboard")
        assert "key_success: Optional[str] = None" in fn
        assert '"key_expiry_days"' in fn

    def test_template_offers_renew_for_live_and_expired_keys(self):
        html = USER_DASHBOARD_HTML
        assert html.count('action="/dashboard/renew-key/{{ key.id }}"') == 2, \
            "one Renew in the active branch, one in the expired branch"
        assert "key_success == 'renewed'" in html
        assert "key_error == 'not_renewable'" in html
        # status=expired (set by the sweeper) renders as expired too
        assert "key.status.value == 'expired' or" in html


class TestKeyExpiryLifecycle:
    """Overdue keys become status=expired hourly; pruning never hits the FK;
    the web UI only ever uses a LIVE key."""

    def test_sweeper_flips_overdue_active_keys(self):
        fn = _extract_function(CRUD_SRC, "expire_overdue_api_keys")
        assert "ApiKey.is_service.is_(False)" in fn
        assert "ApiKey.status == ApiKeyStatus.ACTIVE" in fn
        assert "ApiKey.expires_at < now" in fn
        assert ".values(status=ApiKeyStatus.EXPIRED)" in fn

    def test_prune_excludes_every_referencing_fk(self):
        """The old DELETE failed every run with MariaDB 1451 (requests.api_key_id)."""
        fn = _extract_function(CRUD_SRC, "delete_expired_api_keys")
        assert "_columns_referencing(ApiKey.__table__)" in fn
        assert "~exists().where(column == ApiKey.id)" in fn
        helper = _extract_function(CRUD_SRC, "_columns_referencing")
        assert "fk.column.table is table" in helper
        assert "Base.metadata.tables" in helper

    def test_referencing_fk_discovery_matches_the_models(self, models):
        """Every ForeignKey to api_keys.id in models.py must be one the helper
        would find — mutation guard against a hand-maintained list."""
        src = (_DB_DIR / "models.py").read_text()
        declared = src.count('ForeignKey("api_keys.id")')
        assert declared >= 5, "expected requests, video, telemetry, … to reference api_keys"
        # The helper is metadata-driven, so it needs no edit per new table:
        # no referencing model is named in the prune itself.
        prune = _extract_function(CRUD_SRC, "delete_expired_api_keys")
        for model in ("Request.", "VideoJob.", "VideoAsset.", "Telemetry."):
            assert model not in prune, model

    def test_registry_runs_key_maintenance_in_its_own_loop(self):
        # As of 2.9.50 key maintenance is a dedicated loop, not part of the
        # telemetry cleanup — see TestKeyMaintenanceConfig for the details.
        run = _extract_method(REGISTRY_SRC, "run_api_key_maintenance")
        assert "expire_overdue_api_keys(db)" in run
        assert "delete_expired_api_keys(db, grace_days=grace)" in run

    def test_web_pickers_use_first_live_key_not_index_zero(self):
        for name in ("chat.py", "images.py", "video.py"):
            src = (_DASHBOARD_DIR / name).read_text()
            assert "api_keys[0]" not in src, name
            assert "first_live_api_key(api_keys)" in src, name
            assert "from backend.app.security.api_keys import first_live_api_key" in src, name
        chat = (_DASHBOARD_DIR / "chat.py").read_text()
        assert "ocr_api_keys[0]" not in chat and "first_live_api_key(ocr_api_keys)" in chat

    def test_live_predicate_checks_status_and_expiry_and_spares_service_keys(self):
        fn = _extract_function(SECURITY_SRC, "api_key_is_live")
        assert "db_key.status != ApiKeyStatus.ACTIVE" in fn
        assert "if db_key.is_service:" in fn and "return True" in fn
        assert "expires_at >= datetime.now(timezone.utc)" in fn
        picker = _extract_function(SECURITY_SRC, "first_live_api_key")
        assert "api_key_is_live(k)" in picker

    def test_user_key_list_keeps_expired_rows_visible(self):
        """The dashboard must list an expired key so its owner can Renew it."""
        fn = _extract_function(CRUD_SRC, "get_user_api_keys")
        assert "ApiKey.status.in_([ApiKeyStatus.ACTIVE, ApiKeyStatus.EXPIRED])" in fn


# ===================================================================
# 2.9.50 — admin-configurable API key expiry maintenance
# ===================================================================

API_KEYS_HTML = (_TEMPLATES_DIR / "admin" / "api_keys.html").read_text()


class TestKeyMaintenanceConfig:
    """Sweep interval + prune grace are admin settings on Admin -> API Keys;
    a dedicated background loop reads them; grace 0 = never hard-delete."""

    def test_dedicated_loop_separate_from_telemetry(self):
        # key work was split out of the old _cleanup_old_telemetry loop; that
        # loop is gone too (the retention cycle owns telemetry pruning)
        assert "_cleanup_old_telemetry" not in REGISTRY_SRC
        # it lives in its own loop, started and cancelled like the others
        assert "async def _api_key_maintenance_loop" in REGISTRY_SRC
        assert "self._api_key_maint_task = asyncio.create_task(self._api_key_maintenance_loop())" in REGISTRY_SRC
        assert "self._api_key_maint_task.cancel()" in REGISTRY_SRC

    def test_loop_reads_interval_from_config_and_clamps(self):
        fn = _extract_method(REGISTRY_SRC, "_api_key_maintenance_interval")
        assert '"apikey.maintenance.interval_seconds"' in fn
        assert "API_KEY_MAINT_MIN_INTERVAL" in fn and "API_KEY_MAINT_MAX_INTERVAL" in fn

    def test_run_maintenance_reads_grace_and_zero_means_never_delete(self):
        fn = _extract_method(REGISTRY_SRC, "run_api_key_maintenance")
        assert '"apikey.maintenance.prune_grace_days"' in fn
        assert "expire_overdue_api_keys(db)" in fn
        assert "delete_expired_api_keys(db, grace_days=grace)" in fn
        assert "if grace > 0 else 0" in fn, "grace 0 must skip the delete"

    def test_route_saves_settings_and_runs_now_admin_gated(self):
        fn = _extract_function(ROUTES_SRC, "admin_api_key_maintenance")
        assert "not user.group or not user.group.is_admin" in fn
        assert 'set_config(db, "apikey.maintenance.interval_seconds", interval)' in fn
        assert 'set_config(db, "apikey.maintenance.prune_grace_days", grace)' in fn
        assert 'action == "run_now"' in fn
        assert "run_api_key_maintenance()" in fn
        assert 'action="apikey.maintenance_config"' in fn
        assert '@dashboard_router.post("/admin/api-keys/maintenance")' in ROUTES_SRC

    def test_route_validates_bounds(self):
        fn = _extract_function(ROUTES_SRC, "admin_api_key_maintenance")
        assert "API_KEY_MAINT_MIN_INTERVAL <= interval <= " in fn
        assert "grace < 0 or grace > 3650" in fn

    def test_context_exposes_settings_and_live_counts(self):
        fn = _extract_function(ROUTES_SRC, "_api_key_maintenance_context")
        for key in ("km_interval_seconds", "km_prune_grace_days", "km_count_active",
                    "km_count_expired", "km_overdue_unswept"):
            assert f'"{key}"' in fn, key

    def test_template_has_the_maintenance_card(self):
        html = API_KEYS_HTML
        assert 'action="/admin/api-keys/maintenance"' in html
        assert 'name="interval_seconds"' in html
        assert 'name="prune_grace_days"' in html
        assert 'value="run_now"' in html
        assert "0 = never delete" in html


# ===================================================================
# 2.9.51 — API key expiry reminder emails (admin config + wiring)
# ===================================================================

class TestKeyReminderAdmin:
    def test_route_saves_reminder_settings_and_validates(self):
        fn = _extract_function(ROUTES_SRC, "admin_api_key_maintenance")
        assert 'action == "save_reminders"' in fn
        assert 'set_config(db, "apikey.reminders.enabled"' in fn
        assert 'set_config(db, "apikey.reminders.early_days"' in fn
        assert 'set_config(db, "apikey.reminders.urgent_days"' in fn
        assert "0 <= urgent < early" in fn
        assert 'action="apikey.reminders_config"' in fn

    def test_route_has_test_reminder_actions(self):
        fn = _extract_function(ROUTES_SRC, "admin_api_key_maintenance")
        assert 'action in ("test_early", "test_urgent")' in fn
        assert "send_test_reminder(db, test_to, tier=tier)" in fn

    def test_context_exposes_reminder_settings(self):
        fn = _extract_function(ROUTES_SRC, "_api_key_maintenance_context")
        for key in ("rem_enabled", "rem_early_days", "rem_urgent_days", "rem_test_recipient"):
            assert f'"{key}"' in fn, key

    def test_registry_runs_reminders_in_maintenance(self):
        fn = _extract_method(REGISTRY_SRC, "run_api_key_maintenance")
        assert "send_expiry_reminders(rdb)" in fn
        # own session so an SMTP failure can't roll back the status/prune work
        assert 'logger.error("api_key_reminder_error"' in fn

    def test_renew_re_arms_reminder_watermarks(self):
        fn = _extract_function(CRUD_SRC, "renew_api_key")
        assert "api_key.reminder_early_sent_at = None" in fn
        assert "api_key.reminder_urgent_sent_at = None" in fn

    def test_template_has_reminder_card_with_one_email_note(self):
        html = API_KEYS_HTML
        assert 'value="save_reminders"' in html
        assert 'name="reminders_enabled"' in html
        assert 'name="early_days"' in html and 'name="urgent_days"' in html
        assert 'value="test_early"' in html and 'value="test_urgent"' in html
        assert 'name="test_recipient"' in html
        assert "one email per stage" in html  # the explicit "not one per key" promise

    def test_model_has_reminder_watermark_columns(self, models):
        cols = models.ApiKey.__table__.columns
        assert "reminder_early_sent_at" in cols
        assert "reminder_urgent_sent_at" in cols

    def test_migration_080_adds_columns_and_seeds_config(self):
        mig = (_MIGRATIONS_DIR / "20260821_000000_080_apikey_expiry_reminders.py").read_text()
        assert 'down_revision = "079"' in mig
        assert 'add_column' in mig and 'reminder_early_sent_at' in mig and 'reminder_urgent_sent_at' in mig
        assert 'apikey.reminders.enabled' in mig
        assert 'drop_column' in mig  # downgrade present
