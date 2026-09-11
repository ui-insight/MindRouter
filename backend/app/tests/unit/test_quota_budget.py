############################################################
#
# mindrouter - unit tests for quota budget resolution + grants
#
# GitHub issue #11: approving a quota-increase request recorded
# `granted_tokens` in the audit log and changed nothing the user could
# spend. Migration 023 had moved the budget to the group, leaving the
# per-user workflow with nowhere to write — and the resolution rule
# duplicated across eight call sites, which is why the drift went unseen.
#
############################################################

"""Budget resolution, the grant path, and the anti-recurrence guard."""

import importlib.util
import pathlib
import re
from types import SimpleNamespace as NS

import pytest

_APP = pathlib.Path(__file__).resolve().parents[2]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, _APP / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


qb = _load("quota_budget", "core/quota_budget.py")


def _user(budget=100_000, has_group=True, rpm=30):
    return NS(group=NS(token_budget=budget, rpm_limit=rpm) if has_group else None)


def _quota(override=None, used=0):
    return NS(token_budget_override=override, tokens_used=used)


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------


def test_no_override_inherits_the_group_budget():
    assert qb.effective_token_budget(_user(100_000), _quota()) == 100_000


def test_override_replaces_the_group_budget():
    """The point of the fix: a per-user grant actually takes effect."""
    assert qb.effective_token_budget(_user(100_000), _quota(override=250_000)) == 250_000


def test_override_can_lower_as_well_as_raise():
    assert qb.effective_token_budget(_user(100_000), _quota(override=1_000)) == 1_000


def test_zero_means_unlimited_for_both_override_and_group():
    """One rule, not two — 0 has always meant unlimited on the group."""
    assert qb.effective_token_budget(_user(0), _quota()) == qb.UNLIMITED
    assert qb.effective_token_budget(_user(100_000), _quota(override=0)) == qb.UNLIMITED


def test_groupless_user_has_no_budget():
    assert qb.effective_token_budget(_user(has_group=False), _quota()) == qb.UNLIMITED


def test_missing_quota_falls_back_to_the_group():
    assert qb.effective_token_budget(_user(100_000), None) == 100_000


def test_override_of_zero_is_distinguished_from_absent():
    """`if override:` would treat a 0 grant as absent — it must be `is not None`."""
    assert qb.effective_token_budget(_user(100_000), _quota(override=0)) == 0
    assert qb.effective_token_budget(_user(100_000), _quota(override=None)) == 100_000


# --------------------------------------------------------------------------
# Enforcement
# --------------------------------------------------------------------------


def test_over_budget_uses_the_override_not_the_group():
    """A user at 99,999/100,000 who was granted 250,000 must not be blocked."""
    u, q = _user(100_000), _quota(override=250_000, used=99_999)
    assert qb.is_over_budget(u, q, additional_cost=10) is False
    assert qb.is_over_budget(_user(100_000), _quota(used=99_999), 10) is True


def test_unlimited_is_never_over_budget():
    assert qb.is_over_budget(_user(0), _quota(used=10**12), 10**6) is False
    assert qb.is_over_budget(_user(100_000), _quota(override=0, used=10**12)) is False


def test_exactly_at_budget_is_not_over_until_it_exceeds():
    assert qb.is_over_budget(_user(100), _quota(used=90), 10) is False
    assert qb.is_over_budget(_user(100), _quota(used=90), 11) is True


def test_budget_source_is_reported_for_audit_clarity():
    assert qb.budget_source(_user(), _quota(override=5)) == "user_override"
    assert qb.budget_source(_user(), _quota()) == "group"
    assert qb.budget_source(_user(has_group=False), None) == "none"


# --------------------------------------------------------------------------
# THE ANTI-RECURRENCE GUARD — the reason the bug survived six months
# --------------------------------------------------------------------------


_SITES = [
    "db/crud.py", "dashboard/routes.py", "api/admin_api.py", "api/voice_api.py",
    "api/search_api.py", "api/mcp_server.py", "services/inference.py",
]


@pytest.mark.parametrize("rel", _SITES)
def test_no_module_resolves_the_budget_itself(rel):
    """All eight sites must go through the helper.

    The issue's own suggested fix listed only six of them, omitting
    services/inference.py — the main chat quota gate. That would have shipped a
    grant honoured by voice/search/MCP but silently ignored by inference.
    """
    src = (_APP / rel).read_text()
    assert "group.token_budget" not in src, (
        f"{rel} resolves the budget inline; use effective_token_budget()"
    )


@pytest.mark.parametrize("rel", _SITES)
def test_every_site_imports_the_helper(rel):
    src = (_APP / rel).read_text()
    assert "quota_budget import" in src, f"{rel} does not import the helper"


def test_enforcement_sites_pass_the_quota_row():
    """Passing only the user silently ignores a grant — the original bug."""
    for rel in ("services/inference.py", "api/voice_api.py", "api/search_api.py",
                "api/mcp_server.py", "db/crud.py"):
        src = (_APP / rel).read_text()
        assert "effective_token_budget(user, quota)" in src, rel


def test_helper_lives_outside_services_to_avoid_an_import_cycle():
    """services/__init__ imports InferenceService, so crud importing a
    services.* module creates a cycle. Verified by routes failing to import."""
    assert (_APP / "core" / "quota_budget.py").exists()
    assert not (_APP / "services" / "quota_budget.py").exists()


# --------------------------------------------------------------------------
# The grant is applied in the same call that records the approval
# --------------------------------------------------------------------------


def _crud_fn_src(code_only=False):
    src = (_APP / "db" / "crud.py").read_text()
    i = src.index("async def review_quota_request(")
    body = src[i : src.index("\nasync def ", i + 10)]
    if code_only:
        # Skip the docstring: it describes the override, which would otherwise
        # satisfy source assertions that are meant to be about the code.
        first = body.index('"""')
        body = body[body.index('"""', first + 3) + 3 :]
    return body


def test_review_accepts_and_applies_granted_tokens():
    body = _crud_fn_src()
    assert "granted_tokens: Optional[int] = None" in body
    assert "quota.token_budget_override = int(amount)" in body


def test_approval_defaults_to_the_amount_requested():
    body = _crud_fn_src()
    assert "quota_request.requested_tokens" in body


def test_denial_never_applies_a_grant():
    body = _crud_fn_src(code_only=True)
    i = body.index("if status == QuotaRequestStatus.APPROVED:")
    assert "token_budget_override" not in body[:i], "grant applied before the approval check"


def test_negative_grant_is_refused():
    body = _crud_fn_src()
    assert "int(amount) < 0" in body
    assert "raise ValueError" in body


def test_missing_quota_row_is_created_rather_than_dropping_the_grant():
    body = _crud_fn_src()
    assert "create_quota(" in body
    assert "rpm_limit=" in body


def test_grant_is_replacement_not_addition():
    """Re-approving must not compound the budget."""
    body = _crud_fn_src(code_only=True)
    assert "+=" not in body.split("token_budget_override")[1][:40]


# --------------------------------------------------------------------------
# Both approval paths must pass the grant through
# --------------------------------------------------------------------------


def test_api_path_passes_granted_tokens_to_crud():
    src = (_APP / "api" / "admin_api.py").read_text()
    body = src[src.index("async def review_quota_request("):]
    body = body[: body.index("# User & API Key Provisioning")]
    assert "granted_tokens=review.granted_tokens" in body
    assert "granted_tokens" in body.split("return {")[1]   # reported back


def test_api_path_no_longer_shadows_the_status_module():
    """`status = QuotaRequestStatus...` made status.HTTP_404_NOT_FOUND an
    AttributeError, turning a missing request into a 500."""
    src = (_APP / "api" / "admin_api.py").read_text()
    body = src[src.index("async def review_quota_request("):]
    body = body[: body.index("# User & API Key Provisioning")]
    assert "new_status = QuotaRequestStatus" in body
    assert re.search(r"^\s+status = QuotaRequestStatus", body, re.M) is None


def test_dashboard_path_collects_and_applies_a_grant():
    src = (_APP / "dashboard" / "routes.py").read_text()
    body = src[src.index("async def approve_request("):]
    body = body[: body.index("@dashboard_router.post(\"/admin/requests/{request_id}/deny\")")]
    assert 'form.get("granted_tokens")' in body
    assert "granted_tokens=granted" in body
    assert "cannot+be+negative" in body


def test_dashboard_form_exposes_the_grant_field():
    tpl = (_APP / "dashboard" / "templates" / "admin" / "requests.html").read_text()
    assert 'name="granted_tokens"' in tpl
    assert "req.requested_tokens if req.requested_tokens is not none" in tpl  # prefilled, null-safe
    assert "{% if error %}" in tpl                  # failures are visible


# --------------------------------------------------------------------------
# Migration
# --------------------------------------------------------------------------


def test_migration_is_additive_and_nullable():
    mig = next((_APP / "db" / "migrations" / "versions").glob("*086*.py")).read_text()
    assert 'revision = "086"' in mig
    assert 'down_revision = "085"' in mig
    assert "add_column" in mig and "nullable=True" in mig
    assert "BigInteger" in mig
    assert "drop_column" in mig   # downgrade path exists


def test_model_column_matches_the_migration():
    src = (_APP / "db" / "models.py").read_text()
    assert "token_budget_override" in src
    block = src[src.index("token_budget_override")]
    assert "BigInteger" in src[src.index("token_budget_override") - 200 :
                               src.index("token_budget_override") + 200]


def test_validation_happens_before_any_row_is_mutated():
    """Review finding: get_async_db() commits on normal completion, and the
    dashboard turns a ValueError into a redirect, so mutating status before
    validating would commit an approval with no grant applied."""
    body = _crud_fn_src(code_only=True)
    assert body.index("raise ValueError") < body.index("quota_request.status = status")
