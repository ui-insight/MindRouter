############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# quota_budget.py: the one place a token budget is resolved
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Single source of truth for "how many tokens may this user spend?".

Why this module exists at all
----------------------------
Migration 023 moved the budget from `quotas.token_budget` to
`groups.token_budget` and left eight call sites each resolving it inline as
``user.group.token_budget if user.group else 0``. The quota-increase workflow
was per-user and had nowhere to write a grant, so approvals became silent
no-ops (GitHub issue #11) — and because the rule lived in eight places, the
drift went unnoticed for six months.

Reintroducing a per-user override would have meant editing all eight sites
correctly, and the issue's own suggested fix listed only six of them — missing
``services/inference.py``, the main chat quota gate. That fix would have
produced a grant honoured by voice, search and MCP but silently ignored by
ordinary inference: worse than the bug, because inconsistent.

So the resolution rule now lives here, once, and
``test_quota_budget.py`` fails if any other module reads
``group.token_budget`` directly. The next schema change has one place to
update and a guard that notices if it is missed.

Why it lives in `core` and not `services`
----------------------------------------
``backend/app/services/__init__.py`` imports ``InferenceService``, so importing
any ``services.*`` module drags in ``inference`` -> ``crud``. Since ``crud``
itself needs this helper, putting it under ``services`` creates an import
cycle (``backend/app/core/__init__.py`` is comment-only, which is why
``schema_guard`` lives here too).

The rule
--------
``quota.token_budget_override`` when set, else the user's group budget.
``0`` means unlimited in BOTH (pre-existing group semantics, kept identical so
there is one rule rather than two), and a user with no group has no budget.
"""

from typing import Any, Optional

# A budget of 0 means "no limit" — the semantics groups.token_budget has
# always had. Exported so callers express intent instead of comparing to 0.
UNLIMITED = 0


def effective_token_budget(user: Any, quota: Optional[Any] = None) -> int:
    """Tokens this user may spend per budget period.

    Args:
        user: the User (its ``group`` relationship must be loaded).
        quota: the user's Quota row, if already loaded. When omitted the
            group budget is used — callers that enforce a limit MUST pass the
            quota, or a per-user grant is silently ignored, which is the
            original bug.

    Returns:
        Token budget, where ``UNLIMITED`` (0) means no limit.
    """
    override = getattr(quota, "token_budget_override", None) if quota is not None else None
    if override is not None:
        return int(override)
    group = getattr(user, "group", None)
    if group is None:
        return UNLIMITED
    return int(getattr(group, "token_budget", 0) or 0)


def is_over_budget(user: Any, quota: Optional[Any], additional_cost: int = 0) -> bool:
    """Would this user exceed their budget by spending ``additional_cost``?

    Always False when the budget is UNLIMITED.
    """
    budget = effective_token_budget(user, quota)
    if budget <= UNLIMITED:
        return False
    used = int(getattr(quota, "tokens_used", 0) or 0) if quota is not None else 0
    return used + int(additional_cost) > budget


def budget_source(user: Any, quota: Optional[Any] = None) -> str:
    """Where the budget came from — for admin UI and audit clarity."""
    override = getattr(quota, "token_budget_override", None) if quota is not None else None
    if override is not None:
        return "user_override"
    if getattr(user, "group", None) is not None:
        return "group"
    return "none"
