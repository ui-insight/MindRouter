############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# me_api.py: What the calling key may spend — its requests-per-minute
#     limit and token budget — so a client can pace itself instead of
#     discovering the limits by being refused.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""GET /v1/me/limits — the caller's own rate limit and token budget.

A long-running client (VandalChat's Deep Research runs, which make a few
hundred requests over half an hour on the user's own key) has to stay
inside that user's RPM window without ever tripping it, and inside their
token budget. Until now the only way to learn either was a 429. This
endpoint reports the same numbers `_check_quota` enforces, resolved the
same way: the key's own `rpm_limit` override, else the user's quota row;
the quota's `token_budget_override`, else the group budget (0 = unlimited).
It is read-only, costs no tokens, and is available to any inference-scoped
key for its own identity only.
"""

from datetime import timedelta
from typing import Optional, Tuple

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.auth import authenticate_request
from backend.app.core.quota_budget import effective_token_budget
from backend.app.db import crud
from backend.app.db.crud import _ensure_aware
from backend.app.db.models import ApiKey, User
from backend.app.db.session import get_async_db

router = APIRouter(tags=["me"])


class LimitsResponse(BaseModel):
    """What the calling key may spend."""

    user_id: int
    username: str
    group: Optional[str] = None
    # Requests per minute, as enforced (0 = no limit).
    rpm_limit: int
    # Tokens per budget period; 0 = unlimited, exactly as on the group.
    token_budget: int
    tokens_used: int
    tokens_remaining: Optional[int] = None
    budget_period_days: int
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    key_prefix: Optional[str] = None


@router.get("/v1/me/limits", response_model=LimitsResponse)
async def my_limits(
    db: AsyncSession = Depends(get_async_db),
    auth: Tuple[User, ApiKey] = Depends(authenticate_request),
) -> LimitsResponse:
    """The rate limit and token budget that apply to the calling key."""
    user, api_key = auth
    # Same order as InferenceService._check_quota so the numbers agree.
    await crud.reset_quota_if_needed(db, user.id)
    quota = await crud.get_user_quota(db, user.id)
    budget = effective_token_budget(user, quota)
    rpm_limit = 0
    if api_key is not None and api_key.rpm_limit:
        rpm_limit = int(api_key.rpm_limit)
    elif quota is not None:
        rpm_limit = int(quota.rpm_limit or 0)
    tokens_used = int(quota.tokens_used) if quota is not None else 0
    period_days = int(quota.budget_period_days) if quota is not None else 30
    period_start = _ensure_aware(quota.budget_period_start) if quota is not None else None
    period_end = period_start + timedelta(days=period_days) if period_start else None
    group = getattr(user, "group", None)
    return LimitsResponse(
        user_id=user.id,
        username=user.username,
        group=getattr(group, "name", None) if group is not None else None,
        rpm_limit=rpm_limit,
        token_budget=int(budget),
        tokens_used=tokens_used,
        tokens_remaining=(max(0, int(budget) - tokens_used) if budget > 0 else None),
        budget_period_days=period_days,
        period_start=period_start.isoformat() if period_start else None,
        period_end=period_end.isoformat() if period_end else None,
        key_prefix=getattr(api_key, "key_prefix", None) if api_key is not None else None,
    )
