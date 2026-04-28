############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# crud.py: Database CRUD operations for all entities
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Database CRUD operations for MindRouter."""

import json as _json
from datetime import datetime, timedelta, timezone
from enum import Enum as PyEnum
import time as _time
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import DateTime as SADateTime, and_, case, delete, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.app.db.models import (
    AdminAuditLog,
    ApiKey,
    ApiKeyStatus,
    AppConfig,
    Artifact,
    Backend,
    BackendEngine,
    BackendStatus,
    BackendTelemetry,
    BlogPost,
    DlpAlert,
    DlpSeverity,
    EmailLog,
    GPUDevice,
    GPUDeviceTelemetry,
    Group,
    Model,
    ModelArchivedStats,
    ModelDescriptionCache,
    Modality,
    Node,
    NodeStatus,
    NodeTelemetry,
    Quota,
    QuotaRequest,
    QuotaRequestStatus,
    RegistryMeta,
    Request,
    RequestStatus,
    Response,
    SchedulerDecision,
    ServiceKeyRequest,
    ServiceKeyRequestStatus,
    User,
    UserRole,
)


def _ensure_aware(dt: Optional[datetime]) -> Optional[datetime]:
    """Ensure a datetime is timezone-aware (MariaDB returns naive datetimes)."""
    if dt is not None and dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


# Group CRUD
async def create_group(
    db: AsyncSession,
    name: str,
    display_name: str,
    description: Optional[str] = None,
    token_budget: int = 100000,
    rpm_limit: int = 30,
    scheduler_weight: int = 1,
    is_admin: bool = False,
    is_auditor: bool = False,
    api_key_expiry_days: int = 45,
    max_api_keys: int = 16,
) -> Group:
    """Create a new group."""
    group = Group(
        name=name,
        display_name=display_name,
        description=description,
        token_budget=token_budget,
        rpm_limit=rpm_limit,
        scheduler_weight=scheduler_weight,
        is_admin=is_admin,
        is_auditor=is_auditor,
        api_key_expiry_days=api_key_expiry_days,
        max_api_keys=max_api_keys,
    )
    db.add(group)
    await db.flush()
    return group


async def get_group_by_id(db: AsyncSession, group_id: int) -> Optional[Group]:
    """Get group by ID."""
    result = await db.execute(select(Group).where(Group.id == group_id))
    return result.scalar_one_or_none()


async def get_group_by_name(db: AsyncSession, name: str) -> Optional[Group]:
    """Get group by name."""
    result = await db.execute(select(Group).where(Group.name == name))
    return result.scalar_one_or_none()


async def get_all_groups(db: AsyncSession) -> List[Group]:
    """Get all groups."""
    result = await db.execute(select(Group).order_by(Group.name))
    return list(result.scalars().all())


async def get_all_groups_with_counts(db: AsyncSession) -> List[Tuple[Group, int]]:
    """Get all groups with user counts."""
    result = await db.execute(
        select(Group, func.count(User.id).label("user_count"))
        .outerjoin(User, and_(User.group_id == Group.id, User.deleted_at.is_(None)))
        .group_by(Group.id)
        .order_by(Group.name)
    )
    return [(row[0], row[1]) for row in result.all()]


async def update_group(db: AsyncSession, group_id: int, **kwargs) -> Optional[Group]:
    """Update a group's fields."""
    result = await db.execute(select(Group).where(Group.id == group_id))
    group = result.scalar_one_or_none()
    if not group:
        return None
    for key, value in kwargs.items():
        if hasattr(group, key):
            setattr(group, key, value)
    await db.flush()

    # Propagate quota-related fields to existing user quotas
    quota_fields = {}
    if "rpm_limit" in kwargs:
        quota_fields["rpm_limit"] = kwargs["rpm_limit"]
    if quota_fields:
        user_ids = select(User.id).where(User.group_id == group_id)
        await db.execute(
            update(Quota).where(Quota.user_id.in_(user_ids)).values(**quota_fields)
        )
        await db.flush()

    return group


async def delete_group(db: AsyncSession, group_id: int) -> bool:
    """Delete a group. Fails if users are assigned to it."""
    # Check for users in this group
    result = await db.execute(
        select(User).where(and_(User.group_id == group_id, User.deleted_at.is_(None))).limit(1)
    )
    if result.scalar_one_or_none():
        return False

    result = await db.execute(select(Group).where(Group.id == group_id))
    group = result.scalar_one_or_none()
    if group:
        await db.delete(group)
        await db.flush()
        return True
    return False


# User CRUD
async def get_user_by_id(db: AsyncSession, user_id: int) -> Optional[User]:
    """Get user by ID with group eagerly loaded."""
    result = await db.execute(
        select(User).options(selectinload(User.group)).where(User.id == user_id)
    )
    return result.scalar_one_or_none()


async def get_user_by_username(db: AsyncSession, username: str) -> Optional[User]:
    """Get user by username with group eagerly loaded."""
    result = await db.execute(
        select(User).options(selectinload(User.group)).where(User.username == username)
    )
    return result.scalar_one_or_none()


async def get_user_by_email(db: AsyncSession, email: str) -> Optional[User]:
    """Get user by email with group eagerly loaded."""
    result = await db.execute(
        select(User).options(selectinload(User.group)).where(User.email == email)
    )
    return result.scalar_one_or_none()


async def get_user_by_azure_oid(db: AsyncSession, azure_oid: str) -> Optional[User]:
    """Get user by Azure AD object ID with group eagerly loaded."""
    result = await db.execute(
        select(User).options(selectinload(User.group)).where(User.azure_oid == azure_oid)
    )
    return result.scalar_one_or_none()


async def create_user(
    db: AsyncSession,
    username: str,
    email: str,
    password_hash: Optional[str] = None,
    role: UserRole = UserRole.STUDENT,
    full_name: Optional[str] = None,
    group_id: Optional[int] = None,
    college: Optional[str] = None,
    department: Optional[str] = None,
    intended_use: Optional[str] = None,
) -> User:
    """Create a new user."""
    user = User(
        username=username,
        email=email,
        password_hash=password_hash,
        role=role,
        full_name=full_name,
        group_id=group_id,
        college=college,
        department=department,
        intended_use=intended_use,
    )
    db.add(user)
    await db.flush()
    return user


async def get_active_user_count(db: AsyncSession, since_hours: int = 24) -> int:
    """Count distinct users who made requests in the last N hours."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)
    result = await db.execute(
        select(func.count(func.distinct(Request.user_id)))
        .where(Request.created_at >= cutoff)
    )
    return result.scalar_one() or 0


async def get_users(
    db: AsyncSession,
    skip: int = 0,
    limit: int = 100,
    role: Optional[UserRole] = None,
    group_id: Optional[int] = None,
    is_active: Optional[bool] = None,
    search: Optional[str] = None,
    sort_by: Optional[str] = None,
    sort_dir: str = "desc",
) -> Tuple[List[User], int]:
    """Get list of users with optional filtering and sorting. Returns (users, total_count)."""
    conditions = [User.deleted_at.is_(None)]
    if role:
        conditions.append(User.role == role)
    if group_id is not None:
        conditions.append(User.group_id == group_id)
    if is_active is not None:
        conditions.append(User.is_active == is_active)
    if search:
        search_pattern = f"%{search}%"
        conditions.append(
            or_(
                User.username.ilike(search_pattern),
                User.email.ilike(search_pattern),
                User.full_name.ilike(search_pattern),
            )
        )

    where_clause = and_(*conditions)

    # Total count
    count_result = await db.execute(select(func.count(User.id)).where(where_clause))
    total = count_result.scalar_one()

    # Determine sort order
    asc = sort_dir.lower() == "asc"

    if sort_by == "tokens":
        # Sort by total tokens (live + archived) via subqueries
        token_subq = (
            select(
                Request.user_id,
                func.coalesce(func.sum(Request.total_tokens), 0).label("total_tokens"),
            )
            .where(Request.total_tokens.isnot(None))
            .group_by(Request.user_id)
            .subquery()
        )
        query = (
            select(User)
            .options(selectinload(User.group))
            .outerjoin(token_subq, User.id == token_subq.c.user_id)
            .outerjoin(Quota, User.id == Quota.user_id)
            .where(where_clause)
        )
        order_col = (
            func.coalesce(token_subq.c.total_tokens, 0)
            + func.coalesce(Quota.archived_total_tokens, 0)
        )
        query = query.order_by(order_col.asc() if asc else order_col.desc())
    elif sort_by == "last_login":
        query = (
            select(User)
            .options(selectinload(User.group))
            .where(where_clause)
        )
        # NULLS LAST for asc, NULLS FIRST for desc (so never-logged-in users go to end)
        if asc:
            query = query.order_by(
                case((User.last_login_at.is_(None), 1), else_=0),
                User.last_login_at.asc(),
            )
        else:
            query = query.order_by(
                case((User.last_login_at.is_(None), 1), else_=0),
                User.last_login_at.desc(),
            )
    elif sort_by == "name":
        query = (
            select(User)
            .options(selectinload(User.group))
            .where(where_clause)
            .order_by(User.username.asc() if asc else User.username.desc())
        )
    else:
        # Default: sort by created_at
        query = (
            select(User)
            .options(selectinload(User.group))
            .where(where_clause)
            .order_by(User.created_at.asc() if asc else User.created_at.desc())
        )

    query = query.offset(skip).limit(limit)
    result = await db.execute(query)
    return list(result.scalars().all()), total


async def get_user_token_totals(db: AsyncSession, user_ids: List[int]) -> dict:
    """Get token totals for a list of user IDs.

    Includes archived offsets from the quotas table so totals stay
    correct after retention deletes old requests.

    Returns {user_id: {prompt_tokens, completion_tokens, total_tokens}}.
    """
    if not user_ids:
        return {}

    # Live totals from requests table
    result = await db.execute(
        select(
            Request.user_id,
            func.coalesce(func.sum(Request.prompt_tokens), 0),
            func.coalesce(func.sum(Request.completion_tokens), 0),
            func.coalesce(func.sum(Request.total_tokens), 0),
        )
        .where(Request.user_id.in_(user_ids), Request.total_tokens.isnot(None))
        .group_by(Request.user_id)
    )
    totals = {
        row[0]: {
            "prompt_tokens": int(row[1]),
            "completion_tokens": int(row[2]),
            "total_tokens": int(row[3]),
        }
        for row in result.all()
    }

    # Add archived offsets
    arch_result = await db.execute(
        select(
            Quota.user_id,
            Quota.archived_prompt_tokens,
            Quota.archived_completion_tokens,
            Quota.archived_total_tokens,
        ).where(Quota.user_id.in_(user_ids))
    )
    for uid, a_prompt, a_comp, a_total in arch_result.all():
        if uid in totals:
            totals[uid]["prompt_tokens"] += int(a_prompt)
            totals[uid]["completion_tokens"] += int(a_comp)
            totals[uid]["total_tokens"] += int(a_total)
        elif a_total > 0:
            totals[uid] = {
                "prompt_tokens": int(a_prompt),
                "completion_tokens": int(a_comp),
                "total_tokens": int(a_total),
            }

    return totals


async def get_api_key_token_totals(
    db: AsyncSession, api_key_ids: List[int]
) -> dict:
    """Get token totals for a list of API key IDs.

    Merges live sums from ``requests`` with archived offsets stored on
    ``api_keys`` so that lifetime counters stay correct after retention
    deletes old rows.

    Returns {api_key_id: {prompt_tokens, completion_tokens, total_tokens}}.
    """
    if not api_key_ids:
        return {}

    # Live totals from requests table
    live_result = await db.execute(
        select(
            Request.api_key_id,
            func.coalesce(func.sum(Request.prompt_tokens), 0),
            func.coalesce(func.sum(Request.completion_tokens), 0),
            func.coalesce(func.sum(Request.total_tokens), 0),
        )
        .where(Request.api_key_id.in_(api_key_ids), Request.total_tokens.isnot(None))
        .group_by(Request.api_key_id)
    )
    live_map = {row[0]: (int(row[1]), int(row[2]), int(row[3])) for row in live_result.all()}

    # Archived offsets from api_keys
    arch_result = await db.execute(
        select(
            ApiKey.id,
            ApiKey.archived_prompt_tokens,
            ApiKey.archived_completion_tokens,
            ApiKey.archived_total_tokens,
        ).where(ApiKey.id.in_(api_key_ids))
    )

    merged: dict = {}
    for kid, a_p, a_c, a_t in arch_result.all():
        l_p, l_c, l_t = live_map.get(kid, (0, 0, 0))
        merged[kid] = {
            "prompt_tokens": l_p + int(a_p),
            "completion_tokens": l_c + int(a_c),
            "total_tokens": l_t + int(a_t),
        }
    return merged


async def get_model_token_totals(
    db: AsyncSession, limit: int = 20
) -> List[dict]:
    """Get token totals grouped by model, ordered by total descending.

    Includes archived offsets from model_archived_stats so totals stay
    correct after retention deletes old requests.

    Returns [{model, prompt_tokens, completion_tokens, total_tokens, request_count}].
    """
    # Live totals from requests table
    result = await db.execute(
        select(
            Request.model,
            func.coalesce(func.sum(Request.prompt_tokens), 0),
            func.coalesce(func.sum(Request.completion_tokens), 0),
            func.coalesce(func.sum(Request.total_tokens), 0),
            func.count(Request.id),
        )
        .where(Request.total_tokens.isnot(None))
        .group_by(Request.model)
    )
    by_model: dict[str, dict] = {}
    for row in result.all():
        by_model[row[0]] = {
            "model": row[0],
            "prompt_tokens": int(row[1]),
            "completion_tokens": int(row[2]),
            "total_tokens": int(row[3]),
            "request_count": int(row[4]),
        }

    # Merge archived offsets
    arch_result = await db.execute(select(ModelArchivedStats))
    for arch in arch_result.scalars().all():
        if arch.model in by_model:
            by_model[arch.model]["prompt_tokens"] += arch.archived_prompt_tokens
            by_model[arch.model]["completion_tokens"] += arch.archived_completion_tokens
            by_model[arch.model]["total_tokens"] += arch.archived_total_tokens
            by_model[arch.model]["request_count"] += arch.archived_request_count
        elif arch.archived_total_tokens > 0:
            by_model[arch.model] = {
                "model": arch.model,
                "prompt_tokens": arch.archived_prompt_tokens,
                "completion_tokens": arch.archived_completion_tokens,
                "total_tokens": arch.archived_total_tokens,
                "request_count": arch.archived_request_count,
            }

    # Sort by total_tokens descending, return top N
    sorted_models = sorted(by_model.values(), key=lambda m: m["total_tokens"], reverse=True)
    return sorted_models[:limit]


async def delete_user(db: AsyncSession, user_id: int) -> bool:
    """Hard-delete a user and all child rows (no CASCADE on FKs).

    Deletion order matters due to foreign key constraints:
    1. scheduler_decisions (FK -> requests)
    2. responses (FK -> requests)
    3. artifacts (FK -> requests)
    4. requests (FK -> users, api_keys)
    5. api_keys (FK -> users)
    6. quotas (FK -> users)
    7. quota_requests (FK -> users)
    8. users
    """
    # Get request IDs for this user (needed for child tables of requests)
    req_result = await db.execute(
        select(Request.id).where(Request.user_id == user_id)
    )
    request_ids = [r for (r,) in req_result.all()]

    if request_ids:
        # Delete children of requests
        await db.execute(
            delete(SchedulerDecision).where(
                SchedulerDecision.request_id.in_(request_ids)
            )
        )
        await db.execute(
            delete(Response).where(Response.request_id.in_(request_ids))
        )
        await db.execute(
            delete(Artifact).where(Artifact.request_id.in_(request_ids))
        )

    # Delete direct children of user
    await db.execute(
        delete(Request).where(Request.user_id == user_id)
    )
    await db.execute(
        delete(ApiKey).where(ApiKey.user_id == user_id)
    )
    await db.execute(
        delete(Quota).where(Quota.user_id == user_id)
    )
    await db.execute(
        delete(QuotaRequest).where(QuotaRequest.user_id == user_id)
    )

    # Delete the user
    result = await db.execute(
        delete(User).where(User.id == user_id)
    )
    await db.flush()
    return result.rowcount > 0


# API Key CRUD
async def get_api_key_by_hash(db: AsyncSession, key_hash: str) -> Optional[ApiKey]:
    """Get API key by hash with user and group eagerly loaded."""
    result = await db.execute(
        select(ApiKey)
        .options(selectinload(ApiKey.user).selectinload(User.group))
        .where(ApiKey.key_hash == key_hash)
    )
    return result.scalar_one_or_none()


async def get_api_key_by_prefix(db: AsyncSession, key_prefix: str) -> Optional[ApiKey]:
    """Get API key by prefix (for identification)."""
    result = await db.execute(
        select(ApiKey)
        .options(selectinload(ApiKey.user).selectinload(User.group))
        .where(ApiKey.key_prefix == key_prefix)
    )
    return result.scalar_one_or_none()


async def create_api_key(
    db: AsyncSession,
    user_id: int,
    key_hash: str,
    key_prefix: str,
    name: str,
    expires_at: Optional[datetime] = None,
) -> ApiKey:
    """Create a new API key."""
    api_key = ApiKey(
        user_id=user_id,
        key_hash=key_hash,
        key_prefix=key_prefix,
        name=name,
        expires_at=expires_at,
        status=ApiKeyStatus.ACTIVE,
    )
    db.add(api_key)
    await db.flush()
    return api_key


async def get_user_api_keys(
    db: AsyncSession, user_id: int, include_revoked: bool = False
) -> List[ApiKey]:
    """Get all API keys for a user."""
    query = select(ApiKey).where(ApiKey.user_id == user_id)
    if not include_revoked:
        query = query.where(ApiKey.status == ApiKeyStatus.ACTIVE)
    result = await db.execute(query)
    return list(result.scalars().all())


async def revoke_api_key(db: AsyncSession, api_key_id: int) -> Optional[ApiKey]:
    """Revoke an API key."""
    result = await db.execute(select(ApiKey).where(ApiKey.id == api_key_id))
    api_key = result.scalar_one_or_none()
    if api_key:
        api_key.status = ApiKeyStatus.REVOKED
        await db.flush()
    return api_key


async def get_api_key_by_id(db: AsyncSession, api_key_id: int) -> Optional[ApiKey]:
    """Get an API key by its primary key ID."""
    result = await db.execute(select(ApiKey).where(ApiKey.id == api_key_id))
    return result.scalar_one_or_none()


async def update_api_key_usage(db: AsyncSession, api_key_id: int) -> None:
    """Update API key last used timestamp and usage count.

    Uses an atomic UPDATE to avoid loading the row into the ORM session,
    which prevents holding an exclusive row lock for the entire request.
    """
    await db.execute(
        update(ApiKey)
        .where(ApiKey.id == api_key_id)
        .values(
            last_used_at=func.now(),
            usage_count=ApiKey.usage_count + 1,
        )
    )


# Quota CRUD
async def get_user_quota(db: AsyncSession, user_id: int) -> Optional[Quota]:
    """Get quota for a user."""
    result = await db.execute(select(Quota).where(Quota.user_id == user_id))
    return result.scalar_one_or_none()


async def create_quota(
    db: AsyncSession,
    user_id: int,
    rpm_limit: int,
) -> Quota:
    """Create quota for a user and initialise Redis counter to 0."""
    quota = Quota(
        user_id=user_id,
        rpm_limit=rpm_limit,
    )
    db.add(quota)
    await db.flush()

    # Ensure Redis starts at 0 — prevents inheriting stale keys from
    # a previous database that used the same user_id.
    from backend.app.core.redis_client import set_tokens, is_available
    if is_available():
        await set_tokens(user_id, 0)

    return quota


async def update_quota_usage(
    db: AsyncSession, user_id: int, tokens_used: int
) -> Optional[Quota]:
    """Update quota token usage in DB (Redis increment happens post-commit).

    When Redis is available, this only stages the DB quota object for return;
    the caller must call ``incr_quota_redis(user_id, tokens_used)`` **after**
    a successful ``db.commit()`` to avoid drift between Redis and the ledger.

    When Redis is unavailable, falls back to a direct DB increment that will
    be committed with the rest of the transaction.

    lifetime_tokens_used is always incremented in the DB regardless of Redis
    availability — it is a monotonic counter that never resets.
    """
    from backend.app.core.redis_client import is_available

    result = await db.execute(select(Quota).where(Quota.user_id == user_id))
    quota = result.scalar_one_or_none()
    if not quota:
        return None

    # Always increment lifetime counter in DB
    quota.lifetime_tokens_used += tokens_used

    if not is_available():
        # Fallback: also increment period counter in DB
        quota.tokens_used += tokens_used

    await db.flush()
    return quota


async def incr_quota_redis(user_id: int, tokens_used: int) -> None:
    """Increment quota counter in Redis.  Call only after db.commit() succeeds."""
    from backend.app.core.redis_client import incr_tokens, is_available

    if is_available():
        await incr_tokens(user_id, tokens_used)


async def reset_quota_if_needed(db: AsyncSession, user_id: int) -> Optional[Quota]:
    """Reset quota if period has expired."""
    result = await db.execute(select(Quota).where(Quota.user_id == user_id))
    quota = result.scalar_one_or_none()
    if quota:
        period_end = _ensure_aware(quota.budget_period_start) + timedelta(days=quota.budget_period_days)
        if datetime.now(timezone.utc) >= period_end:
            quota.budget_period_start = datetime.now(timezone.utc)
            quota.tokens_used = 0
            await db.flush()
            # Also reset Redis counter if available
            from backend.app.core.redis_client import reset_tokens, is_available
            if is_available():
                await reset_tokens(user_id)
    return quota


# Node CRUD
async def create_node(
    db: AsyncSession,
    name: str,
    hostname: Optional[str] = None,
    sidecar_url: Optional[str] = None,
    sidecar_key: Optional[str] = None,
) -> Node:
    """Create a new node."""
    node = Node(
        name=name,
        hostname=hostname,
        sidecar_url=sidecar_url,
        sidecar_key=sidecar_key,
        status=NodeStatus.UNKNOWN,
    )
    db.add(node)
    await db.flush()
    return node


async def get_node_by_id(db: AsyncSession, node_id: int) -> Optional[Node]:
    """Get node by ID."""
    result = await db.execute(select(Node).where(Node.id == node_id))
    return result.scalar_one_or_none()


async def get_node_by_name(db: AsyncSession, name: str) -> Optional[Node]:
    """Get node by name."""
    result = await db.execute(select(Node).where(Node.name == name))
    return result.scalar_one_or_none()


async def get_all_nodes(db: AsyncSession) -> List[Node]:
    """Get all nodes."""
    result = await db.execute(select(Node))
    return list(result.scalars().all())


async def update_node_hardware(
    db: AsyncSession,
    node_id: int,
    gpu_count: Optional[int] = None,
    driver_version: Optional[str] = None,
    cuda_version: Optional[str] = None,
    sidecar_version: Optional[str] = None,
    server_power_watts: Optional[int] = None,
) -> Optional[Node]:
    """Update node hardware info from sidecar."""
    result = await db.execute(select(Node).where(Node.id == node_id))
    node = result.scalar_one_or_none()
    if node:
        if gpu_count is not None:
            node.gpu_count = gpu_count
        if driver_version is not None:
            node.driver_version = driver_version
        if cuda_version is not None:
            node.cuda_version = cuda_version
        if sidecar_version is not None:
            node.sidecar_version = sidecar_version
        if server_power_watts is not None:
            node.server_power_watts = server_power_watts
        await db.flush()
    return node


async def update_node_status(
    db: AsyncSession,
    node_id: int,
    status: NodeStatus,
) -> Optional[Node]:
    """Update node status."""
    result = await db.execute(select(Node).where(Node.id == node_id))
    node = result.scalar_one_or_none()
    if node:
        node.status = status
        await db.flush()
    return node


async def delete_node(db: AsyncSession, node_id: int) -> bool:
    """Delete a node. Fails if backends still reference it."""
    # Check for backends referencing this node
    result = await db.execute(
        select(Backend).where(Backend.node_id == node_id).limit(1)
    )
    if result.scalar_one_or_none():
        return False

    # Delete GPU devices on this node (cascade deletes their telemetry)
    gpu_result = await db.execute(
        select(GPUDevice).where(GPUDevice.node_id == node_id)
    )
    for device in gpu_result.scalars().all():
        await db.delete(device)

    result = await db.execute(select(Node).where(Node.id == node_id))
    node = result.scalar_one_or_none()
    if node:
        await db.delete(node)
        await db.flush()
        return True
    return False


# Backend CRUD
async def get_backend_by_id(db: AsyncSession, backend_id: int) -> Optional[Backend]:
    """Get backend by ID."""
    result = await db.execute(
        select(Backend)
        .options(selectinload(Backend.models), selectinload(Backend.node))
        .where(Backend.id == backend_id)
    )
    return result.scalar_one_or_none()


async def get_backend_by_name(db: AsyncSession, name: str) -> Optional[Backend]:
    """Get backend by name."""
    result = await db.execute(select(Backend).where(Backend.name == name))
    return result.scalar_one_or_none()


async def get_backend_by_url(db: AsyncSession, url: str) -> Optional[Backend]:
    """Get backend by URL."""
    result = await db.execute(select(Backend).where(Backend.url == url))
    return result.scalar_one_or_none()


async def get_healthy_backends(
    db: AsyncSession, engine: Optional[BackendEngine] = None
) -> List[Backend]:
    """Get all healthy backends."""
    query = select(Backend).where(Backend.status == BackendStatus.HEALTHY)
    if engine:
        query = query.where(Backend.engine == engine)
    result = await db.execute(query.options(selectinload(Backend.models), selectinload(Backend.node)))
    return list(result.scalars().all())


async def get_all_backends(db: AsyncSession) -> List[Backend]:
    """Get all backends."""
    result = await db.execute(
        select(Backend).options(selectinload(Backend.models), selectinload(Backend.node))
    )
    return list(result.scalars().all())


async def create_backend(
    db: AsyncSession,
    name: str,
    url: str,
    engine: BackendEngine,
    max_concurrent: int = 4,
    gpu_memory_gb: Optional[float] = None,
    gpu_type: Optional[str] = None,
    node_id: Optional[int] = None,
    gpu_indices: Optional[list] = None,
) -> Backend:
    """Register a new backend."""
    backend = Backend(
        name=name,
        url=url,
        engine=engine,
        max_concurrent=max_concurrent,
        gpu_memory_gb=gpu_memory_gb,
        gpu_type=gpu_type,
        node_id=node_id,
        gpu_indices=gpu_indices,
        status=BackendStatus.UNKNOWN,
    )
    db.add(backend)
    await db.flush()
    return backend


async def update_backend_status(
    db: AsyncSession,
    backend_id: int,
    status: BackendStatus,
    version: Optional[str] = None,
) -> Optional[Backend]:
    """Update backend health status."""
    result = await db.execute(select(Backend).where(Backend.id == backend_id))
    backend = result.scalar_one_or_none()
    if backend:
        backend.status = status
        backend.last_health_check = datetime.now(timezone.utc)
        if version:
            backend.version = version
        if status == BackendStatus.HEALTHY:
            backend.consecutive_failures = 0
            backend.last_success = datetime.now(timezone.utc)
        else:
            backend.consecutive_failures += 1
        await db.flush()
    return backend


async def update_backend(
    db: AsyncSession,
    backend_id: int,
    name: Optional[str] = None,
    url: Optional[str] = None,
    engine: Optional[BackendEngine] = None,
    max_concurrent: Optional[int] = None,
    gpu_memory_gb: Optional[float] = None,
    gpu_type: Optional[str] = None,
    priority: Optional[int] = None,
    node_id: Optional[int] = None,
    gpu_indices: Optional[list] = None,
    _clear_fields: Optional[list] = None,
) -> Optional[Backend]:
    """Update editable fields on a backend.

    Only non-None kwargs are applied. To explicitly clear a nullable field,
    include its name in _clear_fields (e.g. _clear_fields=["gpu_memory_gb"]).
    """
    result = await db.execute(
        select(Backend)
        .options(selectinload(Backend.models), selectinload(Backend.node))
        .where(Backend.id == backend_id)
    )
    backend = result.scalar_one_or_none()
    if not backend:
        return None

    clear = set(_clear_fields or [])

    if name is not None:
        backend.name = name
    if url is not None:
        backend.url = url
    if engine is not None:
        backend.engine = engine
    if max_concurrent is not None:
        backend.max_concurrent = max_concurrent
    if gpu_memory_gb is not None:
        backend.gpu_memory_gb = gpu_memory_gb
    elif "gpu_memory_gb" in clear:
        backend.gpu_memory_gb = None
    if gpu_type is not None:
        backend.gpu_type = gpu_type
    elif "gpu_type" in clear:
        backend.gpu_type = None
    if priority is not None:
        backend.priority = priority
    if node_id is not None:
        backend.node_id = node_id
    elif "node_id" in clear:
        backend.node_id = None
    if gpu_indices is not None:
        backend.gpu_indices = gpu_indices
    elif "gpu_indices" in clear:
        backend.gpu_indices = None

    await db.flush()
    return backend


async def update_node(
    db: AsyncSession,
    node_id: int,
    name: Optional[str] = None,
    hostname: Optional[str] = None,
    sidecar_url: Optional[str] = None,
    sidecar_key: Optional[str] = None,
    _clear_fields: Optional[list] = None,
) -> Optional[Node]:
    """Update editable fields on a node.

    Only non-None kwargs are applied. To explicitly clear a nullable field,
    include its name in _clear_fields.
    """
    result = await db.execute(select(Node).where(Node.id == node_id))
    node = result.scalar_one_or_none()
    if not node:
        return None

    clear = set(_clear_fields or [])

    if name is not None:
        node.name = name
    if hostname is not None:
        node.hostname = hostname
    elif "hostname" in clear:
        node.hostname = None
    if sidecar_url is not None:
        node.sidecar_url = sidecar_url
    elif "sidecar_url" in clear:
        node.sidecar_url = None
    if sidecar_key is not None:
        node.sidecar_key = sidecar_key
    elif "sidecar_key" in clear:
        node.sidecar_key = None

    await db.flush()
    return node


async def delete_backend(db: AsyncSession, backend_id: int) -> bool:
    """Delete a backend and its associated data."""
    # Bulk-delete child rows with NOT NULL FK
    await db.execute(
        delete(BackendTelemetry).where(BackendTelemetry.backend_id == backend_id)
    )
    await db.execute(
        delete(Model).where(Model.backend_id == backend_id)
    )
    await db.execute(
        delete(SchedulerDecision).where(SchedulerDecision.selected_backend_id == backend_id)
    )

    # Null out nullable FK references (preserve request history)
    await db.execute(
        update(Request).where(Request.backend_id == backend_id).values(backend_id=None)
    )

    # Delete the backend
    result = await db.execute(select(Backend).where(Backend.id == backend_id))
    backend = result.scalar_one_or_none()
    if backend:
        await db.delete(backend)
        await db.flush()
        return True
    return False


async def update_backend_concurrency(
    db: AsyncSession, backend_id: int, delta: int
) -> Optional[Backend]:
    """Update backend concurrent request count."""
    result = await db.execute(select(Backend).where(Backend.id == backend_id))
    backend = result.scalar_one_or_none()
    if backend:
        backend.current_concurrent = max(0, backend.current_concurrent + delta)
        await db.flush()
    return backend


async def update_backend_latency_ema(
    db: AsyncSession,
    backend_id: int,
    latency_ema_ms: Optional[float],
    ttft_ema_ms: Optional[float],
    throughput_score: float,
) -> None:
    """Persist latency EMA and derived throughput_score to the backend row."""
    await db.execute(
        update(Backend)
        .where(Backend.id == backend_id)
        .values(
            latency_ema_ms=latency_ema_ms,
            ttft_ema_ms=ttft_ema_ms,
            throughput_score=throughput_score,
        )
    )


async def update_backend_circuit_breaker(
    db: AsyncSession,
    backend_id: int,
    live_failure_count: int,
    circuit_open_until: Optional[datetime] = None,
) -> None:
    """Persist circuit breaker state."""
    await db.execute(
        update(Backend)
        .where(Backend.id == backend_id)
        .values(
            live_failure_count=live_failure_count,
            circuit_open_until=circuit_open_until,
        )
    )


# Model CRUD
async def mark_model_loaded(db: AsyncSession, backend_id: int, model_name: str) -> None:
    """Mark a model as loaded on a backend (called after successful inference).

    This bridges the gap between 30-second discovery polls so the scheduler
    knows the model is warm and can prefer this backend for the next request.
    """
    result = await db.execute(
        select(Model).where(
            and_(Model.backend_id == backend_id, Model.name == model_name)
        )
    )
    model = result.scalar_one_or_none()
    if model and not model.is_loaded:
        model.is_loaded = True
        await db.flush()


async def get_models_for_backend(db: AsyncSession, backend_id: int) -> List[Model]:
    """Get all models for a backend."""
    result = await db.execute(
        select(Model).where(Model.backend_id == backend_id)
    )
    return list(result.scalars().all())


async def get_backends_with_model(
    db: AsyncSession,
    model_name: str,
    modality: Optional[Modality] = None,
) -> List[Backend]:
    """Get backends that have a specific model."""
    query = (
        select(Backend)
        .join(Model)
        .where(
            and_(
                Model.name == model_name,
                Backend.status == BackendStatus.HEALTHY,
            )
        )
    )
    if modality:
        query = query.where(Model.modality == modality)
    result = await db.execute(query.options(selectinload(Backend.models)))
    return list(result.scalars().all())


async def upsert_model(
    db: AsyncSession,
    backend_id: int,
    name: str,
    modality: Modality = Modality.CHAT,
    context_length: Optional[int] = None,
    supports_multimodal: bool = False,
    supports_thinking: bool = False,
    supports_tools: bool = False,
    supports_structured_output: bool = True,
    is_loaded: bool = False,
    quantization: Optional[str] = None,
    model_format: Optional[str] = None,
    capabilities_json: Optional[str] = None,
    embedding_length: Optional[int] = None,
    head_count: Optional[int] = None,
    layer_count: Optional[int] = None,
    feed_forward_length: Optional[int] = None,
    parent_model: Optional[str] = None,
    model_max_context: Optional[int] = None,
) -> Model:
    """Create or update a model record.

    Uses SELECT FOR UPDATE to prevent race conditions where concurrent
    discovery cycles could create duplicate rows for the same
    (backend_id, name) pair.
    """
    result = await db.execute(
        select(Model).where(
            and_(Model.backend_id == backend_id, Model.name == name)
        ).with_for_update()
    )
    model = result.scalar_one_or_none()

    # If admin has set an override, use it instead of auto-detected value.
    # Check BOTH this instance's override AND any sibling instance with the
    # same model name — model capabilities like multimodal are inherent to
    # the model itself, not the backend serving it.
    effective_multimodal = supports_multimodal
    effective_thinking = supports_thinking
    effective_tools = supports_tools

    if model:
        if model.multimodal_override is not None:
            effective_multimodal = model.multimodal_override
        if model.thinking_override is not None:
            effective_thinking = model.thinking_override
        if model.tools_override is not None:
            effective_tools = model.tools_override

    # Inherit overrides from sibling instances of the same model name
    # on other backends (model properties are model-level, not backend-level)
    if not (model and model.multimodal_override is not None):
        sibling_result = await db.execute(
            select(Model).where(
                and_(Model.name == name, Model.backend_id != backend_id)
            )
        )
        for sibling in sibling_result.scalars().all():
            if sibling.multimodal_override is not None:
                effective_multimodal = sibling.multimodal_override
                break
            if sibling.supports_multimodal and not effective_multimodal:
                effective_multimodal = True

    if model:
        # UPDATE existing model record
        # Preserve existing non-default modality if discovery passes the
        # default CHAT — prevents overwriting admin-set modalities like
        # IMAGE_GENERATION that discovery heuristics don't detect.
        if modality != Modality.CHAT or model.modality == Modality.CHAT:
            model.modality = modality
        model.context_length = model.context_length_override if model.context_length_override is not None else (context_length if context_length is not None else model.context_length)
        model.model_max_context = model_max_context if model_max_context is not None else model.model_max_context
        model.supports_multimodal = effective_multimodal
        model.supports_thinking = effective_thinking
        model.supports_tools = effective_tools
        model.supports_structured_output = supports_structured_output
        model.is_loaded = is_loaded
        model.quantization = model.quantization_override if model.quantization_override is not None else quantization
        model.model_format = model.model_format_override if model.model_format_override is not None else model_format
        model.capabilities = model.capabilities_override if model.capabilities_override is not None else capabilities_json
        model.embedding_length = model.embedding_length_override if model.embedding_length_override is not None else embedding_length
        model.head_count = model.head_count_override if model.head_count_override is not None else head_count
        model.layer_count = model.layer_count_override if model.layer_count_override is not None else layer_count
        model.feed_forward_length = model.feed_forward_length_override if model.feed_forward_length_override is not None else feed_forward_length
        model.parent_model = model.parent_model_override if model.parent_model_override is not None else parent_model
        model.family = model.family_override if model.family_override is not None else model.family
        model.parameter_count = model.parameter_count_override if model.parameter_count_override is not None else model.parameter_count
    else:
        # INSERT new model record
        model = Model(
            backend_id=backend_id,
            name=name,
            modality=modality,
            context_length=context_length,
            model_max_context=model_max_context,
            supports_multimodal=effective_multimodal,
            supports_thinking=effective_thinking,
            supports_tools=effective_tools,
            supports_structured_output=supports_structured_output,
            is_loaded=is_loaded,
            quantization=quantization,
            model_format=model_format,
            capabilities=capabilities_json,
            embedding_length=embedding_length,
            head_count=head_count,
            layer_count=layer_count,
            feed_forward_length=feed_forward_length,
            parent_model=parent_model,
        )
        db.add(model)

    try:
        await db.flush()
    except Exception as exc:
        # Handle race condition: another thread inserted the same row
        # between our SELECT and INSERT. Roll back and retry as UPDATE.
        if "Duplicate entry" in str(exc) or "IntegrityError" in type(exc).__name__:
            await db.rollback()
            result = await db.execute(
                select(Model).where(
                    and_(Model.backend_id == backend_id, Model.name == name)
                ).with_for_update()
            )
            model = result.scalar_one_or_none()
            if model:
                model.context_length = context_length if context_length is not None else model.context_length
                model.is_loaded = is_loaded
                await db.flush()
                return model
        raise
    return model


async def get_model_by_id(db: AsyncSession, model_id: int) -> Optional[Model]:
    """Get a model by its ID."""
    result = await db.execute(select(Model).where(Model.id == model_id))
    return result.scalar_one_or_none()


async def set_model_multimodal_override(
    db: AsyncSession, model_id: int, value: Optional[bool]
) -> Optional[Model]:
    """Set the multimodal override for a model.

    When value is True/False, admin's choice sticks regardless of auto-detect.
    When value is None, auto-detection controls the value.

    Propagates to all instances of the same model name across all backends,
    since multimodal is a model-level property, not backend-level.
    """
    model = await get_model_by_id(db, model_id)
    if not model:
        return None
    # Propagate to all instances with the same name
    await set_multimodal_override_by_name(db, model.name, value)
    return model


async def get_all_models_with_backends(db: AsyncSession) -> list[Model]:
    """Get all models with their backend relationship eagerly loaded."""
    result = await db.execute(
        select(Model).options(selectinload(Model.backend)).order_by(Model.name)
    )
    return list(result.scalars().all())


async def get_models_grouped_by_name(db: AsyncSession) -> dict:
    """Get all models grouped by name, each with backend eagerly loaded."""
    result = await db.execute(
        select(Model).options(selectinload(Model.backend)).order_by(Model.name)
    )
    models = list(result.scalars().all())
    grouped: dict = {}
    for m in models:
        grouped.setdefault(m.name, []).append(m)
    return grouped


async def set_multimodal_override_by_name(
    db: AsyncSession, model_name: str, value: Optional[bool]
) -> int:
    """Set multimodal override for ALL model rows with the given name."""
    result = await db.execute(select(Model).where(Model.name == model_name))
    models = list(result.scalars().all())
    for m in models:
        m.multimodal_override = value
        if value is not None:
            m.supports_multimodal = value
    await db.flush()
    return len(models)


async def set_thinking_override_by_name(
    db: AsyncSession, model_name: str, value: Optional[bool]
) -> int:
    """Set thinking override for ALL model rows with the given name."""
    result = await db.execute(select(Model).where(Model.name == model_name))
    models = list(result.scalars().all())
    for m in models:
        m.thinking_override = value
        if value is not None:
            m.supports_thinking = value
    await db.flush()
    return len(models)


async def set_tools_override_by_name(
    db: AsyncSession, model_name: str, value: Optional[bool]
) -> int:
    """Set tools override for ALL model rows with the given name."""
    result = await db.execute(select(Model).where(Model.name == model_name))
    models = list(result.scalars().all())
    for m in models:
        m.tools_override = value
        if value is not None:
            m.supports_tools = value
    await db.flush()
    return len(models)


async def update_model_overrides_by_name(
    db: AsyncSession, model_name: str, overrides: dict
) -> int:
    """Set metadata overrides for ALL model rows with the given name.

    Accepts a dict mapping override field names to values.
    A value of None clears the override (back to auto-detect).
    Also applies the effective value to the base field when setting an override.
    """
    # Map override field -> base field
    override_to_base = {
        "context_length_override": "context_length",
        "embedding_length_override": "embedding_length",
        "head_count_override": "head_count",
        "layer_count_override": "layer_count",
        "feed_forward_length_override": "feed_forward_length",
        "capabilities_override": "capabilities",
        "family_override": "family",
        "parameter_count_override": "parameter_count",
        "quantization_override": "quantization",
        "model_format_override": "model_format",
        "parent_model_override": "parent_model",
    }

    # Direct fields (admin-only, no auto-detect source)
    direct_fields = {"description", "model_url", "huggingface_url"}

    result = await db.execute(select(Model).where(Model.name == model_name))
    models = list(result.scalars().all())
    for m in models:
        for field, value in overrides.items():
            if field in direct_fields:
                if hasattr(m, field):
                    setattr(m, field, value)
            elif hasattr(m, field):
                setattr(m, field, value)
                # Also set the base field so the display value updates immediately
                base_field = override_to_base.get(field)
                if base_field and value is not None:
                    setattr(m, base_field, value)
    await db.flush()
    return len(models)


async def remove_stale_models(
    db: AsyncSession, backend_id: int, current_model_names: List[str]
) -> int:
    """Remove models for a backend that are no longer discovered."""
    if not current_model_names:
        return 0
    result = await db.execute(
        delete(Model).where(
            and_(
                Model.backend_id == backend_id,
                Model.name.notin_(current_model_names),
            )
        )
    )
    await db.flush()
    return result.rowcount


async def get_models_needing_enrichment(
    db: AsyncSession, limit: int = 3
) -> List[Model]:
    """Get one Model row per unique model name where description is NULL.

    Returns at most ``limit`` models, picking one representative row per
    distinct name so the enrichment loop does not process the same model
    name multiple times in a single batch.
    """
    # Subquery: one model id per distinct name where description IS NULL
    subq = (
        select(func.min(Model.id).label("mid"))
        .where(Model.description.is_(None))
        .group_by(Model.name)
        .limit(limit)
        .subquery()
    )
    result = await db.execute(
        select(Model).where(Model.id.in_(select(subq.c.mid)))
    )
    return list(result.scalars().all())


async def set_model_description_by_name(
    db: AsyncSession, model_name: str, description: str
) -> int:
    """Set description on ALL Model rows matching the given name."""
    result = await db.execute(
        update(Model)
        .where(Model.name == model_name)
        .values(description=description)
    )
    await db.flush()
    return result.rowcount


async def get_cached_description(db: AsyncSession, model_name: str) -> Optional[str]:
    """Look up a cached description for a model name."""
    result = await db.execute(
        select(ModelDescriptionCache.description)
        .where(ModelDescriptionCache.name == model_name)
    )
    row = result.scalar_one_or_none()
    return row


async def set_cached_description(db: AsyncSession, model_name: str, description: str) -> None:
    """Upsert a description into the model description cache."""
    result = await db.execute(
        select(ModelDescriptionCache).where(ModelDescriptionCache.name == model_name)
    )
    existing = result.scalar_one_or_none()
    if existing:
        existing.description = description
    else:
        db.add(ModelDescriptionCache(name=model_name, description=description))
    await db.flush()


async def get_all_available_models(db: AsyncSession) -> List[Tuple[str, List[Backend]]]:
    """Get all unique model names with their available backends."""
    result = await db.execute(
        select(Model.name, Backend)
        .join(Backend)
        .where(Backend.status == BackendStatus.HEALTHY)
        .order_by(Model.name)
    )
    # Group by model name
    models_dict: dict[str, List[Backend]] = {}
    for row in result.all():
        model_name, backend = row
        if model_name not in models_dict:
            models_dict[model_name] = []
        models_dict[model_name].append(backend)
    return list(models_dict.items())


# Request/Response CRUD
async def create_request(
    db: AsyncSession,
    user_id: int,
    api_key_id: int,
    endpoint: str,
    model: str,
    modality: Modality,
    is_streaming: bool = False,
    messages: Optional[dict] = None,
    prompt: Optional[str] = None,
    parameters: Optional[dict] = None,
    response_format: Optional[dict] = None,
    client_ip: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> Request:
    """Create a new request record."""
    request = Request(
        user_id=user_id,
        api_key_id=api_key_id,
        endpoint=endpoint,
        model=model,
        modality=modality,
        is_streaming=is_streaming,
        messages=messages,
        prompt=prompt,
        parameters=parameters,
        response_format=response_format,
        client_ip=client_ip,
        user_agent=user_agent,
        status=RequestStatus.QUEUED,
    )
    db.add(request)
    await db.flush()
    return request


async def update_request_started(
    db: AsyncSession, request_id: int, backend_id: int
) -> Optional[Request]:
    """Update request when processing starts."""
    result = await db.execute(select(Request).where(Request.id == request_id))
    request = result.scalar_one_or_none()
    if request:
        request.status = RequestStatus.PROCESSING
        request.backend_id = backend_id
        request.started_at = datetime.now(timezone.utc)
        queue_delay = request.started_at - _ensure_aware(request.queued_at)
        request.queue_delay_ms = int(queue_delay.total_seconds() * 1000)
        await db.flush()
    return request


async def update_request_completed(
    db: AsyncSession,
    request_id: int,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    tokens_estimated: bool = False,
    backend_id: Optional[int] = None,
) -> Optional[Request]:
    """Update request when completed."""
    result = await db.execute(select(Request).where(Request.id == request_id))
    request = result.scalar_one_or_none()
    if request:
        request.status = RequestStatus.COMPLETED
        request.completed_at = datetime.now(timezone.utc)
        if backend_id is not None:
            request.backend_id = backend_id
        if request.started_at:
            processing_time = request.completed_at - _ensure_aware(request.started_at)
            request.processing_time_ms = int(processing_time.total_seconds() * 1000)
        total_time = request.completed_at - _ensure_aware(request.queued_at)
        request.total_time_ms = int(total_time.total_seconds() * 1000)
        request.prompt_tokens = prompt_tokens
        request.completion_tokens = completion_tokens
        if prompt_tokens and completion_tokens:
            request.total_tokens = prompt_tokens + completion_tokens
        request.tokens_estimated = tokens_estimated
        await db.flush()
    return request


async def update_request_failed(
    db: AsyncSession, request_id: int, error_message: str, error_code: Optional[str] = None
) -> Optional[Request]:
    """Update request when failed."""
    result = await db.execute(select(Request).where(Request.id == request_id))
    request = result.scalar_one_or_none()
    if request:
        request.status = RequestStatus.FAILED
        request.completed_at = datetime.now(timezone.utc)
        request.error_message = error_message
        request.error_code = error_code
        if request.started_at:
            processing_time = request.completed_at - _ensure_aware(request.started_at)
            request.processing_time_ms = int(processing_time.total_seconds() * 1000)
        total_time = request.completed_at - _ensure_aware(request.queued_at)
        request.total_time_ms = int(total_time.total_seconds() * 1000)
        await db.flush()
    return request


async def create_response(
    db: AsyncSession,
    request_id: int,
    content: Optional[str] = None,
    finish_reason: Optional[str] = None,
    chunk_count: int = 0,
    first_token_time_ms: Optional[int] = None,
    structured_output_valid: Optional[bool] = None,
    validation_errors: Optional[list] = None,
    raw_response: Optional[dict] = None,
) -> Response:
    """Create a response record."""
    response = Response(
        request_id=request_id,
        content=content,
        finish_reason=finish_reason,
        chunk_count=chunk_count,
        first_token_time_ms=first_token_time_ms,
        structured_output_valid=structured_output_valid,
        validation_errors=validation_errors,
        raw_response=raw_response,
    )
    db.add(response)
    await db.flush()
    return response


async def get_user_usage_in_window(
    db: AsyncSession, user_id: int, window_seconds: int
) -> int:
    """Get total tokens used by user in a time window."""
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=window_seconds)
    result = await db.execute(
        select(func.coalesce(func.sum(Request.total_tokens), 0))
        .where(
            and_(
                Request.user_id == user_id,
                Request.created_at >= cutoff,
                Request.total_tokens.isnot(None),
            )
        )
    )
    return result.scalar_one() or 0


# Telemetry CRUD
async def create_telemetry_snapshot(
    db: AsyncSession,
    backend_id: int,
    gpu_utilization: Optional[float] = None,
    gpu_memory_used_gb: Optional[float] = None,
    gpu_memory_total_gb: Optional[float] = None,
    active_requests: int = 0,
    queued_requests: int = 0,
    loaded_models: Optional[list] = None,
    gpu_power_draw_watts: Optional[float] = None,
    gpu_fan_speed_percent: Optional[float] = None,
    gpu_temperature: Optional[float] = None,
) -> BackendTelemetry:
    """Create a telemetry snapshot."""
    telemetry = BackendTelemetry(
        backend_id=backend_id,
        gpu_utilization=gpu_utilization,
        gpu_memory_used_gb=gpu_memory_used_gb,
        gpu_memory_total_gb=gpu_memory_total_gb,
        gpu_temperature=gpu_temperature,
        gpu_power_draw_watts=gpu_power_draw_watts,
        gpu_fan_speed_percent=gpu_fan_speed_percent,
        active_requests=active_requests,
        queued_requests=queued_requests,
        loaded_models=loaded_models,
    )
    db.add(telemetry)
    await db.flush()
    return telemetry


async def get_latest_telemetry(
    db: AsyncSession, backend_id: int
) -> Optional[BackendTelemetry]:
    """Get latest telemetry snapshot for a backend."""
    result = await db.execute(
        select(BackendTelemetry)
        .where(BackendTelemetry.backend_id == backend_id)
        .order_by(BackendTelemetry.timestamp.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


# GPU Device CRUD
async def upsert_gpu_device(
    db: AsyncSession,
    node_id: int,
    gpu_index: int,
    uuid: Optional[str] = None,
    name: Optional[str] = None,
    pci_bus_id: Optional[str] = None,
    compute_capability: Optional[str] = None,
    memory_total_gb: Optional[float] = None,
    power_limit_watts: Optional[float] = None,
) -> GPUDevice:
    """Create or update a GPU device record."""
    result = await db.execute(
        select(GPUDevice).where(
            and_(GPUDevice.node_id == node_id, GPUDevice.gpu_index == gpu_index)
        )
    )
    device = result.scalar_one_or_none()

    if device:
        if uuid is not None:
            device.uuid = uuid
        if name is not None:
            device.name = name
        if pci_bus_id is not None:
            device.pci_bus_id = pci_bus_id
        if compute_capability is not None:
            device.compute_capability = compute_capability
        if memory_total_gb is not None:
            device.memory_total_gb = memory_total_gb
        if power_limit_watts is not None:
            device.power_limit_watts = power_limit_watts
    else:
        device = GPUDevice(
            node_id=node_id,
            gpu_index=gpu_index,
            uuid=uuid,
            name=name,
            pci_bus_id=pci_bus_id,
            compute_capability=compute_capability,
            memory_total_gb=memory_total_gb,
            power_limit_watts=power_limit_watts,
        )
        db.add(device)

    await db.flush()
    return device


async def create_gpu_device_telemetry(
    db: AsyncSession,
    gpu_device_id: int,
    utilization_gpu: Optional[float] = None,
    utilization_memory: Optional[float] = None,
    memory_used_gb: Optional[float] = None,
    memory_free_gb: Optional[float] = None,
    temperature_gpu: Optional[float] = None,
    temperature_memory: Optional[float] = None,
    power_draw_watts: Optional[float] = None,
    fan_speed_percent: Optional[float] = None,
    clock_sm_mhz: Optional[int] = None,
    clock_memory_mhz: Optional[int] = None,
) -> GPUDeviceTelemetry:
    """Create a per-GPU telemetry snapshot."""
    telemetry = GPUDeviceTelemetry(
        gpu_device_id=gpu_device_id,
        utilization_gpu=utilization_gpu,
        utilization_memory=utilization_memory,
        memory_used_gb=memory_used_gb,
        memory_free_gb=memory_free_gb,
        temperature_gpu=temperature_gpu,
        temperature_memory=temperature_memory,
        power_draw_watts=power_draw_watts,
        fan_speed_percent=fan_speed_percent,
        clock_sm_mhz=clock_sm_mhz,
        clock_memory_mhz=clock_memory_mhz,
    )
    db.add(telemetry)
    await db.flush()
    return telemetry


async def get_gpu_devices_for_node(
    db: AsyncSession, node_id: int
) -> List[GPUDevice]:
    """Get all GPU devices for a node."""
    result = await db.execute(
        select(GPUDevice)
        .where(GPUDevice.node_id == node_id)
        .order_by(GPUDevice.gpu_index)
    )
    return list(result.scalars().all())


async def get_gpu_devices_for_backend(
    db: AsyncSession, backend_id: int
) -> List[GPUDevice]:
    """Get GPU devices assigned to a backend (via its node + gpu_indices)."""
    # Look up the backend's node_id and gpu_indices
    result = await db.execute(
        select(Backend.node_id, Backend.gpu_indices).where(Backend.id == backend_id)
    )
    row = result.one_or_none()
    if not row or not row[0]:
        return []

    node_id, gpu_indices = row

    query = (
        select(GPUDevice)
        .where(GPUDevice.node_id == node_id)
        .order_by(GPUDevice.gpu_index)
    )
    if gpu_indices:
        query = query.where(GPUDevice.gpu_index.in_(gpu_indices))

    result = await db.execute(query)
    return list(result.scalars().all())


async def get_all_gpu_devices(db: AsyncSession) -> List[GPUDevice]:
    """Get all GPU devices across all nodes."""
    result = await db.execute(
        select(GPUDevice).order_by(GPUDevice.node_id, GPUDevice.gpu_index)
    )
    return list(result.scalars().all())


async def get_backend_telemetry_history(
    db: AsyncSession,
    backend_id: int,
    start: datetime,
    end: datetime,
    resolution_minutes: int = 5,
) -> List[dict]:
    """Get aggregated backend telemetry history with time bucketing."""
    from sqlalchemy import text

    query = text("""
        SELECT
            DATE_FORMAT(timestamp, :bucket_format) as time_bucket,
            AVG(gpu_utilization) as avg_gpu_utilization,
            MIN(gpu_utilization) as min_gpu_utilization,
            MAX(gpu_utilization) as max_gpu_utilization,
            AVG(gpu_memory_used_gb) as avg_gpu_memory_used_gb,
            AVG(gpu_memory_total_gb) as avg_gpu_memory_total_gb,
            AVG(gpu_temperature) as avg_gpu_temperature,
            AVG(gpu_power_draw_watts) as avg_gpu_power_draw_watts,
            AVG(active_requests) as avg_active_requests,
            AVG(queued_requests) as avg_queued_requests,
            AVG(requests_per_second) as avg_requests_per_second
        FROM backend_telemetry
        WHERE backend_id = :backend_id
          AND timestamp >= :start
          AND timestamp <= :end
        GROUP BY time_bucket
        ORDER BY time_bucket
    """)

    # Choose bucket format based on resolution
    if resolution_minutes <= 1:
        bucket_format = "%Y-%m-%d %H:%i"
    elif resolution_minutes <= 5:
        # Round to 5-minute intervals
        bucket_format = "%Y-%m-%d %H:"
        # Use a more complex expression for 5-min bucketing
        query = text("""
            SELECT
                CONCAT(DATE_FORMAT(timestamp, '%Y-%m-%d %H:'),
                       LPAD(FLOOR(MINUTE(timestamp) / :res_min) * :res_min, 2, '0')) as time_bucket,
                AVG(gpu_utilization) as avg_gpu_utilization,
                MIN(gpu_utilization) as min_gpu_utilization,
                MAX(gpu_utilization) as max_gpu_utilization,
                AVG(gpu_memory_used_gb) as avg_gpu_memory_used_gb,
                AVG(gpu_memory_total_gb) as avg_gpu_memory_total_gb,
                AVG(gpu_temperature) as avg_gpu_temperature,
                AVG(gpu_power_draw_watts) as avg_gpu_power_draw_watts,
                AVG(active_requests) as avg_active_requests,
                AVG(queued_requests) as avg_queued_requests,
                AVG(requests_per_second) as avg_requests_per_second
            FROM backend_telemetry
            WHERE backend_id = :backend_id
              AND timestamp >= :start
              AND timestamp <= :end
            GROUP BY time_bucket
            ORDER BY time_bucket
        """)
    elif resolution_minutes <= 60:
        bucket_format = "%Y-%m-%d %H:00"
    else:
        bucket_format = "%Y-%m-%d"

    params = {
        "backend_id": backend_id,
        "start": start,
        "end": end,
        "bucket_format": bucket_format,
        "res_min": resolution_minutes,
    }

    result = await db.execute(query, params)
    rows = result.mappings().all()

    return [
        {
            "timestamp": row["time_bucket"],
            "gpu_utilization": row["avg_gpu_utilization"],
            "gpu_utilization_min": row["min_gpu_utilization"],
            "gpu_utilization_max": row["max_gpu_utilization"],
            "gpu_memory_used_gb": row["avg_gpu_memory_used_gb"],
            "gpu_memory_total_gb": row["avg_gpu_memory_total_gb"],
            "gpu_temperature": row["avg_gpu_temperature"],
            "gpu_power_draw_watts": row["avg_gpu_power_draw_watts"],
            "active_requests": row["avg_active_requests"],
            "queued_requests": row["avg_queued_requests"],
            "requests_per_second": row["avg_requests_per_second"],
        }
        for row in rows
    ]


async def get_gpu_device_telemetry_history(
    db: AsyncSession,
    gpu_device_id: int,
    start: datetime,
    end: datetime,
    resolution_minutes: int = 5,
) -> List[dict]:
    """Get aggregated per-GPU telemetry history with time bucketing."""
    from sqlalchemy import text

    query = text("""
        SELECT
            CONCAT(DATE_FORMAT(timestamp, '%Y-%m-%d %H:'),
                   LPAD(FLOOR(MINUTE(timestamp) / :res_min) * :res_min, 2, '0')) as time_bucket,
            AVG(utilization_gpu) as avg_utilization_gpu,
            MIN(utilization_gpu) as min_utilization_gpu,
            MAX(utilization_gpu) as max_utilization_gpu,
            AVG(utilization_memory) as avg_utilization_memory,
            AVG(memory_used_gb) as avg_memory_used_gb,
            AVG(memory_free_gb) as avg_memory_free_gb,
            AVG(temperature_gpu) as avg_temperature_gpu,
            AVG(temperature_memory) as avg_temperature_memory,
            AVG(power_draw_watts) as avg_power_draw_watts,
            AVG(fan_speed_percent) as avg_fan_speed_percent,
            AVG(clock_sm_mhz) as avg_clock_sm_mhz,
            AVG(clock_memory_mhz) as avg_clock_memory_mhz
        FROM gpu_device_telemetry
        WHERE gpu_device_id = :gpu_device_id
          AND timestamp >= :start
          AND timestamp <= :end
        GROUP BY time_bucket
        ORDER BY time_bucket
    """)

    result = await db.execute(query, {
        "gpu_device_id": gpu_device_id,
        "start": start,
        "end": end,
        "res_min": resolution_minutes,
    })
    rows = result.mappings().all()

    return [
        {
            "timestamp": row["time_bucket"],
            "utilization_gpu": row["avg_utilization_gpu"],
            "utilization_gpu_min": row["min_utilization_gpu"],
            "utilization_gpu_max": row["max_utilization_gpu"],
            "utilization_memory": row["avg_utilization_memory"],
            "memory_used_gb": row["avg_memory_used_gb"],
            "memory_free_gb": row["avg_memory_free_gb"],
            "temperature_gpu": row["avg_temperature_gpu"],
            "temperature_memory": row["avg_temperature_memory"],
            "power_draw_watts": row["avg_power_draw_watts"],
            "fan_speed_percent": row["avg_fan_speed_percent"],
            "clock_sm_mhz": row["avg_clock_sm_mhz"],
            "clock_memory_mhz": row["avg_clock_memory_mhz"],
        }
        for row in rows
    ]


async def get_latest_gpu_device_telemetry(
    db: AsyncSession, node_id: int
) -> List[GPUDeviceTelemetry]:
    """Get the most recent telemetry for each GPU device on a node."""
    # Get device IDs for this node
    device_result = await db.execute(
        select(GPUDevice.id).where(GPUDevice.node_id == node_id)
    )
    device_ids = [row[0] for row in device_result.all()]

    if not device_ids:
        return []

    # Get max timestamp per device
    from sqlalchemy import text
    placeholders = ",".join(str(d) for d in device_ids)
    subq = text(f"""
        SELECT gpu_device_id, MAX(timestamp) as max_ts
        FROM gpu_device_telemetry
        WHERE gpu_device_id IN ({placeholders})
        GROUP BY gpu_device_id
    """)

    result = await db.execute(subq)
    latest_map = {row[0]: row[1] for row in result.all()}

    if not latest_map:
        return []

    # Fetch the actual rows
    conditions = []
    for device_id, max_ts in latest_map.items():
        conditions.append(
            and_(
                GPUDeviceTelemetry.gpu_device_id == device_id,
                GPUDeviceTelemetry.timestamp == max_ts,
            )
        )

    result = await db.execute(
        select(GPUDeviceTelemetry).where(or_(*conditions))
    )
    return list(result.scalars().all())


async def delete_old_telemetry(
    db: AsyncSession, older_than: datetime
) -> int:
    """Delete backend telemetry data older than the given datetime."""
    result = await db.execute(
        delete(BackendTelemetry).where(BackendTelemetry.timestamp < older_than)
    )
    await db.flush()
    return result.rowcount


async def delete_old_gpu_telemetry(
    db: AsyncSession, older_than: datetime
) -> int:
    """Delete per-GPU telemetry data older than the given datetime."""
    result = await db.execute(
        delete(GPUDeviceTelemetry).where(GPUDeviceTelemetry.timestamp < older_than)
    )
    await db.flush()
    return result.rowcount


async def create_node_telemetry(
    db: AsyncSession,
    node_id: int,
    server_power_watts: Optional[int] = None,
    gpu_power_watts: Optional[float] = None,
) -> NodeTelemetry:
    """Insert a node-level telemetry snapshot (server + GPU power)."""
    entry = NodeTelemetry(
        node_id=node_id,
        server_power_watts=server_power_watts,
        gpu_power_watts=gpu_power_watts,
    )
    db.add(entry)
    await db.flush()
    return entry


async def get_node_telemetry_history(
    db: AsyncSession,
    node_id: int,
    start: datetime,
    end: datetime,
    resolution_minutes: int = 5,
) -> List[dict]:
    """Get aggregated node telemetry history with time bucketing."""
    from sqlalchemy import text

    query = text("""
        SELECT
            CONCAT(DATE_FORMAT(timestamp, '%Y-%m-%d %H:'),
                   LPAD(FLOOR(MINUTE(timestamp) / :res_min) * :res_min, 2, '0')) as time_bucket,
            AVG(server_power_watts) as avg_server_power_watts,
            MIN(server_power_watts) as min_server_power_watts,
            MAX(server_power_watts) as max_server_power_watts,
            AVG(gpu_power_watts) as avg_gpu_power_watts,
            MIN(gpu_power_watts) as min_gpu_power_watts,
            MAX(gpu_power_watts) as max_gpu_power_watts
        FROM node_telemetry
        WHERE node_id = :node_id
          AND timestamp >= :start
          AND timestamp <= :end
        GROUP BY time_bucket
        ORDER BY time_bucket
    """)

    result = await db.execute(query, {
        "node_id": node_id,
        "start": start,
        "end": end,
        "res_min": resolution_minutes,
    })
    rows = result.mappings().all()

    return [
        {
            "timestamp": row["time_bucket"],
            "server_power_watts": row["avg_server_power_watts"],
            "server_power_watts_min": row["min_server_power_watts"],
            "server_power_watts_max": row["max_server_power_watts"],
            "gpu_power_watts": row["avg_gpu_power_watts"],
            "gpu_power_watts_min": row["min_gpu_power_watts"],
            "gpu_power_watts_max": row["max_gpu_power_watts"],
        }
        for row in rows
    ]


async def get_cluster_power_history(
    db: AsyncSession,
    start: datetime,
    end: datetime,
    resolution_minutes: int = 5,
) -> List[dict]:
    """Get aggregated cluster-wide power history (all nodes summed)."""
    from sqlalchemy import text

    query = text("""
        SELECT
            time_bucket,
            SUM(avg_server_power) as total_server_power_watts,
            SUM(avg_gpu_power) as total_gpu_power_watts,
            COUNT(*) as node_count
        FROM (
            SELECT
                CONCAT(DATE_FORMAT(timestamp, '%Y-%m-%d %H:'),
                       LPAD(FLOOR(MINUTE(timestamp) / :res_min) * :res_min, 2, '0')) as time_bucket,
                node_id,
                AVG(server_power_watts) as avg_server_power,
                AVG(gpu_power_watts) as avg_gpu_power
            FROM node_telemetry
            WHERE timestamp >= :start
              AND timestamp <= :end
            GROUP BY time_bucket, node_id
        ) per_node
        GROUP BY time_bucket
        ORDER BY time_bucket
    """)

    result = await db.execute(query, {
        "start": start,
        "end": end,
        "res_min": resolution_minutes,
    })
    rows = result.mappings().all()

    return [
        {
            "timestamp": row["time_bucket"],
            "total_server_power_watts": row["total_server_power_watts"],
            "total_gpu_power_watts": row["total_gpu_power_watts"],
            "node_count": row["node_count"],
        }
        for row in rows
    ]


async def delete_old_node_telemetry(
    db: AsyncSession, older_than: datetime
) -> int:
    """Delete node telemetry data older than the given datetime."""
    result = await db.execute(
        delete(NodeTelemetry).where(NodeTelemetry.timestamp < older_than)
    )
    await db.flush()
    return result.rowcount


async def delete_expired_api_keys(
    db: AsyncSession, grace_days: int = 15
) -> int:
    """Delete API keys that have been expired for more than grace_days.

    Service keys are excluded — they never expire and are admin-managed.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=grace_days)
    result = await db.execute(
        delete(ApiKey).where(
            ApiKey.is_service.is_(False),
            ApiKey.expires_at.isnot(None),
            ApiKey.expires_at < cutoff,
            ApiKey.status.in_([ApiKeyStatus.ACTIVE, ApiKeyStatus.EXPIRED]),
        )
    )
    await db.flush()
    return result.rowcount


async def count_user_active_api_keys(db: AsyncSession, user_id: int) -> int:
    """Count active (non-revoked) API keys for a user."""
    result = await db.execute(
        select(func.count(ApiKey.id)).where(
            ApiKey.user_id == user_id,
            ApiKey.status == ApiKeyStatus.ACTIVE,
        )
    )
    return result.scalar() or 0


# Quota Request CRUD
async def create_quota_request(
    db: AsyncSession,
    request_type: str,
    justification: str,
    user_id: Optional[int] = None,
    requester_name: Optional[str] = None,
    requester_email: Optional[str] = None,
    affiliation: Optional[str] = None,
    requested_tokens: Optional[int] = None,
    requested_rpm: Optional[int] = None,
) -> QuotaRequest:
    """Create a quota or API key request."""
    quota_request = QuotaRequest(
        user_id=user_id,
        requester_name=requester_name,
        requester_email=requester_email,
        affiliation=affiliation,
        request_type=request_type,
        justification=justification,
        requested_tokens=requested_tokens,
        requested_rpm=requested_rpm,
        status=QuotaRequestStatus.PENDING,
    )
    db.add(quota_request)
    await db.flush()
    return quota_request


async def get_pending_quota_requests(db: AsyncSession) -> List[QuotaRequest]:
    """Get all pending quota requests."""
    result = await db.execute(
        select(QuotaRequest)
        .where(QuotaRequest.status == QuotaRequestStatus.PENDING)
        .order_by(QuotaRequest.created_at.asc())
    )
    return list(result.scalars().all())


async def review_quota_request(
    db: AsyncSession,
    request_id: int,
    reviewer_id: int,
    status: QuotaRequestStatus,
    review_notes: Optional[str] = None,
) -> Optional[QuotaRequest]:
    """Review a quota request."""
    result = await db.execute(
        select(QuotaRequest).where(QuotaRequest.id == request_id)
    )
    quota_request = result.scalar_one_or_none()
    if quota_request:
        quota_request.status = status
        quota_request.reviewed_by = reviewer_id
        quota_request.reviewed_at = datetime.now(timezone.utc)
        quota_request.review_notes = review_notes
        await db.flush()
    return quota_request


# Service Key Request CRUD
async def create_service_key_request(
    db: AsyncSession,
    api_key_id: int,
    user_id: int,
    service_name: str,
    reason: str,
    alternative_contacts: Optional[str] = None,
    data_risk_level: str = "low",
    compliance_tags: Optional[str] = None,
    compliance_other: Optional[str] = None,
) -> ServiceKeyRequest:
    """Create a service key promotion request."""
    req = ServiceKeyRequest(
        api_key_id=api_key_id,
        user_id=user_id,
        service_name=service_name,
        reason=reason,
        alternative_contacts=alternative_contacts,
        data_risk_level=data_risk_level,
        compliance_tags=compliance_tags,
        compliance_other=compliance_other,
        status=ServiceKeyRequestStatus.PENDING,
    )
    db.add(req)
    await db.flush()
    return req


async def get_pending_service_key_requests(db: AsyncSession) -> List[ServiceKeyRequest]:
    """Get all pending service key requests with eager-loaded relations."""
    result = await db.execute(
        select(ServiceKeyRequest)
        .where(ServiceKeyRequest.status == ServiceKeyRequestStatus.PENDING)
        .options(
            selectinload(ServiceKeyRequest.api_key),
            selectinload(ServiceKeyRequest.user).selectinload(User.group),
        )
        .order_by(ServiceKeyRequest.created_at.asc())
    )
    return list(result.scalars().all())


async def get_user_service_key_requests(db: AsyncSession, user_id: int) -> List[ServiceKeyRequest]:
    """Get all service key requests for a user."""
    result = await db.execute(
        select(ServiceKeyRequest)
        .where(ServiceKeyRequest.user_id == user_id)
        .options(selectinload(ServiceKeyRequest.api_key))
        .order_by(ServiceKeyRequest.created_at.desc())
    )
    return list(result.scalars().all())


async def review_service_key_request(
    db: AsyncSession,
    request_id: int,
    reviewer_id: int,
    approved: bool,
    review_notes: Optional[str] = None,
) -> Optional[ServiceKeyRequest]:
    """Approve or deny a service key request.

    If approved, promotes the associated API key to service status.
    """
    result = await db.execute(
        select(ServiceKeyRequest)
        .where(ServiceKeyRequest.id == request_id)
        .options(selectinload(ServiceKeyRequest.api_key))
    )
    req = result.scalar_one_or_none()
    if not req:
        return None

    req.status = ServiceKeyRequestStatus.APPROVED if approved else ServiceKeyRequestStatus.DENIED
    req.reviewed_by = reviewer_id
    req.reviewed_at = datetime.now(timezone.utc)
    req.review_notes = review_notes

    if approved and req.api_key:
        key = req.api_key
        key.is_service = True
        key.service_name = req.service_name
        key.service_contacts = req.alternative_contacts
        key.data_risk_level = req.data_risk_level
        key.compliance_tags = req.compliance_tags
        key.promoted_at = datetime.now(timezone.utc)
        key.promoted_by = reviewer_id
        key.expires_at = None  # Service keys never expire

    await db.flush()
    return req


async def request_service_key_revocation(
    db: AsyncSession,
    api_key_id: int,
    reason: str,
) -> Optional[ApiKey]:
    """User requests revocation of a service key (admin must execute)."""
    result = await db.execute(
        select(ApiKey).where(ApiKey.id == api_key_id, ApiKey.is_service.is_(True))
    )
    key = result.scalar_one_or_none()
    if key:
        key.revocation_requested_at = datetime.now(timezone.utc)
        key.revocation_reason = reason
        await db.flush()
    return key


async def revoke_service_key(db: AsyncSession, api_key_id: int) -> Optional[ApiKey]:
    """Admin revokes a service key."""
    result = await db.execute(
        select(ApiKey).where(ApiKey.id == api_key_id, ApiKey.is_service.is_(True))
    )
    key = result.scalar_one_or_none()
    if key:
        key.status = ApiKeyStatus.REVOKED
        await db.flush()
    return key


async def count_pending_service_key_requests(db: AsyncSession) -> int:
    """Count pending service key requests (for sidebar badge)."""
    result = await db.execute(
        select(func.count(ServiceKeyRequest.id)).where(
            ServiceKeyRequest.status == ServiceKeyRequestStatus.PENDING
        )
    )
    return result.scalar() or 0


async def count_pending_quota_requests(db: AsyncSession) -> int:
    """Count pending quota requests (for sidebar badge)."""
    result = await db.execute(
        select(func.count(QuotaRequest.id)).where(
            QuotaRequest.status == QuotaRequestStatus.PENDING
        )
    )
    return result.scalar() or 0


# Scheduler Decision CRUD
async def create_scheduler_decision(
    db: AsyncSession,
    request_id: int,
    selected_backend_id: int,
    candidate_backends: Optional[list] = None,
    scores: Optional[dict] = None,
    user_deficit: Optional[float] = None,
    user_weight: Optional[float] = None,
    user_recent_usage: Optional[int] = None,
    hard_constraints_passed: Optional[list] = None,
    hard_constraints_failed: Optional[list] = None,
) -> SchedulerDecision:
    """Record a scheduler decision."""
    decision = SchedulerDecision(
        request_id=request_id,
        selected_backend_id=selected_backend_id,
        candidate_backends=candidate_backends,
        scores=scores,
        user_deficit=user_deficit,
        user_weight=user_weight,
        user_recent_usage=user_recent_usage,
        hard_constraints_passed=hard_constraints_passed,
        hard_constraints_failed=hard_constraints_failed,
    )
    db.add(decision)
    await db.flush()
    return decision


# Audit Search
def _build_request_filter_conditions(
    user_id: Optional[int] = None,
    model: Optional[str] = None,
    status: Optional[RequestStatus] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    search_text: Optional[str] = None,
) -> list:
    """Build shared filter conditions for request queries."""
    conditions = []
    if user_id:
        conditions.append(Request.user_id == user_id)
    if model:
        conditions.append(Request.model == model)
    if status:
        conditions.append(Request.status == status)
    if start_date:
        conditions.append(Request.created_at >= start_date)
    if end_date:
        conditions.append(Request.created_at <= end_date)
    if search_text:
        conditions.append(
            or_(
                Request.prompt.ilike(f"%{search_text}%"),
                func.json_extract(Request.messages, "$").ilike(f"%{search_text}%"),
            )
        )
    return conditions


async def search_requests(
    db: AsyncSession,
    user_id: Optional[int] = None,
    model: Optional[str] = None,
    status: Optional[RequestStatus] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    search_text: Optional[str] = None,
    cursor_before: Optional[int] = None,
    page: Optional[int] = None,
    last_page: bool = False,
    limit: int = 50,
) -> Tuple[List[Request], int]:
    """Search requests with hybrid pagination.

    Supports three modes:
    - page=N (1-indexed): OFFSET-based, for numbered page links (pages 1-20)
    - cursor_before=ID: keyset-based, for deep navigation via Next
    - last_page=True: fetch the final page of results
    - None of the above: first page (equivalent to page=1)
    """
    base_conditions = _build_request_filter_conditions(
        user_id, model, status, start_date, end_date, search_text
    )
    count_where = and_(*base_conditions) if base_conditions else True
    count_result = await db.execute(
        select(func.count(Request.id)).where(count_where)
    )
    total = count_result.scalar_one()

    conditions = list(base_conditions)

    if last_page:
        # Fetch the last N rows: query ascending, then reverse
        where_clause = and_(*conditions) if conditions else True
        query = (
            select(Request)
            .where(where_clause)
            .order_by(Request.id.asc())
            .limit(limit)
            .options(selectinload(Request.response))
        )
        result = await db.execute(query)
        requests = list(reversed(result.scalars().all()))
    elif cursor_before is not None:
        # Keyset pagination — constant time regardless of depth
        conditions.append(Request.id < cursor_before)
        where_clause = and_(*conditions) if conditions else True
        query = (
            select(Request)
            .where(where_clause)
            .order_by(Request.id.desc())
            .limit(limit)
            .options(selectinload(Request.response))
        )
        result = await db.execute(query)
        requests = list(result.scalars().all())
    else:
        # OFFSET-based for numbered pages (fast for early pages)
        offset = ((page or 1) - 1) * limit
        where_clause = and_(*conditions) if conditions else True
        query = (
            select(Request)
            .where(where_clause)
            .order_by(Request.id.desc())
            .offset(offset)
            .limit(limit)
            .options(selectinload(Request.response))
        )
        result = await db.execute(query)
        requests = list(result.scalars().all())

    return requests, total


async def iter_requests_batched(
    db: AsyncSession,
    user_id: Optional[int] = None,
    model: Optional[str] = None,
    status: Optional[RequestStatus] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    search_text: Optional[str] = None,
    batch_size: int = 1000,
):
    """Yield all matching requests in batches using keyset pagination.

    Memory-efficient: only one batch is loaded at a time.
    """
    conditions = _build_request_filter_conditions(
        user_id, model, status, start_date, end_date, search_text
    )
    cursor = None

    while True:
        batch_conditions = list(conditions)
        if cursor is not None:
            batch_conditions.append(Request.id < cursor)

        where_clause = and_(*batch_conditions) if batch_conditions else True
        query = (
            select(Request)
            .where(where_clause)
            .order_by(Request.id.desc())
            .limit(batch_size)
            .options(selectinload(Request.response))
        )
        result = await db.execute(query)
        batch = list(result.scalars().all())
        if not batch:
            break

        for req in batch:
            yield req

        cursor = batch[-1].id
        if len(batch) < batch_size:
            break


# User stats/detail queries
async def get_user_with_stats(db: AsyncSession, user_id: int) -> Optional[dict]:
    """Get user with usage statistics."""
    user = await get_user_by_id(db, user_id)
    if not user:
        return None

    # Total tokens (with input/output breakdown)
    token_result = await db.execute(
        select(
            func.coalesce(func.sum(Request.prompt_tokens), 0),
            func.coalesce(func.sum(Request.completion_tokens), 0),
            func.coalesce(func.sum(Request.total_tokens), 0),
        )
        .where(Request.user_id == user_id, Request.total_tokens.isnot(None))
    )
    token_row = token_result.one()
    prompt_tokens = int(token_row[0])
    completion_tokens = int(token_row[1])
    total_tokens = int(token_row[2])

    # Request count
    req_result = await db.execute(
        select(func.count(Request.id)).where(Request.user_id == user_id)
    )
    request_count = req_result.scalar_one()

    # Add archived offsets (requests deleted by retention)
    arch_result = await db.execute(
        select(
            Quota.archived_prompt_tokens,
            Quota.archived_completion_tokens,
            Quota.archived_total_tokens,
            Quota.archived_request_count,
        ).where(Quota.user_id == user_id)
    )
    arch_row = arch_result.first()
    if arch_row:
        prompt_tokens += int(arch_row[0])
        completion_tokens += int(arch_row[1])
        total_tokens += int(arch_row[2])
        request_count += int(arch_row[3])

    # Favorite models (top 5)
    model_result = await db.execute(
        select(Request.model, func.count(Request.id).label("cnt"))
        .where(Request.user_id == user_id)
        .group_by(Request.model)
        .order_by(func.count(Request.id).desc())
        .limit(5)
    )
    favorite_models = [(row[0], row[1]) for row in model_result.all()]

    # API key count
    key_result = await db.execute(
        select(func.count(ApiKey.id)).where(ApiKey.user_id == user_id)
    )
    api_key_count = key_result.scalar_one()

    # Quota
    quota = await get_user_quota(db, user_id)

    # API keys
    api_keys = await get_user_api_keys(db, user_id, include_revoked=True)

    return {
        "user": user,
        "total_tokens": total_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "request_count": request_count,
        "favorite_models": favorite_models,
        "api_key_count": api_key_count,
        "quota": quota,
        "api_keys": api_keys,
    }


async def get_user_monthly_usage(
    db: AsyncSession, user_id: int, months: int = 12
) -> List[dict]:
    """Get monthly token usage for a user."""
    from sqlalchemy import text

    query = text("""
        SELECT
            DATE_FORMAT(created_at, '%Y-%m') as month,
            COALESCE(SUM(prompt_tokens), 0) as prompt_tokens,
            COALESCE(SUM(completion_tokens), 0) as completion_tokens,
            COALESCE(SUM(total_tokens), 0) as tokens,
            COUNT(*) as requests
        FROM requests
        WHERE user_id = :user_id
          AND total_tokens IS NOT NULL
          AND created_at >= DATE_SUB(NOW(), INTERVAL :months MONTH)
        GROUP BY month
        ORDER BY month
    """)
    result = await db.execute(query, {"user_id": user_id, "months": months})
    return [
        {
            "month": row[0],
            "prompt_tokens": int(row[1]),
            "completion_tokens": int(row[2]),
            "tokens": int(row[3]),
            "requests": int(row[4]),
        }
        for row in result.all()
    ]


async def get_all_api_keys(
    db: AsyncSession,
    skip: int = 0,
    limit: int = 100,
    search: Optional[str] = None,
    status_filter: Optional[str] = None,
    sort_by: Optional[str] = None,
    sort_dir: str = "desc",
    type_filter: Optional[str] = None,
) -> Tuple[List[ApiKey], int]:
    """Get all API keys with user info, paginated and sortable.

    type_filter: "service" for service keys only, "personal" for non-service only.
    """
    conditions = []
    if type_filter == "service":
        conditions.append(ApiKey.is_service.is_(True))
    elif type_filter == "personal":
        conditions.append(ApiKey.is_service.is_(False))
    if search:
        search_pattern = f"%{search}%"
        conditions.append(
            or_(
                ApiKey.name.ilike(search_pattern),
                ApiKey.key_prefix.ilike(search_pattern),
                User.username.ilike(search_pattern),
                User.email.ilike(search_pattern),
            )
        )
    if status_filter:
        conditions.append(ApiKey.status == ApiKeyStatus(status_filter))

    base_query = select(ApiKey).join(User, ApiKey.user_id == User.id)
    count_query = select(func.count(ApiKey.id)).join(User, ApiKey.user_id == User.id)

    if conditions:
        where_clause = and_(*conditions)
        base_query = base_query.where(where_clause)
        count_query = count_query.where(where_clause)

    count_result = await db.execute(count_query)
    total = count_result.scalar_one()

    # Determine sort order
    asc = sort_dir.lower() == "asc"
    if sort_by == "last_used":
        if asc:
            order = [case((ApiKey.last_used_at.is_(None), 1), else_=0), ApiKey.last_used_at.asc()]
        else:
            order = [case((ApiKey.last_used_at.is_(None), 1), else_=0), ApiKey.last_used_at.desc()]
    elif sort_by == "usage":
        order = [ApiKey.usage_count.asc() if asc else ApiKey.usage_count.desc()]
    else:
        # Default: created_at
        order = [ApiKey.created_at.asc() if asc else ApiKey.created_at.desc()]

    query = (
        base_query
        .options(selectinload(ApiKey.user))
        .order_by(*order)
        .offset(skip)
        .limit(limit)
    )
    result = await db.execute(query)
    return list(result.scalars().all()), total


async def update_user(db: AsyncSession, user_id: int, **kwargs) -> Optional[User]:
    """Update user fields."""
    result = await db.execute(
        select(User).options(selectinload(User.group)).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()
    if not user:
        return None
    for key, value in kwargs.items():
        if hasattr(user, key):
            setattr(user, key, value)
    await db.flush()
    return user


# Blog CRUD
async def get_published_blog_posts(db: AsyncSession) -> List[BlogPost]:
    """Get published blog posts ordered by published_at desc."""
    result = await db.execute(
        select(BlogPost)
        .options(selectinload(BlogPost.author))
        .where(
            and_(
                BlogPost.is_published.is_(True),
                BlogPost.deleted_at.is_(None),
            )
        )
        .order_by(BlogPost.published_at.desc())
    )
    return list(result.scalars().all())


async def get_blog_post_by_slug(db: AsyncSession, slug: str) -> Optional[BlogPost]:
    """Get a single published blog post by slug."""
    result = await db.execute(
        select(BlogPost)
        .options(selectinload(BlogPost.author))
        .where(
            and_(
                BlogPost.slug == slug,
                BlogPost.is_published.is_(True),
                BlogPost.deleted_at.is_(None),
            )
        )
    )
    return result.scalar_one_or_none()


async def get_all_blog_posts(db: AsyncSession) -> List[BlogPost]:
    """Get all blog posts including drafts (admin)."""
    result = await db.execute(
        select(BlogPost)
        .options(selectinload(BlogPost.author))
        .where(BlogPost.deleted_at.is_(None))
        .order_by(BlogPost.created_at.desc())
    )
    return list(result.scalars().all())


async def get_blog_post_by_id(db: AsyncSession, post_id: int) -> Optional[BlogPost]:
    """Get a blog post by ID (admin)."""
    result = await db.execute(
        select(BlogPost)
        .options(selectinload(BlogPost.author))
        .where(
            and_(
                BlogPost.id == post_id,
                BlogPost.deleted_at.is_(None),
            )
        )
    )
    return result.scalar_one_or_none()


async def create_blog_post(
    db: AsyncSession,
    title: str,
    slug: str,
    content: str,
    excerpt: Optional[str],
    author_id: int,
    is_published: bool = False,
) -> BlogPost:
    """Create a new blog post."""
    post = BlogPost(
        title=title,
        slug=slug,
        content=content,
        excerpt=excerpt,
        author_id=author_id,
        is_published=is_published,
        published_at=datetime.now(timezone.utc) if is_published else None,
    )
    db.add(post)
    await db.flush()
    return post


async def update_blog_post(db: AsyncSession, post_id: int, **kwargs) -> Optional[BlogPost]:
    """Update a blog post's fields."""
    result = await db.execute(
        select(BlogPost).where(
            and_(BlogPost.id == post_id, BlogPost.deleted_at.is_(None))
        )
    )
    post = result.scalar_one_or_none()
    if not post:
        return None
    for key, value in kwargs.items():
        if hasattr(post, key):
            setattr(post, key, value)
    await db.flush()
    return post


async def delete_blog_post(db: AsyncSession, post_id: int) -> bool:
    """Soft delete a blog post."""
    result = await db.execute(
        select(BlogPost).where(
            and_(BlogPost.id == post_id, BlogPost.deleted_at.is_(None))
        )
    )
    post = result.scalar_one_or_none()
    if post:
        post.deleted_at = datetime.now(timezone.utc)
        await db.flush()
        return True
    return False


# IP Tracking queries
async def get_api_key_last_ips_batch(
    db: AsyncSession, api_key_ids: List[int]
) -> dict:
    """Get the most recent client_ip for each API key. Returns {api_key_id: ip}."""
    if not api_key_ids:
        return {}
    from sqlalchemy import text

    placeholders = ",".join(str(int(kid)) for kid in api_key_ids)
    query = text(f"""
        SELECT r.api_key_id, r.client_ip
        FROM requests r
        INNER JOIN (
            SELECT api_key_id, MAX(created_at) as max_created
            FROM requests
            WHERE api_key_id IN ({placeholders}) AND client_ip IS NOT NULL
            GROUP BY api_key_id
        ) latest ON r.api_key_id = latest.api_key_id AND r.created_at = latest.max_created
        WHERE r.client_ip IS NOT NULL
    """)
    result = await db.execute(query)
    return {row[0]: row[1] for row in result.all()}


async def get_user_recent_ips(
    db: AsyncSession, user_id: int, days: int = 90
) -> List[Tuple]:
    """Get recent IPs for a user. Returns list of (ip, last_seen, count)."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    result = await db.execute(
        select(
            Request.client_ip,
            func.max(Request.created_at).label("last_seen"),
            func.count(Request.id).label("req_count"),
        )
        .where(
            and_(
                Request.user_id == user_id,
                Request.created_at >= cutoff,
                Request.client_ip.isnot(None),
            )
        )
        .group_by(Request.client_ip)
        .order_by(func.max(Request.created_at).desc())
    )
    return [(row[0], row[1], row[2]) for row in result.all()]


# AppConfig CRUD
async def get_config(db: AsyncSession, key: str) -> Optional[str]:
    """Get a raw config value (JSON string) by key."""
    result = await db.execute(select(AppConfig.value).where(AppConfig.key == key))
    return result.scalar_one_or_none()


async def get_config_json(db: AsyncSession, key: str, default: Any = None) -> Any:
    """Get a config value parsed from JSON."""
    raw = await get_config(db, key)
    if raw is None:
        return default
    try:
        return _json.loads(raw)
    except (ValueError, TypeError):
        return default


async def set_config(
    db: AsyncSession, key: str, value: Any, description: Optional[str] = None
) -> None:
    """Upsert a config value (JSON-encodes the value)."""
    json_value = _json.dumps(value)
    result = await db.execute(select(AppConfig).where(AppConfig.key == key))
    existing = result.scalar_one_or_none()
    if existing:
        existing.value = json_value
        if description is not None:
            existing.description = description
    else:
        row = AppConfig(key=key, value=json_value, description=description)
        db.add(row)
    await db.flush()


# ---------------------------------------------------------------------------
# Use Agreement helpers
# ---------------------------------------------------------------------------


async def get_agreement(db: AsyncSession) -> dict:
    """Return the current use agreement text and version."""
    version = await get_config_json(db, "agreement.version", None)
    text = await get_config_json(db, "agreement.text", "")
    return {"version": int(version) if version is not None else None, "text": text}


async def set_agreement_text(db: AsyncSession, text: str, bump_version: bool = False) -> int:
    """Update agreement text and optionally bump the version (forces re-acceptance)."""
    await set_config(db, "agreement.text", text, description="Use agreement HTML content")
    version = int(await get_config_json(db, "agreement.version", 1))
    if bump_version:
        version += 1
        await set_config(db, "agreement.version", version, description="Current use agreement version number")
    return version


async def accept_agreement(db: AsyncSession, user_id: int, version: int) -> None:
    """Record that a user accepted a specific agreement version."""
    await db.execute(
        update(User)
        .where(User.id == user_id)
        .values(
            agreement_version_accepted=version,
            agreement_accepted_at=datetime.now(timezone.utc),
        )
    )
    await db.flush()


# ---------------------------------------------------------------------------
# Email helpers
# ---------------------------------------------------------------------------


async def create_email_log(
    db: AsyncSession,
    subject: str,
    sent_by: int,
    recipient_count: int,
    body_preview: Optional[str] = None,
    blog_post_id: Optional[int] = None,
) -> EmailLog:
    """Create an email log entry."""
    log = EmailLog(
        subject=subject,
        body_preview=(body_preview or "")[:500],
        recipient_count=recipient_count,
        sent_by=sent_by,
        blog_post_id=blog_post_id,
        status="pending",
    )
    db.add(log)
    await db.flush()
    return log


async def update_email_log(
    db: AsyncSession, log_id: int, **kwargs
) -> None:
    """Update email log fields."""
    result = await db.execute(select(EmailLog).where(EmailLog.id == log_id))
    log = result.scalar_one_or_none()
    if log:
        for k, v in kwargs.items():
            setattr(log, k, v)
        await db.flush()


async def get_email_logs(
    db: AsyncSession, limit: int = 20
) -> List[EmailLog]:
    """Get recent email log entries."""
    result = await db.execute(
        select(EmailLog)
        .options(selectinload(EmailLog.sender), selectinload(EmailLog.blog_post))
        .order_by(EmailLog.created_at.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


async def get_emailable_users(
    db: AsyncSession,
    group_ids: Optional[List[int]] = None,
    user_ids: Optional[List[int]] = None,
    exclude_blog_optout: bool = False,
) -> List[User]:
    """Get active users for emailing, optionally filtered by group/user IDs.

    If exclude_blog_optout=True, exclude users who set email_optout preference.
    """
    stmt = select(User).options(selectinload(User.group)).where(
        User.is_active == True,  # noqa: E712
        User.email.isnot(None),
        User.email != "",
    )
    if group_ids:
        stmt = stmt.where(User.group_id.in_(group_ids))
    if user_ids:
        stmt = stmt.where(User.id.in_(user_ids))

    result = await db.execute(stmt)
    users = list(result.scalars().all())

    if exclude_blog_optout and users:
        # Get opted-out user IDs from AppConfig
        optout_keys = [f"user.{u.id}.email_optout" for u in users]
        cfg_result = await db.execute(
            select(AppConfig.key, AppConfig.value).where(AppConfig.key.in_(optout_keys))
        )
        opted_out_ids = set()
        for key, value in cfg_result.all():
            if value and value.strip('"').lower() in ("true", "1", "on"):
                # Extract user_id from key "user.{id}.email_optout"
                parts = key.split(".")
                if len(parts) == 3:
                    try:
                        opted_out_ids.add(int(parts[1]))
                    except ValueError:
                        pass
        users = [u for u in users if u.id not in opted_out_ids]

    return users


async def get_blog_email_log(db: AsyncSession, blog_post_id: int) -> Optional[EmailLog]:
    """Get the most recent email log for a blog post."""
    result = await db.execute(
        select(EmailLog)
        .where(EmailLog.blog_post_id == blog_post_id)
        .order_by(EmailLog.created_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


# ---------------------------------------------------------------------------
# Node drain helpers
# ---------------------------------------------------------------------------


async def get_active_request_count_for_node_backends(
    db: AsyncSession, backend_ids: List[int]
) -> int:
    """Sum current_concurrent across the given backends."""
    if not backend_ids:
        return 0
    result = await db.execute(
        select(func.coalesce(func.sum(Backend.current_concurrent), 0)).where(
            Backend.id.in_(backend_ids)
        )
    )
    return int(result.scalar())


# ---------------------------------------------------------------------------
# Public stats queries (status page)
# ---------------------------------------------------------------------------

_TREND_BUCKETS = {
    "hour":  (3600,       60),
    "day":   (86400,      900),
    "week":  (604800,     3600),
    "month": (2592000,    21600),
    "year":  (31536000,   86400),
}

# In-memory TTL cache for trend queries.  Week/month/year scans are expensive
# (millions of rows) but the data changes slowly, so we cache results.
_TREND_CACHE_TTL = {
    "hour":  60,     # 1 min  – buckets are 60s
    "day":   120,    # 2 min  – buckets are 15 min
    "week":  300,    # 5 min  – buckets are 1 h
    "month": 600,    # 10 min – buckets are 6 h
    "year":  900,    # 15 min – buckets are 24 h
}
_trend_cache: Dict[str, tuple] = {}  # key -> (expire_ts, data)


async def get_global_token_total(db: AsyncSession, include_offset: bool = True) -> dict:
    """Total tokens ever served (all users, all time).

    Uses per-user quota counters (tokens_used + lifetime_tokens_used)
    as the source of truth so the total survives retention deletions
    and app restarts without drift.

    When *include_offset* is True (the default), the ``stats.token_offset``
    value from app_config is added to the totals.  This lets the homepage
    reflect historical usage from before the current database without
    polluting per-model or per-user metrics.

    Returns {prompt_tokens, completion_tokens, total_tokens}.
    """
    result = await db.execute(
        select(
            func.coalesce(func.sum(Quota.tokens_used), 0),
            func.coalesce(func.sum(Quota.lifetime_tokens_used), 0),
        )
    )
    row = result.one()
    total_tokens = int(row[0]) + int(row[1])

    # prompt/completion breakdown from live requests only (archived
    # breakdown is approximate; total_tokens is the authoritative number)
    detail_result = await db.execute(
        select(
            func.coalesce(func.sum(Request.prompt_tokens), 0),
            func.coalesce(func.sum(Request.completion_tokens), 0),
        )
        .where(Request.total_tokens.isnot(None))
    )
    detail_row = detail_result.one()
    totals = {
        "prompt_tokens": int(detail_row[0]),
        "completion_tokens": int(detail_row[1]),
        "total_tokens": total_tokens,
    }

    if include_offset:
        offset = await get_config_json(db, "stats.token_offset", 0)
        if offset:
            totals["total_tokens"] += int(offset)

    return totals


def _bucket_iso(unix_ts: int) -> str:
    """Convert a UTC unix timestamp to an ISO 8601 string with Z suffix."""
    return datetime.fromtimestamp(unix_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


async def get_token_trend(
    db: AsyncSession, range_name: str = "day"
) -> List[dict]:
    """Token counts bucketed over time for trend chart."""
    cache_key = f"token_trend:{range_name}"
    cached = _trend_cache.get(cache_key)
    if cached and cached[0] > _time.monotonic():
        return cached[1]

    since_sec, bucket_sec = _TREND_BUCKETS.get(range_name, _TREND_BUCKETS["day"])
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=since_sec)

    from sqlalchemy import text
    stmt = text(
        "SELECT"
        "  FLOOR(UNIX_TIMESTAMP(CONVERT_TZ(created_at, '+00:00', '+00:00')) / :bucket) * :bucket AS bucket_ts,"
        "  COALESCE(SUM(prompt_tokens), 0) AS prompt_v,"
        "  COALESCE(SUM(completion_tokens), 0) AS completion_v,"
        "  COALESCE(SUM(total_tokens), 0) AS v"
        " FROM requests"
        " WHERE created_at >= :cutoff AND total_tokens IS NOT NULL"
        " GROUP BY bucket_ts"
        " ORDER BY bucket_ts"
    )
    result = await db.execute(stmt, {"bucket": bucket_sec, "cutoff": cutoff})
    data = [
        {
            "t": _bucket_iso(int(row[0])),
            "prompt_tokens": int(row[1]),
            "completion_tokens": int(row[2]),
            "v": int(row[3]),
        }
        for row in result.all()
    ]

    ttl = _TREND_CACHE_TTL.get(range_name, 120)
    _trend_cache[cache_key] = (_time.monotonic() + ttl, data)
    return data


async def get_active_users_trend(
    db: AsyncSession, range_name: str = "day"
) -> List[dict]:
    """Distinct user counts bucketed over time for trend chart."""
    cache_key = f"active_users_trend:{range_name}"
    cached = _trend_cache.get(cache_key)
    if cached and cached[0] > _time.monotonic():
        return cached[1]

    since_sec, bucket_sec = _TREND_BUCKETS.get(range_name, _TREND_BUCKETS["day"])
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=since_sec)

    from sqlalchemy import text
    stmt = text(
        "SELECT"
        "  FLOOR(UNIX_TIMESTAMP(CONVERT_TZ(created_at, '+00:00', '+00:00')) / :bucket) * :bucket AS bucket_ts,"
        "  COUNT(DISTINCT user_id) AS v"
        " FROM requests"
        " WHERE created_at >= :cutoff"
        " GROUP BY bucket_ts"
        " ORDER BY bucket_ts"
    )
    result = await db.execute(stmt, {"bucket": bucket_sec, "cutoff": cutoff})
    data = [
        {"t": _bucket_iso(int(row[0])), "v": int(row[1])}
        for row in result.all()
    ]

    ttl = _TREND_CACHE_TTL.get(range_name, 120)
    _trend_cache[cache_key] = (_time.monotonic() + ttl, data)
    return data


async def get_top_active_users(
    db: AsyncSession, window_seconds: int = 3600, limit: int = 10
) -> List[dict]:
    """Get top users by total tokens within a time window.

    Returns up to `limit` users sorted descending by total_tokens,
    with per-user stats: username, group, status, top model, request count,
    input/output tokens, and most-used API key.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=window_seconds)

    # Aggregate per-user token usage and request counts
    stmt = (
        select(
            Request.user_id,
            func.coalesce(func.sum(Request.total_tokens), 0).label("total_tokens"),
            func.coalesce(func.sum(Request.prompt_tokens), 0).label("prompt_tokens"),
            func.coalesce(func.sum(Request.completion_tokens), 0).label("completion_tokens"),
            func.count(Request.id).label("request_count"),
        )
        .where(
            and_(
                Request.created_at >= cutoff,
                Request.total_tokens.isnot(None),
            )
        )
        .group_by(Request.user_id)
        .order_by(func.sum(Request.total_tokens).desc())
        .limit(limit)
    )
    result = await db.execute(stmt)
    rows = result.all()

    if not rows:
        return []

    user_ids = [row[0] for row in rows]

    # Fetch user objects with groups
    user_result = await db.execute(
        select(User).options(selectinload(User.group)).where(User.id.in_(user_ids))
    )
    users_by_id = {u.id: u for u in user_result.scalars().all()}

    # Top model per user in the window
    model_stmt = (
        select(
            Request.user_id,
            Request.model,
            func.count(Request.id).label("cnt"),
        )
        .where(
            and_(
                Request.created_at >= cutoff,
                Request.user_id.in_(user_ids),
            )
        )
        .group_by(Request.user_id, Request.model)
    )
    model_result = await db.execute(model_stmt)
    top_model_by_user: dict[int, str] = {}
    model_counts: dict[int, dict[str, int]] = {}
    for uid, model_name, cnt in model_result.all():
        model_counts.setdefault(uid, {})
        model_counts[uid][model_name] = cnt
    for uid, models in model_counts.items():
        top_model_by_user[uid] = max(models, key=models.get)

    # Most-used API key per user in the window
    key_stmt = (
        select(
            Request.user_id,
            Request.api_key_id,
            func.count(Request.id).label("cnt"),
        )
        .where(
            and_(
                Request.created_at >= cutoff,
                Request.user_id.in_(user_ids),
            )
        )
        .group_by(Request.user_id, Request.api_key_id)
    )
    key_result = await db.execute(key_stmt)
    top_key_by_user: dict[int, int] = {}
    key_counts: dict[int, dict[int, int]] = {}
    for uid, key_id, cnt in key_result.all():
        key_counts.setdefault(uid, {})
        key_counts[uid][key_id] = cnt
    for uid, keys in key_counts.items():
        top_key_by_user[uid] = max(keys, key=keys.get)

    # Fetch API key prefixes
    all_key_ids = list(top_key_by_user.values())
    key_prefix_map: dict[int, str] = {}
    if all_key_ids:
        kp_result = await db.execute(
            select(ApiKey.id, ApiKey.key_prefix, ApiKey.name).where(ApiKey.id.in_(all_key_ids))
        )
        for kid, prefix, kname in kp_result.all():
            key_prefix_map[kid] = kname or prefix or str(kid)

    # Build result
    out = []
    for row in rows:
        uid = row[0]
        user = users_by_id.get(uid)
        if not user:
            continue
        top_key_id = top_key_by_user.get(uid)
        out.append({
            "user_id": uid,
            "username": user.username,
            "full_name": user.full_name or "",
            "group_name": user.group.display_name if user.group else "—",
            "is_active": user.is_active,
            "top_model": top_model_by_user.get(uid, "—"),
            "request_count": int(row[4]),
            "prompt_tokens": int(row[2]),
            "completion_tokens": int(row[3]),
            "total_tokens": int(row[1]),
            "top_api_key_id": top_key_id,
            "top_api_key_name": key_prefix_map.get(top_key_id, "—") if top_key_id else "—",
        })

    return out


async def cancel_active_requests_for_backends(
    db: AsyncSession, backend_ids: List[int]
) -> int:
    """Cancel all PROCESSING requests on the given backends and reset counters."""
    if not backend_ids:
        return 0
    now = datetime.now(timezone.utc)
    result = await db.execute(
        update(Request)
        .where(
            and_(
                Request.backend_id.in_(backend_ids),
                Request.status == RequestStatus.PROCESSING,
            )
        )
        .values(
            status=RequestStatus.CANCELLED,
            completed_at=now,
            error_message="Force-drained: node taken offline by admin",
        )
    )
    cancelled = result.rowcount
    # Reset concurrency counters
    await db.execute(
        update(Backend)
        .where(Backend.id.in_(backend_ids))
        .values(current_concurrent=0)
    )
    await db.flush()
    return cancelled


# ---------------------------------------------------------------------------
# Configuration Backup & Restore
# ---------------------------------------------------------------------------

# Tables to export in FK dependency order, with their unique-key column(s)
_CONFIG_TABLES = [
    (Group, "name"),
    (AppConfig, "key"),
    (BlogPost, "slug"),
    (Node, "name"),
    (Backend, "name"),
    (Model, None),          # unique on (backend_id, name) — handled specially
    (GPUDevice, None),      # unique on (node_id, gpu_index) — handled specially
    (User, "username"),
    (ApiKey, "key_hash"),
    (Quota, "user_id"),
]

# Backend runtime fields to reset on import
_BACKEND_RUNTIME_DEFAULTS = {
    "current_concurrent": 0,
    "consecutive_failures": 0,
    "last_health_check": None,
    "last_success": None,
    "live_failure_count": 0,
    "circuit_open_until": None,
    "latency_ema_ms": None,
    "ttft_ema_ms": None,
}


def _row_to_dict(row) -> dict:
    """Convert a SQLAlchemy model instance to a JSON-safe dict."""
    d = {}
    for col in row.__table__.columns:
        val = getattr(row, col.name)
        if isinstance(val, datetime):
            val = val.isoformat()
        elif isinstance(val, PyEnum):
            val = val.value
        d[col.name] = val
    return d


async def export_config_tables(db: AsyncSession) -> dict:
    """Export all configuration tables as a dict suitable for JSON serialization."""
    from backend.app.settings import get_settings

    data = {
        "metadata": {
            "version": get_settings().app_version,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "tables": [cls.__tablename__ for cls, _ in _CONFIG_TABLES],
        }
    }

    for model_cls, _ in _CONFIG_TABLES:
        result = await db.execute(select(model_cls))
        rows = result.scalars().all()
        data[model_cls.__tablename__] = [_row_to_dict(r) for r in rows]

    return data


async def import_config_tables(db: AsyncSession, data: dict) -> dict:
    """Import configuration from a backup dict. Returns summary with inserted/skipped counts."""
    summary: dict[str, dict[str, int]] = {}

    for model_cls, unique_col in _CONFIG_TABLES:
        table_name = model_cls.__tablename__
        rows = data.get(table_name, [])
        inserted = 0
        skipped = 0

        for row_data in rows:
            # Check if row already exists by unique key
            exists = False
            if unique_col:
                pk_val = row_data.get(unique_col)
                if pk_val is not None:
                    if unique_col == "key":
                        # AppConfig uses 'key' as primary key
                        existing = await db.execute(
                            select(model_cls).where(model_cls.key == pk_val)
                        )
                    else:
                        existing = await db.execute(
                            select(model_cls).where(
                                getattr(model_cls, unique_col) == pk_val
                            )
                        )
                    exists = existing.scalars().first() is not None
            elif model_cls is Model:
                # Unique on (backend_id, name)
                bid = row_data.get("backend_id")
                mname = row_data.get("name")
                if bid and mname:
                    existing = await db.execute(
                        select(Model).where(
                            and_(Model.backend_id == bid, Model.name == mname)
                        )
                    )
                    exists = existing.scalars().first() is not None
            elif model_cls is GPUDevice:
                # Unique on (node_id, gpu_index)
                nid = row_data.get("node_id")
                gidx = row_data.get("gpu_index")
                if nid is not None and gidx is not None:
                    existing = await db.execute(
                        select(GPUDevice).where(
                            and_(
                                GPUDevice.node_id == nid,
                                GPUDevice.gpu_index == gidx,
                            )
                        )
                    )
                    exists = existing.scalars().first() is not None

            if exists:
                skipped += 1
                continue

            # Prepare row data — convert ISO strings back to datetimes
            clean = {}
            col_types = {c.name: c for c in model_cls.__table__.columns}
            for col_name, val in row_data.items():
                if col_name not in col_types:
                    continue
                col = col_types[col_name]
                if val is None:
                    clean[col_name] = None
                elif isinstance(col.type, SADateTime) and isinstance(val, str):
                    try:
                        clean[col_name] = datetime.fromisoformat(val)
                    except (ValueError, TypeError):
                        clean[col_name] = None
                else:
                    clean[col_name] = val

            # Reset backend runtime state
            if model_cls is Backend:
                clean.update(_BACKEND_RUNTIME_DEFAULTS)

            obj = model_cls(**clean)
            db.add(obj)
            try:
                await db.flush()
                inserted += 1
            except Exception:
                await db.rollback()
                skipped += 1

        summary[table_name] = {"inserted": inserted, "skipped": skipped}

    await db.commit()
    return summary


# ---------------------------------------------------------------------------
# Admin Audit Log
# ---------------------------------------------------------------------------

async def log_admin_action(
    db: AsyncSession,
    user_id: int,
    action: str,
    entity_type: str,
    entity_id: Optional[str] = None,
    before_value: Optional[dict] = None,
    after_value: Optional[dict] = None,
    ip_address: Optional[str] = None,
    detail: Optional[str] = None,
) -> AdminAuditLog:
    """Record an admin action to the persistent audit log."""
    entry = AdminAuditLog(
        user_id=user_id,
        action=action,
        entity_type=entity_type,
        entity_id=str(entity_id) if entity_id is not None else None,
        before_value=before_value,
        after_value=after_value,
        ip_address=ip_address,
        detail=detail,
    )
    db.add(entry)
    await db.flush()
    return entry


async def get_admin_audit_log(
    db: AsyncSession,
    user_id: Optional[int] = None,
    action: Optional[str] = None,
    entity_type: Optional[str] = None,
    skip: int = 0,
    limit: int = 50,
) -> Tuple[List[AdminAuditLog], int]:
    """Query the admin audit log with filters and pagination."""
    query = select(AdminAuditLog).options(selectinload(AdminAuditLog.user))
    count_query = select(func.count(AdminAuditLog.id))

    if user_id is not None:
        query = query.where(AdminAuditLog.user_id == user_id)
        count_query = count_query.where(AdminAuditLog.user_id == user_id)
    if action is not None:
        query = query.where(AdminAuditLog.action == action)
        count_query = count_query.where(AdminAuditLog.action == action)
    if entity_type is not None:
        query = query.where(AdminAuditLog.entity_type == entity_type)
        count_query = count_query.where(AdminAuditLog.entity_type == entity_type)

    total = (await db.execute(count_query)).scalar() or 0
    result = await db.execute(
        query.order_by(AdminAuditLog.timestamp.desc()).offset(skip).limit(limit)
    )
    return list(result.scalars().all()), total


# ---------------------------------------------------------------------------
# Registry version (cross-worker synchronization)
# ---------------------------------------------------------------------------


async def get_registry_version(db: AsyncSession) -> int:
    """Read the current registry version (single-row table)."""
    result = await db.execute(
        select(RegistryMeta.version).where(RegistryMeta.id == 1)
    )
    return result.scalar_one_or_none() or 0


async def bump_registry_version(db: AsyncSession) -> None:
    """Atomically increment the registry version."""
    await db.execute(
        update(RegistryMeta)
        .where(RegistryMeta.id == 1)
        .values(version=RegistryMeta.version + 1)
    )
    await db.flush()


# ---------------------------------------------------------------------------
# DLP Alerts
# ---------------------------------------------------------------------------


async def create_dlp_alert(
    db: AsyncSession,
    request_id: Optional[int],
    user_id: Optional[int],
    severity: str,
    scanner: str,
    categories: Optional[list] = None,
    entities: Optional[list] = None,
    confidence: Optional[float] = None,
    scan_latency_ms: Optional[int] = None,
    detail: Optional[str] = None,
) -> DlpAlert:
    """Create a new DLP alert."""
    alert = DlpAlert(
        request_id=request_id,
        user_id=user_id,
        severity=severity,
        scanner=scanner,
        categories=categories,
        entities=entities,
        confidence=confidence,
        scan_latency_ms=scan_latency_ms,
        detail=detail,
    )
    db.add(alert)
    await db.flush()
    return alert


async def get_dlp_alerts(
    db: AsyncSession,
    severity: Optional[str] = None,
    scanner: Optional[str] = None,
    user_id: Optional[int] = None,
    search: Optional[str] = None,
    acknowledged: Optional[bool] = None,
    skip: int = 0,
    limit: int = 50,
) -> Tuple[List[DlpAlert], int]:
    """Query DLP alerts with filters and pagination."""
    query = select(DlpAlert).options(
        selectinload(DlpAlert.user),
        selectinload(DlpAlert.acknowledger),
    )
    count_query = select(func.count(DlpAlert.id))

    if severity is not None:
        query = query.where(DlpAlert.severity == severity)
        count_query = count_query.where(DlpAlert.severity == severity)
    if scanner is not None:
        query = query.where(DlpAlert.scanner == scanner)
        count_query = count_query.where(DlpAlert.scanner == scanner)
    if user_id is not None:
        query = query.where(DlpAlert.user_id == user_id)
        count_query = count_query.where(DlpAlert.user_id == user_id)
    if acknowledged is not None:
        query = query.where(DlpAlert.acknowledged == acknowledged)
        count_query = count_query.where(DlpAlert.acknowledged == acknowledged)
    if search:
        like_pat = f"%{search}%"
        filt = or_(
            DlpAlert.detail.ilike(like_pat),
            DlpAlert.severity.ilike(like_pat),
        )
        query = query.where(filt)
        count_query = count_query.where(filt)

    total = (await db.execute(count_query)).scalar() or 0
    result = await db.execute(
        query.order_by(DlpAlert.scanned_at.desc()).offset(skip).limit(limit)
    )
    return list(result.scalars().all()), total


async def get_dlp_alert_by_id(db: AsyncSession, alert_id: int) -> Optional[DlpAlert]:
    """Get a single DLP alert by ID."""
    result = await db.execute(
        select(DlpAlert)
        .options(selectinload(DlpAlert.user), selectinload(DlpAlert.acknowledger))
        .where(DlpAlert.id == alert_id)
    )
    return result.scalar_one_or_none()


async def acknowledge_dlp_alert(
    db: AsyncSession, alert_id: int, user_id: int
) -> Optional[DlpAlert]:
    """Acknowledge a DLP alert."""
    alert = await get_dlp_alert_by_id(db, alert_id)
    if alert is None:
        return None
    alert.acknowledged = True
    alert.acknowledged_by = user_id
    alert.acknowledged_at = datetime.now(timezone.utc)
    await db.flush()
    return alert


async def get_dlp_stats(db: AsyncSession, hours: int = 24) -> dict:
    """Get DLP statistics for the last N hours."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

    # Total alerts
    total = (await db.execute(
        select(func.count(DlpAlert.id)).where(DlpAlert.scanned_at >= cutoff)
    )).scalar() or 0

    # By severity
    sev_rows = (await db.execute(
        select(DlpAlert.severity, func.count(DlpAlert.id))
        .where(DlpAlert.scanned_at >= cutoff)
        .group_by(DlpAlert.severity)
    )).all()
    by_severity = {row[0]: row[1] for row in sev_rows}

    # By scanner
    scan_rows = (await db.execute(
        select(DlpAlert.scanner, func.count(DlpAlert.id))
        .where(DlpAlert.scanned_at >= cutoff)
        .group_by(DlpAlert.scanner)
    )).all()
    by_scanner = {row[0]: row[1] for row in scan_rows}

    # Average latency
    avg_lat = (await db.execute(
        select(func.avg(DlpAlert.scan_latency_ms))
        .where(DlpAlert.scanned_at >= cutoff)
    )).scalar()

    # Unacknowledged count
    unack = (await db.execute(
        select(func.count(DlpAlert.id))
        .where(DlpAlert.scanned_at >= cutoff)
        .where(DlpAlert.acknowledged == False)
    )).scalar() or 0

    return {
        "total": total,
        "by_severity": by_severity,
        "by_scanner": by_scanner,
        "avg_latency_ms": round(avg_lat, 1) if avg_lat else 0,
        "unacknowledged": unack,
        "hours": hours,
    }
