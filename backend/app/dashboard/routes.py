############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# routes.py: Dashboard web routes and views
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Dashboard routes for MindRouter2."""

import csv
import io
import json
import os
import zoneinfo
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Form, HTTPException, Request, Response, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession

from itsdangerous import URLSafeTimedSerializer

from backend.app.core.scheduler.policy import get_scheduler
from backend.app.core.telemetry.registry import get_registry
from backend.app.dashboard.azure_auth import azure_router
from backend.app.db import crud, chat_crud
from backend.app.db.models import BackendEngine, QuotaRequestStatus, UserRole
from backend.app.db.session import get_async_db
from backend.app.logging_config import get_logger
from backend.app.security import generate_api_key, hash_password, verify_password
from backend.app.settings import get_settings

logger = get_logger(__name__)

dashboard_router = APIRouter(tags=["dashboard"])
dashboard_router.include_router(azure_router)

# Setup templates
templates_path = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=templates_path)
templates.env.filters["fromjson"] = lambda s: json.loads(s) if s else []
templates.env.globals["version"] = get_settings().app_version

# ---------------------------------------------------------------------------
# Timezone filter — converts UTC datetimes to the configured app timezone
# ---------------------------------------------------------------------------
_tz_cache = {"name": "America/Los_Angeles"}


def _refresh_tz_cache_sync(tz_name: str) -> None:
    """Update the cached timezone name (called after admin save)."""
    _tz_cache["name"] = tz_name


def localtime_filter(dt, fmt="%Y-%m-%d %H:%M"):
    """Convert a UTC datetime to the configured timezone and format it."""
    if dt is None:
        return ""
    tz_name = _tz_cache["name"]
    try:
        tz = zoneinfo.ZoneInfo(tz_name)
    except Exception:
        tz = zoneinfo.ZoneInfo("America/Los_Angeles")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=zoneinfo.ZoneInfo("UTC"))
    return dt.astimezone(tz).strftime(fmt)


templates.env.filters["localtime"] = localtime_filter


# Session management helpers using signed cookies
def _get_session_serializer() -> URLSafeTimedSerializer:
    """Get a timed serializer for session cookies."""
    settings = get_settings()
    return URLSafeTimedSerializer(settings.secret_key, salt="session")


def get_session_user_id(request: Request) -> Optional[int]:
    """Get user ID from signed session cookie."""
    session_data = request.cookies.get("mindrouter_session")
    if session_data:
        try:
            serializer = _get_session_serializer()
            user_id = serializer.loads(session_data, max_age=86400 * 7)  # 7 days
            return int(user_id)
        except Exception:
            pass
    return None


def set_session_cookie(response: Response, user_id: int) -> None:
    """Set signed session cookie."""
    serializer = _get_session_serializer()
    signed_value = serializer.dumps(user_id)
    response.set_cookie(
        key="mindrouter_session",
        value=signed_value,
        httponly=True,
        samesite="lax",
        max_age=86400 * 7,  # 7 days
    )


def clear_session_cookie(response: Response) -> None:
    """Clear session cookie."""
    response.delete_cookie(key="mindrouter_session")


# ---------------------------------------------------------------------------
# Masquerade helpers
# ---------------------------------------------------------------------------

_MASQUERADE_COOKIE = "mindrouter_masquerade"


def _get_masquerade_serializer() -> URLSafeTimedSerializer:
    settings = get_settings()
    return URLSafeTimedSerializer(settings.secret_key, salt="masquerade")


def get_masquerade_user_id(request: Request) -> Optional[int]:
    """Return the masquerade target user ID from the signed cookie, or None."""
    cookie = request.cookies.get(_MASQUERADE_COOKIE)
    if cookie:
        try:
            ser = _get_masquerade_serializer()
            return int(ser.loads(cookie, max_age=86400))  # 24h expiry
        except Exception:
            pass
    return None


async def get_effective_user_id(request: Request, db: AsyncSession) -> Optional[int]:
    """Return the masquerade target if the real user is admin, else the real user.

    Only for read-only dashboard views — never for admin routes or actions.
    """
    real_user_id = get_session_user_id(request)
    if not real_user_id:
        return None
    masquerade_id = get_masquerade_user_id(request)
    if masquerade_id and masquerade_id != real_user_id:
        # Verify the real user is actually admin
        real_user = await crud.get_user_by_id(db, real_user_id)
        if real_user and real_user.group and real_user.group.is_admin:
            return masquerade_id
    return real_user_id


# Public Dashboard
@dashboard_router.get("/", response_class=HTMLResponse)
async def public_dashboard(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """Public dashboard showing cluster status."""
    settings = get_settings()

    # Get cluster stats
    try:
        registry = get_registry()
        all_backends = await registry.get_all_backends()
        healthy_backends = await registry.get_healthy_backends()
    except Exception:
        all_backends = []
        healthy_backends = []

    # Get models
    models = set()
    for backend in healthy_backends:
        backend_models = await registry.get_backend_models(backend.id)
        for m in backend_models:
            models.add(m.name)

    # Get scheduler stats
    try:
        scheduler = get_scheduler()
        scheduler_stats = await scheduler.get_stats()
        queue_size = scheduler_stats.get("queue", {}).get("total", 0)
    except Exception:
        queue_size = 0

    # Active users in last 24 hours
    try:
        active_users = await crud.get_active_user_count(db)
    except Exception:
        active_users = 0

    # Look up user for navbar rendering
    user_id = get_session_user_id(request)
    user = None
    if user_id:
        user = await crud.get_user_by_id(db, user_id)

    return templates.TemplateResponse(
        "public/index.html",
        {
            "request": request,
            "app_name": settings.app_name,
            "total_backends": len(all_backends),
            "healthy_backends": len(healthy_backends),
            "models": sorted(models),
            "queue_size": queue_size,
            "active_users": active_users,
            "user": user,
            "user_id": user_id,
        },
    )


@dashboard_router.get("/documentation", response_class=HTMLResponse)
async def documentation(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """Public documentation page."""
    user_id = get_session_user_id(request)
    user = None
    if user_id:
        user = await crud.get_user_by_id(db, user_id)

    return templates.TemplateResponse(
        "public/documentation.html",
        {"request": request, "user": user},
    )


# Authentication
@dashboard_router.get("/login", response_class=HTMLResponse)
async def login_form(request: Request, error: Optional[str] = None):
    """Display login form."""
    settings = get_settings()
    return templates.TemplateResponse(
        "public/login.html",
        {
            "request": request,
            "error": error,
            "azure_enabled": settings.azure_ad_enabled,
        },
    )


@dashboard_router.post("/login")
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: AsyncSession = Depends(get_async_db),
):
    """Handle login."""
    settings = get_settings()
    user = await crud.get_user_by_username(db, username)

    if not user:
        return templates.TemplateResponse(
            "public/login.html",
            {
                "request": request,
                "error": "Invalid username or password",
                "azure_enabled": settings.azure_ad_enabled,
            },
        )

    if user.password_hash is None:
        return templates.TemplateResponse(
            "public/login.html",
            {
                "request": request,
                "error": "Please use University of Idaho sign-in",
                "azure_enabled": settings.azure_ad_enabled,
            },
        )

    if not verify_password(password, user.password_hash):
        return templates.TemplateResponse(
            "public/login.html",
            {
                "request": request,
                "error": "Invalid username or password",
                "azure_enabled": settings.azure_ad_enabled,
            },
        )

    if not user.is_active:
        return templates.TemplateResponse(
            "public/login.html",
            {
                "request": request,
                "error": "Account is inactive",
                "azure_enabled": settings.azure_ad_enabled,
            },
        )

    # Update last login
    user.last_login_at = datetime.now(timezone.utc)
    await db.commit()

    response = RedirectResponse(url="/dashboard", status_code=302)
    set_session_cookie(response, user.id)
    return response


@dashboard_router.get("/logout")
async def logout():
    """Handle logout."""
    response = RedirectResponse(url="/", status_code=302)
    clear_session_cookie(response)
    return response


# User Dashboard
@dashboard_router.get("/dashboard", response_class=HTMLResponse)
async def user_dashboard(
    request: Request,
    pw_error: Optional[str] = None,
    pw_success: Optional[str] = None,
    key_error: Optional[str] = None,
    db: AsyncSession = Depends(get_async_db),
):
    """User dashboard."""
    real_user_id = get_session_user_id(request)
    if not real_user_id:
        return RedirectResponse(url="/login", status_code=302)

    # Support masquerade: admin sees target user's dashboard
    effective_id = await get_effective_user_id(request, db)
    user = await crud.get_user_by_id(db, effective_id)
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    masquerade_user = None
    if effective_id != real_user_id:
        masquerade_user = user  # the target user we're viewing as

    # Get user's API keys
    api_keys = await crud.get_user_api_keys(db, effective_id, include_revoked=True)

    # Get quota — prefer Redis for tokens_used (consistent across workers)
    quota = await crud.get_user_quota(db, effective_id)
    from backend.app.core.redis_client import get_tokens as redis_get_tokens, is_available as redis_is_available
    tokens_used_display = 0
    if quota:
        if redis_is_available():
            redis_val = await redis_get_tokens(effective_id)
            tokens_used_display = redis_val if redis_val is not None else quota.tokens_used
        else:
            tokens_used_display = quota.tokens_used

    # Calculate quota usage percentage using group budget
    group_budget = user.group.token_budget if user.group else 0
    usage_percent = 0
    if quota and group_budget > 0:
        usage_percent = min(100, (tokens_used_display / group_budget) * 100)

    # Key limit info
    max_keys = user.group.max_api_keys if user.group else 8
    active_key_count = await crud.count_user_active_api_keys(db, effective_id)

    return templates.TemplateResponse(
        "user/dashboard.html",
        {
            "request": request,
            "user": user,
            "api_keys": api_keys,
            "quota": quota,
            "tokens_used_display": tokens_used_display,
            "group_budget": group_budget,
            "usage_percent": usage_percent,
            "pw_error": pw_error,
            "pw_success": pw_success,
            "key_error": key_error,
            "max_keys": max_keys,
            "active_key_count": active_key_count,
            "now_utc": datetime.now(timezone.utc),
            "masquerade_user": masquerade_user,
        },
    )


@dashboard_router.get("/dashboard/api/token-usage")
async def dashboard_token_usage(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """Return current token usage for the logged-in user (live polling endpoint)."""
    real_user_id = get_session_user_id(request)
    if not real_user_id:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)

    effective_id = await get_effective_user_id(request, db)
    user = await crud.get_user_by_id(db, effective_id)
    if not user:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)

    quota = await crud.get_user_quota(db, effective_id)
    if not quota:
        return JSONResponse({"tokens_used": 0, "budget": 0})

    from backend.app.core.redis_client import get_tokens as redis_get_tokens, is_available as redis_is_available
    if redis_is_available():
        redis_val = await redis_get_tokens(effective_id)
        tokens_used = redis_val if redis_val is not None else quota.tokens_used
    else:
        tokens_used = quota.tokens_used

    group_budget = user.group.token_budget if user.group else 0
    return JSONResponse({"tokens_used": tokens_used, "budget": group_budget})


@dashboard_router.post("/dashboard/change-password")
async def change_password(
    request: Request,
    current_password: str = Form(...),
    new_password: str = Form(...),
    confirm_password: str = Form(...),
    db: AsyncSession = Depends(get_async_db),
):
    """Change password for local (non-SSO) accounts."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or not user.password_hash:
        return RedirectResponse(url="/dashboard", status_code=302)

    # Validate current password
    if not verify_password(current_password, user.password_hash):
        return RedirectResponse(url="/dashboard?pw_error=Current+password+is+incorrect", status_code=302)

    # Validate new password
    if len(new_password) < 8:
        return RedirectResponse(url="/dashboard?pw_error=New+password+must+be+at+least+8+characters", status_code=302)

    if new_password != confirm_password:
        return RedirectResponse(url="/dashboard?pw_error=Passwords+do+not+match", status_code=302)

    # Update password
    user.password_hash = hash_password(new_password)
    await db.commit()

    return RedirectResponse(url="/dashboard?pw_success=Password+changed+successfully", status_code=302)


@dashboard_router.post("/dashboard/create-key")
async def create_api_key(
    request: Request,
    key_name: str = Form(...),
    db: AsyncSession = Depends(get_async_db),
):
    """Create a new API key."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    # Check key limit
    max_keys = user.group.max_api_keys if user.group else 8
    active_count = await crud.count_user_active_api_keys(db, user_id)
    if active_count >= max_keys:
        return RedirectResponse(url="/dashboard?key_error=limit", status_code=302)

    # Calculate expiration from group settings
    expiry_days = user.group.api_key_expiry_days if user.group else 45
    expires_at = datetime.now(timezone.utc) + timedelta(days=expiry_days)

    # Generate new key
    full_key, key_hash, key_prefix = generate_api_key()

    # Store in database
    await crud.create_api_key(
        db=db,
        user_id=user_id,
        key_hash=key_hash,
        key_prefix=key_prefix,
        name=key_name,
        expires_at=expires_at,
    )
    await db.commit()

    # Show the key to user (only time they'll see it)
    return templates.TemplateResponse(
        "user/key_created.html",
        {
            "request": request,
            "api_key": full_key,
            "key_name": key_name,
            "expires_at": expires_at,
        },
    )


@dashboard_router.post("/dashboard/revoke-key/{key_id}")
async def revoke_key(
    request: Request,
    key_id: int,
    db: AsyncSession = Depends(get_async_db),
):
    """Revoke an API key."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    await crud.revoke_api_key(db, key_id)
    await db.commit()

    return RedirectResponse(url="/dashboard", status_code=302)


@dashboard_router.get("/dashboard/request-quota", response_class=HTMLResponse)
async def request_quota_form(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """Display quota request form."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)

    return templates.TemplateResponse(
        "user/request_quota.html",
        {"request": request, "user": user},
    )


@dashboard_router.post("/dashboard/request-quota")
async def submit_quota_request(
    request: Request,
    requested_tokens: int = Form(...),
    justification: str = Form(...),
    db: AsyncSession = Depends(get_async_db),
):
    """Submit quota increase request."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    await crud.create_quota_request(
        db=db,
        user_id=user_id,
        request_type="quota_increase",
        justification=justification,
        requested_tokens=requested_tokens,
    )
    await db.commit()

    return RedirectResponse(url="/dashboard", status_code=302)


# Admin Dashboard
@dashboard_router.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """Admin dashboard."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.is_admin):
        return RedirectResponse(url="/dashboard", status_code=302)

    # Get backends
    registry = get_registry()
    backends = await registry.get_all_backends()

    # Get nodes
    nodes = await crud.get_all_nodes(db)

    # Get pending requests
    pending_requests = await crud.get_pending_quota_requests(db)

    # Get scheduler stats
    scheduler = get_scheduler()
    scheduler_stats = await scheduler.get_stats()

    return templates.TemplateResponse(
        "admin/dashboard.html",
        {
            "request": request,
            "user": user,
            "backends": backends,
            "nodes": nodes,
            "pending_requests": pending_requests,
            "scheduler_stats": scheduler_stats,
            "is_force_offline": registry.is_force_offline,
        },
    )


@dashboard_router.post("/admin/system/toggle-online")
async def admin_toggle_system_online(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """Toggle MindRouter system online/offline."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.is_admin):
        return RedirectResponse(url="/dashboard", status_code=302)

    registry = get_registry()
    if registry.is_force_offline:
        await registry.force_online()
    else:
        await registry.force_offline()

    return RedirectResponse(url="/admin", status_code=303)


@dashboard_router.get("/admin/users", response_class=HTMLResponse)
async def admin_users(
    request: Request,
    search: Optional[str] = None,
    group_id: Optional[int] = None,
    sort: Optional[str] = None,
    dir: Optional[str] = None,
    page: int = 1,
    db: AsyncSession = Depends(get_async_db),
):
    """Admin user management with search, sort, and pagination."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.is_admin):
        return RedirectResponse(url="/dashboard", status_code=302)

    per_page = 25
    skip = (page - 1) * per_page
    sort_dir = dir if dir in ("asc", "desc") else "desc"
    users, total = await crud.get_users(
        db, skip=skip, limit=per_page, group_id=group_id, search=search,
        sort_by=sort, sort_dir=sort_dir,
    )
    groups = await crud.get_all_groups(db)
    total_pages = max(1, (total + per_page - 1) // per_page)

    # Fetch token totals for this page of users
    user_ids = [u.id for u in users]
    user_tokens = await crud.get_user_token_totals(db, user_ids)

    return templates.TemplateResponse(
        "admin/users.html",
        {
            "request": request,
            "user": user,
            "users": users,
            "groups": groups,
            "total": total,
            "page": page,
            "total_pages": total_pages,
            "search": search or "",
            "group_id": group_id,
            "sort": sort or "",
            "dir": sort_dir,
            "user_tokens": user_tokens,
        },
    )


@dashboard_router.get("/admin/requests", response_class=HTMLResponse)
async def admin_requests(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """Admin request management."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.is_admin):
        return RedirectResponse(url="/dashboard", status_code=302)

    pending_requests = await crud.get_pending_quota_requests(db)

    return templates.TemplateResponse(
        "admin/requests.html",
        {"request": request, "user": user, "requests": pending_requests},
    )


@dashboard_router.post("/admin/requests/{request_id}/approve")
async def approve_request(
    request: Request,
    request_id: int,
    db: AsyncSession = Depends(get_async_db),
):
    """Approve a quota/API key request."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.is_admin):
        return RedirectResponse(url="/dashboard", status_code=302)

    await crud.review_quota_request(
        db=db,
        request_id=request_id,
        reviewer_id=user_id,
        status=QuotaRequestStatus.APPROVED,
    )
    await db.commit()

    return RedirectResponse(url="/admin/requests", status_code=302)


@dashboard_router.post("/admin/requests/{request_id}/deny")
async def deny_request(
    request: Request,
    request_id: int,
    db: AsyncSession = Depends(get_async_db),
):
    """Deny a quota/API key request."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.is_admin):
        return RedirectResponse(url="/dashboard", status_code=302)

    await crud.review_quota_request(
        db=db,
        request_id=request_id,
        reviewer_id=user_id,
        status=QuotaRequestStatus.DENIED,
    )
    await db.commit()

    return RedirectResponse(url="/admin/requests", status_code=302)


@dashboard_router.get("/admin/models", response_class=HTMLResponse)
async def admin_models(
    request: Request,
    success: Optional[str] = None,
    error: Optional[str] = None,
    db: AsyncSession = Depends(get_async_db),
):
    """Admin model management."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.is_admin):
        return RedirectResponse(url="/dashboard", status_code=302)

    grouped_models = await crud.get_models_grouped_by_name(db)

    return templates.TemplateResponse(
        "admin/models.html",
        {
            "request": request,
            "user": user,
            "grouped_models": grouped_models,
            "success": success,
            "error": error,
        },
    )


@dashboard_router.post("/admin/models/toggle-multimodal")
async def toggle_model_multimodal(
    request: Request,
    model_name: str = Form(...),
    db: AsyncSession = Depends(get_async_db),
):
    """Toggle the multimodal override for all instances of a model."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.is_admin):
        return RedirectResponse(url="/dashboard", status_code=302)

    # Get current state from first model with this name
    from backend.app.db.models import Model
    from sqlalchemy import select
    result = await db.execute(select(Model).where(Model.name == model_name).limit(1))
    model = result.scalar_one_or_none()
    if not model:
        return RedirectResponse(
            url="/admin/models?error=Model+not+found", status_code=302
        )

    # Toggle: flip the current supports_multimodal value and set as override for all
    new_value = not model.supports_multimodal
    await crud.set_multimodal_override_by_name(db, model_name, new_value)
    await db.commit()

    return RedirectResponse(
        url="/admin/models?success=updated", status_code=302
    )


@dashboard_router.post("/admin/models/reset-multimodal")
async def reset_model_multimodal(
    request: Request,
    model_name: str = Form(...),
    db: AsyncSession = Depends(get_async_db),
):
    """Reset multimodal override to auto-detect for all instances of a model."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.is_admin):
        return RedirectResponse(url="/dashboard", status_code=302)

    await crud.set_multimodal_override_by_name(db, model_name, None)
    await db.commit()

    return RedirectResponse(
        url="/admin/models?success=reset", status_code=302
    )


@dashboard_router.post("/admin/models/toggle-thinking")
async def toggle_model_thinking(
    request: Request,
    model_name: str = Form(...),
    db: AsyncSession = Depends(get_async_db),
):
    """Toggle the thinking override for all instances of a model."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.is_admin):
        return RedirectResponse(url="/dashboard", status_code=302)

    from backend.app.db.models import Model
    from sqlalchemy import select
    result = await db.execute(select(Model).where(Model.name == model_name).limit(1))
    model = result.scalar_one_or_none()
    if not model:
        return RedirectResponse(
            url="/admin/models?error=Model+not+found", status_code=302
        )

    new_value = not model.supports_thinking
    await crud.set_thinking_override_by_name(db, model_name, new_value)
    await db.commit()

    return RedirectResponse(
        url="/admin/models?success=updated", status_code=302
    )


@dashboard_router.post("/admin/models/reset-thinking")
async def reset_model_thinking(
    request: Request,
    model_name: str = Form(...),
    db: AsyncSession = Depends(get_async_db),
):
    """Reset thinking override to auto-detect for all instances of a model."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.is_admin):
        return RedirectResponse(url="/dashboard", status_code=302)

    await crud.set_thinking_override_by_name(db, model_name, None)
    await db.commit()

    return RedirectResponse(
        url="/admin/models?success=reset", status_code=302
    )


@dashboard_router.post("/admin/models/update-metadata")
async def update_model_metadata(
    request: Request,
    model_name: str = Form(...),
    family: Optional[str] = Form(None),
    parameter_count: Optional[str] = Form(None),
    quantization: Optional[str] = Form(None),
    context_length: Optional[str] = Form(None),
    embedding_length: Optional[str] = Form(None),
    head_count: Optional[str] = Form(None),
    layer_count: Optional[str] = Form(None),
    feed_forward_length: Optional[str] = Form(None),
    model_format: Optional[str] = Form(None),
    parent_model: Optional[str] = Form(None),
    capabilities: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    model_url: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_async_db),
):
    """Update metadata overrides for all instances of a model."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.is_admin):
        return RedirectResponse(url="/dashboard", status_code=302)

    def _str_or_none(val: Optional[str]) -> Optional[str]:
        return val.strip() if val and val.strip() else None

    def _int_or_none(val: Optional[str]) -> Optional[int]:
        if val and val.strip():
            try:
                return int(val.strip())
            except ValueError:
                return None
        return None

    # Build overrides dict — empty string means clear the override (None)
    overrides = {
        "family_override": _str_or_none(family),
        "parameter_count_override": _str_or_none(parameter_count),
        "quantization_override": _str_or_none(quantization),
        "context_length_override": _int_or_none(context_length),
        "embedding_length_override": _int_or_none(embedding_length),
        "head_count_override": _int_or_none(head_count),
        "layer_count_override": _int_or_none(layer_count),
        "feed_forward_length_override": _int_or_none(feed_forward_length),
        "model_format_override": _str_or_none(model_format),
        "parent_model_override": _str_or_none(parent_model),
        # Direct fields (admin-only)
        "description": _str_or_none(description),
        "model_url": _str_or_none(model_url),
    }

    # Capabilities: comma-separated -> JSON array, or None to clear
    cap_str = _str_or_none(capabilities)
    if cap_str:
        cap_list = [c.strip() for c in cap_str.split(",") if c.strip()]
        overrides["capabilities_override"] = json.dumps(cap_list)
    else:
        overrides["capabilities_override"] = None

    await crud.update_model_overrides_by_name(db, model_name, overrides)
    await db.commit()

    return RedirectResponse(
        url="/admin/models?success=metadata_updated", status_code=302
    )


@dashboard_router.post("/admin/models/reset-overrides")
async def reset_model_overrides(
    request: Request,
    model_name: str = Form(...),
    db: AsyncSession = Depends(get_async_db),
):
    """Reset all metadata overrides for all instances of a model."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.is_admin):
        return RedirectResponse(url="/dashboard", status_code=302)

    overrides = {
        "family_override": None,
        "parameter_count_override": None,
        "quantization_override": None,
        "context_length_override": None,
        "embedding_length_override": None,
        "head_count_override": None,
        "layer_count_override": None,
        "feed_forward_length_override": None,
        "model_format_override": None,
        "parent_model_override": None,
        "capabilities_override": None,
        "description": None,
        "model_url": None,
    }
    await crud.update_model_overrides_by_name(db, model_name, overrides)
    await db.commit()

    return RedirectResponse(
        url="/admin/models?success=overrides_reset", status_code=302
    )


@dashboard_router.get("/admin/nodes", response_class=HTMLResponse)
async def admin_nodes(
    request: Request,
    success: Optional[str] = None,
    error: Optional[str] = None,
    db: AsyncSession = Depends(get_async_db),
):
    """Admin node management."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.is_admin):
        return RedirectResponse(url="/dashboard", status_code=302)

    nodes = await crud.get_all_nodes(db)

    # Build node data with backends and GPU devices
    node_data = []
    for node in nodes:
        # Get backends on this node
        all_backends = await crud.get_all_backends(db)
        node_backends = [b for b in all_backends if b.node_id == node.id]
        gpu_devices = await crud.get_gpu_devices_for_node(db, node.id)

        node_data.append({
            "node": node,
            "backends": node_backends,
            "gpu_devices": gpu_devices,
        })

    settings = get_settings()
    return templates.TemplateResponse(
        "admin/nodes.html",
        {
            "request": request,
            "user": user,
            "nodes": node_data,
            "app_version": settings.app_version,
            "success": success,
            "error": error,
        },
    )


@dashboard_router.post("/admin/nodes/register")
async def register_node(
    request: Request,
    name: str = Form(...),
    hostname: Optional[str] = Form(None),
    sidecar_url: Optional[str] = Form(None),
    sidecar_key: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_async_db),
):
    """Register a new node."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.is_admin):
        return RedirectResponse(url="/dashboard", status_code=302)

    try:
        existing = await crud.get_node_by_name(db, name)
        if existing:
            return RedirectResponse(
                url="/admin/nodes?error=Node+name+already+exists", status_code=302
            )

        hostname_val = hostname if hostname else None
        sidecar_url_val = sidecar_url if sidecar_url else None
        sidecar_key_val = sidecar_key if sidecar_key else None

        registry = get_registry()
        await registry.register_node(
            name=name,
            hostname=hostname_val,
            sidecar_url=sidecar_url_val,
            sidecar_key=sidecar_key_val,
        )
        return RedirectResponse(url="/admin/nodes?success=registered", status_code=302)
    except Exception:
        return RedirectResponse(url="/admin/nodes?error=Registration+failed", status_code=302)


@dashboard_router.post("/admin/nodes/{node_id}/edit")
async def edit_node(
    request: Request,
    node_id: int,
    name: str = Form(...),
    hostname: Optional[str] = Form(None),
    sidecar_url: Optional[str] = Form(None),
    sidecar_key: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_async_db),
):
    """Edit an existing node."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.is_admin):
        return RedirectResponse(url="/dashboard", status_code=302)

    try:
        kwargs = {"name": name}
        clear_fields = []

        hostname_val = hostname if hostname else None
        if hostname_val is not None:
            kwargs["hostname"] = hostname_val
        else:
            clear_fields.append("hostname")

        sidecar_url_val = sidecar_url if sidecar_url else None
        if sidecar_url_val is not None:
            kwargs["sidecar_url"] = sidecar_url_val
        else:
            clear_fields.append("sidecar_url")

        # Empty sidecar_key means "keep current" — only update if provided
        if sidecar_key and sidecar_key.strip():
            kwargs["sidecar_key"] = sidecar_key.strip()

        if clear_fields:
            kwargs["_clear_fields"] = clear_fields

        registry = get_registry()
        await registry.update_node(node_id, **kwargs)
        return RedirectResponse(url="/admin/nodes?success=updated", status_code=302)
    except Exception as e:
        error_msg = str(e).replace(" ", "+")
        return RedirectResponse(url=f"/admin/nodes?error={error_msg}", status_code=302)


@dashboard_router.post("/admin/nodes/{node_id}/remove")
async def remove_node(
    request: Request,
    node_id: int,
    db: AsyncSession = Depends(get_async_db),
):
    """Remove a node."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.is_admin):
        return RedirectResponse(url="/dashboard", status_code=302)

    registry = get_registry()
    removed = await registry.remove_node(node_id)
    if removed:
        return RedirectResponse(url="/admin/nodes?success=removed", status_code=302)
    else:
        return RedirectResponse(
            url="/admin/nodes?error=Cannot+remove+node+with+active+backends", status_code=302
        )


@dashboard_router.post("/admin/nodes/{node_id}/refresh")
async def refresh_node(
    request: Request,
    node_id: int,
    db: AsyncSession = Depends(get_async_db),
):
    """Refresh node sidecar data."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.is_admin):
        return RedirectResponse(url="/dashboard", status_code=302)

    registry = get_registry()
    await registry.refresh_node(node_id)
    return RedirectResponse(url="/admin/nodes?success=refreshed", status_code=302)


@dashboard_router.post("/admin/nodes/{node_id}/take-offline")
async def take_node_offline(
    request: Request,
    node_id: int,
    db: AsyncSession = Depends(get_async_db),
):
    """Take a node offline: disable all its backends and mark node offline."""
    user_id = get_session_user_id(request)
    if not user_id:
        return JSONResponse({"ok": False, "error": "Not authenticated"}, status_code=401)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.is_admin):
        return JSONResponse({"ok": False, "error": "Forbidden"}, status_code=403)

    from backend.app.db.models import NodeStatus

    all_backends = await crud.get_all_backends(db)
    node_backends = [b for b in all_backends if b.node_id == node_id]

    registry = get_registry()
    for b in node_backends:
        await registry.disable_backend(b.id)

    await crud.update_node_status(db, node_id, NodeStatus.OFFLINE)
    await db.commit()
    return JSONResponse({"ok": True})


@dashboard_router.post("/admin/nodes/{node_id}/bring-online")
async def bring_node_online(
    request: Request,
    node_id: int,
    db: AsyncSession = Depends(get_async_db),
):
    """Bring a node back online: re-enable all its backends."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.is_admin):
        return RedirectResponse(url="/dashboard", status_code=302)

    from backend.app.db.models import NodeStatus

    all_backends = await crud.get_all_backends(db)
    node_backends = [b for b in all_backends if b.node_id == node_id]

    registry = get_registry()
    for b in node_backends:
        await registry.enable_backend(b.id)

    await crud.update_node_status(db, node_id, NodeStatus.ONLINE)
    await db.commit()
    return RedirectResponse(url="/admin/nodes?success=brought_online", status_code=302)


@dashboard_router.get("/admin/nodes/{node_id}/active-requests")
async def node_active_requests(
    request: Request,
    node_id: int,
    db: AsyncSession = Depends(get_async_db),
):
    """Return the count of active (in-flight) requests on this node's backends."""
    user_id = get_session_user_id(request)
    if not user_id:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.is_admin):
        return JSONResponse({"error": "Forbidden"}, status_code=403)

    all_backends = await crud.get_all_backends(db)
    backend_ids = [b.id for b in all_backends if b.node_id == node_id]

    count = await crud.get_active_request_count_for_node_backends(db, backend_ids)
    return JSONResponse({"count": count})


@dashboard_router.post("/admin/nodes/{node_id}/force-drain")
async def force_drain_node(
    request: Request,
    node_id: int,
    db: AsyncSession = Depends(get_async_db),
):
    """Force-cancel all active requests on this node's backends."""
    user_id = get_session_user_id(request)
    if not user_id:
        return JSONResponse({"ok": False, "error": "Not authenticated"}, status_code=401)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.is_admin):
        return JSONResponse({"ok": False, "error": "Forbidden"}, status_code=403)

    all_backends = await crud.get_all_backends(db)
    backend_ids = [b.id for b in all_backends if b.node_id == node_id]

    cancelled = await crud.cancel_active_requests_for_backends(db, backend_ids)
    await db.commit()
    return JSONResponse({"ok": True, "cancelled": cancelled})


@dashboard_router.get("/admin/backends", response_class=HTMLResponse)
async def admin_backends(
    request: Request,
    success: Optional[str] = None,
    error: Optional[str] = None,
    db: AsyncSession = Depends(get_async_db),
):
    """Admin backend management."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.is_admin):
        return RedirectResponse(url="/dashboard", status_code=302)

    registry = get_registry()
    backends = await registry.get_all_backends()
    nodes = await crud.get_all_nodes(db)

    # Get telemetry for each backend
    backend_data = []
    for backend in backends:
        telemetry = await registry.get_telemetry(backend.id)
        models = await registry.get_backend_models(backend.id)
        backend_data.append({
            "backend": backend,
            "telemetry": telemetry,
            "models": models,
        })

    return templates.TemplateResponse(
        "admin/backends.html",
        {
            "request": request,
            "user": user,
            "backends": backend_data,
            "nodes": nodes,
            "success": success,
            "error": error,
        },
    )


@dashboard_router.post("/admin/backends/register")
async def register_backend(
    request: Request,
    name: str = Form(...),
    url: str = Form(...),
    engine: str = Form(...),
    max_concurrent: int = Form(4),
    gpu_memory_gb: Optional[str] = Form(None),
    gpu_type: Optional[str] = Form(None),
    node_id: Optional[str] = Form(None),
    gpu_indices: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_async_db),
):
    """Register a new backend."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.is_admin):
        return RedirectResponse(url="/dashboard", status_code=302)

    try:
        engine_enum = BackendEngine(engine)
        gpu_mem = float(gpu_memory_gb) if gpu_memory_gb else None
        gpu_type_val = gpu_type if gpu_type else None

        # Parse node_id
        node_id_val = int(node_id) if node_id else None

        # Parse gpu_indices (e.g., "0,1,2" -> [0, 1, 2])
        gpu_indices_val = None
        if gpu_indices and gpu_indices.strip():
            gpu_indices_val = [int(x.strip()) for x in gpu_indices.split(",") if x.strip()]

        # Check for duplicate name
        existing = await crud.get_backend_by_name(db, name)
        if existing:
            return RedirectResponse(
                url="/admin/backends?error=Backend+name+already+exists", status_code=302
            )

        registry = get_registry()
        await registry.register_backend(
            name=name,
            url=url,
            engine=engine_enum,
            max_concurrent=max_concurrent,
            gpu_memory_gb=gpu_mem,
            gpu_type=gpu_type_val,
            node_id=node_id_val,
            gpu_indices=gpu_indices_val,
        )
        return RedirectResponse(url="/admin/backends?success=registered", status_code=302)
    except Exception:
        return RedirectResponse(url="/admin/backends?error=Registration+failed", status_code=302)


@dashboard_router.post("/admin/backends/{backend_id}/edit")
async def edit_backend(
    request: Request,
    backend_id: int,
    name: str = Form(...),
    url: str = Form(...),
    engine: str = Form(...),
    max_concurrent: int = Form(4),
    gpu_memory_gb: Optional[str] = Form(None),
    gpu_type: Optional[str] = Form(None),
    node_id: Optional[str] = Form(None),
    gpu_indices: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_async_db),
):
    """Edit an existing backend."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.is_admin):
        return RedirectResponse(url="/dashboard", status_code=302)

    try:
        engine_enum = BackendEngine(engine)
        gpu_mem = float(gpu_memory_gb) if gpu_memory_gb else None
        gpu_type_val = gpu_type if gpu_type else None
        node_id_val = int(node_id) if node_id else None

        gpu_indices_val = None
        if gpu_indices and gpu_indices.strip():
            gpu_indices_val = [int(x.strip()) for x in gpu_indices.split(",") if x.strip()]

        kwargs = {
            "name": name,
            "url": url,
            "engine": engine_enum,
            "max_concurrent": max_concurrent,
        }
        clear_fields = []
        if gpu_mem is not None:
            kwargs["gpu_memory_gb"] = gpu_mem
        else:
            clear_fields.append("gpu_memory_gb")
        if gpu_type_val is not None:
            kwargs["gpu_type"] = gpu_type_val
        else:
            clear_fields.append("gpu_type")
        if node_id_val is not None:
            kwargs["node_id"] = node_id_val
        else:
            clear_fields.append("node_id")
        if gpu_indices_val is not None:
            kwargs["gpu_indices"] = gpu_indices_val
        else:
            clear_fields.append("gpu_indices")
        if clear_fields:
            kwargs["_clear_fields"] = clear_fields

        registry = get_registry()
        await registry.update_backend(backend_id, **kwargs)
        return RedirectResponse(url="/admin/backends?success=updated", status_code=302)
    except Exception as e:
        error_msg = str(e).replace(" ", "+")
        return RedirectResponse(url=f"/admin/backends?error={error_msg}", status_code=302)


@dashboard_router.post("/admin/backends/{backend_id}/disable")
async def disable_backend(
    request: Request,
    backend_id: int,
    db: AsyncSession = Depends(get_async_db),
):
    """Disable a backend."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.is_admin):
        return RedirectResponse(url="/dashboard", status_code=302)

    registry = get_registry()
    await registry.disable_backend(backend_id)
    return RedirectResponse(url="/admin/backends?success=disabled", status_code=302)


@dashboard_router.post("/admin/backends/{backend_id}/drain")
async def drain_backend(
    request: Request,
    backend_id: int,
    db: AsyncSession = Depends(get_async_db),
):
    """Start draining a backend (stop new requests, let in-flight finish)."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.is_admin):
        return RedirectResponse(url="/dashboard", status_code=302)

    registry = get_registry()
    await registry.drain_backend(backend_id)
    return RedirectResponse(url="/admin/backends?success=draining", status_code=302)


@dashboard_router.post("/admin/backends/{backend_id}/enable")
async def enable_backend(
    request: Request,
    backend_id: int,
    db: AsyncSession = Depends(get_async_db),
):
    """Enable a backend."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.is_admin):
        return RedirectResponse(url="/dashboard", status_code=302)

    registry = get_registry()
    await registry.enable_backend(backend_id)
    return RedirectResponse(url="/admin/backends?success=enabled", status_code=302)


@dashboard_router.post("/admin/backends/{backend_id}/remove")
async def remove_backend(
    request: Request,
    backend_id: int,
    db: AsyncSession = Depends(get_async_db),
):
    """Remove/unregister a backend."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.is_admin):
        return RedirectResponse(url="/dashboard", status_code=302)

    registry = get_registry()
    removed = await registry.remove_backend(backend_id)
    if removed:
        return RedirectResponse(url="/admin/backends?success=removed", status_code=302)
    else:
        return RedirectResponse(url="/admin/backends?error=Backend+not+found", status_code=302)


@dashboard_router.post("/admin/backends/{backend_id}/refresh")
async def refresh_backend(
    request: Request,
    backend_id: int,
    db: AsyncSession = Depends(get_async_db),
):
    """Refresh backend capabilities."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.is_admin):
        return RedirectResponse(url="/dashboard", status_code=302)

    registry = get_registry()
    await registry.refresh_backend(backend_id)
    return RedirectResponse(url="/admin/backends?success=refreshed", status_code=302)


@dashboard_router.get("/admin/metrics", response_class=HTMLResponse)
async def admin_metrics(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """Admin GPU metrics dashboard."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.is_admin):
        return RedirectResponse(url="/dashboard", status_code=302)

    return templates.TemplateResponse(
        "admin/metrics.html",
        {"request": request, "user": user},
    )


@dashboard_router.get("/admin/audit", response_class=HTMLResponse)
async def admin_audit(
    request: Request,
    search: Optional[str] = None,
    user_id_filter: Optional[int] = None,
    model_filter: Optional[str] = None,
    status_filter: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    page: int = 1,
    db: AsyncSession = Depends(get_async_db),
):
    """Admin audit log viewer with filters and pagination."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.is_admin):
        return RedirectResponse(url="/dashboard", status_code=302)

    from backend.app.db.models import RequestStatus

    per_page = 50
    skip = (page - 1) * per_page

    # Parse filters
    parsed_status = None
    if status_filter:
        try:
            parsed_status = RequestStatus(status_filter)
        except ValueError:
            pass
    parsed_start = None
    parsed_end = None
    if start_date:
        try:
            parsed_start = datetime.fromisoformat(start_date)
        except ValueError:
            pass
    if end_date:
        try:
            parsed_end = datetime.fromisoformat(end_date)
        except ValueError:
            pass

    audit_requests, total = await crud.search_requests(
        db,
        user_id=user_id_filter,
        model=model_filter,
        status=parsed_status,
        start_date=parsed_start,
        end_date=parsed_end,
        search_text=search,
        skip=skip,
        limit=per_page,
    )
    total_pages = max(1, (total + per_page - 1) // per_page)

    return templates.TemplateResponse(
        "admin/audit.html",
        {
            "request": request,
            "user": user,
            "audit_requests": audit_requests,
            "total": total,
            "page": page,
            "total_pages": total_pages,
            "search": search or "",
            "user_id_filter": user_id_filter or "",
            "model_filter": model_filter or "",
            "status_filter": status_filter or "",
            "start_date": start_date or "",
            "end_date": end_date or "",
        },
    )


@dashboard_router.get("/admin/audit/export")
async def admin_audit_export(
    request: Request,
    format: str = "csv",
    search: Optional[str] = None,
    user_id_filter: Optional[int] = None,
    model_filter: Optional[str] = None,
    status_filter: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    include_content: bool = False,
    db: AsyncSession = Depends(get_async_db),
):
    """Export audit log as CSV or JSON with current filters."""
    session_user_id = get_session_user_id(request)
    if not session_user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, session_user_id)
    if not user or (not user.group or not user.group.is_admin):
        return RedirectResponse(url="/dashboard", status_code=302)

    from backend.app.db.models import RequestStatus

    # Parse filters
    parsed_status = None
    if status_filter:
        try:
            parsed_status = RequestStatus(status_filter)
        except ValueError:
            pass
    parsed_start = None
    parsed_end = None
    if start_date:
        try:
            parsed_start = datetime.fromisoformat(start_date)
        except ValueError:
            pass
    if end_date:
        try:
            parsed_end = datetime.fromisoformat(end_date)
        except ValueError:
            pass

    audit_requests, _ = await crud.search_requests(
        db,
        user_id=user_id_filter,
        model=model_filter,
        status=parsed_status,
        start_date=parsed_start,
        end_date=parsed_end,
        search_text=search,
        skip=0,
        limit=10000,
    )

    # Build export rows
    rows = []
    for req in audit_requests:
        row = {
            "request_uuid": req.request_uuid,
            "created_at": req.created_at.isoformat() if req.created_at else "",
            "user_id": req.user_id,
            "model": req.model,
            "endpoint": req.endpoint,
            "status": req.status.value if req.status else "",
            "prompt_tokens": req.prompt_tokens or 0,
            "completion_tokens": req.completion_tokens or 0,
            "total_tokens": (req.prompt_tokens or 0) + (req.completion_tokens or 0),
            "total_time_ms": req.total_time_ms or "",
            "error_message": req.error_message or "",
        }
        if include_content:
            row["messages"] = json.dumps(req.messages) if req.messages else ""
            row["prompt"] = req.prompt or ""
            row["parameters"] = json.dumps(req.parameters) if req.parameters else ""
            row["response_content"] = req.response.content if req.response else ""
            row["finish_reason"] = req.response.finish_reason if req.response else ""
        rows.append(row)

    if format == "json":
        content = json.dumps(rows, indent=2)
        return StreamingResponse(
            io.BytesIO(content.encode()),
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=audit_log.json"},
        )

    # CSV format
    output = io.StringIO()
    fieldnames = [
        "request_uuid", "created_at", "user_id", "model", "endpoint",
        "status", "prompt_tokens", "completion_tokens", "total_tokens",
        "total_time_ms", "error_message",
    ]
    if include_content:
        fieldnames.extend(["messages", "prompt", "parameters", "response_content", "finish_reason"])
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in fieldnames})

    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=audit_log.csv"},
    )


@dashboard_router.get("/admin/audit/{request_uuid}/detail")
async def admin_audit_detail(
    request: Request,
    request_uuid: str,
    db: AsyncSession = Depends(get_async_db),
):
    """Get request detail with prompt/response content (JSON API for AJAX expand)."""
    session_user_id = get_session_user_id(request)
    if not session_user_id:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)

    user = await crud.get_user_by_id(db, session_user_id)
    if not user or (not user.group or not user.group.is_admin):
        return JSONResponse({"error": "Forbidden"}, status_code=403)

    from sqlalchemy import select
    from sqlalchemy.orm import selectinload
    from backend.app.db.models import Request as RequestModel

    result = await db.execute(
        select(RequestModel)
        .where(RequestModel.request_uuid == request_uuid)
        .options(selectinload(RequestModel.response))
    )
    req = result.scalar_one_or_none()
    if not req:
        return JSONResponse({"error": "Request not found"}, status_code=404)

    detail = {
        "messages": req.messages,
        "prompt": req.prompt,
        "parameters": req.parameters,
        "response_content": req.response.content if req.response else None,
        "finish_reason": req.response.finish_reason if req.response else None,
        "error_message": req.error_message,
    }
    return JSONResponse(detail)


# Group management routes
@dashboard_router.get("/admin/groups", response_class=HTMLResponse)
async def admin_groups(
    request: Request,
    success: Optional[str] = None,
    error: Optional[str] = None,
    db: AsyncSession = Depends(get_async_db),
):
    """Admin group management."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.is_admin):
        return RedirectResponse(url="/dashboard", status_code=302)

    groups_with_counts = await crud.get_all_groups_with_counts(db)

    return templates.TemplateResponse(
        "admin/groups.html",
        {
            "request": request,
            "user": user,
            "groups": groups_with_counts,
            "success": success,
            "error": error,
        },
    )


@dashboard_router.post("/admin/groups")
async def create_group(
    request: Request,
    name: str = Form(...),
    display_name: str = Form(...),
    description: Optional[str] = Form(None),
    token_budget: int = Form(100000),
    rpm_limit: int = Form(30),
    max_concurrent: int = Form(2),
    scheduler_weight: int = Form(1),
    is_admin: Optional[str] = Form(None),
    api_key_expiry_days: int = Form(45),
    max_api_keys: int = Form(8),
    db: AsyncSession = Depends(get_async_db),
):
    """Create a new group."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.is_admin):
        return RedirectResponse(url="/dashboard", status_code=302)

    try:
        existing = await crud.get_group_by_name(db, name)
        if existing:
            return RedirectResponse(url="/admin/groups?error=Group+name+already+exists", status_code=302)

        await crud.create_group(
            db,
            name=name,
            display_name=display_name,
            description=description if description else None,
            token_budget=token_budget,
            rpm_limit=rpm_limit,
            max_concurrent=max_concurrent,
            scheduler_weight=scheduler_weight,
            is_admin=(is_admin == "on"),
            api_key_expiry_days=api_key_expiry_days,
            max_api_keys=max_api_keys,
        )
        await db.commit()
        return RedirectResponse(url="/admin/groups?success=created", status_code=302)
    except Exception:
        return RedirectResponse(url="/admin/groups?error=Creation+failed", status_code=302)


@dashboard_router.post("/admin/groups/{group_id}/edit")
async def edit_group(
    request: Request,
    group_id: int,
    display_name: str = Form(...),
    description: Optional[str] = Form(None),
    token_budget: int = Form(100000),
    rpm_limit: int = Form(30),
    max_concurrent: int = Form(2),
    scheduler_weight: int = Form(1),
    is_admin: Optional[str] = Form(None),
    api_key_expiry_days: int = Form(45),
    max_api_keys: int = Form(8),
    db: AsyncSession = Depends(get_async_db),
):
    """Edit a group."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.is_admin):
        return RedirectResponse(url="/dashboard", status_code=302)

    try:
        await crud.update_group(
            db, group_id,
            display_name=display_name,
            description=description if description else None,
            token_budget=token_budget,
            rpm_limit=rpm_limit,
            max_concurrent=max_concurrent,
            scheduler_weight=scheduler_weight,
            is_admin=(is_admin == "on"),
            api_key_expiry_days=api_key_expiry_days,
            max_api_keys=max_api_keys,
        )
        await db.commit()
        return RedirectResponse(url="/admin/groups?success=updated", status_code=302)
    except Exception as e:
        error_msg = str(e).replace(" ", "+")
        return RedirectResponse(url=f"/admin/groups?error={error_msg}", status_code=302)


@dashboard_router.post("/admin/groups/{group_id}/delete")
async def delete_group(
    request: Request,
    group_id: int,
    db: AsyncSession = Depends(get_async_db),
):
    """Delete a group."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.is_admin):
        return RedirectResponse(url="/dashboard", status_code=302)

    deleted = await crud.delete_group(db, group_id)
    if deleted:
        await db.commit()
        return RedirectResponse(url="/admin/groups?success=deleted", status_code=302)
    else:
        return RedirectResponse(
            url="/admin/groups?error=Cannot+delete+group+with+active+users", status_code=302
        )


# User detail route
@dashboard_router.get("/admin/users/{user_id}", response_class=HTMLResponse)
async def admin_user_detail(
    request: Request,
    user_id: int,
    success: Optional[str] = None,
    error: Optional[str] = None,
    db: AsyncSession = Depends(get_async_db),
):
    """Admin user detail page."""
    session_user_id = get_session_user_id(request)
    if not session_user_id:
        return RedirectResponse(url="/login", status_code=302)

    admin_user = await crud.get_user_by_id(db, session_user_id)
    if not admin_user or (not admin_user.group or not admin_user.group.is_admin):
        return RedirectResponse(url="/dashboard", status_code=302)

    stats = await crud.get_user_with_stats(db, user_id)
    if not stats:
        return RedirectResponse(url="/admin/users", status_code=302)

    monthly_usage = await crud.get_user_monthly_usage(db, user_id)
    groups = await crud.get_all_groups(db)
    recent_ips = await crud.get_user_recent_ips(db, user_id, days=90)

    return templates.TemplateResponse(
        "admin/user_detail.html",
        {
            "request": request,
            "user": admin_user,
            "detail_user": stats["user"],
            "stats": stats,
            "monthly_usage": monthly_usage,
            "groups": groups,
            "recent_ips": recent_ips,
            "success": success,
            "error": error,
        },
    )


@dashboard_router.post("/admin/users/{user_id}/edit")
async def edit_user(
    request: Request,
    user_id: int,
    group_id: int = Form(...),
    full_name: Optional[str] = Form(None),
    college: Optional[str] = Form(None),
    department: Optional[str] = Form(None),
    intended_use: Optional[str] = Form(None),
    rpm_limit: Optional[int] = Form(None),
    max_concurrent: Optional[int] = Form(None),
    weight_override: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_async_db),
):
    """Edit user profile and quota."""
    session_user_id = get_session_user_id(request)
    if not session_user_id:
        return RedirectResponse(url="/login", status_code=302)

    admin_user = await crud.get_user_by_id(db, session_user_id)
    if not admin_user or (not admin_user.group or not admin_user.group.is_admin):
        return RedirectResponse(url="/dashboard", status_code=302)

    try:
        # Update user fields
        await crud.update_user(
            db, user_id,
            group_id=group_id,
            full_name=full_name if full_name else None,
            college=college if college else None,
            department=department if department else None,
            intended_use=intended_use if intended_use else None,
        )

        # Update quota if provided
        quota = await crud.get_user_quota(db, user_id)
        if quota:
            if rpm_limit is not None:
                quota.rpm_limit = rpm_limit
            if max_concurrent is not None:
                quota.max_concurrent = max_concurrent
            quota.weight_override = int(weight_override) if weight_override and weight_override.strip() else None
            await db.flush()

        await db.commit()
        return RedirectResponse(url=f"/admin/users/{user_id}?success=updated", status_code=302)
    except Exception as e:
        error_msg = str(e).replace(" ", "+")
        return RedirectResponse(url=f"/admin/users/{user_id}?error={error_msg}", status_code=302)


# ---------------------------------------------------------------------------
# Admin Masquerade
# ---------------------------------------------------------------------------

@dashboard_router.post("/admin/masquerade/stop")
async def admin_masquerade_stop(request: Request):
    """Stop masquerading and return to admin view."""
    logger.info("masquerade_stop", admin_user_id=get_session_user_id(request))
    response = RedirectResponse(url="/admin/users", status_code=302)
    response.delete_cookie(key=_MASQUERADE_COOKIE)
    return response


@dashboard_router.post("/admin/masquerade/{target_user_id}")
async def admin_masquerade_start(
    request: Request,
    target_user_id: int,
    db: AsyncSession = Depends(get_async_db),
):
    """Start masquerading as the target user (admin only)."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    admin_user = await crud.get_user_by_id(db, user_id)
    if not admin_user or not admin_user.group or not admin_user.group.is_admin:
        return RedirectResponse(url="/dashboard", status_code=302)

    # Verify target user exists
    target = await crud.get_user_by_id(db, target_user_id)
    if not target:
        return RedirectResponse(url="/admin/users?error=User+not+found", status_code=302)

    logger.info("masquerade_start", admin_user_id=user_id, target_user_id=target_user_id)

    ser = _get_masquerade_serializer()
    signed = ser.dumps(target_user_id)
    response = RedirectResponse(url="/dashboard", status_code=302)
    response.set_cookie(
        key=_MASQUERADE_COOKIE,
        value=signed,
        httponly=True,
        samesite="lax",
        max_age=86400,  # 24h
    )
    return response


# API Keys listing route
@dashboard_router.get("/admin/api-keys", response_class=HTMLResponse)
async def admin_api_keys(
    request: Request,
    search: Optional[str] = None,
    key_status: Optional[str] = None,
    sort: Optional[str] = None,
    dir: Optional[str] = None,
    page: int = 1,
    db: AsyncSession = Depends(get_async_db),
):
    """Admin API key listing with sort and IP tracking."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.is_admin):
        return RedirectResponse(url="/dashboard", status_code=302)

    per_page = 50
    skip = (page - 1) * per_page
    sort_dir = dir if dir in ("asc", "desc") else "desc"
    keys, total = await crud.get_all_api_keys(
        db, skip=skip, limit=per_page, search=search, status_filter=key_status,
        sort_by=sort, sort_dir=sort_dir,
    )
    total_pages = max(1, (total + per_page - 1) // per_page)

    # Fetch last IP for each key on this page
    key_ids = [k.id for k in keys]
    key_last_ips = await crud.get_api_key_last_ips_batch(db, key_ids)

    return templates.TemplateResponse(
        "admin/api_keys.html",
        {
            "request": request,
            "user": user,
            "api_keys": keys,
            "total": total,
            "page": page,
            "total_pages": total_pages,
            "search": search or "",
            "key_status": key_status or "",
            "sort": sort or "",
            "dir": sort_dir,
            "key_last_ips": key_last_ips,
            "now_utc": datetime.now(timezone.utc),
        },
    )


# ---------------------------------------------------------------------------
# Admin Conversation Viewer
# ---------------------------------------------------------------------------

@dashboard_router.get("/admin/conversations", response_class=HTMLResponse)
async def admin_conversations(
    request: Request,
    search: Optional[str] = None,
    user_id_filter: Optional[int] = None,
    model_filter: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    page: int = 1,
    db: AsyncSession = Depends(get_async_db),
):
    """Admin conversation list with search, filter, and pagination."""
    session_user_id = get_session_user_id(request)
    if not session_user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, session_user_id)
    if not user or (not user.group or not user.group.is_admin):
        return RedirectResponse(url="/dashboard", status_code=302)

    per_page = 50
    skip = (page - 1) * per_page

    # Parse dates
    parsed_start = None
    parsed_end = None
    if start_date:
        try:
            parsed_start = datetime.fromisoformat(start_date)
        except ValueError:
            pass
    if end_date:
        try:
            parsed_end = datetime.fromisoformat(end_date)
        except ValueError:
            pass

    conversations, total = await chat_crud.search_conversations_admin(
        db,
        user_id=user_id_filter,
        model=model_filter,
        search_text=search,
        start_date=parsed_start,
        end_date=parsed_end,
        skip=skip,
        limit=per_page,
    )
    total_pages = max(1, (total + per_page - 1) // per_page)

    return templates.TemplateResponse(
        "admin/conversations.html",
        {
            "request": request,
            "user": user,
            "conversations": conversations,
            "total": total,
            "page": page,
            "total_pages": total_pages,
            "search": search or "",
            "user_id_filter": user_id_filter or "",
            "model_filter": model_filter or "",
            "start_date": start_date or "",
            "end_date": end_date or "",
        },
    )


@dashboard_router.get("/admin/conversations/{conversation_id}/messages")
async def admin_conversation_messages(
    request: Request,
    conversation_id: int,
    db: AsyncSession = Depends(get_async_db),
):
    """Get conversation messages (JSON API for AJAX expand/reveal)."""
    session_user_id = get_session_user_id(request)
    if not session_user_id:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)

    user = await crud.get_user_by_id(db, session_user_id)
    if not user or (not user.group or not user.group.is_admin):
        return JSONResponse({"error": "Forbidden"}, status_code=403)

    messages = await chat_crud.get_conversation_messages_admin(db, conversation_id)
    if messages is None:
        return JSONResponse({"error": "Conversation not found"}, status_code=404)

    return JSONResponse({"messages": messages})


# ---------------------------------------------------------------------------
# Admin Conversation Export
# ---------------------------------------------------------------------------

@dashboard_router.get("/admin/conversations/export")
async def admin_conversations_export(
    request: Request,
    format: str = "csv",
    search: Optional[str] = None,
    user_id_filter: Optional[int] = None,
    model_filter: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    include_content: bool = False,
    db: AsyncSession = Depends(get_async_db),
):
    """Export conversations as CSV or JSON."""
    session_user_id = get_session_user_id(request)
    if not session_user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, session_user_id)
    if not user or (not user.group or not user.group.is_admin):
        return RedirectResponse(url="/dashboard", status_code=302)

    # Parse dates
    parsed_start = None
    parsed_end = None
    if start_date:
        try:
            parsed_start = datetime.fromisoformat(start_date)
        except ValueError:
            pass
    if end_date:
        try:
            parsed_end = datetime.fromisoformat(end_date)
        except ValueError:
            pass

    conversations, _ = await chat_crud.search_conversations_admin(
        db,
        user_id=user_id_filter,
        model=model_filter,
        search_text=search,
        start_date=parsed_start,
        end_date=parsed_end,
        skip=0,
        limit=10000,
    )

    # Enrich with messages if requested
    if include_content:
        for conv in conversations:
            msgs = await chat_crud.get_conversation_messages_admin(db, conv["id"])
            conv["messages"] = msgs or []

    if format == "json":
        # Serialize datetimes
        def default_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return str(obj)

        content = json.dumps(conversations, default=default_serializer, indent=2)
        return StreamingResponse(
            io.BytesIO(content.encode()),
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=conversations.json"},
        )

    # CSV format
    output = io.StringIO()
    fieldnames = ["id", "user_id", "username", "title", "model", "message_count", "created_at", "updated_at"]
    if include_content:
        fieldnames.append("messages")
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for conv in conversations:
        row = {k: conv.get(k, "") for k in fieldnames}
        if "created_at" in row and hasattr(row["created_at"], "isoformat"):
            row["created_at"] = row["created_at"].isoformat()
        if "updated_at" in row and hasattr(row["updated_at"], "isoformat"):
            row["updated_at"] = row["updated_at"].isoformat()
        if include_content and "messages" in row:
            row["messages"] = json.dumps(row["messages"], default=str)
        writer.writerow(row)

    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=conversations.csv"},
    )


@dashboard_router.get("/api/admin/conversations/export")
async def api_admin_conversations_export(
    request: Request,
    search: Optional[str] = None,
    user_id_filter: Optional[int] = None,
    model_filter: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    include_content: bool = True,
    db: AsyncSession = Depends(get_async_db),
):
    """Bulk API export endpoint for programmatic access (JSON with content)."""
    return await admin_conversations_export(
        request=request,
        format="json",
        search=search,
        user_id_filter=user_id_filter,
        model_filter=model_filter,
        start_date=start_date,
        end_date=end_date,
        include_content=include_content,
        db=db,
    )


# ---------------------------------------------------------------------------
# Admin Chat Config
# ---------------------------------------------------------------------------

@dashboard_router.get("/admin/chat-config", response_class=HTMLResponse)
async def admin_chat_config(
    request: Request,
    success: Optional[str] = None,
    error: Optional[str] = None,
    db: AsyncSession = Depends(get_async_db),
):
    """Admin chat configuration page."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.is_admin):
        return RedirectResponse(url="/dashboard", status_code=302)

    # Get distinct model names from healthy backends (exclude embedding)
    registry = get_registry()
    backends = await registry.get_healthy_backends()
    available_models = set()
    for backend in backends:
        backend_models = await registry.get_backend_models(backend.id)
        for m in backend_models:
            if "embed" not in m.name.lower():
                available_models.add(m.name)

    core_models = await crud.get_config_json(db, "chat.core_models", [])
    default_model = await crud.get_config_json(db, "chat.default_model", None)
    system_prompt = await crud.get_config_json(db, "chat.system_prompt", None)
    chat_max_tokens = await crud.get_config_json(db, "chat.max_tokens", 16384)
    chat_temperature = await crud.get_config_json(db, "chat.temperature", None)
    chat_think = await crud.get_config_json(db, "chat.think", None)

    return templates.TemplateResponse(
        "admin/chat_config.html",
        {
            "request": request,
            "user": user,
            "available_models": sorted(available_models),
            "core_models": core_models,
            "default_model": default_model,
            "system_prompt": system_prompt,
            "chat_max_tokens": chat_max_tokens,
            "chat_temperature": chat_temperature,
            "chat_think": chat_think,
            "success": success,
            "error": error,
        },
    )


@dashboard_router.post("/admin/chat-config")
async def admin_chat_config_post(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """Handle chat config form submissions."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.is_admin):
        return RedirectResponse(url="/dashboard", status_code=302)

    form = await request.form()
    action = form.get("action")

    if action == "set_default":
        default_model = form.get("default_model", "")
        await crud.set_config(db, "chat.default_model", default_model if default_model else None)
        await db.commit()
        return RedirectResponse(url="/admin/chat-config?success=default_updated", status_code=302)

    elif action == "set_core_models":
        selected = form.getlist("core_models")
        await crud.set_config(db, "chat.core_models", selected)
        await db.commit()
        return RedirectResponse(url="/admin/chat-config?success=core_updated", status_code=302)

    elif action == "set_system_prompt":
        prompt_text = form.get("system_prompt", "").strip()
        if prompt_text:
            await crud.set_config(db, "chat.system_prompt", prompt_text)
        else:
            # Blank = remove override, revert to built-in default
            await crud.set_config(db, "chat.system_prompt", None)
        await db.commit()
        return RedirectResponse(url="/admin/chat-config?success=system_prompt_updated", status_code=302)

    elif action == "reset_system_prompt":
        await crud.set_config(db, "chat.system_prompt", None)
        await db.commit()
        return RedirectResponse(url="/admin/chat-config?success=system_prompt_reset", status_code=302)

    elif action == "set_max_tokens":
        try:
            val = int(form.get("max_tokens", "16384"))
            val = max(256, min(131072, val))
        except (ValueError, TypeError):
            val = 16384
        await crud.set_config(db, "chat.max_tokens", val)
        await db.commit()
        return RedirectResponse(url="/admin/chat-config?success=max_tokens_updated", status_code=302)

    elif action == "set_temperature":
        temp_str = form.get("temperature", "").strip()
        if temp_str:
            try:
                temp_val = float(temp_str)
                temp_val = max(0.0, min(2.0, temp_val))
            except (ValueError, TypeError):
                return RedirectResponse(url="/admin/chat-config?error=Invalid+temperature+value", status_code=302)
            await crud.set_config(db, "chat.temperature", temp_val)
        else:
            await crud.set_config(db, "chat.temperature", None)
        await db.commit()
        return RedirectResponse(url="/admin/chat-config?success=temperature_updated", status_code=302)

    elif action == "set_think":
        think_str = form.get("think", "").strip()
        if think_str == "true":
            think_val = True
        elif think_str == "false":
            think_val = False
        elif think_str in ("low", "medium", "high"):
            think_val = think_str
        else:
            think_val = None
        await crud.set_config(db, "chat.think", think_val)
        await db.commit()
        return RedirectResponse(url="/admin/chat-config?success=think_updated", status_code=302)

    return RedirectResponse(url="/admin/chat-config?error=Unknown+action", status_code=302)


# ---------------------------------------------------------------------------
# Admin Site Settings (timezone, etc.)
# ---------------------------------------------------------------------------

# Common timezone choices grouped by region
_TIMEZONE_CHOICES = [
    ("US", [
        "America/New_York",
        "America/Chicago",
        "America/Denver",
        "America/Los_Angeles",
        "America/Anchorage",
        "Pacific/Honolulu",
    ]),
    ("Americas", [
        "America/Toronto",
        "America/Vancouver",
        "America/Mexico_City",
        "America/Sao_Paulo",
        "America/Argentina/Buenos_Aires",
    ]),
    ("Europe", [
        "Europe/London",
        "Europe/Paris",
        "Europe/Berlin",
        "Europe/Moscow",
    ]),
    ("Asia / Pacific", [
        "Asia/Tokyo",
        "Asia/Shanghai",
        "Asia/Kolkata",
        "Asia/Singapore",
        "Australia/Sydney",
        "Pacific/Auckland",
    ]),
    ("Other", [
        "UTC",
    ]),
]


@dashboard_router.get("/admin/settings", response_class=HTMLResponse)
async def admin_settings(
    request: Request,
    success: Optional[str] = None,
    error: Optional[str] = None,
    db: AsyncSession = Depends(get_async_db),
):
    """Admin site settings page."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.is_admin):
        return RedirectResponse(url="/dashboard", status_code=302)

    current_tz = await crud.get_config_json(db, "app.timezone", "America/Los_Angeles")

    # Show current time in configured timezone as preview
    now_in_tz = localtime_filter(datetime.now(timezone.utc), "%Y-%m-%d %H:%M:%S %Z")

    return templates.TemplateResponse(
        "admin/settings.html",
        {
            "request": request,
            "user": user,
            "current_timezone": current_tz,
            "timezone_choices": _TIMEZONE_CHOICES,
            "now_in_tz": now_in_tz,
            "success": success,
            "error": error,
        },
    )


@dashboard_router.post("/admin/settings")
async def admin_settings_post(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """Handle site settings form submissions."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.is_admin):
        return RedirectResponse(url="/dashboard", status_code=302)

    form = await request.form()
    action = form.get("action")

    if action == "set_timezone":
        tz_name = form.get("timezone", "").strip()
        # Validate timezone
        try:
            zoneinfo.ZoneInfo(tz_name)
        except Exception:
            return RedirectResponse(
                url="/admin/settings?error=Invalid+timezone", status_code=302
            )
        await crud.set_config(
            db, "app.timezone", tz_name, description="IANA timezone for displaying dates in the web UI"
        )
        await db.commit()
        _refresh_tz_cache_sync(tz_name)
        return RedirectResponse(url="/admin/settings?success=timezone_updated", status_code=302)

    return RedirectResponse(url="/admin/settings?error=Unknown+action", status_code=302)


async def _init_tz_cache(db: AsyncSession) -> None:
    """Load timezone from DB into cache. Call at startup or first request."""
    tz_name = await crud.get_config_json(db, "app.timezone", "America/Los_Angeles")
    _refresh_tz_cache_sync(tz_name)
