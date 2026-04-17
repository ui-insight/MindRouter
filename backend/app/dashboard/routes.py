############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
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

"""Dashboard routes for MindRouter."""

import csv
import io
import json
import os
import re
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
from backend.app.db.models import ApiKeyStatus, BackendEngine, QuotaRequestStatus, ServiceKeyRequestStatus, UserRole
from backend.app.db.session import get_async_db, get_async_db_context
from backend.app.logging_config import get_logger
from backend.app.security import generate_api_key, hash_password, verify_password
from backend.app.settings import get_settings

logger = get_logger(__name__)

# Strong-reference set for fire-and-forget background tasks.  asyncio
# keeps only weak references to tasks created via ``create_task`` (see
# https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task),
# so without this set a long-running task like a manual retention cycle
# can be garbage-collected mid-execution and silently disappear.
_background_tasks: set = set()

dashboard_router = APIRouter(tags=["dashboard"])
dashboard_router.include_router(azure_router)

# Setup templates
templates_path = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=templates_path)
templates.env.filters["fromjson"] = lambda s: json.loads(s) if s else []
def _mini_md(s: str) -> str:
    """Lightweight markdown: bold, bullets, URLs, paragraphs."""
    if not s:
        return ""
    import markupsafe
    lines = s.split("\n")
    parts = []
    in_list = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_list:
                parts.append("</ul>")
                in_list = False
            continue
        # Bullet points
        bullet = re.match(r'^[-*]\s+(.*)', stripped)
        if bullet:
            if not in_list:
                parts.append('<ul class="mb-1">')
                in_list = True
            content = markupsafe.escape(bullet.group(1))
            content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', str(content))
            content = re.sub(r'(https?://[^\s<>)]+)', r'<a href="\1" target="_blank" rel="noopener">\1</a>', str(content))
            parts.append(f"<li>{content}</li>")
        else:
            if in_list:
                parts.append("</ul>")
                in_list = False
            content = markupsafe.escape(stripped)
            content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', str(content))
            content = re.sub(r'(https?://[^\s<>)]+)', r'<a href="\1" target="_blank" rel="noopener">\1</a>', str(content))
            parts.append(f"<p class='mb-1'>{content}</p>")
    if in_list:
        parts.append("</ul>")
    return "".join(parts)

templates.env.filters["urlize"] = lambda s: re.sub(
    r'(https?://[^\s<>\)]+)',
    r'<a href="\1" target="_blank" rel="noopener">\1</a>',
    s or "",
)
templates.env.filters["mini_md"] = _mini_md
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


def get_client_ip(request: Request) -> Optional[str]:
    """Extract the real client IP from proxy headers, falling back to direct connection.

    Checks X-Forwarded-For (first entry) → X-Real-IP → request.client.host.
    """
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()
    return request.client.host if request.client else None


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
        if real_user and real_user.group and real_user.group.has_admin_read:
            return masquerade_id
    return real_user_id


async def _admin_masquerade_context(request: Request, real_user: "User", db: AsyncSession) -> dict:
    """Return masquerade-aware context keys for admin templates.

    Returns dict with:
      - is_read_only: True when the effective user lacks is_admin
      - masquerade_user: the target User object if masquerading, else None
      - pending_requests_total: count of all pending requests (quota + service key)
    """
    # Count pending requests for sidebar badge (best-effort, don't break page on error)
    pending_total = 0
    try:
        pending_quota = await crud.count_pending_quota_requests(db)
        pending_service = await crud.count_pending_service_key_requests(db)
        pending_total = pending_quota + pending_service
    except Exception:
        pass

    masquerade_id = get_masquerade_user_id(request)
    if masquerade_id and masquerade_id != real_user.id:
        target_user = await crud.get_user_by_id(db, masquerade_id)
        if target_user and target_user.group:
            return {
                "is_read_only": not target_user.group.is_admin,
                "masquerade_user": target_user,
                "pending_requests_total": pending_total,
            }
    return {
        "is_read_only": not real_user.group.is_admin,
        "masquerade_user": None,
        "pending_requests_total": pending_total,
    }


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

    # Total tokens ever served — use live Redis counter
    total_tokens = 0
    try:
        from backend.app.core import redis_client
        totals = await redis_client.get_cluster_tokens()
        if totals:
            total_tokens = totals["total_tokens"]
            offset = await crud.get_config_json(db, "stats.token_offset", 0)
            if offset:
                total_tokens += int(offset)
        else:
            global_tokens = await crud.get_global_token_total(db)
            total_tokens = global_tokens["total_tokens"]
    except Exception:
        total_tokens = 0

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
            "total_tokens": total_tokens,
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


@dashboard_router.get("/models", response_class=HTMLResponse)
async def models_catalog(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """Public models catalog page."""
    user_id = get_session_user_id(request)
    user = None
    if user_id:
        user = await crud.get_user_by_id(db, user_id)

    registry = get_registry()
    backends = await registry.get_healthy_backends()

    model_data: dict = {}
    for backend in backends:
        backend_models = await registry.get_backend_models(backend.id)
        for model in backend_models:
            if model.name not in model_data:
                model_data[model.name] = {
                    "capabilities": {
                        "multimodal": False,
                        "embeddings": False,
                        "structured_output": True,
                        "thinking": False,
                        "tools": False,
                    },
                    "context_length": None,
                    "model_max_context": None,
                    "parameter_count": None,
                    "quantization": None,
                    "family": None,
                    "description": None,
                    "model_url": None,
                    "huggingface_url": None,
                    "modality": None,
                    "model_format": None,
                    "embedding_length": None,
                    "engines": set(),
                    "backend_count": 0,
                }

            md = model_data[model.name]
            md["backend_count"] += 1
            md["engines"].add(backend.engine.value)

            if model.supports_multimodal:
                md["capabilities"]["multimodal"] = True
            if model.supports_thinking:
                md["capabilities"]["thinking"] = True
            if model.supports_tools:
                md["capabilities"]["tools"] = True
            if "embed" in model.name.lower():
                md["capabilities"]["embeddings"] = True

            if model.context_length is not None:
                cur = md["context_length"]
                if cur is None or model.context_length > cur:
                    md["context_length"] = model.context_length

            if model.model_max_context is not None:
                cur = md["model_max_context"]
                if cur is None or model.model_max_context > cur:
                    md["model_max_context"] = model.model_max_context

            if model.parameter_count and not md["parameter_count"]:
                md["parameter_count"] = model.parameter_count
            if model.quantization and not md["quantization"]:
                md["quantization"] = model.quantization
            if model.family and not md["family"]:
                md["family"] = model.family
            if model.description and not md["description"]:
                md["description"] = model.description
            if model.model_url and not md["model_url"]:
                md["model_url"] = model.model_url
            if model.huggingface_url and not md["huggingface_url"]:
                md["huggingface_url"] = model.huggingface_url
            if model.modality and not md["modality"]:
                md["modality"] = model.modality.value
            if model.model_format and not md["model_format"]:
                md["model_format"] = model.model_format
            if model.embedding_length and not md["embedding_length"]:
                md["embedding_length"] = model.embedding_length

    # Build list for JSON serialization
    models_list = []
    for name, data in sorted(model_data.items()):
        modality = data["modality"] or "chat"
        if modality == "embedding":
            category = "Embedding"
        elif modality == "reranking":
            category = "Reranking"
        else:
            category = "LLM"
        models_list.append({
            "name": name,
            "family": data["family"],
            "description": data["description"],
            "model_url": data["model_url"],
            "huggingface_url": data["huggingface_url"],
            "category": category,
            "engines": sorted(data["engines"]),
            "capabilities": data["capabilities"],
            "context_length": data["context_length"],
            "model_max_context": data["model_max_context"],
            "parameter_count": data["parameter_count"],
            "quantization": data["quantization"],
            "model_format": data["model_format"],
            "embedding_length": data["embedding_length"],
            "backend_count": data["backend_count"],
        })

    # Token usage for popularity chart — read from Redis cache only.
    # The background _cache_warm_loop populates this every 30 min.
    # Never block on a DB scan here; if cache is empty, show no chart.
    token_totals = []
    try:
        from backend.app.core import redis_client
        import json as _json
        if redis_client.is_available() and redis_client._redis:
            cached = await redis_client._redis.get("cache:model_token_totals")
            if cached:
                token_totals = _json.loads(cached)
    except Exception:
        token_totals = []

    return templates.TemplateResponse(
        "public/models.html",
        {
            "request": request,
            "user": user,
            "models_json": json.dumps(models_list),
            "token_totals_json": json.dumps(token_totals),
        },
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

    # Redirect to agreement page if user hasn't accepted current version
    target = "/dashboard/agreement" if await _needs_agreement(db, user) else "/dashboard"
    response = RedirectResponse(url=target, status_code=302)
    set_session_cookie(response, user.id)
    return response


@dashboard_router.get("/logout")
async def logout():
    """Handle logout."""
    response = RedirectResponse(url="/", status_code=302)
    clear_session_cookie(response)
    return response


# ---------------------------------------------------------------------------
# Use Agreement
# ---------------------------------------------------------------------------


async def _needs_agreement(db, user) -> bool:
    """Return True if the user needs to accept the current agreement."""
    agreement = await crud.get_agreement(db)
    if agreement["version"] is None:
        return False  # no agreement configured
    return user.agreement_version_accepted != agreement["version"]


@dashboard_router.get("/dashboard/agreement", response_class=HTMLResponse)
async def agreement_page(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """Show the use agreement for acceptance."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    agreement = await crud.get_agreement(db)
    if agreement["version"] is None or user.agreement_version_accepted == agreement["version"]:
        return RedirectResponse(url="/dashboard", status_code=302)

    return templates.TemplateResponse(
        "user/agreement.html",
        {
            "request": request,
            "user": user,
            "agreement_text": agreement["text"],
            "agreement_version": agreement["version"],
        },
    )


@dashboard_router.post("/dashboard/agreement")
async def accept_agreement(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """Handle agreement acceptance."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    form = await request.form()
    submitted_version = int(form.get("version", 0))

    # Verify the submitted version matches the current version
    agreement = await crud.get_agreement(db)
    if agreement["version"] is None:
        return RedirectResponse(url="/dashboard", status_code=302)

    if submitted_version != agreement["version"]:
        # Agreement was updated while user was reading — show the new version
        return RedirectResponse(url="/dashboard/agreement", status_code=302)

    await crud.accept_agreement(db, user_id, submitted_version)
    await db.commit()
    return RedirectResponse(url="/dashboard", status_code=302)


@dashboard_router.get("/dashboard/use-agreement", response_class=HTMLResponse)
async def view_agreement(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """Read-only view of the current use agreement."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    agreement = await crud.get_agreement(db)
    return templates.TemplateResponse(
        "user/use_agreement.html",
        {
            "request": request,
            "user": user,
            "agreement_text": agreement["text"],
            "agreement_version": agreement["version"],
        },
    )


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

    # Check agreement (skip for admin masquerade)
    if not masquerade_user and await _needs_agreement(db, user):
        return RedirectResponse(url="/dashboard/agreement", status_code=302)

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
    max_keys = user.group.max_api_keys if user.group else 16
    active_key_count = await crud.count_user_active_api_keys(db, effective_id)

    # Lifetime token usage: use the higher of the monotonic counter or SUM(requests)
    # (the counter may have been seeded low; SUM(requests) may shrink after retention)
    lifetime_map = await crud.get_user_token_totals(db, [effective_id])
    lifetime_data = lifetime_map.get(effective_id, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    lifetime_counter = quota.lifetime_tokens_used if quota else 0
    lifetime_tokens = max(lifetime_counter, lifetime_data["total_tokens"])

    # TTS preferences
    tts_enabled = await crud.get_config_json(db, "voice.tts_enabled", False)
    user_tts_voice = await crud.get_config_json(db, f"user.{effective_id}.tts_voice", "")
    user_tts_speed = await crud.get_config_json(db, f"user.{effective_id}.tts_speed", None)

    # Email opt-out preference
    email_optout = await crud.get_config_json(db, f"user.{effective_id}.email_optout", "")

    # Service key requests for this user (to show pending status)
    service_key_requests = await crud.get_user_service_key_requests(db, effective_id)

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
            "lifetime_tokens": lifetime_tokens,
            "lifetime_prompt_tokens": lifetime_data["prompt_tokens"],
            "lifetime_completion_tokens": lifetime_data["completion_tokens"],
            "tts_enabled": tts_enabled,
            "user_tts_voice": user_tts_voice or "",
            "user_tts_speed": user_tts_speed,
            "email_optout": email_optout,
            "service_key_requests": service_key_requests,
        },
    )


@dashboard_router.get("/api/tts-voices")
async def api_tts_voices(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """Return available TTS voices from upstream service or config fallback.

    Query params:
        allowed_only=true  — filter to only voices in voice_api.tts_voices config
                             (used by user dashboard to restrict choices)
    """
    user_id = get_session_user_id(request)
    if not user_id:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)

    allowed_only = request.query_params.get("allowed_only", "").lower() == "true"

    # Load the admin-configured allowed voices list
    tts_voices_str = await crud.get_config_json(
        db, "voice_api.tts_voices", "af_heart\naf_bella\nam_adam\nam_michael"
    )
    allowed_voices = {v.strip() for v in tts_voices_str.split("\n") if v.strip()}

    voices = []
    source = "config"

    tts_url = await crud.get_config_json(db, "voice.tts_url", None)
    if tts_url:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{tts_url.rstrip('/')}/v1/audio/voices")
                resp.raise_for_status()
                data = resp.json()
                # Kokoro returns {"voices": [{"id": "af_heart", ...}, ...]}
                voices_raw = data.get("voices", [])
                if voices_raw and isinstance(voices_raw[0], dict):
                    voices = [v.get("id") or v.get("name", "") for v in voices_raw]
                else:
                    voices = [str(v) for v in voices_raw]
                voices = [v for v in voices if v]
                if voices:
                    source = "upstream"
        except Exception:
            pass

    # Fallback to config list if upstream failed
    if not voices:
        voices = sorted(allowed_voices)

    # Filter to allowed voices when requested
    if allowed_only and allowed_voices:
        voices = [v for v in voices if v in allowed_voices]

    default_voice = await crud.get_config_json(db, "voice_api.default_voice", "af_heart")
    return JSONResponse({"voices": sorted(voices), "source": source, "default_voice": default_voice})


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
        return JSONResponse({"tokens_used": 0, "budget": 0, "lifetime_tokens": 0})

    from backend.app.core.redis_client import get_tokens as redis_get_tokens, is_available as redis_is_available
    if redis_is_available():
        redis_val = await redis_get_tokens(effective_id)
        tokens_used = redis_val if redis_val is not None else quota.tokens_used
    else:
        tokens_used = quota.tokens_used

    lifetime_map = await crud.get_user_token_totals(db, [effective_id])
    lifetime_data = lifetime_map.get(effective_id, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    lifetime_counter = quota.lifetime_tokens_used
    lifetime_tokens = max(lifetime_counter, lifetime_data["total_tokens"])

    group_budget = user.group.token_budget if user.group else 0
    return JSONResponse({
        "tokens_used": tokens_used,
        "budget": group_budget,
        "lifetime_tokens": lifetime_tokens,
        "lifetime_prompt_tokens": lifetime_data["prompt_tokens"],
        "lifetime_completion_tokens": lifetime_data["completion_tokens"],
    })


@dashboard_router.post("/dashboard/save-preference")
async def save_preference(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """Save a user preference (e.g. tts_voice)."""
    user_id = get_session_user_id(request)
    if not user_id:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)
    effective_id = await get_effective_user_id(request, db)
    body = await request.json()
    key = body.get("key", "")
    value = body.get("value", "")
    allowed_keys = {"tts_voice", "tts_speed", "email_optout"}
    if key not in allowed_keys:
        return JSONResponse({"error": f"Invalid preference key: {key}"}, status_code=400)
    config_key = f"user.{effective_id}.{key}"
    if value:
        await crud.set_config(db, config_key, value)
    else:
        # Empty value = reset to default (delete the config)
        await crud.set_config(db, config_key, None)
    await db.commit()
    return JSONResponse({"ok": True})


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
    max_keys = user.group.max_api_keys if user.group else 16
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


@dashboard_router.post("/dashboard/request-service-key")
async def submit_service_key_request(
    request: Request,
    api_key_id: int = Form(...),
    service_name: str = Form(...),
    reason: str = Form(...),
    alternative_contacts: str = Form(""),
    data_risk_level: str = Form("low"),
    compliance_other: str = Form(""),
    db: AsyncSession = Depends(get_async_db),
):
    """Submit a request to promote an API key to a service key."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    # Get form data for compliance_tags (multi-select)
    form = await request.form()
    tags = form.getlist("compliance_tags")
    compliance_tags_json = json.dumps(tags) if tags else None

    # Validate key belongs to user, is active, not already service
    key = await crud.get_api_key_by_id(db, api_key_id)
    if not key or key.user_id != user_id:
        return RedirectResponse(url="/dashboard", status_code=302)
    if key.status != ApiKeyStatus.ACTIVE or key.is_service:
        return RedirectResponse(url="/dashboard", status_code=302)

    # Check for existing pending request for this key
    user_requests = await crud.get_user_service_key_requests(db, user_id)
    for r in user_requests:
        if r.api_key_id == api_key_id and r.status == ServiceKeyRequestStatus.PENDING:
            return RedirectResponse(url="/dashboard", status_code=302)

    contacts_json = json.dumps([c.strip() for c in alternative_contacts.split(",") if c.strip()]) if alternative_contacts.strip() else None

    await crud.create_service_key_request(
        db=db,
        api_key_id=api_key_id,
        user_id=user_id,
        service_name=service_name,
        reason=reason,
        alternative_contacts=contacts_json,
        data_risk_level=data_risk_level,
        compliance_tags=compliance_tags_json,
        compliance_other=compliance_other.strip() or None,
    )
    await db.commit()

    return RedirectResponse(url="/dashboard", status_code=302)


@dashboard_router.post("/dashboard/request-service-key-revocation/{key_id}")
async def request_service_key_revocation(
    request: Request,
    key_id: int,
    reason: str = Form(""),
    db: AsyncSession = Depends(get_async_db),
):
    """User requests revocation of their service key."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    # Validate key belongs to user and is a service key
    key = await crud.get_api_key_by_id(db, key_id)
    if not key or key.user_id != user_id or not key.is_service:
        return RedirectResponse(url="/dashboard", status_code=302)

    await crud.request_service_key_revocation(db, key_id, reason or "User requested revocation")
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
    if not user or (not user.group or not user.group.has_admin_read):
        return RedirectResponse(url="/dashboard", status_code=302)

    # Get backends
    registry = get_registry()
    backends = await registry.get_all_backends()

    # Get nodes
    nodes = await crud.get_all_nodes(db)

    # Get pending requests (quota + service key)
    pending_requests = await crud.get_pending_quota_requests(db)
    pending_service_key_requests = await crud.get_pending_service_key_requests(db)

    # Get scheduler stats
    scheduler = get_scheduler()
    scheduler_stats = await scheduler.get_stats()

    masq = await _admin_masquerade_context(request, user, db)
    return templates.TemplateResponse(
        "admin/dashboard.html",
        {
            "request": request,
            "user": user,
            **masq,
            "backends": backends,
            "nodes": nodes,
            "pending_requests": pending_requests,
            "pending_service_key_requests": pending_service_key_requests,
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
    was_offline = registry.is_force_offline
    if was_offline:
        await registry.force_online()
    else:
        await registry.force_offline()

    async with get_async_db_context() as audit_db:
        await crud.log_admin_action(
            audit_db, user_id=user_id, action="system.toggle_online",
            entity_type="system", detail=f"{'online' if was_offline else 'offline'}",
            ip_address=get_client_ip(request),
        )
        await audit_db.commit()

    return RedirectResponse(url="/admin", status_code=303)


@dashboard_router.get("/admin/users", response_class=HTMLResponse)
async def admin_users(
    request: Request,
    search: Optional[str] = None,
    group_id: Optional[str] = None,
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
    if not user or (not user.group or not user.group.has_admin_read):
        return RedirectResponse(url="/dashboard", status_code=302)

    parsed_group_id = int(group_id) if group_id and group_id.strip().isdigit() else None
    per_page = 25
    skip = (page - 1) * per_page
    sort_dir = dir if dir in ("asc", "desc") else "desc"
    users, total = await crud.get_users(
        db, skip=skip, limit=per_page, group_id=parsed_group_id, search=search,
        sort_by=sort, sort_dir=sort_dir,
    )
    groups = await crud.get_all_groups(db)
    total_pages = max(1, (total + per_page - 1) // per_page)

    # Fetch token totals for this page of users
    user_ids = [u.id for u in users]
    user_tokens = await crud.get_user_token_totals(db, user_ids)

    # Fetch quota tokens (current period) for each user from Redis, falling back to DB
    from backend.app.core.redis_client import get_tokens as redis_get_tokens, is_available as redis_is_available
    user_quota_tokens = {}
    for u in users:
        quota = await crud.get_user_quota(db, u.id)
        if quota:
            if redis_is_available():
                redis_val = await redis_get_tokens(u.id)
                user_quota_tokens[u.id] = redis_val if redis_val is not None else quota.tokens_used
            else:
                user_quota_tokens[u.id] = quota.tokens_used
        else:
            user_quota_tokens[u.id] = 0

    masq = await _admin_masquerade_context(request, user, db)
    return templates.TemplateResponse(
        "admin/users.html",
        {
            "request": request,
            "user": user,
            **masq,
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
            "user_quota_tokens": user_quota_tokens,
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
    if not user or (not user.group or not user.group.has_admin_read):
        return RedirectResponse(url="/dashboard", status_code=302)

    pending_requests = await crud.get_pending_quota_requests(db)
    pending_service_key_requests = await crud.get_pending_service_key_requests(db)

    masq = await _admin_masquerade_context(request, user, db)
    return templates.TemplateResponse(
        "admin/requests.html",
        {
            "request": request,
            "user": user,
            **masq,
            "requests": pending_requests,
            "service_key_requests": pending_service_key_requests,
        },
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
    await crud.log_admin_action(
        db, user_id=user_id, action="quota.approve",
        entity_type="quota_request", entity_id=str(request_id),
        ip_address=get_client_ip(request),
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
    await crud.log_admin_action(
        db, user_id=user_id, action="quota.deny",
        entity_type="quota_request", entity_id=str(request_id),
        ip_address=get_client_ip(request),
    )
    await db.commit()

    return RedirectResponse(url="/admin/requests", status_code=302)


@dashboard_router.post("/admin/service-key-requests/{request_id}/approve")
async def approve_service_key_request(
    request: Request,
    request_id: int,
    review_notes: str = Form(""),
    db: AsyncSession = Depends(get_async_db),
):
    """Approve a service key request."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.is_admin):
        return RedirectResponse(url="/dashboard", status_code=302)

    await crud.review_service_key_request(
        db=db,
        request_id=request_id,
        reviewer_id=user_id,
        approved=True,
        review_notes=review_notes.strip() or None,
    )
    await crud.log_admin_action(
        db, user_id=user_id, action="service_key.approve",
        entity_type="service_key_request", entity_id=str(request_id),
        ip_address=get_client_ip(request),
    )
    await db.commit()

    return RedirectResponse(url="/admin/requests", status_code=302)


@dashboard_router.post("/admin/service-key-requests/{request_id}/deny")
async def deny_service_key_request(
    request: Request,
    request_id: int,
    review_notes: str = Form(""),
    db: AsyncSession = Depends(get_async_db),
):
    """Deny a service key request."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.is_admin):
        return RedirectResponse(url="/dashboard", status_code=302)

    await crud.review_service_key_request(
        db=db,
        request_id=request_id,
        reviewer_id=user_id,
        approved=False,
        review_notes=review_notes.strip() or None,
    )
    await crud.log_admin_action(
        db, user_id=user_id, action="service_key.deny",
        entity_type="service_key_request", entity_id=str(request_id),
        ip_address=get_client_ip(request),
    )
    await db.commit()

    return RedirectResponse(url="/admin/requests", status_code=302)


@dashboard_router.post("/admin/service-keys/{key_id}/revoke")
async def admin_revoke_service_key(
    request: Request,
    key_id: int,
    db: AsyncSession = Depends(get_async_db),
):
    """Admin revokes a service key."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.is_admin):
        return RedirectResponse(url="/dashboard", status_code=302)

    await crud.revoke_service_key(db, key_id)
    await crud.log_admin_action(
        db, user_id=user_id, action="service_key.revoke",
        entity_type="api_key", entity_id=str(key_id),
        ip_address=get_client_ip(request),
    )
    await db.commit()

    return RedirectResponse(url="/admin/api-keys?type_filter=service", status_code=302)


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
    if not user or (not user.group or not user.group.has_admin_read):
        return RedirectResponse(url="/dashboard", status_code=302)

    grouped_models = await crud.get_models_grouped_by_name(db)

    # Get Ollama backends for pull/delete UI
    all_backends = await crud.get_all_backends(db)
    ollama_backends = [
        {
            "id": b.id,
            "name": b.name,
            "url": b.url,
            "node_id": b.node_id,
            "node_name": b.node.name if b.node else None,
            "has_sidecar": bool(b.node and b.node.sidecar_url),
            "status": b.status.value,
        }
        for b in all_backends
        if b.engine == BackendEngine.OLLAMA
    ]

    masq = await _admin_masquerade_context(request, user, db)
    return templates.TemplateResponse(
        "admin/models.html",
        {
            "request": request,
            "user": user,
            **masq,
            "grouped_models": grouped_models,
            "ollama_backends": ollama_backends,
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
    await crud.log_admin_action(
        db, user_id=user_id, action="model.toggle_multimodal",
        entity_type="model", entity_id=model_name,
        after_value={"supports_multimodal": new_value},
        ip_address=get_client_ip(request),
    )
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
    await crud.log_admin_action(
        db, user_id=user_id, action="model.reset_multimodal",
        entity_type="model", entity_id=model_name,
        ip_address=get_client_ip(request),
    )
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
    await crud.log_admin_action(
        db, user_id=user_id, action="model.toggle_thinking",
        entity_type="model", entity_id=model_name,
        after_value={"supports_thinking": new_value},
        ip_address=get_client_ip(request),
    )
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
    await crud.log_admin_action(
        db, user_id=user_id, action="model.reset_thinking",
        entity_type="model", entity_id=model_name,
        ip_address=get_client_ip(request),
    )
    await db.commit()

    return RedirectResponse(
        url="/admin/models?success=reset", status_code=302
    )


@dashboard_router.post("/admin/models/toggle-tools")
async def toggle_model_tools(
    request: Request,
    model_name: str = Form(...),
    db: AsyncSession = Depends(get_async_db),
):
    """Toggle the tools override for all instances of a model."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.is_admin):
        return RedirectResponse(url="/dashboard", status_code=302)

    from sqlalchemy import select
    from backend.app.db.models import Model

    result = await db.execute(select(Model).where(Model.name == model_name).limit(1))
    model = result.scalar_one_or_none()
    if not model:
        return RedirectResponse(
            url="/admin/models?error=Model+not+found", status_code=302
        )

    new_value = not model.supports_tools
    await crud.set_tools_override_by_name(db, model_name, new_value)
    await crud.log_admin_action(
        db, user_id=user_id, action="model.toggle_tools",
        entity_type="model", entity_id=model_name,
        after_value={"supports_tools": new_value},
        ip_address=get_client_ip(request),
    )
    await db.commit()

    return RedirectResponse(
        url="/admin/models?success=updated", status_code=302
    )


@dashboard_router.post("/admin/models/reset-tools")
async def reset_model_tools(
    request: Request,
    model_name: str = Form(...),
    db: AsyncSession = Depends(get_async_db),
):
    """Reset tools override to auto-detect for all instances of a model."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.is_admin):
        return RedirectResponse(url="/dashboard", status_code=302)

    await crud.set_tools_override_by_name(db, model_name, None)
    await crud.log_admin_action(
        db, user_id=user_id, action="model.reset_tools",
        entity_type="model", entity_id=model_name,
        ip_address=get_client_ip(request),
    )
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
    huggingface_url: Optional[str] = Form(None),
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
        "huggingface_url": _str_or_none(huggingface_url),
    }

    # Capabilities: comma-separated -> JSON array, or None to clear
    cap_str = _str_or_none(capabilities)
    if cap_str:
        cap_list = [c.strip() for c in cap_str.split(",") if c.strip()]
        overrides["capabilities_override"] = json.dumps(cap_list)
    else:
        overrides["capabilities_override"] = None

    await crud.update_model_overrides_by_name(db, model_name, overrides)
    await crud.log_admin_action(
        db, user_id=user_id, action="model.update_metadata",
        entity_type="model", entity_id=model_name,
        after_value=overrides,
        ip_address=get_client_ip(request),
    )
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
    await crud.log_admin_action(
        db, user_id=user_id, action="model.reset_overrides",
        entity_type="model", entity_id=model_name,
        ip_address=get_client_ip(request),
    )
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
    if not user or (not user.group or not user.group.has_admin_read):
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
    masq = await _admin_masquerade_context(request, user, db)
    return templates.TemplateResponse(
        "admin/nodes.html",
        {
            "request": request,
            "user": user,
            **masq,
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
        async with get_async_db_context() as audit_db:
            await crud.log_admin_action(
                audit_db, user_id=user_id, action="node.register",
                entity_type="node", entity_id=name,
                after_value={"hostname": hostname_val, "sidecar_url": sidecar_url_val},
                ip_address=get_client_ip(request),
            )
            await audit_db.commit()
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
        async with get_async_db_context() as audit_db:
            await crud.log_admin_action(
                audit_db, user_id=user_id, action="node.edit",
                entity_type="node", entity_id=str(node_id),
                after_value={"name": name, "hostname": hostname_val, "sidecar_url": sidecar_url_val},
                ip_address=get_client_ip(request),
            )
            await audit_db.commit()
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
        async with get_async_db_context() as audit_db:
            await crud.log_admin_action(
                audit_db, user_id=user_id, action="node.remove",
                entity_type="node", entity_id=str(node_id),
                ip_address=get_client_ip(request),
            )
            await audit_db.commit()
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
    async with get_async_db_context() as audit_db:
        await crud.log_admin_action(
            audit_db, user_id=user_id, action="node.refresh",
            entity_type="node", entity_id=str(node_id),
            ip_address=get_client_ip(request),
        )
        await audit_db.commit()
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
    await crud.log_admin_action(
        db, user_id=user_id, action="node.take_offline",
        entity_type="node", entity_id=str(node_id),
        ip_address=get_client_ip(request),
    )
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
    await crud.log_admin_action(
        db, user_id=user_id, action="node.bring_online",
        entity_type="node", entity_id=str(node_id),
        ip_address=get_client_ip(request),
    )
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
    if not user or (not user.group or not user.group.has_admin_read):
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
    await crud.log_admin_action(
        db, user_id=user_id, action="node.force_drain",
        entity_type="node", entity_id=str(node_id),
        detail=f"cancelled {cancelled} requests",
        ip_address=get_client_ip(request),
    )
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
    if not user or (not user.group or not user.group.has_admin_read):
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

    masq = await _admin_masquerade_context(request, user, db)
    return templates.TemplateResponse(
        "admin/backends.html",
        {
            "request": request,
            "user": user,
            **masq,
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
        async with get_async_db_context() as audit_db:
            await crud.log_admin_action(
                audit_db, user_id=user_id, action="backend.register",
                entity_type="backend", entity_id=name,
                after_value={"url": url, "engine": engine, "max_concurrent": max_concurrent},
                ip_address=get_client_ip(request),
            )
            await audit_db.commit()
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
        async with get_async_db_context() as audit_db:
            await crud.log_admin_action(
                audit_db, user_id=user_id, action="backend.edit",
                entity_type="backend", entity_id=str(backend_id),
                after_value={"name": name, "url": url, "engine": engine, "max_concurrent": max_concurrent},
                ip_address=get_client_ip(request),
            )
            await audit_db.commit()
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
    async with get_async_db_context() as audit_db:
        await crud.log_admin_action(
            audit_db, user_id=user_id, action="backend.disable",
            entity_type="backend", entity_id=str(backend_id),
            ip_address=get_client_ip(request),
        )
        await audit_db.commit()
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
    async with get_async_db_context() as audit_db:
        await crud.log_admin_action(
            audit_db, user_id=user_id, action="backend.drain",
            entity_type="backend", entity_id=str(backend_id),
            ip_address=get_client_ip(request),
        )
        await audit_db.commit()
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
    async with get_async_db_context() as audit_db:
        await crud.log_admin_action(
            audit_db, user_id=user_id, action="backend.enable",
            entity_type="backend", entity_id=str(backend_id),
            ip_address=get_client_ip(request),
        )
        await audit_db.commit()
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
        async with get_async_db_context() as audit_db:
            await crud.log_admin_action(
                audit_db, user_id=user_id, action="backend.remove",
                entity_type="backend", entity_id=str(backend_id),
                ip_address=get_client_ip(request),
            )
            await audit_db.commit()
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
    async with get_async_db_context() as audit_db:
        await crud.log_admin_action(
            audit_db, user_id=user_id, action="backend.refresh",
            entity_type="backend", entity_id=str(backend_id),
            ip_address=get_client_ip(request),
        )
        await audit_db.commit()
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
    if not user or (not user.group or not user.group.has_admin_read):
        return RedirectResponse(url="/dashboard", status_code=302)

    masq = await _admin_masquerade_context(request, user, db)
    current_tz = await crud.get_config_json(db, "display.timezone", _tz_cache["name"])
    return templates.TemplateResponse(
        "admin/metrics.html",
        {"request": request, "user": user, "current_timezone": current_tz, **masq},
    )


@dashboard_router.get("/admin/energy", response_class=HTMLResponse)
async def admin_energy(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """Admin energy & power monitoring dashboard."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.has_admin_read):
        return RedirectResponse(url="/dashboard", status_code=302)

    current_tz = await crud.get_config_json(db, "display.timezone", _tz_cache["name"])
    masq = await _admin_masquerade_context(request, user, db)
    return templates.TemplateResponse(
        "admin/energy.html",
        {"request": request, "user": user, **masq, "current_timezone": current_tz},
    )


@dashboard_router.get("/admin/audit", response_class=HTMLResponse)
async def admin_audit(
    request: Request,
    search: Optional[str] = None,
    user_id_filter: Optional[str] = None,
    model_filter: Optional[str] = None,
    status_filter: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    page: Optional[str] = None,
    cursor: Optional[str] = None,
    db: AsyncSession = Depends(get_async_db),
):
    """Admin audit log viewer with hybrid pagination.

    Pages 1-20: OFFSET-based (fast, gives page numbers).
    Beyond page 20 or cursor param: keyset-based (constant time).
    page=last: jump to the final page.
    """
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.has_admin_read):
        return RedirectResponse(url="/dashboard", status_code=302)

    from backend.app.db.models import RequestStatus

    per_page = 50
    max_numbered_pages = 20

    # Parse filters
    parsed_user_id: Optional[int] = None
    if user_id_filter and user_id_filter.strip().isdigit():
        parsed_user_id = int(user_id_filter.strip())
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

    # Determine pagination mode
    cursor_before = None
    current_page = None
    is_last_page = False
    using_cursor = False

    if page == "last":
        is_last_page = True
    elif cursor and cursor.isdigit():
        cursor_before = int(cursor)
        using_cursor = True
    elif page and page.isdigit():
        current_page = max(1, int(page))
    else:
        current_page = 1

    filter_kwargs = dict(
        user_id=parsed_user_id, model=model_filter, status=parsed_status,
        start_date=parsed_start, end_date=parsed_end, search_text=search,
    )

    audit_requests, total = await crud.search_requests(
        db,
        **filter_kwargs,
        cursor_before=cursor_before,
        page=current_page,
        last_page=is_last_page,
        limit=per_page,
    )

    total_pages = max(1, (total + per_page - 1) // per_page)

    # Build next-page cursor from the last row's ID
    next_cursor = None
    if len(audit_requests) == per_page:
        next_cursor = audit_requests[-1].id

    # Determine which page numbers to show (up to max_numbered_pages)
    show_numbered = min(total_pages, max_numbered_pages)

    # Figure out the effective page for highlighting
    if is_last_page:
        effective_page = total_pages
    elif using_cursor:
        effective_page = None  # beyond numbered range
    else:
        effective_page = current_page

    masq = await _admin_masquerade_context(request, user, db)
    return templates.TemplateResponse(
        "admin/audit.html",
        {
            "request": request,
            "user": user,
            **masq,
            "audit_requests": audit_requests,
            "total": total,
            "page": effective_page,
            "total_pages": total_pages,
            "show_numbered": show_numbered,
            "next_cursor": next_cursor,
            "using_cursor": using_cursor,
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
    user_id_filter: Optional[str] = None,
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
    if not user or (not user.group or not user.group.has_admin_read):
        return RedirectResponse(url="/dashboard", status_code=302)

    from backend.app.db.models import RequestStatus

    # Parse filters
    parsed_user_id: Optional[int] = None
    if user_id_filter and user_id_filter.strip().isdigit():
        parsed_user_id = int(user_id_filter.strip())
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

    fieldnames = [
        "request_uuid", "created_at", "user_id", "model", "endpoint",
        "status", "prompt_tokens", "completion_tokens", "total_tokens",
        "total_time_ms", "error_message",
    ]
    if include_content:
        fieldnames.extend(["messages", "prompt", "parameters", "response_content", "finish_reason"])

    def _row(req) -> dict:
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
        return row

    # Stream all matching rows using batched keyset pagination (no row limit)
    if format == "json":
        async def json_stream():
            yield "[\n"
            first = True
            async for req in crud.iter_requests_batched(
                db, user_id=parsed_user_id, model=model_filter,
                status=parsed_status, start_date=parsed_start,
                end_date=parsed_end, search_text=search,
            ):
                if not first:
                    yield ",\n"
                first = False
                yield json.dumps(_row(req), indent=2)
            yield "\n]\n"

        return StreamingResponse(
            json_stream(),
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=audit_log.json"},
        )

    # CSV format — stream rows as they're read
    async def csv_stream():
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        yield buf.getvalue()

        async for req in crud.iter_requests_batched(
            db, user_id=parsed_user_id, model=model_filter,
            status=parsed_status, start_date=parsed_start,
            end_date=parsed_end, search_text=search,
        ):
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=fieldnames)
            writer.writerow({k: _row(req).get(k, "") for k in fieldnames})
            yield buf.getvalue()

    return StreamingResponse(
        csv_stream(),
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
    if not user or (not user.group or not user.group.has_admin_read):
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


@dashboard_router.get("/admin/admin-audit", response_class=HTMLResponse)
async def admin_admin_audit(
    request: Request,
    action: Optional[str] = None,
    entity_type: Optional[str] = None,
    user_id: Optional[str] = None,
    page: int = 1,
    db: AsyncSession = Depends(get_async_db),
):
    """Admin audit log viewer — tracks all admin actions."""
    session_user_id = get_session_user_id(request)
    if not session_user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, session_user_id)
    if not user or (not user.group or not user.group.has_admin_read):
        return RedirectResponse(url="/dashboard", status_code=302)

    from sqlalchemy import select, distinct
    from backend.app.db.models import AdminAuditLog, User

    parsed_uid = int(user_id) if user_id and user_id.strip().isdigit() else None
    page_size = 50
    skip = (page - 1) * page_size

    entries, total = await crud.get_admin_audit_log(
        db,
        user_id=parsed_uid,
        action=action,
        entity_type=entity_type,
        skip=skip,
        limit=page_size,
    )

    # Get distinct actions and entity types for filter dropdowns
    actions_result = await db.execute(
        select(distinct(AdminAuditLog.action)).order_by(AdminAuditLog.action)
    )
    actions = [r[0] for r in actions_result.all()]

    entity_types_result = await db.execute(
        select(distinct(AdminAuditLog.entity_type)).order_by(AdminAuditLog.entity_type)
    )
    entity_types = [r[0] for r in entity_types_result.all()]

    # Get admin users for user filter
    admin_users_result = await db.execute(
        select(User).join(User.group).where(User.group.has(is_admin=True)).order_by(User.username)
    )
    admin_users = admin_users_result.scalars().all()

    masq = await _admin_masquerade_context(request, user, db)
    return templates.TemplateResponse(
        "admin/admin_audit.html",
        {
            "request": request,
            "user": user,
            **masq,
            "entries": entries,
            "total": total,
            "page": page,
            "page_size": page_size,
            "actions": actions,
            "entity_types": entity_types,
            "admin_users": admin_users,
            "action_filter": action,
            "entity_filter": entity_type,
            "user_filter": user_id,
        },
    )


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
    if not user or (not user.group or not user.group.has_admin_read):
        return RedirectResponse(url="/dashboard", status_code=302)

    groups_with_counts = await crud.get_all_groups_with_counts(db)

    masq = await _admin_masquerade_context(request, user, db)
    return templates.TemplateResponse(
        "admin/groups.html",
        {
            "request": request,
            "user": user,
            **masq,
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
    scheduler_weight: int = Form(1),
    is_admin: Optional[str] = Form(None),
    is_auditor: Optional[str] = Form(None),
    api_key_expiry_days: int = Form(45),
    max_api_keys: int = Form(16),
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
            scheduler_weight=scheduler_weight,
            is_admin=(is_admin == "on"),
            is_auditor=(is_auditor == "on"),
            api_key_expiry_days=api_key_expiry_days,
            max_api_keys=max_api_keys,
        )
        await crud.log_admin_action(
            db, user_id=user_id, action="group.create",
            entity_type="group", entity_id=name,
            after_value={"display_name": display_name, "token_budget": token_budget},
            ip_address=get_client_ip(request),
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
    scheduler_weight: int = Form(1),
    is_admin: Optional[str] = Form(None),
    is_auditor: Optional[str] = Form(None),
    api_key_expiry_days: int = Form(45),
    max_api_keys: int = Form(16),
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
            scheduler_weight=scheduler_weight,
            is_admin=(is_admin == "on"),
            is_auditor=(is_auditor == "on"),
            api_key_expiry_days=api_key_expiry_days,
            max_api_keys=max_api_keys,
        )
        await crud.log_admin_action(
            db, user_id=user_id, action="group.edit",
            entity_type="group", entity_id=str(group_id),
            after_value={"display_name": display_name, "token_budget": token_budget},
            ip_address=get_client_ip(request),
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
        await crud.log_admin_action(
            db, user_id=user_id, action="group.delete",
            entity_type="group", entity_id=str(group_id),
            ip_address=get_client_ip(request),
        )
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
    if not admin_user or (not admin_user.group or not admin_user.group.has_admin_read):
        return RedirectResponse(url="/dashboard", status_code=302)

    stats = await crud.get_user_with_stats(db, user_id)
    if not stats:
        return RedirectResponse(url="/admin/users", status_code=302)

    monthly_usage = await crud.get_user_monthly_usage(db, user_id)
    groups = await crud.get_all_groups(db)
    recent_ips = await crud.get_user_recent_ips(db, user_id, days=90)

    masq = await _admin_masquerade_context(request, admin_user, db)
    return templates.TemplateResponse(
        "admin/user_detail.html",
        {
            "request": request,
            "user": admin_user,
            **masq,
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
            quota.weight_override = int(weight_override) if weight_override and weight_override.strip() else None
            await db.flush()

        await crud.log_admin_action(
            db, user_id=session_user_id, action="user.edit",
            entity_type="user", entity_id=str(user_id),
            after_value={"group_id": group_id, "full_name": full_name},
            ip_address=get_client_ip(request),
        )
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
    user_id = get_session_user_id(request)
    logger.info("masquerade_stop", admin_user_id=user_id)
    if user_id:
        async with get_async_db_context() as audit_db:
            await crud.log_admin_action(
                audit_db, user_id=user_id, action="masquerade.stop",
                entity_type="user",
                ip_address=get_client_ip(request),
            )
            await audit_db.commit()
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

    async with get_async_db_context() as audit_db:
        await crud.log_admin_action(
            audit_db, user_id=user_id, action="masquerade.start",
            entity_type="user", entity_id=str(target_user_id),
            ip_address=get_client_ip(request),
        )
        await audit_db.commit()

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
    type_filter: Optional[str] = None,
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
    if not user or (not user.group or not user.group.has_admin_read):
        return RedirectResponse(url="/dashboard", status_code=302)

    per_page = 50
    skip = (page - 1) * per_page
    sort_dir = dir if dir in ("asc", "desc") else "desc"
    keys, total = await crud.get_all_api_keys(
        db, skip=skip, limit=per_page, search=search, status_filter=key_status,
        sort_by=sort, sort_dir=sort_dir, type_filter=type_filter,
    )
    total_pages = max(1, (total + per_page - 1) // per_page)

    # Fetch last IP for each key on this page
    key_ids = [k.id for k in keys]
    key_last_ips = await crud.get_api_key_last_ips_batch(db, key_ids)

    # Fetch all active service keys for the dedicated section (with their request history)
    service_keys, _ = await crud.get_all_api_keys(
        db, skip=0, limit=200, type_filter="service",
        sort_by="created", sort_dir="desc",
    )
    service_key_ids = [k.id for k in service_keys]
    service_last_ips = await crud.get_api_key_last_ips_batch(db, service_key_ids)

    masq = await _admin_masquerade_context(request, user, db)
    return templates.TemplateResponse(
        "admin/api_keys.html",
        {
            "request": request,
            "user": user,
            **masq,
            "api_keys": keys,
            "service_keys": service_keys,
            "service_last_ips": service_last_ips,
            "total": total,
            "page": page,
            "total_pages": total_pages,
            "search": search or "",
            "key_status": key_status or "",
            "type_filter": type_filter or "",
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
    user_id_filter: Optional[str] = None,
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
    if not user or (not user.group or not user.group.has_admin_read):
        return RedirectResponse(url="/dashboard", status_code=302)

    parsed_uid = int(user_id_filter) if user_id_filter and user_id_filter.strip().isdigit() else None
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
        user_id=parsed_uid,
        model=model_filter,
        search_text=search,
        start_date=parsed_start,
        end_date=parsed_end,
        skip=skip,
        limit=per_page,
    )
    total_pages = max(1, (total + per_page - 1) // per_page)

    masq = await _admin_masquerade_context(request, user, db)
    return templates.TemplateResponse(
        "admin/conversations.html",
        {
            "request": request,
            "user": user,
            **masq,
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
    if not user or (not user.group or not user.group.has_admin_read):
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
    user_id_filter: Optional[str] = None,
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
    if not user or (not user.group or not user.group.has_admin_read):
        return RedirectResponse(url="/dashboard", status_code=302)

    parsed_uid = int(user_id_filter) if user_id_filter and user_id_filter.strip().isdigit() else None

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
        user_id=parsed_uid,
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
    user_id_filter: Optional[str] = None,
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
    if not user or (not user.group or not user.group.has_admin_read):
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

    # Voice settings (chat-specific)
    tts_enabled = await crud.get_config_json(db, "voice.tts_enabled", False)
    tts_provider = await crud.get_config_json(db, "voice.tts_provider", "kokoro")
    tts_voice = await crud.get_config_json(db, "voice.tts_voice", "af_heart")
    tts_speed = await crud.get_config_json(db, "voice.tts_speed", 1.0)
    stt_enabled = await crud.get_config_json(db, "voice.stt_enabled", False)

    masq = await _admin_masquerade_context(request, user, db)
    return templates.TemplateResponse(
        "admin/chat_config.html",
        {
            "request": request,
            "user": user,
            **masq,
            "available_models": sorted(available_models),
            "core_models": core_models,
            "default_model": default_model,
            "system_prompt": system_prompt,
            "chat_max_tokens": chat_max_tokens,
            "chat_temperature": chat_temperature,
            "chat_think": chat_think,
            "tts_enabled": tts_enabled,
            "tts_provider": tts_provider,
            "tts_voice": tts_voice,
            "tts_speed": tts_speed,
            "stt_enabled": stt_enabled,
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

    _ip = get_client_ip(request)

    if action == "set_default":
        default_model = form.get("default_model", "")
        await crud.set_config(db, "chat.default_model", default_model if default_model else None)
        await crud.log_admin_action(
            db, user_id=user_id, action="chat_config.set_default",
            entity_type="config", detail=f"default_model={default_model}", ip_address=_ip,
        )
        await db.commit()
        return RedirectResponse(url="/admin/chat-config?success=default_updated", status_code=302)

    elif action == "set_core_models":
        selected = form.getlist("core_models")
        await crud.set_config(db, "chat.core_models", selected)
        await crud.log_admin_action(
            db, user_id=user_id, action="chat_config.set_core_models",
            entity_type="config", after_value={"core_models": selected}, ip_address=_ip,
        )
        await db.commit()
        return RedirectResponse(url="/admin/chat-config?success=core_updated", status_code=302)

    elif action == "set_system_prompt":
        prompt_text = form.get("system_prompt", "").strip()
        if prompt_text:
            await crud.set_config(db, "chat.system_prompt", prompt_text)
        else:
            # Blank = remove override, revert to built-in default
            await crud.set_config(db, "chat.system_prompt", None)
        await crud.log_admin_action(
            db, user_id=user_id, action="chat_config.set_system_prompt",
            entity_type="config", ip_address=_ip,
        )
        await db.commit()
        return RedirectResponse(url="/admin/chat-config?success=system_prompt_updated", status_code=302)

    elif action == "reset_system_prompt":
        await crud.set_config(db, "chat.system_prompt", None)
        await crud.log_admin_action(
            db, user_id=user_id, action="chat_config.reset_system_prompt",
            entity_type="config", ip_address=_ip,
        )
        await db.commit()
        return RedirectResponse(url="/admin/chat-config?success=system_prompt_reset", status_code=302)

    elif action == "set_max_tokens":
        try:
            val = int(form.get("max_tokens", "16384"))
            val = max(256, min(131072, val))
        except (ValueError, TypeError):
            val = 16384
        await crud.set_config(db, "chat.max_tokens", val)
        await crud.log_admin_action(
            db, user_id=user_id, action="chat_config.set_max_tokens",
            entity_type="config", detail=f"max_tokens={val}", ip_address=_ip,
        )
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
        await crud.log_admin_action(
            db, user_id=user_id, action="chat_config.set_temperature",
            entity_type="config", detail=f"temperature={temp_str}", ip_address=_ip,
        )
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
        await crud.log_admin_action(
            db, user_id=user_id, action="chat_config.set_think",
            entity_type="config", detail=f"think={think_val}", ip_address=_ip,
        )
        await db.commit()
        return RedirectResponse(url="/admin/chat-config?success=think_updated", status_code=302)

    elif action == "save_chat_tts":
        tts_enabled = form.get("tts_enabled") == "on"
        tts_provider = form.get("tts_provider", "kokoro")
        tts_voice = form.get("tts_voice", "").strip() or "af_heart"
        try:
            tts_speed = float(form.get("tts_speed", "1.0"))
            tts_speed = max(0.5, min(2.0, tts_speed))
        except (ValueError, TypeError):
            tts_speed = 1.0
        await crud.set_config(db, "voice.tts_enabled", tts_enabled)
        await crud.set_config(db, "voice.tts_provider", tts_provider)
        await crud.set_config(db, "voice.tts_voice", tts_voice)
        await crud.set_config(db, "voice.tts_speed", tts_speed)
        await crud.log_admin_action(
            db, user_id=user_id, action="chat_config.save_tts",
            entity_type="config",
            after_value={"tts_enabled": tts_enabled, "tts_provider": tts_provider, "tts_voice": tts_voice},
            ip_address=_ip,
        )
        await db.commit()
        return RedirectResponse(url="/admin/chat-config?success=voice_tts_updated", status_code=302)

    elif action == "save_chat_stt":
        stt_enabled = form.get("stt_enabled") == "on"
        await crud.set_config(db, "voice.stt_enabled", stt_enabled)
        await crud.log_admin_action(
            db, user_id=user_id, action="chat_config.save_stt",
            entity_type="config", detail=f"stt_enabled={stt_enabled}", ip_address=_ip,
        )
        await db.commit()
        return RedirectResponse(url="/admin/chat-config?success=voice_stt_updated", status_code=302)

    return RedirectResponse(url="/admin/chat-config?error=Unknown+action", status_code=302)


# ---------------------------------------------------------------------------
# Admin Voice Config (TTS / STT)
# ---------------------------------------------------------------------------


@dashboard_router.get("/admin/voice-config")
async def admin_voice_config(
    request: Request,
    success: Optional[str] = None,
    error: Optional[str] = None,
    db: AsyncSession = Depends(get_async_db),
):
    """Admin voice configuration page."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.has_admin_read):
        return RedirectResponse(url="/dashboard", status_code=302)

    tts_url = await crud.get_config_json(db, "voice.tts_url", None)
    tts_api_key = await crud.get_config_json(db, "voice.tts_api_key", None)
    tts_voices = await crud.get_config_json(db, "voice_api.tts_voices", "af_heart\naf_bella\nam_adam\nam_michael")
    default_voice = await crud.get_config_json(db, "voice_api.default_voice", "af_heart")
    stt_url = await crud.get_config_json(db, "voice.stt_url", None)
    stt_api_key = await crud.get_config_json(db, "voice.stt_api_key", None)
    stt_model = await crud.get_config_json(db, "voice.stt_model", "whisper-large-v3-turbo")
    tts_quota_tokens = await crud.get_config_json(db, "voice_api.tts_quota_tokens", 100)
    stt_quota_tokens = await crud.get_config_json(db, "voice_api.stt_quota_tokens", 200)

    masq = await _admin_masquerade_context(request, user, db)
    return templates.TemplateResponse(
        "admin/voice_config.html",
        {
            "request": request,
            "user": user,
            **masq,
            "tts_url": tts_url,
            "tts_api_key": tts_api_key,
            "tts_voices": tts_voices,
            "default_voice": default_voice,
            "stt_url": stt_url,
            "stt_api_key": stt_api_key,
            "stt_model": stt_model,
            "tts_quota_tokens": tts_quota_tokens,
            "stt_quota_tokens": stt_quota_tokens,
            "success": success,
            "error": error,
        },
    )


@dashboard_router.post("/admin/voice-config")
async def admin_voice_config_post(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """Handle voice config form submissions."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.is_admin):
        return RedirectResponse(url="/dashboard", status_code=302)

    form = await request.form()
    action = form.get("action")

    _ip = get_client_ip(request)

    if action == "save_voice_tts_backend":
        tts_url = form.get("tts_url", "").strip() or None
        tts_api_key = form.get("tts_api_key", "").strip() or None
        tts_voices = form.get("tts_voices", "").strip()
        default_voice = form.get("default_voice", "").strip() or "af_heart"

        await crud.set_config(db, "voice.tts_url", tts_url)
        await crud.set_config(db, "voice.tts_api_key", tts_api_key)
        await crud.set_config(db, "voice_api.tts_voices", tts_voices)
        await crud.set_config(db, "voice_api.default_voice", default_voice)
        await crud.log_admin_action(
            db, user_id=user_id, action="voice_config.save_tts_backend",
            entity_type="config", after_value={"tts_url": tts_url, "default_voice": default_voice},
            ip_address=_ip,
        )
        await db.commit()
        return RedirectResponse(url="/admin/voice-config?success=tts_backend_updated", status_code=302)

    elif action == "save_voice_stt_backend":
        stt_url = form.get("stt_url", "").strip() or None
        stt_api_key = form.get("stt_api_key", "").strip() or None
        stt_model = form.get("stt_model", "").strip() or "whisper-large-v3-turbo"

        await crud.set_config(db, "voice.stt_url", stt_url)
        await crud.set_config(db, "voice.stt_api_key", stt_api_key)
        await crud.set_config(db, "voice.stt_model", stt_model)
        await crud.log_admin_action(
            db, user_id=user_id, action="voice_config.save_stt_backend",
            entity_type="config", after_value={"stt_url": stt_url, "stt_model": stt_model},
            ip_address=_ip,
        )
        await db.commit()
        return RedirectResponse(url="/admin/voice-config?success=stt_backend_updated", status_code=302)

    elif action == "save_voice_quota":
        try:
            tts_qt = int(form.get("tts_quota_tokens", "100"))
            tts_qt = max(0, min(100000, tts_qt))
        except (ValueError, TypeError):
            tts_qt = 100
        try:
            stt_qt = int(form.get("stt_quota_tokens", "200"))
            stt_qt = max(0, min(100000, stt_qt))
        except (ValueError, TypeError):
            stt_qt = 200
        await crud.set_config(db, "voice_api.tts_quota_tokens", tts_qt)
        await crud.set_config(db, "voice_api.stt_quota_tokens", stt_qt)
        await crud.log_admin_action(
            db, user_id=user_id, action="voice_config.save_quota",
            entity_type="config",
            after_value={"tts_quota_tokens": tts_qt, "stt_quota_tokens": stt_qt},
            ip_address=_ip,
        )
        await db.commit()
        return RedirectResponse(url="/admin/voice-config?success=quota_updated", status_code=302)

    return RedirectResponse(url="/admin/voice-config?error=Unknown+action", status_code=302)


# ---------------------------------------------------------------------------
# Admin OCR Configuration
# ---------------------------------------------------------------------------


@dashboard_router.get("/admin/ocr-config")
async def admin_ocr_config(
    request: Request,
    success: Optional[str] = None,
    error: Optional[str] = None,
    db: AsyncSession = Depends(get_async_db),
):
    """Admin OCR configuration page."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.has_admin_read):
        return RedirectResponse(url="/dashboard", status_code=302)

    # Get multimodal model names for the dropdown
    all_models = await crud.get_all_models_with_backends(db)
    multimodal_models = sorted({
        m.name for m in all_models
        if m.supports_multimodal
    })

    masq = await _admin_masquerade_context(request, user, db)
    return templates.TemplateResponse(
        "admin/ocr_config.html",
        {
            "request": request,
            "user": user,
            **masq,
            "multimodal_models": multimodal_models,
            "enabled": await crud.get_config_json(db, "ocr.enabled", True),
            "default_model": await crud.get_config_json(db, "ocr.default_model", "qwen/qwen3.5-122b"),
            "chunk_size": await crud.get_config_json(db, "ocr.chunk_size", 6),
            "overlap": await crud.get_config_json(db, "ocr.overlap", 2),
            "dpi": await crud.get_config_json(db, "ocr.dpi", 200),
            "max_pages": await crud.get_config_json(db, "ocr.max_pages", 200),
            "max_file_size_mb": await crud.get_config_json(db, "ocr.max_file_size_mb", 100),
            "max_concurrent_chunks": await crud.get_config_json(db, "ocr.max_concurrent_chunks", 4),
            "min_chars_per_page": await crud.get_config_json(db, "ocr.min_chars_per_page", 400),
            "max_retries": await crud.get_config_json(db, "ocr.max_retries", 2),
            "max_tokens": await crud.get_config_json(db, "ocr.max_tokens", 16384),
            "temperature": await crud.get_config_json(db, "ocr.temperature", 0.1),
            "success": success,
            "error": error,
        },
    )


@dashboard_router.post("/admin/ocr-config")
async def admin_ocr_config_post(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """Handle OCR config form submission."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.is_admin):
        return RedirectResponse(url="/dashboard", status_code=302)

    form = await request.form()

    await crud.set_config(db, "ocr.enabled", "enabled" in form)
    await crud.set_config(db, "ocr.default_model", form.get("default_model", "").strip() or "qwen/qwen3.5-122b")

    # Integer configs with bounds
    int_configs = {
        "ocr.chunk_size": ("chunk_size", 6, 1, 20),
        "ocr.overlap": ("overlap", 2, 0, 10),
        "ocr.dpi": ("dpi", 200, 72, 600),
        "ocr.max_pages": ("max_pages", 200, 1, 1000),
        "ocr.max_file_size_mb": ("max_file_size_mb", 100, 1, 500),
        "ocr.max_concurrent_chunks": ("max_concurrent_chunks", 4, 1, 32),
        "ocr.min_chars_per_page": ("min_chars_per_page", 400, 0, 2000),
        "ocr.max_retries": ("max_retries", 2, 0, 5),
        "ocr.max_tokens": ("max_tokens", 16384, 1024, 65536),
    }
    for key, (field, default, lo, hi) in int_configs.items():
        try:
            val = int(form.get(field, str(default)))
            val = max(lo, min(hi, val))
        except (ValueError, TypeError):
            val = default
        await crud.set_config(db, key, val)

    # Float config
    try:
        temp = float(form.get("temperature", "0.1"))
        temp = max(0.0, min(1.0, temp))
    except (ValueError, TypeError):
        temp = 0.1
    await crud.set_config(db, "ocr.temperature", temp)

    _ip = get_client_ip(request)
    await crud.log_admin_action(
        db, user_id=user_id, action="ocr_config.save",
        entity_type="config",
        after_value={"model": form.get("default_model"), "enabled": "enabled" in form},
        ip_address=_ip,
    )
    await db.commit()
    return RedirectResponse(url="/admin/ocr-config?success=config_updated", status_code=302)


# ---------------------------------------------------------------------------
# Admin Image Generation Config
# ---------------------------------------------------------------------------

@dashboard_router.get("/admin/images-config")
async def admin_images_config(
    request: Request,
    success: Optional[str] = None,
    error: Optional[str] = None,
    search: Optional[str] = None,
    page: int = 1,
    db: AsyncSession = Depends(get_async_db),
):
    """Admin image generation configuration page."""
    from sqlalchemy import select, func
    from backend.app.db.models import Modality, User as UserModel

    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.has_admin_read):
        return RedirectResponse(url="/dashboard", status_code=302)

    # Model names for dropdowns
    all_models = await crud.get_all_models_with_backends(db)
    diffusion_models = sorted({
        m.name for m in all_models
        if m.modality == Modality.IMAGE_GENERATION
    })
    chat_models = sorted({
        m.name for m in all_models
        if m.modality in (Modality.CHAT, Modality.MULTIMODAL)
    })

    # User list with pagination
    per_page = 50
    users_list, total_count = await crud.get_users(
        db, skip=(page - 1) * per_page, limit=per_page,
        search=search, is_active=True, sort_by="username", sort_dir="asc",
    )
    total_pages = max(1, (total_count + per_page - 1) // per_page)

    # Count enabled users
    enabled_result = await db.execute(
        select(func.count()).select_from(UserModel).where(
            UserModel.image_generation_enabled == True,  # noqa: E712
            UserModel.deleted_at.is_(None),
            UserModel.is_active == True,  # noqa: E712
        )
    )
    enabled_count = enabled_result.scalar() or 0

    masq = await _admin_masquerade_context(request, user, db)
    return templates.TemplateResponse(
        "admin/images_config.html",
        {
            "request": request,
            "user": user,
            **masq,
            "diffusion_models": diffusion_models,
            "chat_models": chat_models,
            "enabled": await crud.get_config_json(db, "img.enabled", True),
            "default_model": await crud.get_config_json(db, "img.default_model", "black-forest-labs/FLUX.2-dev"),
            "default_size": await crud.get_config_json(db, "img.default_size", "1024x1024"),
            "max_n": await crud.get_config_json(db, "img.max_n", 4),
            "default_steps": await crud.get_config_json(db, "img.default_steps", 20),
            "max_steps": await crud.get_config_json(db, "img.max_steps", 50),
            "default_guidance_scale": await crud.get_config_json(db, "img.default_guidance_scale", 3.5),
            "allowed_sizes": await crud.get_config_json(db, "img.allowed_sizes", "512x512,768x768,1024x1024,1024x768,768x1024"),
            "policy": await crud.get_config_json(db, "img.policy", ""),
            "judge_model": await crud.get_config_json(db, "img.judge_model", ""),
            "judge_model_secondary": await crud.get_config_json(db, "img.judge_model_secondary", ""),
            "users": users_list,
            "total_users": total_count,
            "total_pages": total_pages,
            "page": page,
            "search": search,
            "enabled_count": enabled_count,
            "success": success,
            "error": error,
        },
    )


@dashboard_router.post("/admin/images-config")
async def admin_images_config_post(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """Handle image config form submissions."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.is_admin):
        return RedirectResponse(url="/dashboard", status_code=302)

    form = await request.form()
    action = form.get("action", "")
    _ip = get_client_ip(request)

    if action == "save_config":
        await crud.set_config(db, "img.enabled", "enabled" in form)
        await crud.set_config(db, "img.default_model", form.get("default_model", "").strip() or "black-forest-labs/FLUX.2-dev")
        await crud.set_config(db, "img.default_size", form.get("default_size", "").strip() or "1024x1024")
        await crud.set_config(db, "img.allowed_sizes", form.get("allowed_sizes", "").strip())
        await crud.set_config(db, "img.policy", form.get("policy", "").strip())
        await crud.set_config(db, "img.judge_model", form.get("judge_model", "").strip())
        await crud.set_config(db, "img.judge_model_secondary", form.get("judge_model_secondary", "").strip())

        int_configs = {
            "img.max_n": ("max_n", 4, 1, 10),
            "img.default_steps": ("default_steps", 20, 1, 100),
            "img.max_steps": ("max_steps", 50, 1, 200),
        }
        for key, (field, default, lo, hi) in int_configs.items():
            try:
                val = int(form.get(field, str(default)))
                val = max(lo, min(hi, val))
            except (ValueError, TypeError):
                val = default
            await crud.set_config(db, key, val)

        # Float config
        try:
            guidance = float(form.get("default_guidance_scale", "3.5"))
            guidance = max(0.0, min(20.0, guidance))
        except (ValueError, TypeError):
            guidance = 3.5
        await crud.set_config(db, "img.default_guidance_scale", guidance)

        await crud.log_admin_action(
            db, user_id=user_id, action="images_config.save",
            entity_type="config",
            after_value={"enabled": "enabled" in form, "model": form.get("default_model")},
            ip_address=_ip,
        )
        await db.commit()
        return RedirectResponse(url="/admin/images-config?success=config_updated", status_code=302)

    elif action == "toggle_user":
        target_id = int(form.get("user_id", 0))
        target_user = await crud.get_user_by_id(db, target_id)
        if not target_user:
            return RedirectResponse(url="/admin/images-config?error=User+not+found", status_code=302)

        new_val = not target_user.image_generation_enabled
        target_user.image_generation_enabled = new_val

        await crud.log_admin_action(
            db, user_id=user_id, action="images_config.toggle_user",
            entity_type="user", entity_id=str(target_id),
            before_value={"image_generation_enabled": not new_val},
            after_value={"image_generation_enabled": new_val},
            ip_address=_ip,
        )
        await db.commit()
        status_msg = "user_enabled" if new_val else "user_disabled"
        return RedirectResponse(url=f"/admin/images-config?success={status_msg}", status_code=302)

    return RedirectResponse(url="/admin/images-config?error=Unknown+action", status_code=302)


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
    if not user or (not user.group or not user.group.has_admin_read):
        return RedirectResponse(url="/dashboard", status_code=302)

    site_url = await crud.get_config_json(db, "app.base_url", get_settings().app_base_url)
    current_tz = await crud.get_config_json(db, "app.timezone", "America/Los_Angeles")
    enforce_num_ctx = await crud.get_config_json(db, "ollama.enforce_num_ctx", True)

    # Model auto-enrichment config
    auto_enrich = await crud.get_config_json(db, "catalog.auto_enrich", False)
    enrich_model = await crud.get_config_json(db, "catalog.enrich_model", "")
    enrich_api_key = await crud.get_config_json(db, "catalog.enrich_api_key", "")
    brave_api_key = await crud.get_config_json(db, "catalog.brave_api_key", "")

    # Available model names for the enrichment model dropdown
    from sqlalchemy import select, distinct
    from backend.app.db.models import Model
    result = await db.execute(select(distinct(Model.name)).order_by(Model.name))
    available_model_names = [row[0] for row in result.all()]

    # Show current time in configured timezone as preview
    now_in_tz = localtime_filter(datetime.now(timezone.utc), "%Y-%m-%d %H:%M:%S %Z")

    # Use agreement
    agreement = await crud.get_agreement(db)

    # Email / SMTP config
    from backend.app.services.email_service import get_smtp_config
    smtp_config = await get_smtp_config(db)

    masq = await _admin_masquerade_context(request, user, db)
    return templates.TemplateResponse(
        "admin/settings.html",
        {
            "request": request,
            "user": user,
            **masq,
            "site_url": site_url,
            "current_timezone": current_tz,
            "timezone_choices": _TIMEZONE_CHOICES,
            "now_in_tz": now_in_tz,
            "enforce_num_ctx": enforce_num_ctx,
            "auto_enrich": auto_enrich,
            "enrich_model": enrich_model,
            "enrich_api_key": enrich_api_key,
            "brave_api_key": brave_api_key,
            "available_model_names": available_model_names,
            "agreement_text": agreement["text"],
            "agreement_version": agreement["version"],
            "smtp_host": smtp_config["host"],
            "smtp_port": smtp_config["port"],
            "smtp_username": smtp_config["username"],
            "smtp_password": smtp_config["password"],
            "smtp_use_tls": smtp_config["use_tls"],
            "smtp_default_sender": smtp_config["default_sender"],
            "smtp_test_address": smtp_config["test_address"],
            "smtp_blog_sender": smtp_config["blog_sender"],
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
    _ip = get_client_ip(request)

    if action == "set_site_url":
        url = form.get("site_url", "").strip().rstrip("/")
        if not url:
            return RedirectResponse(url="/admin/settings?error=Site+URL+cannot+be+empty", status_code=302)
        await crud.set_config(
            db, "app.base_url", url, description="Public-facing base URL for this MindRouter installation"
        )
        await crud.log_admin_action(
            db, user_id=user_id, action="settings.set_site_url",
            entity_type="config", detail=f"site_url={url}", ip_address=_ip,
        )
        await db.commit()
        return RedirectResponse(url="/admin/settings?success=site_url_updated", status_code=302)

    elif action == "set_timezone":
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
        await crud.log_admin_action(
            db, user_id=user_id, action="settings.set_timezone",
            entity_type="config", detail=f"timezone={tz_name}", ip_address=_ip,
        )
        await db.commit()
        _refresh_tz_cache_sync(tz_name)
        return RedirectResponse(url="/admin/settings?success=timezone_updated", status_code=302)

    elif action == "set_enforce_num_ctx":
        val = form.get("enforce_num_ctx") == "on"
        await crud.set_config(
            db, "ollama.enforce_num_ctx", val,
            description="Override user-supplied num_ctx with model config context_length"
        )
        await crud.log_admin_action(
            db, user_id=user_id, action="settings.set_enforce_num_ctx",
            entity_type="config", detail=f"enforce_num_ctx={val}", ip_address=_ip,
        )
        await db.commit()
        return RedirectResponse(url="/admin/settings?success=enforce_num_ctx_updated", status_code=302)

    elif action == "set_auto_enrich":
        val = form.get("auto_enrich") == "on"
        await crud.set_config(
            db, "catalog.auto_enrich", val,
            description="Enable automatic model description enrichment"
        )
        await crud.log_admin_action(
            db, user_id=user_id, action="settings.set_auto_enrich",
            entity_type="config", detail=f"auto_enrich={val}", ip_address=_ip,
        )
        await db.commit()
        return RedirectResponse(url="/admin/settings?success=auto_enrich_updated", status_code=302)

    elif action == "set_enrich_config":
        model_name = form.get("enrich_model", "").strip()
        api_key_val = form.get("enrich_api_key", "").strip()
        brave_key_val = form.get("brave_api_key", "").strip()
        await crud.set_config(
            db, "catalog.enrich_model", model_name,
            description="Model used for auto-enrichment LLM calls"
        )
        if api_key_val:
            await crud.set_config(
                db, "catalog.enrich_api_key", api_key_val,
                description="API key for internal MindRouter enrichment calls"
            )
        if brave_key_val:
            await crud.set_config(
                db, "catalog.brave_api_key", brave_key_val,
                description="Brave Search API key for enrichment (overrides env var)"
            )
        await crud.log_admin_action(
            db, user_id=user_id, action="settings.set_enrich_config",
            entity_type="config", detail=f"enrich_model={model_name}", ip_address=_ip,
        )
        await db.commit()
        return RedirectResponse(url="/admin/settings?success=enrich_config_updated", status_code=302)

    elif action == "set_agreement":
        text = form.get("agreement_text", "").strip()
        if not text:
            return RedirectResponse(url="/admin/settings?error=Agreement+text+cannot+be+empty", status_code=302)
        bump = form.get("bump_version") == "on"
        await crud.set_agreement_text(db, text, bump_version=bump)
        await crud.log_admin_action(
            db, user_id=user_id, action="settings.set_agreement",
            entity_type="config", detail=f"bump_version={bump}", ip_address=_ip,
        )
        await db.commit()
        msg = "agreement_updated_and_bumped" if bump else "agreement_updated"
        return RedirectResponse(url=f"/admin/settings?success={msg}", status_code=302)

    elif action == "set_email_config":
        from backend.app.services import email_service
        keys_map = {
            "smtp_host": ("email.smtp_host", "SMTP server hostname"),
            "smtp_port": ("email.smtp_port", "SMTP server port"),
            "smtp_username": ("email.smtp_username", "SMTP username"),
            "smtp_password": ("email.smtp_password", "SMTP password"),
            "smtp_default_sender": ("email.default_sender", "Default sender address"),
            "smtp_test_address": ("email.test_address", "Test recipient address"),
            "smtp_blog_sender": ("email.blog_sender", "Blog notification sender address"),
        }
        for field, (config_key, desc) in keys_map.items():
            val = form.get(field, "").strip()
            # Skip empty password field (keep existing)
            if field == "smtp_password" and not val:
                continue
            await crud.set_config(db, config_key, val, description=desc)
        # TLS is a checkbox
        use_tls = form.get("smtp_use_tls") == "on"
        await crud.set_config(db, "email.use_tls", use_tls, description="Use TLS for SMTP")
        await crud.log_admin_action(
            db, user_id=user_id, action="settings.set_email_config",
            entity_type="config", ip_address=_ip,
        )
        await db.commit()
        return RedirectResponse(url="/admin/settings?success=email_config_updated", status_code=302)

    elif action == "test_email":
        from backend.app.services import email_service
        smtp_config = await email_service.get_smtp_config(db)
        if not email_service.is_smtp_configured(smtp_config):
            return RedirectResponse(url="/admin/settings?error=SMTP+not+configured", status_code=302)
        test_addr = smtp_config.get("test_address") or smtp_config.get("default_sender")
        if not test_addr:
            return RedirectResponse(url="/admin/settings?error=No+test+address+configured", status_code=302)
        try:
            smtp = await email_service._open_smtp(smtp_config)
            try:
                await email_service._send_one(
                    smtp, smtp_config["default_sender"], test_addr,
                    "MindRouter Test Email",
                    email_service._wrap_html("<p>This is a test email from MindRouter.</p>", base_url=await email_service.get_base_url(db)),
                )
            finally:
                await smtp.quit()
            return RedirectResponse(url=f"/admin/settings?success=test_email_sent", status_code=302)
        except Exception as e:
            from urllib.parse import quote_plus
            return RedirectResponse(url=f"/admin/settings?error={quote_plus(str(e))}", status_code=302)

    return RedirectResponse(url="/admin/settings?error=Unknown+action", status_code=302)


async def _init_tz_cache(db: AsyncSession) -> None:
    """Load timezone from DB into cache. Call at startup or first request."""
    tz_name = await crud.get_config_json(db, "app.timezone", "America/Los_Angeles")
    _refresh_tz_cache_sync(tz_name)


# ------------------------------------------------------------------
# Data Retention
# ------------------------------------------------------------------


@dashboard_router.get("/admin/retention", response_class=HTMLResponse)
async def admin_retention(
    request: Request,
    tab: str = "policies",
    page: int = 1,
    category: str = "requests",
    model_filter: Optional[str] = None,
    user_id_filter: Optional[str] = None,
    success: Optional[str] = None,
    error: Optional[str] = None,
    db: AsyncSession = Depends(get_async_db),
):
    """Admin data retention page."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.has_admin_read):
        return RedirectResponse(url="/dashboard", status_code=302)

    from backend.app.services.retention import (
        browse_archive,
        get_app_db_counts,
        get_archive_stats,
        get_retention_config,
        is_retention_running,
    )
    from backend.app.settings import get_settings

    parsed_uid = int(user_id_filter) if user_id_filter and user_id_filter.strip().isdigit() else None
    settings = get_settings()
    archive_configured = settings.archive_database_url is not None

    config = await get_retention_config(db)
    app_counts = await get_app_db_counts(db)
    retention_running = await is_retention_running()

    archive_stats = None
    browse_rows = []
    browse_total = 0

    if archive_configured:
        if tab == "stats" or tab == "browse":
            try:
                from backend.app.db.session import get_archive_db_context
                async with get_archive_db_context() as archive_db:
                    if tab == "stats":
                        archive_stats = await get_archive_stats(archive_db)
                    elif tab == "browse":
                        archive_stats = await get_archive_stats(archive_db)
                        browse_rows, browse_total = await browse_archive(
                            archive_db, category, page, 50,
                            model_filter=model_filter,
                            user_id_filter=parsed_uid,
                        )
            except Exception as e:
                error = f"Archive DB error: {e}"

    total_pages = max(1, (browse_total + 49) // 50)

    masq = await _admin_masquerade_context(request, user, db)
    return templates.TemplateResponse(
        "admin/retention.html",
        {
            "request": request,
            "user": user,
            **masq,
            "tab": tab,
            "config": config,
            "app_counts": app_counts,
            "archive_configured": archive_configured,
            "retention_running": retention_running,
            "archive_stats": archive_stats,
            "browse_rows": browse_rows,
            "browse_total": browse_total,
            "browse_category": category,
            "model_filter": model_filter or "",
            "user_id_filter": user_id_filter or "",
            "page": page,
            "total_pages": total_pages,
            "success": success,
            "error": error,
        },
    )


@dashboard_router.post("/admin/retention")
async def admin_retention_post(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """Handle retention form submissions."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.is_admin):
        return RedirectResponse(url="/dashboard", status_code=302)

    form = await request.form()
    action = form.get("action")

    if action == "update_policies":
        from backend.app.services.retention import save_retention_config

        updates = {}
        for key in [
            "retention.requests.tier1_days",
            "retention.requests.tier2_days",
            "retention.chat.tier1_days",
            "retention.chat.tier2_days",
            "retention.telemetry.tier1_days",
            "retention.telemetry.tier2_days",
            "retention.cleanup_interval",
            "retention.batch_size",
        ]:
            val = form.get(key)
            if val is not None:
                try:
                    updates[key] = int(val)
                except ValueError:
                    pass

        await save_retention_config(db, updates)
        await crud.log_admin_action(
            db, user_id=user_id, action="retention.update_policies",
            entity_type="config", after_value=updates,
            ip_address=get_client_ip(request),
        )
        await db.commit()
        return RedirectResponse(
            url="/admin/retention?success=policies_updated", status_code=302
        )

    elif action == "run_now":
        import asyncio
        from backend.app.services.retention import (
            is_retention_running,
            try_run_retention_with_lock,
        )

        # Refuse if a cycle is already running anywhere in the cluster.
        # The background task also re-checks via GET_LOCK to close the
        # race window between this probe and task start.
        if await is_retention_running():
            return RedirectResponse(
                url="/admin/retention?error=retention_already_running",
                status_code=302,
            )

        async def _run_manual_retention() -> None:
            try:
                await try_run_retention_with_lock("manual")
            except Exception:
                logger.exception("retention_manual_cycle_error")

        # Run in background so the redirect completes quickly.  Keep a
        # strong reference in _background_tasks so the event loop's
        # weak reference doesn't let the task get GC'd mid-run.
        task = asyncio.create_task(_run_manual_retention())
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)
        async with get_async_db_context() as audit_db:
            await crud.log_admin_action(
                audit_db, user_id=user_id, action="retention.run_now",
                entity_type="config",
                ip_address=get_client_ip(request),
            )
            await audit_db.commit()
        return RedirectResponse(
            url="/admin/retention?success=retention_triggered", status_code=302
        )

    return RedirectResponse(
        url="/admin/retention?error=Unknown+action", status_code=302
    )


# ---------------------------------------------------------------------------
# Backup & Restore
# ---------------------------------------------------------------------------


@dashboard_router.get("/admin/backup", response_class=HTMLResponse)
async def admin_backup(
    request: Request,
    success: Optional[str] = None,
    error: Optional[str] = None,
    db: AsyncSession = Depends(get_async_db),
):
    """Admin backup & restore page."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.has_admin_read):
        return RedirectResponse(url="/dashboard", status_code=302)

    masq = await _admin_masquerade_context(request, user, db)
    return templates.TemplateResponse(
        "admin/backup.html",
        {
            "request": request,
            "user": user,
            **masq,
            "success": success,
            "error": error,
            "restore_summary": None,
        },
    )


@dashboard_router.get("/admin/backup/export")
async def admin_backup_export(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """Download a JSON backup of all configuration tables."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.has_admin_read):
        return RedirectResponse(url="/dashboard", status_code=302)

    data = await crud.export_config_tables(db)
    content = json.dumps(data, indent=2, ensure_ascii=False)
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    filename = f"mindrouter-config-{date_str}.json"

    return StreamingResponse(
        iter([content]),
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@dashboard_router.post("/admin/backup/restore", response_class=HTMLResponse)
async def admin_backup_restore(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """Restore configuration from an uploaded JSON backup file."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.is_admin):
        return RedirectResponse(url="/dashboard", status_code=302)

    form = await request.form()
    backup_file = form.get("backup_file")

    if not backup_file or not hasattr(backup_file, "read"):
        return templates.TemplateResponse(
            "admin/backup.html",
            {
                "request": request,
                "user": user,
                "success": None,
                "error": "No file uploaded.",
                "restore_summary": None,
            },
        )

    try:
        contents = await backup_file.read()
        data = json.loads(contents)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return templates.TemplateResponse(
            "admin/backup.html",
            {
                "request": request,
                "user": user,
                "success": None,
                "error": f"Invalid JSON file: {exc}",
                "restore_summary": None,
            },
        )

    if "metadata" not in data:
        return templates.TemplateResponse(
            "admin/backup.html",
            {
                "request": request,
                "user": user,
                "success": None,
                "error": "Invalid backup file: missing metadata section.",
                "restore_summary": None,
            },
        )

    try:
        summary = await crud.import_config_tables(db, data)
        await crud.log_admin_action(
            db, user_id=user_id, action="backup.restore",
            entity_type="config",
            detail=f"restored from uploaded backup",
            ip_address=get_client_ip(request),
        )
        await db.commit()
    except Exception as exc:
        logger.error("Backup restore failed", error=str(exc))
        return templates.TemplateResponse(
            "admin/backup.html",
            {
                "request": request,
                "user": user,
                "success": None,
                "error": f"Restore failed: {exc}",
                "restore_summary": None,
            },
        )

    total_inserted = sum(v["inserted"] for v in summary.values())
    return templates.TemplateResponse(
        "admin/backup.html",
        {
            "request": request,
            "user": user,
            "success": f"Restore complete — {total_inserted} rows inserted.",
            "error": None,
            "restore_summary": summary,
        },
    )


# ── Queue Monitor ──────────────────────────────────────────
@dashboard_router.get("/admin/queue", response_class=HTMLResponse)
async def admin_queue(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """Admin queue monitor page."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.has_admin_read):
        return RedirectResponse(url="/dashboard", status_code=302)

    return templates.TemplateResponse(
        "admin/queue.html",
        {"request": request, "user": user},
    )


# ---- Web Search Config ----


@dashboard_router.get("/admin/search-config")
async def admin_search_config(
    request: Request,
    success: Optional[str] = None,
    error: Optional[str] = None,
    test_count: Optional[int] = None,
    db: AsyncSession = Depends(get_async_db),
):
    """Admin web search configuration page."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.has_admin_read):
        return RedirectResponse(url="/dashboard", status_code=302)

    from backend.app.services.search.registry import (
        get_search_config,
        list_providers,
        PROVIDERS,
    )

    config = await get_search_config(db)

    # Run health checks for all providers
    health_checks = []
    for p in PROVIDERS.values():
        healthy, message = await p.health_check(config)
        health_checks.append({
            "name": p.display_name,
            "key": p.provider_key,
            "healthy": healthy,
            "message": message,
        })

    masq = await _admin_masquerade_context(request, user, db)
    return templates.TemplateResponse(
        "admin/search_config.html",
        {
            "request": request,
            "user": user,
            **masq,
            "search_enabled": config.get("search.enabled", True),
            "active_provider": config.get("search.provider", "brave"),
            "max_results": config.get("search.max_results", 10),
            "quota_tokens": config.get("search.quota_tokens_per_request", 50),
            "brave_api_key": config.get("search.brave.api_key", ""),
            "brave_endpoint": config.get("search.brave.endpoint", ""),
            "searxng_endpoint": config.get("search.searxng.endpoint", ""),
            "providers": list_providers(),
            "health_checks": health_checks,
            "success": success,
            "error": error,
            "test_count": test_count,
            "test_results": None,
            "test_query": None,
        },
    )


@dashboard_router.post("/admin/search-config")
async def admin_search_config_post(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """Handle search config form submissions."""
    user_id = get_session_user_id(request)
    if not user_id:
        return RedirectResponse(url="/login", status_code=302)

    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.is_admin):
        return RedirectResponse(url="/dashboard", status_code=302)

    from backend.app.services.search.registry import (
        get_search_config,
        save_search_config,
        list_providers,
        PROVIDERS,
    )

    form = await request.form()
    action = form.get("action")
    _ip = get_client_ip(request)

    if action == "save_general":
        updates = {
            "search.enabled": bool(form.get("search_enabled")),
            "search.provider": form.get("provider", "brave"),
            "search.max_results": max(1, min(50, int(form.get("max_results", 10)))),
            "search.quota_tokens_per_request": max(0, min(10000, int(form.get("quota_tokens", 50)))),
        }
        await save_search_config(db, updates)
        await crud.log_admin_action(
            db, user_id=user_id, action="search_config.save_general",
            entity_type="config", after_value=updates, ip_address=_ip,
        )
        await db.commit()
        return RedirectResponse(url="/admin/search-config?success=settings_updated", status_code=302)

    elif action == "save_brave":
        updates = {
            "search.brave.api_key": form.get("brave_api_key", "").strip(),
            "search.brave.endpoint": form.get("brave_endpoint", "").strip()
                or "https://api.search.brave.com/res/v1/web/search",
        }
        await save_search_config(db, updates)
        await crud.log_admin_action(
            db, user_id=user_id, action="search_config.save_brave",
            entity_type="config",
            after_value={"endpoint": updates["search.brave.endpoint"]},
            ip_address=_ip,
        )
        await db.commit()
        return RedirectResponse(url="/admin/search-config?success=provider_updated", status_code=302)

    elif action == "save_searxng":
        updates = {
            "search.searxng.endpoint": form.get("searxng_endpoint", "").strip(),
        }
        await save_search_config(db, updates)
        await crud.log_admin_action(
            db, user_id=user_id, action="search_config.save_searxng",
            entity_type="config",
            after_value=updates,
            ip_address=_ip,
        )
        await db.commit()
        return RedirectResponse(url="/admin/search-config?success=provider_updated", status_code=302)

    elif action == "test_search":
        test_query = form.get("test_query", "").strip()
        if not test_query:
            return RedirectResponse(url="/admin/search-config?error=Empty+query", status_code=302)

        config = await get_search_config(db)
        provider_key = config.get("search.provider", "brave")
        provider = PROVIDERS.get(provider_key)

        if not provider:
            return RedirectResponse(
                url="/admin/search-config?error=Unknown+provider", status_code=302
            )

        try:
            results = await provider.search(
                test_query,
                max_results=int(config.get("search.max_results", 5)),
                config=config,
            )
        except Exception as e:
            return RedirectResponse(
                url=f"/admin/search-config?error={str(e)[:100]}", status_code=302
            )

        # Re-render page with test results inline
        health_checks = []
        for p in PROVIDERS.values():
            healthy, message = await p.health_check(config)
            health_checks.append({
                "name": p.display_name,
                "key": p.provider_key,
                "healthy": healthy,
                "message": message,
            })

        masq = await _admin_masquerade_context(request, user, db)
        return templates.TemplateResponse(
            "admin/search_config.html",
            {
                "request": request,
                "user": user,
                **masq,
                "search_enabled": config.get("search.enabled", True),
                "active_provider": config.get("search.provider", "brave"),
                "max_results": config.get("search.max_results", 10),
                "quota_tokens": config.get("search.quota_tokens_per_request", 50),
                "brave_api_key": config.get("search.brave.api_key", ""),
                "brave_endpoint": config.get("search.brave.endpoint", ""),
                "searxng_endpoint": config.get("search.searxng.endpoint", ""),
                "providers": list_providers(),
                "health_checks": health_checks,
                "success": f"test_ok",
                "error": None,
                "test_count": len(results),
                "test_results": [r.to_dict() for r in results],
                "test_query": test_query,
            },
        )

    return RedirectResponse(url="/admin/search-config", status_code=302)
