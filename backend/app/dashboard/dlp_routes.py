############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# dlp_routes.py: Admin DLP configuration and alerts routes
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Admin DLP (Data Loss Prevention) routes for MindRouter."""

import json
from typing import Optional

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db import crud
from backend.app.db.session import get_async_db
from backend.app.dashboard.routes import get_client_ip, get_session_user_id, _admin_masquerade_context, templates

dlp_router = APIRouter(tags=["dlp"])


async def _require_admin_read(request: Request, db: AsyncSession):
    """Helper to require admin or auditor access (read-only admin)."""
    user_id = get_session_user_id(request)
    if not user_id:
        return None, RedirectResponse("/login", status_code=302)
    user = await crud.get_user_by_id(db, user_id)
    if not user or not user.group or not user.group.has_admin_read:
        return None, RedirectResponse("/dashboard", status_code=302)
    return user, None


async def _require_admin(request: Request, db: AsyncSession):
    """Helper to require full admin access (mutating actions)."""
    user_id = get_session_user_id(request)
    if not user_id:
        return None, RedirectResponse("/login", status_code=302)
    user = await crud.get_user_by_id(db, user_id)
    if not user or not user.group or not user.group.is_admin:
        return None, RedirectResponse("/dashboard", status_code=302)
    return user, None


@dlp_router.get("/admin/dlp", response_class=HTMLResponse)
async def admin_dlp_page(
    request: Request,
    success: Optional[str] = None,
    error: Optional[str] = None,
    severity: Optional[str] = None,
    scanner: Optional[str] = None,
    search: Optional[str] = None,
    page: int = 1,
    db: AsyncSession = Depends(get_async_db),
):
    """Admin DLP configuration and alerts page."""
    user, redirect = await _require_admin_read(request, db)
    if redirect:
        return redirect

    # Load DLP config
    config = {
        "enabled": await crud.get_config_json(db, "dlp.enabled", False),
        "regex_enabled": await crud.get_config_json(db, "dlp.regex.enabled", True),
        "regex_patterns": await crud.get_config_json(db, "dlp.regex.patterns", []),
        "regex_keywords": await crud.get_config_json(db, "dlp.regex.keywords", []),
        "gliner_enabled": await crud.get_config_json(db, "dlp.gliner.enabled", False),
        "gliner_threshold": await crud.get_config_json(db, "dlp.gliner.threshold", 0.5),
        "gliner_categories": await crud.get_config_json(db, "dlp.gliner.categories", []),
        "llm_enabled": await crud.get_config_json(db, "dlp.llm.enabled", False),
        "llm_model": await crud.get_config_json(db, "dlp.llm.model", ""),
        "llm_system_prompt": await crud.get_config_json(db, "dlp.llm.system_prompt", ""),
        "severity_rules": await crud.get_config_json(db, "dlp.severity_rules", {}),
        "email_minor": await crud.get_config_json(db, "dlp.email.minor_recipients", ""),
        "email_moderate": await crud.get_config_json(db, "dlp.email.moderate_recipients", ""),
        "email_major": await crud.get_config_json(db, "dlp.email.major_recipients", ""),
    }

    # Load alerts with pagination
    per_page = 25
    skip = (page - 1) * per_page
    alerts, total = await crud.get_dlp_alerts(
        db, severity=severity, scanner=scanner, search=search,
        skip=skip, limit=per_page,
    )
    total_pages = max(1, (total + per_page - 1) // per_page)

    # Load stats
    stats = await crud.get_dlp_stats(db)

    masq = await _admin_masquerade_context(request, user, db)
    return templates.TemplateResponse(
        "admin/dlp.html",
        {
            "request": request,
            "user": user,
            **masq,
            "config": config,
            "alerts": alerts,
            "stats": stats,
            "total": total,
            "page": page,
            "total_pages": total_pages,
            "severity_filter": severity,
            "scanner_filter": scanner,
            "search": search or "",
            "success": success,
            "error": error,
            "active": "dlp",
        },
    )


@dlp_router.post("/admin/dlp/config")
async def save_dlp_config(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """Save DLP configuration (requires admin)."""
    user, redirect = await _require_admin(request, db)
    if redirect:
        return RedirectResponse("/admin/dlp?error=Unauthorized", status_code=302)

    try:
        form = await request.form()

        await crud.set_config(db, "dlp.enabled", form.get("enabled") == "on")
        await crud.set_config(db, "dlp.regex.enabled", form.get("regex_enabled") == "on")
        await crud.set_config(db, "dlp.gliner.enabled", form.get("gliner_enabled") == "on")
        await crud.set_config(db, "dlp.llm.enabled", form.get("llm_enabled") == "on")

        # GLiNER settings
        threshold = form.get("gliner_threshold", "0.5")
        try:
            await crud.set_config(db, "dlp.gliner.threshold", float(threshold))
        except ValueError:
            pass

        categories = form.getlist("gliner_categories")
        if categories:
            await crud.set_config(db, "dlp.gliner.categories", categories)

        # LLM settings
        llm_model = form.get("llm_model", "")
        await crud.set_config(db, "dlp.llm.model", llm_model)

        llm_prompt = form.get("llm_system_prompt", "")
        await crud.set_config(db, "dlp.llm.system_prompt", llm_prompt)

        # Severity rules
        severity_rules_str = form.get("severity_rules", "{}")
        try:
            severity_rules = json.loads(severity_rules_str)
            await crud.set_config(db, "dlp.severity_rules", severity_rules)
        except json.JSONDecodeError:
            pass

        # Custom regex patterns
        regex_patterns_str = form.get("regex_patterns", "[]")
        try:
            regex_patterns = json.loads(regex_patterns_str)
            await crud.set_config(db, "dlp.regex.patterns", regex_patterns)
        except json.JSONDecodeError:
            pass

        # Keywords
        keywords_str = form.get("regex_keywords", "")
        keywords = [k.strip() for k in keywords_str.split("\n") if k.strip()]
        await crud.set_config(db, "dlp.regex.keywords", keywords)

        # Email recipients
        await crud.set_config(db, "dlp.email.minor_recipients", form.get("email_minor", ""))
        await crud.set_config(db, "dlp.email.moderate_recipients", form.get("email_moderate", ""))
        await crud.set_config(db, "dlp.email.major_recipients", form.get("email_major", ""))

        # Ensure internal API key exists if LLM scanner is enabled
        if form.get("llm_enabled") == "on":
            from backend.app.services.dlp_worker import ensure_internal_api_key
            await ensure_internal_api_key(db)

        await crud.log_admin_action(
            db, user.id, "update", "dlp_config",
            detail="DLP configuration updated",
            ip_address=get_client_ip(request),
        )
        await db.commit()

        return RedirectResponse("/admin/dlp?success=DLP+configuration+saved", status_code=302)
    except Exception as e:
        await db.rollback()
        return RedirectResponse(f"/admin/dlp?error={str(e)[:100]}", status_code=302)


@dlp_router.post("/admin/dlp/acknowledge/{alert_id}")
async def acknowledge_alert(
    request: Request,
    alert_id: int,
    db: AsyncSession = Depends(get_async_db),
):
    """Acknowledge a DLP alert (requires admin)."""
    user, redirect = await _require_admin(request, db)
    if redirect:
        return JSONResponse({"error": "Unauthorized"}, status_code=403)

    alert = await crud.acknowledge_dlp_alert(db, alert_id, user.id)
    if alert is None:
        return JSONResponse({"error": "Alert not found"}, status_code=404)

    await crud.log_admin_action(
        db, user.id, "acknowledge", "dlp_alert",
        entity_id=str(alert_id),
        ip_address=get_client_ip(request),
    )
    await db.commit()

    return JSONResponse({"ok": True, "alert_id": alert_id})


@dlp_router.get("/admin/dlp/stats")
async def dlp_stats(
    request: Request,
    hours: int = 24,
    db: AsyncSession = Depends(get_async_db),
):
    """JSON stats endpoint for DLP dashboard cards."""
    user, redirect = await _require_admin_read(request, db)
    if redirect:
        return JSONResponse({"error": "Unauthorized"}, status_code=403)

    stats = await crud.get_dlp_stats(db, hours=hours)
    return JSONResponse(stats)
