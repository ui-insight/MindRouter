############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# email_routes.py: Admin email compose and send routes
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Admin email routes for MindRouter."""

import asyncio
from typing import Optional

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db import crud
from backend.app.db.session import get_async_db
from backend.app.dashboard.routes import get_session_user_id, templates
from backend.app.services import email_service
from backend.app.settings import get_settings

email_router = APIRouter(tags=["email"])


async def _require_admin(request: Request, db: AsyncSession):
    """Helper to require admin access."""
    user_id = get_session_user_id(request)
    if not user_id:
        return None, RedirectResponse("/login", status_code=302)
    user = await crud.get_user_by_id(db, user_id)
    if not user or not user.group or not user.group.is_admin:
        return None, RedirectResponse("/dashboard", status_code=302)
    return user, None


@email_router.get("/admin/email", response_class=HTMLResponse)
async def admin_email_page(
    request: Request,
    success: Optional[str] = None,
    error: Optional[str] = None,
    db: AsyncSession = Depends(get_async_db),
):
    """Admin email compose page."""
    user, redirect = await _require_admin(request, db)
    if redirect:
        return redirect

    groups = await crud.get_all_groups(db)
    smtp_config = await email_service.get_smtp_config(db)
    smtp_ready = email_service.is_smtp_configured(smtp_config)
    email_logs = await crud.get_email_logs(db, limit=15)

    return templates.TemplateResponse(
        "admin/email.html",
        {
            "request": request,
            "user": user,
            "groups": groups,
            "smtp_ready": smtp_ready,
            "email_logs": email_logs,
            "success": success,
            "error": error,
            "active": "email",
        },
    )


@email_router.post("/admin/email/recipient-count")
async def recipient_count(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """AJAX: return count of recipients for given selection."""
    user, redirect = await _require_admin(request, db)
    if redirect:
        return JSONResponse({"error": "Unauthorized"}, status_code=403)

    body = await request.json()
    mode = body.get("mode", "all")
    group_ids = body.get("group_ids", [])
    user_ids = body.get("user_ids", [])

    if mode == "all":
        users = await crud.get_emailable_users(db)
    elif mode == "groups":
        users = await crud.get_emailable_users(db, group_ids=[int(g) for g in group_ids])
    elif mode == "users":
        users = await crud.get_emailable_users(db, user_ids=[int(u) for u in user_ids])
    else:
        users = []

    return JSONResponse({"count": len(users)})


@email_router.post("/admin/email/send")
async def send_email(
    request: Request,
    subject: str = Form(...),
    body: str = Form(...),
    recipient_mode: str = Form("all"),
    group_ids: Optional[str] = Form(None),
    user_ids: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_async_db),
):
    """Send bulk email."""
    user, redirect = await _require_admin(request, db)
    if redirect:
        return redirect

    smtp_config = await email_service.get_smtp_config(db)
    if not email_service.is_smtp_configured(smtp_config):
        return RedirectResponse("/admin/email?error=SMTP+not+configured", status_code=302)

    # Resolve recipients
    gids = [int(g) for g in group_ids.split(",") if g.strip()] if group_ids else None
    uids = [int(u) for u in user_ids.split(",") if u.strip()] if user_ids else None

    if recipient_mode == "groups" and gids:
        users = await crud.get_emailable_users(db, group_ids=gids)
    elif recipient_mode == "users" and uids:
        users = await crud.get_emailable_users(db, user_ids=uids)
    else:
        users = await crud.get_emailable_users(db)

    if not users:
        return RedirectResponse("/admin/email?error=No+recipients+found", status_code=302)

    # Wrap body in HTML email template
    base_url = get_settings().app_base_url
    html_body = email_service._wrap_html(body, base_url=base_url)

    # Create log
    log = await crud.create_email_log(
        db, subject=subject, sent_by=user.id,
        recipient_count=len(users), body_preview=subject,
    )
    await db.commit()

    # Build recipient list
    recipient_list = [
        {"email": u.email, "username": u.username, "full_name": u.full_name or ""}
        for u in users
    ]

    # Fire and forget
    asyncio.create_task(
        email_service.send_bulk_email(
            log.id, subject, html_body, recipient_list,
            smtp_config["default_sender"], smtp_config,
        )
    )

    return RedirectResponse(
        f"/admin/email?success=Sending+to+{len(users)}+recipients", status_code=302
    )


@email_router.post("/admin/email/send-test")
async def send_test_compose(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """Send a test of the composed email to the test address."""
    user, redirect = await _require_admin(request, db)
    if redirect:
        return JSONResponse({"error": "Unauthorized"}, status_code=403)

    body = await request.json()
    subject = body.get("subject", "Test Email")
    content = body.get("body", "")

    smtp_config = await email_service.get_smtp_config(db)
    if not email_service.is_smtp_configured(smtp_config):
        return JSONResponse({"error": "SMTP not configured"}, status_code=400)

    test_addr = smtp_config.get("test_address") or smtp_config.get("default_sender")
    if not test_addr:
        return JSONResponse({"error": "No test address configured"}, status_code=400)

    # Personalize with test values
    base_url = get_settings().app_base_url
    html_body = email_service._wrap_html(content, base_url=base_url)
    test_user = {"email": test_addr, "username": "testuser", "full_name": "Test User"}
    personalized = email_service._personalize(html_body, test_user)

    err = await email_service.send_test_email.__wrapped__(smtp_config, test_addr) if False else ""
    # Actually send directly
    try:
        smtp = await email_service._open_smtp(smtp_config)
        try:
            await email_service._send_one(
                smtp, smtp_config["default_sender"], test_addr, subject, personalized
            )
        finally:
            await smtp.quit()
        return JSONResponse({"ok": True, "message": f"Test sent to {test_addr}"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@email_router.get("/admin/email/user-search")
async def user_search(
    request: Request,
    q: str = "",
    db: AsyncSession = Depends(get_async_db),
):
    """AJAX: search users for the recipient picker."""
    user, redirect = await _require_admin(request, db)
    if redirect:
        return JSONResponse({"error": "Unauthorized"}, status_code=403)

    if len(q) < 2:
        return JSONResponse({"users": []})

    users, _ = await crud.get_users(db, search=q, limit=20, is_active=True)
    return JSONResponse({
        "users": [
            {
                "id": u.id,
                "username": u.username,
                "full_name": u.full_name or "",
                "email": u.email,
                "group": u.group.display_name if u.group else "",
            }
            for u in users
        ]
    })
