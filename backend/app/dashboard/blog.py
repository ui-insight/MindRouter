############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# blog.py: Blog routes (public viewing + admin CRUD)
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Blog routes for MindRouter."""

import asyncio
import re
from datetime import datetime, timezone
from typing import Optional

import markdown
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db import crud
from backend.app.db.session import get_async_db
from backend.app.dashboard.routes import get_session_user_id, _admin_masquerade_context, templates
from backend.app.services import email_service
from backend.app.storage.artifacts import get_artifact_storage

blog_router = APIRouter(tags=["blog"])

# Allowed image MIME types for blog uploads
_ALLOWED_IMAGE_TYPES = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/svg+xml": ".svg",
}
_MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB


def _render_markdown(text: str) -> str:
    """Render markdown to HTML with syntax highlighting."""
    return markdown.markdown(
        text,
        extensions=["fenced_code", "codehilite", "tables", "toc"],
        extension_configs={
            "codehilite": {"css_class": "codehilite", "guess_lang": False},
        },
    )


def _slugify(title: str) -> str:
    """Generate a URL slug from a title."""
    slug = title.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[-\s]+", "-", slug)
    return slug.strip("-")


# Public routes
@blog_router.get("/blog", response_class=HTMLResponse)
async def blog_index(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """Public blog listing."""
    user_id = get_session_user_id(request)
    user = None
    if user_id:
        user = await crud.get_user_by_id(db, user_id)

    posts = await crud.get_published_blog_posts(db)

    return templates.TemplateResponse(
        "blog/index.html",
        {"request": request, "user": user, "posts": posts},
    )


@blog_router.get("/blog/{slug}", response_class=HTMLResponse)
async def blog_post(
    request: Request,
    slug: str,
    db: AsyncSession = Depends(get_async_db),
):
    """Public single post view."""
    user_id = get_session_user_id(request)
    user = None
    if user_id:
        user = await crud.get_user_by_id(db, user_id)

    post = await crud.get_blog_post_by_slug(db, slug)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    content_html = _render_markdown(post.content)

    return templates.TemplateResponse(
        "blog/post.html",
        {"request": request, "user": user, "post": post, "content_html": content_html},
    )


# Admin routes
async def _require_admin_read(request: Request, db: AsyncSession):
    """Helper to require admin or auditor access (read-only admin)."""
    user_id = get_session_user_id(request)
    if not user_id:
        return None, RedirectResponse("/login", status_code=302)
    user = await crud.get_user_by_id(db, user_id)
    if not user or not user.group or not user.group.has_admin_read:
        raise HTTPException(status_code=403, detail="Admin access required")
    return user, None


async def _require_admin(request: Request, db: AsyncSession):
    """Helper to require full admin access (mutating actions)."""
    user_id = get_session_user_id(request)
    if not user_id:
        return None, RedirectResponse("/login", status_code=302)
    user = await crud.get_user_by_id(db, user_id)
    if not user or not user.group or not user.group.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    return user, None


@blog_router.get("/admin/blog", response_class=HTMLResponse)
async def admin_blog_list(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """Admin: list all blog posts."""
    user, redirect = await _require_admin_read(request, db)
    if redirect:
        return redirect

    posts = await crud.get_all_blog_posts(db)

    masq = await _admin_masquerade_context(request, user, db)
    return templates.TemplateResponse(
        "admin/blog.html",
        {"request": request, "user": user, **masq, "posts": posts, "active": "blog"},
    )


@blog_router.get("/admin/blog/new", response_class=HTMLResponse)
async def admin_blog_new(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """Admin: new post form."""
    user, redirect = await _require_admin_read(request, db)
    if redirect:
        return redirect

    masq = await _admin_masquerade_context(request, user, db)
    return templates.TemplateResponse(
        "admin/blog_edit.html",
        {"request": request, "user": user, **masq, "post": None, "active": "blog"},
    )


@blog_router.post("/admin/blog/new", response_class=HTMLResponse)
async def admin_blog_create(
    request: Request,
    title: str = Form(...),
    slug: str = Form(...),
    content: str = Form(...),
    excerpt: str = Form(""),
    is_published: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_async_db),
):
    """Admin: create a new blog post."""
    user, redirect = await _require_admin(request, db)
    if redirect:
        return redirect

    publish = is_published == "on"
    await crud.create_blog_post(
        db,
        title=title,
        slug=slug,
        content=content,
        excerpt=excerpt or None,
        author_id=user.id,
        is_published=publish,
    )
    await db.commit()

    return RedirectResponse("/admin/blog", status_code=302)


@blog_router.get("/admin/blog/{post_id}/edit", response_class=HTMLResponse)
async def admin_blog_edit(
    request: Request,
    post_id: int,
    success: Optional[str] = None,
    error: Optional[str] = None,
    db: AsyncSession = Depends(get_async_db),
):
    """Admin: edit post form."""
    user, redirect = await _require_admin_read(request, db)
    if redirect:
        return redirect

    post = await crud.get_blog_post_by_id(db, post_id)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    smtp_config = await email_service.get_smtp_config(db)
    smtp_ready = email_service.is_smtp_configured(smtp_config)
    blog_email_log = await crud.get_blog_email_log(db, post_id)

    masq = await _admin_masquerade_context(request, user, db)
    return templates.TemplateResponse(
        "admin/blog_edit.html",
        {
            "request": request,
            "user": user,
            **masq,
            "post": post,
            "active": "blog",
            "smtp_ready": smtp_ready,
            "blog_email_log": blog_email_log,
            "success": success,
            "error": error,
        },
    )


@blog_router.post("/admin/blog/{post_id}/edit", response_class=HTMLResponse)
async def admin_blog_update(
    request: Request,
    post_id: int,
    title: str = Form(...),
    slug: str = Form(...),
    content: str = Form(...),
    excerpt: str = Form(""),
    is_published: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_async_db),
):
    """Admin: update a blog post."""
    user, redirect = await _require_admin(request, db)
    if redirect:
        return redirect

    publish = is_published == "on"
    kwargs = {
        "title": title,
        "slug": slug,
        "content": content,
        "excerpt": excerpt or None,
        "is_published": publish,
    }
    # Set published_at when first publishing
    post = await crud.get_blog_post_by_id(db, post_id)
    if post and publish and not post.published_at:
        kwargs["published_at"] = datetime.now(timezone.utc)

    await crud.update_blog_post(db, post_id, **kwargs)
    await db.commit()

    return RedirectResponse("/admin/blog", status_code=302)


@blog_router.post("/admin/blog/{post_id}/delete", response_class=HTMLResponse)
async def admin_blog_delete(
    request: Request,
    post_id: int,
    db: AsyncSession = Depends(get_async_db),
):
    """Admin: soft delete a blog post."""
    user, redirect = await _require_admin(request, db)
    if redirect:
        return redirect

    await crud.delete_blog_post(db, post_id)
    await db.commit()

    return RedirectResponse("/admin/blog", status_code=302)


@blog_router.post("/admin/blog/{post_id}/publish", response_class=HTMLResponse)
async def admin_blog_toggle_publish(
    request: Request,
    post_id: int,
    db: AsyncSession = Depends(get_async_db),
):
    """Admin: toggle publish status."""
    user, redirect = await _require_admin(request, db)
    if redirect:
        return redirect

    post = await crud.get_blog_post_by_id(db, post_id)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    kwargs = {"is_published": not post.is_published}
    if not post.is_published and not post.published_at:
        kwargs["published_at"] = datetime.now(timezone.utc)

    await crud.update_blog_post(db, post_id, **kwargs)
    await db.commit()

    return RedirectResponse("/admin/blog", status_code=302)


@blog_router.post("/admin/blog/{post_id}/send-email")
async def admin_blog_send_email(
    request: Request,
    post_id: int,
    db: AsyncSession = Depends(get_async_db),
):
    """Send blog post as email to all users (excluding opt-outs)."""
    user, redirect = await _require_admin(request, db)
    if redirect:
        return redirect

    post = await crud.get_blog_post_by_id(db, post_id)
    if not post or not post.is_published:
        return RedirectResponse(f"/admin/blog/{post_id}/edit?error=Post+must+be+published", status_code=302)

    smtp_config = await email_service.get_smtp_config(db)
    if not email_service.is_smtp_configured(smtp_config):
        return RedirectResponse(f"/admin/blog/{post_id}/edit?error=SMTP+not+configured", status_code=302)

    users = await crud.get_emailable_users(db, exclude_blog_optout=True)
    if not users:
        return RedirectResponse(f"/admin/blog/{post_id}/edit?error=No+recipients", status_code=302)

    base_url = await email_service.get_base_url(db)

    subject = f"MindRouter Blog: {post.title}"
    html_body = email_service._render_blog_email(
        post.title, post.content, post.slug,
        user.full_name or user.username, base_url,
    )

    sender = smtp_config.get("blog_sender") or smtp_config.get("default_sender")

    log = await crud.create_email_log(
        db, subject=subject, sent_by=user.id,
        recipient_count=len(users), body_preview=post.title,
        blog_post_id=post.id,
    )
    await db.commit()

    recipient_list = [
        {"email": u.email, "username": u.username, "full_name": u.full_name or ""}
        for u in users
    ]

    asyncio.create_task(
        email_service.send_bulk_email(
            log.id, subject, html_body, recipient_list, sender, smtp_config,
        )
    )

    return RedirectResponse(
        f"/admin/blog/{post_id}/edit?success=Sending+to+{len(users)}+recipients", status_code=302
    )


@blog_router.post("/admin/blog/{post_id}/send-test-email")
async def admin_blog_send_test_email(
    request: Request,
    post_id: int,
    db: AsyncSession = Depends(get_async_db),
):
    """Send blog post as test email to the configured test address."""
    user, redirect = await _require_admin(request, db)
    if redirect:
        return redirect

    post = await crud.get_blog_post_by_id(db, post_id)
    if not post:
        return JSONResponse({"error": "Post not found"}, status_code=404)

    smtp_config = await email_service.get_smtp_config(db)
    if not email_service.is_smtp_configured(smtp_config):
        return JSONResponse({"error": "SMTP not configured"}, status_code=400)

    test_addr = smtp_config.get("test_address") or smtp_config.get("default_sender")
    if not test_addr:
        return JSONResponse({"error": "No test address configured"}, status_code=400)

    base_url = await email_service.get_base_url(db)

    subject = f"[TEST] MindRouter Blog: {post.title}"
    html_body = email_service._render_blog_email(
        post.title, post.content, post.slug,
        user.full_name or user.username, base_url,
    )
    sender = smtp_config.get("blog_sender") or smtp_config.get("default_sender")

    test_user = {"email": test_addr, "username": "testuser", "full_name": "Test User"}
    personalized = email_service._personalize(html_body, test_user)

    try:
        smtp = await email_service._open_smtp(smtp_config)
        try:
            await email_service._send_one(smtp, sender, test_addr, subject, personalized)
        finally:
            await smtp.quit()
        return JSONResponse({"ok": True, "message": f"Test sent to {test_addr}"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------------------------------------------------------------------------
# Blog image upload & serving
# ---------------------------------------------------------------------------

@blog_router.post("/admin/blog/upload-image")
async def admin_blog_upload_image(
    request: Request,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_async_db),
):
    """Upload an image for use in blog posts. Returns the markdown image tag."""
    user_id = get_session_user_id(request)
    if not user_id:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)
    user = await crud.get_user_by_id(db, user_id)
    if not user or (not user.group or not user.group.is_admin):
        return JSONResponse({"error": "Admin access required"}, status_code=403)

    # Validate content type
    content_type = file.content_type or ""
    if content_type not in _ALLOWED_IMAGE_TYPES:
        return JSONResponse(
            {"error": f"Unsupported image type: {content_type}. Allowed: {', '.join(_ALLOWED_IMAGE_TYPES.keys())}"},
            status_code=400,
        )

    # Read and validate size
    data = await file.read()
    if len(data) > _MAX_IMAGE_SIZE:
        return JSONResponse(
            {"error": f"Image too large ({len(data) / 1024 / 1024:.1f} MB). Maximum: {_MAX_IMAGE_SIZE / 1024 / 1024:.0f} MB"},
            status_code=400,
        )

    # Store via artifact storage
    storage = get_artifact_storage()
    storage_path, sha256, size = await storage.store(data, file.filename or "image", content_type)

    # Build the URL for the blog image
    image_url = f"/blog/images/{storage_path}"
    markdown_tag = f"![{file.filename or 'image'}]({image_url})"

    return JSONResponse({
        "url": image_url,
        "markdown": markdown_tag,
        "filename": file.filename,
        "size": size,
    })


@blog_router.get("/blog/images/{path:path}")
async def serve_blog_image(path: str):
    """Serve a blog image from artifact storage."""
    storage = get_artifact_storage()
    data = await storage.retrieve(path)
    if data is None:
        raise HTTPException(status_code=404, detail="Image not found")

    # Determine content type from extension
    ext = "." + path.rsplit(".", 1)[-1] if "." in path else ""
    content_types = {v: k for k, v in _ALLOWED_IMAGE_TYPES.items()}
    content_type = content_types.get(ext, "application/octet-stream")

    return Response(
        content=data,
        media_type=content_type,
        headers={"Cache-Control": "public, max-age=86400"},
    )
