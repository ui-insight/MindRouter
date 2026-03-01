############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
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

"""Blog routes for MindRouter2."""

import os
import re
from datetime import datetime, timezone
from typing import Optional

import markdown
from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db import crud
from backend.app.db.session import get_async_db
from backend.app.dashboard.routes import get_session_user_id
from backend.app.settings import get_settings

blog_router = APIRouter(tags=["blog"])

# Setup templates
templates_path = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=templates_path)


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
async def _require_admin(request: Request, db: AsyncSession):
    """Helper to require admin access."""
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
    user, redirect = await _require_admin(request, db)
    if redirect:
        return redirect

    posts = await crud.get_all_blog_posts(db)

    return templates.TemplateResponse(
        "admin/blog.html",
        {"request": request, "user": user, "posts": posts, "active": "blog"},
    )


@blog_router.get("/admin/blog/new", response_class=HTMLResponse)
async def admin_blog_new(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """Admin: new post form."""
    user, redirect = await _require_admin(request, db)
    if redirect:
        return redirect

    return templates.TemplateResponse(
        "admin/blog_edit.html",
        {"request": request, "user": user, "post": None, "active": "blog"},
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
    db: AsyncSession = Depends(get_async_db),
):
    """Admin: edit post form."""
    user, redirect = await _require_admin(request, db)
    if redirect:
        return redirect

    post = await crud.get_blog_post_by_id(db, post_id)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    return templates.TemplateResponse(
        "admin/blog_edit.html",
        {"request": request, "user": user, "post": post, "active": "blog"},
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
