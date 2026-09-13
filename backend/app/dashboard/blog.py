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
import html
import re
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlencode

import markdown
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from sqlalchemy.exc import DataError, IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db import crud
from backend.app.db.session import get_async_db
from backend.app.dashboard.routes import get_session_user_id, _admin_masquerade_context, templates
from backend.app.services import email_service
from backend.app.storage.artifacts import get_artifact_storage
from backend.app.dashboard import blog_export
from backend.app.services import branding as branding_service
from backend.app.settings import get_settings

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


# ---------------------------------------------------------------------------
# HTML sanitization for rendered markdown.
#
# Blog posts are admin-authored, but the rendered HTML is emitted with |safe on
# the PUBLIC blog page, so we strip active content as defense-in-depth. When a
# real HTML sanitizer (nh3) is present we use it; otherwise we fall back to a
# conservative regex scrub. Code fences are unaffected because markdown already
# entity-encodes their content (a `<script>` sample renders as `&lt;script&gt;`).
# ---------------------------------------------------------------------------
try:  # pragma: no cover - optional dependency
    import nh3 as _nh3
except Exception:  # pragma: no cover
    _nh3 = None

_ACTIVE_TAG_PAIR_RE = re.compile(
    r"<\s*(script|iframe|object|embed|form|style|link|meta|base)\b[^>]*>.*?<\s*/\s*\1\s*>",
    re.IGNORECASE | re.DOTALL,
)
_ACTIVE_TAG_RE = re.compile(
    r"<\s*/?\s*(script|iframe|object|embed|form|style|link|meta|base)\b[^>]*>",
    re.IGNORECASE,
)
_ON_ATTR_RE = re.compile(
    r"""\son\w+\s*=\s*("[^"]*"|'[^']*'|[^\s>]+)""", re.IGNORECASE
)
_JS_URI_RE = re.compile(r"""(=\s*["']?)\s*javascript:""", re.IGNORECASE)


# Explicit nh3 allowlist covering exactly what the markdown pipeline emits
# (fenced_code + codehilite + tables + toc). nh3's DEFAULT allowlist drops
# `span`/`div` and strips the `class` attribute, which would silently break
# code-syntax highlighting (codehilite emits `<span class=...>` / `<div
# class="codehilite">`) and tables. Anything not listed here — script,
# iframe, on* handlers, javascript: — is still removed.
_NH3_TAGS = {
    "a", "abbr", "b", "blockquote", "br", "code", "div", "em",
    "h1", "h2", "h3", "h4", "h5", "h6", "hr", "i", "img", "li",
    "ol", "p", "pre", "span", "strong", "sub", "sup",
    "table", "tbody", "td", "th", "thead", "tr", "ul",
}
_NH3_ATTRS = {
    "a": {"href", "title"},
    "img": {"src", "alt", "title"},
    "span": {"class"},
    "code": {"class"},
    "pre": {"class"},
    "div": {"class"},
    "table": {"class"},
    "th": {"align"},
    "td": {"align"},
}


def _sanitize_html(html: str) -> str:
    """Neutralize active content in rendered markdown HTML (defense-in-depth)."""
    if _nh3 is not None:
        return _nh3.clean(html, tags=_NH3_TAGS, attributes=_NH3_ATTRS)
    html = _ACTIVE_TAG_PAIR_RE.sub("", html)
    html = _ACTIVE_TAG_RE.sub("", html)
    html = _ON_ATTR_RE.sub("", html)
    html = _JS_URI_RE.sub(r"\1unsafe:", html)
    return html


# Python-Markdown's raw-HTML scanner is built on CPython's html.parser. On
# current CPython builds (3.14, and Ubuntu 24.04's security-patched 3.12) a
# closing-tag-looking sequence such as `</dev/null` inside an INLINE code span
# is parsed as raw HTML and swallows the rest of the post: a 145 KB post was
# silently rendered up to that span and nothing after it (2026-09-13, Markdown
# 3.10.3). Fenced blocks are stashed before that scanner runs and are safe.
_FENCE_RE = re.compile(r"^(`{3,}|~{3,})[^\n]*\n.*?^\1[ \t]*$", re.MULTILINE | re.DOTALL)
_CODE_SPAN_RE = re.compile(r"(?<!`)`([^`\n]*</[^`\n]*)`(?!`)")


def _protect_code_spans(text: str) -> str:
    """Turn inline code spans that contain ``</`` into explicit
    ``<code>&lt;/…</code>`` HTML so the raw-HTML scanner never sees a tag."""
    def _span(m: re.Match) -> str:
        return "<code>" + html.escape(m.group(1), quote=False) + "</code>"

    out, pos = [], 0
    for fence in _FENCE_RE.finditer(text):
        out.append(_CODE_SPAN_RE.sub(_span, text[pos:fence.start()]))
        out.append(fence.group(0))
        pos = fence.end()
    out.append(_CODE_SPAN_RE.sub(_span, text[pos:]))
    return "".join(out)


def _render_markdown(text: str) -> str:
    """Render markdown to HTML with syntax highlighting (sanitized for the public page)."""
    text = _protect_code_spans(text)
    rendered = markdown.markdown(
        text,
        extensions=["fenced_code", "codehilite", "tables", "toc"],
        extension_configs={
            "codehilite": {"css_class": "codehilite", "guess_lang": False},
        },
    )
    return _sanitize_html(rendered)


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


# ---------------------------------------------------------------------------
# Public syndication feeds (pull model).
#
# External sites (e.g. a marketing/static site) PULL the admin-selected posts
# from here and render them with their own templates. The gateway never pushes
# and holds no external credentials. Both endpoints are public: they expose
# only posts that are already app-published AND flagged for syndication.
# ---------------------------------------------------------------------------

async def _syndication_base_url(db: AsyncSession) -> str:
    """This installation's public base URL (admin-configurable)."""
    base = await crud.get_config_json(db, "app.base_url", get_settings().app_base_url)
    return (base or "").rstrip("/")


@blog_router.get("/blog/feed.xml")
async def blog_feed_xml(db: AsyncSession = Depends(get_async_db)):
    """RSS 2.0 feed of syndicated posts. Public, cacheable."""
    posts = await crud.get_website_published_blog_posts(db)
    base_url = await _syndication_base_url(db)
    site_name = branding_service.get_branding()["app_name"]
    xml = blog_export.render_feed_xml(posts, base_url, site_name=site_name)
    return Response(
        content=xml,
        media_type="application/rss+xml",
        headers={"Cache-Control": "public, max-age=300"},
    )


@blog_router.get("/api/blog/syndicated")
async def blog_syndicated_json(db: AsyncSession = Depends(get_async_db)):
    """JSON feed of syndicated posts for external site builders.

    Carries raw markdown (render with your own pipeline), pre-rendered HTML
    (codehilite classes), and an absolute-URL image manifest so a puller can
    download and rehost the images.
    """
    posts = await crud.get_website_published_blog_posts(db)
    base_url = await _syndication_base_url(db)

    def _iso(dt):
        return dt.isoformat() if dt else None

    items = []
    for post in posts:
        image_paths = blog_export.collect_image_paths(post.content)
        items.append({
            "slug": post.slug,
            "title": post.title,
            "description": blog_export.derive_description(
                getattr(post, "excerpt", None), post.content
            ),
            "author": getattr(post, "author_name", None) or None,
            "published_at": _iso(getattr(post, "published_at", None)),
            "syndicated_at": _iso(getattr(post, "website_published_at", None)),
            "updated_at": _iso(getattr(post, "updated_at", None)),
            "content_markdown": post.content,
            "content_html": blog_export.render_markdown(post.content),
            "images": [
                {"path": p, "url": f"{base_url}/blog/images/{p}"}
                for p in image_paths
            ],
        })
    return JSONResponse(
        {"object": "list", "base_url": base_url, "posts": items},
        headers={"Cache-Control": "public, max-age=300"},
    )


# NOTE: registered BEFORE /blog/{slug} — the catch-all slug route would
# otherwise swallow /blog/feed.xml as a slug (404 'Post not found').
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


def _blog_edit_redirect(post_id: int, **params):
    """Redirect to a post's edit page with url-encoded success/error flash params."""
    qs = urlencode({k: v for k, v in params.items() if v})
    suffix = f"?{qs}" if qs else ""
    return RedirectResponse(f"/admin/blog/{post_id}/edit{suffix}", status_code=302)


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
    error: Optional[str] = None,
    db: AsyncSession = Depends(get_async_db),
):
    """Admin: new post form."""
    user, redirect = await _require_admin_read(request, db)
    if redirect:
        return redirect

    masq = await _admin_masquerade_context(request, user, db)
    return templates.TemplateResponse(
        "admin/blog_edit.html",
        {"request": request, "user": user, **masq, "post": None, "active": "blog", "error": error},
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
    try:
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
    except IntegrityError:
        await db.rollback()
        qs = urlencode({"error": f"A post with the slug '{slug}' already exists."})
        return RedirectResponse(f"/admin/blog/new?{qs}", status_code=302)
    except DataError:
        await db.rollback()
        qs = urlencode({"error": "Content is too large to save. Try reducing the post size."})
        return RedirectResponse(f"/admin/blog/new?{qs}", status_code=302)

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

    try:
        await crud.update_blog_post(db, post_id, **kwargs)
        await db.commit()
    except IntegrityError:
        await db.rollback()
        qs = urlencode({"error": f"A post with the slug '{slug}' already exists."})
        return RedirectResponse(f"/admin/blog/{post_id}/edit?{qs}", status_code=302)
    except DataError:
        await db.rollback()
        qs = urlencode({"error": "Content is too large to save. Try reducing the post size."})
        return RedirectResponse(f"/admin/blog/{post_id}/edit?{qs}", status_code=302)

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


@blog_router.post("/admin/blog/{post_id}/website-publish")
async def admin_blog_website_publish(
    request: Request,
    post_id: int,
    db: AsyncSession = Depends(get_async_db),
):
    """Include a post in the public syndication feed.

    Requires the post to be app-published first, so drafts never leak. This
    only flips the selection flag: external sites PULL the selected posts via
    ``GET /blog/feed.xml`` (RSS) or ``GET /api/blog/syndicated`` (JSON) on
    their own schedule — the gateway pushes nothing and holds no credentials
    for any external site.
    """
    user, redirect = await _require_admin(request, db)
    if redirect:
        return redirect

    post = await crud.get_blog_post_by_id(db, post_id)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    if not post.is_published:
        return _blog_edit_redirect(
            post_id, error="Publish the post before adding it to the syndication feed."
        )

    await crud.update_blog_post(
        db, post_id,
        website_published=True,
        website_published_at=datetime.now(timezone.utc),
    )
    await db.commit()
    return _blog_edit_redirect(
        post_id,
        success="Added to the public syndication feed — external sites pick it up on their next sync.",
    )


@blog_router.post("/admin/blog/{post_id}/website-unpublish")
async def admin_blog_website_unpublish(
    request: Request,
    post_id: int,
    db: AsyncSession = Depends(get_async_db),
):
    """Remove a post from the public syndication feed.

    Flag-only, like publish: external sites notice the post is gone from the
    feed on their next sync and remove their copy.
    """
    user, redirect = await _require_admin(request, db)
    if redirect:
        return redirect

    post = await crud.get_blog_post_by_id(db, post_id)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    await crud.update_blog_post(
        db, post_id,
        website_published=False,
        website_published_at=None,
        website_commit_sha=None,
    )
    await db.commit()
    return _blog_edit_redirect(
        post_id,
        success="Removed from the public syndication feed — external sites drop it on their next sync.",
    )


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
    # Images travel inside the message (shrunk, CID-referenced) so the email
    # is self-contained; expandable <details> notes are web-only.
    inline_images = await email_service.load_blog_inline_images(post.content)
    html_body = email_service._render_blog_email(
        post.title, post.content, post.slug,
        user.full_name or user.username, base_url,
        inline_images=inline_images,
    )
    attachments = email_service.blog_inline_attachments(inline_images)

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
            inline_attachments=attachments,
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
    inline_images = await email_service.load_blog_inline_images(post.content)
    html_body = email_service._render_blog_email(
        post.title, post.content, post.slug,
        user.full_name or user.username, base_url,
        inline_images=inline_images,
    )
    attachments = email_service.blog_inline_attachments(inline_images)
    sender = smtp_config.get("blog_sender") or smtp_config.get("default_sender")

    test_user = {"email": test_addr, "username": "testuser", "full_name": "Test User"}
    personalized = email_service._personalize(html_body, test_user)

    try:
        smtp = await email_service._open_smtp(smtp_config)
        try:
            await email_service._send_one(
                smtp, sender, test_addr, subject, personalized, attachments,
            )
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

    headers = {"Cache-Control": "public, max-age=86400"}
    # SVG (and unknown/unmapped types) can carry active content and would execute
    # if opened as a top-level document. Force download + disable MIME sniffing.
    # Inline <img src> embeds in blog posts still display: browsers ignore
    # Content-Disposition for image subresources.
    if content_type in ("image/svg+xml", "application/octet-stream"):
        headers["Content-Disposition"] = "attachment"
        headers["X-Content-Type-Options"] = "nosniff"

    return Response(
        content=data,
        media_type=content_type,
        headers=headers,
    )
