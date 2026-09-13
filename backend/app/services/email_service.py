############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# email_service.py: Async email sending service
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Async email sending service for MindRouter."""

import asyncio
import html as _html
import logging
import re
from datetime import datetime, timezone
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional, Tuple

import aiosmtplib
import markdown

from backend.app.db import crud
from backend.app.db.session import get_async_db_context
from backend.app.services import branding as _branding

from backend.app.settings import get_settings

logger = logging.getLogger(__name__)


async def get_base_url(db=None) -> str:
    """Return the configured site URL from AppConfig, falling back to settings."""
    async def _load(db):
        return await crud.get_config_json(db, "app.base_url", get_settings().app_base_url)
    if db:
        return await _load(db)
    async with get_async_db_context() as db:
        return await _load(db)

# ---------------------------------------------------------------------------
# HTML email wrapper template (inline CSS for email client compatibility)
# ---------------------------------------------------------------------------

# Opening and closing of the email shell. The header row and content/footer are
# assembled in _wrap_html() so branding (logo, accent color) can be injected and
# so user content is never passed through str.format() (it may contain braces).
_EMAIL_HEAD = """\
<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"></head>
<body style="margin:0;padding:0;background:#f4f4f7;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#f4f4f7;">
<tr><td align="center" style="padding:24px 0;">
<table role="presentation" width="600" cellpadding="0" cellspacing="0" style="background:#ffffff;border-radius:8px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.08);">
"""

_EMAIL_FOOT = """\
</table>
</td></tr>
</table>
</body>
</html>"""

# Email headings use a neutral near-black, NOT the brand accent: a light accent
# (e.g. University of Idaho Pride Gold) reads poorly as headline text and its
# muted shades are brand-reserved. Buttons/links keep the accent.
_EMAIL_HEADING_COLOR = "#231f20"

# Footer templates use {base_url} and {link_color} placeholders, filled in
# _wrap_html (link_color = the brand's accessible accent "ink").
_DEFAULT_FOOTER = (
    "You received this email because you are a registered {app_name} user. "
    "To manage your email preferences, visit your "
    '<a href="{base_url}/dashboard" style="color:{link_color};">dashboard settings</a>.'
)

_BLOG_FOOTER = (
    "You received this email because you are subscribed to {app_name} blog notifications. "
    "To opt out, visit your "
    '<a href="{base_url}/dashboard" style="color:{link_color};">dashboard settings</a> '
    "and toggle the email preference."
)


# ---------------------------------------------------------------------------
# SMTP configuration helpers
# ---------------------------------------------------------------------------


async def get_smtp_config(db=None) -> Dict[str, Any]:
    """Load SMTP configuration from AppConfig."""
    async def _load(db):
        return {
            "host": await crud.get_config_json(db, "email.smtp_host", ""),
            "port": await crud.get_config_json(db, "email.smtp_port", 587),
            "username": await crud.get_config_json(db, "email.smtp_username", ""),
            "password": await crud.get_config_json(db, "email.smtp_password", ""),
            "use_tls": await crud.get_config_json(db, "email.use_tls", True),
            "default_sender": await crud.get_config_json(db, "email.default_sender", ""),
            "test_address": await crud.get_config_json(db, "email.test_address", ""),
            "blog_sender": await crud.get_config_json(db, "email.blog_sender", ""),
        }

    if db:
        return await _load(db)
    async with get_async_db_context() as db:
        return await _load(db)


def is_smtp_configured(config: Dict[str, Any]) -> bool:
    """Check if SMTP is minimally configured."""
    return bool(config.get("host") and config.get("default_sender"))


# ---------------------------------------------------------------------------
# Template personalization
# ---------------------------------------------------------------------------


def _personalize(text: str, user: Dict[str, str]) -> str:
    """Replace template variables in text."""
    full_name = user.get("full_name", "") or ""
    parts = full_name.split() if full_name else []
    first_name = parts[0] if parts else user.get("username", "")
    last_name = parts[-1] if len(parts) > 1 else ""

    return (
        text
        .replace("{{first_name}}", first_name)
        .replace("{{last_name}}", last_name)
        .replace("{{username}}", user.get("username", ""))
        .replace("{{email}}", user.get("email", ""))
    )


def _style_code_blocks(html: str) -> str:
    """Add inline CSS to code/pre blocks for email client compatibility."""
    import re

    _mono = "'SFMono-Regular',Consolas,'Liberation Mono',Menlo,monospace"

    # 1) Style <pre> blocks — bordered container, wrap text, small mono font
    html = re.sub(
        r'<pre(?![^>]*style=)>',
        f'<pre style="background:#f6f8fa;border:1px solid #333333;'
        f'padding:12px 16px;border-radius:6px;'
        f'overflow-x:auto;white-space:pre-wrap;word-wrap:break-word;'
        f'font-family:{_mono};font-size:13px;line-height:1.45;">',
        html,
    )

    # 2) Style <code> inside <pre> — transparent background, inherit font
    html = re.sub(
        r'(<pre[^>]*>)\s*<code([^>]*)>',
        rf'\1<code\2 style="font-family:inherit;font-size:inherit;'
        rf'background:transparent;padding:0;white-space:inherit;word-wrap:inherit;">',
        html,
    )

    # 3) Style remaining inline <code> tags (those not already styled by step 2)
    html = re.sub(
        r'<code(?![^>]*style=)([^>]*)>',
        f'<code\\1 style="background:#f6f8fa;padding:2px 6px;border-radius:3px;'
        f'font-family:{_mono};font-size:13px;">',
        html,
    )

    return html


def _style_tables(html: str) -> str:
    """Add inline CSS with visible black borders to tables for email.

    Email clients strip <style> blocks, so borders must be inline on
    each element.  cellspacing="0" + border-collapse keeps Outlook from
    inserting gaps between the (border-collapsed) cells.
    """
    import re

    table_style = (
        "border-collapse:collapse;border:2px solid #000000;"
        "width:100%;margin:16px 0;font-size:14px;"
    )
    cell = "border:1px solid #000000;padding:8px 10px;"
    th_style = cell + "background:#f0f0f0;text-align:left;"

    # <table> is always emitted bare by python-markdown.
    html = re.sub(
        r"<table(?![^>]*style=)([^>]*)>",
        rf'<table\1 cellspacing="0" cellpadding="0" style="{table_style}">',
        html,
    )

    # <th>/<td> may already carry style="text-align:..." from column
    # alignment — merge onto the existing style, else add a new one.
    # The (?=[\s>]) guard anchors on the whole tag name so <th> does not
    # also match <thead>.
    for tag, style in (("th", th_style), ("td", cell)):
        html = re.sub(
            rf'<{tag}(?=[\s>])([^>]*) style="([^"]*)"',
            rf'<{tag}\1 style="\2;{style}"',
            html,
        )
        html = re.sub(
            rf"<{tag}(?=[\s>])(?![^>]*style=)([^>]*)>",
            rf'<{tag}\1 style="{style}">',
            html,
        )

    return html


def _wrap_html(content_html: str, footer_html: str = "", base_url: str = "") -> str:
    """Wrap content in the branded email base template.

    Branding (organization name, logo, accent color) is pulled from the shared
    branding config so emails match the web UI. The header shows the email logo
    (referenced as ``cid:brandlogo`` and embedded by ``_send_one`` — email
    clients don't render SVG, so a raster ``email_logo`` is used) when set,
    otherwise the organization name. Accent colors are contrast-adjusted: the
    footer/title link color uses the accessible "ink" so a light brand accent
    (e.g. Pride Gold) stays legible on white.
    """
    content_html = _style_code_blocks(content_html)
    content_html = _style_tables(content_html)

    brand = _branding.get_branding()
    accent = brand["primary_light"]
    accent_ink = brand["primary_light_ink"]
    app_name = brand["app_name"]

    name_esc = _html.escape(app_name)
    logo_alt = _html.escape(brand.get("tagline") or app_name)
    name_html = f'<span style="font-size:26px;font-weight:700;color:#231f20;">{name_esc}</span>'
    if brand.get("email_logo_file"):
        # Logo left, org/app name right-justified on the same row (email-safe
        # two-cell table — float/flex are unreliable in mail clients).
        header_inner = (
            '<table role="presentation" width="100%" cellpadding="0" cellspacing="0"><tr>'
            '<td align="left" valign="middle">'
            f'<img src="cid:brandlogo" alt="{logo_alt}" height="44" '
            'style="display:block;height:44px;width:auto;border:0;"></td>'
            f'<td align="right" valign="middle">{name_html}</td>'
            '</tr></table>'
        )
    else:
        header_inner = name_html

    footer_tmpl = footer_html or _DEFAULT_FOOTER
    footer = footer_tmpl.format(base_url=base_url, link_color=accent_ink, app_name=_html.escape(app_name))

    header_row = (
        f'  <tr><td style="background:#ffffff;padding:20px 32px;'
        f'border-bottom:3px solid {accent};">{header_inner}</td></tr>\n'
    )
    content_row = (
        '  <tr><td style="padding:32px;color:#333333;font-size:15px;line-height:1.6;">\n'
        f'{content_html}\n  </td></tr>\n'
    )
    footer_row = (
        '  <tr><td style="padding:16px 32px;background:#f8f9fa;border-top:1px solid #e9ecef;'
        f'color:#999999;font-size:12px;">{footer}</td></tr>\n'
    )
    return _EMAIL_HEAD + header_row + content_row + footer_row + _EMAIL_FOOT


# Blog images are embedded in the email as inline (CID) attachments so the
# message is self-contained — no remote fetch, so it renders even where the
# client blocks external images. They are shrunk first: email bodies are
# ~600px wide and a post can carry dozens of retina screenshots.
_EMAIL_IMAGE_MAX_WIDTH = 640
_EMAIL_IMAGE_JPEG_QUALITY = 78
# Total budget for inline images per message; past it the remaining images
# fall back to remote links so a large post never produces a 20 MB email.
_EMAIL_IMAGE_BUDGET_BYTES = 6 * 1024 * 1024

_DETAILS_RE = re.compile(r"<details\b.*?</details>", re.IGNORECASE | re.DOTALL)
_DETAILS_OMITTED_NOTE = (
    '<p style="color:#999999;font-size:13px;">'
    "(Expandable troubleshooting notes are omitted from the email version "
    "— see the post on the web.)</p>"
)
_BLOG_IMAGE_SRC_RE = re.compile(r'src="(?:https?://[^"/]+)?/blog/images/([^"]+)"')


def _strip_details(content_md: str) -> str:
    """Drop ``<details>`` blocks (expandable, web-only notes) from email copy.

    Mail clients cannot collapse them, so a long post's gotchas would be
    dumped inline. Each block is replaced by a one-line pointer to the web.
    """
    return _DETAILS_RE.sub(_DETAILS_OMITTED_NOTE, content_md)


def shrink_image_for_email(data: bytes, max_width: int = _EMAIL_IMAGE_MAX_WIDTH,
                           quality: int = _EMAIL_IMAGE_JPEG_QUALITY) -> Tuple[bytes, str]:
    """Downscale an image for inline email use. Returns ``(bytes, subtype)``.

    Photos and flat screenshots both compress well as JPEG; images with
    transparency (logos, diagrams on transparent backgrounds) stay PNG so
    the background is not painted black. Anything Pillow cannot read is
    returned untouched with a best-effort subtype.
    """
    from io import BytesIO
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - Pillow is a hard dependency
        return data, "png"
    try:
        im = Image.open(BytesIO(data))
        im.load()
    except Exception:
        return data, "png"
    fmt = (im.format or "png").lower()
    if fmt == "gif":
        return data, "gif"  # keep animations intact
    if im.width > max_width:
        im = im.resize((max_width, max(1, round(im.height * max_width / im.width))), Image.LANCZOS)
    has_alpha = im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info)
    out = BytesIO()
    if has_alpha:
        im.convert("RGBA").save(out, "PNG", optimize=True)
        return out.getvalue(), "png"
    im.convert("RGB").save(out, "JPEG", quality=quality, optimize=True, progressive=True)
    return out.getvalue(), "jpeg"


async def load_blog_inline_images(content_md: str) -> Dict[str, Tuple[bytes, str]]:
    """Fetch every ``/blog/images/…`` the post references from artifact
    storage and shrink it for email.

    Returns ``{storage_path: (bytes, subtype)}`` in document order, capped by
    ``_EMAIL_IMAGE_BUDGET_BYTES`` — images past the budget are left out so
    ``_render_blog_email`` keeps them as remote links instead.
    """
    # Lazy import: the storage module pulls in settings, which the
    # email-render unit tests stub out.
    from backend.app.storage.artifacts import get_artifact_storage

    paths: List[str] = []
    for m in _BLOG_IMAGE_SRC_RE.finditer(content_md):
        if m.group(1) not in paths:
            paths.append(m.group(1))
    for m in re.finditer(r'!\[[^\]]*\]\(/blog/images/([^)\s]+)\)', content_md):
        if m.group(1) not in paths:
            paths.append(m.group(1))
    if not paths:
        return {}

    storage = get_artifact_storage()
    images: Dict[str, Tuple[bytes, str]] = {}
    total = 0
    for path in paths:
        data = await storage.retrieve(path)
        if data is None:
            continue
        shrunk = shrink_image_for_email(data)
        if total + len(shrunk[0]) > _EMAIL_IMAGE_BUDGET_BYTES:
            logger.warning(
                "blog_email_image_budget_exceeded: %d images embedded, rest left as links", len(images)
            )
            break
        images[path] = shrunk
        total += len(shrunk[0])
    return images


def _render_blog_email(
    title: str, content_md: str, slug: str, author_name: str, base_url: str,
    inline_images: Optional[Dict[str, Tuple[bytes, str]]] = None,
) -> str:
    """Render a blog post as an HTML email body.

    ``inline_images`` (from :func:`load_blog_inline_images`) maps storage
    paths to shrunken image bytes; each referenced image becomes
    ``src="cid:blogimgN"`` and is attached by :func:`_send_one` via
    :func:`blog_inline_attachments`. Images not in the map keep absolute URLs.
    """
    # Strip the [TOC] table-of-contents marker: the email renderer has no
    # 'toc' extension (anchor navigation is unreliable in mail clients),
    # so the literal token would otherwise appear as text.
    content_md = re.sub(r"\[TOC\]", "", content_md, flags=re.IGNORECASE)
    content_md = _strip_details(content_md)

    # Convert relative image URLs to absolute so email clients can fetch them
    content_md = re.sub(
        r'src="(/blog/images/)',
        f'src="{base_url}/blog/images/',
        content_md,
    )
    content_md = re.sub(
        r'!\[([^\]]*)\]\((/blog/images/[^)]+)\)',
        rf'![\1]({base_url}\2)',
        content_md,
    )
    content_html = markdown.markdown(
        content_md,
        extensions=["fenced_code", "tables"],
    )
    if inline_images:
        cids = _blog_image_cids(inline_images)

        def _to_cid(m: "re.Match[str]") -> str:
            path = m.group(1)
            return f'src="cid:{cids[path]}"' if path in cids else m.group(0)

        content_html = _BLOG_IMAGE_SRC_RE.sub(_to_cid, content_html)
    # Email clients ignore stylesheets: size every image inline so a wide
    # screenshot cannot blow out the 600px layout.
    content_html = re.sub(
        r"<img\b(?![^>]*\bstyle=)",
        '<img style="max-width:100%;height:auto;"',
        content_html,
    )
    post_url = f"{base_url}/blog/{slug}"
    brand = _branding.get_branding()
    accent = brand["primary_light"]          # fill (button background)
    accent_on = brand["primary_light_on"]    # legible text on the accent fill
    headline = brand.get("headline_color") or _EMAIL_HEADING_COLOR  # admin-configurable; neutral by default
    body = (
        f'<h2 style="margin:0 0 16px 0;color:{headline};">{title}</h2>'
        f'{content_html}'
        f'<p style="margin-top:24px;">'
        f'<a href="{post_url}" style="display:inline-block;padding:10px 24px;'
        f'background:{accent};color:{accent_on};text-decoration:none;border-radius:4px;'
        f'font-weight:600;">Read on the Web</a></p>'
        f'<p style="color:#999999;font-size:13px;margin-top:16px;">Posted by {author_name}</p>'
    )
    # Pass the raw footer template — _wrap_html fills base_url/link_color/app_name.
    return _wrap_html(body, _BLOG_FOOTER, base_url)


def _blog_image_cids(inline_images: Dict[str, Tuple[bytes, str]]) -> Dict[str, str]:
    """Stable ``storage_path -> Content-ID`` assignment (document order)."""
    return {path: f"blogimg{i}" for i, path in enumerate(inline_images, start=1)}


def blog_inline_attachments(
    inline_images: Optional[Dict[str, Tuple[bytes, str]]],
) -> Dict[str, Tuple[bytes, str]]:
    """``{cid: (bytes, subtype)}`` for :func:`_send_one`, matching the ids
    :func:`_render_blog_email` wrote into the HTML."""
    if not inline_images:
        return {}
    cids = _blog_image_cids(inline_images)
    return {cids[path]: img for path, img in inline_images.items()}


# ---------------------------------------------------------------------------
# Core send functions
# ---------------------------------------------------------------------------


async def _send_one(
    smtp: aiosmtplib.SMTP,
    sender: str,
    recipient: str,
    subject: str,
    html_body: str,
    inline_attachments: Optional[Dict[str, Tuple[bytes, str]]] = None,
) -> None:
    """Send a single HTML email via an open SMTP connection.

    When the body references the branding email logo (``cid:brandlogo``), the
    logo is embedded as an inline (CID) attachment so it renders even when the
    client blocks remote images — the structure becomes ``multipart/related``
    wrapping the ``multipart/alternative`` text+html parts. ``inline_attachments``
    (``{cid: (bytes, subtype)}``, e.g. a blog post's images) are embedded the
    same way.
    """
    # Plain text fallback (strip tags crudely)
    plain = re.sub(r"<[^>]+>", "", html_body)
    plain = re.sub(r"\n{3,}", "\n\n", plain).strip()

    inline: Dict[str, Tuple[bytes, str]] = dict(inline_attachments or {})
    logo = _branding.read_email_logo() if "cid:brandlogo" in html_body else None
    if logo:
        inline["brandlogo"] = logo
    if inline:
        msg = MIMEMultipart("related")
        alt = MIMEMultipart("alternative")
        alt.attach(MIMEText(plain, "plain", "utf-8"))
        alt.attach(MIMEText(html_body, "html", "utf-8"))
        msg.attach(alt)
        for cid, (data, subtype) in inline.items():
            img = MIMEImage(data, _subtype=subtype)
            img.add_header("Content-ID", f"<{cid}>")
            img.add_header("Content-Disposition", "inline", filename=f"{cid}.{subtype}")
            msg.attach(img)
    else:
        msg = MIMEMultipart("alternative")
        msg.attach(MIMEText(plain, "plain", "utf-8"))
        msg.attach(MIMEText(html_body, "html", "utf-8"))

    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = subject
    await smtp.send_message(msg)


async def _open_smtp(config: Dict[str, Any]) -> aiosmtplib.SMTP:
    """Open and authenticate an SMTP connection."""
    import socket
    local_fqdn = config.get("helo_hostname") or socket.getfqdn() or "mindrouter.uidaho.edu"
    # If running in Docker the FQDN is the container ID — fall back to a real hostname
    if "." not in local_fqdn:
        local_fqdn = "mindrouter.uidaho.edu"
    smtp = aiosmtplib.SMTP(
        hostname=config["host"],
        port=int(config["port"]),
        start_tls=bool(config.get("use_tls", True)),
        timeout=30,
        local_hostname=local_fqdn,
    )
    await smtp.connect()
    if config.get("username") and config.get("password"):
        await smtp.login(config["username"], config["password"])
    return smtp


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def send_test_email(config: Dict[str, Any], recipient: str, base_url: str = "") -> str:
    """Send a test email. Returns empty string on success, error message on failure."""
    if not recipient:
        return "No test address configured"
    try:
        smtp = await _open_smtp(config)
        try:
            app_name = _html.escape(_branding.get_branding()["app_name"])
            body = _wrap_html(
                f"<p>This is a test email from <strong>{app_name}</strong>.</p>"
                "<p>If you can read this, your SMTP configuration is working correctly.</p>",
                base_url=base_url,
            )
            await _send_one(smtp, config["default_sender"], recipient, f"{app_name} Test Email", body)
        finally:
            await smtp.quit()
        return ""
    except Exception as e:
        return str(e)


async def send_notification_email(
    config: Dict[str, Any],
    recipients: List[str],
    subject: str,
    body_html: str,
    base_url: str = "",
) -> int:
    """Send one operational notification to a list of plain addresses.

    For system-generated alerts (no per-user personalization, no EmailLog row) —
    unlike ``send_bulk_email``, which drives admin-composed campaigns.  Opens a
    single SMTP connection for the whole batch.  Returns the number of messages
    successfully sent; never raises.
    """
    if not recipients:
        return 0
    if not is_smtp_configured(config):
        logger.warning("notification email skipped: SMTP not configured")
        return 0

    # A recipient carrying CR/LF would inject extra SMTP headers.
    clean = [r.strip() for r in recipients if r and "\n" not in r and "\r" not in r]
    if not clean:
        return 0

    sent = 0
    try:
        smtp = await _open_smtp(config)
        try:
            body = _wrap_html(body_html, base_url=base_url)
            for recipient in clean:
                try:
                    await _send_one(smtp, config["default_sender"], recipient, subject, body)
                    sent += 1
                except Exception as e:
                    logger.warning("notification email failed for %s: %s", recipient, e)
                await asyncio.sleep(0.05)  # throttle
        finally:
            try:
                await smtp.quit()
            except Exception:
                pass
    except Exception as e:
        logger.error("notification email send error: %s", e)
    return sent


async def send_bulk_email(
    email_log_id: int,
    subject: str,
    body_html: str,
    recipients: List[Dict[str, str]],
    sender: str,
    config: Dict[str, Any],
    inline_attachments: Optional[Dict[str, Tuple[bytes, str]]] = None,
) -> None:
    """Send personalized emails to a list of recipients (fire-and-forget background task).

    recipients: list of dicts with keys: email, username, full_name
    inline_attachments: ``{cid: (bytes, subtype)}`` embedded in every message
    (see :func:`blog_inline_attachments`).
    """
    errors = []
    success = 0

    try:
        async with get_async_db_context() as db:
            await crud.update_email_log(db, email_log_id, status="sending")
            await db.commit()

        smtp = await _open_smtp(config)
        try:
            for user in recipients:
                try:
                    personalized = _personalize(body_html, user)
                    personalized_subject = _personalize(subject, user)
                    await _send_one(
                        smtp, sender, user["email"], personalized_subject, personalized,
                        inline_attachments,
                    )
                    success += 1
                except Exception as e:
                    errors.append(f"{user['email']}: {e}")
                    logger.warning("email_send_failed: %s — %s", user["email"], e)
                await asyncio.sleep(0.05)  # throttle
        finally:
            try:
                await smtp.quit()
            except Exception:
                pass

        async with get_async_db_context() as db:
            await crud.update_email_log(
                db, email_log_id,
                status="completed",
                success_count=success,
                fail_count=len(errors),
                error_message="\n".join(errors) if errors else None,
                completed_at=datetime.now(timezone.utc),
            )
            await db.commit()

    except Exception as e:
        logger.error("email_bulk_send_error: %s", e)
        try:
            async with get_async_db_context() as db:
                await crud.update_email_log(
                    db, email_log_id,
                    status="failed",
                    success_count=success,
                    fail_count=len(errors),
                    error_message=str(e),
                    completed_at=datetime.now(timezone.utc),
                )
                await db.commit()
        except Exception:
            pass


async def send_blog_email(
    email_log_id: int,
    post_title: str,
    post_content: str,
    post_slug: str,
    author_name: str,
    recipients: List[Dict[str, str]],
    sender: str,
    config: Dict[str, Any],
    base_url: str,
) -> None:
    """Send a blog post as email to opted-in recipients (fire-and-forget)."""
    html_body = _render_blog_email(post_title, post_content, post_slug, author_name, base_url)
    subject = f"{_branding.get_branding()['app_name']} Blog: {post_title}"
    await send_bulk_email(email_log_id, subject, html_body, recipients, sender, config)
