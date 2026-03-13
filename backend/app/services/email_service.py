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
import logging
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

import aiosmtplib
import markdown

from backend.app.db import crud
from backend.app.db.session import get_async_db_context

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HTML email wrapper template (inline CSS for email client compatibility)
# ---------------------------------------------------------------------------

_EMAIL_WRAPPER = """\
<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"></head>
<body style="margin:0;padding:0;background:#f4f4f7;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#f4f4f7;">
<tr><td align="center" style="padding:24px 0;">
<table role="presentation" width="600" cellpadding="0" cellspacing="0" style="background:#ffffff;border-radius:8px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.08);">
  <tr><td style="background:#003DA5;padding:20px 32px;">
    <h1 style="margin:0;color:#ffffff;font-size:20px;font-weight:600;">MindRouter</h1>
  </td></tr>
  <tr><td style="padding:32px;color:#333333;font-size:15px;line-height:1.6;">
    {content}
  </td></tr>
  <tr><td style="padding:16px 32px;background:#f8f9fa;border-top:1px solid #e9ecef;color:#999999;font-size:12px;">
    {footer}
  </td></tr>
</table>
</td></tr>
</table>
</body>
</html>"""

_DEFAULT_FOOTER = (
    "You received this email because you are a registered MindRouter user at the "
    "University of Idaho. To manage your email preferences, visit your "
    '<a href="{base_url}/dashboard" style="color:#003DA5;">dashboard settings</a>.'
)

_BLOG_FOOTER = (
    "You received this email because you are subscribed to MindRouter blog notifications. "
    "To opt out, visit your "
    '<a href="{base_url}/dashboard" style="color:#003DA5;">dashboard settings</a> '
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


def _wrap_html(content_html: str, footer_html: str = "", base_url: str = "") -> str:
    """Wrap content in the email base template."""
    footer = footer_html or _DEFAULT_FOOTER.format(base_url=base_url)
    return _EMAIL_WRAPPER.format(content=content_html, footer=footer)


def _render_blog_email(
    title: str, content_md: str, slug: str, author_name: str, base_url: str
) -> str:
    """Render a blog post as an HTML email body."""
    content_html = markdown.markdown(
        content_md,
        extensions=["fenced_code", "tables"],
    )
    post_url = f"{base_url}/blog/{slug}"
    body = (
        f'<h2 style="margin:0 0 16px 0;color:#003DA5;">{title}</h2>'
        f'{content_html}'
        f'<p style="margin-top:24px;">'
        f'<a href="{post_url}" style="display:inline-block;padding:10px 24px;'
        f'background:#003DA5;color:#ffffff;text-decoration:none;border-radius:4px;'
        f'font-weight:600;">Read on the Web</a></p>'
        f'<p style="color:#999999;font-size:13px;margin-top:16px;">Posted by {author_name}</p>'
    )
    footer = _BLOG_FOOTER.format(base_url=base_url)
    return _wrap_html(body, footer, base_url)


# ---------------------------------------------------------------------------
# Core send functions
# ---------------------------------------------------------------------------


async def _send_one(
    smtp: aiosmtplib.SMTP,
    sender: str,
    recipient: str,
    subject: str,
    html_body: str,
) -> None:
    """Send a single HTML email via an open SMTP connection."""
    msg = MIMEMultipart("alternative")
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = subject
    # Plain text fallback (strip tags crudely)
    import re
    plain = re.sub(r"<[^>]+>", "", html_body)
    plain = re.sub(r"\n{3,}", "\n\n", plain).strip()
    msg.attach(MIMEText(plain, "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))
    await smtp.send_message(msg)


async def _open_smtp(config: Dict[str, Any]) -> aiosmtplib.SMTP:
    """Open and authenticate an SMTP connection."""
    smtp = aiosmtplib.SMTP(
        hostname=config["host"],
        port=int(config["port"]),
        start_tls=bool(config.get("use_tls", True)),
        timeout=30,
    )
    await smtp.connect()
    if config.get("username") and config.get("password"):
        await smtp.login(config["username"], config["password"])
    return smtp


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def send_test_email(config: Dict[str, Any], recipient: str) -> str:
    """Send a test email. Returns empty string on success, error message on failure."""
    if not recipient:
        return "No test address configured"
    try:
        smtp = await _open_smtp(config)
        try:
            body = _wrap_html(
                "<p>This is a test email from <strong>MindRouter</strong>.</p>"
                "<p>If you can read this, your SMTP configuration is working correctly.</p>",
                _DEFAULT_FOOTER.format(base_url=""),
            )
            await _send_one(smtp, config["default_sender"], recipient, "MindRouter Test Email", body)
        finally:
            await smtp.quit()
        return ""
    except Exception as e:
        return str(e)


async def send_bulk_email(
    email_log_id: int,
    subject: str,
    body_html: str,
    recipients: List[Dict[str, str]],
    sender: str,
    config: Dict[str, Any],
) -> None:
    """Send personalized emails to a list of recipients (fire-and-forget background task).

    recipients: list of dicts with keys: email, username, full_name
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
                    await _send_one(smtp, sender, user["email"], personalized_subject, personalized)
                    success += 1
                except Exception as e:
                    errors.append(f"{user['email']}: {e}")
                    logger.warning("email_send_failed", email=user["email"], error=str(e))
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
        logger.error("email_bulk_send_error", error=str(e))
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
    subject = f"MindRouter Blog: {post_title}"
    await send_bulk_email(email_log_id, subject, html_body, recipients, sender, config)
