############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# sso/base.py: shared SSO profile, CSRF state, JIT provisioning
#
############################################################

"""Provider-agnostic SSO plumbing.

Every provider driver reduces its IdP response to an :class:`SSOProfile`,
then calls :func:`find_or_create_sso_user`. Identity is keyed on the
``(sso_provider, sso_subject)`` pair (``users`` table); a pre-existing account
matched by email is adopted only when NO identity provider has claimed it yet.
Both this path and the Azure driver refuse an email match against an account
that already carries an ``azure_oid`` or an ``sso_provider``.
"""

import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from itsdangerous import URLSafeTimedSerializer
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db import crud
from backend.app.logging_config import get_logger
from backend.app.settings import get_settings

logger = get_logger(__name__)

STATE_MAX_AGE = 600  # seconds; CSRF state lifetime (matches Azure driver)


@dataclass
class SSOProfile:
    """Normalized identity returned by a provider driver."""

    provider: str                     # "google" | "oidc" | "saml"
    subject: str                      # stable unique id at the IdP (sub / NameID)
    email: str
    display_name: Optional[str] = None
    username_hint: Optional[str] = None  # e.g. eduPersonPrincipalName
    department: Optional[str] = None
    college: Optional[str] = None
    raw: dict = field(default_factory=dict)


def state_serializer() -> URLSafeTimedSerializer:
    """Timed serializer for CSRF state cookies (same signer as Azure driver)."""
    return URLSafeTimedSerializer(get_settings().secret_key)


def new_signed_state() -> str:
    """Generate a signed random CSRF state token."""
    return state_serializer().dumps(secrets.token_urlsafe(32))


def validate_state(query_state: Optional[str], cookie_state: Optional[str]) -> bool:
    """Validate the round-tripped CSRF state against the cookie + signature age."""
    if not query_state or not cookie_state or query_state != cookie_state:
        return False
    try:
        state_serializer().loads(query_state, max_age=STATE_MAX_AGE)
        return True
    except Exception:
        return False


class SSOConfigError(Exception):
    """SSO cannot proceed because required deployment configuration is absent.

    Raised by :func:`public_base_url` when ``app_base_url`` is unset. Provider
    drivers catch it and return a login redirect with an error banner rather
    than letting it surface as a bare 500 at the IdP callback.
    """


def public_base_url(request=None) -> str:
    """Return this deployment's public base URL (``scheme://host``, no trailing
    slash), FAILING CLOSED when it is not configured.

    The IdP ``redirect_uri``, the SAML Destination/Recipient/ACS URL, and SP
    metadata must all match one externally-visible origin exactly. Deriving that
    origin from request headers (``Host`` / ``X-Forwarded-*``) is attacker-
    controllable — a spoofed ``Host`` would mint a redirect pointing at a host
    the attacker chose — so this reads only the operator-set ``app_base_url``
    and refuses the login when it is empty rather than trusting the request.

    ``request`` is accepted for call-site symmetry but is deliberately NOT
    consulted: there is no safe header-derived fallback here.
    """
    base = (get_settings().app_base_url or "").rstrip("/")
    if base:
        return base
    logger.error(
        "sso_app_base_url_missing",
        note="set APP_BASE_URL to the public https origin; SSO redirect_uri / ACS / metadata must match it exactly",
    )
    raise SSOConfigError("app_base_url is not configured")


async def find_or_create_sso_user(
    db: AsyncSession,
    profile: SSOProfile,
    default_group_name: str,
):
    """JIT-provision (or link) a user from a normalized SSO profile.

    Lookup order mirrors the Azure driver: (provider, subject) first, then
    email (links pre-existing local accounts without clearing their password).
    Returns the user, or None when the profile is unusable.
    """
    if not profile.subject or not profile.email:
        return None

    email = profile.email.lower()

    user = await crud.get_user_by_sso_subject(db, profile.provider, profile.subject)
    if not user:
        candidate = await crud.get_user_by_email(db, email)
        if candidate is not None:
            # Only adopt an account no IdP has claimed yet. Without this, a
            # second enabled provider could assert someone else's address and
            # inherit their account — including Azure-provisioned admins.
            # Email is an IdP-supplied attribute, not proof of ownership.
            if candidate.azure_oid or candidate.sso_provider:
                logger.warning(
                    "sso_email_link_refused",
                    provider=profile.provider,
                    email=email,
                    reason="account already bound to another identity provider",
                )
                return None
            user = candidate

    if user:
        if not user.sso_provider:
            user.sso_provider = profile.provider
            user.sso_subject = profile.subject
        if profile.display_name:
            user.full_name = profile.display_name
        if profile.department:
            user.department = profile.department
        if profile.college:
            user.college = profile.college
        user.last_login_at = datetime.now(timezone.utc)
        user.last_direct_login_at = user.last_login_at   # an SSO web sign-in is a direct one
        await db.flush()
        return user

    group = await crud.get_group_by_name(db, default_group_name)
    if group is None:
        # users.group_id is NOT NULL (migration 009), so provisioning with
        # no group would raise IntegrityError and surface as a bare 500 at
        # the IdP callback.  Refuse cleanly instead — finish_login turns
        # None into a redirect with an error banner.
        logger.error(
            "sso_default_group_missing",
            provider=profile.provider,
            group=default_group_name,
            note="create the group or fix the provider's default-group setting",
        )
        return None

    username = (profile.username_hint or email).split("@")[0]
    if await crud.get_user_by_username(db, username):
        username = f"{username}_{profile.subject[:8]}"

    user = await crud.create_user(
        db=db,
        username=username,
        email=email,
        password_hash=None,
        full_name=profile.display_name,
        group_id=group.id,
        department=profile.department,
        college=profile.college,
    )
    user.sso_provider = profile.provider
    user.sso_subject = profile.subject
    user.last_login_at = datetime.now(timezone.utc)
    user.last_direct_login_at = user.last_login_at
    await db.flush()

    await crud.create_quota(db=db, user_id=user.id, rpm_limit=group.rpm_limit)

    logger.info(
        "sso_user_provisioned",
        provider=profile.provider,
        user_id=user.id,
        username=user.username,
        group=group.name,
    )
    return user


async def finish_login(db: AsyncSession, user, clear_cookie: Optional[str] = None):
    """Common tail of every SSO callback: commit, set session, redirect."""
    from fastapi.responses import RedirectResponse

    from backend.app.dashboard.routes import _needs_agreement, set_session_cookie

    if not user:
        return RedirectResponse(url="/login?error=Failed+to+provision+user+account", status_code=302)
    if not user.is_active:
        return RedirectResponse(url="/login?error=Account+is+inactive", status_code=302)

    await db.commit()
    target = "/dashboard/agreement" if await _needs_agreement(db, user) else "/dashboard"
    response = RedirectResponse(url=target, status_code=302)
    set_session_cookie(response, user.id)
    if clear_cookie:
        response.delete_cookie(key=clear_cookie)
    return response
