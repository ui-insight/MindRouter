############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# apps_api.py: Registered-application provisioning
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Let a registered first-party app put its SSO users on MindRouter.

A registered app calls this once per user sign-in with two credentials:

  * its OWN key, carrying the `apps:provision` scope — proves the caller is
    that app's server;
  * the user's Entra id_token — proves that specific person just
    authenticated.

Neither alone is enough. Without the token, the app credential would be an
unbounded impersonation primitive: whoever held it could act as any user in
the tenant. Without the app credential, a stolen token would be usable by
anyone who obtained it. Requiring both means a compromised app can only reach
users who are actually signing in to it.

In return the app receives an inference-only key for that user, which it holds
server-side and uses for inference. The user never sees it and never logs in
to MindRouter.

WHAT THIS DELIBERATELY DOES NOT DO: it does not accept an email, username, or
any other identity claim from the request body. Identity comes from the
verified token's `oid` alone. An app asserting who its users are is exactly
the attack this design exists to prevent.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.auth import require_scope
from backend.app.core.client_ip import get_client_ip
from backend.app.db import crud
from backend.app.db.models import ApiKey, User
from backend.app.db.session import get_async_db
from backend.app.logging_config import get_logger
from backend.app.security.api_keys import generate_api_key
from backend.app.security.entra_tokens import EntraTokenError, verify_entra_id_token
from backend.app.security.scopes import APP_USER_KEY_SCOPES, SCOPE_APP_PROVISION, format_scopes

logger = get_logger(__name__)

router = APIRouter(prefix="/api/apps", tags=["apps"])

# Hand back the existing key while it still has this much life left; mint a
# fresh one otherwise. Reusing avoids a new credential per browser tab, while
# the threshold guarantees the app always holds a key that outlives the
# session it is about to serve.
REUSE_IF_REMAINING_FRACTION = 0.5

# Ceiling on live keys per (app, user). Concurrent sessions legitimately need
# more than one; an app looping on the endpoint should not mint without bound.
MAX_LIVE_KEYS_PER_APP_USER = 10

# How often one (app, user) pair may FORCE a new key. Minting runs Argon2, so
# an app that sets force_rotate on every call would turn each sign-in into
# deliberate CPU work. Recovering a lost key cache is a rare event; retrying a
# minute later is not a hardship.
MIN_FORCED_ROTATE_INTERVAL_SECONDS = 60


def _as_utc(value: Optional[datetime]) -> Optional[datetime]:
    """Stamp UTC on a datetime MariaDB handed back without a timezone.

    Values this module constructs are aware; values loaded from the database
    are not. Mixing the two in one field is how a caller ends up parsing a
    timestamp as local time.
    """
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


class AppSessionRequest(BaseModel):
    """Only the token. Identity is never taken from the body."""

    id_token: str = Field(..., min_length=1)

    # Not an identity claim, and it grants nothing the caller was not already
    # entitled to. It exists because an app that loses its server-side key
    # cache — a restart with in-memory storage, a redeploy, a wiped Redis —
    # otherwise cannot recover: a live key is reused, and MindRouter stores
    # only a hash, so the plaintext can never be shown again. Its users would
    # be locked out until that key aged past half its lifetime, which is two
    # weeks at the default TTL.
    force_rotate: bool = False


class AppSessionResponse(BaseModel):
    # None when an existing key was reused: MindRouter stores hashes only, so
    # the plaintext cannot be re-shown and the caller keeps what it holds.
    # Deliberately not an empty string — an app that assigns this straight into
    # its key store should end up with an obviously-absent value rather than a
    # plausible-looking one.
    api_key: Optional[str] = None
    expires_at: datetime
    rotated: bool
    user_id: int
    username: str
    # Lets a caller confirm the key it holds is the one MindRouter considers
    # live, and ask for a rotation if not.
    key_prefix: str


@router.post("/{slug}/sessions", response_model=AppSessionResponse)
async def create_app_session(
    slug: str,
    body: AppSessionRequest,
    request: Request,
    db: AsyncSession = Depends(get_async_db),
    auth: Tuple[User, ApiKey] = Depends(require_scope(SCOPE_APP_PROVISION)),
):
    """Provision (if needed) and return an inference key for one app user."""
    _caller, caller_key = auth

    app = await crud.get_app_by_slug(db, slug)
    if app is None or app.status != "active":
        # Same response whether the app is absent or disabled: a caller
        # holding some other app's credential learns nothing about which apps
        # exist here.
        raise HTTPException(status_code=404, detail="Unknown application")

    # NAMESPACE ENFORCEMENT: the credential must belong to the app it is
    # acting for. This is what makes the scope bounded rather than global —
    # without it, any app's credential could provision for every other app.
    if caller_key.app_id != app.id:
        logger.warning(
            "app_session_wrong_namespace",
            key_id=caller_key.id,
            key_app_id=caller_key.app_id,
            requested_app_id=app.id,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This credential does not belong to that application",
        )

    if not app.entra_client_id or not app.entra_tenant_id:
        logger.error("app_session_unconfigured", app_id=app.id, slug=slug)
        raise HTTPException(
            status_code=500,
            detail="Application is not configured for token verification",
        )

    try:
        identity = await verify_entra_id_token(
            body.id_token,
            expected_client_id=app.entra_client_id,
            expected_tenant_id=app.entra_tenant_id,
        )
    except EntraTokenError as e:
        # Log the reason, return a generic message: the detail describes why
        # verification failed and would help an attacker tune a forgery.
        logger.warning("app_session_token_rejected", app_id=app.id, reason=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="The supplied identity token was not accepted",
        )

    try:
        user, created = await _provision_user(db, identity)
    except IntegrityError:
        # users.username and users.email are both UNIQUE, and provisioning is a
        # read-then-insert with no lock. Prod runs two uvicorn workers, so a
        # user opening the app in two tabs at once can have both processes
        # decide the account does not exist. Whoever loses retries and finds
        # the row the winner committed. An app fires these far more readily
        # than a human clicking a login link.
        await db.rollback()
        logger.info("app_session_provision_raced", app_id=app.id, oid=identity.oid)
        user, created = await _provision_user(db, identity)

    if user is None:
        # find_or_create refuses when the email is already bound to another
        # identity provider — the cross-provider takeover guard.
        logger.warning(
            "app_session_provision_refused", app_id=app.id, oid=identity.oid
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account for this identity could not be provisioned",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Account is inactive"
        )

    api_key_str, key_row, rotated = await _issue_or_reuse_key(
        db, app, user, force_rotate=body.force_rotate
    )

    await crud.log_admin_action(
        db,
        user_id=None,          # the actor is an app, not a person
        action="apps.session",
        entity_type="api_key",
        entity_id=str(key_row.id),
        detail=(
            f"app={app.slug} user_id={user.id} "
            f"{'created_user ' if created else ''}"
            f"{'rotated_key' if rotated else 'reused_key'}"
        ),
        # The point of this row is which app server called, so it has to be the
        # forwarded address: behind nginx, request.client.host is the proxy and
        # is identical for every caller, legitimate or not.
        ip_address=get_client_ip(request),
    )
    await db.commit()

    logger.info(
        "app_session_issued",
        app_id=app.id,
        user_id=user.id,
        key_id=key_row.id,
        created_user=created,
        rotated=rotated,
    )

    return AppSessionResponse(
        api_key=api_key_str,
        # A freshly minted key carries the aware datetime just constructed; a
        # reused one carries what MariaDB returned, which is naive. Serialized
        # raw, the same field would sometimes have a UTC offset and sometimes
        # not, and a client parsing the offset-less form as local time would
        # believe the key lives hours longer than it does.
        expires_at=_as_utc(key_row.expires_at),
        rotated=rotated,
        user_id=user.id,
        username=user.username,
        key_prefix=key_row.key_prefix,
    )


async def _provision_user(db: AsyncSession, identity) -> Tuple[Optional[User], bool]:
    """Find or create the MindRouter account for a verified Entra identity.

    Routed through the Azure driver's own predicate so the account-linking
    rules apply unchanged — in particular that an SSO identity may adopt a
    local account by email ONLY if no provider has claimed it. A second
    provisioning door that reimplemented that rule would be the bypass.

    The profile is shaped like Microsoft Graph's /me because that is what the
    driver consumes, but jobTitle is absent: an id_token does not carry it. A
    newly created user is therefore marked unclassified so their group is
    settled on their first direct MindRouter sign-in.

    "Newly created" is decided by looking the account up under BOTH keys the
    driver can match on. It has two non-creating outcomes, not one: the oid,
    and — for an account no identity provider has claimed — the email. Deciding
    on the oid alone would classify an adopted local account as new and mark it
    for re-grouping, which is precisely what migration 074 promises never
    happens to an existing user. The account that fits that description best is
    the local bootstrap admin, whose group is the deployment's way back in.
    """
    from backend.app.dashboard.azure_auth import find_or_create_azure_user

    if not identity.email:
        # Distinct from the takeover refusal below and much more likely: the
        # `email` claim is optional in Entra and off by default, and without
        # `profile` scope there is no `preferred_username` either. The driver
        # would return None here and the caller would report an account
        # conflict, sending the integrator hunting a duplicate that does not
        # exist.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "The identity token carries no email address. Add the optional "
                "`email` claim, or request the `profile` scope, in the "
                "application's Entra registration."
            ),
        )

    email = identity.email.lower()
    existing = await crud.get_user_by_azure_oid(db, identity.oid)
    if existing is None:
        existing = await crud.get_user_by_email(db, email)

    profile = {
        "id": identity.oid,
        "mail": email,
        "userPrincipalName": email,
        "displayName": identity.display_name,
        # jobTitle intentionally omitted — see docstring.
    }
    # direct=False: an app acting for the person is not the person signing in
    # to MindRouter. The account stays "app-only" for email audiences until
    # they do.
    user = await find_or_create_azure_user(db, profile, direct=False)
    if user is None:
        return None, False

    created = existing is None
    if created:
        user.group_classified = False
        await db.flush()

    return user, created


async def _issue_or_reuse_key(
    db: AsyncSession, app, user: User, force_rotate: bool = False
) -> Tuple[Optional[str], ApiKey, bool]:
    """Return a live key for (app, user), minting one when needed.

    Silent rotation: the app calls this on every sign-in, so a key is replaced
    well before it lapses and a leaked one stops working within the app's TTL.
    An existing key is reused while it retains enough life to outlast the
    session about to start, which keeps concurrent sessions from invalidating
    each other.

    The plaintext key can only be returned when freshly minted — MindRouter
    stores hashes only — so a reused key returns None and the caller keeps
    using what it already has.
    """
    ttl_days = max(1, int(app.key_ttl_days or 30))
    existing = await crud.get_active_app_user_key(db, app.id, user.id)

    if force_rotate and existing is not None:
        # Minting runs Argon2, which is deliberately expensive. The ordinary
        # path pays that only when a key is actually due for replacement, but a
        # caller that sets force_rotate on every request — a restart loop, a
        # misread of the contract — would pay it on every sign-in and could pin
        # CPU. Honour the request only once per interval and tell the caller to
        # come back, rather than silently reusing a key it says it cannot see.
        created = _as_utc(existing.created_at)
        if created is not None:
            age = (datetime.now(timezone.utc) - created).total_seconds()
            if age < MIN_FORCED_ROTATE_INTERVAL_SECONDS:
                logger.warning(
                    "app_session_rotate_throttled",
                    app_id=app.id,
                    user_id=user.id,
                    key_age_seconds=int(age),
                )
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="A key for this user was just issued; retry shortly",
                    headers={
                        "Retry-After": str(
                            int(MIN_FORCED_ROTATE_INTERVAL_SECONDS - age) + 1
                        )
                    },
                )
        # The caller has told us it no longer holds a usable key, so reuse
        # would hand back a credential it cannot see. Mint below.
        existing = None

    if existing is not None and existing.expires_at is not None:
        expires_at = _as_utc(existing.expires_at)
        remaining = (expires_at - datetime.now(timezone.utc)).total_seconds()
        if remaining > ttl_days * 86400 * REUSE_IF_REMAINING_FRACTION:
            return None, existing, False

    live = await crud.count_active_app_user_keys(db, app.id, user.id)
    if live >= MAX_LIVE_KEYS_PER_APP_USER:
        # Retire this user's keys for this app rather than accumulate. Any
        # session still holding one re-provisions on its next call.
        await crud.revoke_app_keys(db, app.id, user_id=user.id)

    full_key, key_hash, key_prefix, key_sha256 = generate_api_key()
    key_row = await crud.create_api_key(
        db,
        user_id=user.id,
        key_hash=key_hash,
        key_prefix=key_prefix,
        name=f"{app.slug} session",
        expires_at=datetime.now(timezone.utc) + timedelta(days=ttl_days),
        key_sha256=key_sha256,
    )
    key_row.app_id = app.id
    # Inference only, whoever owns it — an administrator using a first-party
    # app must not hand that app an admin credential.
    key_row.scopes = format_scopes(APP_USER_KEY_SCOPES)
    key_row.hidden = True
    await db.flush()

    return full_key, key_row, True
