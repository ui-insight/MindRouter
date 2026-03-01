############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# azure_auth.py: Azure AD OAuth2 SSO routes
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Azure AD OAuth2 SSO authentication for MindRouter2."""

import json
import logging
import secrets
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

import httpx
from fastapi import APIRouter, Depends, Request
from fastapi.responses import RedirectResponse
from itsdangerous import URLSafeTimedSerializer
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db import crud
from backend.app.db.models import UserRole
from backend.app.db.session import get_async_db
from backend.app.settings import get_settings

azure_router = APIRouter(tags=["azure-auth"])

# Azure AD endpoints
AZURE_AUTHORIZE_URL = "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/authorize"
AZURE_TOKEN_URL = "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"
GRAPH_ME_URL = "https://graph.microsoft.com/v1.0/me"

# Scopes for Azure AD
AZURE_SCOPES = "openid profile email User.Read"


def _get_serializer() -> URLSafeTimedSerializer:
    """Get a timed serializer for CSRF state and session cookies."""
    settings = get_settings()
    return URLSafeTimedSerializer(settings.secret_key)


def _map_job_title_to_group(job_title: Optional[str]) -> str:
    """Map Azure AD jobTitle to MindRouter2 group name."""
    settings = get_settings()
    if not job_title:
        return settings.azure_ad_default_group

    title_lower = job_title.lower()
    if "student" in title_lower:
        return "students"
    elif "faculty" in title_lower or "professor" in title_lower:
        return "faculty"
    elif "staff" in title_lower:
        return "staff"
    return settings.azure_ad_default_group


def _map_group_to_role(group_name: str) -> UserRole:
    """Map group name to UserRole enum."""
    mapping = {
        "students": UserRole.STUDENT,
        "faculty": UserRole.FACULTY,
        "staff": UserRole.STAFF,
        "admin": UserRole.ADMIN,
    }
    return mapping.get(group_name, UserRole.STUDENT)


@azure_router.get("/login/azure")
async def azure_login(request: Request):
    """Redirect to Azure AD authorization endpoint."""
    settings = get_settings()
    if not settings.azure_ad_enabled:
        return RedirectResponse(url="/login?error=Azure+AD+SSO+is+not+configured", status_code=302)

    # Generate CSRF state token
    state = secrets.token_urlsafe(32)
    serializer = _get_serializer()
    signed_state = serializer.dumps(state)

    params = {
        "client_id": settings.azure_ad_client_id,
        "response_type": "code",
        "redirect_uri": settings.azure_ad_redirect_uri,
        "response_mode": "query",
        "scope": AZURE_SCOPES,
        "state": signed_state,
    }

    authorize_url = AZURE_AUTHORIZE_URL.format(tenant=settings.azure_ad_tenant_id)
    redirect_url = f"{authorize_url}?{urlencode(params)}"

    response = RedirectResponse(url=redirect_url, status_code=302)
    # Store state in cookie for validation on callback
    response.set_cookie(
        key="azure_oauth_state",
        value=signed_state,
        httponly=True,
        samesite="lax",
        max_age=600,  # 10 minutes
    )
    return response


@azure_router.get("/login/azure/authorized")
async def azure_callback(
    request: Request,
    code: Optional[str] = None,
    state: Optional[str] = None,
    error: Optional[str] = None,
    error_description: Optional[str] = None,
    db: AsyncSession = Depends(get_async_db),
):
    """Handle Azure AD OAuth2 callback."""
    settings = get_settings()

    # Check for Azure AD error response
    if error:
        return RedirectResponse(
            url=f"/login?error=Azure+login+failed:+{error_description or error}",
            status_code=302,
        )

    if not code or not state:
        return RedirectResponse(url="/login?error=Invalid+callback+parameters", status_code=302)

    # Validate CSRF state
    cookie_state = request.cookies.get("azure_oauth_state")
    if not cookie_state or cookie_state != state:
        return RedirectResponse(url="/login?error=Invalid+state+parameter", status_code=302)

    # Verify the state hasn't expired (10 minutes)
    serializer = _get_serializer()
    try:
        serializer.loads(state, max_age=600)
    except Exception:
        return RedirectResponse(url="/login?error=State+token+expired", status_code=302)

    # Exchange authorization code for tokens
    token_url = AZURE_TOKEN_URL.format(tenant=settings.azure_ad_tenant_id)
    token_data = {
        "client_id": settings.azure_ad_client_id,
        "client_secret": settings.azure_ad_client_secret,
        "code": code,
        "redirect_uri": settings.azure_ad_redirect_uri,
        "grant_type": "authorization_code",
        "scope": AZURE_SCOPES,
    }

    async with httpx.AsyncClient() as client:
        token_response = await client.post(token_url, data=token_data)
        if token_response.status_code != 200:
            return RedirectResponse(
                url="/login?error=Failed+to+exchange+authorization+code",
                status_code=302,
            )

        tokens = token_response.json()
        access_token = tokens.get("access_token")
        if not access_token:
            return RedirectResponse(
                url="/login?error=No+access+token+received", status_code=302
            )

        # Fetch user profile from Microsoft Graph
        graph_response = await client.get(
            GRAPH_ME_URL,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        if graph_response.status_code != 200:
            return RedirectResponse(
                url="/login?error=Failed+to+fetch+user+profile", status_code=302
            )

        profile = graph_response.json()

    # Log full Azure AD profile for debugging field availability
    logger.info("Azure AD profile for %s: %s", profile.get("mail") or profile.get("userPrincipalName"), json.dumps(profile, indent=2, default=str))

    # JIT provision or update user
    user = await find_or_create_azure_user(db, profile)
    if not user:
        return RedirectResponse(
            url="/login?error=Failed+to+provision+user+account", status_code=302
        )

    if not user.is_active:
        return RedirectResponse(
            url="/login?error=Account+is+inactive", status_code=302
        )

    await db.commit()

    # Set signed session cookie and redirect to dashboard
    from backend.app.dashboard.routes import set_session_cookie

    response = RedirectResponse(url="/dashboard", status_code=302)
    set_session_cookie(response, user.id)
    # Clear the OAuth state cookie
    response.delete_cookie(key="azure_oauth_state")
    return response


async def find_or_create_azure_user(db: AsyncSession, profile: dict):
    """Find or create a user from Azure AD profile (JIT provisioning).

    Profile fields from Microsoft Graph /me:
    - id: Azure AD object ID (OID)
    - displayName: Full name
    - mail: Email address
    - userPrincipalName: UPN (fallback for email)
    - jobTitle: Job title (used to map to group)
    - department: Department name
    - officeLocation: Office/college location
    """
    azure_oid = profile.get("id")
    email = profile.get("mail") or profile.get("userPrincipalName")
    display_name = profile.get("displayName")
    job_title = profile.get("jobTitle")
    department = profile.get("department")
    college = profile.get("officeLocation")

    if not azure_oid or not email:
        return None

    email = email.lower()

    # 1. Look up by azure_oid first
    user = await crud.get_user_by_azure_oid(db, azure_oid)

    # 2. Fallback: look up by email (link pre-existing local accounts)
    if not user:
        user = await crud.get_user_by_email(db, email)

    if user:
        # Update existing user with latest Azure AD info
        if not user.azure_oid:
            user.azure_oid = azure_oid
        if display_name:
            user.full_name = display_name
        if department:
            user.department = department
        if college:
            user.college = college
        user.last_login_at = datetime.now(timezone.utc)
        await db.flush()
        return user

    # 3. Create new user via JIT provisioning
    group_name = _map_job_title_to_group(job_title)
    group = await crud.get_group_by_name(db, group_name)

    # Fallback to "other" group if mapped group doesn't exist
    if not group:
        settings = get_settings()
        group = await crud.get_group_by_name(db, settings.azure_ad_default_group)

    role = _map_group_to_role(group_name)

    # Use email prefix as username
    username = email.split("@")[0]
    # Handle potential username conflicts
    existing = await crud.get_user_by_username(db, username)
    if existing:
        username = f"{username}_{azure_oid[:8]}"

    user = await crud.create_user(
        db=db,
        username=username,
        email=email,
        password_hash=None,
        role=role,
        full_name=display_name,
        group_id=group.id if group else None,
        college=college,
        department=department,
    )
    user.azure_oid = azure_oid
    user.last_login_at = datetime.now(timezone.utc)
    await db.flush()

    # Create quota from group defaults
    if group:
        await crud.create_quota(
            db=db,
            user_id=user.id,
            rpm_limit=group.rpm_limit,
            max_concurrent=group.max_concurrent,
        )

    return user
