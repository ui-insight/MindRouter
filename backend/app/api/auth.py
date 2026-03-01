############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# auth.py: API authentication and authorization middleware
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""API authentication and authorization."""

from datetime import datetime, timezone
from typing import Optional, Tuple

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from itsdangerous import URLSafeTimedSerializer

from backend.app.db import crud
from backend.app.db.models import ApiKey, ApiKeyStatus, User, UserRole, Group
from backend.app.db.session import get_async_db
from backend.app.security.api_keys import verify_api_key
from backend.app.logging_config import get_logger
from backend.app.settings import get_settings

logger = get_logger(__name__)

# Security scheme
bearer_scheme = HTTPBearer(auto_error=False)


async def get_api_key_from_request(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> Optional[str]:
    """
    Extract API key from request.

    Supports:
    - Authorization: Bearer <key>
    - X-API-Key: <key>
    """
    # Try Authorization header first
    if credentials and credentials.credentials:
        return credentials.credentials

    # Try X-API-Key header
    x_api_key = request.headers.get("X-API-Key")
    if x_api_key:
        return x_api_key

    return None


async def authenticate_request(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
    api_key_str: Optional[str] = Depends(get_api_key_from_request),
) -> Tuple[User, ApiKey]:
    """
    Authenticate a request using API key.

    Args:
        request: The incoming request
        db: Database session
        api_key_str: API key from request

    Returns:
        Tuple of (User, ApiKey)

    Raises:
        HTTPException: If authentication fails
    """
    if not api_key_str:
        logger.warning("missing_api_key", path=request.url.path)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide via 'Authorization: Bearer <key>' or 'X-API-Key: <key>'",
        )

    # Verify API key
    api_key = await verify_api_key(db, api_key_str)

    if not api_key:
        logger.warning(
            "invalid_api_key",
            path=request.url.path,
            key_prefix=api_key_str[:8] if len(api_key_str) >= 8 else "short",
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    # Check key status
    if api_key.status != ApiKeyStatus.ACTIVE:
        logger.warning(
            "inactive_api_key",
            key_id=api_key.id,
            status=api_key.status.value,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"API key is {api_key.status.value}",
        )

    # Check expiration (MariaDB returns naive datetimes)
    expires_at = api_key.expires_at
    if expires_at and expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if expires_at and expires_at < datetime.now(timezone.utc):
        logger.warning("expired_api_key", key_id=api_key.id)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key has expired",
        )

    # Get user
    user = api_key.user
    if not user or not user.is_active or user.deleted_at:
        logger.warning(
            "inactive_user",
            key_id=api_key.id,
            user_id=user.id if user else None,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is inactive",
        )

    # NOTE: api_key usage update moved to request completion phase
    # to avoid holding a row lock for the entire request duration.

    return user, api_key


async def require_role(
    required_role: UserRole,
    user: User = Depends(lambda u=Depends(authenticate_request): u[0]),
) -> User:
    """
    Require a minimum role level.

    Role hierarchy: admin > faculty > staff > student
    """
    role_hierarchy = {
        UserRole.STUDENT: 0,
        UserRole.STAFF: 1,
        UserRole.FACULTY: 2,
        UserRole.ADMIN: 3,
    }

    user_level = role_hierarchy.get(user.role, 0)
    required_level = role_hierarchy.get(required_role, 0)

    if user_level < required_level:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Requires {required_role.value} role or higher",
        )

    return user


def require_admin():
    """Dependency that requires admin role (via group.is_admin)."""
    async def check_admin(
        auth_result: Tuple[User, ApiKey] = Depends(authenticate_request),
    ) -> User:
        user, _ = auth_result
        if not user.group or not user.group.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required",
            )
        return user
    return check_admin


def require_admin_or_session():
    """
    Dependency that requires admin role via API key OR session cookie.

    This enables admin-only API endpoints to be called from both:
    - Programmatic clients (via API key in Authorization/X-API-Key header)
    - Dashboard AJAX calls (via session cookie from browser login)
    """
    async def check_admin(
        request: Request,
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
        db: AsyncSession = Depends(get_async_db),
    ) -> User:
        # Try API key auth first
        api_key_str = None
        if credentials and credentials.credentials:
            api_key_str = credentials.credentials
        if not api_key_str:
            api_key_str = request.headers.get("X-API-Key")

        if api_key_str:
            # API key path — delegate to standard auth
            from backend.app.security.api_keys import verify_api_key as _verify
            api_key = await _verify(db, api_key_str)
            if api_key and api_key.status == ApiKeyStatus.ACTIVE:
                user = api_key.user
                if user and user.is_active and user.group and user.group.is_admin:
                    return user
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key or insufficient permissions",
            )

        # Fallback to session cookie (signed with itsdangerous)
        session_data = request.cookies.get("mindrouter_session")
        if session_data:
            try:
                settings = get_settings()
                serializer = URLSafeTimedSerializer(settings.secret_key, salt="session")
                user_id = int(serializer.loads(session_data, max_age=86400 * 7))
                user = await crud.get_user_by_id(db, user_id)
                if user and user.is_active and user.group and user.group.is_admin:
                    return user
            except Exception:
                pass

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )
    return check_admin


class AuthenticatedUser:
    """Dependency class for getting authenticated user."""

    def __init__(self, require_admin: bool = False, require_role: Optional[UserRole] = None):
        self.require_admin_flag = require_admin
        self.require_role = require_role

    async def __call__(
        self,
        auth_result: Tuple[User, ApiKey] = Depends(authenticate_request),
    ) -> User:
        user, _ = auth_result

        if self.require_admin_flag:
            if not user.group or not user.group.is_admin:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Admin access required",
                )
        elif self.require_role:
            # Legacy role hierarchy check (kept for backward compat)
            role_hierarchy = {
                UserRole.STUDENT: 0,
                UserRole.STAFF: 1,
                UserRole.FACULTY: 2,
                UserRole.ADMIN: 3,
            }
            user_level = role_hierarchy.get(user.role, 0)
            required_level = role_hierarchy.get(self.require_role, 0)

            if user_level < required_level:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Requires {self.require_role.value} role or higher",
                )

        return user


class AuthenticatedApiKey:
    """Dependency class for getting authenticated API key."""

    async def __call__(
        self,
        auth_result: Tuple[User, ApiKey] = Depends(authenticate_request),
    ) -> ApiKey:
        _, api_key = auth_result
        return api_key


# Convenience dependencies
get_current_user = AuthenticatedUser()
get_current_user_admin = AuthenticatedUser(require_admin=True)
get_current_api_key = AuthenticatedApiKey()
