############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# rate_limits.py: Token bucket rate limiting implementation
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Rate limiting implementation."""

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from backend.app.settings import get_settings
from backend.app.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class RateLimitState:
    """Rate limit state for a single key."""

    # Request tracking for RPM
    request_timestamps: list = field(default_factory=list)

    # Concurrent request tracking
    active_requests: int = 0

    # Last cleanup time
    last_cleanup: float = field(default_factory=time.time)


class RateLimiter:
    """
    In-memory rate limiter.

    Tracks:
    - Requests per minute (RPM)
    - Concurrent requests
    """

    def __init__(self):
        self._settings = get_settings()
        self._states: Dict[str, RateLimitState] = defaultdict(RateLimitState)
        self._lock = asyncio.Lock()

    async def check_rate_limit(
        self,
        key: str,
        rpm_limit: int,
        max_concurrent: int,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if a request is allowed under rate limits.

        Args:
            key: Rate limit key (e.g., api_key_id or user_id)
            rpm_limit: Requests per minute limit
            max_concurrent: Maximum concurrent requests

        Returns:
            Tuple of (allowed, reason_if_denied)
        """
        async with self._lock:
            state = self._states[key]
            now = time.time()

            # Cleanup old timestamps (older than 1 minute)
            cutoff = now - 60
            state.request_timestamps = [
                ts for ts in state.request_timestamps if ts > cutoff
            ]

            # Check RPM
            if len(state.request_timestamps) >= rpm_limit:
                return False, f"Rate limit exceeded: {rpm_limit} requests per minute"

            # Check concurrent
            if state.active_requests >= max_concurrent:
                return False, f"Too many concurrent requests: max {max_concurrent}"

            # Record this request
            state.request_timestamps.append(now)
            state.active_requests += 1

            return True, None

    async def release_request(self, key: str) -> None:
        """
        Release a concurrent request slot.

        Call this when a request completes.

        Args:
            key: Rate limit key
        """
        async with self._lock:
            state = self._states[key]
            state.active_requests = max(0, state.active_requests - 1)

    async def get_state(self, key: str) -> Dict:
        """Get rate limit state for a key."""
        async with self._lock:
            state = self._states[key]
            now = time.time()

            # Count recent requests
            cutoff = now - 60
            recent = len([ts for ts in state.request_timestamps if ts > cutoff])

            return {
                "requests_last_minute": recent,
                "active_requests": state.active_requests,
            }

    async def cleanup(self) -> None:
        """Clean up old state entries."""
        async with self._lock:
            now = time.time()
            cutoff = now - 300  # 5 minutes

            keys_to_remove = []
            for key, state in self._states.items():
                # Remove if no recent activity and no active requests
                if (
                    state.active_requests == 0
                    and state.last_cleanup < cutoff
                    and not state.request_timestamps
                ):
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del self._states[key]


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


class RateLimitMiddleware:
    """
    FastAPI middleware for rate limiting.

    Usage:
        app.add_middleware(RateLimitMiddleware)
    """

    def __init__(self, app):
        self.app = app
        self._limiter = get_rate_limiter()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Rate limiting is handled per-request in the auth layer
        # This middleware is just a placeholder for global limits

        await self.app(scope, receive, send)
