"""Redis client for atomic token metrics across workers."""

import asyncio
from typing import Optional

from backend.app.logging_config import get_logger
from backend.app.settings import get_settings

logger = get_logger(__name__)

_redis = None
_available = False


async def init_redis() -> None:
    """Initialize the Redis connection. No-op if redis_url is not configured."""
    global _redis, _available
    settings = get_settings()
    if not settings.redis_url:
        logger.info("redis_disabled", reason="no redis_url configured")
        return
    try:
        import redis.asyncio as aioredis

        _redis = aioredis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=5,
        )
        await _redis.ping()
        _available = True
        logger.info("redis_connected", url=settings.redis_url)
    except Exception:
        logger.exception("redis_connect_failed")
        _redis = None
        _available = False


async def close_redis() -> None:
    """Close the Redis connection."""
    global _redis, _available
    if _redis:
        try:
            await _redis.aclose()
        except Exception:
            pass
    _redis = None
    _available = False


def is_available() -> bool:
    """Check whether Redis is connected and available."""
    return _available


async def incr_tokens(user_id: int, amount: int) -> Optional[int]:
    """Atomically increment token counter. Returns new value or None on failure."""
    if not _available or not _redis:
        return None
    try:
        return await _redis.incrby(f"quota:tokens:{user_id}", amount)
    except Exception:
        logger.exception("redis_incr_tokens_failed", user_id=user_id)
        return None


async def get_tokens(user_id: int) -> Optional[int]:
    """Get current token count from Redis. Returns None if unavailable."""
    if not _available or not _redis:
        return None
    try:
        val = await _redis.get(f"quota:tokens:{user_id}")
        return int(val) if val is not None else None
    except Exception:
        logger.exception("redis_get_tokens_failed", user_id=user_id)
        return None


async def set_tokens(user_id: int, value: int) -> bool:
    """Set token counter to a specific value. Returns success."""
    if not _available or not _redis:
        return False
    try:
        await _redis.set(f"quota:tokens:{user_id}", value)
        return True
    except Exception:
        logger.exception("redis_set_tokens_failed", user_id=user_id)
        return False


async def reset_tokens(user_id: int) -> bool:
    """Delete token counter for a user. Returns success."""
    if not _available or not _redis:
        return False
    try:
        await _redis.delete(f"quota:tokens:{user_id}")
        return True
    except Exception:
        logger.exception("redis_reset_tokens_failed", user_id=user_id)
        return False


async def get_all_token_keys() -> dict[int, int]:
    """Scan all quota:tokens:* keys and return {user_id: tokens} dict."""
    if not _available or not _redis:
        return {}
    result = {}
    try:
        async for key in _redis.scan_iter(match="quota:tokens:*", count=200):
            uid_str = key.split(":")[-1]
            try:
                uid = int(uid_str)
                val = await _redis.get(key)
                if val is not None:
                    result[uid] = int(val)
            except (ValueError, TypeError):
                continue
    except Exception:
        logger.exception("redis_scan_tokens_failed")
    return result
