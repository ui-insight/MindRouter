"""Redis client for atomic token metrics across workers."""

import asyncio
from typing import Optional

from backend.app.logging_config import get_logger
from backend.app.settings import get_settings

logger = get_logger(__name__)

_redis = None
_available = False

INFLIGHT_KEY = "streaming:inflight_tokens"
INFLIGHT_TTL_SECONDS = 30  # Auto-expire if no streaming activity

# Cluster-wide token totals (atomically incremented on each request completion)
_CLUSTER_PROMPT_KEY = "cluster:prompt_tokens"
_CLUSTER_COMPLETION_KEY = "cluster:completion_tokens"
_CLUSTER_TOTAL_KEY = "cluster:total_tokens_counter"


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
        # Reset inflight streaming counter on startup (clears orphans from crashes)
        await _redis.set(INFLIGHT_KEY, 0)
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


async def incr_inflight_tokens(amount: int) -> Optional[int]:
    """Atomically increment the inflight streaming token counter.

    Sets a TTL on the key so that leaked counters auto-expire if no
    streaming activity refreshes it within INFLIGHT_TTL_SECONDS.
    """
    if not _available or not _redis or amount <= 0:
        return None
    try:
        pipe = _redis.pipeline(transaction=False)
        pipe.incrby(INFLIGHT_KEY, amount)
        pipe.expire(INFLIGHT_KEY, INFLIGHT_TTL_SECONDS)
        results = await pipe.execute()
        return results[0]
    except Exception:
        logger.exception("redis_incr_inflight_failed")
        return None


async def decr_inflight_tokens(amount: int) -> Optional[int]:
    """Atomically decrement the inflight streaming token counter."""
    if not _available or not _redis or amount <= 0:
        return None
    try:
        return await _redis.decrby(INFLIGHT_KEY, amount)
    except Exception:
        logger.exception("redis_decr_inflight_failed")
        return None


async def get_inflight_tokens() -> int:
    """Get current inflight streaming token estimate. Returns 0 if unavailable."""
    if not _available or not _redis:
        return 0
    try:
        val = await _redis.get(INFLIGHT_KEY)
        return max(0, int(val)) if val is not None else 0
    except Exception:
        logger.exception("redis_get_inflight_failed")
        return 0


# ------------------------------------------------------------------
# Cluster-wide token totals (live counter, no TTL)
# ------------------------------------------------------------------


async def incr_cluster_tokens(
    prompt_tokens: int, completion_tokens: int, total_tokens: int
) -> None:
    """Atomically increment the cluster-wide token counters."""
    if not _available or not _redis:
        return
    try:
        pipe = _redis.pipeline(transaction=False)
        pipe.incrby(_CLUSTER_PROMPT_KEY, prompt_tokens)
        pipe.incrby(_CLUSTER_COMPLETION_KEY, completion_tokens)
        pipe.incrby(_CLUSTER_TOTAL_KEY, total_tokens)
        await pipe.execute()
    except Exception:
        pass  # Best-effort, don't break the completion path


async def get_cluster_tokens() -> dict | None:
    """Read cluster-wide token totals. Returns None if not seeded yet."""
    if not _available or not _redis:
        return None
    try:
        pipe = _redis.pipeline(transaction=False)
        pipe.get(_CLUSTER_PROMPT_KEY)
        pipe.get(_CLUSTER_COMPLETION_KEY)
        pipe.get(_CLUSTER_TOTAL_KEY)
        vals = await pipe.execute()
        if vals[2] is None:
            return None  # Not seeded yet
        return {
            "prompt_tokens": int(vals[0] or 0),
            "completion_tokens": int(vals[1] or 0),
            "total_tokens": int(vals[2] or 0),
        }
    except Exception:
        return None


async def seed_cluster_tokens(
    prompt_tokens: int, completion_tokens: int, total_tokens: int
) -> None:
    """Seed the cluster token counters (called once at startup from DB)."""
    if not _available or not _redis:
        return
    try:
        pipe = _redis.pipeline(transaction=False)
        pipe.set(_CLUSTER_PROMPT_KEY, prompt_tokens)
        pipe.set(_CLUSTER_COMPLETION_KEY, completion_tokens)
        pipe.set(_CLUSTER_TOTAL_KEY, total_tokens)
        await pipe.execute()
    except Exception:
        logger.exception("redis_seed_cluster_tokens_failed")
