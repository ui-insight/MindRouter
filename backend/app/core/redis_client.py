"""Redis client for atomic token metrics across workers."""

import asyncio
import time
import uuid
from typing import Iterable, Optional

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

        # socket_timeout bounds EVERY command, not just connects: a wedged
        # Redis (VM pause, partition without RST) raises redis.TimeoutError
        # instead of blocking forever, so all callers' except-Exception
        # handlers fail open as designed. redis-py's retry_on_timeout
        # defaults to False, so timeouts fail fast without retry
        # multiplication.
        _redis = aioredis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
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


# ── Leader lease (single-active-worker election) ──────────────────────────
# Compare-and-act so only the token owner can renew or release — a stalled
# leader whose key expired can never clobber the new leader.
_RENEW_LUA = (
    "if redis.call('get', KEYS[1]) == ARGV[1] "
    "then return redis.call('expire', KEYS[1], ARGV[2]) else return 0 end"
)
_RELEASE_LUA = (
    "if redis.call('get', KEYS[1]) == ARGV[1] "
    "then return redis.call('del', KEYS[1]) else return 0 end"
)


async def acquire_lease(key: str, token: str, ttl_seconds: int) -> bool:
    """Claim `key` for `token` iff currently unheld (SET NX EX). True if won."""
    if not _available or not _redis:
        return False
    try:
        return bool(await _redis.set(key, token, nx=True, ex=ttl_seconds))
    except Exception:
        logger.exception("lease_acquire_failed", extra={"key": key})
        return False


async def renew_lease(key: str, token: str, ttl_seconds: int) -> bool:
    """Extend the lease iff we still own it. False means we lost it."""
    if not _available or not _redis:
        return False
    try:
        return bool(await _redis.eval(_RENEW_LUA, 1, key, token, ttl_seconds))
    except Exception:
        logger.exception("lease_renew_failed", extra={"key": key})
        return False


async def release_lease(key: str, token: str) -> None:
    """Release the lease iff we still own it (safe no-op otherwise)."""
    if not _available or not _redis:
        return
    try:
        await _redis.eval(_RELEASE_LUA, 1, key, token)
    except Exception:
        logger.exception("lease_release_failed", extra={"key": key})


# ── Concurrency slots (fleet-wide "at most N in flight") ──────────────────
# Each holder is a member of a sorted set scored by its expiry time, so a
# worker that dies holding a slot frees it when the score passes instead of
# leaking it forever. Holders renew while they work.
_SLOT_ACQUIRE_LUA = """
local now = tonumber(ARGV[1])
local expiry = tonumber(ARGV[2])
local token = ARGV[3]
local ttl = tonumber(ARGV[6])
for i = 1, #KEYS do
  redis.call('zremrangebyscore', KEYS[i], '-inf', now)
end
local user_limit = tonumber(ARGV[4])
if user_limit > 0 and redis.call('zcard', KEYS[1]) >= user_limit then
  return 1
end
local global_limit = tonumber(ARGV[5])
if global_limit > 0 and redis.call('zcard', KEYS[2]) >= global_limit then
  return 2
end
for i = 1, #KEYS do
  redis.call('zadd', KEYS[i], expiry, token)
  redis.call('expire', KEYS[i], ttl)
end
return 0
"""

SLOT_OK, SLOT_USER_FULL, SLOT_GLOBAL_FULL = 0, 1, 2


async def acquire_slot(
    user_key: str, global_key: str, token: str,
    user_limit: int, global_limit: int, ttl_seconds: int,
) -> Optional[int]:
    """Take one slot in both sets, or none if either is full.

    Returns SLOT_OK, SLOT_USER_FULL or SLOT_GLOBAL_FULL, or None when Redis
    is unavailable (the caller falls back to an in-process count). A limit of
    0 means unlimited.
    """
    if not _available or not _redis:
        return None
    now = time.time()
    try:
        return int(await _redis.eval(
            _SLOT_ACQUIRE_LUA, 2, user_key, global_key,
            now, now + ttl_seconds, token, user_limit, global_limit, ttl_seconds,
        ))
    except Exception:
        logger.exception("slot_acquire_failed", key=user_key)
        return None


# Waiting for a slot. Waiters sit in sorted sets scored by arrival time;
# entries older than `stale_before` belong to waiters that died (no waiter
# waits that long) and are dropped. Within one user, slots go to the OLDEST
# waiter, so an account's documents start in the order they arrived.
_WAIT_ENQUEUE_LUA = """
local now = tonumber(ARGV[1])
local token = ARGV[2]
local stale_before = tonumber(ARGV[5])
local ttl = tonumber(ARGV[6])
for i = 1, #KEYS do
  redis.call('zremrangebyscore', KEYS[i], '-inf', stale_before)
end
local user_limit = tonumber(ARGV[3])
if user_limit > 0 and redis.call('zcard', KEYS[1]) >= user_limit then
  return 1
end
local global_limit = tonumber(ARGV[4])
if global_limit > 0 and redis.call('zcard', KEYS[2]) >= global_limit then
  return 2
end
for i = 1, #KEYS do
  redis.call('zadd', KEYS[i], now, token)
  redis.call('expire', KEYS[i], ttl)
end
return 0
"""

_SLOT_ACQUIRE_QUEUED_LUA = """
local now = tonumber(ARGV[1])
local expiry = tonumber(ARGV[2])
local token = ARGV[3]
local ttl = tonumber(ARGV[6])
local stale_before = tonumber(ARGV[7])
redis.call('zremrangebyscore', KEYS[1], '-inf', now)
redis.call('zremrangebyscore', KEYS[2], '-inf', now)
redis.call('zremrangebyscore', KEYS[3], '-inf', stale_before)
redis.call('zremrangebyscore', KEYS[4], '-inf', stale_before)
if not redis.call('zscore', KEYS[3], token) then
  return 3
end
local head = redis.call('zrange', KEYS[3], 0, 0)[1]
if head ~= token then
  return 4
end
local user_limit = tonumber(ARGV[4])
if user_limit > 0 and redis.call('zcard', KEYS[1]) >= user_limit then
  return 1
end
local global_limit = tonumber(ARGV[5])
if global_limit > 0 and redis.call('zcard', KEYS[2]) >= global_limit then
  return 2
end
redis.call('zadd', KEYS[1], expiry, token)
redis.call('zadd', KEYS[2], expiry, token)
redis.call('expire', KEYS[1], ttl)
redis.call('expire', KEYS[2], ttl)
redis.call('zrem', KEYS[3], token)
redis.call('zrem', KEYS[4], token)
return 0
"""

SLOT_NOT_QUEUED, SLOT_NOT_YOUR_TURN = 3, 4


async def enqueue_waiter(
    user_wait_key: str, global_wait_key: str, token: str,
    user_limit: int, global_limit: int, stale_seconds: int,
) -> Optional[int]:
    """Join the wait line for a slot, or not if the line itself is full.

    Returns SLOT_OK, SLOT_USER_FULL or SLOT_GLOBAL_FULL (the WAITING limits),
    or None when Redis is unavailable. A limit of 0 means unlimited.
    """
    if not _available or not _redis:
        return None
    now = time.time()
    try:
        return int(await _redis.eval(
            _WAIT_ENQUEUE_LUA, 2, user_wait_key, global_wait_key,
            now, token, user_limit, global_limit, now - stale_seconds, stale_seconds,
        ))
    except Exception:
        logger.exception("slot_enqueue_failed", key=user_wait_key)
        return None


async def acquire_queued_slot(
    user_key: str, global_key: str, user_wait_key: str, global_wait_key: str,
    token: str, user_limit: int, global_limit: int, ttl_seconds: int,
    stale_seconds: int,
) -> Optional[int]:
    """Move a waiter into a slot if it is first in its user's line and
    both limits have room. Returns SLOT_OK, SLOT_USER_FULL, SLOT_GLOBAL_FULL,
    SLOT_NOT_QUEUED (its wait entry is gone), SLOT_NOT_YOUR_TURN, or None
    when Redis is unavailable."""
    if not _available or not _redis:
        return None
    now = time.time()
    try:
        return int(await _redis.eval(
            _SLOT_ACQUIRE_QUEUED_LUA, 4, user_key, global_key, user_wait_key, global_wait_key,
            now, now + ttl_seconds, token, user_limit, global_limit, ttl_seconds,
            now - stale_seconds,
        ))
    except Exception:
        logger.exception("slot_acquire_failed", key=user_key)
        return None


async def renew_slot(keys: Iterable[str], token: str, ttl_seconds: int) -> None:
    """Push the holder's expiry forward (XX: never re-adds a released slot)."""
    if not _available or not _redis:
        return
    try:
        pipe = _redis.pipeline(transaction=False)
        for key in keys:
            pipe.zadd(key, {token: time.time() + ttl_seconds}, xx=True)
            pipe.expire(key, ttl_seconds)
        await pipe.execute()
    except Exception:
        logger.exception("slot_renew_failed")


async def release_slot(keys: Iterable[str], token: str) -> None:
    """Give the slot back (safe no-op if it already expired)."""
    if not _available or not _redis:
        return
    try:
        pipe = _redis.pipeline(transaction=False)
        for key in keys:
            pipe.zrem(key, token)
        await pipe.execute()
    except Exception:
        logger.exception("slot_release_failed")


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


# ------------------------------------------------------------------
# Per-backend admission slots (fleet-wide max_concurrent enforcement)
# ------------------------------------------------------------------
# Each worker owns its own subkey mr:adm:{backend_id}:{worker_id}, so no
# worker ever decrements another worker's count.  A worker that dies stops
# refreshing its TTLs and its subkeys expire — leaked slots self-heal
# within _ADM_TTL_SECONDS.  The fleet-wide count is the sum of all live
# subkeys for a backend.

WORKER_ID = uuid.uuid4().hex
_ADM_KEY_PREFIX = "mr:adm:"
_ADM_TTL_SECONDS = 90  # ~3 missed 30s reconcile cycles before slots leak away

# Cooldown breaker for the admission helpers, which sit on the routing
# critical path: after a failure, skip Redis for a short window so an
# outage costs at most one bounded failure per worker per window (routing
# instantly falls back to local per-worker counters) instead of a
# connect-timeout plus log flood on every request. The 30s maintenance
# reconcile naturally re-probes Redis after cooldown and restores the
# shared counters via absolute SET.
_ADM_COOLDOWN_S = 15.0
_adm_down_until = 0.0

# Per-worker micro-cache of the admission snapshot: the full-keyspace SCAN
# is O(total keys) and route_job runs on every routing attempt (amplified
# by the capacity-waiter thundering herd), so back-to-back attempts reuse
# one fetch. 250ms staleness is dwarfed by the existing 90s key-TTL / 30s
# reconcile drift bounds.
_ADM_SNAPSHOT_TTL = 0.25  # seconds
_adm_snapshot: Optional[tuple[float, dict[int, int]]] = None


def _adm_ready() -> bool:
    return _available and _redis is not None and time.monotonic() >= _adm_down_until


def _adm_trip() -> None:
    global _adm_down_until
    _adm_down_until = time.monotonic() + _ADM_COOLDOWN_S

# DECR with a floor of zero (a decrement can arrive after the subkey
# expired or was reconciled down — never let the count go negative).
_ADM_DECR_LUA = (
    "local v = redis.call('decr', KEYS[1]) "
    "if v < 0 then redis.call('set', KEYS[1], 0) v = 0 end "
    "redis.call('expire', KEYS[1], ARGV[1]) "
    "return v"
)


def _adm_key(backend_id: int, worker_id: str) -> str:
    return f"{_ADM_KEY_PREFIX}{backend_id}:{worker_id}"


async def incr_backend_inflight(
    backend_id: int, worker_id: str = WORKER_ID
) -> Optional[int]:
    """Claim one admission slot on a backend for this worker.

    Returns this worker's new subkey count, or None if Redis is
    unavailable (caller falls back to per-worker admission).
    """
    if not _adm_ready():
        return None
    try:
        key = _adm_key(backend_id, worker_id)
        pipe = _redis.pipeline(transaction=False)
        pipe.incr(key)
        pipe.expire(key, _ADM_TTL_SECONDS)
        results = await pipe.execute()
        # Keep the snapshot micro-cache honest for this worker's own claims
        if _adm_snapshot is not None:
            _adm_snapshot[1][backend_id] = _adm_snapshot[1].get(backend_id, 0) + 1
        return int(results[0])
    except Exception:
        logger.exception("redis_incr_backend_inflight_failed", backend_id=backend_id)
        _adm_trip()
        return None


async def decr_backend_inflight(
    backend_id: int, worker_id: str = WORKER_ID
) -> Optional[int]:
    """Release one admission slot on a backend for this worker (floor 0)."""
    if not _adm_ready():
        return None
    try:
        key = _adm_key(backend_id, worker_id)
        result = int(await _redis.eval(_ADM_DECR_LUA, 1, key, _ADM_TTL_SECONDS))
        # Mirror the release into the snapshot micro-cache (floor 0)
        if _adm_snapshot is not None:
            _adm_snapshot[1][backend_id] = max(
                0, _adm_snapshot[1].get(backend_id, 0) - 1
            )
        return result
    except Exception:
        logger.exception("redis_decr_backend_inflight_failed", backend_id=backend_id)
        _adm_trip()
        return None


async def set_backend_inflight(
    backend_id: int, count: int, worker_id: str = WORKER_ID
) -> None:
    """Reconcile this worker's slot count to its actual local in-flight count.

    An absolute SET (not INCR) bounds drift from any missed decrements;
    called periodically it also refreshes the subkey TTL.
    """
    if not _adm_ready():
        return
    try:
        key = _adm_key(backend_id, worker_id)
        if count > 0:
            await _redis.set(key, count, ex=_ADM_TTL_SECONDS)
        else:
            await _redis.delete(key)
    except Exception:
        logger.exception("redis_set_backend_inflight_failed", backend_id=backend_id)
        _adm_trip()


async def get_backend_inflight_snapshot(
    backend_ids: Iterable[int],
) -> Optional[dict[int, int]]:
    """Sum live admission slots per backend across all workers.

    Returns {backend_id: fleet_wide_inflight} covering every requested id
    (0 when no worker holds slots), or None if Redis is unavailable —
    callers must then fall back to their local per-worker counts.

    The unfiltered per-backend sums are micro-cached per worker for
    _ADM_SNAPSHOT_TTL and projected onto the requested ids per call, so
    waiter storms and back-to-back routing attempts reuse one SCAN+MGET.
    """
    global _adm_snapshot
    if not _adm_ready():
        return None
    ids = list(backend_ids)
    now = time.monotonic()
    if _adm_snapshot is not None and now - _adm_snapshot[0] < _ADM_SNAPSHOT_TTL:
        sums = _adm_snapshot[1]
        return {bid: sums.get(bid, 0) for bid in ids}
    try:
        sums: dict[int, int] = {}
        keys = []
        async for key in _redis.scan_iter(match=f"{_ADM_KEY_PREFIX}*", count=200):
            keys.append(key)
        if keys:
            vals = await _redis.mget(keys)
            for key, val in zip(keys, vals):
                if val is None:
                    continue  # expired between SCAN and MGET
                try:
                    # mr:adm:{backend_id}:{worker_id}
                    bid = int(key.split(":")[2])
                    count = int(val)
                except (IndexError, ValueError):
                    continue
                if count > 0:
                    sums[bid] = sums.get(bid, 0) + count
        _adm_snapshot = (now, sums)
        return {bid: sums.get(bid, 0) for bid in ids}
    except Exception:
        logger.exception("redis_backend_inflight_snapshot_failed")
        _adm_trip()
        return None  # never cache failures — fail-open stays instant


async def clear_backend_inflight(worker_id: str = WORKER_ID) -> None:
    """Delete this worker's admission subkeys (graceful-shutdown cleanup).

    Without this, requests still in flight when a worker is stopped die
    without decrementing, and replacement workers see the dead worker's
    phantom slots until the 90s TTL expires. Only self-owned keys match
    (WORKER_ID is a uuid4 hex, no glob metachars), so this is safe with
    concurrent workers; the TTL remains the crash-only backstop.
    """
    if not _available or not _redis:
        return
    try:
        keys = []
        async for key in _redis.scan_iter(
            match=f"{_ADM_KEY_PREFIX}*:{worker_id}", count=200
        ):
            keys.append(key)
        if keys:
            await _redis.delete(*keys)
    except Exception:
        logger.exception("redis_clear_backend_inflight_failed")


# ------------------------------------------------------------------
# RPM rate limiting (shared across all workers via Redis)
# ------------------------------------------------------------------

_RPM_KEY_PREFIX = "rpm:"
_RPM_WINDOW_SECONDS = 60


def _rpm_fallback(key: str, rpm_limit: int) -> tuple[bool, int]:
    """Consult the bounded in-process limiter when Redis is unavailable.

    Gated by ``settings.rate_limit_local_fallback`` (default True). When
    disabled, preserves the historical fully-open behaviour. The local
    limiter enforces the same fixed-window contract as :func:`check_rpm`,
    so callers are unaffected.
    """
    if not getattr(get_settings(), "rate_limit_local_fallback", True):
        return True, 0
    try:
        from backend.app.security.rate_limits import check_rpm_local

        return check_rpm_local(key, rpm_limit)
    except Exception:
        logger.exception("rpm_local_fallback_failed", key=key)
        return True, 0  # never let the limiter itself break the request path


async def check_rpm(key: str, rpm_limit: int) -> tuple[bool, int]:
    """Check whether a request is allowed under the RPM limit.

    Uses INCR + EXPIRE for a fixed 60-second sliding window.
    When Redis is unavailable, falls back to a bounded in-process limiter
    (``settings.rate_limit_local_fallback``, default True) instead of
    admitting everything.

    Args:
        key: Rate limit key (e.g., ``"user:123"``).
        rpm_limit: Maximum requests allowed per 60-second window.

    Returns:
        ``(allowed, current_count)``  — *current_count* includes this
        request if allowed.
    """
    if rpm_limit <= 0:
        return True, 0
    if not _available or not _redis:
        return _rpm_fallback(key, rpm_limit)
    try:
        redis_key = f"{_RPM_KEY_PREFIX}{key}"
        pipe = _redis.pipeline(transaction=True)
        pipe.incr(redis_key)
        pipe.ttl(redis_key)
        results = await pipe.execute()
        current = int(results[0])
        ttl = int(results[1])
        # Only set EXPIRE when the key is new (first request in window).
        # Setting it on every request would reset the TTL and prevent the
        # counter from ever expiring while requests keep arriving — even
        # rejected ones — permanently locking out the user.
        if ttl < 0:
            await _redis.expire(redis_key, _RPM_WINDOW_SECONDS)
        if current > rpm_limit:
            # Over limit — decrement back so the window stays accurate
            await _redis.decr(redis_key)
            return False, current - 1
        return True, current
    except Exception:
        logger.exception("redis_check_rpm_failed", key=key)
        return _rpm_fallback(key, rpm_limit)
