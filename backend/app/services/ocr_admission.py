"""Admission control for OCR: how many documents may be in flight at once.

Background (2026-11-05). One client sent PDFs to /v1/ocr far faster than they
could be processed, and retried the ones that timed out. Nothing limited how
many documents the gateway accepted, and each one turns its pages into images
on the gateway's own CPUs (pdftoppm) before any GPU sees them: 39 conversions
ran at once, load average reached ~90 on 32 cores, and the starved app then
held database connections until MariaDB refused new ones (error 1040). Every
user got 500s, not just the one sending the documents.

Two limits on documents IN PROGRESS, both fleet-wide via Redis so they hold
across uvicorn workers:

- per user: one account may have only N documents in progress;
- whole server: at most M documents in progress from everyone together.

A document over a limit WAITS for a slot, up to `max_wait` seconds, so a
client that sends a modest burst just gets slower answers instead of errors.
Within one account, waiting documents start in arrival order. The wait line
is itself bounded (per user and server-wide), because a waiting request still
holds its uploaded file in memory and an open connection; past those bounds,
or after `max_wait`, the document is refused with 429 and Retry-After before
any of its pages are converted. A waiter whose client disconnects leaves the
line, and a waiter that died (its worker crashed, or it was cancelled at an
awkward moment) is dropped within seconds via a liveness mark it refreshes on
every poll, so it cannot block its account's line. In-progress limits of 0
mean unlimited; `max_wait` 0 means refuse at once. Across accounts there is
no strict order: a document from an account with nobody waiting may take a
freed server-wide slot ahead of another account's waiter, which then keeps
waiting (never past `max_wait`).

If Redis is unavailable the same in-progress limits apply per worker instead
(so the server-wide figure is effectively multiplied by the worker count) and
nothing waits: an over-limit document is refused at once, since waiting
requests per worker would multiply the memory they hold by the worker count
too. A Redis error part way through a wait is treated as "not yet" (a brief
timeout must not let the document bypass the fleet-wide limits); only several
failures in a row fall back to per-worker counting.
"""

import asyncio
import contextlib
import time
import uuid
from collections import Counter
from typing import AsyncIterator, Awaitable, Callable, Optional

from backend.app.core import redis_client
from backend.app.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_MAX_PER_USER = 2
DEFAULT_MAX_GLOBAL = 8
DEFAULT_MAX_WAIT_SECONDS = 60
DEFAULT_MAX_WAITING_PER_USER = 4
DEFAULT_MAX_WAITING = 16
# Upper bound of the admin setting; also fixes how old a wait-line entry must
# be to count as dead, so that callers with different (or stale cached)
# max_wait values never evict each other's live waiters.
MAX_WAIT_CAP_SECONDS = 600
STALE_SECONDS = MAX_WAIT_CAP_SECONDS + 60

# A slot outlives a dead worker by at most this long; live holders renew it.
SLOT_TTL_SECONDS = 300
HEARTBEAT_SECONDS = 60
RETRY_AFTER_SECONDS = 30
# How often a waiting document checks for a free slot, and how long its
# liveness mark lasts without a check (generous: a busy event loop can be late).
POLL_SECONDS = 1.0
LIVE_TTL_SECONDS = 10
# Consecutive Redis failures before falling back to per-worker counting.
REDIS_FAILURES_BEFORE_FALLBACK = 3

_GLOBAL_KEY = "ocr:docs:all"
_GLOBAL_WAIT_KEY = "ocr:wait:all"
_LIVE_PREFIX = "ocr:waiter:"


def _user_key(user_id: int) -> str:
    return f"ocr:docs:user:{user_id}"


def _user_wait_key(user_id: int) -> str:
    return f"ocr:wait:user:{user_id}"


class OcrBusy(Exception):
    """The document was refused because a concurrency limit is reached."""

    def __init__(self, scope: str, limit: int, waited: bool = False):
        self.scope = scope
        self.limit = limit
        self.waited = waited
        self.retry_after = RETRY_AFTER_SECONDS
        if scope == "user":
            msg = (
                f"Too many OCR documents in progress for this account "
                f"(limit {limit} at a time). Wait for one to finish, then retry."
            )
        elif scope == "user_waiting":
            msg = (
                f"Too many OCR documents already waiting for this account "
                f"(limit {limit}). Send documents one or two at a time and wait "
                f"for each response."
            )
        elif scope == "global_waiting":
            msg = (
                f"The OCR service queue is full ({limit} documents waiting). "
                f"Retry in {RETRY_AFTER_SECONDS} seconds."
            )
        else:
            msg = (
                f"The OCR service is at capacity ({limit} documents in "
                f"progress). Retry in {RETRY_AFTER_SECONDS} seconds."
            )
        if waited:
            msg += " (Waited for a free slot first.)"
        super().__init__(msg)


class WaitAbandoned(Exception):
    """The client went away while its document was waiting for a slot."""


# In-process fallback counts, used only while Redis is unavailable.
_local_users: Counter = Counter()
_local_total = 0


def _local_try(user_id: int, per_user: int, global_limit: int) -> Optional[str]:
    """Take a local slot if both limits allow; else the scope that is full."""
    global _local_total
    if per_user > 0 and _local_users[user_id] >= per_user:
        return "user"
    if global_limit > 0 and _local_total >= global_limit:
        return "global"
    _local_users[user_id] += 1
    _local_total += 1
    return None


def _local_release(user_id: int) -> None:
    global _local_total
    _local_users[user_id] -= 1
    if _local_users[user_id] <= 0:
        del _local_users[user_id]
    _local_total = max(0, _local_total - 1)


async def _heartbeat(keys, token: str) -> None:
    while True:
        await asyncio.sleep(HEARTBEAT_SECONDS)
        await redis_client.renew_slot(keys, token, SLOT_TTL_SECONDS)


def _full(verdict: Optional[int], per_user: int, global_limit: int, waited: bool) -> OcrBusy:
    """The refusal for the last thing that stopped us (behind our own
    account's earlier documents counts as the account's limit)."""
    if verdict == redis_client.SLOT_GLOBAL_FULL:
        return OcrBusy("global", global_limit, waited)
    return OcrBusy("user", per_user, waited)


class _RedisDown(Exception):
    """Redis failed several times in a row: fall back to local counting."""


async def _admit_redis(
    user_id: int, token: str, per_user: int, global_limit: int,
    max_wait: float, max_waiting_per_user: int, max_waiting: int,
    should_abort: Optional[Callable[[], Awaitable[bool]]],
) -> None:
    """Get a fleet-wide slot, waiting in line if need be.

    Returns with the slot held; raises OcrBusy, WaitAbandoned or _RedisDown.
    Leaves the line on every exit only via the caller's cleanup (which
    removes the token from every set), so nothing here needs its own.
    """
    keys = (_user_key(user_id), _GLOBAL_KEY)
    wait_keys = (_user_wait_key(user_id), _GLOBAL_WAIT_KEY)
    deadline = time.monotonic() + max_wait
    failures = 0
    queued = False
    last: Optional[int] = None
    while True:
        if queued:
            verdict = await redis_client.acquire_queued_slot(
                keys[0], keys[1], wait_keys[0], wait_keys[1], token,
                per_user, global_limit, SLOT_TTL_SECONDS, STALE_SECONDS,
                _LIVE_PREFIX, LIVE_TTL_SECONDS,
            )
        else:
            verdict = await redis_client.try_or_enqueue_slot(
                keys[0], keys[1], wait_keys[0], wait_keys[1], token,
                per_user, global_limit, SLOT_TTL_SECONDS, STALE_SECONDS,
                max_waiting_per_user, max_waiting, _LIVE_PREFIX, LIVE_TTL_SECONDS,
            )

        if verdict is None:
            failures += 1
            if failures >= REDIS_FAILURES_BEFORE_FALLBACK:
                raise _RedisDown()
        else:
            failures = 0
            if verdict == redis_client.SLOT_OK:
                return
            if verdict == redis_client.SLOT_WAIT_USER_FULL:
                raise OcrBusy("user_waiting", max_waiting_per_user, waited=queued)
            if verdict == redis_client.SLOT_WAIT_GLOBAL_FULL:
                raise OcrBusy("global_waiting", max_waiting, waited=queued)
            if verdict == redis_client.SLOT_QUEUED:
                if not queued:
                    logger.info("ocr_waiting", user_id=user_id)
                queued = True
            elif verdict == redis_client.SLOT_NOT_QUEUED:
                # Our entry was dropped (e.g. our liveness lapsed while the
                # event loop was starved): rejoin through the front door.
                queued = False
            else:
                last = verdict

        if time.monotonic() >= deadline:
            raise _full(last, per_user, global_limit, waited=True)
        if should_abort is not None and await should_abort():
            raise WaitAbandoned()
        await asyncio.sleep(POLL_SECONDS)


@contextlib.asynccontextmanager
async def document_slot(
    user_id: int,
    per_user: int = DEFAULT_MAX_PER_USER,
    global_limit: int = DEFAULT_MAX_GLOBAL,
    max_wait_seconds: float = DEFAULT_MAX_WAIT_SECONDS,
    max_waiting_per_user: int = DEFAULT_MAX_WAITING_PER_USER,
    max_waiting: int = DEFAULT_MAX_WAITING,
    should_abort: Optional[Callable[[], Awaitable[bool]]] = None,
) -> AsyncIterator[None]:
    """Hold one OCR document slot for the duration of the block.

    Waits up to ``max_wait_seconds`` for a slot (see module docstring), then
    raises OcrBusy — always before any work. Raises WaitAbandoned if
    ``should_abort()`` turns true (the client left) while waiting.
    """
    per_user, global_limit = max(0, int(per_user)), max(0, int(global_limit))
    max_wait = min(float(MAX_WAIT_CAP_SECONDS), max(0.0, float(max_wait_seconds)))
    max_waiting_per_user, max_waiting = max(0, int(max_waiting_per_user)), max(0, int(max_waiting))
    keys = (_user_key(user_id), _GLOBAL_KEY)
    all_keys = keys + (_user_wait_key(user_id), _GLOBAL_WAIT_KEY)
    token = uuid.uuid4().hex
    live_key = _LIVE_PREFIX + token

    async def _leave() -> None:
        await redis_client.leave_slot(all_keys, live_key, token)

    in_redis = redis_client.is_available()
    if in_redis:
        try:
            if max_wait <= 0:
                verdict = await redis_client.acquire_slot(
                    keys[0], keys[1], token, per_user, global_limit, SLOT_TTL_SECONDS,
                )
                if verdict is None:
                    raise _RedisDown()
                if verdict != redis_client.SLOT_OK:
                    raise _full(verdict, per_user, global_limit, waited=False)
            else:
                await _admit_redis(
                    user_id, token, per_user, global_limit, max_wait,
                    max_waiting_per_user, max_waiting, should_abort,
                )
        except _RedisDown:
            # A script may have run even though its reply was lost.
            await asyncio.shield(_leave())
            in_redis = False
            logger.warning("ocr_admission_redis_fallback", user_id=user_id)
        except OcrBusy as busy:
            await asyncio.shield(_leave())
            logger.warning("ocr_refused", scope=busy.scope, user_id=user_id,
                           limit=busy.limit, waited=busy.waited)
            raise
        except BaseException:
            # WaitAbandoned, cancellation, anything: leave the line and drop
            # any slot a script may have granted.
            await asyncio.shield(_leave())
            raise

    if not in_redis:
        full = _local_try(user_id, per_user, global_limit)
        if full is not None:
            busy = OcrBusy(full, per_user if full == "user" else global_limit)
            logger.warning("ocr_refused", scope=busy.scope, user_id=user_id,
                           limit=busy.limit, fallback="local")
            raise busy
        try:
            yield
        finally:
            _local_release(user_id)
        return

    beat = asyncio.create_task(_heartbeat(keys, token))
    try:
        yield
    finally:
        beat.cancel()
        try:
            # gather(return_exceptions) absorbs the heartbeat's own
            # cancellation without swallowing one aimed at us ...
            await asyncio.gather(beat, return_exceptions=True)
        finally:
            # ... and the slot is released even if one arrives right here.
            await asyncio.shield(_leave())
