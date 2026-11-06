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
line. Limits of 0 mean unlimited; `max_wait` 0 means refuse at once.

If Redis is down the same limits apply per worker instead (so the server-wide
figures are effectively multiplied by the worker count): looser, but bounded.
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

# A slot outlives a dead worker by at most this long; live holders renew it.
SLOT_TTL_SECONDS = 300
HEARTBEAT_SECONDS = 60
RETRY_AFTER_SECONDS = 30
# How often a waiting document checks for a free slot.
POLL_SECONDS = 1.0
# A wait-line entry older than max_wait + this belongs to a dead waiter.
STALE_MARGIN_SECONDS = 60

_GLOBAL_KEY = "ocr:docs:all"
_GLOBAL_WAIT_KEY = "ocr:wait:all"


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
_local_waiting_users: Counter = Counter()
_local_waiting_total = 0


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


def _full(verdict: int, per_user: int, global_limit: int, waited: bool) -> OcrBusy:
    if verdict == redis_client.SLOT_USER_FULL:
        return OcrBusy("user", per_user, waited)
    return OcrBusy("global", global_limit, waited)


async def _wait_redis(
    user_id: int, token: str, per_user: int, global_limit: int,
    max_wait: float, max_waiting_per_user: int, max_waiting: int,
    should_abort: Optional[Callable[[], Awaitable[bool]]],
) -> Optional[bool]:
    """Join the line and wait for a Redis slot.

    Returns True once the slot is held, None if Redis became unavailable
    (the caller falls back to local counting), or raises OcrBusy /
    WaitAbandoned. Our wait-line entry is always gone on return.
    """
    keys = (_user_key(user_id), _GLOBAL_KEY)
    wait_keys = (_user_wait_key(user_id), _GLOBAL_WAIT_KEY)
    stale = int(max_wait + STALE_MARGIN_SECONDS)
    joined = await redis_client.enqueue_waiter(
        wait_keys[0], wait_keys[1], token, max_waiting_per_user, max_waiting, stale,
    )
    if joined is None:
        return None
    if joined == redis_client.SLOT_USER_FULL:
        raise OcrBusy("user_waiting", max_waiting_per_user)
    if joined == redis_client.SLOT_GLOBAL_FULL:
        raise OcrBusy("global_waiting", max_waiting)

    deadline = time.monotonic() + max_wait
    first = True
    last = redis_client.SLOT_NOT_YOUR_TURN
    try:
        while True:
            verdict = await redis_client.acquire_queued_slot(
                keys[0], keys[1], wait_keys[0], wait_keys[1], token,
                per_user, global_limit, SLOT_TTL_SECONDS, stale,
            )
            if verdict == redis_client.SLOT_OK:
                return True
            if verdict is None:
                return None
            if verdict == redis_client.SLOT_NOT_QUEUED:
                # Our entry aged out (a very long stall): rejoin at the back.
                joined = await redis_client.enqueue_waiter(
                    wait_keys[0], wait_keys[1], token,
                    max_waiting_per_user, max_waiting, stale,
                )
                if joined != redis_client.SLOT_OK:
                    raise _full(last, per_user, global_limit, waited=True)
            else:
                last = verdict
            if time.monotonic() >= deadline:
                if last == redis_client.SLOT_NOT_YOUR_TURN:
                    # Behind this account's own earlier documents.
                    last = redis_client.SLOT_USER_FULL
                raise _full(last, per_user, global_limit, waited=not first)
            if first:
                logger.info("ocr_waiting", user_id=user_id)
                first = False
            if should_abort is not None and await should_abort():
                raise WaitAbandoned()
            await asyncio.sleep(POLL_SECONDS)
    finally:
        # Leave the line however we got here (no-op once the slot was taken).
        await asyncio.shield(redis_client.release_slot(wait_keys, token))


async def _wait_local(
    user_id: int, per_user: int, global_limit: int,
    max_wait: float, max_waiting_per_user: int, max_waiting: int,
    should_abort: Optional[Callable[[], Awaitable[bool]]],
) -> None:
    """The same wait, counted in this worker only (Redis unavailable).
    Not strictly ordered; bounded the same way. Returns with a local slot
    held, or raises OcrBusy / WaitAbandoned."""
    global _local_waiting_total
    full = _local_try(user_id, per_user, global_limit)
    if full is None:
        return
    if max_wait <= 0:
        raise OcrBusy(full, per_user if full == "user" else global_limit)
    if max_waiting_per_user > 0 and _local_waiting_users[user_id] >= max_waiting_per_user:
        raise OcrBusy("user_waiting", max_waiting_per_user)
    if max_waiting > 0 and _local_waiting_total >= max_waiting:
        raise OcrBusy("global_waiting", max_waiting)
    _local_waiting_users[user_id] += 1
    _local_waiting_total += 1
    try:
        deadline = time.monotonic() + max_wait
        while True:
            if should_abort is not None and await should_abort():
                raise WaitAbandoned()
            await asyncio.sleep(POLL_SECONDS)
            full = _local_try(user_id, per_user, global_limit)
            if full is None:
                return
            if time.monotonic() >= deadline:
                raise OcrBusy(full, per_user if full == "user" else global_limit, waited=True)
    finally:
        _local_waiting_users[user_id] -= 1
        if _local_waiting_users[user_id] <= 0:
            del _local_waiting_users[user_id]
        _local_waiting_total = max(0, _local_waiting_total - 1)


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
    max_wait = max(0.0, float(max_wait_seconds))
    max_waiting_per_user, max_waiting = max(0, int(max_waiting_per_user)), max(0, int(max_waiting))
    keys = (_user_key(user_id), _GLOBAL_KEY)
    token = uuid.uuid4().hex

    held_in_redis: Optional[bool]
    try:
        if max_wait <= 0:
            verdict = await redis_client.acquire_slot(
                keys[0], keys[1], token, per_user, global_limit, SLOT_TTL_SECONDS,
            )
            if verdict is not None and verdict != redis_client.SLOT_OK:
                raise _full(verdict, per_user, global_limit, waited=False)
            held_in_redis = None if verdict is None else True
        else:
            held_in_redis = await _wait_redis(
                user_id, token, per_user, global_limit, max_wait,
                max_waiting_per_user, max_waiting, should_abort,
            )
    except asyncio.CancelledError:
        # A script may have run before we were cancelled; don't leave the
        # slot held until it expires.
        await asyncio.shield(redis_client.release_slot(keys, token))
        raise
    except OcrBusy as busy:
        logger.warning("ocr_refused", scope=busy.scope, user_id=user_id,
                       limit=busy.limit, waited=busy.waited)
        raise

    if held_in_redis is None:
        # Redis unavailable — or its reply was lost after a script ran, so
        # also drop our token there (a no-op if Redis is really down).
        await redis_client.release_slot(keys, token)
        try:
            await _wait_local(user_id, per_user, global_limit, max_wait,
                              max_waiting_per_user, max_waiting, should_abort)
        except OcrBusy as busy:
            logger.warning("ocr_refused", scope=busy.scope, user_id=user_id,
                           limit=busy.limit, waited=busy.waited, fallback="local")
            raise
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
        # gather(return_exceptions) absorbs the heartbeat's own cancellation
        # without swallowing a cancellation aimed at us.
        await asyncio.gather(beat, return_exceptions=True)
        await asyncio.shield(redis_client.release_slot(keys, token))
