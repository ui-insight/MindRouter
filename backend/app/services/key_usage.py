"""API key usage counters (usage_count, last_used_at), written in batches.

Background (2026-11-05). Every completed or failed request used to bump its
key's row — `UPDATE api_keys SET usage_count = usage_count + 1` — as the
FIRST statement of its accounting transaction, so the row's exclusive lock
was held while the request row was updated and the response (often large)
was inserted. One key making many requests at once therefore serialized all
of its accounting on that row. When an OCR flood also starved the app of CPU,
those transactions slowed, ~110 of them queued on the one row, each holding a
pooled connection, and MariaDB ran out of connections for everyone.

Now the accounting transaction never touches api_keys. Each worker counts
uses in memory and writes them every FLUSH_INTERVAL_SECONDS, one short
single-row transaction per key. A busy key costs one UPDATE per worker per
interval instead of one per request, and a lock on it can no longer hold up
anyone's accounting.

Trade-off: last_used_at can lag by up to the interval, and a worker that
dies without its shutdown flush loses at most one interval of counts. Both
are display fields; tokens and quotas are accounted elsewhere, per request.
"""

import asyncio
from typing import Dict

from backend.app.logging_config import get_logger

logger = get_logger(__name__)

FLUSH_INTERVAL_SECONDS = 15
# Don't wait for the row at all: while an X-lock request waits, InnoDB queues
# new FK S-locks on that row behind it, so even a 1 s wait stalled new
# requests on a busy key (measured 0.81 s, review of PR #37). A busy row just
# fails with 1205 at once and the count is retried next interval.
LOCK_WAIT_SECONDS = 0

_pending: Dict[int, int] = {}


def record(api_key_id) -> None:
    """Count one use of a key (no I/O)."""
    if api_key_id is None:
        return
    _pending[api_key_id] = _pending.get(api_key_id, 0) + 1


def pending() -> Dict[int, int]:
    """Uses counted but not yet written (a copy)."""
    return dict(_pending)


async def flush() -> int:
    """Write the counted uses; returns how many keys were written.

    Counts that fail to write are put back and retried on the next flush.
    """
    if not _pending:
        return 0
    batch = dict(_pending)
    _pending.clear()

    from backend.app.db import crud
    from backend.app.db.session import isolated_async_session

    written = 0
    try:
        for key_id in sorted(batch):
            count = batch[key_id]
            try:
                async with isolated_async_session() as db:
                    # Give up at once if the row is busy (LOCK_WAIT_SECONDS).
                    await crud.add_api_key_usage(
                        db, key_id, count, lock_wait_seconds=LOCK_WAIT_SECONDS,
                    )
                    await db.commit()
                del batch[key_id]
                written += 1
            except Exception as e:
                logger.warning("key_usage_flush_failed", api_key_id=key_id,
                               count=count, error=type(e).__name__)
    finally:
        # Anything not written (a failure, or cancellation mid-flush at
        # shutdown) goes back to be written next time.
        for key_id, count in batch.items():
            _pending[key_id] = _pending.get(key_id, 0) + count
    return written


async def flush_loop() -> None:
    """Background task: flush every FLUSH_INTERVAL_SECONDS until cancelled."""
    while True:
        await asyncio.sleep(FLUSH_INTERVAL_SECONDS)
        try:
            await flush()
        except Exception:
            logger.exception("key_usage_flush_loop_error")
