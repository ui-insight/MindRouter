############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# retention.py: Tiered data retention — archive, cleanup,
#     stats, and browsing for the admin UI.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Tiered data retention service.

Tier 1 (App DB): configurable retention periods per category.
Tier 2 (Archive DB): longer retention.  Requests and chat keep full rows;
telemetry keeps 5-minute averages (rolled up continuously, see
archive_expired_telemetry).
"""

import asyncio
import json as _json
import os
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, NamedTuple, Optional

from sqlalchemy import bindparam, delete, func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.logging_config import get_logger

logger = get_logger(__name__)

# Telemetry table names are interpolated into DELETE statements below. Table
# identifiers cannot be passed as bind parameters, so they must come from this
# closed allowlist — asserted at each interpolation site — never from input.
_TELEMETRY_TABLES = frozenset(
    {"backend_telemetry", "gpu_device_telemetry", "node_telemetry"}
)

# Default retention config values
# Fallbacks used when an app_config row is missing.  The requests/chat/
# telemetry/interval/batch keys MUST match migration 029's seeds so the
# documented defaults hold on every install (the three-way docs/seed/code
# disagreement was a confirmed 2.9.4-audit doc bug); request_images,
# responses_store, and conversations are seeded nowhere and take their
# defaults from here until an admin saves the Retention page.
_DEFAULTS: dict[str, Any] = {
    "retention.requests.tier1_days": 90,
    "retention.requests.tier2_days": 730,
    "retention.chat.tier1_days": 90,
    "retention.chat.tier2_days": 730,
    "retention.telemetry.tier1_days": 30,
    "retention.telemetry.tier2_days": 0,
    "retention.request_images_days": 180,
    "retention.responses_store_days": 30,
    "retention.conversations_days": 0,
    # 0 = keep forever, matching conversations.  DLP alert rows carry masked
    # snippets of flagged content, so they are governed data rather than an
    # audit trail — but they were retained forever by omission until 2.9.9, and
    # a nonzero default here would mass-delete existing history on the first
    # cycle after deploy.  An admin opts in on the Retention page.
    "retention.dlp_alerts_days": 0,
    # 0 = keep forever.  Web-search logs are an audit trail (who sent what to
    # which third party), so the default preserves them; but they carry the
    # verbatim provider response and grow fastest of any audit table, which is
    # why they get their own dial rather than riding on requests.
    "retention.web_search_logs_days": 0,
    "retention.cleanup_interval": 3600,
    "retention.batch_size": 500,
}


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------


async def get_retention_config(db: AsyncSession) -> dict[str, Any]:
    """Load all retention keys from AppConfig, falling back to defaults."""
    from backend.app.db.models import AppConfig

    result = await db.execute(
        select(AppConfig.key, AppConfig.value).where(
            AppConfig.key.like("retention.%")
        )
    )
    rows = {r.key: r.value for r in result.all()}

    config: dict[str, Any] = {}
    for key, default in _DEFAULTS.items():
        raw = rows.get(key)
        if raw is not None:
            try:
                config[key] = _json.loads(raw)
            except (ValueError, TypeError):
                config[key] = default
        else:
            config[key] = default
    return config


async def save_retention_config(
    db: AsyncSession, updates: dict[str, Any]
) -> None:
    """Persist retention config values to AppConfig."""
    from backend.app.db import crud

    for key, value in updates.items():
        if key in _DEFAULTS:
            await crud.set_config(
                db, key, value,
                description=f"Data retention: {key}",
            )
    await db.flush()


# ------------------------------------------------------------------
# Archive helpers
# ------------------------------------------------------------------


def _row_to_dict(row) -> dict:
    """Convert a SQLAlchemy row to a plain dict of column values."""
    return {c.name: getattr(row, c.name) for c in row.__table__.columns}


# Retention rows are dumped here before being reloaded into the archive
# DB row-by-row.  Kept on partial failure for triage; removed on full
# success.  The dump sidesteps aio-libs/aiomysql#450 (broken multi-packet
# splitter on queries > 16 MB) by making every INSERT a single row.
_RETENTION_DUMP_DIR = "/tmp/mindrouter_retention"


def _jsonl_default(o: Any) -> Any:
    """JSON encoder for SQLAlchemy row values we dump to disk.

    Dict/list/str/int/float/bool/None serialize natively.  Datetime,
    date, Decimal, and bytes get wrapped in typed markers so the
    decoder can restore them exactly.
    """
    if isinstance(o, datetime):
        return {"__type__": "datetime", "value": o.isoformat()}
    if isinstance(o, date):
        return {"__type__": "date", "value": o.isoformat()}
    if isinstance(o, Decimal):
        return {"__type__": "decimal", "value": str(o)}
    if isinstance(o, (bytes, bytearray)):
        return {"__type__": "bytes", "value": bytes(o).hex()}
    raise TypeError(f"Not JSON-serializable: {type(o).__name__}")


def _jsonl_object_hook(d: dict) -> Any:
    t = d.get("__type__")
    if t == "datetime":
        return datetime.fromisoformat(d["value"])
    if t == "date":
        return date.fromisoformat(d["value"])
    if t == "decimal":
        return Decimal(d["value"])
    if t == "bytes":
        return bytes.fromhex(d["value"])
    return d


async def _bulk_insert_ignore(
    archive_db: AsyncSession,
    model_class,
    rows: list[dict],
) -> tuple[int, list[dict]]:
    """Insert rows into archive table via single-row INSERT IGNORE.

    Rows are dumped to a JSONL file under ``_RETENTION_DUMP_DIR`` first,
    then reloaded one at a time.  Each row is wrapped in a SAVEPOINT so
    an individual failure (oversized payload, encoding issue, etc.) is
    logged and skipped instead of aborting the whole batch.

    On full success the dump file is deleted.  On any skipped row the
    file is retained for forensics.

    Returns ``(inserted_count, skipped_rows)``.
    """
    if not rows:
        return 0, []

    table = model_class.__table__
    now = datetime.now(timezone.utc)

    # The dump carries plaintext prompts/completions/IPs, so it must never be
    # world-readable: restrict the directory to 0o700 and each file to 0o600.
    # os.makedirs honours the process umask, so chmod the directory explicitly
    # in case it (or an ancestor left by an earlier looser run) is too open.
    os.makedirs(_RETENTION_DUMP_DIR, exist_ok=True)
    try:
        os.chmod(_RETENTION_DUMP_DIR, 0o700)
    except OSError:
        pass
    dump_path = os.path.join(
        _RETENTION_DUMP_DIR,
        f"{table.name}_{now.strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}.jsonl",
    )

    # Open with O_EXCL and mode 0o600 so the file is created private from the
    # start (never a world-readable window between open and a later chmod).
    fd = os.open(dump_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)

    # Deliberately retained on partial failure for forensics; removed in the
    # finally on full success OR on an unexpected error, so plaintext PII is
    # never left on disk except in the controlled skip-for-triage case.
    retain = False
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for row in rows:
                row["archived_at"] = now
                fh.write(_json.dumps(row, default=_jsonl_default) + "\n")

        logger.info(
            "retention_dump_written",
            table=table.name,
            path=dump_path,
            rows=len(rows),
        )

        total = 0
        skipped: list[dict] = []
        with open(dump_path, "r", encoding="utf-8") as fh:
            for line in fh:
                row = _json.loads(line, object_hook=_jsonl_object_hook)
                sp = await archive_db.begin_nested()
                try:
                    stmt = table.insert().prefix_with("IGNORE").values(**row)
                    result = await archive_db.execute(stmt)
                    total += result.rowcount
                    await sp.commit()
                except Exception as exc:
                    await sp.rollback()
                    logger.warning(
                        "retention_row_insert_failed",
                        table=table.name,
                        row_id=row.get("id"),
                        error=str(exc),
                    )
                    skipped.append(row)

        if skipped:
            retain = True
            logger.warning(
                "retention_dump_retained",
                path=dump_path,
                inserted=total,
                skipped=len(skipped),
            )

        return total, skipped
    finally:
        if not retain:
            try:
                os.remove(dump_path)
            except OSError:
                pass


# ------------------------------------------------------------------
# Pre-deletion aggregate snapshots
# ------------------------------------------------------------------


_SNAPSHOT_DEADLOCK_RETRIES = 3
_SNAPSHOT_DEADLOCK_DELAY = 0.5  # seconds, multiplied by attempt number


async def _snapshot_request_aggregates(
    db: AsyncSession, request_ids: list[int]
) -> None:
    """Accumulate per-user, per-model, and per-api-key token/count offsets
    for a batch of requests that are about to be deleted.

    Retries on deadlock (error 1213) — concurrent discovery and request
    processing also touch ``quotas``, so transient deadlocks are expected.
    A deadlock rolls back the entire transaction, so a full retry from
    the top is safe (no double-counting).
    """
    if not request_ids:
        return

    from sqlalchemy.exc import OperationalError

    for attempt in range(_SNAPSHOT_DEADLOCK_RETRIES + 1):
        try:
            await _snapshot_request_aggregates_inner(db, request_ids)
            return
        except OperationalError as exc:
            if "1213" in str(exc) and attempt < _SNAPSHOT_DEADLOCK_RETRIES:
                logger.warning(
                    "retention_snapshot_deadlock_retry",
                    attempt=attempt + 1,
                    max_retries=_SNAPSHOT_DEADLOCK_RETRIES,
                )
                await asyncio.sleep(
                    _SNAPSHOT_DEADLOCK_DELAY * (attempt + 1)
                )
                continue
            raise


async def _snapshot_request_aggregates_inner(
    db: AsyncSession, request_ids: list[int]
) -> None:
    """Inner implementation — called by the retry wrapper above."""

    # Per-user aggregates → quotas.archived_*
    user_agg = await db.execute(
        text(
            "SELECT user_id,"
            " COALESCE(SUM(total_tokens), 0),"
            " COALESCE(SUM(prompt_tokens), 0),"
            " COALESCE(SUM(completion_tokens), 0),"
            " COUNT(*)"
            " FROM requests"
            " WHERE id IN :ids"
            " GROUP BY user_id"
        ).bindparams(bindparam("ids", expanding=True)),
        {"ids": request_ids},
    )
    for user_id, total_tok, prompt_tok, comp_tok, req_count in user_agg.all():
        await db.execute(
            text(
                "UPDATE quotas SET"
                " archived_total_tokens = archived_total_tokens + :total_tok,"
                " archived_prompt_tokens = archived_prompt_tokens + :prompt_tok,"
                " archived_completion_tokens = archived_completion_tokens + :comp_tok,"
                " archived_request_count = archived_request_count + :req_count"
                " WHERE user_id = :uid"
            ),
            {
                "uid": user_id,
                "total_tok": int(total_tok),
                "prompt_tok": int(prompt_tok),
                "comp_tok": int(comp_tok),
                "req_count": int(req_count),
            },
        )

    # Per-model aggregates → model_archived_stats
    model_agg = await db.execute(
        text(
            "SELECT model,"
            " COALESCE(SUM(total_tokens), 0),"
            " COALESCE(SUM(prompt_tokens), 0),"
            " COALESCE(SUM(completion_tokens), 0),"
            " COUNT(*)"
            " FROM requests"
            " WHERE id IN :ids"
            " GROUP BY model"
        ).bindparams(bindparam("ids", expanding=True)),
        {"ids": request_ids},
    )
    for model_name, total_tok, prompt_tok, comp_tok, req_count in model_agg.all():
        await db.execute(
            text(
                "INSERT INTO model_archived_stats"
                " (model, archived_total_tokens, archived_prompt_tokens,"
                "  archived_completion_tokens, archived_request_count)"
                " VALUES (:model, :total_tok, :prompt_tok, :comp_tok, :req_count)"
                " ON DUPLICATE KEY UPDATE"
                " archived_total_tokens = archived_total_tokens + VALUES(archived_total_tokens),"
                " archived_prompt_tokens = archived_prompt_tokens + VALUES(archived_prompt_tokens),"
                " archived_completion_tokens = archived_completion_tokens + VALUES(archived_completion_tokens),"
                " archived_request_count = archived_request_count + VALUES(archived_request_count)"
            ),
            {
                "model": model_name,
                "total_tok": int(total_tok),
                "prompt_tok": int(prompt_tok),
                "comp_tok": int(comp_tok),
                "req_count": int(req_count),
            },
        )

    # Per-api-key aggregates → api_keys.archived_*
    key_agg = await db.execute(
        text(
            "SELECT api_key_id,"
            " COALESCE(SUM(total_tokens), 0),"
            " COALESCE(SUM(prompt_tokens), 0),"
            " COALESCE(SUM(completion_tokens), 0),"
            " COUNT(*)"
            " FROM requests"
            " WHERE id IN :ids AND api_key_id IS NOT NULL"
            " GROUP BY api_key_id"
        ).bindparams(bindparam("ids", expanding=True)),
        {"ids": request_ids},
    )
    for key_id, total_tok, prompt_tok, comp_tok, req_count in key_agg.all():
        await db.execute(
            text(
                "UPDATE api_keys SET"
                " archived_total_tokens = archived_total_tokens + :total_tok,"
                " archived_prompt_tokens = archived_prompt_tokens + :prompt_tok,"
                " archived_completion_tokens = archived_completion_tokens + :comp_tok,"
                " archived_request_count = archived_request_count + :req_count"
                " WHERE id = :key_id"
            ),
            {
                "key_id": key_id,
                "total_tok": int(total_tok),
                "prompt_tok": int(prompt_tok),
                "comp_tok": int(comp_tok),
                "req_count": int(req_count),
            },
        )


# ------------------------------------------------------------------
# Tier 1: archive + delete from app DB
# ------------------------------------------------------------------


async def _detach_request_references(
    app_db: AsyncSession, request_ids: list[int]
) -> None:
    """SET NULL the nullable audit back-references to requests about to
    be deleted (user_images, video_jobs, stored_responses, dlp_alerts,
    web_search_logs).  Their FKs are RESTRICT with no cascade, so deleting a
    referenced request would otherwise raise IntegrityError."""
    if not request_ids:
        return
    from backend.app.db.models import (
        DlpAlert,
        StoredResponse,
        UserImage,
        VideoJob,
        WebSearchLog,
    )

    await app_db.execute(
        update(UserImage)
        .where(UserImage.request_id.in_(request_ids))
        .values(request_id=None)
    )
    await app_db.execute(
        update(VideoJob)
        .where(VideoJob.request_id.in_(request_ids))
        .values(request_id=None)
    )
    await app_db.execute(
        update(StoredResponse)
        .where(StoredResponse.request_id.in_(request_ids))
        .values(request_id=None)
    )
    # A web-search log outlives the inference request that triggered it: the
    # third-party call is the auditable event, not the request row.
    await app_db.execute(
        update(WebSearchLog)
        .where(WebSearchLog.request_id.in_(request_ids))
        .values(request_id=None)
    )
    await app_db.execute(
        update(DlpAlert)
        .where(DlpAlert.request_id.in_(request_ids))
        .values(request_id=None)
    )


async def archive_expired_requests(
    app_db: AsyncSession,
    archive_db: AsyncSession,
    cutoff: datetime,
    batch_size: int,
) -> dict[str, int]:
    """Archive requests (and children) older than cutoff.

    Returns counts of archived rows per table.
    Uses a minimum batch size of 5000 to handle large backlogs
    efficiently.  Each batch is committed independently so that
    progress is durable and InnoDB lock duration stays bounded.
    """
    from backend.app.db.models import (
        Artifact,
        Request,
        Response,
        SchedulerDecision,
    )
    from backend.app.db.archive_models import (
        ArchivedArtifact,
        ArchivedRequest,
        ArchivedResponse,
        ArchivedSchedulerDecision,
    )

    counts = {
        "requests": 0,
        "responses": 0,
        "scheduler_decisions": 0,
        "artifacts": 0,
    }

    # Outer batch size is bounded to keep per-batch memory manageable:
    # each Response row can carry a multi-KB JSON payload, so loading
    # thousands at once consumed > 15 GB of RAM in practice.  The inner
    # ``_bulk_insert_ignore`` chunks the VALUES list separately.
    effective_batch = batch_size if batch_size and batch_size > 0 else 500

    while True:
        # Fetch a batch of expired request IDs (ID-only, cheap)
        result = await app_db.execute(
            select(Request.id)
            .where(Request.created_at < cutoff)
            .limit(effective_batch)
        )
        request_ids = [r[0] for r in result.all()]
        if not request_ids:
            break

        # Fetch full rows for each child table
        resp_result = await app_db.execute(
            select(Response).where(Response.request_id.in_(request_ids))
        )
        responses = list(resp_result.scalars().all())

        sched_result = await app_db.execute(
            select(SchedulerDecision).where(
                SchedulerDecision.request_id.in_(request_ids)
            )
        )
        sched_rows = list(sched_result.scalars().all())

        art_result = await app_db.execute(
            select(Artifact).where(Artifact.request_id.in_(request_ids))
        )
        art_rows = list(art_result.scalars().all())

        req_result = await app_db.execute(
            select(Request).where(Request.id.in_(request_ids))
        )
        req_rows = list(req_result.scalars().all())

        # Archive: bulk insert into archive DB.  Collect IDs of any
        # parent requests whose archival failed (either directly or via
        # a failed child row) — we must NOT delete those from the live
        # DB or we'd silently lose data.
        skipped_req_ids: set[int] = set()

        resp_inserted, resp_skipped = await _bulk_insert_ignore(
            archive_db, ArchivedResponse,
            [_row_to_dict(r) for r in responses],
        )
        counts["responses"] += resp_inserted
        for r in resp_skipped:
            if r.get("request_id") is not None:
                skipped_req_ids.add(r["request_id"])

        sched_inserted, sched_skipped = await _bulk_insert_ignore(
            archive_db, ArchivedSchedulerDecision,
            [_row_to_dict(r) for r in sched_rows],
        )
        counts["scheduler_decisions"] += sched_inserted
        for r in sched_skipped:
            if r.get("request_id") is not None:
                skipped_req_ids.add(r["request_id"])

        art_inserted, art_skipped = await _bulk_insert_ignore(
            archive_db, ArchivedArtifact,
            [_row_to_dict(r) for r in art_rows],
        )
        counts["artifacts"] += art_inserted
        for r in art_skipped:
            if r.get("request_id") is not None:
                skipped_req_ids.add(r["request_id"])

        req_inserted, req_skipped = await _bulk_insert_ignore(
            archive_db, ArchivedRequest,
            [_row_to_dict(r) for r in req_rows],
        )
        counts["requests"] += req_inserted
        for r in req_skipped:
            if r.get("id") is not None:
                skipped_req_ids.add(r["id"])

        await archive_db.flush()

        safe_ids = [rid for rid in request_ids if rid not in skipped_req_ids]

        if skipped_req_ids:
            logger.warning(
                "retention_requests_archive_skipped",
                count=len(skipped_req_ids),
                request_ids=sorted(skipped_req_ids)[:50],
            )

        if not safe_ids:
            # Whole batch failed to archive — break to avoid an infinite
            # loop over the same rows.  Dump files retained on disk.
            logger.error(
                "retention_batch_fully_skipped",
                batch_size=len(request_ids),
            )
            break

        # Snapshot per-user and per-model aggregates before deletion
        await _snapshot_request_aggregates(app_db, safe_ids)

        # Detach nullable audit back-references before deleting the
        # requests they point at (RESTRICT FKs, no cascade).
        await _detach_request_references(app_db, safe_ids)

        # Delete from app DB in FK order
        await app_db.execute(
            delete(SchedulerDecision).where(
                SchedulerDecision.request_id.in_(safe_ids)
            )
        )
        await app_db.execute(
            delete(Response).where(Response.request_id.in_(safe_ids))
        )
        await app_db.execute(
            delete(Artifact).where(Artifact.request_id.in_(safe_ids))
        )
        await app_db.execute(
            delete(Request).where(Request.id.in_(safe_ids))
        )
        await app_db.flush()
        await app_db.commit()
        await archive_db.commit()

        logger.info(
            "retention_requests_batch",
            deleted=len(safe_ids),
            total=counts["requests"],
        )

    return counts


async def archive_expired_chats(
    app_db: AsyncSession,
    archive_db: AsyncSession,
    cutoff: datetime,
    batch_size: int,
) -> dict[str, int]:
    """Archive chat conversations (and children) older than cutoff.

    Returns counts of archived rows per table.
    """
    from backend.app.db.models import (
        ChatAttachment,
        ChatConversation,
        ChatMessage,
    )
    from backend.app.db.archive_models import (
        ArchivedChatAttachment,
        ArchivedChatConversation,
        ArchivedChatMessage,
    )
    from backend.app.db.chat_crud import _remove_attachment_files

    counts = {
        "chat_conversations": 0,
        "chat_messages": 0,
        "chat_attachments": 0,
    }

    while True:
        result = await app_db.execute(
            select(ChatConversation)
            .where(ChatConversation.updated_at < cutoff)
            .limit(batch_size)
        )
        batch = list(result.scalars().all())
        if not batch:
            break

        conv_ids = [c.id for c in batch]

        # Fetch messages
        msg_result = await app_db.execute(
            select(ChatMessage).where(ChatMessage.conversation_id.in_(conv_ids))
        )
        messages = list(msg_result.scalars().all())
        msg_ids = [m.id for m in messages]

        # Fetch attachments
        attachments = []
        if msg_ids:
            att_result = await app_db.execute(
                select(ChatAttachment).where(ChatAttachment.message_id.in_(msg_ids))
            )
            attachments = list(att_result.scalars().all())

        # Remove attachment files from disk
        for att in attachments:
            _remove_attachment_files(att)

        # Archive to archive DB.  Track any conversation whose archival
        # failed (directly or via a failed message/attachment) so we
        # don't delete its rows from the live DB.
        skipped_conv_ids: set[int] = set()
        msg_to_conv: dict[int, int] = {m.id: m.conversation_id for m in messages}

        att_inserted, att_skipped = await _bulk_insert_ignore(
            archive_db, ArchivedChatAttachment,
            [_row_to_dict(a) for a in attachments],
        )
        counts["chat_attachments"] += att_inserted
        for r in att_skipped:
            conv_id = msg_to_conv.get(r.get("message_id"))
            if conv_id is not None:
                skipped_conv_ids.add(conv_id)

        msg_inserted, msg_skipped = await _bulk_insert_ignore(
            archive_db, ArchivedChatMessage,
            [_row_to_dict(m) for m in messages],
        )
        counts["chat_messages"] += msg_inserted
        for r in msg_skipped:
            if r.get("conversation_id") is not None:
                skipped_conv_ids.add(r["conversation_id"])

        conv_inserted, conv_skipped = await _bulk_insert_ignore(
            archive_db, ArchivedChatConversation,
            [_row_to_dict(c) for c in batch],
        )
        counts["chat_conversations"] += conv_inserted
        for r in conv_skipped:
            if r.get("id") is not None:
                skipped_conv_ids.add(r["id"])

        await archive_db.flush()

        safe_conv_ids = [cid for cid in conv_ids if cid not in skipped_conv_ids]
        safe_msg_ids = [mid for mid in msg_ids if msg_to_conv.get(mid) in safe_conv_ids]

        if skipped_conv_ids:
            logger.warning(
                "retention_chat_archive_skipped",
                count=len(skipped_conv_ids),
                conversation_ids=sorted(skipped_conv_ids)[:50],
            )

        if not safe_conv_ids:
            logger.error(
                "retention_chat_batch_fully_skipped",
                batch_size=len(conv_ids),
            )
            break

        # Delete from app DB in FK order
        if safe_msg_ids:
            await app_db.execute(
                delete(ChatAttachment).where(
                    ChatAttachment.message_id.in_(safe_msg_ids)
                )
            )
        await app_db.execute(
            delete(ChatMessage).where(
                ChatMessage.conversation_id.in_(safe_conv_ids)
            )
        )
        await app_db.execute(
            delete(ChatConversation).where(
                ChatConversation.id.in_(safe_conv_ids)
            )
        )
        await app_db.flush()
        await app_db.commit()
        await archive_db.commit()

    return counts


# ------------------------------------------------------------------
# Telemetry: continuous 5-minute rollups into the archive DB
# ------------------------------------------------------------------
#
# Raw telemetry (one row per GPU, node and backend every few seconds, ~35M
# rows a month) is far too much to archive row for row.  What is worth keeping
# for the long term is the shape of the series, power above all, so when an
# archive DB is configured and retention.telemetry.tier2_days > 0 the
# retention cycle aggregates raw readings into 5-minute rows (avg/min/max and
# sample counts, archive_models.Archived*Telemetry5m) kept for tier2_days,
# while the raw rows stay in the app DB for tier1_days.
#
# Rollups run on every cycle, independent of tier 1, so the archive trails the
# live tables by about one cycle rather than by tier1_days, and tier 1 = 0
# (never delete raw rows) still archives.  A raw row is deleted only after the
# bucket it falls in has been committed to the archive: the retention cycle
# caps its delete at the rollup it just ran.  The cycle is the ONLY deleter of
# raw telemetry (the backend registry used to run its own 30-day loop).
#
# Progress is read back from the archive itself: the newest bucket_start in a
# rollup table is that table's watermark, and the next pass resumes after it.
# That is only sound because each window is written in ONE archive
# transaction.  A failure mid-window rolls the whole window back, so the
# watermark never reflects a window that was written for some entities (some
# GPUs, say) but not for the rest.
#
# A pass never starts before now - tier2_days.  Older buckets are deleted by
# the same cycle's purge, so writing them is wasted work, and worse: when they
# are all a pass wrote, the purge empties the rollup table, the watermark is
# lost, and the next pass starts over at the oldest live reading, every cycle,
# without ever reaching recent buckets.

_ROLLUP_BUCKET = timedelta(minutes=5)
# A bucket is complete once it ended at least this long ago.  Readings are
# stamped by the DB's NOW() when inserted, so a reading lands in a bucket that
# has already been rolled up only if its writing transaction stays open longer
# than this.
_ROLLUP_SAFETY_MARGIN = timedelta(minutes=10)
# Aggregate and commit in 6-hour windows.  Each GROUP BY then reads a bounded
# slice of the timestamp index (~200k rows of gpu_device_telemetry), each
# archive transaction stays short, and a failure loses one window of work.
_ROLLUP_WINDOW = timedelta(hours=6)
# At most this much history per table per cycle, so the first-deploy backfill
# of 30-90 days spreads over a few hourly cycles instead of one long run that
# holds the retention lock.  A manual purge lifts the cap (max_span=None): it
# is about to delete, so it catches up in one pass first.
_ROLLUP_MAX_SPAN_PER_CYCLE = timedelta(days=14)
# Rows per INSERT IGNORE call.  The rows go in as executemany parameters of
# the single-row statement, which SQLAlchemy compiles once and caches; the
# driver packs them into multi-row INSERTs of at most ~1 MB, far below
# max_allowed_packet and the 16 MB multi-packet bug that _bulk_insert_ignore
# works around (aiomysql#450).  A multi-VALUES statement (.values(rows)) has
# no cache key: every batch would be compiled from scratch, ~100 ms of CPU
# for 1000 GPU rows, on the event loop of a worker that is serving requests.
_ROLLUP_INSERT_BATCH = 1000
# Rollup rows per DELETE when tier 2 expires them.  Shortening tier 2 can
# expire millions of rows at once.
_ROLLUP_PURGE_BATCH = 10000
# Seek origin for a table that has never been rolled up, and the epoch of the
# bucket expression below.  No reading predates it.
_ROLLUP_EPOCH = datetime(2000, 1, 1)


@dataclass(frozen=True)
class _RollupSpec:
    """The static SQL that rolls one live telemetry table into the archive.

    Every statement is written out in full, with no interpolated
    identifiers, so each can be read, grepped and EXPLAINed as it runs.

    All three rollup statements bucket readings with the same expression::

        TIMESTAMPADD(MINUTE,
            FLOOR(TIMESTAMPDIFF(MINUTE, TIMESTAMP'2000-01-01 00:00:00',
                                t.timestamp) / 5) * 5,
            TIMESTAMP'2000-01-01 00:00:00')

    It is plain DATETIME arithmetic against a fixed epoch, so it does not
    depend on the session time_zone, which UNIX_TIMESTAMP()/FROM_UNIXTIME()
    convert through.  The epoch is a TIMESTAMP literal rather than a string
    because TIMESTAMPADD over a string returns VARCHAR, and over a DATETIME
    returns DATETIME.  Readings are aggregated in a derived table and the
    dimension table is joined afterwards, once per (entity, bucket) rather
    than once per reading.
    """

    # Live app-DB table; also this table's key in the cycle summary.
    table: str
    # Oldest reading at or after :after.  The pass starts there, so it can
    # jump over a stretch with no telemetry (a fleet outage, or a pause in
    # archiving) instead of rescanning empty windows every cycle.
    next_reading_sql: str
    # One row per (entity, 5-minute bucket) for readings in [:start, :end).
    # The column aliases are the archive model's column names.
    rollup_sql: str
    # Batched tier-1 delete.  The caller caps :cutoff at the rollup frontier.
    delete_sql: str


_NODE_ROLLUP = _RollupSpec(
    table="node_telemetry",
    next_reading_sql=(
        "SELECT MIN(timestamp) FROM node_telemetry WHERE timestamp >= :after"
    ),
    rollup_sql="""
        SELECT
            a.node_id AS node_id,
            n.name AS node_name,
            a.bucket_start AS bucket_start,
            a.sample_count AS sample_count,
            a.server_power_watts_avg AS server_power_watts_avg,
            a.server_power_watts_min AS server_power_watts_min,
            a.server_power_watts_max AS server_power_watts_max,
            a.server_power_samples AS server_power_samples,
            a.gpu_power_watts_avg AS gpu_power_watts_avg,
            a.gpu_power_watts_min AS gpu_power_watts_min,
            a.gpu_power_watts_max AS gpu_power_watts_max,
            a.gpu_power_samples AS gpu_power_samples
        FROM (
            SELECT
                t.node_id AS node_id,
                TIMESTAMPADD(MINUTE,
                    FLOOR(TIMESTAMPDIFF(MINUTE, TIMESTAMP'2000-01-01 00:00:00',
                                        t.timestamp) / 5) * 5,
                    TIMESTAMP'2000-01-01 00:00:00') AS bucket_start,
                COUNT(*) AS sample_count,
                AVG(t.server_power_watts) AS server_power_watts_avg,
                MIN(t.server_power_watts) AS server_power_watts_min,
                MAX(t.server_power_watts) AS server_power_watts_max,
                COUNT(t.server_power_watts) AS server_power_samples,
                AVG(t.gpu_power_watts) AS gpu_power_watts_avg,
                MIN(t.gpu_power_watts) AS gpu_power_watts_min,
                MAX(t.gpu_power_watts) AS gpu_power_watts_max,
                COUNT(t.gpu_power_watts) AS gpu_power_samples
            FROM node_telemetry t
            WHERE t.timestamp >= :start AND t.timestamp < :end
            GROUP BY t.node_id, bucket_start
        ) a
        LEFT JOIN nodes n ON n.id = a.node_id
    """,
    delete_sql=(
        "DELETE FROM node_telemetry WHERE timestamp < :cutoff LIMIT :batch"
    ),
)

_GPU_DEVICE_ROLLUP = _RollupSpec(
    table="gpu_device_telemetry",
    next_reading_sql=(
        "SELECT MIN(timestamp) FROM gpu_device_telemetry"
        " WHERE timestamp >= :after"
    ),
    rollup_sql="""
        SELECT
            a.gpu_device_id AS gpu_device_id,
            g.node_id AS node_id,
            g.gpu_index AS gpu_index,
            g.name AS gpu_name,
            a.bucket_start AS bucket_start,
            a.sample_count AS sample_count,
            a.power_draw_watts_avg AS power_draw_watts_avg,
            a.power_draw_watts_min AS power_draw_watts_min,
            a.power_draw_watts_max AS power_draw_watts_max,
            a.power_samples AS power_samples,
            a.utilization_gpu_avg AS utilization_gpu_avg,
            a.utilization_gpu_max AS utilization_gpu_max,
            a.utilization_memory_avg AS utilization_memory_avg,
            a.memory_used_gb_avg AS memory_used_gb_avg,
            a.memory_used_gb_max AS memory_used_gb_max,
            a.temperature_gpu_avg AS temperature_gpu_avg,
            a.temperature_gpu_max AS temperature_gpu_max,
            a.clock_sm_mhz_avg AS clock_sm_mhz_avg
        FROM (
            SELECT
                t.gpu_device_id AS gpu_device_id,
                TIMESTAMPADD(MINUTE,
                    FLOOR(TIMESTAMPDIFF(MINUTE, TIMESTAMP'2000-01-01 00:00:00',
                                        t.timestamp) / 5) * 5,
                    TIMESTAMP'2000-01-01 00:00:00') AS bucket_start,
                COUNT(*) AS sample_count,
                AVG(t.power_draw_watts) AS power_draw_watts_avg,
                MIN(t.power_draw_watts) AS power_draw_watts_min,
                MAX(t.power_draw_watts) AS power_draw_watts_max,
                COUNT(t.power_draw_watts) AS power_samples,
                AVG(t.utilization_gpu) AS utilization_gpu_avg,
                MAX(t.utilization_gpu) AS utilization_gpu_max,
                AVG(t.utilization_memory) AS utilization_memory_avg,
                AVG(t.memory_used_gb) AS memory_used_gb_avg,
                MAX(t.memory_used_gb) AS memory_used_gb_max,
                AVG(t.temperature_gpu) AS temperature_gpu_avg,
                MAX(t.temperature_gpu) AS temperature_gpu_max,
                AVG(t.clock_sm_mhz) AS clock_sm_mhz_avg
            FROM gpu_device_telemetry t
            WHERE t.timestamp >= :start AND t.timestamp < :end
            GROUP BY t.gpu_device_id, bucket_start
        ) a
        LEFT JOIN gpu_devices g ON g.id = a.gpu_device_id
    """,
    delete_sql=(
        "DELETE FROM gpu_device_telemetry WHERE timestamp < :cutoff LIMIT :batch"
    ),
)

_BACKEND_ROLLUP = _RollupSpec(
    table="backend_telemetry",
    next_reading_sql=(
        "SELECT MIN(timestamp) FROM backend_telemetry WHERE timestamp >= :after"
    ),
    rollup_sql="""
        SELECT
            a.backend_id AS backend_id,
            b.name AS backend_name,
            b.node_id AS node_id,
            a.bucket_start AS bucket_start,
            a.sample_count AS sample_count,
            a.gpu_power_draw_watts_avg AS gpu_power_draw_watts_avg,
            a.gpu_power_draw_watts_max AS gpu_power_draw_watts_max,
            a.power_samples AS power_samples,
            a.gpu_utilization_avg AS gpu_utilization_avg,
            a.gpu_memory_used_gb_avg AS gpu_memory_used_gb_avg,
            a.active_requests_avg AS active_requests_avg,
            a.active_requests_max AS active_requests_max,
            a.queued_requests_avg AS queued_requests_avg,
            a.queued_requests_max AS queued_requests_max,
            a.requests_per_second_avg AS requests_per_second_avg
        FROM (
            SELECT
                t.backend_id AS backend_id,
                TIMESTAMPADD(MINUTE,
                    FLOOR(TIMESTAMPDIFF(MINUTE, TIMESTAMP'2000-01-01 00:00:00',
                                        t.timestamp) / 5) * 5,
                    TIMESTAMP'2000-01-01 00:00:00') AS bucket_start,
                COUNT(*) AS sample_count,
                AVG(t.gpu_power_draw_watts) AS gpu_power_draw_watts_avg,
                MAX(t.gpu_power_draw_watts) AS gpu_power_draw_watts_max,
                COUNT(t.gpu_power_draw_watts) AS power_samples,
                AVG(t.gpu_utilization) AS gpu_utilization_avg,
                AVG(t.gpu_memory_used_gb) AS gpu_memory_used_gb_avg,
                AVG(t.active_requests) AS active_requests_avg,
                MAX(t.active_requests) AS active_requests_max,
                AVG(t.queued_requests) AS queued_requests_avg,
                MAX(t.queued_requests) AS queued_requests_max,
                AVG(t.requests_per_second) AS requests_per_second_avg
            FROM backend_telemetry t
            WHERE t.timestamp >= :start AND t.timestamp < :end
            GROUP BY t.backend_id, bucket_start
        ) a
        LEFT JOIN backends b ON b.id = a.backend_id
    """,
    delete_sql=(
        "DELETE FROM backend_telemetry WHERE timestamp < :cutoff LIMIT :batch"
    ),
)

# Node first: server power is the series the archive exists for.
_ROLLUP_SPECS = (_NODE_ROLLUP, _GPU_DEVICE_ROLLUP, _BACKEND_ROLLUP)


def _rollup_models() -> dict[str, Any]:
    """Map each live telemetry table to its 5-minute archive model.

    Imported at call time, like every archive model in this module.
    """
    from backend.app.db.archive_models import (
        ArchivedBackendTelemetry5m,
        ArchivedGPUDeviceTelemetry5m,
        ArchivedNodeTelemetry5m,
    )

    return {
        "node_telemetry": ArchivedNodeTelemetry5m,
        "gpu_device_telemetry": ArchivedGPUDeviceTelemetry5m,
        "backend_telemetry": ArchivedBackendTelemetry5m,
    }


def _as_naive_utc(ts: datetime) -> datetime:
    """Convert an aware datetime to the naive UTC that DATETIME columns hold.

    A naive value is returned unchanged.
    """
    if ts.tzinfo is not None:
        ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
    return ts


def _floor_to_bucket(ts: datetime) -> datetime:
    """Start of the 5-minute bucket that contains ``ts`` (naive UTC)."""
    ts = _as_naive_utc(ts)
    return ts.replace(minute=ts.minute - ts.minute % 5, second=0, microsecond=0)


class _RollupPlan(NamedTuple):
    """One table's rollup pass, as planned by _plan_rollup_windows."""

    # Rollup frontier before this pass: every live reading older than it has
    # already been aggregated, never existed, or is older than tier 2 keeps
    # (its average would be purged at once).  None only when nothing anchors
    # it: an empty archive, no tier-2 limit and no live reading.
    start: Optional[datetime]
    # Contiguous, bucket-aligned [from, to) windows to aggregate, in order.
    windows: list[tuple[datetime, datetime]]
    # Last complete bucket boundary: floor(now - margin).
    horizon: datetime


def _plan_rollup_windows(
    watermark: Optional[datetime],
    next_reading: Optional[datetime],
    now: datetime,
    *,
    keep_since: Optional[datetime] = None,
    window: timedelta = _ROLLUP_WINDOW,
    max_span: Optional[timedelta] = _ROLLUP_MAX_SPAN_PER_CYCLE,
    margin: timedelta = _ROLLUP_SAFETY_MARGIN,
) -> _RollupPlan:
    """Plan one table's rollup pass.  Pure: no I/O.

    ``watermark`` is the newest bucket_start already in the archive (None
    if the rollup table is empty).  ``next_reading`` is the oldest live
    reading at or after the point the pass resumes from (below; None if
    there is none).  ``now`` comes from the clock that stamped the readings.
    ``keep_since`` is where tier 2 begins, now - tier2_days on that clock
    (None = averages never expire).

    The pass resumes at the watermark's next bucket, or at ``keep_since``'s
    bucket when that is later or the archive is empty: an older bucket is
    deleted by the same cycle's purge, so it is never written.  The start
    then moves forward to ``next_reading``'s bucket when the readings begin
    later, because nothing in between exists to aggregate.  Windows run up
    to the horizon and stop after ``max_span`` in total (None = no cap);
    ``window`` and ``max_span`` must be whole multiples of 5 minutes so
    every window stays bucket-aligned.
    """
    horizon = _floor_to_bucket(_as_naive_utc(now) - margin)
    resume = None
    if watermark is not None:
        resume = _floor_to_bucket(watermark) + _ROLLUP_BUCKET
    if keep_since is not None:
        oldest_kept = _floor_to_bucket(keep_since)
        if resume is None or resume < oldest_kept:
            resume = oldest_kept
    if next_reading is None:
        return _RollupPlan(resume, [], horizon)

    start = _floor_to_bucket(next_reading)
    if resume is not None and start < resume:
        # The caller seeks from `resume`, so this only guards a misuse:
        # never re-plan buckets the archive already holds, or ones that
        # tier 2 does not keep.
        start = resume

    stop = horizon if max_span is None else min(horizon, start + max_span)
    windows: list[tuple[datetime, datetime]] = []
    cursor = start
    while cursor < stop:
        upper = min(cursor + window, stop)
        windows.append((cursor, upper))
        cursor = upper
    return _RollupPlan(start, windows, horizon)


def _effective_delete_cutoff(
    cutoff: Optional[datetime], rolled_through: Optional[datetime]
) -> Optional[datetime]:
    """Delete raw readings older than this; None means delete nothing.

    The tier-1 cutoff, capped at the table's rollup frontier so that no
    reading is deleted before its bucket is in the archive.  None when
    there is no tier-1 cutoff (tier 1 = 0, or a rollup-only call) or no
    known frontier.
    """
    if cutoff is None or rolled_through is None:
        return None
    return min(_as_naive_utc(cutoff), _as_naive_utc(rolled_through))


def _iso(ts: Optional[datetime]) -> Optional[str]:
    return ts.isoformat() if ts is not None else None


async def _rollback_quietly(db: AsyncSession) -> None:
    """Roll back after a failure without masking the error that caused it."""
    try:
        await db.rollback()
    except Exception:
        pass


async def _telemetry_db_now(app_db: AsyncSession) -> datetime:
    """Read the app DB's clock, which is the clock that stamped the readings.

    Telemetry timestamps come from the columns' server default, NOW().
    Judging bucket completeness by the app host's clock instead would shift
    every window by any host clock skew, or by a DB time zone other than UTC.
    """
    now = (await app_db.execute(text("SELECT NOW()"))).scalar()
    await app_db.commit()
    return now


def _telemetry_delete_batch(batch_size: Optional[int]) -> int:
    """Rows per raw-telemetry DELETE.

    Larger than the request tables use: telemetry rows are small and have
    no FK children.
    """
    return max(batch_size or 0, 10000)


async def _newest_archived_bucket(
    archive_db: AsyncSession, model: Any
) -> Optional[datetime]:
    """Newest bucket_start in one rollup table, its watermark (None if empty).

    Commits to end the read, so whatever the caller does next on the archive
    starts a transaction of its own.
    """
    newest = (
        await archive_db.execute(select(func.max(model.bucket_start)))
    ).scalar()
    await archive_db.commit()
    return newest


async def _rollup_telemetry_table(
    app_db: AsyncSession,
    archive_db: AsyncSession,
    spec: _RollupSpec,
    model: Any,
    now: datetime,
    insert_batch: int = _ROLLUP_INSERT_BATCH,
    *,
    keep_since: Optional[datetime] = None,
    max_span: Optional[timedelta] = _ROLLUP_MAX_SPAN_PER_CYCLE,
) -> dict[str, Any]:
    """Archive one table's complete buckets that are not in the archive yet.

    Each window is aggregated in SQL on the app DB (raw rows never reach
    Python), then written with INSERT IGNORE, the rows passed as executemany
    parameters (see _ROLLUP_INSERT_BATCH), inside one archive transaction
    committed once at the end of the window.  Buckets older than
    ``keep_since`` (where tier 2 begins) are never written, and ``max_span``
    caps the history one pass covers (None = everything up to the horizon).

    Returns ``rolled_through``, the end of the last committed window: every
    live reading older than it has been aggregated, or is older than
    ``keep_since``.  It is None only when the table has never held a
    reading and there is no ``keep_since``.  Raises on failure.  Windows
    committed before the failure stay committed, and the caller must leave
    this table's raw rows alone for the cycle.
    """
    # The single-row statement, executed with a list of rows: SQLAlchemy
    # compiles it once and serves every later batch from its cache.
    insert_ignore = model.__table__.insert().prefix_with("IGNORE")

    # The read is committed, so each window below is exactly one archive
    # transaction.
    watermark = await _newest_archived_bucket(archive_db, model)

    after = (
        _floor_to_bucket(watermark) + _ROLLUP_BUCKET
        if watermark is not None
        else _ROLLUP_EPOCH
    )
    if keep_since is not None:
        after = max(after, _floor_to_bucket(keep_since))
    next_reading = (
        await app_db.execute(text(spec.next_reading_sql), {"after": after})
    ).scalar()
    await app_db.commit()

    plan = _plan_rollup_windows(
        watermark, next_reading, now, keep_since=keep_since, max_span=max_span
    )
    rolled_through = plan.start
    rows_written = 0

    for window_start, window_end in plan.windows:
        result = await app_db.execute(
            text(spec.rollup_sql), {"start": window_start, "end": window_end}
        )
        rows = [dict(row) for row in result.mappings().all()]
        # End the app-DB read now.  A snapshot held across the archive write,
        # or across the whole backfill, would pin undo history on the busiest
        # tables in the app DB.
        await app_db.commit()

        window_written = 0
        try:
            for i in range(0, len(rows), insert_batch):
                inserted = await archive_db.execute(
                    insert_ignore, rows[i:i + insert_batch]
                )
                window_written += inserted.rowcount or 0
            await archive_db.commit()
        except BaseException:
            await _rollback_quietly(archive_db)
            raise

        rows_written += window_written
        rolled_through = window_end

    return {
        "rows_written": rows_written,
        "windows": len(plan.windows),
        "rolled_through": rolled_through,
        "more_pending": bool(plan.windows) and rolled_through < plan.horizon,
    }


async def _delete_telemetry_before(
    app_db: AsyncSession, spec: _RollupSpec, before: datetime, batch: int
) -> int:
    """Delete one table's readings older than ``before`` in batches.

    Commits after each batch so InnoDB lock duration stays bounded.
    """
    total = 0
    while True:
        result = await app_db.execute(
            text(spec.delete_sql), {"cutoff": before, "batch": batch}
        )
        deleted = result.rowcount
        total += deleted
        await app_db.commit()
        if deleted > 0:
            logger.info(
                "retention_telemetry_batch",
                table=spec.table,
                deleted=deleted,
                total=total,
            )
        if deleted < batch:
            return total


async def archive_expired_telemetry(
    app_db: AsyncSession,
    archive_db: AsyncSession,
    cutoff: Optional[datetime],
    batch_size: int,
    *,
    now: Optional[datetime] = None,
    tier2_days: Optional[int] = None,
    max_span: Optional[timedelta] = _ROLLUP_MAX_SPAN_PER_CYCLE,
) -> dict[str, Any]:
    """Roll telemetry up into 5-minute archive rows, then delete expired raw rows.

    Two steps for each live table (node, GPU device, backend):

    1. Rollup.  Every complete 5-minute bucket that is not in the archive
       yet, and that tier 2 keeps (the last ``tier2_days``; None = no
       limit), is aggregated in SQL (avg/min/max plus sample counts) and
       written to the table's Archived*Telemetry5m rollup table, in 6-hour
       windows that are each one archive transaction, at most ``max_span``
       (14 days) of history per table per call; None lifts the cap, for a
       manual purge that must catch up before it deletes.  It runs whether
       or not anything is due for deletion (``cutoff`` None = rollup only),
       so the archive stays about one retention cycle behind the live tables.
    2. Delete.  Raw readings older than ``cutoff`` are deleted in batches,
       but never past the table's rollup frontier: a reading goes only after
       its bucket has been committed to the archive (or when it is older
       than tier 2 keeps).  When a table's rollup fails, its raw rows are
       left alone this cycle.

    Raw readings are not copied to the archive one by one; the rollups are
    the long-term record.  The raw archived_*_telemetry tables only hold
    legacy rows.  ``now`` defaults to the app DB's clock, which stamped the
    readings; tier 2 is measured on the same clock.

    Returns, per table: rollup rows written, windows aggregated, the rollup
    frontier (ISO string), raw rows deleted, and ``error`` when the rollup
    failed.
    """
    models = _rollup_models()
    if now is None:
        now = await _telemetry_db_now(app_db)
    keep_since = (
        _as_naive_utc(now) - timedelta(days=tier2_days) if tier2_days else None
    )
    tel_batch = _telemetry_delete_batch(batch_size)

    summary: dict[str, Any] = {}
    for spec in _ROLLUP_SPECS:
        entry: dict[str, Any] = {
            "rolled_up": 0,
            "windows": 0,
            "rolled_through": None,
            "deleted": 0,
        }
        summary[spec.table] = entry

        try:
            progress = await _rollup_telemetry_table(
                app_db, archive_db, spec, models[spec.table], now,
                keep_since=keep_since, max_span=max_span,
            )
        except Exception as exc:
            await _rollback_quietly(app_db)
            await _rollback_quietly(archive_db)
            entry["error"] = f"{type(exc).__name__}: {exc}"[:500]
            logger.warning(
                "retention_telemetry_rollup_failed",
                table=spec.table,
                error=entry["error"],
                exc_info=True,
            )
            if cutoff is not None:
                logger.warning(
                    "retention_telemetry_delete_skipped",
                    table=spec.table,
                    reason="rollup_failed",
                )
            continue

        entry["rolled_up"] = progress["rows_written"]
        entry["windows"] = progress["windows"]
        entry["rolled_through"] = _iso(progress["rolled_through"])
        if progress["windows"]:
            logger.info(
                "retention_telemetry_rollup",
                table=spec.table,
                rolled_up=progress["rows_written"],
                windows=progress["windows"],
                rolled_through=entry["rolled_through"],
                more_pending=progress["more_pending"],
            )

        if cutoff is None:
            continue
        before = _effective_delete_cutoff(cutoff, progress["rolled_through"])
        if before is None:
            # No frontier means the live table has never held a reading
            # (and there is no tier-2 limit), so there is nothing to delete.
            continue
        entry["deleted"] = await _delete_telemetry_before(
            app_db, spec, before, tel_batch
        )

    return summary


async def cleanup_expired_telemetry(
    app_db: AsyncSession, cutoff: datetime
) -> dict[str, int]:
    """Delete expired telemetry in batches (no archival). Fallback path."""
    counts = {
        "backend_telemetry": 0,
        "gpu_device_telemetry": 0,
        "node_telemetry": 0,
    }

    for table_name, key in [
        ("backend_telemetry", "backend_telemetry"),
        ("gpu_device_telemetry", "gpu_device_telemetry"),
        ("node_telemetry", "node_telemetry"),
    ]:
        assert table_name in _TELEMETRY_TABLES  # closed allowlist, not input
        while True:
            result = await app_db.execute(
                text(
                    f"DELETE FROM {table_name}"
                    f" WHERE timestamp < :cutoff"
                    f" LIMIT :batch"
                ),
                {"cutoff": cutoff, "batch": 10000},
            )
            deleted = result.rowcount
            counts[key] += deleted
            await app_db.commit()
            if deleted < 10000:
                break

    return counts


# ------------------------------------------------------------------
# Tier 2: purge old archives
# ------------------------------------------------------------------


async def _purge_rollups_before(
    archive_db: AsyncSession, model: Any, cutoff: datetime,
    batch: int = _ROLLUP_PURGE_BATCH,
) -> int:
    """Delete one rollup table's buckets that start before ``cutoff``.

    Runs in batches, committing after each one.
    """
    total = 0
    while True:
        result = await archive_db.execute(
            delete(model)
            .where(model.bucket_start < cutoff)
            .with_dialect_options(mysql_limit=batch)
        )
        deleted = result.rowcount or 0
        total += deleted
        await archive_db.commit()
        if deleted < batch:
            return total


async def purge_expired_archives(
    archive_db: AsyncSession, config: dict[str, Any]
) -> dict[str, int]:
    """Delete archive rows older than each category's tier2_days.

    Archived requests, chat and legacy raw telemetry expire by archived_at.
    The 5-minute telemetry rollups expire by bucket_start, the time they
    describe, so tier 2 is the amount of history kept.  A tier 2 of 0 purges
    nothing, as for every category: for telemetry it stops new rollups, but
    the averages already in the archive are kept (the Retention page says
    so).
    """
    from backend.app.db.archive_models import (
        ArchivedArtifact,
        ArchivedBackendTelemetry,
        ArchivedChatAttachment,
        ArchivedChatConversation,
        ArchivedChatMessage,
        ArchivedGPUDeviceTelemetry,
        ArchivedNodeTelemetry,
        ArchivedRequest,
        ArchivedResponse,
        ArchivedSchedulerDecision,
    )

    counts: dict[str, int] = {}

    # Purge request archives
    req_tier2 = config.get("retention.requests.tier2_days", 730)
    if req_tier2 > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(days=req_tier2)
        # Find expired request IDs to cascade delete children
        result = await archive_db.execute(
            select(ArchivedRequest.id).where(ArchivedRequest.archived_at < cutoff)
        )
        req_ids = [r[0] for r in result.all()]

        if req_ids:
            for model, key in [
                (ArchivedSchedulerDecision, "scheduler_decisions"),
                (ArchivedResponse, "responses"),
                (ArchivedArtifact, "artifacts"),
            ]:
                r = await archive_db.execute(
                    delete(model).where(model.request_id.in_(req_ids))
                )
                counts[key] = r.rowcount

            r = await archive_db.execute(
                delete(ArchivedRequest).where(ArchivedRequest.id.in_(req_ids))
            )
            counts["requests"] = r.rowcount

    # Purge chat archives
    chat_tier2 = config.get("retention.chat.tier2_days", 730)
    if chat_tier2 > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(days=chat_tier2)
        result = await archive_db.execute(
            select(ArchivedChatConversation.id).where(
                ArchivedChatConversation.archived_at < cutoff
            )
        )
        conv_ids = [r[0] for r in result.all()]

        if conv_ids:
            # Get message IDs for attachment cleanup
            msg_result = await archive_db.execute(
                select(ArchivedChatMessage.id).where(
                    ArchivedChatMessage.conversation_id.in_(conv_ids)
                )
            )
            msg_ids = [r[0] for r in msg_result.all()]

            if msg_ids:
                r = await archive_db.execute(
                    delete(ArchivedChatAttachment).where(
                        ArchivedChatAttachment.message_id.in_(msg_ids)
                    )
                )
                counts["chat_attachments"] = r.rowcount

            r = await archive_db.execute(
                delete(ArchivedChatMessage).where(
                    ArchivedChatMessage.conversation_id.in_(conv_ids)
                )
            )
            counts["chat_messages"] = r.rowcount

            r = await archive_db.execute(
                delete(ArchivedChatConversation).where(
                    ArchivedChatConversation.id.in_(conv_ids)
                )
            )
            counts["chat_conversations"] = r.rowcount

    # Purge telemetry archives: the legacy raw snapshots, archived before
    # 5-minute rollups replaced them, by archive time.
    tel_tier2 = config.get("retention.telemetry.tier2_days", 0)
    if tel_tier2 > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(days=tel_tier2)
        for model, key in [
            (ArchivedBackendTelemetry, "backend_telemetry"),
            (ArchivedGPUDeviceTelemetry, "gpu_device_telemetry"),
            (ArchivedNodeTelemetry, "node_telemetry"),
        ]:
            r = await archive_db.execute(
                delete(model).where(model.archived_at < cutoff)
            )
            counts[key] = r.rowcount

    await archive_db.flush()
    await archive_db.commit()

    # The 5-minute rollups, by the time they describe.  Batched with a commit
    # per batch, because shortening tier 2 can expire millions of rows at once.
    if tel_tier2 > 0:
        rollup_cutoff = _as_naive_utc(
            datetime.now(timezone.utc) - timedelta(days=tel_tier2)
        )
        # One table at a time, each on its own: a rollup table that is missing
        # (create_all failed at startup) or broken must not stop the others,
        # nor - now that steps are isolated - the rest of the purge's caller.
        for table_name, model in _rollup_models().items():
            key = f"{table_name}_5m"
            try:
                counts[key] = await _purge_rollups_before(
                    archive_db, model, rollup_cutoff
                )
            except Exception as exc:
                await _rollback_quietly(archive_db)
                counts[key] = {"error": f"{type(exc).__name__}: {exc}"[:500]}
                logger.warning(
                    "retention_rollup_purge_failed", table=key, error=counts[key]["error"]
                )

    return counts


# ------------------------------------------------------------------
# Stats & browsing
# ------------------------------------------------------------------


async def get_archive_stats(archive_db: AsyncSession) -> dict[str, Any]:
    """Get row counts and disk usage per archive table."""
    tables = [
        "archived_requests",
        "archived_responses",
        "archived_scheduler_decisions",
        "archived_artifacts",
        "archived_chat_conversations",
        "archived_chat_messages",
        "archived_chat_attachments",
        "archived_backend_telemetry",
        "archived_gpu_device_telemetry",
        "archived_node_telemetry",
        "archived_node_telemetry_5m",
        "archived_gpu_device_telemetry_5m",
        "archived_backend_telemetry_5m",
    ]

    stats: dict[str, Any] = {"tables": {}, "total_rows": 0, "total_size_mb": 0.0}

    for table_name in tables:
        result = await archive_db.execute(
            text(
                "SELECT table_rows, "
                "ROUND((data_length + index_length) / 1024 / 1024, 2) AS size_mb "
                "FROM information_schema.tables "
                "WHERE table_schema = DATABASE() AND table_name = :tbl"
            ),
            {"tbl": table_name},
        )
        row = result.first()
        if row:
            stats["tables"][table_name] = {
                "rows": row[0] or 0,
                "size_mb": float(row[1] or 0),
            }
            stats["total_rows"] += row[0] or 0
            stats["total_size_mb"] += float(row[1] or 0)
        else:
            stats["tables"][table_name] = {"rows": 0, "size_mb": 0.0}

    stats["total_size_mb"] = round(stats["total_size_mb"], 2)
    return stats


async def get_app_db_counts(app_db: AsyncSession) -> dict[str, int]:
    """Get row counts for the main app tables that retention manages."""
    from backend.app.db.models import (
        Artifact,
        ChatConversation,
        ChatMessage,
        DlpAlert,
        Request,
        Response,
        SchedulerDecision,
    )

    counts = {}
    for label, model in [
        ("requests", Request),
        ("responses", Response),
        ("scheduler_decisions", SchedulerDecision),
        ("artifacts", Artifact),
        ("chat_conversations", ChatConversation),
        ("chat_messages", ChatMessage),
        ("dlp_alerts", DlpAlert),
    ]:
        result = await app_db.execute(select(func.count()).select_from(model))
        counts[label] = result.scalar() or 0
    return counts


async def browse_archive(
    archive_db: AsyncSession,
    category: str,
    page: int = 1,
    per_page: int = 50,
    model_filter: Optional[str] = None,
    user_id_filter: Optional[int] = None,
) -> tuple[list[dict], int]:
    """Paginated query on archived tables.

    Returns (rows_as_dicts, total_count).
    """
    from backend.app.db.archive_models import (
        ArchivedChatConversation,
        ArchivedRequest,
    )

    if category == "chat":
        table_model = ArchivedChatConversation
    else:
        table_model = ArchivedRequest

    # Build base query
    query = select(table_model)
    count_query = select(func.count()).select_from(table_model)

    # Apply filters
    if model_filter and hasattr(table_model, "model"):
        query = query.where(table_model.model.like(f"%{model_filter}%"))
        count_query = count_query.where(table_model.model.like(f"%{model_filter}%"))

    if user_id_filter is not None and hasattr(table_model, "user_id"):
        query = query.where(table_model.user_id == user_id_filter)
        count_query = count_query.where(table_model.user_id == user_id_filter)

    # Get total
    total_result = await archive_db.execute(count_query)
    total = total_result.scalar() or 0

    # Paginate
    offset = (page - 1) * per_page
    query = query.order_by(table_model.archived_at.desc()).offset(offset).limit(per_page)

    result = await archive_db.execute(query)
    rows = [_row_to_dict(r) for r in result.scalars().all()]

    return rows, total


# ------------------------------------------------------------------
# Cross-worker mutual exclusion
# ------------------------------------------------------------------

# MariaDB advisory lock name used to serialize retention cycles across
# all uvicorn workers (and across scheduled + manual triggers).
_RETENTION_LOCK_NAME = "mindrouter_retention"


async def is_retention_running() -> bool:
    """Return True if some worker is currently running a retention cycle.

    Uses ``IS_FREE_LOCK`` which returns 1 if the named advisory lock is
    free, 0 if held, NULL on error.  Any non-1 result is treated as
    "assume busy" — better to show a false positive than to launch a
    duplicate cycle.
    """
    from backend.app.db.session import get_async_db_context

    try:
        async with get_async_db_context() as db:
            is_free = await db.scalar(
                text("SELECT IS_FREE_LOCK(:name)"),
                {"name": _RETENTION_LOCK_NAME},
            )
        return is_free != 1
    except Exception:
        logger.exception("retention_lock_check_failed")
        return False


async def try_run_retention_with_lock(trigger: str) -> dict[str, Any]:
    """Run ``run_retention_cycle`` while holding the advisory lock.

    The lock is held on a dedicated session that stays open for the
    entire cycle, so other workers (or a stacked manual click) see the
    lock as held and bail out immediately.

    Returns the cycle summary, plus a ``trigger`` tag and ``skipped``
    flag.  When skipped, ``reason`` explains why.
    """
    from backend.app.db.session import AsyncSessionLocal

    # Dedicated session — NOT from the shared context manager because
    # we need to keep this one connection bound to the caller for the
    # lock's full lifetime.
    lock_session = AsyncSessionLocal()
    try:
        acquired = await lock_session.scalar(
            text("SELECT GET_LOCK(:name, 0)"),
            {"name": _RETENTION_LOCK_NAME},
        )
        if acquired != 1:
            logger.info(
                "retention_cycle_skipped_lock_held", trigger=trigger
            )
            return {
                "skipped": True,
                "reason": "already_running",
                "trigger": trigger,
            }

        try:
            logger.info("retention_cycle_start", trigger=trigger)
            summary = await run_retention_cycle()
            logger.info(
                "retention_cycle_complete", trigger=trigger, summary=summary
            )
            summary["trigger"] = trigger
            return summary
        finally:
            try:
                await lock_session.execute(
                    text("SELECT RELEASE_LOCK(:name)"),
                    {"name": _RETENTION_LOCK_NAME},
                )
                await lock_session.commit()
            except Exception:
                logger.exception("retention_lock_release_failed")
    finally:
        try:
            await lock_session.close()
        except Exception:
            pass


# ------------------------------------------------------------------
# Full retention cycle (called by background loop or "Run Now")
# ------------------------------------------------------------------


async def cleanup_expired_request_images(
    artifact_storage_path: str, days: int, force_all: bool = False
) -> dict[str, Any]:
    """Delete ``request_images/<uuid>/`` directories older than ``days``.

    These are audit-offload images written by inference: base64 images
    extracted from multimodal (OCR/vision) requests so the DB audit record
    stays small. Nothing reads them back -- they exist only for forensic
    inspection -- so they are reaped by directory mtime. A value of 0
    disables cleanup (keep forever) unless ``force_all`` is set (manual
    purge of everything).
    """
    import shutil
    from pathlib import Path

    base = Path(artifact_storage_path) / "request_images"
    if (days <= 0 and not force_all) or not base.is_dir():
        return {"deleted_dirs": 0, "skipped": True}

    cutoff_ts = (
        datetime.now(timezone.utc) - timedelta(days=days)
    ).timestamp()

    def _sweep() -> dict[str, int]:
        scanned = deleted = errors = 0
        with os.scandir(base) as it:
            for entry in it:
                scanned += 1
                try:
                    if not entry.is_dir(follow_symlinks=False):
                        continue
                    if entry.stat(follow_symlinks=False).st_mtime >= cutoff_ts:
                        continue
                    shutil.rmtree(entry.path, ignore_errors=False)
                    deleted += 1
                except Exception:
                    errors += 1
        return {"scanned": scanned, "deleted_dirs": deleted, "errors": errors}

    # Filesystem walk over potentially hundreds of thousands of dirs --
    # run off the event loop so it never blocks request handling.
    result = await asyncio.to_thread(_sweep)
    logger.info(
        "retention_request_images_cleanup", cutoff_days=days, **result
    )
    return result


async def cleanup_expired_stored_responses(
    app_db: AsyncSession, cutoff: datetime, batch_size: int
) -> dict[str, int]:
    """Delete expired stored_responses rows in batches (no archival).

    Offloaded image files are removed per-row (row-driven, not
    mtime-driven — a chain's files must die with its row), before the
    DELETE so a crash leaves rows pointing at missing files (harmless,
    reinflate is best-effort) rather than orphaned files.
    """
    from backend.app.services.responses_store import remove_artifacts

    deleted_total = 0
    files_removed = 0

    while True:
        result = await app_db.execute(
            text(
                "SELECT id, response_id, offloaded_images IS NOT NULL AS has_files"
                " FROM stored_responses WHERE created_at < :cutoff"
                " ORDER BY id LIMIT :batch"
            ),
            {"cutoff": cutoff, "batch": batch_size},
        )
        rows = result.fetchall()
        if not rows:
            break

        for row in rows:
            if row.has_files:
                remove_artifacts(row.response_id)
                files_removed += 1

        ids = [row.id for row in rows]
        await app_db.execute(
            text("DELETE FROM stored_responses WHERE id IN :ids").bindparams(
                bindparam("ids", expanding=True)
            ),
            {"ids": ids},
        )
        deleted_total += len(rows)
        await app_db.commit()

        logger.info(
            "retention_responses_store_batch",
            deleted=len(rows), total=deleted_total,
        )

        if len(rows) < batch_size:
            break

    return {"deleted": deleted_total, "artifact_dirs_removed": files_removed}


async def cleanup_expired_conversations(
    app_db: AsyncSession, cutoff: datetime, batch_size: int
) -> dict[str, int]:
    """Delete conversations idle since before the cutoff (by updated_at
    — item appends touch the conversation row), including their items
    and offloaded artifact directories."""
    from backend.app.services.conversations_store import (
        remove_conversation_artifacts,
    )

    deleted_convs = 0
    deleted_items = 0

    while True:
        result = await app_db.execute(
            text(
                "SELECT id, conversation_id FROM conversations"
                " WHERE updated_at < :cutoff ORDER BY id LIMIT :batch"
            ),
            {"cutoff": cutoff, "batch": batch_size},
        )
        rows = result.fetchall()
        if not rows:
            break

        ids = [row.id for row in rows]
        items_result = await app_db.execute(
            text(
                "DELETE FROM conversation_items WHERE conversation_pk IN :ids"
            ).bindparams(bindparam("ids", expanding=True)),
            {"ids": ids},
        )
        deleted_items += items_result.rowcount or 0
        await app_db.execute(
            text("DELETE FROM conversations WHERE id IN :ids").bindparams(
                bindparam("ids", expanding=True)
            ),
            {"ids": ids},
        )
        for row in rows:
            remove_conversation_artifacts(row.conversation_id)
        deleted_convs += len(rows)
        await app_db.commit()

        logger.info(
            "retention_conversations_batch",
            deleted=len(rows), total=deleted_convs,
        )
        if len(rows) < batch_size:
            break

    return {"deleted": deleted_convs, "items_deleted": deleted_items}


async def cleanup_expired_dlp_alerts(
    app_db: AsyncSession, cutoff: datetime, batch_size: int
) -> dict[str, int]:
    """Delete DLP alerts scanned before the cutoff.

    Delete-only, like responses_store: alerts have no archive table and no file
    artifacts, and nothing references dlp_alerts, so no FK ordering is needed.
    """
    deleted = 0

    while True:
        result = await app_db.execute(
            text(
                "SELECT id FROM dlp_alerts WHERE scanned_at < :cutoff"
                " ORDER BY id LIMIT :batch"
            ),
            {"cutoff": cutoff, "batch": batch_size},
        )
        rows = result.fetchall()
        if not rows:
            break

        ids = [row.id for row in rows]
        await app_db.execute(
            text("DELETE FROM dlp_alerts WHERE id IN :ids").bindparams(
                bindparam("ids", expanding=True)
            ),
            {"ids": ids},
        )
        deleted += len(rows)
        await app_db.commit()

        logger.info(
            "retention_dlp_alerts_batch", deleted=len(rows), total=deleted,
        )
        if len(rows) < batch_size:
            break

    return {"deleted": deleted}


async def delete_expired_requests_no_archive(
    app_db: AsyncSession, cutoff: datetime, batch_size: int
) -> dict[str, int]:
    """Delete requests (and children) older than cutoff without
    archiving, snapshotting aggregates first.  Shared by the retention
    cycle (archive DB unconfigured) and manual purge."""
    from backend.app.db.models import (
        Artifact, Request, Response,
        SchedulerDecision,
    )

    total = 0
    while True:
        result = await app_db.execute(
            select(Request.id)
            .where(Request.created_at < cutoff)
            .limit(batch_size)
        )
        ids = [r[0] for r in result.all()]
        if not ids:
            break
        # Snapshot aggregates before deletion
        await _snapshot_request_aggregates(app_db, ids)
        await _detach_request_references(app_db, ids)
        await app_db.execute(
            delete(SchedulerDecision).where(
                SchedulerDecision.request_id.in_(ids)
            )
        )
        await app_db.execute(
            delete(Response).where(Response.request_id.in_(ids))
        )
        await app_db.execute(
            delete(Artifact).where(Artifact.request_id.in_(ids))
        )
        await app_db.execute(
            delete(Request).where(Request.id.in_(ids))
        )
        await app_db.flush()
        await app_db.commit()
        total += len(ids)
        logger.info(
            "retention_requests_batch_no_archive",
            deleted=len(ids),
            total=total,
        )
    return {"deleted_without_archive": total}


async def cleanup_expired_web_search_logs(
    app_db: AsyncSession, cutoff: datetime, batch_size: int
) -> dict[str, int]:
    """Delete web-search audit rows created before the cutoff.

    Delete-only, like dlp_alerts: no archive table, no file artifacts, and
    nothing references web_search_logs, so no FK ordering is needed.  Pure ORM
    (no text() with interpolated ids) so the statement can never become an
    injection finding.
    """
    from backend.app.db.models import WebSearchLog

    deleted = 0
    while True:
        result = await app_db.execute(
            select(WebSearchLog.id)
            .where(WebSearchLog.created_at < cutoff)
            .order_by(WebSearchLog.id)
            .limit(batch_size)
        )
        ids = [row_id for (row_id,) in result.all()]
        if not ids:
            break

        await app_db.execute(
            delete(WebSearchLog).where(WebSearchLog.id.in_(ids))
        )
        deleted += len(ids)
        await app_db.commit()
        logger.info("retention_web_search_logs_batch", deleted=len(ids), total=deleted)

    return {"deleted": deleted}


# Categories an admin may purge on demand.  admin_audit_log is
# deliberately absent: the admin audit trail is retained permanently.
PURGE_CATEGORIES = (
    "requests",
    "chat",
    "telemetry",
    "request_images",
    "responses_store",
    "conversations",
    "dlp_alerts",
    "web_search_logs",
)


async def run_purge(category: str, older_than_days: int) -> dict[str, Any]:
    """Purge one category of records older than a cutoff, immediately.

    Runs the same archive-then-delete pipeline the retention cycle
    uses, just with an explicit cutoff — so an archive DB (when
    configured) still receives requests/chat rows, and telemetry's
    5-minute averages (best effort, see the telemetry branch), before
    they leave the app DB, and file artifacts are removed with their rows.
    older_than_days=0 purges everything in the category.
    """
    if category not in PURGE_CATEGORIES:
        raise ValueError(f"Unknown purge category: {category}")

    from backend.app.db.session import get_archive_db_context, get_async_db_context
    from backend.app.settings import get_settings

    settings = get_settings()
    cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)

    async with get_async_db_context() as app_db:
        config = await get_retention_config(app_db)
    batch = config.get("retention.batch_size", 500) or 500
    archive_configured = settings.archive_database_url is not None

    summary: dict[str, Any] = {"category": category, "older_than_days": older_than_days}

    if category == "requests":
        if archive_configured:
            async with get_async_db_context() as app_db:
                async with get_archive_db_context() as archive_db:
                    summary["result"] = await archive_expired_requests(
                        app_db, archive_db, cutoff, batch
                    )
        else:
            async with get_async_db_context() as app_db:
                summary["result"] = await delete_expired_requests_no_archive(
                    app_db, cutoff, batch
                )
    elif category == "chat":
        if archive_configured:
            async with get_async_db_context() as app_db:
                async with get_archive_db_context() as archive_db:
                    summary["result"] = await archive_expired_chats(
                        app_db, archive_db, cutoff, batch
                    )
        else:
            from backend.app.db import chat_crud

            async with get_async_db_context() as app_db:
                deleted = await chat_crud.delete_expired_conversations(
                    app_db, older_than_days,
                )
                orphans = await chat_crud.delete_all_orphan_attachments(app_db)
                await app_db.commit()
                summary["result"] = {
                    "conversations_deleted": deleted,
                    "orphan_attachments": orphans,
                }
    elif category == "telemetry":
        tel_tier2 = config.get("retention.telemetry.tier2_days", 0)
        if archive_configured and tel_tier2 > 0:
            # Bring the 5-minute averages up to date first, so complete
            # buckets reach the archive before their raw rows go.  One pass
            # without the 14-day cap: right after archiving was enabled, or
            # after an archive outage, the backlog is longer than a
            # scheduled cycle covers, and a capped pass would leave every
            # reading past its frontier to be purged unaveraged.  Best
            # effort all the same: the purge is an explicit admin action and
            # runs regardless.  Readings the rollup could not cover (the last
            # ~10 minutes, a table whose rollup failed, everything if the
            # archive is down) are deleted without averages, and the summary
            # and a warning name the tables that fell short.
            failed: list[str] = []
            try:
                async with get_async_db_context() as app_db:
                    async with get_archive_db_context() as archive_db:
                        summary["rollup"] = await archive_expired_telemetry(
                            app_db, archive_db, None, batch,
                            tier2_days=tel_tier2, max_span=None,
                        )
                failed = sorted(
                    table for table, entry in summary["rollup"].items()
                    if "error" in entry
                )
            except Exception as exc:
                logger.warning("purge_telemetry_rollup_failed", error=str(exc))
                summary["rollup"] = {"error": str(exc)[:500]}
                failed = sorted(_TELEMETRY_TABLES)
            if failed:
                summary["rollup_incomplete"] = failed
                logger.warning(
                    "purge_telemetry_rollup_incomplete",
                    tables=failed,
                    cutoff=cutoff.isoformat(),
                )
        async with get_async_db_context() as app_db:
            summary["result"] = await cleanup_expired_telemetry(app_db, cutoff)
    elif category == "request_images":
        summary["result"] = await cleanup_expired_request_images(
            settings.artifact_storage_path, older_than_days,
            force_all=(older_than_days == 0),
        )
    elif category == "responses_store":
        async with get_async_db_context() as app_db:
            summary["result"] = await cleanup_expired_stored_responses(
                app_db, cutoff, batch
            )
    elif category == "conversations":
        async with get_async_db_context() as app_db:
            summary["result"] = await cleanup_expired_conversations(
                app_db, cutoff, batch
            )
    elif category == "dlp_alerts":
        async with get_async_db_context() as app_db:
            summary["result"] = await cleanup_expired_dlp_alerts(
                app_db, cutoff, batch
            )
    elif category == "web_search_logs":
        async with get_async_db_context() as app_db:
            summary["result"] = await cleanup_expired_web_search_logs(
                app_db, cutoff, batch
            )

    return summary


async def try_run_purge_with_lock(
    category: str, older_than_days: int, trigger: str
) -> dict[str, Any]:
    """Run ``run_purge`` under the shared retention advisory lock, so a
    purge never overlaps a retention cycle (or another purge)."""
    from backend.app.db.session import AsyncSessionLocal

    lock_session = AsyncSessionLocal()
    try:
        acquired = await lock_session.scalar(
            text("SELECT GET_LOCK(:name, 0)"),
            {"name": _RETENTION_LOCK_NAME},
        )
        if acquired != 1:
            logger.info("purge_skipped_lock_held", trigger=trigger)
            return {
                "skipped": True,
                "reason": "already_running",
                "trigger": trigger,
            }
        try:
            logger.info(
                "purge_start", category=category,
                older_than_days=older_than_days, trigger=trigger,
            )
            summary = await run_purge(category, older_than_days)
            logger.info("purge_complete", trigger=trigger, summary=summary)
            summary["trigger"] = trigger
            return summary
        finally:
            try:
                await lock_session.execute(
                    text("SELECT RELEASE_LOCK(:name)"),
                    {"name": _RETENTION_LOCK_NAME},
                )
                await lock_session.commit()
            except Exception:
                logger.exception("retention_lock_release_failed")
    finally:
        try:
            await lock_session.close()
        except Exception:
            pass


async def run_retention_cycle() -> dict[str, Any]:
    """Execute one full retention cycle.

    Every category runs in its own ``_step``: an exception in one (the
    archive DB unreachable, /tmp full for the request dump, deadlock retries
    exhausted) is logged as ``retention_step_failed``, recorded in the summary
    as ``{"error": ...}`` and listed in ``failed_steps``, and the remaining
    categories still run.  The cycle used to run straight-line, so one failing
    early step silently skipped every later one - including raw-telemetry
    pruning, which nothing else performs.

    Returns a summary dict of all actions taken.
    """
    from backend.app.db.session import (
        get_archive_db_context,
        get_async_db_context,
    )
    from backend.app.settings import get_settings

    settings = get_settings()
    summary: dict[str, Any] = {}
    failed: list[str] = []

    async with get_async_db_context() as app_db:
        config = await get_retention_config(app_db)

    archive_configured = settings.archive_database_url is not None

    async def _step(name: str, run) -> None:
        try:
            result = await run()
        except Exception as exc:
            failed.append(name)
            summary[name] = {"error": f"{type(exc).__name__}: {exc}"[:500]}
            logger.exception("retention_step_failed", step=name)
            return
        if result is not None:
            summary[name] = result

    # Tier 1: archive + delete from app DB
    async def _requests():
        req_tier1 = config.get("retention.requests.tier1_days", 90)
        if req_tier1 <= 0:
            return None
        cutoff = datetime.now(timezone.utc) - timedelta(days=req_tier1)
        if archive_configured:
            async with get_async_db_context() as app_db:
                async with get_archive_db_context() as archive_db:
                    return await archive_expired_requests(
                        app_db, archive_db, cutoff,
                        config.get("retention.batch_size", 500),
                    )
        # No archive DB — just delete
        async with get_async_db_context() as app_db:
            return await delete_expired_requests_no_archive(
                app_db, cutoff,
                config.get("retention.batch_size", 500) or 500,
            )

    async def _chats():
        chat_tier1 = config.get("retention.chat.tier1_days", 90)
        if chat_tier1 <= 0:
            return None
        cutoff = datetime.now(timezone.utc) - timedelta(days=chat_tier1)
        if archive_configured:
            async with get_async_db_context() as app_db:
                async with get_archive_db_context() as archive_db:
                    return await archive_expired_chats(
                        app_db, archive_db, cutoff,
                        config.get("retention.batch_size", 500),
                    )
        # Fall back to existing cleanup (no archival)
        from backend.app.db import chat_crud
        async with get_async_db_context() as app_db:
            deleted = await chat_crud.delete_expired_conversations(
                app_db, chat_tier1,
            )
            orphans = await chat_crud.delete_all_orphan_attachments(app_db)
            await app_db.commit()
            return {
                "conversations_deleted": deleted,
                "orphan_attachments": orphans,
            }

    # Telemetry.  With an archive DB and tier 2 > 0, 5-minute averages roll
    # into the archive on every cycle, even when tier 1 is 0 (keep raw rows
    # forever), and expired raw rows are deleted only once their buckets are
    # archived.  The rollup skips buckets older than tier 2, which the purge
    # below would delete again.  Otherwise expired raw rows are simply
    # deleted.  This is the only place raw telemetry is pruned.
    async def _telemetry():
        telemetry_tier1 = config.get("retention.telemetry.tier1_days", 30)
        telemetry_tier2 = config.get("retention.telemetry.tier2_days", 0)
        if archive_configured and telemetry_tier2 > 0:
            cutoff = (
                datetime.now(timezone.utc) - timedelta(days=telemetry_tier1)
                if telemetry_tier1 > 0
                else None
            )
            async with get_async_db_context() as app_db:
                async with get_archive_db_context() as archive_db:
                    return await archive_expired_telemetry(
                        app_db, archive_db, cutoff,
                        config.get("retention.batch_size", 500),
                        tier2_days=telemetry_tier2,
                    )
        if telemetry_tier1 > 0:
            cutoff = datetime.now(timezone.utc) - timedelta(days=telemetry_tier1)
            async with get_async_db_context() as app_db:
                return await cleanup_expired_telemetry(app_db, cutoff)
        return None

    # Tier 2: purge old archives
    async def _archive_purge():
        if not archive_configured:
            return None
        async with get_archive_db_context() as archive_db:
            return await purge_expired_archives(archive_db, config)

    # Orphan attachments cleanup (always)
    async def _orphan_attachments():
        from backend.app.db import chat_crud
        async with get_async_db_context() as app_db:
            orphans = await chat_crud.delete_all_orphan_attachments(app_db)
            await app_db.commit()
            return orphans

    # Audit-offload request images: reap directories older than the
    # configured window (filesystem-only, no DB rows involved).
    async def _request_images():
        return await cleanup_expired_request_images(
            settings.artifact_storage_path,
            config.get("retention.request_images_days", 180),
        )

    # Stored Responses API state (delete-only, no archival — the audit
    # copy lives in requests/responses; offloaded image files are
    # removed row-driven, with their row).
    async def _responses_store():
        responses_days = config.get("retention.responses_store_days", 30)
        if responses_days <= 0:
            return None
        cutoff = datetime.now(timezone.utc) - timedelta(days=responses_days)
        async with get_async_db_context() as app_db:
            return await cleanup_expired_stored_responses(
                app_db, cutoff, config.get("retention.batch_size", 5000)
            )

    # Conversations (durable state — default 0 = never delete; sweeps
    # by last activity, not creation).
    async def _conversations():
        conversations_days = config.get("retention.conversations_days", 0)
        if conversations_days <= 0:
            return None
        cutoff = datetime.now(timezone.utc) - timedelta(days=conversations_days)
        async with get_async_db_context() as app_db:
            return await cleanup_expired_conversations(
                app_db, cutoff, config.get("retention.batch_size", 5000)
            )

    # DLP alerts (delete-only, no archival — default 0 = never delete).
    async def _dlp_alerts():
        dlp_days = config.get("retention.dlp_alerts_days", 0)
        if dlp_days <= 0:
            return None
        cutoff = datetime.now(timezone.utc) - timedelta(days=dlp_days)
        async with get_async_db_context() as app_db:
            return await cleanup_expired_dlp_alerts(
                app_db, cutoff, config.get("retention.batch_size", 5000)
            )

    # Web-search audit logs (delete-only — default 0 = never delete).
    async def _web_search_logs():
        ws_days = config.get("retention.web_search_logs_days", 0)
        if ws_days <= 0:
            return None
        cutoff = datetime.now(timezone.utc) - timedelta(days=ws_days)
        async with get_async_db_context() as app_db:
            return await cleanup_expired_web_search_logs(
                app_db, cutoff, config.get("retention.batch_size", 5000)
            )

    for name, run in (
        ("requests", _requests),
        ("chats", _chats),
        ("telemetry", _telemetry),
        ("archive_purge", _archive_purge),
        ("orphan_attachments", _orphan_attachments),
        ("request_images", _request_images),
        ("responses_store", _responses_store),
        ("conversations", _conversations),
        ("dlp_alerts", _dlp_alerts),
        ("web_search_logs", _web_search_logs),
    ):
        await _step(name, run)

    if failed:
        summary["failed_steps"] = failed
        logger.warning("retention_cycle_partial", failed_steps=failed)
    return summary
