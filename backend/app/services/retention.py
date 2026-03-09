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
Tier 2 (Archive DB): longer retention with full data preserved.
"""

import json as _json
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import delete, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.logging_config import get_logger

logger = get_logger(__name__)

# Default retention config values
_DEFAULTS: dict[str, Any] = {
    "retention.requests.tier1_days": 90,
    "retention.requests.tier2_days": 730,
    "retention.chat.tier1_days": 90,
    "retention.chat.tier2_days": 730,
    "retention.telemetry.tier1_days": 30,
    "retention.telemetry.tier2_days": 0,   # 0 = no archival, just delete
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


async def _bulk_insert_ignore(
    archive_db: AsyncSession,
    model_class,
    rows: list[dict],
) -> int:
    """Insert rows into archive table, skipping duplicates (INSERT IGNORE).

    Returns the number of rows inserted.
    """
    if not rows:
        return 0

    table = model_class.__table__
    now = datetime.now(timezone.utc)

    for row in rows:
        row["archived_at"] = now

    # Use INSERT IGNORE via prefix_with
    stmt = table.insert().prefix_with("IGNORE").values(rows)
    result = await archive_db.execute(stmt)
    return result.rowcount


# ------------------------------------------------------------------
# Tier 1: archive + delete from app DB
# ------------------------------------------------------------------


async def archive_expired_requests(
    app_db: AsyncSession,
    archive_db: AsyncSession,
    cutoff: datetime,
    batch_size: int,
) -> dict[str, int]:
    """Archive requests (and children) older than cutoff.

    Returns counts of archived rows per table.
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

    while True:
        # Fetch a batch of expired request IDs
        result = await app_db.execute(
            select(Request.id)
            .where(Request.created_at < cutoff)
            .limit(batch_size)
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

        # Archive: bulk insert into archive DB
        counts["responses"] += await _bulk_insert_ignore(
            archive_db, ArchivedResponse,
            [_row_to_dict(r) for r in responses],
        )
        counts["scheduler_decisions"] += await _bulk_insert_ignore(
            archive_db, ArchivedSchedulerDecision,
            [_row_to_dict(r) for r in sched_rows],
        )
        counts["artifacts"] += await _bulk_insert_ignore(
            archive_db, ArchivedArtifact,
            [_row_to_dict(r) for r in art_rows],
        )
        counts["requests"] += await _bulk_insert_ignore(
            archive_db, ArchivedRequest,
            [_row_to_dict(r) for r in req_rows],
        )
        await archive_db.flush()

        # Delete from app DB in FK order
        if request_ids:
            await app_db.execute(
                delete(SchedulerDecision).where(
                    SchedulerDecision.request_id.in_(request_ids)
                )
            )
            await app_db.execute(
                delete(Response).where(Response.request_id.in_(request_ids))
            )
            await app_db.execute(
                delete(Artifact).where(Artifact.request_id.in_(request_ids))
            )
            await app_db.execute(
                delete(Request).where(Request.id.in_(request_ids))
            )
            await app_db.flush()
            await app_db.commit()
            await archive_db.commit()

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

        # Archive to archive DB
        counts["chat_attachments"] += await _bulk_insert_ignore(
            archive_db, ArchivedChatAttachment,
            [_row_to_dict(a) for a in attachments],
        )
        counts["chat_messages"] += await _bulk_insert_ignore(
            archive_db, ArchivedChatMessage,
            [_row_to_dict(m) for m in messages],
        )
        counts["chat_conversations"] += await _bulk_insert_ignore(
            archive_db, ArchivedChatConversation,
            [_row_to_dict(c) for c in batch],
        )
        await archive_db.flush()

        # Delete from app DB in FK order
        if msg_ids:
            await app_db.execute(
                delete(ChatAttachment).where(
                    ChatAttachment.message_id.in_(msg_ids)
                )
            )
        await app_db.execute(
            delete(ChatMessage).where(
                ChatMessage.conversation_id.in_(conv_ids)
            )
        )
        await app_db.execute(
            delete(ChatConversation).where(
                ChatConversation.id.in_(conv_ids)
            )
        )
        await app_db.flush()
        await app_db.commit()
        await archive_db.commit()

        counts["chat_conversations"] += 0  # already counted above

    return counts


async def cleanup_expired_telemetry(
    app_db: AsyncSession, cutoff: datetime
) -> dict[str, int]:
    """Delete expired telemetry data (no archival)."""
    from backend.app.db.crud import delete_old_gpu_telemetry, delete_old_telemetry

    backend_count = await delete_old_telemetry(app_db, cutoff)
    gpu_count = await delete_old_gpu_telemetry(app_db, cutoff)
    await app_db.commit()

    return {"backend_telemetry": backend_count, "gpu_telemetry": gpu_count}


# ------------------------------------------------------------------
# Tier 2: purge old archives
# ------------------------------------------------------------------


async def purge_expired_archives(
    archive_db: AsyncSession, config: dict[str, Any]
) -> dict[str, int]:
    """Delete from archive DB where archived_at is older than tier2_days."""
    from backend.app.db.archive_models import (
        ArchivedArtifact,
        ArchivedChatAttachment,
        ArchivedChatConversation,
        ArchivedChatMessage,
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

    await archive_db.flush()
    await archive_db.commit()
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
# Full retention cycle (called by background loop or "Run Now")
# ------------------------------------------------------------------


async def run_retention_cycle() -> dict[str, Any]:
    """Execute one full retention cycle.

    Returns a summary dict of all actions taken.
    """
    from backend.app.db.session import (
        get_archive_db_context,
        get_archive_engine,
        get_async_db_context,
    )
    from backend.app.settings import get_settings

    settings = get_settings()
    summary: dict[str, Any] = {}

    async with get_async_db_context() as app_db:
        config = await get_retention_config(app_db)

    archive_configured = settings.archive_database_url is not None

    # Tier 1: archive + delete from app DB
    req_tier1 = config.get("retention.requests.tier1_days", 90)
    if req_tier1 > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(days=req_tier1)
        if archive_configured:
            async with get_async_db_context() as app_db:
                async with get_archive_db_context() as archive_db:
                    summary["requests"] = await archive_expired_requests(
                        app_db, archive_db, cutoff,
                        config.get("retention.batch_size", 500),
                    )
        else:
            # No archive DB — just delete
            from backend.app.db.models import (
                Artifact, Request, Response,
                SchedulerDecision,
            )
            async with get_async_db_context() as app_db:
                batch_size = config.get("retention.batch_size", 500)
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
                summary["requests"] = {"deleted_without_archive": total}

    chat_tier1 = config.get("retention.chat.tier1_days", 90)
    if chat_tier1 > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(days=chat_tier1)
        if archive_configured:
            async with get_async_db_context() as app_db:
                async with get_archive_db_context() as archive_db:
                    summary["chats"] = await archive_expired_chats(
                        app_db, archive_db, cutoff,
                        config.get("retention.batch_size", 500),
                    )
        else:
            # Fall back to existing cleanup (no archival)
            from backend.app.db import chat_crud
            async with get_async_db_context() as app_db:
                deleted = await chat_crud.delete_expired_conversations(
                    app_db, chat_tier1,
                )
                orphans = await chat_crud.delete_all_orphan_attachments(app_db)
                await app_db.commit()
                summary["chats"] = {
                    "conversations_deleted": deleted,
                    "orphan_attachments": orphans,
                }

    telemetry_tier1 = config.get("retention.telemetry.tier1_days", 30)
    if telemetry_tier1 > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(days=telemetry_tier1)
        async with get_async_db_context() as app_db:
            summary["telemetry"] = await cleanup_expired_telemetry(app_db, cutoff)

    # Tier 2: purge old archives
    if archive_configured:
        async with get_archive_db_context() as archive_db:
            summary["archive_purge"] = await purge_expired_archives(
                archive_db, config
            )

    # Orphan attachments cleanup (always)
    from backend.app.db import chat_crud
    async with get_async_db_context() as app_db:
        orphans = await chat_crud.delete_all_orphan_attachments(app_db)
        await app_db.commit()
        summary["orphan_attachments"] = orphans

    return summary
