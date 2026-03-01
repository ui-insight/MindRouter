############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# chat_crud.py: Database CRUD operations for chat entities
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Database CRUD operations for chat conversations, messages, and attachments."""

import os
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from sqlalchemy import func

from backend.app.db.models import ChatAttachment, ChatConversation, ChatMessage, User


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------

async def create_conversation(
    db: AsyncSession,
    user_id: int,
    title: str = "New Chat",
    model: Optional[str] = None,
) -> ChatConversation:
    conv = ChatConversation(user_id=user_id, title=title, model=model)
    db.add(conv)
    await db.flush()
    return conv


async def get_user_conversations(
    db: AsyncSession,
    user_id: int,
    limit: int = 50,
) -> List[ChatConversation]:
    result = await db.execute(
        select(ChatConversation)
        .where(ChatConversation.user_id == user_id)
        .order_by(ChatConversation.updated_at.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


async def get_conversation(
    db: AsyncSession,
    conversation_id: int,
    user_id: int,
) -> Optional[ChatConversation]:
    result = await db.execute(
        select(ChatConversation).where(
            ChatConversation.id == conversation_id,
            ChatConversation.user_id == user_id,
        )
    )
    return result.scalar_one_or_none()


async def get_conversation_with_messages(
    db: AsyncSession,
    conversation_id: int,
    user_id: int,
) -> Optional[dict]:
    result = await db.execute(
        select(ChatConversation)
        .where(
            ChatConversation.id == conversation_id,
            ChatConversation.user_id == user_id,
        )
        .options(
            selectinload(ChatConversation.messages).selectinload(ChatMessage.attachments)
        )
    )
    conv = result.scalar_one_or_none()
    if not conv:
        return None

    messages = []
    for msg in conv.messages:
        attachments = []
        for att in msg.attachments:
            # Determine thumbnail value: URL for filesystem thumbnails
            if att.thumbnail_path:
                thumb = f"/chat/api/attachments/{att.id}/thumbnail"
            else:
                thumb = None
            attachments.append({
                "id": att.id,
                "filename": att.filename,
                "is_image": att.is_image,
                "thumbnail": thumb,
                "content_type": att.content_type,
            })
        messages.append({
            "id": msg.id,
            "role": msg.role,
            "content": msg.content,
            "attachments": attachments,
        })

    return {
        "id": conv.id,
        "title": conv.title,
        "model": conv.model,
        "messages": messages,
    }


async def update_conversation(
    db: AsyncSession,
    conversation_id: int,
    user_id: int,
    title: Optional[str] = None,
    model: Optional[str] = None,
) -> Optional[ChatConversation]:
    conv = await get_conversation(db, conversation_id, user_id)
    if not conv:
        return None
    if title is not None:
        conv.title = title
    if model is not None:
        conv.model = model
    await db.flush()
    return conv


async def delete_conversation(
    db: AsyncSession,
    conversation_id: int,
    user_id: int,
) -> bool:
    conv = await get_conversation(db, conversation_id, user_id)
    if not conv:
        return False

    # Load messages with attachments to clean up files
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.conversation_id == conversation_id)
        .options(selectinload(ChatMessage.attachments))
    )
    messages = list(result.scalars().all())

    # Collect file paths to delete (include medium and small thumbnails)
    file_paths = []
    for msg in messages:
        for att in msg.attachments:
            if att.storage_path:
                file_paths.append(att.storage_path)
                medium = att.storage_path.replace('.jpg', '_medium.jpg')
                file_paths.append(medium)
            if att.thumbnail_path:
                file_paths.append(att.thumbnail_path)

    # Delete attachments, messages, then conversation
    for msg in messages:
        await db.execute(
            delete(ChatAttachment).where(ChatAttachment.message_id == msg.id)
        )
    # Also delete orphan attachments owned by this user that might reference this conv
    await db.execute(
        delete(ChatMessage).where(ChatMessage.conversation_id == conversation_id)
    )
    await db.delete(conv)
    await db.flush()

    # Clean up files from filesystem
    for path in file_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass

    return True


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

async def create_message(
    db: AsyncSession,
    conversation_id: int,
    role: str,
    content: Optional[str] = None,
) -> ChatMessage:
    msg = ChatMessage(conversation_id=conversation_id, role=role, content=content)
    db.add(msg)
    await db.flush()
    return msg


async def get_conversation_messages(
    db: AsyncSession,
    conversation_id: int,
) -> List[ChatMessage]:
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.conversation_id == conversation_id)
        .options(selectinload(ChatMessage.attachments))
        .order_by(ChatMessage.created_at)
    )
    return list(result.scalars().all())


# ---------------------------------------------------------------------------
# Attachments
# ---------------------------------------------------------------------------

async def create_attachment(
    db: AsyncSession,
    user_id: int,
    filename: str,
    content_type: Optional[str] = None,
    is_image: bool = False,
    storage_path: Optional[str] = None,
    extracted_text: Optional[str] = None,
    file_size: Optional[int] = None,
) -> ChatAttachment:
    att = ChatAttachment(
        user_id=user_id,
        filename=filename,
        content_type=content_type,
        is_image=is_image,
        storage_path=storage_path,
        extracted_text=extracted_text,
        file_size=file_size,
    )
    db.add(att)
    await db.flush()
    return att


async def link_attachments_to_message(
    db: AsyncSession,
    attachment_ids: List[int],
    message_id: int,
    user_id: int,
) -> None:
    if not attachment_ids:
        return
    await db.execute(
        update(ChatAttachment)
        .where(
            ChatAttachment.id.in_(attachment_ids),
            ChatAttachment.user_id == user_id,
            ChatAttachment.message_id.is_(None),
        )
        .values(message_id=message_id)
    )
    await db.flush()


async def get_attachment(
    db: AsyncSession,
    attachment_id: int,
    user_id: int,
) -> Optional[ChatAttachment]:
    result = await db.execute(
        select(ChatAttachment).where(
            ChatAttachment.id == attachment_id,
            ChatAttachment.user_id == user_id,
        )
    )
    return result.scalar_one_or_none()


async def delete_orphan_attachments(
    db: AsyncSession,
    user_id: int,
    max_age_hours: int = 24,
) -> int:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    result = await db.execute(
        select(ChatAttachment).where(
            ChatAttachment.user_id == user_id,
            ChatAttachment.message_id.is_(None),
            ChatAttachment.created_at < cutoff,
        )
    )
    orphans = list(result.scalars().all())

    for att in orphans:
        if att.storage_path:
            try:
                if os.path.exists(att.storage_path):
                    os.remove(att.storage_path)
            except OSError:
                pass
            # Also remove medium thumbnail if it exists
            medium = att.storage_path.replace('.jpg', '_medium.jpg')
            try:
                if os.path.exists(medium):
                    os.remove(medium)
            except OSError:
                pass
        if att.thumbnail_path:
            try:
                if os.path.exists(att.thumbnail_path):
                    os.remove(att.thumbnail_path)
            except OSError:
                pass

    if orphans:
        await db.execute(
            delete(ChatAttachment).where(
                ChatAttachment.user_id == user_id,
                ChatAttachment.message_id.is_(None),
                ChatAttachment.created_at < cutoff,
            )
        )
        await db.flush()

    return len(orphans)


# ---------------------------------------------------------------------------
# Global Orphan Cleanup
# ---------------------------------------------------------------------------

async def delete_all_orphan_attachments(
    db: AsyncSession,
    max_age_hours: int = 24,
) -> int:
    """Delete all orphan attachments (unlinked to any message) older than max_age_hours."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    result = await db.execute(
        select(ChatAttachment).where(
            ChatAttachment.message_id.is_(None),
            ChatAttachment.created_at < cutoff,
        )
    )
    orphans = list(result.scalars().all())

    for att in orphans:
        _remove_attachment_files(att)

    if orphans:
        await db.execute(
            delete(ChatAttachment).where(
                ChatAttachment.message_id.is_(None),
                ChatAttachment.created_at < cutoff,
            )
        )
        await db.flush()

    return len(orphans)


def _remove_attachment_files(att: ChatAttachment) -> None:
    """Remove all filesystem files for an attachment."""
    if att.storage_path:
        for path in [att.storage_path, att.storage_path.replace('.jpg', '_medium.jpg')]:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass
    if att.thumbnail_path:
        try:
            if os.path.exists(att.thumbnail_path):
                os.remove(att.thumbnail_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Conversation Retention
# ---------------------------------------------------------------------------

async def delete_expired_conversations(
    db: AsyncSession,
    retention_days: int,
    batch_size: int = 100,
) -> int:
    """Delete conversations older than retention_days in batches.

    Returns total number of conversations deleted.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    total_deleted = 0

    while True:
        # Find a batch of expired conversations
        result = await db.execute(
            select(ChatConversation)
            .where(ChatConversation.updated_at < cutoff)
            .limit(batch_size)
        )
        batch = list(result.scalars().all())
        if not batch:
            break

        conv_ids = [c.id for c in batch]

        # Load attachments for file cleanup
        att_result = await db.execute(
            select(ChatAttachment)
            .join(ChatMessage, ChatAttachment.message_id == ChatMessage.id)
            .where(ChatMessage.conversation_id.in_(conv_ids))
        )
        attachments = list(att_result.scalars().all())

        # Collect file paths
        for att in attachments:
            _remove_attachment_files(att)

        # Delete in order: attachments -> messages -> conversations
        if conv_ids:
            # Delete attachments linked to messages in these conversations
            await db.execute(
                delete(ChatAttachment).where(
                    ChatAttachment.message_id.in_(
                        select(ChatMessage.id).where(
                            ChatMessage.conversation_id.in_(conv_ids)
                        )
                    )
                )
            )
            await db.execute(
                delete(ChatMessage).where(
                    ChatMessage.conversation_id.in_(conv_ids)
                )
            )
            await db.execute(
                delete(ChatConversation).where(
                    ChatConversation.id.in_(conv_ids)
                )
            )
            await db.flush()
            await db.commit()

        total_deleted += len(batch)

    return total_deleted


# ---------------------------------------------------------------------------
# Admin Conversation Queries
# ---------------------------------------------------------------------------

async def search_conversations_admin(
    db: AsyncSession,
    user_id: Optional[int] = None,
    model: Optional[str] = None,
    search_text: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    skip: int = 0,
    limit: int = 50,
) -> tuple:
    """Search conversations for admin view.

    Returns (list of dicts, total_count).
    """
    # Base query with message count
    msg_count = (
        select(func.count(ChatMessage.id))
        .where(ChatMessage.conversation_id == ChatConversation.id)
        .correlate(ChatConversation)
        .scalar_subquery()
    )

    query = (
        select(
            ChatConversation,
            User.username,
            msg_count.label("message_count"),
        )
        .join(User, ChatConversation.user_id == User.id)
    )
    count_query = (
        select(func.count(ChatConversation.id))
        .join(User, ChatConversation.user_id == User.id)
    )

    conditions = []
    if user_id:
        conditions.append(ChatConversation.user_id == user_id)
    if model:
        conditions.append(ChatConversation.model == model)
    if search_text:
        conditions.append(
            ChatConversation.title.ilike(f"%{search_text}%")
        )
    if start_date:
        conditions.append(ChatConversation.created_at >= start_date)
    if end_date:
        conditions.append(ChatConversation.created_at <= end_date)

    if conditions:
        for cond in conditions:
            query = query.where(cond)
            count_query = count_query.where(cond)

    count_result = await db.execute(count_query)
    total = count_result.scalar_one()

    query = query.order_by(ChatConversation.updated_at.desc()).offset(skip).limit(limit)
    result = await db.execute(query)
    rows = result.all()

    conversations = []
    for conv, username, msg_count_val in rows:
        conversations.append({
            "id": conv.id,
            "user_id": conv.user_id,
            "username": username,
            "title": conv.title,
            "model": conv.model,
            "message_count": msg_count_val or 0,
            "created_at": conv.created_at,
            "updated_at": conv.updated_at,
        })

    return conversations, total


async def get_conversation_messages_admin(
    db: AsyncSession,
    conversation_id: int,
) -> Optional[list]:
    """Get full messages for a conversation (admin access, no user_id check).

    Returns list of message dicts, or None if conversation not found.
    """
    result = await db.execute(
        select(ChatConversation).where(ChatConversation.id == conversation_id)
    )
    conv = result.scalar_one_or_none()
    if not conv:
        return None

    msg_result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.conversation_id == conversation_id)
        .options(selectinload(ChatMessage.attachments))
        .order_by(ChatMessage.created_at)
    )
    messages = list(msg_result.scalars().all())

    return [
        {
            "id": msg.id,
            "role": msg.role,
            "content": msg.content,
            "created_at": msg.created_at.isoformat() if msg.created_at else None,
            "attachment_count": len(msg.attachments),
        }
        for msg in messages
    ]
