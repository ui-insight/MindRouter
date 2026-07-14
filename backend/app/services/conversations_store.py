############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# conversations_store.py: Server-side state for the OpenAI
# Conversations API (conv_* objects).
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Item storage and wire-shaping for the Conversations API.

Conversations are durable, user-owned item logs consumed by
POST /v1/responses via the ``conversation`` parameter: stored items are
prepended to the request input, and the response's input + output items
are appended to the conversation after it completes (per spec — failed
responses append nothing).

Reuses the responses_store image-offload machinery (server-controlled
offload maps + realpath containment) with per-item artifact keys
``conversations/<conv_id>/<item_id>``.
"""

import json
from typing import Any, Dict, List, Optional

from backend.app.db import crud
from backend.app.db.models import Conversation
from backend.app.db.session import get_async_db_context
from backend.app.logging_config import get_logger
from backend.app.services.responses_store import (
    offload_images,
    reinflate_item_images,
    remove_artifacts,
    stamp_item_ids,
)
from backend.app.settings import get_settings

logger = get_logger(__name__)

_SUBDIR = "conversations"


def conversation_object(conversation: Conversation) -> Dict[str, Any]:
    """Wire shape: the Conversation object."""
    return {
        "id": conversation.conversation_id,
        "object": "conversation",
        "metadata": conversation.meta or {},
        "created_at": int(conversation.created_at.timestamp()),
    }


def deleted_object(conversation_id: str) -> Dict[str, Any]:
    return {
        "id": conversation_id,
        "object": "conversation.deleted",
        "deleted": True,
    }


def validate_items(raw_items: Any, max_count: Optional[int] = None) -> List[Dict[str, Any]]:
    """Validate an incoming items array. Raises ValueError."""
    if not isinstance(raw_items, list):
        raise ValueError("'items' must be an array of input items")
    if max_count is not None and len(raw_items) > max_count:
        raise ValueError(f"You may add up to {max_count} items at a time.")
    items = []
    for item in raw_items:
        if not isinstance(item, dict):
            raise ValueError("items must be objects")
        items.append(dict(item))

    max_bytes = get_settings().conversations_max_item_bytes
    if max_bytes:
        for item in items:
            if len(json.dumps(item)) > max_bytes:
                raise ValueError(
                    f"an item exceeds the maximum size of {max_bytes} bytes"
                )
    return items


async def append_items(
    db,
    conversation: Conversation,
    raw_items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Stamp, offload, and append items. Returns the stamped wire items
    (with original inline images — offload placeholders never leave the
    server). Caller commits."""
    settings = get_settings()
    max_items = settings.conversations_max_items
    if max_items:
        count = await crud.count_conversation_items(db, conversation.id)
        if count + len(raw_items) > max_items:
            raise ValueError(
                f"conversation item limit reached ({max_items})"
            )

    stamped = stamp_item_ids(raw_items)
    stored_items = []
    offload_maps = []
    for item in stamped:
        offloaded, offload_map = offload_images(
            [item],
            f"{conversation.conversation_id}/{item['id']}",
            subdir=_SUBDIR,
        )
        stored_items.append(offloaded[0])
        # Per-item offload maps are keyed "0.<part>" (single-item batch)
        offload_maps.append(offload_map)

    await crud.add_conversation_items(
        db, conversation, stored_items, offload_maps
    )
    return stamped


def _reinflate_row(conversation: Conversation, row) -> Dict[str, Any]:
    items = reinflate_item_images(
        [row.item],
        row.offloaded_images,
        f"{conversation.conversation_id}/{row.item_id}",
        subdir=_SUBDIR,
    )
    return items[0]


async def load_context_items(db, conversation: Conversation) -> List[Dict[str, Any]]:
    """All conversation items in append order, images restored — the
    history prepended to a /v1/responses request."""
    rows = await crud.get_conversation_items(db, conversation.id)
    return [_reinflate_row(conversation, row) for row in rows]


async def list_items_wire(
    db,
    conversation: Conversation,
    limit: int = 20,
    order: str = "desc",
    after: Optional[str] = None,
) -> Dict[str, Any]:
    """ConversationItemList envelope with cursor pagination."""
    rows = await crud.get_conversation_items(db, conversation.id)
    if order != "asc":
        rows = list(reversed(rows))
    if after:
        idx = next((i for i, r in enumerate(rows) if r.item_id == after), None)
        rows = rows[idx + 1:] if idx is not None else []
    page = rows[:limit]
    data = [_reinflate_row(conversation, row) for row in page]
    return {
        "object": "list",
        "data": data,
        "first_id": page[0].item_id if page else None,
        "last_id": page[-1].item_id if page else None,
        "has_more": len(rows) > limit,
    }


def remove_conversation_artifacts(conversation_id: str) -> None:
    """Remove ALL offloaded files for a conversation (its whole dir)."""
    remove_artifacts(conversation_id, subdir=_SUBDIR)


def remove_item_artifacts(conversation_id: str, item_id: str) -> None:
    remove_artifacts(f"{conversation_id}/{item_id}", subdir=_SUBDIR)


async def append_from_response(
    conversation_id: str,
    user_id: int,
    input_items: List[Dict[str, Any]],
    output_items: List[Dict[str, Any]],
) -> None:
    """Post-terminal append of a response's delta input + output items.

    Opens its own DB session (safe from a streaming generator's finally
    block under asyncio.shield). Never raises.
    """
    try:
        async with get_async_db_context() as db:
            conversation = await crud.get_conversation(
                db, conversation_id, user_id
            )
            if not conversation:
                return
            items = [dict(i) for i in (input_items + output_items)]
            # Drop server ids from output items that could collide with
            # ids already stored (e.g. resent history); stamping below
            # assigns fresh ones only where missing.
            existing = {
                row.item_id
                for row in await crud.get_conversation_items(db, conversation.id)
            }
            for item in items:
                if item.get("id") in existing:
                    item.pop("id", None)
            await append_items(db, conversation, items)
            await db.commit()
        logger.info(
            "conversation_appended",
            conversation_id=conversation_id,
            items=len(input_items) + len(output_items),
        )
    except Exception as e:
        logger.warning(
            "conversation_append_failed",
            conversation_id=conversation_id, error=str(e),
        )
