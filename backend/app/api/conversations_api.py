############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# conversations_api.py: OpenAI Conversations API compatible
# endpoints (conv_* objects).
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""OpenAI Conversations API (/v1/conversations).

Durable server-side conversation state consumed by POST /v1/responses
via the ``conversation`` parameter. All endpoints are owner-scoped and
use the OpenAI error envelope. Gated by the same feature flag as the
Responses API (they are one feature family).
"""

from typing import Any, Optional, Tuple

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.auth import authenticate_request
from backend.app.api.responses_api import error_json
from backend.app.db import crud
from backend.app.db.models import ApiKey, User
from backend.app.db.session import get_async_db
from backend.app.logging_config import get_logger
from backend.app.services import conversations_store
from backend.app.settings import get_settings

logger = get_logger(__name__)
router = APIRouter(tags=["conversations"])

_MAX_CREATE_ITEMS = 20  # per spec: "You may add up to 20 items at a time."


def _disabled():
    return error_json(404, "The Conversations API is not enabled on this server.",
                      code="not_found")


def _not_found(conversation_id: str):
    return error_json(
        404, f"Conversation with id '{conversation_id}' not found.",
        code="conversation_not_found",
    )


@router.post("/v1/conversations")
async def create_conversation(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
    auth: Tuple[User, ApiKey] = Depends(authenticate_request),
):
    """Create a conversation, optionally seeded with up to 20 items."""
    user, api_key = auth
    settings = get_settings()
    if not settings.responses_api_enabled:
        return _disabled()

    try:
        body = await request.json() if await request.body() else {}
    except Exception:
        return error_json(400, "Invalid JSON body")
    if not isinstance(body, dict):
        return error_json(400, "Request body must be a JSON object")

    max_convs = settings.conversations_max_per_user
    if max_convs and await crud.count_conversations_for_user(db, user.id) >= max_convs:
        return error_json(
            400,
            f"Conversation limit reached ({max_convs}). Delete unused "
            "conversations and try again.",
        )

    metadata = body.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        return error_json(400, "'metadata' must be an object", param="metadata")

    items = None
    if body.get("items") is not None:
        try:
            items = conversations_store.validate_items(
                body["items"], max_count=_MAX_CREATE_ITEMS
            )
        except ValueError as e:
            return error_json(400, str(e), param="items")

    conversation = await crud.create_conversation(
        db, user_id=user.id, api_key_id=api_key.id, metadata=metadata,
    )
    if items:
        try:
            await conversations_store.append_items(db, conversation, items)
        except ValueError as e:
            return error_json(400, str(e), param="items")

    return conversations_store.conversation_object(conversation)


@router.get("/v1/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    db: AsyncSession = Depends(get_async_db),
    auth: Tuple[User, ApiKey] = Depends(authenticate_request),
):
    user, _ = auth
    if not get_settings().responses_api_enabled:
        return _disabled()
    conversation = await crud.get_conversation(db, conversation_id, user.id)
    if not conversation:
        return _not_found(conversation_id)
    return conversations_store.conversation_object(conversation)


@router.post("/v1/conversations/{conversation_id}")
async def update_conversation(
    conversation_id: str,
    request: Request,
    db: AsyncSession = Depends(get_async_db),
    auth: Tuple[User, ApiKey] = Depends(authenticate_request),
):
    """Update conversation metadata."""
    user, _ = auth
    if not get_settings().responses_api_enabled:
        return _disabled()
    conversation = await crud.get_conversation(db, conversation_id, user.id)
    if not conversation:
        return _not_found(conversation_id)

    try:
        body = await request.json()
    except Exception:
        return error_json(400, "Invalid JSON body")
    metadata = body.get("metadata")
    if not isinstance(metadata, dict):
        return error_json(400, "'metadata' must be an object", param="metadata")

    conversation.meta = metadata
    await crud.touch_conversation(db, conversation)
    return conversations_store.conversation_object(conversation)


@router.delete("/v1/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    db: AsyncSession = Depends(get_async_db),
    auth: Tuple[User, ApiKey] = Depends(authenticate_request),
):
    user, _ = auth
    if not get_settings().responses_api_enabled:
        return _disabled()
    conversation = await crud.delete_conversation(db, conversation_id, user.id)
    if not conversation:
        return _not_found(conversation_id)
    conversations_store.remove_conversation_artifacts(conversation_id)
    return conversations_store.deleted_object(conversation_id)


@router.post("/v1/conversations/{conversation_id}/items")
async def create_items(
    conversation_id: str,
    request: Request,
    db: AsyncSession = Depends(get_async_db),
    auth: Tuple[User, ApiKey] = Depends(authenticate_request),
):
    """Append items to a conversation."""
    user, _ = auth
    if not get_settings().responses_api_enabled:
        return _disabled()
    conversation = await crud.get_conversation(db, conversation_id, user.id)
    if not conversation:
        return _not_found(conversation_id)

    try:
        body = await request.json()
    except Exception:
        return error_json(400, "Invalid JSON body")
    try:
        items = conversations_store.validate_items(
            body.get("items"), max_count=_MAX_CREATE_ITEMS
        )
        stamped = await conversations_store.append_items(db, conversation, items)
    except ValueError as e:
        return error_json(400, str(e), param="items")

    return {
        "object": "list",
        "data": stamped,
        "first_id": stamped[0]["id"] if stamped else None,
        "last_id": stamped[-1]["id"] if stamped else None,
        "has_more": False,
    }


@router.get("/v1/conversations/{conversation_id}/items")
async def list_items(
    conversation_id: str,
    db: AsyncSession = Depends(get_async_db),
    auth: Tuple[User, ApiKey] = Depends(authenticate_request),
    limit: int = Query(20, ge=1, le=100),
    order: str = Query("desc"),
    after: Optional[str] = Query(None),
):
    user, _ = auth
    if not get_settings().responses_api_enabled:
        return _disabled()
    conversation = await crud.get_conversation(db, conversation_id, user.id)
    if not conversation:
        return _not_found(conversation_id)
    return await conversations_store.list_items_wire(
        db, conversation, limit=limit, order=order, after=after
    )


@router.get("/v1/conversations/{conversation_id}/items/{item_id}")
async def get_item(
    conversation_id: str,
    item_id: str,
    db: AsyncSession = Depends(get_async_db),
    auth: Tuple[User, ApiKey] = Depends(authenticate_request),
):
    user, _ = auth
    if not get_settings().responses_api_enabled:
        return _disabled()
    conversation = await crud.get_conversation(db, conversation_id, user.id)
    if not conversation:
        return _not_found(conversation_id)
    row = await crud.get_conversation_item(db, conversation.id, item_id)
    if not row:
        return error_json(
            404, f"Item with id '{item_id}' not found.", code="item_not_found",
        )
    return conversations_store._reinflate_row(conversation, row)


@router.delete("/v1/conversations/{conversation_id}/items/{item_id}")
async def delete_item(
    conversation_id: str,
    item_id: str,
    db: AsyncSession = Depends(get_async_db),
    auth: Tuple[User, ApiKey] = Depends(authenticate_request),
):
    """Delete an item. Per spec, returns the Conversation object."""
    user, _ = auth
    if not get_settings().responses_api_enabled:
        return _disabled()
    conversation = await crud.get_conversation(db, conversation_id, user.id)
    if not conversation:
        return _not_found(conversation_id)
    row = await crud.delete_conversation_item(db, conversation.id, item_id)
    if not row:
        return error_json(
            404, f"Item with id '{item_id}' not found.", code="item_not_found",
        )
    conversations_store.remove_item_artifacts(conversation_id, item_id)
    return conversations_store.conversation_object(conversation)
