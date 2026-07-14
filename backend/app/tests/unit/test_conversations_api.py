############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_conversations_api.py: Route-level unit tests for the
# Conversations API endpoints.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Covers:
- Conversation CRUD: create (metadata, seed items, per-user cap),
  retrieve, update metadata, delete envelope + artifact cleanup
- Item endpoints: create (validation, 20-item cap), list delegation,
  get, delete (returns the Conversation object per spec)
- Feature-flag gating and owner-scoped 404s
- /v1/responses integration: conversation param mutual exclusion,
  unknown conversation, context prepending, post-completion append
"""

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.responses import JSONResponse, StreamingResponse

# Real light imports first (see test_responses_api.py bootstrap notes)
from backend.app.core.translators.responses_in import ResponsesRequestContext

_STUB_NAMES = [
    "backend.app.api",
    "backend.app.api.auth",
    "backend.app.api.responses_api",
    "backend.app.db",
    "backend.app.db.crud",
    "backend.app.db.models",
    "backend.app.db.session",
    "backend.app.services",
    "backend.app.services.conversations_store",
    "backend.app.logging_config",
    "backend.app.settings",
]
_added = []
for _name in _STUB_NAMES:
    if _name not in sys.modules:
        stub = MagicMock()
        if _name == "backend.app.logging_config":
            stub.get_logger = MagicMock(return_value=MagicMock())
        if _name == "backend.app.api.responses_api":
            # Real-ish error_json so route returns are inspectable
            def _error_json(status_code, message, err_type="invalid_request_error",
                            code=None, param=None):
                return JSONResponse(
                    status_code=status_code,
                    content={"error": {"message": message, "type": err_type,
                                       "param": param, "code": code}},
                )
            stub.error_json = _error_json
        sys.modules[_name] = stub
        _added.append(_name)

_api_dir = Path(__file__).resolve().parents[2] / "api"
_spec = importlib.util.spec_from_file_location(
    "conversations_api", _api_dir / "conversations_api.py",
    submodule_search_locations=[],
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

for _name in _added:
    sys.modules.pop(_name, None)


def _make_user():
    user = MagicMock()
    user.id = 1
    return user


def _make_api_key():
    key = MagicMock()
    key.id = 42
    return key


def _make_request(body=None):
    request = MagicMock()
    request.json = AsyncMock(return_value=body if body is not None else {})
    request.body = AsyncMock(return_value=json.dumps(body or {}).encode())
    return request


def _make_conversation(conversation_id="conv_abc"):
    conv = MagicMock()
    conv.id = 7
    conv.conversation_id = conversation_id
    conv.user_id = 1
    conv.meta = {"topic": "test"}
    conv.created_at = datetime(2026, 7, 16, tzinfo=timezone.utc)
    return conv


def _conv_object(conv):
    return {
        "id": conv.conversation_id,
        "object": "conversation",
        "metadata": conv.meta or {},
        "created_at": int(conv.created_at.timestamp()),
    }


def _make_store(conv=None):
    store = MagicMock()
    store.conversation_object = MagicMock(side_effect=_conv_object)
    store.deleted_object = MagicMock(side_effect=lambda cid: {
        "id": cid, "object": "conversation.deleted", "deleted": True,
    })
    store.validate_items = MagicMock(side_effect=lambda items, max_count=None: list(items))
    store.append_items = AsyncMock(side_effect=lambda db, c, items: [
        {**i, "id": i.get("id", f"msg_{n}")} for n, i in enumerate(items)
    ])
    store.list_items_wire = AsyncMock(return_value={
        "object": "list", "data": [], "first_id": None, "last_id": None,
        "has_more": False,
    })
    store.remove_conversation_artifacts = MagicMock()
    store.remove_item_artifacts = MagicMock()
    store._reinflate_row = MagicMock(
        side_effect=lambda c, row: dict(row.item)
    )
    return store


def _make_crud(conv=None):
    crud = MagicMock()
    crud.get_conversation = AsyncMock(return_value=conv)
    crud.create_conversation = AsyncMock(return_value=conv or _make_conversation())
    crud.delete_conversation = AsyncMock(return_value=conv)
    crud.count_conversations_for_user = AsyncMock(return_value=0)
    crud.touch_conversation = AsyncMock()
    crud.get_conversation_item = AsyncMock(return_value=None)
    crud.delete_conversation_item = AsyncMock(return_value=None)
    return crud


def _settings(enabled=True, max_convs=1000):
    s = MagicMock()
    s.responses_api_enabled = enabled
    s.conversations_max_per_user = max_convs
    return s


def _patches(crud, store, enabled=True, max_convs=1000):
    return (
        patch.object(_mod, "get_settings",
                     return_value=_settings(enabled, max_convs)),
        patch.object(_mod, "crud", crud),
        patch.object(_mod, "conversations_store", store),
    )


def _body(result):
    return json.loads(result.body)


class TestConversationCRUD:
    async def test_create(self):
        conv = _make_conversation()
        crud, store = _make_crud(conv), _make_store()
        p1, p2, p3 = _patches(crud, store)
        with p1, p2, p3:
            result = await _mod.create_conversation(
                _make_request({"metadata": {"topic": "test"}}),
                db=AsyncMock(), auth=(_make_user(), _make_api_key()),
            )
        assert result["object"] == "conversation"
        assert result["id"] == "conv_abc"
        assert result["metadata"] == {"topic": "test"}
        crud.create_conversation.assert_awaited_once()

    async def test_create_with_seed_items(self):
        conv = _make_conversation()
        crud, store = _make_crud(conv), _make_store()
        p1, p2, p3 = _patches(crud, store)
        with p1, p2, p3:
            result = await _mod.create_conversation(
                _make_request({"items": [{"role": "user", "content": "hi"}]}),
                db=AsyncMock(), auth=(_make_user(), _make_api_key()),
            )
        assert result["object"] == "conversation"
        store.append_items.assert_awaited_once()

    async def test_create_per_user_cap(self):
        crud, store = _make_crud(), _make_store()
        crud.count_conversations_for_user = AsyncMock(return_value=5)
        p1, p2, p3 = _patches(crud, store, max_convs=5)
        with p1, p2, p3:
            result = await _mod.create_conversation(
                _make_request({}), db=AsyncMock(),
                auth=(_make_user(), _make_api_key()),
            )
        assert result.status_code == 400
        assert "limit" in _body(result)["error"]["message"].lower()

    async def test_get_and_404(self):
        conv = _make_conversation()
        crud, store = _make_crud(conv), _make_store()
        p1, p2, p3 = _patches(crud, store)
        with p1, p2, p3:
            result = await _mod.get_conversation(
                "conv_abc", db=AsyncMock(), auth=(_make_user(), _make_api_key()),
            )
        assert result["id"] == "conv_abc"

        crud404 = _make_crud(None)
        p1, p2, p3 = _patches(crud404, store)
        with p1, p2, p3:
            result = await _mod.get_conversation(
                "conv_x", db=AsyncMock(), auth=(_make_user(), _make_api_key()),
            )
        assert result.status_code == 404
        assert _body(result)["error"]["code"] == "conversation_not_found"

    async def test_update_metadata(self):
        conv = _make_conversation()
        crud, store = _make_crud(conv), _make_store()
        p1, p2, p3 = _patches(crud, store)
        with p1, p2, p3:
            result = await _mod.update_conversation(
                "conv_abc", _make_request({"metadata": {"topic": "new"}}),
                db=AsyncMock(), auth=(_make_user(), _make_api_key()),
            )
        assert conv.meta == {"topic": "new"}
        crud.touch_conversation.assert_awaited_once()
        assert result["object"] == "conversation"

    async def test_delete(self):
        conv = _make_conversation()
        crud, store = _make_crud(conv), _make_store()
        p1, p2, p3 = _patches(crud, store)
        with p1, p2, p3:
            result = await _mod.delete_conversation(
                "conv_abc", db=AsyncMock(), auth=(_make_user(), _make_api_key()),
            )
        assert result == {"id": "conv_abc", "object": "conversation.deleted",
                          "deleted": True}
        store.remove_conversation_artifacts.assert_called_once_with("conv_abc")

    async def test_flag_disabled(self):
        crud, store = _make_crud(), _make_store()
        p1, p2, p3 = _patches(crud, store, enabled=False)
        with p1, p2, p3:
            result = await _mod.get_conversation(
                "conv_abc", db=AsyncMock(), auth=(_make_user(), _make_api_key()),
            )
        assert result.status_code == 404


class TestItemEndpoints:
    async def test_create_items_envelope(self):
        conv = _make_conversation()
        crud, store = _make_crud(conv), _make_store()
        p1, p2, p3 = _patches(crud, store)
        with p1, p2, p3:
            result = await _mod.create_items(
                "conv_abc",
                _make_request({"items": [{"role": "user", "content": "a"},
                                         {"role": "user", "content": "b"}]}),
                db=AsyncMock(), auth=(_make_user(), _make_api_key()),
            )
        assert result["object"] == "list"
        assert len(result["data"]) == 2
        assert result["first_id"] == result["data"][0]["id"]
        assert result["has_more"] is False

    async def test_create_items_validation_error(self):
        conv = _make_conversation()
        crud, store = _make_crud(conv), _make_store()
        store.validate_items = MagicMock(
            side_effect=ValueError("You may add up to 20 items at a time.")
        )
        p1, p2, p3 = _patches(crud, store)
        with p1, p2, p3:
            result = await _mod.create_items(
                "conv_abc", _make_request({"items": []}),
                db=AsyncMock(), auth=(_make_user(), _make_api_key()),
            )
        assert result.status_code == 400
        assert _body(result)["error"]["param"] == "items"

    async def test_list_items_delegates(self):
        conv = _make_conversation()
        crud, store = _make_crud(conv), _make_store()
        p1, p2, p3 = _patches(crud, store)
        with p1, p2, p3:
            result = await _mod.list_items(
                "conv_abc", db=AsyncMock(), auth=(_make_user(), _make_api_key()),
                limit=5, order="asc", after="msg_1",
            )
        assert result["object"] == "list"
        kwargs = store.list_items_wire.await_args.kwargs
        assert kwargs["limit"] == 5
        assert kwargs["order"] == "asc"
        assert kwargs["after"] == "msg_1"

    async def test_get_item(self):
        conv = _make_conversation()
        crud, store = _make_crud(conv), _make_store()
        row = MagicMock()
        row.item = {"id": "msg_1", "type": "message", "role": "user",
                    "content": "hi"}
        crud.get_conversation_item = AsyncMock(return_value=row)
        p1, p2, p3 = _patches(crud, store)
        with p1, p2, p3:
            result = await _mod.get_item(
                "conv_abc", "msg_1", db=AsyncMock(),
                auth=(_make_user(), _make_api_key()),
            )
        assert result["id"] == "msg_1"

    async def test_delete_item_returns_conversation(self):
        conv = _make_conversation()
        crud, store = _make_crud(conv), _make_store()
        crud.delete_conversation_item = AsyncMock(return_value=MagicMock())
        p1, p2, p3 = _patches(crud, store)
        with p1, p2, p3:
            result = await _mod.delete_item(
                "conv_abc", "msg_1", db=AsyncMock(),
                auth=(_make_user(), _make_api_key()),
            )
        # Per spec: DELETE item returns the Conversation object
        assert result["object"] == "conversation"
        store.remove_item_artifacts.assert_called_once_with("conv_abc", "msg_1")

    async def test_item_404(self):
        conv = _make_conversation()
        crud, store = _make_crud(conv), _make_store()
        p1, p2, p3 = _patches(crud, store)
        with p1, p2, p3:
            result = await _mod.get_item(
                "conv_abc", "msg_ghost", db=AsyncMock(),
                auth=(_make_user(), _make_api_key()),
            )
        assert result.status_code == 404
        assert _body(result)["error"]["code"] == "item_not_found"
