############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_responses_api.py: Route-level unit tests for the
# /v1/responses endpoint.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Covers:
- Feature flag gating (404 when disabled)
- Request validation: invalid JSON, missing model
- Tier-1 state errors: previous_response_id, background
- Model existence check and OpenAI-shaped 404 envelope
- Pre-flight quota (429 envelope; skip_quota_check=True passed through)
- Happy paths: non-streaming Response object, streaming media type
- Service HTTPException reshaping into the OpenAI error envelope
"""

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

# ----------------------------------------------------------------
# Direct-load responses_api.py, importing its LIGHT dependencies for
# real (translators + canonical schemas have no DB import chain) and
# stubbing only the heavy leaves.  Stubs added here are removed after
# module load so later test files see a clean sys.modules.
# ----------------------------------------------------------------

# Real light imports first — ensures backend/backend.app/backend.app.core
# are genuine packages in sys.modules before any stubbing.
from backend.app.core.translators.responses_in import (  # noqa: E402
    ResponsesInTranslator,
    ResponsesRequestContext,
)

_STUB_NAMES = [
    "backend.app.api",
    "backend.app.api.auth",
    "backend.app.db",
    "backend.app.db.crud",
    "backend.app.db.models",
    "backend.app.db.session",
    "backend.app.services",
    "backend.app.services.inference",
    "backend.app.services.responses_store",
    "backend.app.core.telemetry",
    "backend.app.core.telemetry.registry",
    "backend.app.logging_config",
    "backend.app.settings",
]
_added_stubs = []
for _name in _STUB_NAMES:
    if _name not in sys.modules:
        stub = MagicMock()
        if _name == "backend.app.logging_config":
            stub.get_logger = MagicMock(return_value=MagicMock())
            stub.bind_request_context = MagicMock()
        sys.modules[_name] = stub
        _added_stubs.append(_name)

_api_dir = Path(__file__).resolve().parents[2] / "api"
_spec = importlib.util.spec_from_file_location(
    "responses_api", _api_dir / "responses_api.py",
    submodule_search_locations=[],
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

for _name in _added_stubs:
    sys.modules.pop(_name, None)

responses_endpoint = _mod.responses
count_input_tokens_endpoint = _mod.count_input_tokens
get_response_endpoint = _mod.get_response
delete_response_endpoint = _mod.delete_response
list_input_items_endpoint = _mod.list_input_items
cancel_response_endpoint = _mod.cancel_response
error_json = _mod.error_json


# ================================================================
# Helpers
# ================================================================

def _make_mock_user():
    user = MagicMock()
    user.id = 1
    return user


def _make_mock_api_key():
    api_key = MagicMock()
    api_key.id = 42
    return api_key


def _make_mock_request(body=None, raise_json=False):
    request = MagicMock()
    if raise_json:
        request.json = AsyncMock(side_effect=ValueError("bad json"))
    else:
        request.json = AsyncMock(return_value=body or {})
    request.headers = {"user-agent": "test-client/1.0"}
    request.client = MagicMock()
    request.client.host = "127.0.0.1"
    return request


_CHAT_RESPONSE = {
    "id": "internal-uuid",
    "object": "chat.completion",
    "created": 1,
    "model": "m",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "hello!"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
}


async def _fake_stream():
    yield b'data: {"choices": [{"index": 0, "delta": {"content": "hi"}}]}\n\n'
    yield b'data: {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}\n\n'
    yield b"data: [DONE]\n\n"


def _make_mock_service(streaming=False):
    service = MagicMock()
    service._check_quota = AsyncMock()
    service.chat_completion = AsyncMock(return_value=dict(_CHAT_RESPONSE))
    service.stream_chat_completion = MagicMock(return_value=_fake_stream())
    service._count_input_tokens = AsyncMock(return_value=(123, False))
    return service


def _make_mock_registry(model_exists=True):
    registry = MagicMock()
    registry.resolve_alias = MagicMock(side_effect=lambda m: (m, None))
    registry.model_exists = AsyncMock(return_value=model_exists)
    vllm_backend = MagicMock()
    vllm_backend.engine = MagicMock(value="vllm")
    registry.get_backends_with_model = AsyncMock(
        return_value=[vllm_backend] if model_exists else []
    )
    return registry


def _settings(enabled=True, web_search=False):
    settings = MagicMock()
    settings.responses_api_enabled = enabled
    settings.responses_store_max_chain_depth = 20
    settings.responses_web_search_enabled = web_search
    settings.responses_web_search_max_calls = 4
    return settings


def _body(payload):
    return json.loads(payload.body)


def _normalize_items(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [{"type": "message", "role": "user", "content": value}]
    return list(value)


def _make_mock_store():
    store = MagicMock()
    store.normalize_input_to_items = MagicMock(side_effect=_normalize_items)
    store.persist_response = AsyncMock()
    store.remove_artifacts = MagicMock()
    store.reinflate_images = MagicMock(
        side_effect=lambda stored: list(stored.input_items or [])
    )
    store.rebuild_input_from_chain = MagicMock(
        side_effect=lambda chain, new: list(new)
    )
    return store


def _make_mock_crud(chain_error="Previous response with id 'resp_x' not found."):
    crud = MagicMock()
    if chain_error:
        crud.get_stored_response_chain = AsyncMock(
            side_effect=ValueError(chain_error)
        )
    else:
        crud.get_stored_response_chain = AsyncMock(return_value=[])
    crud.get_stored_response = AsyncMock(return_value=None)
    crud.delete_stored_response = AsyncMock(return_value=None)
    return crud


def _make_mock_conv_store(history=None):
    cs = MagicMock()
    cs.load_context_items = AsyncMock(return_value=list(history or []))
    cs.append_from_response = AsyncMock()
    return cs


async def _call(body=None, raise_json=False, service=None, registry=None,
                enabled=True, store=None, crud=None, web_search=False,
                websearch_mod=None, conv_store=None):
    """Invoke the endpoint with all module-level collaborators patched."""
    service = service or _make_mock_service()
    registry = registry or _make_mock_registry()
    store = store or _make_mock_store()
    crud = crud or _make_mock_crud()
    websearch_mod = websearch_mod or _make_mock_websearch()
    conv_store = conv_store or _make_mock_conv_store()
    with patch.object(_mod, "get_settings",
                      return_value=_settings(enabled, web_search)), \
         patch.object(_mod, "get_registry", return_value=registry), \
         patch.object(_mod, "InferenceService", return_value=service), \
         patch.object(_mod, "responses_store", store), \
         patch.object(_mod, "responses_websearch", websearch_mod), \
         patch.object(_mod, "conversations_store", conv_store), \
         patch.object(_mod, "crud", crud):
        result = await responses_endpoint(
            _make_mock_request(body, raise_json=raise_json),
            db=AsyncMock(),
            auth=(_make_mock_user(), _make_mock_api_key()),
        )
    return result, service


def _make_mock_websearch(wants=False):
    ws = MagicMock()
    ws.wants_web_search = MagicMock(return_value=wants)
    ws.has_client_web_search_function = MagicMock(return_value=False)
    ws.WEB_SEARCH_TOOL_TYPES = {"web_search"}
    ws.synthetic_search_tool = MagicMock(return_value=MagicMock())
    ws.make_search_executor = MagicMock(return_value=AsyncMock())
    ws.run_web_search_loop = AsyncMock(return_value={
        "id": "resp_ws", "object": "response", "status": "completed",
        "output": [], "usage": None, "error": None,
    })
    ws.stream_with_web_search = MagicMock(return_value=_fake_stream())
    return ws


class TestValidation:
    async def test_feature_flag_disabled_404(self):
        result, _ = await _call({"model": "m", "input": "hi"}, enabled=False)
        assert isinstance(result, JSONResponse)
        assert result.status_code == 404
        assert _body(result)["error"]["code"] == "not_found"

    async def test_invalid_json_400(self):
        result, _ = await _call(raise_json=True)
        assert result.status_code == 400
        assert _body(result)["error"]["type"] == "invalid_request_error"

    async def test_missing_model_400(self):
        result, _ = await _call({"input": "hi"})
        assert result.status_code == 400
        assert _body(result)["error"]["param"] == "model"

    async def test_previous_response_id_unknown_400(self):
        # crud mock raises "not found" for the chain walk by default
        result, _ = await _call(
            {"model": "m", "input": "hi", "previous_response_id": "resp_x"}
        )
        assert result.status_code == 400
        err = _body(result)["error"]
        assert err["code"] == "previous_response_not_found"
        assert err["param"] == "previous_response_id"

    async def test_previous_response_id_chain_rebuilds_input(self):
        crud = _make_mock_crud(chain_error=None)
        store = _make_mock_store()
        store.rebuild_input_from_chain = MagicMock(
            return_value=[
                {"type": "message", "role": "user", "content": "old turn"},
                {"type": "message", "role": "user", "content": "hi"},
            ]
        )
        result, service = await _call(
            {"model": "m", "input": "hi", "previous_response_id": "resp_prev"},
            store=store, crud=crud,
        )
        assert result.status_code == 200
        canonical = service.chat_completion.call_args.args[0]
        assert len(canonical.messages) == 2
        assert canonical.messages[0].content == "old turn"

    async def test_background_400(self):
        result, _ = await _call({"model": "m", "input": "hi", "background": True})
        assert result.status_code == 400

    async def test_model_not_found_404(self):
        result, _ = await _call(
            {"model": "ghost", "input": "hi"},
            registry=_make_mock_registry(model_exists=False),
        )
        assert result.status_code == 404
        err = _body(result)["error"]
        assert err["code"] == "model_not_found"
        assert "ghost" in err["message"]

    async def test_translation_error_400(self):
        result, _ = await _call(
            {"model": "m", "input": [{"type": "item_reference", "id": "msg_1"}]}
        )
        assert result.status_code == 400
        assert "Invalid request" in _body(result)["error"]["message"]


class TestQuotaPreflight:
    async def test_quota_429_reshaped(self):
        service = _make_mock_service()
        service._check_quota = AsyncMock(
            side_effect=HTTPException(status_code=429, detail="Token quota exceeded")
        )
        result, _ = await _call({"model": "m", "input": "hi"}, service=service)
        assert result.status_code == 429
        err = _body(result)["error"]
        assert err["code"] == "insufficient_quota"

    async def test_rate_limit_429_code(self):
        service = _make_mock_service()
        service._check_quota = AsyncMock(
            side_effect=HTTPException(
                status_code=429,
                detail="Rate limit exceeded: 60 requests per minute",
            )
        )
        result, _ = await _call({"model": "m", "input": "hi"}, service=service)
        assert _body(result)["error"]["code"] == "rate_limit_exceeded"

    async def test_service_skips_second_quota_check(self):
        _, service = await _call({"model": "m", "input": "hi"})
        assert service.chat_completion.call_args.kwargs["skip_quota_check"] is True


class TestHappyPaths:
    async def test_non_streaming_response_object(self):
        result, service = await _call({"model": "m", "input": "hi"})
        assert isinstance(result, JSONResponse)
        assert result.status_code == 200
        body = _body(result)
        assert body["object"] == "response"
        assert body["id"].startswith("resp_")
        assert body["status"] == "completed"
        assert body["output"][0]["type"] == "message"
        assert body["output"][0]["content"][0]["text"] == "hello!"
        assert result.headers["x-request-id"] == body["id"]
        # Service call contract
        kwargs = service.chat_completion.call_args.kwargs
        assert kwargs["endpoint"] == "/v1/responses"
        assert kwargs["skip_quota_check"] is True
        assert kwargs["extra_parameters"]["response_id"] == body["id"]
        service._check_quota.assert_awaited_once()

    async def test_streaming_returns_event_stream(self):
        result, service = await _call({"model": "m", "input": "hi", "stream": True})
        assert isinstance(result, StreamingResponse)
        assert result.media_type == "text/event-stream"
        assert "x-request-id" in result.headers
        kwargs = service.stream_chat_completion.call_args.kwargs
        assert kwargs["endpoint"] == "/v1/responses"
        assert kwargs["skip_quota_check"] is True

    async def test_alias_resolution_applied(self):
        registry = _make_mock_registry()
        registry.resolve_alias = MagicMock(return_value=("real/model", "alias"))
        _, service = await _call({"model": "alias", "input": "hi"}, registry=registry)
        canonical = service.chat_completion.call_args.args[0]
        assert canonical.model == "real/model"


class TestServiceErrorReshaping:
    async def test_502_reshaped_to_openai_envelope(self):
        service = _make_mock_service()
        service.chat_completion = AsyncMock(
            side_effect=HTTPException(
                status_code=502, detail="All 3 backend attempts failed"
            )
        )
        result, _ = await _call({"model": "m", "input": "hi"}, service=service)
        assert result.status_code == 502
        err = _body(result)["error"]
        assert err["type"] == "server_error"
        assert err["code"] == "server_error"

    async def test_service_404_detail_passthrough(self):
        service = _make_mock_service()
        service.chat_completion = AsyncMock(
            side_effect=HTTPException(
                status_code=404,
                detail={"error": {"message": "gone", "type": "invalid_request_error",
                                  "code": "model_not_found"}},
            )
        )
        result, _ = await _call({"model": "m", "input": "hi"}, service=service)
        assert result.status_code == 404
        assert _body(result)["error"]["code"] == "model_not_found"


# ================================================================
# Tier 2: store persistence + companion endpoints
# ================================================================

def _make_stored_row(response_id="resp_stored1"):
    from datetime import datetime, timezone

    stored = MagicMock()
    stored.response_id = response_id
    stored.user_id = 1
    stored.api_key_id = 42
    stored.model = "m"
    stored.status = MagicMock()
    stored.status.value = "completed"
    stored.previous_response_id = None
    stored.instructions = None
    stored.input_items = [
        {"id": "msg_in1", "type": "message", "role": "user", "content": "hi"},
        {"id": "msg_in2", "type": "message", "role": "user", "content": "again"},
    ]
    stored.output_items = [
        {"id": "msg_out1", "type": "message", "status": "completed",
         "role": "assistant",
         "content": [{"type": "output_text", "text": "hello",
                      "annotations": [], "logprobs": []}]}
    ]
    stored.parameters = {"model": "m", "created_at": 1700000000}
    stored.usage = {"input_tokens": 1, "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 2, "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 3}
    stored.error = None
    stored.offloaded_images = None
    stored.created_at = datetime(2026, 7, 15, tzinfo=timezone.utc)
    stored.updated_at = datetime(2026, 7, 15, tzinfo=timezone.utc)
    return stored


async def _call_companion(endpoint_fn, response_id="resp_stored1", stored=None,
                          crud=None, store=None, enabled=True, **kwargs):
    crud = crud or _make_mock_crud()
    store = store or _make_mock_store()
    if stored is not None:
        crud.get_stored_response = AsyncMock(return_value=stored)
        crud.delete_stored_response = AsyncMock(return_value=stored)
    with patch.object(_mod, "get_settings", return_value=_settings(enabled)), \
         patch.object(_mod, "responses_store", store), \
         patch.object(_mod, "crud", crud):
        if endpoint_fn is get_response_endpoint:
            request = MagicMock()
            request.query_params = kwargs.pop("query_params", {})
            return await endpoint_fn(
                response_id, request, db=AsyncMock(),
                auth=(_make_mock_user(), _make_mock_api_key()),
            ), crud, store
        if endpoint_fn is list_input_items_endpoint:
            # Direct calls bypass FastAPI DI — Query defaults would leak
            # as sentinel objects, so pass every param explicitly.
            kwargs.setdefault("limit", 20)
            kwargs.setdefault("order", "desc")
            kwargs.setdefault("after", None)
        return await endpoint_fn(
            response_id, db=AsyncMock(),
            auth=(_make_mock_user(), _make_mock_api_key()), **kwargs,
        ), crud, store


class TestConversationIntegration:
    async def test_mutual_exclusion_400(self):
        result, _ = await _call({
            "model": "m", "input": "hi",
            "conversation": "conv_a", "previous_response_id": "resp_b",
        })
        assert result.status_code == 400
        assert "conjunction" in _body(result)["error"]["message"]

    async def test_unknown_conversation_404(self):
        crud = _make_mock_crud()
        crud.get_conversation = AsyncMock(return_value=None)
        result, _ = await _call(
            {"model": "m", "input": "hi", "conversation": "conv_ghost"},
            crud=crud,
        )
        assert result.status_code == 404
        assert _body(result)["error"]["code"] == "conversation_not_found"

    async def test_history_prepended_and_appended(self):
        crud = _make_mock_crud()
        crud.get_conversation = AsyncMock(return_value=MagicMock())
        conv_store = _make_mock_conv_store(history=[
            {"id": "msg_h1", "type": "message", "role": "user",
             "content": "remember: teal"},
        ])
        result, service = await _call(
            {"model": "m", "input": "what color?",
             "conversation": {"id": "conv_a"}, "store": False},
            crud=crud, conv_store=conv_store,
        )
        assert result.status_code == 200
        # History item precedes the new input in the canonical request
        canonical = service.chat_completion.call_args.args[0]
        assert canonical.messages[0].content == "remember: teal"
        assert canonical.messages[1].content == "what color?"
        # Echoed on the Response object
        assert _body(result)["conversation"] == {"id": "conv_a"}
        # Post-completion append: delta input + output items
        conv_store.append_from_response.assert_awaited_once()
        args = conv_store.append_from_response.await_args.args
        assert args[0] == "conv_a"
        assert args[2][0]["content"] == "what color?"  # delta input only
        assert args[3][0]["type"] == "message"  # output items

    async def test_item_reference_resolves_from_conversation(self):
        crud = _make_mock_crud()
        crud.get_conversation = AsyncMock(return_value=MagicMock())
        conv_store = _make_mock_conv_store(history=[
            {"id": "msg_ref", "type": "message", "role": "user",
             "content": "referenced text"},
        ])
        result, service = await _call(
            {"model": "m", "conversation": "conv_a", "store": False,
             "input": [{"type": "item_reference", "id": "msg_ref"}]},
            crud=crud, conv_store=conv_store,
        )
        assert result.status_code == 200
        canonical = service.chat_completion.call_args.args[0]
        # history + resolved reference = same content twice
        assert [m.content for m in canonical.messages] == [
            "referenced text", "referenced text",
        ]

    async def test_streaming_appends_via_wrapper(self):
        crud = _make_mock_crud()
        crud.get_conversation = AsyncMock(return_value=MagicMock())
        conv_store = _make_mock_conv_store()
        service = _make_mock_service()
        registry = _make_mock_registry()
        with patch.object(_mod, "get_settings", return_value=_settings(True)), \
             patch.object(_mod, "get_registry", return_value=registry), \
             patch.object(_mod, "InferenceService", return_value=service), \
             patch.object(_mod, "responses_store", _make_mock_store()), \
             patch.object(_mod, "responses_websearch", _make_mock_websearch()), \
             patch.object(_mod, "conversations_store", conv_store), \
             patch.object(_mod, "crud", crud):
            result = await responses_endpoint(
                _make_mock_request({"model": "m", "input": "hi",
                                    "conversation": "conv_a", "store": False,
                                    "stream": True}),
                db=AsyncMock(),
                auth=(_make_mock_user(), _make_mock_api_key()),
            )
            assert isinstance(result, StreamingResponse)
            async for _ in result.body_iterator:
                pass
        conv_store.append_from_response.assert_awaited_once()


class TestCountInputTokens:
    async def _count(self, body, service=None, registry=None, enabled=True):
        service = service or _make_mock_service()
        registry = registry or _make_mock_registry()
        with patch.object(_mod, "get_settings",
                          return_value=_settings(enabled)), \
             patch.object(_mod, "get_registry", return_value=registry), \
             patch.object(_mod, "InferenceService", return_value=service), \
             patch.object(_mod, "responses_store", _make_mock_store()), \
             patch.object(_mod, "crud", _make_mock_crud()):
            return await count_input_tokens_endpoint(
                _make_mock_request(body),
                db=AsyncMock(),
                auth=(_make_mock_user(), _make_mock_api_key()),
            ), service

    async def test_returns_spec_shape(self):
        result, service = await self._count({"model": "m", "input": "hello"})
        assert result == {"object": "response.input_tokens", "input_tokens": 123}
        canonical = service._count_input_tokens.await_args.args[0]
        assert canonical.messages[0].content == "hello"

    async def test_counts_tools_and_instructions(self):
        result, service = await self._count({
            "model": "m", "input": "hi", "instructions": "be terse",
            "tools": [{"type": "function", "name": "f", "parameters": {}}],
        })
        canonical = service._count_input_tokens.await_args.args[0]
        assert canonical.tools is not None
        assert canonical.messages[0].role.value == "system"

    async def test_missing_model_400(self):
        result, _ = await self._count({"input": "hi"})
        assert result.status_code == 400
        assert _body(result)["error"]["param"] == "model"

    async def test_unknown_model_404(self):
        result, _ = await self._count(
            {"model": "ghost", "input": "hi"},
            registry=_make_mock_registry(model_exists=False),
        )
        assert result.status_code == 404
        assert _body(result)["error"]["code"] == "model_not_found"

    async def test_flag_disabled_404(self):
        result, _ = await self._count({"model": "m", "input": "hi"}, enabled=False)
        assert result.status_code == 404


class TestWebSearchDispatch:
    async def test_hosted_web_search_routes_to_loop(self):
        ws = _make_mock_websearch(wants=True)
        result, service = await _call(
            {"model": "m", "input": "latest news?",
             "tools": [{"type": "web_search"}]},
            web_search=True, websearch_mod=ws,
        )
        assert result.status_code == 200
        ws.run_web_search_loop.assert_awaited_once()
        service.chat_completion.assert_not_awaited()

    async def test_web_search_disabled_strips_tool(self):
        ws = _make_mock_websearch(wants=True)
        result, service = await _call(
            {"model": "m", "input": "hi", "tools": [{"type": "web_search"}]},
            web_search=False, websearch_mod=ws,
        )
        assert result.status_code == 200
        ws.run_web_search_loop.assert_not_awaited()
        service.chat_completion.assert_awaited_once()

    async def test_client_function_collision_disables_hosted(self):
        ws = _make_mock_websearch(wants=True)
        ws.has_client_web_search_function = MagicMock(return_value=True)
        result, service = await _call(
            {"model": "m", "input": "hi",
             "tools": [{"type": "web_search"},
                       {"type": "function", "name": "web_search",
                        "parameters": {}}]},
            web_search=True, websearch_mod=ws,
        )
        assert result.status_code == 200
        ws.run_web_search_loop.assert_not_awaited()
        service.chat_completion.assert_awaited_once()

    async def test_streaming_web_search_dispatch(self):
        ws = _make_mock_websearch(wants=True)
        result, service = await _call(
            {"model": "m", "input": "hi", "stream": True,
             "tools": [{"type": "web_search"}]},
            web_search=True, websearch_mod=ws,
        )
        assert isinstance(result, StreamingResponse)
        ws.stream_with_web_search.assert_called_once()
        service.stream_chat_completion.assert_not_called()


class TestStorePersistence:
    async def test_non_streaming_store_true_persists(self):
        store = _make_mock_store()
        result, _ = await _call({"model": "m", "input": "hi"}, store=store)
        assert result.status_code == 200
        store.persist_response.assert_awaited_once()
        args = store.persist_response.await_args
        ctx = args.args[0]
        assert ctx.response_id == _body(result)["id"]
        assert args.args[1] == [{"type": "message", "role": "user", "content": "hi"}]
        assert args.args[4] == "completed"

    async def test_non_streaming_store_false_skips_persist(self):
        store = _make_mock_store()
        result, _ = await _call(
            {"model": "m", "input": "hi", "store": False}, store=store
        )
        assert result.status_code == 200
        store.persist_response.assert_not_awaited()

    async def test_streaming_store_true_wraps_generator(self):
        # The persist happens inside the generator's finally block, which
        # runs when the body is drained — keep the patches active for it.
        store = _make_mock_store()
        service = _make_mock_service()
        registry = _make_mock_registry()
        with patch.object(_mod, "get_settings", return_value=_settings(True)), \
             patch.object(_mod, "get_registry", return_value=registry), \
             patch.object(_mod, "InferenceService", return_value=service), \
             patch.object(_mod, "responses_store", store), \
             patch.object(_mod, "crud", _make_mock_crud()):
            result = await responses_endpoint(
                _make_mock_request({"model": "m", "input": "hi", "stream": True}),
                db=AsyncMock(),
                auth=(_make_mock_user(), _make_mock_api_key()),
            )
            assert isinstance(result, StreamingResponse)
            async for _ in result.body_iterator:
                pass
        store.persist_response.assert_awaited_once()
        # Terminal state came from the adapter's capture dict
        args = store.persist_response.await_args
        assert args.args[4] == "completed"


class TestCompanionEndpoints:
    async def test_get_response_rebuilds_snapshot(self):
        result, _, _ = await _call_companion(
            get_response_endpoint, stored=_make_stored_row()
        )
        body = _body(result)
        assert body["id"] == "resp_stored1"
        assert body["object"] == "response"
        assert body["status"] == "completed"
        assert body["output"][0]["id"] == "msg_out1"
        assert body["usage"]["total_tokens"] == 3

    async def test_get_response_404(self):
        result, _, _ = await _call_companion(get_response_endpoint)
        assert result.status_code == 404
        assert _body(result)["error"]["code"] == "response_not_found"

    async def test_get_response_stream_replay_400(self):
        result, _, _ = await _call_companion(
            get_response_endpoint, stored=_make_stored_row(),
            query_params={"stream": "true"},
        )
        assert result.status_code == 400

    async def test_delete_response(self):
        result, _, store = await _call_companion(
            delete_response_endpoint, stored=_make_stored_row()
        )
        assert result == {"id": "resp_stored1", "object": "response", "deleted": True}
        store.remove_artifacts.assert_called_once_with("resp_stored1")

    async def test_delete_response_404(self):
        result, _, _ = await _call_companion(delete_response_endpoint)
        assert result.status_code == 404

    async def test_input_items_pagination(self):
        stored = _make_stored_row()
        result, _, _ = await _call_companion(
            list_input_items_endpoint, stored=stored, limit=1, order="asc",
        )
        assert result["object"] == "list"
        assert result["data"][0]["id"] == "msg_in1"
        assert result["first_id"] == "msg_in1"
        assert result["has_more"] is True

        result2, _, _ = await _call_companion(
            list_input_items_endpoint, stored=stored, limit=10, order="asc",
            after="msg_in1",
        )
        assert [i["id"] for i in result2["data"]] == ["msg_in2"]
        assert result2["has_more"] is False

    async def test_input_items_default_desc(self):
        result, _, _ = await _call_companion(
            list_input_items_endpoint, stored=_make_stored_row(),
            limit=20, order="desc",
        )
        assert result["data"][0]["id"] == "msg_in2"

    async def test_cancel_unsupported_400(self):
        result, _, _ = await _call_companion(
            cancel_response_endpoint, stored=_make_stored_row()
        )
        assert result.status_code == 400

    async def test_cancel_unknown_404(self):
        result, _, _ = await _call_companion(cancel_response_endpoint)
        assert result.status_code == 404
