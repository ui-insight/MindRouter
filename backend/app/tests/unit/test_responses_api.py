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
    "backend.app.db.models",
    "backend.app.db.session",
    "backend.app.services",
    "backend.app.services.inference",
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
    return service


def _make_mock_registry(model_exists=True):
    registry = MagicMock()
    registry.resolve_alias = MagicMock(side_effect=lambda m: (m, None))
    registry.model_exists = AsyncMock(return_value=model_exists)
    return registry


def _settings(enabled=True):
    settings = MagicMock()
    settings.responses_api_enabled = enabled
    return settings


def _body(payload):
    return json.loads(payload.body)


async def _call(body=None, raise_json=False, service=None, registry=None,
                enabled=True):
    """Invoke the endpoint with all module-level collaborators patched."""
    service = service or _make_mock_service()
    registry = registry or _make_mock_registry()
    with patch.object(_mod, "get_settings", return_value=_settings(enabled)), \
         patch.object(_mod, "get_registry", return_value=registry), \
         patch.object(_mod, "InferenceService", return_value=service):
        result = await responses_endpoint(
            _make_mock_request(body, raise_json=raise_json),
            db=AsyncMock(),
            auth=(_make_mock_user(), _make_mock_api_key()),
        )
    return result, service


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

    async def test_previous_response_id_400(self):
        result, _ = await _call(
            {"model": "m", "input": "hi", "previous_response_id": "resp_x"}
        )
        assert result.status_code == 400
        err = _body(result)["error"]
        assert err["code"] == "previous_response_not_found"
        assert err["param"] == "previous_response_id"

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
