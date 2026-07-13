############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# responses_api.py: OpenAI Responses API compatible endpoint
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""OpenAI Responses API compatible endpoint (/v1/responses).

Serves the OpenAI python/node SDKs and — the primary driver — the
Codex agent in the ChatGPT desktop app, configured as a custom
provider:

    [model_providers.mindrouter]
    base_url = "https://<host>/v1"
    env_key = "MINDROUTER_API_KEY"
    wire_api = "responses"
    requires_openai_auth = false

Tier 1 is stateless: ``store`` is accepted and echoed but nothing
persists, and ``previous_response_id`` returns the documented
``previous_response_not_found`` error (Codex never sends it over HTTP —
it resends the full transcript each turn).

Unlike the older dialects, route-level errors here use the OpenAI
error envelope ``{"error": {message, type, param, code}}`` — Codex and
the SDKs parse it.
"""

from typing import Any, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.auth import authenticate_request
from backend.app.core.telemetry.registry import get_registry
from backend.app.core.translators.responses_in import (
    ResponsesInTranslator,
    ResponsesRequestContext,
)
from backend.app.core.translators.responses_stream import stream_responses_events
from backend.app.db.models import ApiKey, User
from backend.app.db.session import get_async_db
from backend.app.logging_config import bind_request_context, get_logger
from backend.app.services.inference import InferenceService
from backend.app.settings import get_settings

logger = get_logger(__name__)
router = APIRouter(tags=["responses"])


def error_json(
    status_code: int,
    message: str,
    err_type: str = "invalid_request_error",
    code: Optional[str] = None,
    param: Optional[str] = None,
) -> JSONResponse:
    """OpenAI-shaped error envelope."""
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": err_type,
                "param": param,
                "code": code,
            }
        },
    )


def _reshape_http_exception(e: HTTPException) -> JSONResponse:
    """Re-shape a service HTTPException into the OpenAI error envelope."""
    detail = e.detail
    # Service 404s already carry an OpenAI-style {"error": {...}} detail.
    if isinstance(detail, dict) and "error" in detail:
        return JSONResponse(status_code=e.status_code, content=detail)
    message = detail if isinstance(detail, str) else str(detail)
    if e.status_code == 429:
        code = "insufficient_quota" if "quota" in message.lower() else "rate_limit_exceeded"
        return error_json(429, message, err_type="requests", code=code)
    if e.status_code >= 500:
        return error_json(e.status_code, message, err_type="server_error", code="server_error")
    return error_json(e.status_code, message)


@router.post("/v1/responses")
async def responses(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
    auth: Tuple[User, ApiKey] = Depends(authenticate_request),
):
    """
    OpenAI Responses API compatible endpoint.

    Supports:
    - Streaming (typed SSE events) and non-streaming
    - String and item-array ``input`` (messages, function_call,
      function_call_output, reasoning items)
    - Function tools (flat Responses shape) with round-trip via call_id
    - Reasoning models (reasoning items / reasoning_text deltas)
    - instructions, text.format structured output, vision inputs
    """
    user, api_key = auth

    settings = get_settings()
    if not settings.responses_api_enabled:
        return error_json(
            404, "The Responses API is not enabled on this server.", code="not_found"
        )

    try:
        body = await request.json()
    except Exception:
        return error_json(400, "Invalid JSON body")

    if not isinstance(body, dict) or not body.get("model"):
        return error_json(400, "you must provide a model parameter", param="model")

    ctx = ResponsesRequestContext.from_body(body)
    bind_request_context(request_id=ctx.response_id, user_id=user.id)

    # Tier 1 has no server-side response store.
    if ctx.previous_response_id:
        return error_json(
            400,
            f"Previous response with id '{ctx.previous_response_id}' not found.",
            param="previous_response_id",
            code="previous_response_not_found",
        )
    if ctx.background:
        return error_json(
            400, "Background responses are not supported.", param="background"
        )

    stripped = ctx.stripped_tool_types()
    if stripped:
        # Codex sends a web_search tool by default; stripping (not
        # erroring) keeps default configs working.
        logger.info("responses_tools_stripped", types=stripped, user_id=user.id)

    try:
        canonical = ResponsesInTranslator.translate_responses_request(body)
        canonical.request_id = ctx.response_id
        canonical.user_id = user.id
        canonical.api_key_id = api_key.id
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("responses_translation_error", error=str(e))
        return error_json(400, f"Invalid request: {str(e)}")

    # Resolve alias for routing; echo the raw client model string.
    registry = get_registry()
    canonical.model, _ = registry.resolve_alias(canonical.model)
    if not await registry.model_exists(canonical.model):
        logger.warning(
            "responses_model_not_found",
            model=canonical.model,
            raw_model=body.get("model"),
        )
        return error_json(
            404,
            f"The model '{ctx.model}' does not exist or you do not have access to it.",
            code="model_not_found",
        )

    service = InferenceService(db)

    # Pre-flight quota exactly once, before HTTP 200 headers are
    # committed (the service-internal check runs on first generator
    # iteration — too late for a clean 429, and check_rpm increments
    # a Redis counter so it must not run twice).
    try:
        await service._check_quota(user, api_key)
    except HTTPException as e:
        return _reshape_http_exception(e)

    extra_parameters = {"response_id": ctx.response_id}

    if ctx.stream:
        return StreamingResponse(
            stream_responses_events(
                service.stream_chat_completion(
                    canonical,
                    user,
                    api_key,
                    request,
                    endpoint="/v1/responses",
                    skip_quota_check=True,
                    extra_parameters=extra_parameters,
                ),
                ctx,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Request-ID": ctx.response_id,
            },
        )

    try:
        chat_response = await service.chat_completion(
            canonical,
            user,
            api_key,
            request,
            endpoint="/v1/responses",
            skip_quota_check=True,
            extra_parameters=extra_parameters,
        )
    except HTTPException as e:
        return _reshape_http_exception(e)

    result = ResponsesInTranslator.format_response(chat_response, ctx)
    return JSONResponse(content=result, headers={"X-Request-ID": ctx.response_id})
