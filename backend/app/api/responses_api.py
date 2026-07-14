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

Server-side state (Tier 2): ``store`` (default true, matching OpenAI)
persists responses to the stored_responses table; ``previous_response_id``
rebuilds conversation context by walking the stored chain; GET/DELETE
/v1/responses/{id} and GET .../input_items are served from the store.
Codex never uses any of this over HTTP — it resends the full transcript
each turn with store=false — but the OpenAI SDKs default to it.

Unlike the older dialects, route-level errors here use the OpenAI
error envelope ``{"error": {message, type, param, code}}`` — Codex and
the SDKs parse it.
"""

import asyncio
from typing import Any, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.auth import authenticate_request
from backend.app.core.telemetry.registry import get_registry
from backend.app.core.translators.responses_in import (
    ResponsesInTranslator,
    ResponsesRequestContext,
)
from backend.app.core.translators.responses_stream import stream_responses_events
from backend.app.db import crud
from backend.app.db.models import ApiKey, User
from backend.app.db.session import get_async_db
from backend.app.logging_config import bind_request_context, get_logger
from backend.app.services import responses_store, responses_websearch
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
    ctx.user_id = user.id
    ctx.api_key_id = api_key.id
    bind_request_context(request_id=ctx.response_id, user_id=user.id)

    if ctx.background:
        return error_json(
            400, "Background responses are not supported.", param="background"
        )

    # Normalize the request's own (delta) input items; these are what
    # gets persisted for store=true rows.
    try:
        delta_items = responses_store.normalize_input_to_items(body.get("input"))
    except ValueError as e:
        return error_json(400, str(e), param="input")

    # previous_response_id: rebuild conversation context from the store.
    if ctx.previous_response_id:
        try:
            chain = await crud.get_stored_response_chain(
                db,
                ctx.previous_response_id,
                user.id,
                max_depth=settings.responses_store_max_chain_depth,
            )
            combined = responses_store.rebuild_input_from_chain(chain, delta_items)
        except ValueError as e:
            message = str(e)
            code = (
                "previous_response_not_found" if "not found" in message else None
            )
            return error_json(
                400, message, param="previous_response_id", code=code
            )
        body = {**body, "input": combined}

    # Hosted web_search: executed server-side when enabled (the hosted
    # tool type is still stripped from what the backend sees; a
    # synthetic function tool is injected instead).
    use_web_search = (
        settings.responses_web_search_enabled
        and responses_websearch.wants_web_search(ctx.tools)
        and not responses_websearch.has_client_web_search_function(ctx.tools)
    )

    stripped = [
        t for t in ctx.stripped_tool_types()
        if not (use_web_search and t in responses_websearch.WEB_SEARCH_TOOL_TYPES)
    ]
    if stripped:
        # Non-executable hosted tools are stripped, not errored, so
        # default agent configs keep working.
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

    if use_web_search:
        canonical.tools = (canonical.tools or []) + [
            responses_websearch.synthetic_search_tool()
        ]
        max_tool_calls = body.get("max_tool_calls")
        max_calls = min(
            max_tool_calls or settings.responses_web_search_max_calls, 10
        )
        executor = responses_websearch.make_search_executor(db, user, api_key)

    if ctx.stream:
        capture: dict = {}
        if use_web_search:
            events = responses_websearch.stream_with_web_search(
                lambda: service.stream_chat_completion(
                    canonical,
                    user,
                    api_key,
                    request,
                    endpoint="/v1/responses",
                    skip_quota_check=True,
                    extra_parameters=extra_parameters,
                ),
                canonical,
                ctx,
                executor,
                max_calls,
                capture=capture,
            )
        else:
            events = stream_responses_events(
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
                capture=capture,
            )

        if ctx.store:
            events = _stream_and_persist(events, ctx, delta_items, capture)

        return StreamingResponse(
            events,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Request-ID": ctx.response_id,
            },
        )

    try:
        if use_web_search:
            result = await responses_websearch.run_web_search_loop(
                lambda c: service.chat_completion(
                    c,
                    user,
                    api_key,
                    request,
                    endpoint="/v1/responses",
                    skip_quota_check=True,
                    extra_parameters=extra_parameters,
                ),
                canonical,
                ctx,
                executor,
                max_calls,
            )
        else:
            chat_response = await service.chat_completion(
                canonical,
                user,
                api_key,
                request,
                endpoint="/v1/responses",
                skip_quota_check=True,
                extra_parameters=extra_parameters,
            )
            result = ResponsesInTranslator.format_response(chat_response, ctx)
    except HTTPException as e:
        return _reshape_http_exception(e)

    if ctx.store:
        await responses_store.persist_response(
            ctx,
            delta_items,
            result.get("output") or [],
            result.get("usage"),
            result.get("status", "completed"),
            error=result.get("error"),
        )

    return JSONResponse(content=result, headers={"X-Request-ID": ctx.response_id})


async def _stream_and_persist(events, ctx, delta_items, capture):
    """Wrap the event stream so store=true rows are persisted even when
    the client disconnects mid-stream (dashboard/chat.py precedent:
    finally + asyncio.shield survives ASGI task cancellation)."""
    try:
        async for frame in events:
            yield frame
    finally:
        if capture.get("terminal"):
            persist = responses_store.persist_response(
                ctx,
                delta_items,
                capture.get("output") or [],
                capture.get("usage"),
                capture.get("status", "completed"),
                error=capture.get("error"),
            )
        else:
            # Stream aborted before a terminal event (disconnect or
            # cancellation) — still leave a row so tokens consumed have
            # a trace and any planned chain fails loudly, not silently.
            persist = responses_store.persist_response(
                ctx,
                delta_items,
                capture.get("output") or [],
                None,
                "failed",
                error={
                    "code": "server_error",
                    "message": "stream aborted before completion",
                },
            )
        try:
            await asyncio.shield(persist)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning("responses_store_stream_persist_error", error=str(e))


@router.get("/v1/responses/{response_id}")
async def get_response(
    response_id: str,
    request: Request,
    db: AsyncSession = Depends(get_async_db),
    auth: Tuple[User, ApiKey] = Depends(authenticate_request),
):
    """Retrieve a stored response."""
    user, _ = auth
    if not get_settings().responses_api_enabled:
        return error_json(404, "The Responses API is not enabled on this server.",
                          code="not_found")
    if request.query_params.get("stream") in ("true", "1"):
        return error_json(400, "Stream replay of stored responses is not supported.",
                          param="stream")

    stored = await crud.get_stored_response(db, response_id, user.id)
    if not stored:
        return error_json(
            404, f"Response with id '{response_id}' not found.",
            code="response_not_found",
        )

    ctx = ResponsesRequestContext.from_stored(stored)
    status = stored.status.value
    incomplete_details = (
        {"reason": "max_output_tokens"} if status == "incomplete" else None
    )
    snapshot = ResponsesInTranslator.build_snapshot(
        ctx,
        status=status,
        output=stored.output_items or [],
        usage=stored.usage,
        error=stored.error,
        incomplete_details=incomplete_details,
        completed_at=int(stored.updated_at.timestamp()) if stored.updated_at else None,
    )
    return JSONResponse(content=snapshot)


@router.delete("/v1/responses/{response_id}")
async def delete_response(
    response_id: str,
    db: AsyncSession = Depends(get_async_db),
    auth: Tuple[User, ApiKey] = Depends(authenticate_request),
):
    """Delete a stored response (and its offloaded artifacts)."""
    user, _ = auth
    if not get_settings().responses_api_enabled:
        return error_json(404, "The Responses API is not enabled on this server.",
                          code="not_found")

    stored = await crud.delete_stored_response(db, response_id, user.id)
    if not stored:
        return error_json(
            404, f"Response with id '{response_id}' not found.",
            code="response_not_found",
        )
    responses_store.remove_artifacts(response_id)
    return {"id": response_id, "object": "response", "deleted": True}


@router.get("/v1/responses/{response_id}/input_items")
async def list_input_items(
    response_id: str,
    db: AsyncSession = Depends(get_async_db),
    auth: Tuple[User, ApiKey] = Depends(authenticate_request),
    limit: int = Query(20, ge=1, le=100),
    order: str = Query("desc"),
    after: Optional[str] = Query(None),
):
    """List a stored response's input items (paginated)."""
    user, _ = auth
    if not get_settings().responses_api_enabled:
        return error_json(404, "The Responses API is not enabled on this server.",
                          code="not_found")

    stored = await crud.get_stored_response(db, response_id, user.id)
    if not stored:
        return error_json(
            404, f"Response with id '{response_id}' not found.",
            code="response_not_found",
        )

    items = responses_store.reinflate_images(stored)
    if order != "asc":
        items = list(reversed(items))

    if after:
        idx = next(
            (i for i, item in enumerate(items) if item.get("id") == after), None
        )
        items = items[idx + 1:] if idx is not None else []

    page = items[:limit]
    return {
        "object": "list",
        "data": page,
        "first_id": page[0].get("id") if page else None,
        "last_id": page[-1].get("id") if page else None,
        "has_more": len(items) > limit,
    }


@router.post("/v1/responses/{response_id}/cancel")
async def cancel_response(
    response_id: str,
    db: AsyncSession = Depends(get_async_db),
    auth: Tuple[User, ApiKey] = Depends(authenticate_request),
):
    """Cancel a background response — background mode is unsupported."""
    user, _ = auth
    if not get_settings().responses_api_enabled:
        return error_json(404, "The Responses API is not enabled on this server.",
                          code="not_found")

    stored = await crud.get_stored_response(db, response_id, user.id)
    if not stored:
        return error_json(
            404, f"Response with id '{response_id}' not found.",
            code="response_not_found",
        )
    return error_json(
        400,
        "Only responses created with background=true can be cancelled, "
        "and background mode is not supported on this server.",
    )
