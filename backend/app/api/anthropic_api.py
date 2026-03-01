############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# anthropic_api.py: Anthropic Messages API compatible endpoint
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Anthropic Messages API compatible endpoint."""

import json
import uuid
from typing import AsyncIterator, Tuple

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.auth import authenticate_request
from backend.app.core.telemetry.registry import get_registry
from backend.app.core.translators.anthropic_in import AnthropicInTranslator
from backend.app.db.models import ApiKey, User
from backend.app.db.session import get_async_db
from backend.app.logging_config import bind_request_context, get_logger
from backend.app.services.inference import InferenceService

logger = get_logger(__name__)
router = APIRouter(prefix="/anthropic", tags=["anthropic"])


@router.post("/v1/messages")
async def messages(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
    auth: Tuple[User, ApiKey] = Depends(authenticate_request),
):
    """
    Anthropic Messages API compatible endpoint.

    Clients configure: base_url="https://mindrouter.example.edu/anthropic"

    Supports:
    - Streaming and non-streaming responses
    - Multi-turn conversations
    - Vision/multimodal inputs (base64 and URL)
    - System prompts (string and content block array)
    - Thinking/reasoning mode
    - Structured outputs via output_config
    """
    user, api_key = auth

    # Parse request body
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON body",
        )

    # Generate Anthropic-style request ID
    request_id = f"msg_{uuid.uuid4().hex[:24]}"
    bind_request_context(request_id=request_id, user_id=user.id)

    # Translate to canonical format
    try:
        canonical = AnthropicInTranslator.translate_messages_request(body)
        canonical.request_id = request_id
        canonical.user_id = user.id
        canonical.api_key_id = api_key.id
    except Exception as e:
        logger.warning("anthropic_translation_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request format: {str(e)}",
        )

    model = body.get("model", canonical.model)

    # Early model validation — reject unknown models before queuing
    registry = get_registry()
    if not await registry.model_exists(canonical.model):
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "type": "error",
                "error": {
                    "type": "not_found_error",
                    "message": f"model: {canonical.model}",
                },
            },
        )

    # Create inference service
    service = InferenceService(db)

    # Handle streaming vs non-streaming
    if canonical.stream:
        return StreamingResponse(
            _stream_anthropic_events(
                service.stream_chat_completion(canonical, user, api_key, request),
                request_id,
                model,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Request-ID": request_id,
            },
        )
    else:
        response = await service.chat_completion(canonical, user, api_key, request)
        return AnthropicInTranslator.format_response(response, model)


async def _stream_anthropic_events(
    openai_stream: AsyncIterator[bytes],
    request_id: str,
    model: str,
) -> AsyncIterator[str]:
    """Convert OpenAI SSE stream to Anthropic SSE stream.

    Takes the OpenAI-format SSE byte stream from InferenceService and
    re-emits it as Anthropic Messages API streaming events.

    Handles both text content and tool_use blocks. Text block emission is
    deferred until actual text content arrives, allowing tool-call-only
    responses to work correctly.
    """
    fmt = AnthropicInTranslator.format_stream_event

    # Emit message_start
    yield fmt("message_start", {
        "type": "message_start",
        "message": {
            "id": request_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })

    output_tokens = 0
    content_block_index = 0
    thinking_block_started = False
    text_block_started = False
    finished = False
    # Track active tool blocks: tool_call_index → content_block_index
    active_tool_blocks: dict = {}
    # Accumulate tool call argument fragments per tool call index
    tool_arg_buffers: dict = {}

    async for chunk_bytes in openai_stream:
        # After emitting message_stop, keep consuming the inner stream
        # so that stream_chat_completion() can run its cleanup code
        # (_complete_streaming_request). Without this, the inner generator
        # is abandoned and backend capacity is never released.
        if finished:
            continue

        chunk_str = chunk_bytes.decode("utf-8") if isinstance(chunk_bytes, bytes) else chunk_bytes

        # Process each SSE line
        for line in chunk_str.strip().split("\n"):
            line = line.strip()
            if not line.startswith("data: "):
                continue

            data_str = line[6:]  # Strip "data: " prefix

            if data_str == "[DONE]":
                continue

            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            # Extract delta content from OpenAI chunk
            choices = data.get("choices", [])
            if not choices:
                continue

            choice = choices[0]
            delta = choice.get("delta", {})
            content = delta.get("content")
            reasoning_content = delta.get("reasoning_content")
            tool_calls = delta.get("tool_calls")
            finish_reason = choice.get("finish_reason")

            # Handle reasoning/thinking content
            if reasoning_content:
                if not thinking_block_started:
                    yield fmt("content_block_start", {
                        "type": "content_block_start",
                        "index": content_block_index,
                        "content_block": {"type": "thinking", "thinking": ""},
                    })
                    thinking_block_started = True

                yield fmt("content_block_delta", {
                    "type": "content_block_delta",
                    "index": content_block_index,
                    "delta": {"type": "thinking_delta", "thinking": reasoning_content},
                })

            # Handle text content
            if content:
                if not text_block_started:
                    # Close thinking block if it was open
                    if thinking_block_started:
                        yield fmt("content_block_stop", {
                            "type": "content_block_stop",
                            "index": content_block_index,
                        })
                        content_block_index += 1
                        thinking_block_started = False

                    yield fmt("content_block_start", {
                        "type": "content_block_start",
                        "index": content_block_index,
                        "content_block": {"type": "text", "text": ""},
                    })
                    text_block_started = True

                output_tokens += 1  # Approximate token count
                yield fmt("content_block_delta", {
                    "type": "content_block_delta",
                    "index": content_block_index,
                    "delta": {"type": "text_delta", "text": content},
                })

            # Handle tool calls
            if tool_calls:
                for tc_delta in tool_calls:
                    tc_index = tc_delta.get("index", 0)
                    tc_func = tc_delta.get("function", {})

                    if tc_index not in active_tool_blocks:
                        # New tool call — close text block if open
                        if text_block_started:
                            yield fmt("content_block_stop", {
                                "type": "content_block_stop",
                                "index": content_block_index,
                            })
                            content_block_index += 1
                            text_block_started = False

                        # Assign a content block index
                        active_tool_blocks[tc_index] = content_block_index
                        tool_arg_buffers[tc_index] = ""
                        content_block_index += 1

                        # Emit content_block_start for this tool_use
                        yield fmt("content_block_start", {
                            "type": "content_block_start",
                            "index": active_tool_blocks[tc_index],
                            "content_block": {
                                "type": "tool_use",
                                "id": tc_delta.get("id", f"toolu_{tc_index}"),
                                "name": tc_func.get("name", ""),
                                "input": {},
                            },
                        })

                    # Emit argument fragments as input_json_delta
                    args_fragment = tc_func.get("arguments", "")
                    if args_fragment:
                        tool_arg_buffers[tc_index] += args_fragment
                        yield fmt("content_block_delta", {
                            "type": "content_block_delta",
                            "index": active_tool_blocks[tc_index],
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": args_fragment,
                            },
                        })

            if finish_reason:
                stop_reason = AnthropicInTranslator._map_finish_reason(finish_reason)

                # Close text block if still open (text block is always at index 0 when present)
                if text_block_started:
                    yield fmt("content_block_stop", {
                        "type": "content_block_stop",
                        "index": 0,
                    })

                # Close all open tool blocks
                for tc_idx in sorted(active_tool_blocks.keys()):
                    yield fmt("content_block_stop", {
                        "type": "content_block_stop",
                        "index": active_tool_blocks[tc_idx],
                    })

                # Emit message_delta with stop_reason and usage
                usage = data.get("usage", {})
                yield fmt("message_delta", {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": stop_reason,
                        "stop_sequence": None,
                    },
                    "usage": {
                        "output_tokens": usage.get("completion_tokens", output_tokens),
                    },
                })

                # Emit message_stop
                yield fmt("message_stop", {"type": "message_stop"})
                finished = True

    # If stream ends without explicit finish_reason, close gracefully
    if not finished:
        if text_block_started:
            yield fmt("content_block_stop", {
                "type": "content_block_stop",
                "index": 0,
            })
        for tc_idx in sorted(active_tool_blocks.keys()):
            yield fmt("content_block_stop", {
                "type": "content_block_stop",
                "index": active_tool_blocks[tc_idx],
            })
        yield fmt("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        })
        yield fmt("message_stop", {"type": "message_stop"})
