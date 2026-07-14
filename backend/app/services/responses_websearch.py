############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# responses_websearch.py: Hosted web_search tool execution for
# the OpenAI Responses API dialect.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Server-side execution of the Responses API ``web_search`` hosted tool.

OpenAI executes ``{"type": "web_search"}`` tools inside the platform.
MindRouter's backends (vLLM/Ollama) cannot, so this module runs the
agentic loop at the gateway:

1. The hosted tool is translated into a synthetic *function* tool
   (``web_search(query)``) that the backend model can call.
2. Model calls to that function are intercepted (never surfaced to the
   client), executed against MindRouter's internal search service
   (the same provider/quota stack behind ``/v1/search``), and fed back
   to the model as tool results.
3. The loop continues until the model answers, a client-owned function
   is called, or the call budget (``max_tool_calls`` or the server
   default) is exhausted — at which point the synthetic tool is removed
   so the model must answer from what it has.
4. Each executed search appears in the Response ``output`` as a
   ``web_search_call`` item, with
   ``response.web_search_call.in_progress/.searching/.completed``
   events on the streaming path.

Search failures are reported to the *model* as tool output text, never
raised — a failed search should degrade the answer, not the request.
"""

import asyncio
import json
import time
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from backend.app.core.canonical_schemas import (
    CanonicalChatRequest,
    CanonicalFunctionCall,
    CanonicalMessage,
    CanonicalToolCall,
    CanonicalToolDefinition,
    MessageRole,
)
from backend.app.core.translators.responses_in import (
    ResponsesInTranslator,
    ResponsesRequestContext,
    _gen_id,
)
from backend.app.core.translators import responses_stream as rs
from backend.app.logging_config import get_logger
from backend.app.settings import get_settings

logger = get_logger(__name__)

SEARCH_TOOL_NAME = "web_search"

WEB_SEARCH_TOOL_TYPES = {
    "web_search",
    "web_search_preview",
    "web_search_preview_2025_03_11",
}


def wants_web_search(tools: List[Dict[str, Any]]) -> bool:
    """True if the request includes a hosted web_search tool."""
    return any(
        isinstance(t, dict) and t.get("type") in WEB_SEARCH_TOOL_TYPES
        for t in tools or []
    )


def has_client_web_search_function(tools: List[Dict[str, Any]]) -> bool:
    """True if the client defined its OWN function named web_search —
    the client's tool wins and hosted execution is disabled."""
    return any(
        isinstance(t, dict)
        and t.get("type") == "function"
        and t.get("name") == SEARCH_TOOL_NAME
        for t in tools or []
    )


def synthetic_search_tool() -> CanonicalToolDefinition:
    """The function tool injected into the backend request."""
    return CanonicalToolDefinition(
        type="function",
        function={
            "name": SEARCH_TOOL_NAME,
            "description": (
                "Search the public web for current information. Returns a "
                "list of results with title, URL, and snippet. Use for "
                "facts you are unsure about or anything after your "
                "training cutoff."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    }
                },
                "required": ["query"],
            },
        },
    )


def build_web_search_call_item(query: str) -> Dict[str, Any]:
    return {
        "id": _gen_id("ws"),
        "type": "web_search_call",
        "status": "completed",
        "action": {"type": "search", "query": query},
    }


def _parse_query(arguments: str) -> str:
    try:
        parsed = json.loads(arguments or "{}")
        if isinstance(parsed, dict) and parsed.get("query"):
            return str(parsed["query"])
    except (json.JSONDecodeError, TypeError):
        pass
    return (arguments or "").strip() or "(empty query)"


def make_search_executor(db, user, api_key) -> Callable:
    """Build an async executor(query) -> result text for the model.

    Reuses the /v1/search provider registry and quota accounting.
    Never raises.
    """

    async def executor(query: str) -> str:
        try:
            from backend.app.api.search_api import (
                _check_search_quota,
                _deduct_search_tokens,
            )
            from backend.app.services.search.registry import (
                PROVIDERS,
                get_search_config,
            )

            config = await get_search_config(db)
            if not config.get("search.enabled", True):
                return "Web search is not enabled on this server."

            try:
                await _check_search_quota(db, user, config)
            except Exception as e:
                detail = getattr(e, "detail", None) or str(e)
                return f"Web search unavailable: {detail}"

            provider_key = config.get("search.provider", "brave")
            provider = PROVIDERS.get(provider_key)
            if not provider:
                return "Web search is not available (no provider configured)."

            settings = get_settings()
            max_results = min(
                int(config.get("search.max_results", 10)),
                settings.responses_web_search_max_results,
            )
            results = await provider.search(
                query, max_results=max_results, config=config
            )
            try:
                await _deduct_search_tokens(db, user, api_key, config, 0)
            except Exception:
                pass  # accounting is best-effort

            if not results:
                return f"No results found for: {query}"

            lines = []
            for i, r in enumerate(results, 1):
                snippet = (r.snippet or "").strip()
                if len(snippet) > 400:
                    snippet = snippet[:400] + "…"
                lines.append(f"{i}. {r.title}\n   URL: {r.url}\n   {snippet}")
            return f"Web search results for {query!r}:\n\n" + "\n\n".join(lines)
        except Exception as e:
            logger.warning("responses_web_search_failed", error=str(e))
            return f"Web search failed: {e}"

    return executor


def _append_round_messages(
    canonical: CanonicalChatRequest,
    content: Optional[str],
    calls: List[Dict[str, str]],
    results: List[str],
) -> None:
    """Record the model's tool-call turn and the search results in the
    conversation for the next round."""
    canonical.messages.append(
        CanonicalMessage(
            role=MessageRole.ASSISTANT,
            content=content or None,
            tool_calls=[
                CanonicalToolCall(
                    id=c["call_id"],
                    type="function",
                    function=CanonicalFunctionCall(
                        name=c["name"], arguments=c["arguments"]
                    ),
                )
                for c in calls
            ],
        )
    )
    for call, result in zip(calls, results):
        canonical.messages.append(
            CanonicalMessage(
                role=MessageRole.TOOL,
                content=result,
                tool_call_id=call["call_id"],
            )
        )


def _remove_synthetic_tool(canonical: CanonicalChatRequest) -> None:
    canonical.tools = [
        t
        for t in canonical.tools or []
        if not (t.type == "function" and t.function.get("name") == SEARCH_TOOL_NAME)
    ] or None


# ---------------------------------------------------------------------------
# Streaming orchestrator
# ---------------------------------------------------------------------------

async def stream_with_web_search(
    make_inner: Callable[[], AsyncIterator[bytes]],
    canonical: CanonicalChatRequest,
    ctx: ResponsesRequestContext,
    executor: Callable,
    max_calls: int,
    capture: Optional[Dict[str, Any]] = None,
) -> AsyncIterator[str]:
    """Multi-round Responses stream with hosted web_search execution.

    ``make_inner`` starts a fresh backend round against the (mutated)
    canonical request.  Client-visible items stream normally across all
    rounds; synthetic web_search calls are suppressed, executed, and
    surfaced as web_search_call items instead.
    """
    st = rs._StreamState()
    for f in rs.prologue_frames(st, ctx):
        yield f

    calls_made = 0
    rounds = 0
    try:
        while True:
            rounds += 1
            round_text_before = len(st.completed_output)
            async for f in rs.stream_round(
                make_inner(), st, suppress_tool=SEARCH_TOOL_NAME
            ):
                yield f

            calls = st.take_suppressed()
            if st.pending_error is not None or not calls:
                break
            if rounds > max_calls + 2:
                logger.warning("responses_web_search_round_cap", rounds=rounds)
                break

            # The round's visible text (if any) was already streamed and
            # closed into completed_output; find it for the tool-call turn.
            round_content = None
            for item in st.completed_output[round_text_before:]:
                if item.get("type") == "message":
                    round_content = item["content"][0]["text"]

            results: List[str] = []
            for call in calls:
                query = _parse_query(call["arguments"])
                st.output_index += 1
                ws_item_id = _gen_id("ws")
                yield rs._frame(
                    st.next_seq(),
                    "response.output_item.added",
                    output_index=st.output_index,
                    item={
                        "id": ws_item_id,
                        "type": "web_search_call",
                        "status": "in_progress",
                    },
                )
                yield rs._frame(
                    st.next_seq(),
                    "response.web_search_call.in_progress",
                    item_id=ws_item_id,
                    output_index=st.output_index,
                )
                yield rs._frame(
                    st.next_seq(),
                    "response.web_search_call.searching",
                    item_id=ws_item_id,
                    output_index=st.output_index,
                )
                results.append(await executor(query))
                calls_made += 1
                ws_item = {
                    "id": ws_item_id,
                    "type": "web_search_call",
                    "status": "completed",
                    "action": {"type": "search", "query": query},
                }
                yield rs._frame(
                    st.next_seq(),
                    "response.web_search_call.completed",
                    item_id=ws_item_id,
                    output_index=st.output_index,
                )
                yield rs._frame(
                    st.next_seq(),
                    "response.output_item.done",
                    output_index=st.output_index,
                    item=ws_item,
                )
                st.completed_output.append(ws_item)

            _append_round_messages(canonical, round_content, calls, results)
            if calls_made >= max_calls:
                # Budget exhausted: the model must answer next round.
                _remove_synthetic_tool(canonical)
            st.reset_for_next_round()

    except (asyncio.CancelledError, GeneratorExit):
        raise
    except Exception as e:  # noqa: BLE001 — terminal-event guarantee
        rs.apply_exception(st, e)
        for f in rs.terminal_frames(st, ctx, capture):
            yield f
        return

    rs.check_abnormal_eof(st)
    for f in rs.terminal_frames(st, ctx, capture):
        yield f


# ---------------------------------------------------------------------------
# Non-streaming orchestrator
# ---------------------------------------------------------------------------

async def run_web_search_loop(
    call_backend: Callable,
    canonical: CanonicalChatRequest,
    ctx: ResponsesRequestContext,
    executor: Callable,
    max_calls: int,
) -> Dict[str, Any]:
    """Non-streaming hosted web_search loop.  Returns the final
    Response object dict (snapshot shape)."""
    output: List[Dict[str, Any]] = []
    usage_acc = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    calls_made = 0
    rounds = 0
    status = "completed"
    incomplete = None

    while True:
        rounds += 1
        chat_response = await call_backend(canonical)

        usage = chat_response.get("usage") or {}
        for key in usage_acc:
            usage_acc[key] += usage.get(key, 0) or 0

        choices = chat_response.get("choices") or [{}]
        message = dict(choices[0].get("message") or {})
        finish_reason = choices[0].get("finish_reason")

        all_calls = message.get("tool_calls") or []
        internal = [
            tc for tc in all_calls
            if (tc.get("function") or {}).get("name") == SEARCH_TOOL_NAME
        ]
        message["tool_calls"] = [tc for tc in all_calls if tc not in internal]

        output.extend(ResponsesInTranslator.build_output_items(message))

        if not internal:
            status, incomplete = ResponsesInTranslator.map_finish_reason(
                finish_reason
            )
            break
        if rounds > max_calls + 2:
            logger.warning("responses_web_search_round_cap", rounds=rounds)
            break

        calls = [
            {
                "call_id": tc.get("id") or _gen_id("call"),
                "name": SEARCH_TOOL_NAME,
                "arguments": (tc.get("function") or {}).get("arguments") or "{}",
            }
            for tc in internal
        ]
        results = []
        for call in calls:
            query = _parse_query(call["arguments"])
            results.append(await executor(query))
            calls_made += 1
            output.append(build_web_search_call_item(query))

        _append_round_messages(canonical, message.get("content"), calls, results)
        if calls_made >= max_calls:
            _remove_synthetic_tool(canonical)

    return ResponsesInTranslator.build_snapshot(
        ctx,
        status=status,
        output=output,
        usage=ResponsesInTranslator.map_usage(usage_acc),
        incomplete_details=incomplete,
        completed_at=int(time.time()),
    )
