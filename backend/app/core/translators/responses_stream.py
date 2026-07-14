############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# responses_stream.py: OpenAI chat-completions SSE stream to
# OpenAI Responses API SSE event stream adapter.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Re-frame an OpenAI chat.completion.chunk SSE byte stream as
OpenAI Responses API typed SSE events.

Wire invariants this adapter guarantees (strict clients — Codex, the
OpenAI SDKs — rely on all of them):

- ``response.created`` (seq 0) then ``response.in_progress`` (seq 1)
  are emitted before anything else.
- ``response.content_part.added`` precedes the first
  ``response.output_text.delta`` of a message item.
- ``sequence_number`` is present on every event and strictly
  monotonic from 0; ``item_id`` is stable across an item's events;
  items never interleave.
- ``usage`` appears only in the terminal snapshot.
- Exactly one terminal event (``response.completed`` /
  ``response.incomplete`` / ``response.failed``) is emitted, then the
  stream closes.  There is NO ``data: [DONE]`` sentinel.
- The terminal event is emitted at inner-stream exhaustion, not at the
  finish_reason chunk: vLLM's ``include_usage`` chunk (empty choices,
  real token counts) arrives AFTER finish_reason and must land in the
  terminal snapshot.

The inner generator is always driven to natural exhaustion so that
``InferenceService.stream_chat_completion`` runs its post-[DONE]
cleanup (``_complete_streaming_request`` — accounting and backend
capacity release).  Never break out of the loop early.

The module is organised as composable round primitives
(``_StreamState`` + ``stream_round`` + ``prologue_frames`` /
``terminal_frames``) so orchestrators (the hosted web_search loop in
``services/responses_websearch.py``) can run several backend rounds
inside one Responses stream.  ``stream_responses_events`` is the
single-round composition used for plain requests.
"""

import asyncio
import json
import time
from typing import Any, AsyncIterator, Dict, List, Optional

from backend.app.core.translators.responses_in import (
    ResponsesInTranslator,
    ResponsesRequestContext,
    _gen_id,
)


def _frame(seq: int, event_type: str, **fields: Any) -> str:
    """Format one SSE frame: ``event:`` line + compact JSON ``data:`` line."""
    payload: Dict[str, Any] = {"type": event_type}
    payload.update(fields)
    payload["sequence_number"] = seq
    return f"event: {event_type}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n"


def _map_error(code: Any, message: str) -> str:
    """Map an inner error to a Responses error code Codex understands."""
    msg = (message or "").lower()
    if code == 429 or "rate limit" in msg:
        if "quota" in msg:
            return "insufficient_quota"
        return "rate_limit_exceeded"
    if "quota" in msg:
        return "insufficient_quota"
    if "context" in msg and ("length" in msg or "window" in msg or "exceed" in msg):
        return "context_length_exceeded"
    return "server_error"


class _StreamState:
    """Bookkeeping shared across all rounds of one Responses stream:
    the currently-open output item, finished item snapshots, sequence
    numbering, and pending terminal state."""

    def __init__(self) -> None:
        self.seq = 0
        self.output_index = -1  # incremented when an item opens
        self.kind: Optional[str] = None  # None | reasoning | message | fc
        self.item_id = ""
        self.text_buf = ""
        # Function-call bookkeeping.  Item opening is DEFERRED until the
        # call's name is known so internal (hosted-tool) calls can be
        # suppressed without ever emitting events for them.
        self.fc_call_id = ""
        self.fc_name = ""
        self.fc_args = ""
        self.fc_tc_index: Optional[int] = None
        self.fc_opened = False
        self.fc_suppressed = False
        self.fc_pending_frags: List[str] = []
        self.suppressed_calls: List[Dict[str, str]] = []
        self.completed_output: List[Dict[str, Any]] = []
        self.content_chars = 0  # for usage estimation
        # Terminal state, resolved at exhaustion
        self.pending_status: Optional[str] = None
        self.pending_incomplete: Optional[Dict[str, Any]] = None
        self.pending_error: Optional[Dict[str, Any]] = None
        self.harvested_usage: Optional[Dict[str, Any]] = None
        self.saw_error_frame = False

    def next_seq(self) -> int:
        seq = self.seq
        self.seq += 1
        return seq

    def take_suppressed(self) -> List[Dict[str, str]]:
        calls = self.suppressed_calls
        self.suppressed_calls = []
        return calls

    def reset_for_next_round(self) -> None:
        """Clear per-round terminal state before another backend round."""
        self.pending_status = None
        self.pending_incomplete = None

    # -- open helpers ---------------------------------------------------

    def open_item(self, kind: str, prefix: str) -> None:
        self.output_index += 1
        self.kind = kind
        self.item_id = _gen_id(prefix)
        self.text_buf = ""

    # -- close: returns the frames that end the open item ----------------

    def close_current(self) -> List[str]:
        if self.kind == "reasoning":
            return self._close_reasoning()
        if self.kind == "message":
            return self._close_message()
        if self.kind == "fc":
            return self._close_fc()
        return []

    def _close_reasoning(self) -> List[str]:
        item = {
            "id": self.item_id,
            "type": "reasoning",
            "summary": [],
            "content": [{"type": "reasoning_text", "text": self.text_buf}],
            "status": "completed",
        }
        frames = [
            _frame(
                self.next_seq(),
                "response.reasoning_text.done",
                item_id=self.item_id,
                output_index=self.output_index,
                content_index=0,
                text=self.text_buf,
            ),
            _frame(
                self.next_seq(),
                "response.output_item.done",
                output_index=self.output_index,
                item=item,
            ),
        ]
        self.completed_output.append(item)
        self.kind = None
        return frames

    def _close_message(self) -> List[str]:
        part = {
            "type": "output_text",
            "text": self.text_buf,
            "annotations": [],
            "logprobs": [],
        }
        item = {
            "id": self.item_id,
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [part],
        }
        frames = [
            _frame(
                self.next_seq(),
                "response.output_text.done",
                item_id=self.item_id,
                output_index=self.output_index,
                content_index=0,
                text=self.text_buf,
                logprobs=[],
            ),
            _frame(
                self.next_seq(),
                "response.content_part.done",
                item_id=self.item_id,
                output_index=self.output_index,
                content_index=0,
                part=part,
            ),
            _frame(
                self.next_seq(),
                "response.output_item.done",
                output_index=self.output_index,
                item=item,
            ),
        ]
        self.completed_output.append(item)
        self.kind = None
        return frames

    def _open_fc_frames(self) -> List[str]:
        """Emit the deferred output_item.added + buffered arg deltas."""
        self.output_index += 1
        self.item_id = _gen_id("fc")
        self.fc_opened = True
        frames = [
            _frame(
                self.next_seq(),
                "response.output_item.added",
                output_index=self.output_index,
                item={
                    "id": self.item_id,
                    "type": "function_call",
                    "status": "in_progress",
                    "arguments": "",
                    "call_id": self.fc_call_id,
                    "name": self.fc_name,
                },
            )
        ]
        for frag in self.fc_pending_frags:
            frames.append(
                _frame(
                    self.next_seq(),
                    "response.function_call_arguments.delta",
                    item_id=self.item_id,
                    output_index=self.output_index,
                    delta=frag,
                )
            )
        self.fc_pending_frags = []
        return frames

    def _close_fc(self) -> List[str]:
        self.kind = None
        self.fc_tc_index = None
        if self.fc_suppressed:
            self.suppressed_calls.append(
                {
                    "call_id": self.fc_call_id,
                    "name": self.fc_name,
                    "arguments": self.fc_args or "{}",
                }
            )
            self.fc_suppressed = False
            self.fc_pending_frags = []
            return []

        frames: List[str] = []
        if not self.fc_opened:
            # Name never arrived — open now so the item is well-formed.
            frames.extend(self._open_fc_frames())
        item = {
            "id": self.item_id,
            "type": "function_call",
            "status": "completed",
            "call_id": self.fc_call_id,
            "name": self.fc_name,
            "arguments": self.fc_args or "{}",
        }
        frames.extend(
            [
                _frame(
                    self.next_seq(),
                    "response.function_call_arguments.done",
                    item_id=self.item_id,
                    output_index=self.output_index,
                    name=self.fc_name,
                    arguments=self.fc_args or "{}",
                ),
                _frame(
                    self.next_seq(),
                    "response.output_item.done",
                    output_index=self.output_index,
                    item=item,
                ),
            ]
        )
        self.completed_output.append(item)
        self.fc_opened = False
        return frames


def prologue_frames(st: _StreamState, ctx: ResponsesRequestContext) -> List[str]:
    """response.created + response.in_progress, emitted before anything."""
    snapshot = ResponsesInTranslator.build_snapshot
    return [
        _frame(
            st.next_seq(),
            "response.created",
            response=snapshot(ctx, status="in_progress", output=[], usage=None),
        ),
        _frame(
            st.next_seq(),
            "response.in_progress",
            response=snapshot(ctx, status="in_progress", output=[], usage=None),
        ),
    ]


def terminal_frames(
    st: _StreamState,
    ctx: ResponsesRequestContext,
    capture: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Close any open item and emit the single terminal event."""
    frames = st.close_current()
    usage = st.harvested_usage
    if usage is not None:
        usage = ResponsesInTranslator.map_usage(usage)
    elif st.pending_error is None:
        est = max(1, st.content_chars // 4) if st.content_chars else 0
        usage = {
            "input_tokens": 0,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": est,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": est,
        }

    if st.pending_error is not None:
        event_type, status = "response.failed", "failed"
    elif st.pending_status == "incomplete":
        event_type, status = "response.incomplete", "incomplete"
    else:
        event_type, status = "response.completed", "completed"

    if capture is not None:
        capture["output"] = st.completed_output
        capture["usage"] = usage
        capture["status"] = status
        capture["error"] = st.pending_error
        capture["terminal"] = True

    frames.append(
        _frame(
            st.next_seq(),
            event_type,
            response=ResponsesInTranslator.build_snapshot(
                ctx,
                status=status,
                output=st.completed_output,
                usage=usage,
                error=st.pending_error,
                incomplete_details=st.pending_incomplete,
                completed_at=int(time.time()),
            ),
        )
    )
    return frames


async def stream_round(
    inner: AsyncIterator[bytes],
    st: _StreamState,
    suppress_tool: Optional[str] = None,
) -> AsyncIterator[str]:
    """Run one backend round: parse the inner chat-SSE byte stream to
    exhaustion, yielding Responses event frames and updating ``st``.

    Tool calls whose function name equals ``suppress_tool`` produce NO
    events — they accumulate in ``st.suppressed_calls`` for a hosted-tool
    orchestrator to execute between rounds.
    """
    async for chunk_bytes in inner:
        # After the terminal condition is known we still drain the
        # inner generator (harvesting late usage) so its cleanup runs.
        chunk_str = (
            chunk_bytes.decode("utf-8")
            if isinstance(chunk_bytes, bytes)
            else chunk_bytes
        )

        for line in chunk_str.strip().split("\n"):
            line = line.strip()
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                continue
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            # Inner error frame (backend 4xx before first chunk).
            if "error" in data and "choices" not in data:
                err = data.get("error") or {}
                st.pending_error = {
                    "code": _map_error(err.get("code"), str(err.get("message"))),
                    "message": str(err.get("message") or "backend error"),
                }
                st.saw_error_frame = True
                continue

            # Harvest usage wherever it appears — vLLM's include_usage
            # chunk has empty choices and arrives after finish_reason.
            if data.get("usage"):
                st.harvested_usage = data["usage"]

            choices = data.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}
            reasoning_delta = delta.get("reasoning_content")
            content_delta = delta.get("content")
            tool_deltas = delta.get("tool_calls")
            finish_reason = choice.get("finish_reason")

            if st.pending_status is not None or st.pending_error is not None:
                continue  # already finished; draining

            if reasoning_delta:
                if st.kind != "reasoning":
                    for f in st.close_current():
                        yield f
                    st.open_item("reasoning", "rs")
                    yield _frame(
                        st.next_seq(),
                        "response.output_item.added",
                        output_index=st.output_index,
                        item={
                            "id": st.item_id,
                            "type": "reasoning",
                            "summary": [],
                        },
                    )
                st.text_buf += reasoning_delta
                yield _frame(
                    st.next_seq(),
                    "response.reasoning_text.delta",
                    item_id=st.item_id,
                    output_index=st.output_index,
                    content_index=0,
                    delta=reasoning_delta,
                )

            if content_delta:
                if st.kind != "message":
                    for f in st.close_current():
                        yield f
                    st.open_item("message", "msg")
                    yield _frame(
                        st.next_seq(),
                        "response.output_item.added",
                        output_index=st.output_index,
                        item={
                            "id": st.item_id,
                            "type": "message",
                            "status": "in_progress",
                            "content": [],
                            "role": "assistant",
                        },
                    )
                    yield _frame(
                        st.next_seq(),
                        "response.content_part.added",
                        item_id=st.item_id,
                        output_index=st.output_index,
                        content_index=0,
                        part={
                            "type": "output_text",
                            "text": "",
                            "annotations": [],
                            "logprobs": [],
                        },
                    )
                st.text_buf += content_delta
                st.content_chars += len(content_delta)
                yield _frame(
                    st.next_seq(),
                    "response.output_text.delta",
                    item_id=st.item_id,
                    output_index=st.output_index,
                    content_index=0,
                    delta=content_delta,
                    logprobs=[],
                )

            if tool_deltas:
                for tc in tool_deltas:
                    tc_index = tc.get("index", 0)
                    tc_func = tc.get("function") or {}
                    if st.kind != "fc" or st.fc_tc_index != tc_index:
                        for f in st.close_current():
                            yield f
                        st.kind = "fc"
                        st.fc_tc_index = tc_index
                        st.fc_call_id = tc.get("id") or _gen_id("call")
                        st.fc_name = tc_func.get("name") or ""
                        st.fc_args = ""
                        st.fc_opened = False
                        st.fc_suppressed = False
                        st.fc_pending_frags = []
                    # Fragments may deliver id/name after the first
                    # chunk; keep the freshest values.
                    if tc.get("id"):
                        st.fc_call_id = tc["id"]
                    if tc_func.get("name"):
                        st.fc_name = tc_func["name"]

                    # Item opening is deferred until the name is known
                    # so hosted-tool calls never emit events.
                    if st.fc_name and not st.fc_opened and not st.fc_suppressed:
                        if suppress_tool and st.fc_name == suppress_tool:
                            st.fc_suppressed = True
                        else:
                            for f in st._open_fc_frames():
                                yield f

                    args_fragment = tc_func.get("arguments") or ""
                    if args_fragment:
                        st.fc_args += args_fragment
                        if st.fc_opened:
                            yield _frame(
                                st.next_seq(),
                                "response.function_call_arguments.delta",
                                item_id=st.item_id,
                                output_index=st.output_index,
                                delta=args_fragment,
                            )
                        elif not st.fc_suppressed:
                            st.fc_pending_frags.append(args_fragment)

            if finish_reason:
                for f in st.close_current():
                    yield f
                (
                    st.pending_status,
                    st.pending_incomplete,
                ) = ResponsesInTranslator.map_finish_reason(finish_reason)


def apply_exception(st: _StreamState, e: BaseException) -> None:
    """Record a raised exception as the stream's terminal error."""
    status_code = getattr(e, "status_code", None)
    detail = getattr(e, "detail", None)
    message = str(detail) if detail is not None else str(e) or "internal error"
    st.pending_error = {
        "code": _map_error(status_code, message),
        "message": message,
    }


def check_abnormal_eof(st: _StreamState) -> None:
    """Flag a stream that died mid-generation as failed rather than
    presenting a truncated answer as completed.

    A healthy inner stream always ends with a finish_reason chunk (and
    usually usage); content with neither means the backend died.
    """
    if (
        st.pending_status is None
        and st.pending_error is None
        and (st.kind is not None or st.completed_output)
        and not st.saw_error_frame
        and st.harvested_usage is None
    ):
        st.pending_error = {
            "code": "server_error",
            "message": "upstream stream ended unexpectedly",
        }


async def stream_responses_events(
    inner: AsyncIterator[bytes],
    ctx: ResponsesRequestContext,
    capture: Optional[Dict[str, Any]] = None,
) -> AsyncIterator[str]:
    """Single-round adaptation of the OpenAI-chunk SSE byte stream from
    ``InferenceService.stream_chat_completion`` into Responses events.

    ``capture``, when provided, is filled with the terminal state
    (output, usage, status, error) as the terminal event is emitted —
    the store=true persistence path reads it from a finally block.
    """
    st = _StreamState()

    for f in prologue_frames(st, ctx):
        yield f

    try:
        async for f in stream_round(inner, st):
            yield f
    except (asyncio.CancelledError, GeneratorExit):
        # Client disconnect or task cancellation (worker shutdown).
        # GeneratorExit forbids further yields.
        raise
    except Exception as e:  # noqa: BLE001 — terminal-event guarantee
        # Includes HTTPException raised by the inner generator.  EOF
        # without a terminal event makes Codex re-POST the entire
        # request up to 5 times — always emit response.failed instead.
        apply_exception(st, e)
        for f in terminal_frames(st, ctx, capture):
            yield f
        return

    check_abnormal_eof(st)
    for f in terminal_frames(st, ctx, capture):
        yield f
