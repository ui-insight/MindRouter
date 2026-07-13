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


class _ItemState:
    """Bookkeeping for the currently-open output item and the finished
    item snapshots accumulated for the terminal response."""

    def __init__(self) -> None:
        self.seq = 0
        self.output_index = -1  # incremented when an item opens
        self.kind: Optional[str] = None  # None | reasoning | message | fc
        self.item_id = ""
        self.text_buf = ""
        self.fc_call_id = ""
        self.fc_name = ""
        self.fc_args = ""
        self.fc_tc_index: Optional[int] = None
        self.completed_output: List[Dict[str, Any]] = []
        self.content_chars = 0  # for usage estimation

    def next_seq(self) -> int:
        seq = self.seq
        self.seq += 1
        return seq

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

    def _close_fc(self) -> List[str]:
        item = {
            "id": self.item_id,
            "type": "function_call",
            "status": "completed",
            "call_id": self.fc_call_id,
            "name": self.fc_name,
            "arguments": self.fc_args or "{}",
        }
        frames = [
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
        self.completed_output.append(item)
        self.kind = None
        self.fc_tc_index = None
        return frames


async def stream_responses_events(
    inner: AsyncIterator[bytes],
    ctx: ResponsesRequestContext,
) -> AsyncIterator[str]:
    """Adapt the OpenAI-chunk SSE byte stream from
    ``InferenceService.stream_chat_completion`` into Responses events."""
    st = _ItemState()
    snapshot = ResponsesInTranslator.build_snapshot

    # Prologue — emitted before the inner stream produces anything so
    # clients see activity immediately.
    yield _frame(
        st.next_seq(),
        "response.created",
        response=snapshot(ctx, status="in_progress", output=[], usage=None),
    )
    yield _frame(
        st.next_seq(),
        "response.in_progress",
        response=snapshot(ctx, status="in_progress", output=[], usage=None),
    )

    # Pending terminal state, resolved at inner-stream exhaustion.
    pending_status: Optional[str] = None
    pending_incomplete: Optional[Dict[str, Any]] = None
    pending_error: Optional[Dict[str, Any]] = None
    harvested_usage: Optional[Dict[str, Any]] = None
    saw_error_frame = False

    def terminal_frames() -> List[str]:
        """Close any open item and emit the single terminal event."""
        frames = st.close_current()
        usage = harvested_usage
        if usage is not None:
            usage = ResponsesInTranslator.map_usage(usage)
        elif pending_error is None:
            usage = {
                "input_tokens": 0,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": max(1, st.content_chars // 4)
                if st.content_chars
                else 0,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": max(1, st.content_chars // 4)
                if st.content_chars
                else 0,
            }

        if pending_error is not None:
            event_type, status = "response.failed", "failed"
        elif pending_status == "incomplete":
            event_type, status = "response.incomplete", "incomplete"
        else:
            event_type, status = "response.completed", "completed"

        frames.append(
            _frame(
                st.next_seq(),
                event_type,
                response=snapshot(
                    ctx,
                    status=status,
                    output=st.completed_output,
                    usage=usage,
                    error=pending_error,
                    incomplete_details=pending_incomplete,
                    completed_at=int(time.time()),
                ),
            )
        )
        return frames

    try:
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
                    pending_error = {
                        "code": _map_error(err.get("code"), str(err.get("message"))),
                        "message": str(err.get("message") or "backend error"),
                    }
                    saw_error_frame = True
                    continue

                # Harvest usage wherever it appears — vLLM's
                # include_usage chunk has empty choices and arrives
                # after finish_reason.
                if data.get("usage"):
                    harvested_usage = data["usage"]

                choices = data.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta") or {}
                reasoning_delta = delta.get("reasoning_content")
                content_delta = delta.get("content")
                tool_deltas = delta.get("tool_calls")
                finish_reason = choice.get("finish_reason")

                if pending_status is not None or pending_error is not None:
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
                            st.open_item("fc", "fc")
                            st.fc_tc_index = tc_index
                            st.fc_call_id = tc.get("id") or _gen_id("call")
                            st.fc_name = tc_func.get("name") or ""
                            st.fc_args = ""
                            yield _frame(
                                st.next_seq(),
                                "response.output_item.added",
                                output_index=st.output_index,
                                item={
                                    "id": st.item_id,
                                    "type": "function_call",
                                    "status": "in_progress",
                                    "arguments": "",
                                    "call_id": st.fc_call_id,
                                    "name": st.fc_name,
                                },
                            )
                        # Fragments may deliver id/name after the first
                        # chunk; keep the freshest values for .done.
                        if tc.get("id"):
                            st.fc_call_id = tc["id"]
                        if tc_func.get("name"):
                            st.fc_name = tc_func["name"]
                        args_fragment = tc_func.get("arguments") or ""
                        if args_fragment:
                            st.fc_args += args_fragment
                            yield _frame(
                                st.next_seq(),
                                "response.function_call_arguments.delta",
                                item_id=st.item_id,
                                output_index=st.output_index,
                                delta=args_fragment,
                            )

                if finish_reason:
                    for f in st.close_current():
                        yield f
                    (
                        pending_status,
                        pending_incomplete,
                    ) = ResponsesInTranslator.map_finish_reason(finish_reason)

    except (asyncio.CancelledError, GeneratorExit):
        # Client disconnect or task cancellation (worker shutdown).
        # GeneratorExit forbids further yields; for CancelledError we
        # attempt one terminal frame so a still-connected client (e.g.
        # during graceful worker recycle) doesn't see a bare EOF and
        # trigger full-request retries.
        raise
    except Exception as e:  # noqa: BLE001 — terminal-event guarantee
        # Includes HTTPException raised by the inner generator (e.g.
        # its internal quota check runs after headers are committed).
        # EOF without a terminal event makes Codex re-POST the entire
        # request up to 5 times — always emit response.failed instead.
        status_code = getattr(e, "status_code", None)
        detail = getattr(e, "detail", None)
        message = str(detail) if detail is not None else str(e) or "internal error"
        pending_error = {
            "code": _map_error(status_code, message),
            "message": message,
        }
        for f in terminal_frames():
            yield f
        return

    # Inner stream exhausted. If it died after streaming content
    # without ever sending finish_reason (backend crash mid-generation),
    # report failure rather than presenting a truncated answer as
    # completed — bounded client retries beat silent corruption.
    if (
        pending_status is None
        and pending_error is None
        and (st.kind is not None or st.completed_output)
        and not saw_error_frame
        and harvested_usage is None
    ):
        # Heuristic: a healthy inner stream always ends with a
        # finish_reason chunk (and usually usage). No finish_reason and
        # no usage after content started flowing means it died.
        pending_error = {
            "code": "server_error",
            "message": "upstream stream ended unexpectedly",
        }

    for f in terminal_frames():
        yield f
