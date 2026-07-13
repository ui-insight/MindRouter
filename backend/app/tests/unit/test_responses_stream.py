############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_responses_stream.py: Unit tests for the chat-SSE ->
# Responses-API SSE stream adapter.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Covers:
- Canonical event sequences: text, reasoning+text, function calls
- Wire invariants: created/in_progress prologue, content_part.added
  before deltas, monotonic sequence_number, stable item ids, event
  line == payload type, no [DONE] sentinel
- Deferred terminal emission: usage chunk arriving AFTER finish_reason
  (vLLM include_usage) still lands in response.completed
- finish_reason length -> response.incomplete
- Inner error frames and raised exceptions -> terminal response.failed
- Abnormal EOF (content, no finish_reason) -> response.failed
- Drain contract: the inner generator is always fully consumed
"""

import json

import pytest

from backend.app.core.translators.responses_in import ResponsesRequestContext
from backend.app.core.translators.responses_stream import stream_responses_events


async def _async_iter(chunks):
    """Convert a list of bytes into an async iterator."""
    for chunk in chunks:
        yield chunk


async def _collect_stream(async_gen):
    items = []
    async for item in async_gen:
        items.append(item)
    return items


def _sse(obj) -> bytes:
    return f"data: {json.dumps(obj)}\n\n".encode()


def _chunk(delta=None, finish_reason=None, usage=None, choices=True) -> bytes:
    data = {
        "id": "req-uuid",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "m",
        "choices": [],
    }
    if choices:
        choice = {"index": 0, "delta": delta or {}}
        if finish_reason:
            choice["finish_reason"] = finish_reason
        data["choices"] = [choice]
    if usage is not None:
        data["usage"] = usage
    return _sse(data)


_DONE = b"data: [DONE]\n\n"
_USAGE = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}


def _parse(frames):
    """Parse SSE frames into (event_type, payload) pairs, asserting framing."""
    parsed = []
    for frame in frames:
        assert frame.endswith("\n\n")
        lines = frame.strip().split("\n")
        assert lines[0].startswith("event: ")
        assert lines[1].startswith("data: ")
        event_type = lines[0][7:]
        payload = json.loads(lines[1][6:])
        assert payload["type"] == event_type  # event line mirrors json type
        parsed.append((event_type, payload))
    return parsed


def _ctx(**overrides):
    body = {"model": "test-model", "input": "hi", "stream": True}
    body.update(overrides)
    return ResponsesRequestContext.from_body(body)


class TestTextStream:
    async def test_canonical_text_sequence(self):
        chunks = [
            _chunk(delta={"role": "assistant", "content": "Hel"}),
            _chunk(delta={"content": "lo"}),
            _chunk(finish_reason="stop", usage=_USAGE),
            _DONE,
        ]
        events = _parse(
            await _collect_stream(
                stream_responses_events(_async_iter(chunks), _ctx())
            )
        )
        types = [t for t, _ in events]
        assert types == [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ]
        # Monotonic sequence numbers from 0
        seqs = [p["sequence_number"] for _, p in events]
        assert seqs == list(range(len(events)))
        # Stable item id across the item's events
        item_id = events[2][1]["item"]["id"]
        assert item_id.startswith("msg_")
        for _, p in events[3:9]:
            assert p.get("item_id", p.get("item", {}).get("id")) == item_id
        # Terminal snapshot
        completed = events[-1][1]["response"]
        assert completed["status"] == "completed"
        assert completed["output"][0]["content"][0]["text"] == "Hello"
        assert completed["usage"]["input_tokens"] == 11
        assert completed["usage"]["output_tokens"] == 7
        # Prologue snapshots carry no usage
        assert events[0][1]["response"]["usage"] is None
        assert events[0][1]["response"]["status"] == "in_progress"
        # No [DONE] forwarded
        assert not any("[DONE]" in f for t, f in [(t, json.dumps(p)) for t, p in events])

    async def test_usage_chunk_after_finish_reason(self):
        # vLLM include_usage: usage rides an empty-choices chunk AFTER
        # the finish_reason chunk; terminal emission must wait for it.
        chunks = [
            _chunk(delta={"content": "Hi"}),
            _chunk(finish_reason="stop"),
            _chunk(choices=False, usage=_USAGE),
            _DONE,
        ]
        events = _parse(
            await _collect_stream(
                stream_responses_events(_async_iter(chunks), _ctx())
            )
        )
        assert events[-1][0] == "response.completed"
        assert events[-1][1]["response"]["usage"]["input_tokens"] == 11

    async def test_usage_estimated_when_absent(self):
        chunks = [
            _chunk(delta={"content": "Hello"}),
            _chunk(finish_reason="stop"),
            _DONE,
        ]
        events = _parse(
            await _collect_stream(
                stream_responses_events(_async_iter(chunks), _ctx())
            )
        )
        usage = events[-1][1]["response"]["usage"]
        assert usage["output_tokens"] == 1  # ceil-ish 5 chars // 4
        assert usage["input_tokens"] == 0

    async def test_length_finish_becomes_incomplete(self):
        chunks = [
            _chunk(delta={"content": "partial"}),
            _chunk(finish_reason="length"),
            _DONE,
        ]
        events = _parse(
            await _collect_stream(
                stream_responses_events(_async_iter(chunks), _ctx())
            )
        )
        assert events[-1][0] == "response.incomplete"
        resp = events[-1][1]["response"]
        assert resp["status"] == "incomplete"
        assert resp["incomplete_details"] == {"reason": "max_output_tokens"}

    async def test_empty_stream_completes(self):
        events = _parse(
            await _collect_stream(
                stream_responses_events(_async_iter([_DONE]), _ctx())
            )
        )
        assert [t for t, _ in events] == [
            "response.created",
            "response.in_progress",
            "response.completed",
        ]
        assert events[-1][1]["response"]["output"] == []


class TestReasoningStream:
    async def test_reasoning_then_text(self):
        chunks = [
            _chunk(delta={"reasoning_content": "think"}),
            _chunk(delta={"reasoning_content": "ing"}),
            _chunk(delta={"content": "answer"}),
            _chunk(finish_reason="stop", usage=_USAGE),
            _DONE,
        ]
        events = _parse(
            await _collect_stream(
                stream_responses_events(_async_iter(chunks), _ctx())
            )
        )
        types = [t for t, _ in events]
        assert types == [
            "response.created",
            "response.in_progress",
            "response.output_item.added",       # reasoning, output_index 0
            "response.reasoning_text.delta",
            "response.reasoning_text.delta",
            "response.reasoning_text.done",      # closed when text starts
            "response.output_item.done",
            "response.output_item.added",       # message, output_index 1
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ]
        rs_added = events[2][1]
        assert rs_added["item"]["id"].startswith("rs_")
        assert rs_added["output_index"] == 0
        msg_added = events[7][1]
        assert msg_added["output_index"] == 1
        output = events[-1][1]["response"]["output"]
        assert output[0]["type"] == "reasoning"
        assert output[0]["content"][0]["text"] == "thinking"
        assert output[1]["type"] == "message"


class TestToolCallStream:
    async def test_single_function_call(self):
        chunks = [
            _chunk(
                delta={
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_abc",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": ""},
                        }
                    ]
                }
            ),
            _chunk(
                delta={
                    "tool_calls": [
                        {"index": 0, "function": {"arguments": '{"location":'}}
                    ]
                }
            ),
            _chunk(
                delta={
                    "tool_calls": [
                        {"index": 0, "function": {"arguments": '"x"}'}}
                    ]
                }
            ),
            _chunk(finish_reason="tool_calls", usage=_USAGE),
            _DONE,
        ]
        events = _parse(
            await _collect_stream(
                stream_responses_events(_async_iter(chunks), _ctx())
            )
        )
        types = [t for t, _ in events]
        assert types == [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.output_item.done",
            "response.completed",
        ]
        added_item = events[2][1]["item"]
        assert added_item["type"] == "function_call"
        assert added_item["call_id"] == "call_abc"
        assert added_item["name"] == "get_weather"
        args_done = events[5][1]
        assert args_done["arguments"] == '{"location":"x"}'
        assert args_done["name"] == "get_weather"
        item_done = events[6][1]["item"]
        assert item_done["status"] == "completed"
        assert item_done["call_id"] == "call_abc"
        # No content_part events for function_call items
        assert "response.content_part.added" not in types

    async def test_text_then_two_function_calls(self):
        chunks = [
            _chunk(delta={"content": "checking"}),
            _chunk(
                delta={
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "function": {"name": "a", "arguments": "{}"},
                        }
                    ]
                }
            ),
            _chunk(
                delta={
                    "tool_calls": [
                        {
                            "index": 1,
                            "id": "call_2",
                            "function": {"name": "b", "arguments": "{}"},
                        }
                    ]
                }
            ),
            _chunk(finish_reason="tool_calls"),
            _DONE,
        ]
        events = _parse(
            await _collect_stream(
                stream_responses_events(_async_iter(chunks), _ctx())
            )
        )
        output = events[-1][1]["response"]["output"]
        assert [item["type"] for item in output] == [
            "message",
            "function_call",
            "function_call",
        ]
        assert output[1]["call_id"] == "call_1"
        assert output[2]["call_id"] == "call_2"
        # Items never interleave: message closes before fc 1 opens, etc.
        added_indexes = [
            p["output_index"]
            for t, p in events
            if t == "response.output_item.added"
        ]
        assert added_indexes == [0, 1, 2]


class TestErrorPaths:
    async def test_inner_error_frame_becomes_failed(self):
        error_frame = (
            b'data: {"error": {"message": "Token quota exceeded", '
            b'"type": "backend_error", "code": 429}}\n\n' + _DONE
        )
        events = _parse(
            await _collect_stream(
                stream_responses_events(_async_iter([error_frame]), _ctx())
            )
        )
        assert events[-1][0] == "response.failed"
        resp = events[-1][1]["response"]
        assert resp["status"] == "failed"
        assert resp["error"]["code"] == "insufficient_quota"
        assert "quota" in resp["error"]["message"].lower()
        assert resp["usage"] is None

    async def test_raised_exception_becomes_failed(self):
        class FakeHTTPException(Exception):
            status_code = 429
            detail = "Rate limit exceeded: 60 requests per minute"

        async def raising_inner():
            yield _chunk(delta={"content": "par"})
            raise FakeHTTPException()

        events = _parse(
            await _collect_stream(
                stream_responses_events(raising_inner(), _ctx())
            )
        )
        assert events[-1][0] == "response.failed"
        assert events[-1][1]["response"]["error"]["code"] == "rate_limit_exceeded"
        # The open message item was closed before the terminal event
        types = [t for t, _ in events]
        assert "response.output_item.done" in types

    async def test_abnormal_eof_after_content_becomes_failed(self):
        chunks = [_chunk(delta={"content": "trunca"})]  # no finish_reason
        events = _parse(
            await _collect_stream(
                stream_responses_events(_async_iter(chunks), _ctx())
            )
        )
        assert events[-1][0] == "response.failed"
        assert events[-1][1]["response"]["error"]["code"] == "server_error"


class TestDrainContract:
    async def test_inner_generator_fully_consumed(self):
        consumed = {"done": False}

        async def inner():
            yield _chunk(delta={"content": "hi"})
            yield _chunk(finish_reason="stop", usage=_USAGE)
            yield _DONE
            # Post-[DONE] work models stream_chat_completion's cleanup
            # (accounting, capacity release) which only runs if the
            # wrapper drives the generator to exhaustion.
            consumed["done"] = True

        events = _parse(
            await _collect_stream(stream_responses_events(inner(), _ctx()))
        )
        assert consumed["done"] is True
        assert events[-1][0] == "response.completed"
