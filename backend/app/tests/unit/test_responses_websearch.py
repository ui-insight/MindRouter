############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_responses_websearch.py: Unit tests for hosted web_search
# tool execution in the Responses API.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Covers:
- Tool detection helpers and the synthetic function tool
- Non-streaming loop: search round -> answer round, message threading,
  usage accumulation, web_search_call items, call-budget exhaustion,
  client-function passthrough
- Streaming loop: suppression of synthetic calls (no function_call
  events), web_search_call event lifecycle, terminal correctness
- stream_round suppression primitive
"""

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.app.core.canonical_schemas import (
    CanonicalChatRequest,
    CanonicalMessage,
    MessageRole,
)
from backend.app.core.translators.responses_in import ResponsesRequestContext
from backend.app.core.translators import responses_stream as rs

# ----------------------------------------------------------------
# Direct-load responses_websearch with heavy leaves stubbed (it
# imports logging_config and settings; search internals are imported
# lazily inside the executor, which these tests replace anyway).
# ----------------------------------------------------------------

_STUB_NAMES = ["backend.app.logging_config", "backend.app.settings"]
_added = []
for _name in _STUB_NAMES:
    if _name not in sys.modules:
        stub = MagicMock()
        if _name == "backend.app.logging_config":
            stub.get_logger = MagicMock(return_value=MagicMock())
        sys.modules[_name] = stub
        _added.append(_name)

_svc_dir = Path(__file__).resolve().parents[2] / "services"
_spec = importlib.util.spec_from_file_location(
    "responses_websearch", _svc_dir / "responses_websearch.py",
    submodule_search_locations=[],
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

for _name in _added:
    sys.modules.pop(_name, None)


def _ctx(**overrides):
    body = {"model": "m", "input": "hi", "stream": True}
    body.update(overrides)
    return ResponsesRequestContext.from_body(body)


def _canonical(tools=True):
    req = CanonicalChatRequest(
        model="m",
        messages=[CanonicalMessage(role=MessageRole.USER, content="hi")],
    )
    if tools:
        req.tools = [_mod.synthetic_search_tool()]
    return req


def _chat_response(content=None, tool_calls=None, finish="stop", usage=None):
    return {
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content,
                        "tool_calls": tool_calls},
            "finish_reason": finish,
        }],
        "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5,
                           "total_tokens": 15},
    }


def _search_call(call_id="call_s1", query="idaho news"):
    return {"id": call_id, "type": "function",
            "function": {"name": "web_search",
                         "arguments": json.dumps({"query": query})}}


class TestHelpers:
    def test_wants_web_search(self):
        assert _mod.wants_web_search([{"type": "web_search"}]) is True
        assert _mod.wants_web_search([{"type": "web_search_preview"}]) is True
        assert _mod.wants_web_search([{"type": "function", "name": "f"}]) is False
        assert _mod.wants_web_search([]) is False

    def test_client_function_collision(self):
        assert _mod.has_client_web_search_function(
            [{"type": "function", "name": "web_search"}]
        ) is True
        assert _mod.has_client_web_search_function(
            [{"type": "web_search"}]
        ) is False

    def test_synthetic_tool_shape(self):
        tool = _mod.synthetic_search_tool()
        assert tool.type == "function"
        assert tool.function["name"] == "web_search"
        assert tool.function["parameters"]["required"] == ["query"]
        assert isinstance(tool.function["description"], str)

    def test_arm_web_search_adds_tool_and_nudge(self):
        canonical = _canonical(tools=False)
        _mod.arm_web_search(canonical)
        assert canonical.tools[-1].function["name"] == "web_search"
        nudge = canonical.messages[-1]
        assert nudge.role == MessageRole.SYSTEM
        assert "web_search" in nudge.content
        assert "real-time" in nudge.content

    def test_parse_query(self):
        assert _mod._parse_query('{"query": "cats"}') == "cats"
        assert _mod._parse_query("not json") == "not json"
        assert _mod._parse_query("") == "(empty query)"

    def test_ws_item_shape(self):
        item = _mod.build_web_search_call_item("cats")
        assert item["id"].startswith("ws_")
        assert item["type"] == "web_search_call"
        assert item["status"] == "completed"
        assert item["action"] == {"type": "search", "query": "cats"}


class TestNonStreamingLoop:
    async def test_search_then_answer(self):
        responses = [
            _chat_response(tool_calls=[_search_call()], finish="tool_calls"),
            _chat_response(content="Sunny in Moscow.", finish="stop"),
        ]
        backend = AsyncMock(side_effect=responses)
        queries = []

        async def executor(q):
            queries.append(q)
            return "1. Weather site\n   URL: https://x\n   sunny"

        canonical = _canonical()
        result = await _mod.run_web_search_loop(
            backend, canonical, _ctx(stream=False), executor, max_calls=4
        )

        assert queries == ["idaho news"]
        types = [i["type"] for i in result["output"]]
        assert types == ["web_search_call", "message"]
        assert result["output"][0]["action"]["query"] == "idaho news"
        assert result["status"] == "completed"
        # Usage summed across both rounds
        assert result["usage"]["total_tokens"] == 30
        # Conversation was threaded: assistant tool_calls + tool result
        roles = [m.role for m in canonical.messages]
        assert roles == [MessageRole.USER, MessageRole.ASSISTANT, MessageRole.TOOL]
        assert canonical.messages[2].tool_call_id == "call_s1"
        assert "sunny" in canonical.messages[2].content

    async def test_call_budget_removes_tool(self):
        # Model searches every round; after max_calls the synthetic tool
        # is removed and the model must answer.
        n = {"count": 0}

        async def backend(canonical):
            if canonical.tools:
                n["count"] += 1
                return _chat_response(
                    tool_calls=[_search_call(f"call_{n['count']}")],
                    finish="tool_calls",
                )
            return _chat_response(content="best effort answer", finish="stop")

        canonical = _canonical()
        result = await _mod.run_web_search_loop(
            backend, canonical, _ctx(stream=False),
            AsyncMock(return_value="results"), max_calls=2,
        )
        assert n["count"] == 2  # capped
        assert canonical.tools is None
        types = [i["type"] for i in result["output"]]
        assert types.count("web_search_call") == 2
        assert types[-1] == "message"

    async def test_client_function_passes_through(self):
        # A client-owned function call ends the loop and is surfaced.
        client_call = {"id": "call_c1", "type": "function",
                       "function": {"name": "get_weather",
                                    "arguments": "{}"}}
        backend = AsyncMock(return_value=_chat_response(
            tool_calls=[client_call], finish="tool_calls"
        ))
        result = await _mod.run_web_search_loop(
            backend, _canonical(), _ctx(stream=False),
            AsyncMock(), max_calls=4,
        )
        assert backend.await_count == 1
        fc = [i for i in result["output"] if i["type"] == "function_call"]
        assert fc and fc[0]["name"] == "get_weather"
        assert fc[0]["call_id"] == "call_c1"


def _sse(obj) -> bytes:
    return f"data: {json.dumps(obj)}\n\n".encode()


def _chunk(delta=None, finish_reason=None, usage=None, choices=True) -> bytes:
    data = {"id": "u", "object": "chat.completion.chunk", "created": 1,
            "model": "m", "choices": []}
    if choices:
        choice = {"index": 0, "delta": delta or {}}
        if finish_reason:
            choice["finish_reason"] = finish_reason
        data["choices"] = [choice]
    if usage is not None:
        data["usage"] = usage
    return _sse(data)


_DONE = b"data: [DONE]\n\n"


async def _aiter(chunks):
    for c in chunks:
        yield c


def _parse(frames):
    out = []
    for frame in frames:
        lines = frame.strip().split("\n")
        out.append((lines[0][7:], json.loads(lines[1][6:])))
    return out


class TestStreamRoundSuppression:
    async def test_suppressed_call_emits_no_events(self):
        st = rs._StreamState()
        chunks = [
            _chunk(delta={"tool_calls": [{"index": 0, "id": "call_1",
                    "function": {"name": "web_search",
                                 "arguments": '{"query": "cats"}'}}]}),
            _chunk(finish_reason="tool_calls"),
            _DONE,
        ]
        frames = [f async for f in rs.stream_round(_aiter(chunks), st,
                                                   suppress_tool="web_search")]
        assert frames == []  # nothing surfaced
        assert st.suppressed_calls == [{
            "call_id": "call_1", "name": "web_search",
            "arguments": '{"query": "cats"}',
        }]


class TestStreamingLoop:
    async def test_two_round_stream(self):
        round1 = [
            _chunk(delta={"tool_calls": [{"index": 0, "id": "call_1",
                    "function": {"name": "web_search",
                                 "arguments": '{"query": "cats"}'}}]}),
            _chunk(finish_reason="tool_calls"),
            _DONE,
        ]
        round2 = [
            _chunk(delta={"content": "Cats are great."}),
            _chunk(finish_reason="stop",
                   usage={"prompt_tokens": 9, "completion_tokens": 3,
                          "total_tokens": 12}),
            _DONE,
        ]
        rounds = [round1, round2]

        def make_inner():
            return _aiter(rounds.pop(0))

        async def executor(q):
            return f"results for {q}"

        canonical = _canonical()
        capture = {}
        frames = [
            f async for f in _mod.stream_with_web_search(
                make_inner, canonical, _ctx(), executor, 4, capture=capture
            )
        ]
        events = _parse(frames)
        types = [t for t, _ in events]
        assert types == [
            "response.created",
            "response.in_progress",
            "response.output_item.added",        # web_search_call
            "response.web_search_call.in_progress",
            "response.web_search_call.searching",
            "response.web_search_call.completed",
            "response.output_item.done",
            "response.output_item.added",        # message (round 2)
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ]
        # No function_call events anywhere (call was suppressed)
        assert not any("function_call" in t for t in types)
        # Sequence numbers monotonic across rounds
        seqs = [p["sequence_number"] for _, p in events]
        assert seqs == list(range(len(events)))
        # Terminal output: ws item then message; real usage from round 2
        final = events[-1][1]["response"]
        assert [i["type"] for i in final["output"]] == [
            "web_search_call", "message"
        ]
        assert final["usage"]["input_tokens"] == 9
        assert capture["status"] == "completed"
        # Conversation threaded for round 2
        assert canonical.messages[-1].role == MessageRole.TOOL
        assert canonical.messages[-1].content == "results for cats"

    async def test_stream_error_round_still_terminates(self):
        round1 = [
            b'data: {"error": {"message": "boom", "code": 500}}\n\n',
            _DONE,
        ]

        def make_inner():
            return _aiter(round1)

        frames = [
            f async for f in _mod.stream_with_web_search(
                make_inner, _canonical(), _ctx(), AsyncMock(), 4
            )
        ]
        events = _parse(frames)
        assert events[-1][0] == "response.failed"
