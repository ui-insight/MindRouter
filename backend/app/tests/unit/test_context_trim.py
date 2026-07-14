############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_context_trim.py: Unit tests for truncation:"auto"
# context trimming.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Covers: turn grouping, oldest-first dropping, tool-call pairing
preservation, system-message and final-turn protection, no-op when
the input already fits."""

from backend.app.core.canonical_schemas import (
    CanonicalFunctionCall,
    CanonicalMessage,
    CanonicalToolCall,
    MessageRole,
)
from backend.app.core.context_trim import _group_messages, trim_messages_to_fit


def _est(s: str) -> int:
    return len(s) // 4


def _msg(role, content=None, tool_calls=None, tool_call_id=None):
    return CanonicalMessage(
        role=role, content=content, tool_calls=tool_calls,
        tool_call_id=tool_call_id,
    )


def _tool_call(call_id="call_1"):
    return CanonicalToolCall(
        id=call_id, type="function",
        function=CanonicalFunctionCall(name="f", arguments="{}"),
    )


class TestGrouping:
    def test_system_head_and_turn_groups(self):
        messages = [
            _msg(MessageRole.SYSTEM, "sys"),
            _msg(MessageRole.USER, "q1"),
            _msg(MessageRole.ASSISTANT, "a1"),
            _msg(MessageRole.USER, "q2"),
        ]
        head, groups = _group_messages(messages)
        assert len(head) == 1
        assert [len(g) for g in groups] == [1, 1, 1]

    def test_tool_messages_stay_with_their_assistant(self):
        messages = [
            _msg(MessageRole.USER, "q"),
            _msg(MessageRole.ASSISTANT, None, tool_calls=[_tool_call()]),
            _msg(MessageRole.TOOL, "result", tool_call_id="call_1"),
            _msg(MessageRole.ASSISTANT, "answer"),
        ]
        _, groups = _group_messages(messages)
        # assistant + its tool results form one atomic group; the
        # follow-up assistant answer is its own droppable group
        assert [len(g) for g in groups] == [1, 2, 1]
        assert groups[1][0].role == MessageRole.ASSISTANT
        assert groups[1][1].role == MessageRole.TOOL


class TestTrimming:
    def test_no_op_when_fits(self):
        messages = [_msg(MessageRole.USER, "short")]
        trimmed, dropped = trim_messages_to_fit(messages, 1000, _est)
        assert dropped == 0
        assert trimmed is messages

    def test_drops_oldest_first(self):
        messages = [
            _msg(MessageRole.SYSTEM, "sys"),
            _msg(MessageRole.USER, "x" * 400),      # ~100 tokens
            _msg(MessageRole.ASSISTANT, "y" * 400),
            _msg(MessageRole.USER, "z" * 400),
        ]
        trimmed, dropped = trim_messages_to_fit(messages, 130, _est)
        assert dropped == 2
        assert trimmed[0].role == MessageRole.SYSTEM
        assert trimmed[1].content == "z" * 400  # newest turn kept

    def test_never_drops_final_group_or_system(self):
        messages = [
            _msg(MessageRole.SYSTEM, "s" * 4000),
            _msg(MessageRole.USER, "u" * 4000),
        ]
        trimmed, dropped = trim_messages_to_fit(messages, 10, _est)
        assert dropped == 0  # nothing droppable
        assert len(trimmed) == 2

    def test_tool_pairs_dropped_atomically(self):
        messages = [
            _msg(MessageRole.USER, "q1" * 200),
            _msg(MessageRole.ASSISTANT, None, tool_calls=[_tool_call()]),
            _msg(MessageRole.TOOL, "r" * 400, tool_call_id="call_1"),
            _msg(MessageRole.USER, "q2"),
        ]
        trimmed, dropped = trim_messages_to_fit(messages, 20, _est)
        roles = [m.role for m in trimmed]
        # No orphaned tool message may survive
        assert MessageRole.TOOL not in roles or (
            MessageRole.ASSISTANT in roles[: roles.index(MessageRole.TOOL)]
        )
        assert trimmed[-1].content == "q2"
