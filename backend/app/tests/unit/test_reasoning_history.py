############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_reasoning_history.py: Unit tests for earlier-turn
# reasoning in multi-turn and tool-calling conversations.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Covers: every inbound dialect keeps an earlier assistant turn's reasoning
(OpenAI reasoning_content/reasoning, Anthropic thinking blocks, Responses
reasoning items, Ollama thinking); the outbound translators forward it on
assistant messages only; the gateway policy (fleet setting + per-model
override) keeps or drops it before the token count; context trimming counts
it and drops it before whole turns."""

import asyncio
from types import SimpleNamespace

import pytest

from backend.app.core.canonical_schemas import (
    CanonicalChatRequest,
    CanonicalFunctionCall,
    CanonicalMessage,
    CanonicalToolCall,
    MessageRole,
)
from backend.app.core.context_trim import trim_messages_to_fit
from backend.app.core.reasoning import forward_history_for, history_reasoning_text
from backend.app.core.translators import (
    AnthropicInTranslator,
    OllamaInTranslator,
    OllamaOutTranslator,
    OpenAIInTranslator,
    VLLMOutTranslator,
)
from backend.app.core.translators.responses_in import ResponsesInTranslator

R1 = "First compare the two prices, then answer."
R2 = "The tool said rain; mention the umbrella."


def _call(call_id="call_1", name="get_weather", args='{"city": "Moscow"}'):
    return CanonicalToolCall(id=call_id, type="function", function=CanonicalFunctionCall(name=name, arguments=args))


def _by_role(messages, role):
    return [m for m in messages if m.role == role]


# ---------------------------------------------------------------------------
# Helpers in core/reasoning.py
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_history_reasoning_text_keeps_only_non_blank_strings(self):
        assert history_reasoning_text(R1) == R1
        for value in (None, "", "   \n", 42, ["a"], {"text": "x"}):
            assert history_reasoning_text(value) is None

    @pytest.mark.parametrize("override,fleet,expected", [
        (None, False, False), (None, True, True),
        (True, False, True), (False, True, False),
    ])
    def test_override_wins_over_fleet_setting(self, override, fleet, expected):
        assert forward_history_for(override, fleet) is expected


# ---------------------------------------------------------------------------
# Inbound translators
# ---------------------------------------------------------------------------

class TestOpenAIIn:
    def _translate(self, assistant_extra):
        return OpenAIInTranslator.translate_chat_request({
            "model": "m",
            "messages": [
                {"role": "user", "content": "Which is cheaper?", "reasoning_content": "ignored on user"},
                {"role": "assistant", "content": "The first.", **assistant_extra},
                {"role": "user", "content": "By how much?"},
            ],
        })

    def test_reasoning_content_is_kept_on_the_assistant_message(self):
        req = self._translate({"reasoning_content": R1})
        assert _by_role(req.messages, MessageRole.ASSISTANT)[0].reasoning == R1
        assert all(m.reasoning is None for m in _by_role(req.messages, MessageRole.USER))

    def test_vllm_reasoning_field_is_accepted(self):
        req = self._translate({"reasoning": R1})
        assert _by_role(req.messages, MessageRole.ASSISTANT)[0].reasoning == R1

    def test_non_text_reasoning_is_ignored(self):
        req = self._translate({"reasoning": {"effort": "high"}, "reasoning_content": "  "})
        assert _by_role(req.messages, MessageRole.ASSISTANT)[0].reasoning is None

    def test_tool_call_turn_keeps_reasoning(self):
        req = OpenAIInTranslator.translate_chat_request({
            "model": "m",
            "messages": [
                {"role": "user", "content": "Weather?"},
                {"role": "assistant", "content": None, "reasoning_content": R2, "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "rain"},
            ],
        })
        assistant = _by_role(req.messages, MessageRole.ASSISTANT)[0]
        assert assistant.reasoning == R2 and assistant.tool_calls


class TestAnthropicIn:
    def _translate(self, assistant_content):
        return AnthropicInTranslator.translate_messages_request({
            "model": "m", "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "Weather?"},
                {"role": "assistant", "content": assistant_content},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "rain"}]},
            ],
        })

    def test_thinking_blocks_become_reasoning_on_a_tool_use_turn(self):
        req = self._translate([
            {"type": "thinking", "thinking": R1, "signature": "sig-1"},
            {"type": "redacted_thinking", "data": "opaque"},
            {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Moscow"}},
            {"type": "thinking", "thinking": R2, "signature": "sig-2"},
        ])
        assistant = _by_role(req.messages, MessageRole.ASSISTANT)[0]
        assert assistant.reasoning == f"{R1}\n\n{R2}"
        assert assistant.tool_calls[0].function.name == "get_weather"

    def test_thinking_on_a_text_turn(self):
        req = self._translate([
            {"type": "thinking", "thinking": R1, "signature": "s"},
            {"type": "text", "text": "Let me check."},
        ])
        assistant = _by_role(req.messages, MessageRole.ASSISTANT)[0]
        assert assistant.reasoning == R1
        assert assistant.content[0].text == "Let me check."

    def test_turn_without_thinking_has_none(self):
        req = self._translate([{"type": "text", "text": "Hi"}])
        assert _by_role(req.messages, MessageRole.ASSISTANT)[0].reasoning is None


class TestResponsesIn:
    def _reasoning_item(self, text, summary=None):
        return {"type": "reasoning", "id": "rs_1", "summary": summary or [],
                "content": [{"type": "reasoning_text", "text": text}]}

    def test_reasoning_attaches_to_the_following_message_and_its_function_call(self):
        req = ResponsesInTranslator.translate_responses_request({
            "model": "m",
            "input": [
                {"role": "user", "content": "Weather?"},
                self._reasoning_item(R1),
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Checking."}]},
                {"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "rain"},
                self._reasoning_item(R2),
                {"type": "function_call", "call_id": "call_2", "name": "get_weather", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "call_2", "output": "rain"},
            ],
        })
        assistants = _by_role(req.messages, MessageRole.ASSISTANT)
        assert len(assistants) == 2
        assert assistants[0].reasoning == R1 and assistants[0].content == "Checking."
        assert [tc.id for tc in assistants[0].tool_calls] == ["call_1"]
        assert assistants[1].reasoning == R2 and [tc.id for tc in assistants[1].tool_calls] == ["call_2"]

    def test_summary_and_encrypted_content_are_not_replayed(self):
        req = ResponsesInTranslator.translate_responses_request({
            "model": "m",
            "input": [
                {"role": "user", "content": "Hi"},
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "a summary"}],
                 "encrypted_content": "gAAAA..."},
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Hello"}]},
            ],
        })
        assert _by_role(req.messages, MessageRole.ASSISTANT)[0].reasoning is None

    def test_reasoning_not_followed_by_an_assistant_turn_is_dropped(self):
        req = ResponsesInTranslator.translate_responses_request({
            "model": "m",
            "input": [
                self._reasoning_item(R1),
                {"role": "user", "content": "Hi"},
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Hello"}]},
            ],
        })
        assert all(m.reasoning is None for m in req.messages)


class TestOllamaIn:
    def test_thinking_is_kept_on_assistant_messages_only(self):
        req = OllamaInTranslator.translate_chat_request({
            "model": "m",
            "messages": [
                {"role": "user", "content": "Hi", "thinking": "not mine"},
                {"role": "assistant", "content": "Hello", "thinking": R1},
                {"role": "user", "content": "Again"},
            ],
        })
        assert _by_role(req.messages, MessageRole.ASSISTANT)[0].reasoning == R1
        assert all(m.reasoning is None for m in _by_role(req.messages, MessageRole.USER))


# ---------------------------------------------------------------------------
# Outbound translators
# ---------------------------------------------------------------------------

def _request(**kw):
    messages = [
        CanonicalMessage(role=MessageRole.USER, content="Weather?"),
        CanonicalMessage(role=MessageRole.ASSISTANT, content=None, reasoning=R2, tool_calls=[_call()]),
        CanonicalMessage(role=MessageRole.TOOL, content="rain", tool_call_id="call_1"),
    ]
    return CanonicalChatRequest(model=kw.pop("model", "qwen/qwen3.6-35b"), messages=messages, **kw)


class TestOutbound:
    def test_vllm_gets_reasoning_on_the_assistant_message(self):
        payload = VLLMOutTranslator.translate_chat_request(_request())
        assistant = [m for m in payload["messages"] if m["role"] == "assistant"][0]
        assert assistant["reasoning"] == R2
        assert "reasoning_content" not in assistant
        assert all("reasoning" not in m for m in payload["messages"] if m["role"] != "assistant")

    def test_vllm_omits_the_key_when_there_is_none(self):
        req = _request()
        req.messages[1].reasoning = None
        payload = VLLMOutTranslator.translate_chat_request(req)
        assert all("reasoning" not in m for m in payload["messages"])

    def test_non_assistant_reasoning_never_goes_upstream(self):
        req = _request()
        req.messages[0].reasoning = "user-side text"
        payload = VLLMOutTranslator.translate_chat_request(req)
        assert "reasoning" not in payload["messages"][0]

    def test_ollama_gets_thinking(self):
        payload = OllamaOutTranslator.translate_chat_request(_request())
        assistant = [m for m in payload["messages"] if m["role"] == "assistant"][0]
        assert assistant["thinking"] == R2


# ---------------------------------------------------------------------------
# Gateway policy (services/inference.apply_reasoning_history_policy)
# ---------------------------------------------------------------------------

@pytest.fixture
def policy(monkeypatch):
    """apply_reasoning_history_policy with the fleet setting stubbed."""
    from backend.app.services import inference as inf

    state = {"fleet": False, "reads": 0}

    async def _fleet():
        state["reads"] += 1
        return state["fleet"]

    monkeypatch.setattr(inf, "_get_forward_history_default", _fleet)

    def run(request, override=None, fleet=False):
        state["fleet"] = fleet
        row = SimpleNamespace(reasoning_history_override=override)
        asyncio.run(inf.apply_reasoning_history_policy(request, request.model, row))
        return request

    run.state = state
    return run


class TestPolicy:
    def test_fleet_off_drops_it(self, policy):
        req = policy(_request(), fleet=False)
        assert all(m.reasoning is None for m in req.messages)

    def test_fleet_on_keeps_it(self, policy):
        req = policy(_request(), fleet=True)
        assert req.messages[1].reasoning == R2

    def test_override_on_beats_fleet_off_without_reading_the_setting(self, policy):
        req = policy(_request(), override=True, fleet=False)
        assert req.messages[1].reasoning == R2
        assert policy.state["reads"] == 0

    def test_override_off_beats_fleet_on(self, policy):
        req = policy(_request(), override=False, fleet=True)
        assert req.messages[1].reasoning is None

    def test_reasoning_on_other_roles_is_always_dropped(self, policy):
        req = _request()
        req.messages[0].reasoning = "user-side text"
        policy(req, fleet=True)
        assert req.messages[0].reasoning is None and req.messages[1].reasoning == R2

    def test_stripping_invalidates_memoized_token_counts(self, policy):
        req = _request()
        req._est_input_tokens = 123
        req._exact_input_tokens = 456
        policy(req, fleet=False)
        assert req._est_input_tokens is None and req._exact_input_tokens is None

    def test_no_reasoning_means_no_setting_read(self, policy):
        req = _request()
        req.messages[1].reasoning = None
        policy(req)
        assert policy.state["reads"] == 0

    def test_missing_model_row_uses_the_fleet_setting(self, policy):
        from backend.app.services import inference as inf
        req = _request()
        policy.state["fleet"] = True
        asyncio.run(inf.apply_reasoning_history_policy(req, req.model, None))
        assert req.messages[1].reasoning == R2


# ---------------------------------------------------------------------------
# Context trimming (truncation: "auto")
# ---------------------------------------------------------------------------

def _est(s: str) -> int:
    return len(s) // 4


class TestContextTrim:
    def _conversation(self):
        big = "x" * 400  # 100 tokens of earlier reasoning per turn
        return [
            CanonicalMessage(role=MessageRole.SYSTEM, content="sys"),
            CanonicalMessage(role=MessageRole.USER, content="q1"),
            CanonicalMessage(role=MessageRole.ASSISTANT, content="a1", reasoning=big),
            CanonicalMessage(role=MessageRole.USER, content="q2"),
            CanonicalMessage(role=MessageRole.ASSISTANT, content="a2", reasoning=big),
        ]

    def test_reasoning_is_counted_and_dropped_before_any_turn(self):
        messages = self._conversation()
        # Everything ~227 tokens; without the first turn's reasoning ~127.
        trimmed, dropped = trim_messages_to_fit(messages, 150, _est)
        assert dropped == 0
        assert trimmed is not messages and len(trimmed) == len(messages)
        assert trimmed[2].reasoning is None
        assert trimmed[4].reasoning == "x" * 400  # the final turn keeps its reasoning
        assert messages[2].reasoning == "x" * 400  # the caller's list is untouched

    def test_turns_are_dropped_when_dropping_reasoning_is_not_enough(self):
        # Groups (each user message and each assistant reply apart): sys 4 |
        # q1 4 | a1 4 once its reasoning is gone | q2 4 | a2 104 → 120 > 113;
        # dropping q1 and a1 brings it to 112.
        trimmed, dropped = trim_messages_to_fit(self._conversation(), 113, _est)
        assert dropped == 2
        assert [m.content for m in trimmed] == ["sys", "q2", "a2"]

    def test_nothing_changes_when_it_fits(self):
        messages = self._conversation()
        trimmed, dropped = trim_messages_to_fit(messages, 10_000, _est)
        assert trimmed is messages and dropped == 0


# ---------------------------------------------------------------------------
# Wiring in the inference service
# ---------------------------------------------------------------------------

class TestServiceWiring:
    @pytest.mark.parametrize("method", ["_proxy_with_retry", "_proxy_stream_with_retry"])
    def test_policy_runs_before_max_tokens_is_capped(self, method):
        """The outbound translators forward reasoning whenever a message carries
        it, so both attempt paths must apply the policy, and before the token
        count (cap_max_tokens) so the count matches what is sent."""
        import inspect
        from backend.app.services import inference as inf

        src = inspect.getsource(getattr(inf.InferenceService, method))
        assert "apply_reasoning_history_policy(" in src
        assert src.index("apply_reasoning_history_policy(") < src.index("cap_max_tokens(")

    def test_dlp_redaction_reaches_earlier_reasoning(self):
        from backend.app.services import inference as inf

        req = _request()
        req.messages[1].reasoning = "user SSN is 123-45-6789"
        inf.InferenceService._apply_prompt_redactions(req, [("123-45-6789", "ssn")])
        assert req.messages[1].reasoning == "user SSN is [REDACTED: ssn]"
        payload = VLLMOutTranslator.translate_chat_request(req)
        assert "123-45-6789" not in str(payload)
