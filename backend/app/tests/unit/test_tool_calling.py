############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# test_tool_calling.py: Tool calling support tests
#
############################################################

"""Tests for tool calling across all translators and schemas."""

import json
import importlib.util
import os
import pytest

# Load translator modules directly to avoid DB import chain
_BASE = os.path.join(os.path.dirname(__file__), "..", "..", "core")

def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

schemas = _load_module(
    "canonical_schemas",
    os.path.join(_BASE, "canonical_schemas.py"),
)
openai_in_mod = _load_module(
    "openai_in",
    os.path.join(_BASE, "translators", "openai_in.py"),
)
ollama_in_mod = _load_module(
    "ollama_in",
    os.path.join(_BASE, "translators", "ollama_in.py"),
)
anthropic_in_mod = _load_module(
    "anthropic_in",
    os.path.join(_BASE, "translators", "anthropic_in.py"),
)
vllm_out_mod = _load_module(
    "vllm_out",
    os.path.join(_BASE, "translators", "vllm_out.py"),
)
ollama_out_mod = _load_module(
    "ollama_out",
    os.path.join(_BASE, "translators", "ollama_out.py"),
)

CanonicalFunctionCall = schemas.CanonicalFunctionCall
CanonicalToolCall = schemas.CanonicalToolCall
CanonicalToolDefinition = schemas.CanonicalToolDefinition
CanonicalStreamToolCallDelta = schemas.CanonicalStreamToolCallDelta
CanonicalMessage = schemas.CanonicalMessage
CanonicalChatRequest = schemas.CanonicalChatRequest
CanonicalStreamDelta = schemas.CanonicalStreamDelta
MessageRole = schemas.MessageRole

OpenAIInTranslator = openai_in_mod.OpenAIInTranslator
OllamaInTranslator = ollama_in_mod.OllamaInTranslator
AnthropicInTranslator = anthropic_in_mod.AnthropicInTranslator
VLLMOutTranslator = vllm_out_mod.VLLMOutTranslator
OllamaOutTranslator = ollama_out_mod.OllamaOutTranslator


# ── Fixtures ──────────────────────────────────────────────

SAMPLE_TOOLS_OPENAI = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

SAMPLE_TOOLS_ANTHROPIC = [
    {
        "name": "get_weather",
        "description": "Get current weather",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
]


# ── Phase 1: Schema Tests ────────────────────────────────

class TestCanonicalSchemas:
    """Tests for new canonical schema models."""

    def test_function_call_serialization(self):
        fc = CanonicalFunctionCall(name="get_weather", arguments='{"city":"NYC"}')
        assert fc.name == "get_weather"
        assert json.loads(fc.arguments) == {"city": "NYC"}

    def test_tool_call_serialization(self):
        tc = CanonicalToolCall(
            id="call_123",
            function=CanonicalFunctionCall(name="test", arguments="{}"),
        )
        assert tc.type == "function"
        d = tc.model_dump()
        assert d["id"] == "call_123"
        assert d["function"]["name"] == "test"

    def test_tool_definition(self):
        td = CanonicalToolDefinition(
            function={"name": "test", "description": "A test", "parameters": {}},
        )
        assert td.type == "function"
        assert td.function["name"] == "test"

    def test_stream_tool_call_delta(self):
        delta = CanonicalStreamToolCallDelta(
            index=0, id="call_1", type="function",
            function={"name": "test", "arguments": ""},
        )
        assert delta.index == 0
        assert delta.id == "call_1"

    def test_message_with_tool_calls(self):
        msg = CanonicalMessage(
            role=MessageRole.ASSISTANT,
            content=None,
            tool_calls=[
                CanonicalToolCall(
                    id="call_1",
                    function=CanonicalFunctionCall(name="f", arguments="{}"),
                )
            ],
        )
        assert msg.content is None
        assert len(msg.tool_calls) == 1
        assert msg.get_text_content() == ""

    def test_message_content_none(self):
        msg = CanonicalMessage(role=MessageRole.ASSISTANT, content=None)
        assert msg.get_text_content() == ""
        assert msg.has_images() is False

    def test_message_with_tool_call_id(self):
        msg = CanonicalMessage(
            role=MessageRole.TOOL,
            content="result here",
            tool_call_id="call_1",
        )
        assert msg.tool_call_id == "call_1"

    def test_chat_request_with_tools(self):
        req = CanonicalChatRequest(
            model="test",
            messages=[CanonicalMessage(role=MessageRole.USER, content="hi")],
            tools=[
                CanonicalToolDefinition(
                    function={"name": "test", "parameters": {}},
                )
            ],
            tool_choice="auto",
        )
        assert len(req.tools) == 1
        assert req.tool_choice == "auto"

    def test_stream_delta_with_tool_calls(self):
        delta = CanonicalStreamDelta(
            tool_calls=[
                CanonicalStreamToolCallDelta(
                    index=0, id="call_1", function={"name": "f", "arguments": ""},
                )
            ],
        )
        assert len(delta.tool_calls) == 1


# ── Phase 2a: OpenAI In ──────────────────────────────────

class TestOpenAIInToolCalling:
    """Tests for OpenAI inbound tool calling translation."""

    def test_tools_extraction(self):
        data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": SAMPLE_TOOLS_OPENAI,
            "tool_choice": "auto",
        }
        result = OpenAIInTranslator.translate_chat_request(data)
        assert len(result.tools) == 1
        assert result.tools[0].function["name"] == "get_weather"
        assert result.tool_choice == "auto"

    def test_assistant_tool_calls(self):
        data = {
            "model": "gpt-4",
            "messages": [{
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_abc",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city":"Seattle"}',
                    },
                }],
            }],
        }
        result = OpenAIInTranslator.translate_chat_request(data)
        msg = result.messages[0]
        assert msg.content is None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].id == "call_abc"
        assert msg.tool_calls[0].function.name == "get_weather"
        assert json.loads(msg.tool_calls[0].function.arguments) == {"city": "Seattle"}

    def test_tool_role_message(self):
        data = {
            "model": "gpt-4",
            "messages": [{
                "role": "tool",
                "content": "72F and sunny",
                "tool_call_id": "call_abc",
            }],
        }
        result = OpenAIInTranslator.translate_chat_request(data)
        msg = result.messages[0]
        assert msg.role == MessageRole.TOOL
        assert msg.tool_call_id == "call_abc"
        assert msg.get_text_content() == "72F and sunny"

    def test_no_tools_returns_none(self):
        data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hi"}],
        }
        result = OpenAIInTranslator.translate_chat_request(data)
        assert result.tools is None
        assert result.tool_choice is None


# ── Phase 2b: Ollama In ──────────────────────────────────

class TestOllamaInToolCalling:
    """Tests for Ollama inbound tool calling translation."""

    def test_tools_extraction(self):
        data = {
            "model": "llama3",
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": SAMPLE_TOOLS_OPENAI,
        }
        result = OllamaInTranslator.translate_chat_request(data)
        assert len(result.tools) == 1
        assert result.tools[0].function["name"] == "get_weather"

    def test_tool_calls_dict_to_string(self):
        """Ollama sends arguments as dict, canonical uses JSON string."""
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "function": {
                    "name": "get_weather",
                    "arguments": {"city": "Seattle"},
                },
            }],
        }
        result = OllamaInTranslator._translate_message(msg)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.arguments == '{"city": "Seattle"}'
        assert result.tool_calls[0].id == "call_0"  # synthetic ID

    def test_synthetic_ids(self):
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "f1", "arguments": {}}},
                {"function": {"name": "f2", "arguments": {}}},
            ],
        }
        result = OllamaInTranslator._translate_message(msg)
        assert result.tool_calls[0].id == "call_0"
        assert result.tool_calls[1].id == "call_1"


# ── Phase 2c: Anthropic In ───────────────────────────────

class TestAnthropicInToolCalling:
    """Tests for Anthropic inbound tool calling translation."""

    def test_tools_translation(self):
        """Anthropic input_schema → canonical parameters."""
        data = {
            "model": "claude-3",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": SAMPLE_TOOLS_ANTHROPIC,
        }
        result = AnthropicInTranslator.translate_messages_request(data)
        assert len(result.tools) == 1
        assert result.tools[0].function["name"] == "get_weather"
        assert "properties" in result.tools[0].function["parameters"]

    def test_tool_choice_auto(self):
        data = {
            "model": "claude-3",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": {"type": "auto"},
        }
        result = AnthropicInTranslator.translate_messages_request(data)
        assert result.tool_choice == "auto"

    def test_tool_choice_any_to_required(self):
        data = {
            "model": "claude-3",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": {"type": "any"},
        }
        result = AnthropicInTranslator.translate_messages_request(data)
        assert result.tool_choice == "required"

    def test_tool_choice_specific_tool(self):
        data = {
            "model": "claude-3",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": {"type": "tool", "name": "get_weather"},
        }
        result = AnthropicInTranslator.translate_messages_request(data)
        assert result.tool_choice == {
            "type": "function",
            "function": {"name": "get_weather"},
        }

    def test_tool_use_to_tool_calls(self):
        """Assistant message with tool_use → CanonicalMessage.tool_calls."""
        data = {
            "model": "claude-3",
            "max_tokens": 1024,
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me check."},
                    {
                        "type": "tool_use",
                        "id": "toolu_123",
                        "name": "get_weather",
                        "input": {"city": "Seattle"},
                    },
                ],
            }],
        }
        result = AnthropicInTranslator.translate_messages_request(data)
        msg = result.messages[0]
        assert msg.role == MessageRole.ASSISTANT
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].id == "toolu_123"
        assert msg.tool_calls[0].function.name == "get_weather"
        assert json.loads(msg.tool_calls[0].function.arguments) == {"city": "Seattle"}

    def test_tool_use_with_text_content(self):
        """Assistant message with text + tool_use preserves text blocks."""
        data = {
            "model": "claude-3",
            "max_tokens": 1024,
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Checking weather..."},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "get_weather",
                        "input": {"city": "NYC"},
                    },
                ],
            }],
        }
        result = AnthropicInTranslator.translate_messages_request(data)
        msg = result.messages[0]
        # Text content preserved as content blocks
        assert isinstance(msg.content, list)
        assert msg.content[0].text == "Checking weather..."
        # Tool call also present
        assert len(msg.tool_calls) == 1

    def test_tool_result_expansion(self):
        """User message with tool_result → expanded to TOOL role messages."""
        data = {
            "model": "claude-3",
            "max_tokens": 1024,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_123",
                        "content": "72F and sunny",
                    },
                ],
            }],
        }
        result = AnthropicInTranslator.translate_messages_request(data)
        # Should expand to a TOOL role message
        assert len(result.messages) == 1
        msg = result.messages[0]
        assert msg.role == MessageRole.TOOL
        assert msg.tool_call_id == "toolu_123"
        assert msg.get_text_content() == "72F and sunny"

    def test_tool_result_with_text(self):
        """User message with text + tool_result → user msg + tool msg."""
        data = {
            "model": "claude-3",
            "max_tokens": 1024,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here are the results:"},
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "result data",
                    },
                ],
            }],
        }
        result = AnthropicInTranslator.translate_messages_request(data)
        assert len(result.messages) == 2
        assert result.messages[0].role == MessageRole.USER
        assert result.messages[1].role == MessageRole.TOOL
        assert result.messages[1].tool_call_id == "toolu_1"

    def test_format_response_with_tool_calls(self):
        """format_response emits tool_use content blocks."""
        response = {
            "id": "resp_1",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Let me check.",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city":"Seattle"}',
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        }
        result = AnthropicInTranslator.format_response(response, "claude-3")
        assert result["stop_reason"] == "tool_use"
        # Should have text block + tool_use block
        blocks = result["content"]
        assert any(b["type"] == "text" for b in blocks)
        assert any(b["type"] == "tool_use" for b in blocks)
        tool_block = [b for b in blocks if b["type"] == "tool_use"][0]
        assert tool_block["name"] == "get_weather"
        assert tool_block["input"] == {"city": "Seattle"}

    def test_format_response_tool_only(self):
        """format_response with no text content, only tool_calls."""
        response = {
            "id": "resp_1",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city":"NYC"}',
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        result = AnthropicInTranslator.format_response(response, "claude-3")
        blocks = result["content"]
        # Should only have tool_use, no empty text block
        assert len(blocks) == 1
        assert blocks[0]["type"] == "tool_use"


# ── Phase 3a: vLLM Out ───────────────────────────────────

class TestVLLMOutToolCalling:
    """Tests for vLLM outbound tool calling translation."""

    def _make_canonical_with_tools(self):
        return CanonicalChatRequest(
            model="llama3",
            messages=[CanonicalMessage(role=MessageRole.USER, content="hi")],
            tools=[
                CanonicalToolDefinition(
                    function={
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                )
            ],
            tool_choice="auto",
        )

    def test_tools_passthrough(self):
        canonical = self._make_canonical_with_tools()
        payload = VLLMOutTranslator.translate_chat_request(canonical)
        assert "tools" in payload
        assert len(payload["tools"]) == 1
        assert payload["tools"][0]["function"]["name"] == "get_weather"
        assert payload["tool_choice"] == "auto"

    def test_tool_calls_in_message(self):
        msg = CanonicalMessage(
            role=MessageRole.ASSISTANT,
            content=None,
            tool_calls=[
                CanonicalToolCall(
                    id="call_1",
                    function=CanonicalFunctionCall(
                        name="get_weather", arguments='{"city":"NYC"}',
                    ),
                )
            ],
        )
        result = VLLMOutTranslator._translate_message(msg)
        assert result["content"] is None
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["id"] == "call_1"

    def test_tool_call_id_in_message(self):
        msg = CanonicalMessage(
            role=MessageRole.TOOL,
            content="72F",
            tool_call_id="call_1",
        )
        result = VLLMOutTranslator._translate_message(msg)
        assert result["tool_call_id"] == "call_1"
        assert result["role"] == "tool"

    def test_response_parsing_with_tool_calls(self):
        openai_resp = {
            "id": "resp_1",
            "created": 1234567890,
            "model": "llama3",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city":"NYC"}',
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        result = VLLMOutTranslator.translate_chat_response(openai_resp)
        msg = result.choices[0].message
        assert msg.content is None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].function.name == "get_weather"
        assert result.choices[0].finish_reason == "tool_calls"

    def test_no_tools_no_field(self):
        canonical = CanonicalChatRequest(
            model="llama3",
            messages=[CanonicalMessage(role=MessageRole.USER, content="hi")],
        )
        payload = VLLMOutTranslator.translate_chat_request(canonical)
        assert "tools" not in payload
        assert "tool_choice" not in payload


# ── Phase 3b: Ollama Out ─────────────────────────────────

class TestOllamaOutToolCalling:
    """Tests for Ollama outbound tool calling translation."""

    def test_tools_passthrough(self):
        canonical = CanonicalChatRequest(
            model="llama3",
            messages=[CanonicalMessage(role=MessageRole.USER, content="hi")],
            tools=[
                CanonicalToolDefinition(
                    function={
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                )
            ],
        )
        payload = OllamaOutTranslator.translate_chat_request(canonical)
        assert "tools" in payload
        assert payload["tools"][0]["function"]["name"] == "get_weather"

    def test_arguments_string_to_dict(self):
        """Canonical JSON string args → Ollama dict args."""
        msg = CanonicalMessage(
            role=MessageRole.ASSISTANT,
            content="",
            tool_calls=[
                CanonicalToolCall(
                    id="call_1",
                    function=CanonicalFunctionCall(
                        name="get_weather", arguments='{"city":"NYC"}',
                    ),
                )
            ],
        )
        result = OllamaOutTranslator._translate_message(msg)
        assert "tool_calls" in result
        assert result["tool_calls"][0]["function"]["arguments"] == {"city": "NYC"}

    def test_response_parsing_with_tool_calls(self):
        ollama_resp = {
            "model": "llama3",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "NYC"},
                    },
                }],
            },
            "done": True,
            "prompt_eval_count": 10,
            "eval_count": 5,
        }
        result = OllamaOutTranslator.translate_chat_response(
            ollama_resp, "req_1", "llama3"
        )
        msg = result.choices[0].message
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].function.name == "get_weather"
        # Arguments should be JSON string in canonical
        assert json.loads(msg.tool_calls[0].function.arguments) == {"city": "NYC"}
        assert msg.tool_calls[0].id == "call_0"  # synthetic ID
        assert result.choices[0].finish_reason == "tool_calls"

    def test_content_none_handling(self):
        msg = CanonicalMessage(
            role=MessageRole.ASSISTANT, content=None,
        )
        result = OllamaOutTranslator._translate_message(msg)
        assert result["content"] == ""


# ── Round-trip Tests ──────────────────────────────────────

class TestToolCallingRoundTrips:
    """End-to-end round-trip tests for tool calling."""

    def test_openai_to_vllm_roundtrip(self):
        """OpenAI request → canonical → vLLM payload."""
        openai_data = {
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city":"Seattle"}',
                        },
                    }],
                },
                {
                    "role": "tool",
                    "content": "72F and sunny",
                    "tool_call_id": "call_1",
                },
            ],
            "tools": SAMPLE_TOOLS_OPENAI,
            "tool_choice": "auto",
        }
        canonical = OpenAIInTranslator.translate_chat_request(openai_data)
        vllm_payload = VLLMOutTranslator.translate_chat_request(canonical)

        # Tools preserved
        assert len(vllm_payload["tools"]) == 1
        assert vllm_payload["tool_choice"] == "auto"

        # Messages preserved
        msgs = vllm_payload["messages"]
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["content"] is None
        assert msgs[1]["tool_calls"][0]["id"] == "call_1"
        assert msgs[2]["role"] == "tool"
        assert msgs[2]["tool_call_id"] == "call_1"

    def test_openai_to_ollama_roundtrip(self):
        """OpenAI request → canonical → Ollama payload."""
        openai_data = {
            "model": "gpt-4",
            "messages": [{
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city":"Seattle"}',
                    },
                }],
            }],
            "tools": SAMPLE_TOOLS_OPENAI,
        }
        canonical = OpenAIInTranslator.translate_chat_request(openai_data)
        ollama_payload = OllamaOutTranslator.translate_chat_request(canonical)

        # Tools preserved
        assert "tools" in ollama_payload
        # Arguments converted from JSON string to dict for Ollama
        msg = ollama_payload["messages"][0]
        assert msg["tool_calls"][0]["function"]["arguments"] == {"city": "Seattle"}

    def test_anthropic_to_vllm_roundtrip(self):
        """Anthropic request → canonical → vLLM payload."""
        anthro_data = {
            "model": "claude-3",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Let me check."},
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "get_weather",
                            "input": {"city": "Seattle"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "72F and sunny",
                    }],
                },
            ],
            "tools": SAMPLE_TOOLS_ANTHROPIC,
            "tool_choice": {"type": "auto"},
        }
        canonical = AnthropicInTranslator.translate_messages_request(anthro_data)
        vllm_payload = VLLMOutTranslator.translate_chat_request(canonical)

        # Tools translated
        assert len(vllm_payload["tools"]) == 1
        assert vllm_payload["tools"][0]["function"]["name"] == "get_weather"
        assert vllm_payload["tool_choice"] == "auto"

        # Messages: user, assistant (with tool_calls), tool
        msgs = vllm_payload["messages"]
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"
        assert len(msgs[1]["tool_calls"]) == 1
        assert msgs[1]["tool_calls"][0]["id"] == "toolu_1"
        # tool_result expanded to tool role
        assert msgs[2]["role"] == "tool"
        assert msgs[2]["tool_call_id"] == "toolu_1"

    def test_full_tool_cycle(self):
        """Complete: request with tools → backend response with tool_calls → Anthropic format."""
        # Simulate: Anthropic client sends request with tools
        anthro_data = {
            "model": "claude-3",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Weather in NYC?"}],
            "tools": SAMPLE_TOOLS_ANTHROPIC,
        }
        canonical = AnthropicInTranslator.translate_messages_request(anthro_data)

        # Simulate: backend returns tool_calls in OpenAI format
        backend_response = {
            "id": "resp_1",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city":"NYC"}',
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 50, "completion_tokens": 10},
        }

        # Format as Anthropic response
        anthro_resp = AnthropicInTranslator.format_response(
            backend_response, "claude-3"
        )

        assert anthro_resp["stop_reason"] == "tool_use"
        blocks = anthro_resp["content"]
        tool_blocks = [b for b in blocks if b["type"] == "tool_use"]
        assert len(tool_blocks) == 1
        assert tool_blocks[0]["name"] == "get_weather"
        assert tool_blocks[0]["input"] == {"city": "NYC"}
        assert tool_blocks[0]["id"] == "call_abc"
