############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# test_anthropic_translator.py: Unit tests for Anthropic Messages API translator
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Unit tests for Anthropic Messages API translator."""

import json

import pytest

from backend.app.core.translators.anthropic_in import AnthropicInTranslator
from backend.app.core.canonical_schemas import (
    CanonicalChatRequest,
    CanonicalMessage,
    ImageBase64Content,
    ImageUrlContent,
    MessageRole,
    ResponseFormatType,
    TextContent,
)


class TestAnthropicInTranslator:
    """Tests for Anthropic Messages API to Canonical translation."""

    def test_basic_request(self):
        """Test basic messages request translation."""
        data = {
            "model": "claude-3-opus-20240229",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "Hello!"}
            ],
        }

        result = AnthropicInTranslator.translate_messages_request(data)

        assert isinstance(result, CanonicalChatRequest)
        assert result.model == "claude-3-opus-20240229"
        assert result.max_tokens == 1024
        assert len(result.messages) == 1
        assert result.messages[0].role == MessageRole.USER
        assert result.messages[0].content == "Hello!"
        assert result.stream is False

    def test_system_as_string(self):
        """Test system prompt as a simple string."""
        data = {
            "model": "claude-3-sonnet",
            "max_tokens": 512,
            "system": "You are a helpful assistant.",
            "messages": [
                {"role": "user", "content": "Hi"}
            ],
        }

        result = AnthropicInTranslator.translate_messages_request(data)

        assert len(result.messages) == 2
        assert result.messages[0].role == MessageRole.SYSTEM
        assert result.messages[0].content == "You are a helpful assistant."
        assert result.messages[1].role == MessageRole.USER

    def test_system_as_content_blocks(self):
        """Test system prompt as array of content blocks."""
        data = {
            "model": "claude-3-sonnet",
            "max_tokens": 512,
            "system": [
                {"type": "text", "text": "You are helpful."},
                {"type": "text", "text": "Be concise."},
            ],
            "messages": [
                {"role": "user", "content": "Hi"}
            ],
        }

        result = AnthropicInTranslator.translate_messages_request(data)

        assert len(result.messages) == 2
        assert result.messages[0].role == MessageRole.SYSTEM
        assert "You are helpful." in result.messages[0].content
        assert "Be concise." in result.messages[0].content

    def test_all_parameters(self):
        """Test all supported parameters mapping."""
        data = {
            "model": "claude-3-opus",
            "max_tokens": 2048,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "stop_sequences": ["END", "STOP"],
            "stream": True,
            "messages": [{"role": "user", "content": "Hi"}],
        }

        result = AnthropicInTranslator.translate_messages_request(data)

        assert result.max_tokens == 2048
        assert result.temperature == 0.7
        assert result.top_p == 0.9
        assert result.top_k == 40
        assert result.stop == ["END", "STOP"]
        assert result.stream is True

    def test_stream_defaults_false(self):
        """Test that stream defaults to False (Anthropic default)."""
        data = {
            "model": "claude-3-sonnet",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}],
        }

        result = AnthropicInTranslator.translate_messages_request(data)
        assert result.stream is False

    def test_multimodal_base64_image(self):
        """Test base64 image content block translation."""
        data = {
            "model": "claude-3-sonnet",
            "max_tokens": 1024,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": "abc123base64data",
                        },
                    },
                ],
            }],
        }

        result = AnthropicInTranslator.translate_messages_request(data)

        msg = result.messages[0]
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2
        assert isinstance(msg.content[0], TextContent)
        assert msg.content[0].text == "What is this?"
        assert isinstance(msg.content[1], ImageBase64Content)
        assert msg.content[1].data == "abc123base64data"
        assert msg.content[1].media_type == "image/jpeg"

    def test_multimodal_url_image(self):
        """Test URL image content block translation."""
        data = {
            "model": "claude-3-sonnet",
            "max_tokens": 1024,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this."},
                    {
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": "https://example.com/image.png",
                        },
                    },
                ],
            }],
        }

        result = AnthropicInTranslator.translate_messages_request(data)

        msg = result.messages[0]
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2
        assert isinstance(msg.content[1], ImageUrlContent)
        assert msg.content[1].image_url["url"] == "https://example.com/image.png"

    def test_thinking_enabled(self):
        """Test thinking mode enabled."""
        data = {
            "model": "claude-3-opus",
            "max_tokens": 4096,
            "thinking": {"type": "enabled", "budget_tokens": 2048},
            "messages": [{"role": "user", "content": "Think step by step."}],
        }

        result = AnthropicInTranslator.translate_messages_request(data)
        assert result.think is True

    def test_thinking_adaptive(self):
        """Test thinking mode adaptive maps to think=True."""
        data = {
            "model": "claude-3-opus",
            "max_tokens": 4096,
            "thinking": {"type": "adaptive"},
            "messages": [{"role": "user", "content": "Hi"}],
        }

        result = AnthropicInTranslator.translate_messages_request(data)
        assert result.think is True

    def test_thinking_disabled(self):
        """Test thinking mode disabled maps to think=False."""
        data = {
            "model": "claude-3-opus",
            "max_tokens": 4096,
            "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": "Hi"}],
        }

        result = AnthropicInTranslator.translate_messages_request(data)
        assert result.think is False

    def test_metadata_user_id(self):
        """Test metadata.user_id mapping to canonical user field."""
        data = {
            "model": "claude-3-sonnet",
            "max_tokens": 100,
            "metadata": {"user_id": "user-abc-123"},
            "messages": [{"role": "user", "content": "Hi"}],
        }

        result = AnthropicInTranslator.translate_messages_request(data)
        assert result.user == "user-abc-123"

    def test_structured_output_json_schema(self):
        """Test structured output via output_config.format."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        data = {
            "model": "claude-3-sonnet",
            "max_tokens": 1024,
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "name": "person",
                    "schema": schema,
                },
            },
            "messages": [{"role": "user", "content": "Give me JSON"}],
        }

        result = AnthropicInTranslator.translate_messages_request(data)

        assert result.response_format is not None
        assert result.response_format.type == ResponseFormatType.JSON_SCHEMA
        assert result.response_format.json_schema == {
            "name": "person",
            "schema": schema,
        }

    def test_multi_turn_conversation(self):
        """Test multi-turn conversation with user and assistant messages."""
        data = {
            "model": "claude-3-sonnet",
            "max_tokens": 512,
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
                {"role": "user", "content": "And 3+3?"},
            ],
        }

        result = AnthropicInTranslator.translate_messages_request(data)

        assert len(result.messages) == 3
        assert result.messages[0].role == MessageRole.USER
        assert result.messages[1].role == MessageRole.ASSISTANT
        assert result.messages[1].content == "4"
        assert result.messages[2].role == MessageRole.USER

    def test_tool_use_block_proper_extraction(self):
        """Test that tool_use blocks are properly extracted as tool_calls."""
        data = {
            "model": "claude-3-sonnet",
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
        # Text content preserved
        assert isinstance(msg.content, list)
        assert len(msg.content) == 1
        assert isinstance(msg.content[0], TextContent)
        assert msg.content[0].text == "Let me check."
        # Tool call extracted
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].id == "toolu_123"
        assert msg.tool_calls[0].function.name == "get_weather"


class TestAnthropicResponseFormatting:
    """Tests for canonical response to Anthropic Messages format conversion."""

    def test_non_streaming_response(self):
        """Test non-streaming response formatting."""
        openai_response = {
            "id": "chatcmpl-abc123",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello there!"},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }

        result = AnthropicInTranslator.format_response(openai_response, "claude-3-sonnet")

        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["model"] == "claude-3-sonnet"
        assert result["content"] == [{"type": "text", "text": "Hello there!"}]
        assert result["stop_reason"] == "end_turn"
        assert result["stop_sequence"] is None
        assert result["usage"]["input_tokens"] == 10
        assert result["usage"]["output_tokens"] == 5

    def test_finish_reason_length(self):
        """Test finish_reason 'length' maps to 'max_tokens'."""
        openai_response = {
            "id": "chatcmpl-abc",
            "choices": [{
                "message": {"content": "partial"},
                "finish_reason": "length",
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 100},
        }

        result = AnthropicInTranslator.format_response(openai_response, "model")
        assert result["stop_reason"] == "max_tokens"

    def test_finish_reason_tool_calls(self):
        """Test finish_reason 'tool_calls' maps to 'tool_use'."""
        openai_response = {
            "id": "chatcmpl-abc",
            "choices": [{
                "message": {"content": ""},
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10},
        }

        result = AnthropicInTranslator.format_response(openai_response, "model")
        assert result["stop_reason"] == "tool_use"

    def test_format_stream_event(self):
        """Test SSE event formatting."""
        event = AnthropicInTranslator.format_stream_event(
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}},
        )

        assert event.startswith("event: content_block_delta\n")
        assert "data: " in event
        assert event.endswith("\n\n")

        # Verify the data is valid JSON
        data_line = event.split("\n")[1]
        data_json = json.loads(data_line[6:])  # Strip "data: "
        assert data_json["type"] == "content_block_delta"
