############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# test_translators.py: Unit tests for API translation layer
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Unit tests for API translators."""

import pytest

from backend.app.core.translators import (
    OpenAIInTranslator,
    OllamaInTranslator,
    OllamaOutTranslator,
    VLLMOutTranslator,
)
from backend.app.core.canonical_schemas import (
    CanonicalChatRequest,
    CanonicalMessage,
    MessageRole,
    ResponseFormatType,
)


class TestOpenAIInTranslator:
    """Tests for OpenAI to Canonical translation."""

    def test_basic_chat_request(self):
        """Test basic chat request translation."""
        data = {
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "Hello!"}
            ],
        }

        result = OpenAIInTranslator.translate_chat_request(data)

        assert isinstance(result, CanonicalChatRequest)
        assert result.model == "gpt-4"
        assert len(result.messages) == 1
        assert result.messages[0].role == MessageRole.USER
        assert result.messages[0].content == "Hello!"
        assert result.stream is False

    def test_chat_request_with_parameters(self):
        """Test chat request with optional parameters."""
        data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0.7,
            "max_tokens": 100,
            "stream": True,
            "stop": ["END"],
        }

        result = OpenAIInTranslator.translate_chat_request(data)

        assert result.temperature == 0.7
        assert result.max_tokens == 100
        assert result.stream is True
        assert result.stop == ["END"]

    def test_chat_request_with_system_message(self):
        """Test chat request with system message."""
        data = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello!"},
            ],
        }

        result = OpenAIInTranslator.translate_chat_request(data)

        assert len(result.messages) == 2
        assert result.messages[0].role == MessageRole.SYSTEM
        assert result.get_system_prompt() == "You are helpful."

    def test_chat_request_with_json_mode(self):
        """Test chat request with JSON response format."""
        data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "List 3 items as JSON"}],
            "response_format": {"type": "json_object"},
        }

        result = OpenAIInTranslator.translate_chat_request(data)

        assert result.response_format is not None
        assert result.response_format.type == ResponseFormatType.JSON_OBJECT
        assert result.requires_structured_output() is True

    def test_chat_request_with_json_schema(self):
        """Test chat request with JSON schema."""
        data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Generate user data"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"type": "object", "properties": {"name": {"type": "string"}}},
            },
        }

        result = OpenAIInTranslator.translate_chat_request(data)

        assert result.response_format.type == ResponseFormatType.JSON_SCHEMA
        assert result.response_format.json_schema is not None

    def test_multimodal_content(self):
        """Test multimodal (image) content parsing."""
        data = {
            "model": "gpt-4-vision",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
                    ],
                }
            ],
        }

        result = OpenAIInTranslator.translate_chat_request(data)

        assert len(result.messages) == 1
        assert result.messages[0].has_images() is True
        assert result.requires_multimodal() is True

    def test_embedding_request(self):
        """Test embedding request translation."""
        data = {
            "model": "text-embedding-ada-002",
            "input": "Hello world",
        }

        result = OpenAIInTranslator.translate_embedding_request(data)

        assert result.model == "text-embedding-ada-002"
        assert result.input == "Hello world"


class TestOllamaInTranslator:
    """Tests for Ollama to Canonical translation."""

    def test_basic_chat_request(self):
        """Test basic Ollama chat request."""
        data = {
            "model": "llama3.2",
            "messages": [
                {"role": "user", "content": "Hello!"}
            ],
        }

        result = OllamaInTranslator.translate_chat_request(data)

        assert result.model == "llama3.2"
        assert len(result.messages) == 1
        assert result.stream is True  # Ollama defaults to streaming

    def test_chat_with_options(self):
        """Test chat request with Ollama options."""
        data = {
            "model": "llama3.2",
            "messages": [{"role": "user", "content": "Hi"}],
            "options": {
                "temperature": 0.5,
                "num_predict": 200,
            },
            "stream": False,
        }

        result = OllamaInTranslator.translate_chat_request(data)

        assert result.temperature == 0.5
        assert result.max_tokens == 200
        assert result.stream is False

    def test_chat_with_json_format(self):
        """Test chat with Ollama JSON format."""
        data = {
            "model": "llama3.2",
            "messages": [{"role": "user", "content": "List items"}],
            "format": "json",
        }

        result = OllamaInTranslator.translate_chat_request(data)

        assert result.response_format is not None
        assert result.response_format.type == ResponseFormatType.JSON_OBJECT

    def test_generate_request(self):
        """Test Ollama generate request."""
        data = {
            "model": "llama3.2",
            "prompt": "Once upon a time",
            "system": "You are a storyteller.",
        }

        result = OllamaInTranslator.translate_generate_request(data)

        assert result.model == "llama3.2"
        assert "You are a storyteller." in result.prompt
        assert "Once upon a time" in result.prompt


class TestOllamaOutTranslator:
    """Tests for Canonical to Ollama translation."""

    def test_basic_chat_request(self):
        """Test canonical to Ollama chat format."""
        canonical = CanonicalChatRequest(
            model="llama3.2",
            messages=[
                CanonicalMessage(role=MessageRole.USER, content="Hello!"),
            ],
            temperature=0.7,
            max_tokens=100,
            stream=False,
        )

        result = OllamaOutTranslator.translate_chat_request(canonical)

        assert result["model"] == "llama3.2"
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "Hello!"
        assert result["stream"] is False
        assert result["options"]["temperature"] == 0.7
        assert result["options"]["num_predict"] == 100

    def test_json_format_translation(self):
        """Test JSON format translation to Ollama."""
        from backend.app.core.canonical_schemas import ResponseFormat

        canonical = CanonicalChatRequest(
            model="llama3.2",
            messages=[CanonicalMessage(role=MessageRole.USER, content="List items")],
            response_format=ResponseFormat(type=ResponseFormatType.JSON_OBJECT),
        )

        result = OllamaOutTranslator.translate_chat_request(canonical)

        assert result["format"] == "json"


class TestVLLMOutTranslator:
    """Tests for Canonical to vLLM/OpenAI translation."""

    def test_basic_chat_request(self):
        """Test canonical to OpenAI format."""
        canonical = CanonicalChatRequest(
            model="llama3.2",
            messages=[
                CanonicalMessage(role=MessageRole.SYSTEM, content="Be helpful"),
                CanonicalMessage(role=MessageRole.USER, content="Hello!"),
            ],
            temperature=0.8,
            stream=True,
        )

        result = VLLMOutTranslator.translate_chat_request(canonical)

        assert result["model"] == "llama3.2"
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][1]["role"] == "user"
        assert result["temperature"] == 0.8
        assert result["stream"] is True

    def test_embedding_request(self):
        """Test embedding request translation."""
        from backend.app.core.canonical_schemas import CanonicalEmbeddingRequest

        canonical = CanonicalEmbeddingRequest(
            model="text-embedding",
            input=["Hello", "World"],
        )

        result = VLLMOutTranslator.translate_embedding_request(canonical)

        assert result["model"] == "text-embedding"
        assert result["input"] == ["Hello", "World"]
