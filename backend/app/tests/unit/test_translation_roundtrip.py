"""Cross-format round-trip fidelity tests.

Tests that translating a request from one API format through canonical
and out to the other format preserves all data correctly.
"""

import base64
import pytest

from backend.app.core.translators import (
    OpenAIInTranslator,
    OllamaInTranslator,
    OllamaOutTranslator,
    VLLMOutTranslator,
)
from backend.app.core.canonical_schemas import (
    CanonicalChatRequest,
    CanonicalEmbeddingRequest,
    CanonicalMessage,
    MessageRole,
    ResponseFormat,
    ResponseFormatType,
    ImageBase64Content,
    ImageUrlContent,
    TextContent,
)


class TestOllamaToVLLMRoundTrip:
    """Ollama request -> Canonical -> vLLM payload round-trip."""

    def test_basic_chat(self):
        """Messages and model preserved through Ollama -> Canonical -> vLLM."""
        ollama_data = {
            "model": "llama3.2",
            "messages": [
                {"role": "user", "content": "Hello!"},
            ],
        }

        canonical = OllamaInTranslator.translate_chat_request(ollama_data)
        vllm_payload = VLLMOutTranslator.translate_chat_request(canonical)

        assert vllm_payload["model"] == "llama3.2"
        assert len(vllm_payload["messages"]) == 1
        assert vllm_payload["messages"][0]["role"] == "user"
        assert vllm_payload["messages"][0]["content"] == "Hello!"

    def test_all_parameters(self, all_params_ollama_request):
        """Every Ollama option maps correctly to vLLM equivalents."""
        canonical = OllamaInTranslator.translate_chat_request(all_params_ollama_request)
        vllm_payload = VLLMOutTranslator.translate_chat_request(canonical)

        assert vllm_payload["temperature"] == 0.8
        assert vllm_payload["top_p"] == 0.95
        assert vllm_payload["max_tokens"] == 256
        assert vllm_payload["stop"] == ["\n", "END"]
        assert vllm_payload["presence_penalty"] == 0.5
        assert vllm_payload["frequency_penalty"] == 0.3
        assert vllm_payload["seed"] == 42

    def test_system_message(self):
        """Ollama system in messages -> vLLM system message."""
        ollama_data = {
            "model": "llama3.2",
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Hello!"},
            ],
        }

        canonical = OllamaInTranslator.translate_chat_request(ollama_data)
        vllm_payload = VLLMOutTranslator.translate_chat_request(canonical)

        assert vllm_payload["messages"][0]["role"] == "system"
        assert vllm_payload["messages"][0]["content"] == "Be concise."
        assert vllm_payload["messages"][1]["role"] == "user"

    def test_multi_turn_conversation(self):
        """Multiple user/assistant turns preserved in order."""
        ollama_data = {
            "model": "llama3.2",
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "Good."},
                {"role": "user", "content": "Great."},
            ],
        }

        canonical = OllamaInTranslator.translate_chat_request(ollama_data)
        vllm_payload = VLLMOutTranslator.translate_chat_request(canonical)

        assert len(vllm_payload["messages"]) == 5
        roles = [m["role"] for m in vllm_payload["messages"]]
        assert roles == ["user", "assistant", "user", "assistant", "user"]
        assert vllm_payload["messages"][0]["content"] == "Hi"
        assert vllm_payload["messages"][4]["content"] == "Great."

    def test_streaming_flag_true(self):
        """Ollama stream:true (default) -> vLLM stream:true."""
        ollama_data = {
            "model": "llama3.2",
            "messages": [{"role": "user", "content": "Hi"}],
        }

        canonical = OllamaInTranslator.translate_chat_request(ollama_data)
        vllm_payload = VLLMOutTranslator.translate_chat_request(canonical)

        assert canonical.stream is True
        assert vllm_payload["stream"] is True

    def test_streaming_flag_false(self):
        """Explicit Ollama stream:false -> vLLM stream:false."""
        ollama_data = {
            "model": "llama3.2",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
        }

        canonical = OllamaInTranslator.translate_chat_request(ollama_data)
        vllm_payload = VLLMOutTranslator.translate_chat_request(canonical)

        assert canonical.stream is False
        assert vllm_payload["stream"] is False

    def test_json_format_to_response_format(self):
        """Ollama format:"json" -> Canonical JSON_OBJECT -> vLLM response_format."""
        ollama_data = {
            "model": "llama3.2",
            "messages": [{"role": "user", "content": "List items as JSON"}],
            "format": "json",
        }

        canonical = OllamaInTranslator.translate_chat_request(ollama_data)
        assert canonical.response_format.type == ResponseFormatType.JSON_OBJECT

        vllm_payload = VLLMOutTranslator.translate_chat_request(canonical)
        assert vllm_payload["response_format"] == {
            "type": "json_schema",
            "json_schema": {"name": "json_response", "schema": {"type": "object"}},
        }

    def test_json_schema_format_to_response_format(self, simple_json_schema):
        """Ollama format:{schema} -> Canonical JSON_SCHEMA -> vLLM response_format."""
        ollama_data = {
            "model": "llama3.2",
            "messages": [{"role": "user", "content": "Generate data"}],
            "format": simple_json_schema,
        }

        canonical = OllamaInTranslator.translate_chat_request(ollama_data)
        assert canonical.response_format.type == ResponseFormatType.JSON_SCHEMA
        assert canonical.response_format.json_schema["schema"] == simple_json_schema

        vllm_payload = VLLMOutTranslator.translate_chat_request(canonical)
        assert vllm_payload["response_format"]["type"] == "json_schema"
        assert "json_schema" in vllm_payload["response_format"]

    def test_images_base64(self):
        """Ollama images:["base64..."] -> Canonical ImageBase64Content -> vLLM data URL."""
        # Minimal valid PNG header in base64
        png_header = base64.b64encode(b'\x89PNG\r\n\x1a\n' + b'\x00' * 20).decode()

        ollama_data = {
            "model": "llava",
            "messages": [
                {
                    "role": "user",
                    "content": "What is this?",
                    "images": [png_header],
                }
            ],
        }

        canonical = OllamaInTranslator.translate_chat_request(ollama_data)
        assert canonical.requires_multimodal() is True

        vllm_payload = VLLMOutTranslator.translate_chat_request(canonical)
        content_blocks = vllm_payload["messages"][0]["content"]
        assert isinstance(content_blocks, list)

        # Find the image block
        image_block = [b for b in content_blocks if b.get("type") == "image_url"][0]
        url = image_block["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
        assert png_header in url


class TestVLLMToOllamaRoundTrip:
    """OpenAI request -> Canonical -> Ollama payload round-trip."""

    def test_basic_chat(self):
        """Messages and model preserved through OpenAI -> Canonical -> Ollama."""
        openai_data = {
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "Hello!"},
            ],
        }

        canonical = OpenAIInTranslator.translate_chat_request(openai_data)
        ollama_payload = OllamaOutTranslator.translate_chat_request(canonical)

        assert ollama_payload["model"] == "gpt-4"
        assert len(ollama_payload["messages"]) == 1
        assert ollama_payload["messages"][0]["role"] == "user"
        assert ollama_payload["messages"][0]["content"] == "Hello!"

    def test_all_parameters(self, all_params_openai_request):
        """Every OpenAI param -> Ollama options mapping."""
        canonical = OpenAIInTranslator.translate_chat_request(all_params_openai_request)
        ollama_payload = OllamaOutTranslator.translate_chat_request(canonical)

        options = ollama_payload.get("options", {})
        assert options["temperature"] == 0.8
        assert options["top_p"] == 0.95
        assert options["num_predict"] == 256
        assert options["stop"] == ["\n", "END"]
        assert options["presence_penalty"] == 0.5
        assert options["frequency_penalty"] == 0.3
        assert options["seed"] == 42

    def test_json_object_to_format(self):
        """OpenAI response_format:{"type":"json_object"} -> Canonical -> Ollama format:"json"."""
        openai_data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "List items"}],
            "response_format": {"type": "json_object"},
        }

        canonical = OpenAIInTranslator.translate_chat_request(openai_data)
        assert canonical.response_format.type == ResponseFormatType.JSON_OBJECT

        ollama_payload = OllamaOutTranslator.translate_chat_request(canonical)
        assert ollama_payload["format"] == "json"

    def test_json_schema_to_format(self, simple_json_schema):
        """OpenAI json_schema -> Canonical -> Ollama format:{schema}."""
        openai_data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Generate data"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"schema": simple_json_schema},
            },
        }

        canonical = OpenAIInTranslator.translate_chat_request(openai_data)
        assert canonical.response_format.type == ResponseFormatType.JSON_SCHEMA

        ollama_payload = OllamaOutTranslator.translate_chat_request(canonical)
        assert isinstance(ollama_payload["format"], dict)

    def test_image_url_to_base64(self):
        """OpenAI image_url content (data URI) -> Canonical -> Ollama images."""
        b64_data = base64.b64encode(b'\x89PNG' + b'\x00' * 20).decode()

        openai_data = {
            "model": "gpt-4-vision",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_data}"},
                        },
                    ],
                }
            ],
        }

        canonical = OpenAIInTranslator.translate_chat_request(openai_data)
        assert canonical.requires_multimodal() is True

        ollama_payload = OllamaOutTranslator.translate_chat_request(canonical)
        msg = ollama_payload["messages"][0]
        assert "images" in msg
        assert msg["images"][0] == b64_data

    def test_n_and_user_fields(self):
        """OpenAI-specific fields n, user handled when going to Ollama."""
        openai_data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "n": 2,
            "user": "test-user-123",
        }

        canonical = OpenAIInTranslator.translate_chat_request(openai_data)
        assert canonical.n == 2
        assert canonical.user == "test-user-123"

        # Ollama doesn't support n/user, but translation should not fail
        ollama_payload = OllamaOutTranslator.translate_chat_request(canonical)
        assert ollama_payload["model"] == "gpt-4"
        assert "n" not in ollama_payload
        assert "user" not in ollama_payload


class TestResponseRoundTrip:
    """Response translation tests."""

    def test_ollama_response_to_canonical(self):
        """Ollama non-streaming response -> Canonical."""
        ollama_response = {
            "model": "llama3.2",
            "message": {"role": "assistant", "content": "Hello there!"},
            "done": True,
            "prompt_eval_count": 15,
            "eval_count": 5,
            "total_duration": 500000000,
        }

        canonical = OllamaOutTranslator.translate_chat_response(
            ollama_response, "req-123", "llama3.2"
        )

        assert canonical.id == "req-123"
        assert canonical.model == "llama3.2"
        assert len(canonical.choices) == 1
        assert canonical.choices[0].message.content == "Hello there!"
        assert canonical.choices[0].finish_reason == "stop"
        assert canonical.usage.prompt_tokens == 15
        assert canonical.usage.completion_tokens == 5
        assert canonical.usage.total_tokens == 20

    def test_vllm_response_to_canonical(self):
        """vLLM/OpenAI response -> Canonical."""
        vllm_response = {
            "id": "chatcmpl-abc",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "llama3.2",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello there!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 3,
                "total_tokens": 13,
            },
        }

        canonical = VLLMOutTranslator.translate_chat_response(vllm_response, "req-456")

        assert canonical.id == "req-456"
        assert canonical.model == "llama3.2"
        assert len(canonical.choices) == 1
        assert canonical.choices[0].message.content == "Hello there!"
        assert canonical.choices[0].finish_reason == "stop"
        assert canonical.usage.prompt_tokens == 10
        assert canonical.usage.completion_tokens == 3
        assert canonical.usage.total_tokens == 13

    def test_embedding_ollama_to_canonical(self):
        """Ollama embedding response format -> Canonical."""
        ollama_response = {
            "embedding": [0.1, 0.2, 0.3],
            "prompt_eval_count": 5,
        }

        canonical = OllamaOutTranslator.translate_embedding_response(
            ollama_response, "nomic-embed-text"
        )

        assert canonical.model == "nomic-embed-text"
        assert len(canonical.data) == 1
        assert canonical.data[0]["embedding"] == [0.1, 0.2, 0.3]
        assert canonical.usage.prompt_tokens == 5

    def test_embedding_vllm_to_canonical(self):
        """vLLM embedding response format -> Canonical."""
        vllm_response = {
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0}
            ],
            "model": "text-embedding",
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }

        canonical = VLLMOutTranslator.translate_embedding_response(vllm_response)

        assert canonical.model == "text-embedding"
        assert len(canonical.data) == 1
        assert canonical.data[0]["embedding"] == [0.1, 0.2, 0.3]
        assert canonical.usage.prompt_tokens == 5


# ===========================================================================
# Vision edge cases
# ===========================================================================

class TestVisionEdgeCases:
    """Edge cases for vision/multimodal content translation."""

    def test_multiple_images_openai_to_ollama(self):
        """Multiple images in one message."""
        b64_1 = base64.b64encode(b'\x89PNG' + b'\x00' * 20).decode()
        b64_2 = base64.b64encode(b'\xff\xd8\xff' + b'\x00' * 20).decode()

        openai_data = {
            "model": "gpt-4-vision",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare these images"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_1}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_2}"}},
                ],
            }],
        }
        canonical = OpenAIInTranslator.translate_chat_request(openai_data)
        assert canonical.requires_multimodal() is True

        ollama_payload = OllamaOutTranslator.translate_chat_request(canonical)
        msg = ollama_payload["messages"][0]
        assert len(msg["images"]) == 2

    def test_image_detail_level_preserved(self):
        """Image detail level preserved through OpenAI → Canonical → vLLM."""
        openai_data = {
            "model": "gpt-4-vision",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://example.com/img.png",
                            "detail": "high",
                        },
                    },
                ],
            }],
        }
        canonical = OpenAIInTranslator.translate_chat_request(openai_data)
        vllm_payload = VLLMOutTranslator.translate_chat_request(canonical)

        content = vllm_payload["messages"][0]["content"]
        img_block = [b for b in content if b.get("type") == "image_url"][0]
        assert img_block["image_url"]["detail"] == "high"

    def test_url_only_images_dropped_by_ollama(self):
        """URL-only images (no base64) are silently dropped in Ollama output."""
        canonical = CanonicalChatRequest(
            model="llava",
            messages=[
                CanonicalMessage(
                    role=MessageRole.USER,
                    content=[
                        TextContent(text="Describe"),
                        ImageUrlContent(image_url={
                            "url": "https://example.com/img.png",
                            "detail": "auto",
                        }),
                    ],
                )
            ],
        )
        ollama_payload = OllamaOutTranslator.translate_chat_request(canonical)
        msg = ollama_payload["messages"][0]
        assert "images" not in msg  # URL images can't be sent to Ollama
        assert msg["content"] == "Describe"

    def test_mixed_base64_and_url_images(self):
        """Base64 images kept, URL images dropped for Ollama output."""
        b64_data = base64.b64encode(b'\x89PNG' + b'\x00' * 20).decode()
        canonical = CanonicalChatRequest(
            model="llava",
            messages=[
                CanonicalMessage(
                    role=MessageRole.USER,
                    content=[
                        TextContent(text="Describe"),
                        ImageBase64Content(data=b64_data, media_type="image/png"),
                        ImageUrlContent(image_url={
                            "url": "https://example.com/img.png",
                            "detail": "auto",
                        }),
                    ],
                )
            ],
        )
        ollama_payload = OllamaOutTranslator.translate_chat_request(canonical)
        msg = ollama_payload["messages"][0]
        assert len(msg["images"]) == 1  # Only base64 image
        assert msg["images"][0] == b64_data


# ===========================================================================
# Embedding parameter gaps
# ===========================================================================

class TestEmbeddingParameterGaps:
    """encoding_format, dimensions, and multi-input embedding edge cases."""

    def test_encoding_format_preserved(self):
        data = {
            "model": "text-embedding",
            "input": "Hello",
            "encoding_format": "base64",
        }
        canonical = OpenAIInTranslator.translate_embedding_request(data)
        assert canonical.encoding_format == "base64"

        vllm_payload = VLLMOutTranslator.translate_embedding_request(canonical)
        assert vllm_payload["encoding_format"] == "base64"

    def test_encoding_format_default_float(self):
        data = {"model": "text-embedding", "input": "Hello"}
        canonical = OpenAIInTranslator.translate_embedding_request(data)
        assert canonical.encoding_format == "float"

        vllm_payload = VLLMOutTranslator.translate_embedding_request(canonical)
        assert "encoding_format" not in vllm_payload  # float is default, omitted

    def test_dimensions_preserved(self):
        data = {
            "model": "text-embedding",
            "input": "Hello",
            "dimensions": 512,
        }
        canonical = OpenAIInTranslator.translate_embedding_request(data)
        assert canonical.dimensions == 512

        vllm_payload = VLLMOutTranslator.translate_embedding_request(canonical)
        assert vllm_payload["dimensions"] == 512

    def test_multi_input_to_ollama_uses_first(self):
        """Ollama only supports single input — documents taking first element."""
        canonical = CanonicalEmbeddingRequest(
            model="nomic-embed-text",
            input=["first", "second", "third"],
        )
        ollama_payload = OllamaOutTranslator.translate_embedding_request(canonical)
        assert ollama_payload["prompt"] == "first"

    def test_multi_input_to_vllm_preserved(self):
        """vLLM supports list inputs natively."""
        canonical = CanonicalEmbeddingRequest(
            model="text-embedding",
            input=["first", "second"],
        )
        vllm_payload = VLLMOutTranslator.translate_embedding_request(canonical)
        assert vllm_payload["input"] == ["first", "second"]

    def test_ollama_embedding_input_or_prompt(self):
        """Ollama embedding accepts both 'input' and 'prompt' keys."""
        data1 = {"model": "m", "input": "hello"}
        data2 = {"model": "m", "prompt": "hello"}
        c1 = OllamaInTranslator.translate_embedding_request(data1)
        c2 = OllamaInTranslator.translate_embedding_request(data2)
        assert c1.input == "hello"
        assert c2.input == "hello"


# ===========================================================================
# Edge cases and negative tests
# ===========================================================================

class TestEdgeCasesAndNegative:
    """stop as string vs array, unknown fields, tool messages, empty messages."""

    def test_stop_as_string(self):
        """stop can be a string (not just an array)."""
        openai_data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hi"}],
            "stop": "END",
        }
        canonical = OpenAIInTranslator.translate_chat_request(openai_data)
        assert canonical.stop == "END"

        # Round-trip through both outputs
        vllm_payload = VLLMOutTranslator.translate_chat_request(canonical)
        assert vllm_payload["stop"] == "END"

        ollama_payload = OllamaOutTranslator.translate_chat_request(canonical)
        assert ollama_payload["options"]["stop"] == "END"

    def test_stop_as_array(self):
        openai_data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hi"}],
            "stop": ["\n", "END"],
        }
        canonical = OpenAIInTranslator.translate_chat_request(openai_data)
        assert canonical.stop == ["\n", "END"]

    def test_unknown_fields_ignored(self):
        """Unknown top-level fields in OpenAI request don't cause errors."""
        openai_data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hi"}],
            "unknown_field": "should be ignored",
            "another_unknown": 42,
        }
        canonical = OpenAIInTranslator.translate_chat_request(openai_data)
        assert canonical.model == "gpt-4"

    def test_tool_message_role(self):
        """Tool role messages translated correctly."""
        openai_data = {
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "Use the calculator"},
                {"role": "tool", "content": "42", "name": "calculator"},
            ],
        }
        canonical = OpenAIInTranslator.translate_chat_request(openai_data)
        assert canonical.messages[1].role == MessageRole.TOOL
        assert canonical.messages[1].content == "42"
        assert canonical.messages[1].name == "calculator"

    def test_empty_content_message(self):
        """Empty content string handled gracefully."""
        openai_data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": ""}],
        }
        canonical = OpenAIInTranslator.translate_chat_request(openai_data)
        assert canonical.messages[0].content == ""
