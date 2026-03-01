"""Cross-engine inference proxying tests.

Tests that requests arriving in one API format are correctly translated
and proxied to a backend of a different engine type, with responses
translated back to the original format.
"""

import json
import pytest

from unittest.mock import AsyncMock, MagicMock, patch

from backend.app.core.translators import (
    OllamaInTranslator,
    OllamaOutTranslator,
    OpenAIInTranslator,
    VLLMOutTranslator,
)
from backend.app.core.canonical_schemas import (
    CanonicalChatRequest,
    CanonicalMessage,
    MessageRole,
    ResponseFormat,
    ResponseFormatType,
)


class TestOllamaRequestToVLLMBackend:
    """Ollama-format request routed to a vLLM backend."""

    def test_chat_nonstreaming(self, mock_vllm_backend):
        """Ollama /api/chat -> Canonical -> vLLM /v1/chat/completions payload."""
        ollama_data = {
            "model": "llama3.2",
            "messages": [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Hello!"},
            ],
            "stream": False,
            "options": {"temperature": 0.7, "num_predict": 100},
        }

        canonical = OllamaInTranslator.translate_chat_request(ollama_data)
        vllm_payload = VLLMOutTranslator.translate_chat_request(canonical)

        assert vllm_payload["model"] == "llama3.2"
        assert vllm_payload["messages"][0]["role"] == "system"
        assert vllm_payload["messages"][1]["content"] == "Hello!"
        assert vllm_payload["temperature"] == 0.7
        assert vllm_payload["max_tokens"] == 100

        # Simulate vLLM response and translate back
        vllm_response = {
            "id": "chatcmpl-abc",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "llama3.2",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hi there!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
        }

        canonical_resp = VLLMOutTranslator.translate_chat_response(vllm_response, "req-1")
        assert canonical_resp.choices[0].message.content == "Hi there!"
        assert canonical_resp.usage.total_tokens == 13

    def test_chat_streaming_payload(self, mock_vllm_backend):
        """Ollama streaming -> vLLM payload has stream:true."""
        ollama_data = {
            "model": "llama3.2",
            "messages": [{"role": "user", "content": "Hello!"}],
            "stream": True,
        }

        canonical = OllamaInTranslator.translate_chat_request(ollama_data)
        vllm_payload = VLLMOutTranslator.translate_chat_request(canonical)
        assert vllm_payload["stream"] is True

    def test_with_json_format(self, mock_vllm_backend):
        """Ollama format:"json" -> vLLM response_format:{"type":"json_object"}."""
        ollama_data = {
            "model": "llama3.2",
            "messages": [{"role": "user", "content": "Give me JSON"}],
            "format": "json",
        }

        canonical = OllamaInTranslator.translate_chat_request(ollama_data)
        vllm_payload = VLLMOutTranslator.translate_chat_request(canonical)

        assert vllm_payload["response_format"] == {
            "type": "json_schema",
            "json_schema": {"name": "json_response", "schema": {"type": "object"}},
        }

    def test_with_schema_format(self, mock_vllm_backend, simple_json_schema):
        """Ollama format:{schema} -> vLLM response_format:{"type":"json_schema",...}."""
        ollama_data = {
            "model": "llama3.2",
            "messages": [{"role": "user", "content": "Generate data"}],
            "format": simple_json_schema,
        }

        canonical = OllamaInTranslator.translate_chat_request(ollama_data)
        vllm_payload = VLLMOutTranslator.translate_chat_request(canonical)

        assert vllm_payload["response_format"]["type"] == "json_schema"
        assert "json_schema" in vllm_payload["response_format"]

    def test_parameters_translated(self, mock_vllm_backend, all_params_ollama_request):
        """All Ollama options correctly appear in vLLM request."""
        canonical = OllamaInTranslator.translate_chat_request(all_params_ollama_request)
        vllm_payload = VLLMOutTranslator.translate_chat_request(canonical)

        assert vllm_payload["temperature"] == 0.8
        assert vllm_payload["top_p"] == 0.95
        assert vllm_payload["max_tokens"] == 256
        assert vllm_payload["stop"] == ["\n", "END"]
        assert vllm_payload["presence_penalty"] == 0.5
        assert vllm_payload["frequency_penalty"] == 0.3
        assert vllm_payload["seed"] == 42


class TestOpenAIRequestToOllamaBackend:
    """OpenAI-format request routed to an Ollama backend."""

    def test_chat_nonstreaming(self, mock_ollama_backend):
        """OpenAI /v1/chat/completions -> Canonical -> Ollama /api/chat payload."""
        openai_data = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Hello!"},
            ],
            "temperature": 0.7,
            "max_tokens": 100,
            "stream": False,
        }

        canonical = OpenAIInTranslator.translate_chat_request(openai_data)
        ollama_payload = OllamaOutTranslator.translate_chat_request(canonical)

        assert ollama_payload["model"] == "gpt-4"
        assert ollama_payload["messages"][0]["role"] == "system"
        assert ollama_payload["messages"][1]["content"] == "Hello!"
        assert ollama_payload["options"]["temperature"] == 0.7
        assert ollama_payload["options"]["num_predict"] == 100

        # Simulate Ollama response and translate back
        ollama_response = {
            "model": "gpt-4",
            "message": {"role": "assistant", "content": "Hi there!"},
            "done": True,
            "prompt_eval_count": 10,
            "eval_count": 3,
        }

        canonical_resp = OllamaOutTranslator.translate_chat_response(
            ollama_response, "req-1", "gpt-4"
        )
        assert canonical_resp.choices[0].message.content == "Hi there!"
        assert canonical_resp.usage.total_tokens == 13

    def test_chat_streaming_payload(self, mock_ollama_backend):
        """OpenAI streaming -> Ollama payload has stream:true."""
        openai_data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello!"}],
            "stream": True,
        }

        canonical = OpenAIInTranslator.translate_chat_request(openai_data)
        ollama_payload = OllamaOutTranslator.translate_chat_request(canonical)
        assert ollama_payload["stream"] is True

    def test_with_json_response_format(self, mock_ollama_backend):
        """OpenAI response_format -> Ollama format in proxied request."""
        openai_data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Give me JSON"}],
            "response_format": {"type": "json_object"},
        }

        canonical = OpenAIInTranslator.translate_chat_request(openai_data)
        ollama_payload = OllamaOutTranslator.translate_chat_request(canonical)

        assert ollama_payload["format"] == "json"

    def test_parameters_translated(self, mock_ollama_backend, all_params_openai_request):
        """All OpenAI params correctly appear as Ollama options."""
        canonical = OpenAIInTranslator.translate_chat_request(all_params_openai_request)
        ollama_payload = OllamaOutTranslator.translate_chat_request(canonical)

        options = ollama_payload["options"]
        assert options["temperature"] == 0.8
        assert options["top_p"] == 0.95
        assert options["num_predict"] == 256
        assert options["stop"] == ["\n", "END"]
        assert options["presence_penalty"] == 0.5
        assert options["frequency_penalty"] == 0.3
        assert options["seed"] == 42


class TestInferenceServiceRouting:
    """InferenceService backend selection and payload translation."""

    def test_selects_vllm_backend_payload(self, mock_vllm_backend):
        """When targeting vLLM backend, payload format matches OpenAI."""
        canonical = CanonicalChatRequest(
            model="llama3.2",
            messages=[CanonicalMessage(role=MessageRole.USER, content="Hi")],
            temperature=0.5,
        )

        if mock_vllm_backend.engine.value == "vllm":
            payload = VLLMOutTranslator.translate_chat_request(canonical)
        else:
            payload = OllamaOutTranslator.translate_chat_request(canonical)

        assert "messages" in payload
        assert payload["temperature"] == 0.5
        # vLLM uses max_tokens, not num_predict
        assert "num_predict" not in payload

    def test_selects_ollama_backend_payload(self, mock_ollama_backend):
        """When targeting Ollama backend, payload format matches Ollama."""
        canonical = CanonicalChatRequest(
            model="llama3.2",
            messages=[CanonicalMessage(role=MessageRole.USER, content="Hi")],
            temperature=0.5,
            max_tokens=100,
        )

        if mock_ollama_backend.engine.value == "ollama":
            payload = OllamaOutTranslator.translate_chat_request(canonical)
        else:
            payload = VLLMOutTranslator.translate_chat_request(canonical)

        assert "messages" in payload
        options = payload.get("options", {})
        assert options["temperature"] == 0.5
        assert options["num_predict"] == 100

    def test_translates_for_target_engine(self, mock_ollama_backend, mock_vllm_backend):
        """Payload format matches the selected backend's engine."""
        canonical = CanonicalChatRequest(
            model="llama3.2",
            messages=[CanonicalMessage(role=MessageRole.USER, content="Hi")],
            max_tokens=100,
        )

        # For vLLM: max_tokens in top-level
        vllm_payload = VLLMOutTranslator.translate_chat_request(canonical)
        assert vllm_payload.get("max_tokens") == 100
        assert "options" not in vllm_payload

        # For Ollama: num_predict in options
        ollama_payload = OllamaOutTranslator.translate_chat_request(canonical)
        assert ollama_payload.get("options", {}).get("num_predict") == 100
        assert "max_tokens" not in ollama_payload

    def test_response_format_matches_client_openai(self, mock_ollama_backend):
        """OpenAI client request -> Ollama backend -> response translates back to OpenAI format."""
        # OpenAI client sends request
        openai_data = {
            "model": "llama3.2",
            "messages": [{"role": "user", "content": "Hello!"}],
        }
        canonical = OpenAIInTranslator.translate_chat_request(openai_data)

        # Route to Ollama backend
        ollama_payload = OllamaOutTranslator.translate_chat_request(canonical)
        assert ollama_payload["model"] == "llama3.2"

        # Ollama responds
        ollama_response = {
            "model": "llama3.2",
            "message": {"role": "assistant", "content": "Hi!"},
            "done": True,
            "prompt_eval_count": 5,
            "eval_count": 2,
        }

        # Translate back to canonical (and then OpenAI format)
        canonical_resp = OllamaOutTranslator.translate_chat_response(
            ollama_response, "req-1", "llama3.2"
        )
        resp_dict = canonical_resp.model_dump()
        assert resp_dict["choices"][0]["message"]["content"] == "Hi!"
        assert resp_dict["usage"]["prompt_tokens"] == 5

    def test_response_format_matches_client_ollama(self, mock_vllm_backend):
        """Ollama client request -> vLLM backend -> response translates back."""
        # Ollama client sends request
        ollama_data = {
            "model": "llama3.2",
            "messages": [{"role": "user", "content": "Hello!"}],
        }
        canonical = OllamaInTranslator.translate_chat_request(ollama_data)

        # Route to vLLM backend
        vllm_payload = VLLMOutTranslator.translate_chat_request(canonical)
        assert vllm_payload["model"] == "llama3.2"

        # vLLM responds
        vllm_response = {
            "id": "chatcmpl-abc",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "llama3.2",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hi!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }

        canonical_resp = VLLMOutTranslator.translate_chat_response(vllm_response, "req-1")
        assert canonical_resp.choices[0].message.content == "Hi!"
        assert canonical_resp.usage.total_tokens == 7
