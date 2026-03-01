"""Completion endpoint translation tests.

Tests for /v1/completions (OpenAI) and /api/generate (Ollama) translation
paths, including the CanonicalCompletionRequest.to_chat_request() bridge.
"""

import pytest

from backend.app.core.translators import (
    OpenAIInTranslator,
    OllamaInTranslator,
    OllamaOutTranslator,
    VLLMOutTranslator,
)
from backend.app.core.canonical_schemas import (
    CanonicalCompletionRequest,
    MessageRole,
)


# ===========================================================================
# OpenAI /v1/completions translation
# ===========================================================================

class TestOpenAICompletionTranslation:
    """translate_completion_request with all params."""

    def test_basic_completion(self):
        data = {"model": "gpt-3.5-turbo-instruct", "prompt": "Once upon a time"}
        canonical = OpenAIInTranslator.translate_completion_request(data)
        assert canonical.model == "gpt-3.5-turbo-instruct"
        assert canonical.prompt == "Once upon a time"
        assert canonical.stream is False

    def test_all_standard_params(self):
        data = {
            "model": "gpt-3.5-turbo-instruct",
            "prompt": "Hello",
            "temperature": 0.9,
            "top_p": 0.8,
            "max_tokens": 100,
            "stream": True,
            "stop": ["\n"],
            "presence_penalty": 0.5,
            "frequency_penalty": 0.3,
            "seed": 42,
            "suffix": " END",
            "echo": True,
            "n": 3,
            "best_of": 5,
        }
        canonical = OpenAIInTranslator.translate_completion_request(data)
        assert canonical.temperature == 0.9
        assert canonical.top_p == 0.8
        assert canonical.max_tokens == 100
        assert canonical.stream is True
        assert canonical.stop == ["\n"]
        assert canonical.presence_penalty == 0.5
        assert canonical.frequency_penalty == 0.3
        assert canonical.seed == 42
        assert canonical.suffix == " END"
        assert canonical.echo is True
        assert canonical.n == 3
        assert canonical.best_of == 5

    def test_extended_params(self):
        data = {
            "model": "gpt-3.5-turbo-instruct",
            "prompt": "Hello",
            "top_k": 40,
            "repetition_penalty": 1.2,
            "min_p": 0.05,
        }
        canonical = OpenAIInTranslator.translate_completion_request(data)
        assert canonical.top_k == 40
        assert canonical.repeat_penalty == 1.2
        assert canonical.min_p == 0.05

    def test_list_prompt(self):
        data = {
            "model": "gpt-3.5-turbo-instruct",
            "prompt": ["Hello", "World"],
        }
        canonical = OpenAIInTranslator.translate_completion_request(data)
        assert canonical.prompt == ["Hello", "World"]

    def test_defaults(self):
        data = {"model": "m", "prompt": "p"}
        canonical = OpenAIInTranslator.translate_completion_request(data)
        assert canonical.echo is False
        assert canonical.n == 1
        assert canonical.best_of == 1
        assert canonical.stream is False


# ===========================================================================
# CanonicalCompletionRequest.to_chat_request()
# ===========================================================================

class TestCompletionToChatConversion:
    """to_chat_request() preserves all fields."""

    def test_basic_conversion(self):
        comp = CanonicalCompletionRequest(model="m", prompt="Hello world")
        chat = comp.to_chat_request()
        assert chat.model == "m"
        assert len(chat.messages) == 1
        assert chat.messages[0].role == MessageRole.USER
        assert chat.messages[0].content == "Hello world"

    def test_standard_params_forwarded(self):
        comp = CanonicalCompletionRequest(
            model="m",
            prompt="p",
            temperature=0.5,
            top_p=0.9,
            max_tokens=200,
            stream=True,
            stop=["x"],
            presence_penalty=0.1,
            frequency_penalty=0.2,
            seed=7,
            n=2,
        )
        chat = comp.to_chat_request()
        assert chat.temperature == 0.5
        assert chat.top_p == 0.9
        assert chat.max_tokens == 200
        assert chat.stream is True
        assert chat.stop == ["x"]
        assert chat.presence_penalty == 0.1
        assert chat.frequency_penalty == 0.2
        assert chat.seed == 7
        assert chat.n == 2

    def test_extended_params_forwarded(self):
        comp = CanonicalCompletionRequest(
            model="m",
            prompt="p",
            top_k=40,
            repeat_penalty=1.2,
            min_p=0.05,
            backend_options={"mirostat": 2},
        )
        chat = comp.to_chat_request()
        assert chat.top_k == 40
        assert chat.repeat_penalty == 1.2
        assert chat.min_p == 0.05
        assert chat.backend_options == {"mirostat": 2}

    def test_list_prompt_uses_first(self):
        comp = CanonicalCompletionRequest(model="m", prompt=["first", "second"])
        chat = comp.to_chat_request()
        assert chat.messages[0].content == "first"

    def test_metadata_forwarded(self):
        comp = CanonicalCompletionRequest(
            model="m", prompt="p",
            request_id="req-1", user_id=42, api_key_id=7,
        )
        chat = comp.to_chat_request()
        assert chat.request_id == "req-1"
        assert chat.user_id == 42
        assert chat.api_key_id == 7


# ===========================================================================
# Ollama /api/generate round-trips
# ===========================================================================

class TestOllamaGenerateRoundTrip:
    """/api/generate → canonical → Ollama/vLLM out."""

    def test_basic_generate(self):
        data = {"model": "llama3.2", "prompt": "Hello"}
        canonical = OllamaInTranslator.translate_generate_request(data)
        assert canonical.model == "llama3.2"
        assert canonical.prompt == "Hello"
        assert canonical.stream is True  # Ollama default

    def test_system_prompt_combined(self):
        data = {
            "model": "llama3.2",
            "prompt": "What is 2+2?",
            "system": "Be concise.",
        }
        canonical = OllamaInTranslator.translate_generate_request(data)
        assert "Be concise." in canonical.prompt
        assert "What is 2+2?" in canonical.prompt

    def test_generate_options(self):
        data = {
            "model": "llama3.2",
            "prompt": "Hello",
            "options": {
                "temperature": 0.5,
                "top_k": 30,
                "repeat_penalty": 1.1,
                "min_p": 0.05,
                "num_predict": 100,
            },
        }
        canonical = OllamaInTranslator.translate_generate_request(data)
        assert canonical.temperature == 0.5
        assert canonical.top_k == 30
        assert canonical.repeat_penalty == 1.1
        assert canonical.min_p == 0.05
        assert canonical.max_tokens == 100

    def test_generate_backend_options(self):
        data = {
            "model": "llama3.2",
            "prompt": "Hello",
            "options": {"mirostat": 2, "num_ctx": 4096},
        }
        canonical = OllamaInTranslator.translate_generate_request(data)
        assert canonical.backend_options == {"mirostat": 2, "num_ctx": 4096}

    def test_generate_to_ollama_out(self):
        data = {
            "model": "llama3.2",
            "prompt": "Hello",
            "options": {"temperature": 0.5},
        }
        canonical = OllamaInTranslator.translate_generate_request(data)
        payload = OllamaOutTranslator.translate_generate_request(canonical)
        assert payload["model"] == "llama3.2"
        assert payload["prompt"] == "Hello"
        assert payload["options"]["temperature"] == 0.5

    def test_generate_to_vllm_out(self):
        data = {
            "model": "llama3.2",
            "prompt": "Hello",
            "options": {"temperature": 0.5, "top_k": 40},
        }
        canonical = OllamaInTranslator.translate_generate_request(data)
        payload = VLLMOutTranslator.translate_completion_request(canonical)
        assert payload["model"] == "llama3.2"
        assert payload["prompt"] == "Hello"
        assert payload["temperature"] == 0.5
        assert payload["top_k"] == 40


# ===========================================================================
# Completion output translation
# ===========================================================================

class TestCompletionOutTranslation:
    """Canonical completion → vLLM and Ollama output format."""

    def test_vllm_completion_output(self):
        canonical = CanonicalCompletionRequest(
            model="m", prompt="Hello",
            temperature=0.5, max_tokens=100,
            top_k=40, repeat_penalty=1.2, min_p=0.05,
        )
        payload = VLLMOutTranslator.translate_completion_request(canonical)
        assert payload["prompt"] == "Hello"
        assert payload["temperature"] == 0.5
        assert payload["max_tokens"] == 100
        assert payload["top_k"] == 40
        assert payload["repetition_penalty"] == 1.2
        assert payload["min_p"] == 0.05

    def test_ollama_generate_output(self):
        canonical = CanonicalCompletionRequest(
            model="m", prompt="Hello",
            temperature=0.5, max_tokens=100,
            top_k=40, repeat_penalty=1.2, min_p=0.05,
        )
        payload = OllamaOutTranslator.translate_generate_request(canonical)
        opts = payload["options"]
        assert opts["temperature"] == 0.5
        assert opts["num_predict"] == 100
        assert opts["top_k"] == 40
        assert opts["repeat_penalty"] == 1.2
        assert opts["min_p"] == 0.05

    def test_vllm_completion_suffix_echo(self):
        canonical = CanonicalCompletionRequest(
            model="m", prompt="Hello",
            suffix=" END", echo=True, n=3,
        )
        payload = VLLMOutTranslator.translate_completion_request(canonical)
        assert payload["suffix"] == " END"
        assert payload["echo"] is True
        assert payload["n"] == 3

    def test_vllm_completion_ignores_backend_options(self):
        canonical = CanonicalCompletionRequest(
            model="m", prompt="Hello",
            backend_options={"mirostat": 2},
        )
        payload = VLLMOutTranslator.translate_completion_request(canonical)
        assert "mirostat" not in payload
        assert "backend_options" not in payload

    def test_ollama_generate_merges_backend_options(self):
        canonical = CanonicalCompletionRequest(
            model="m", prompt="Hello",
            backend_options={"mirostat": 2, "num_ctx": 4096},
        )
        payload = OllamaOutTranslator.translate_generate_request(canonical)
        assert payload["options"]["mirostat"] == 2
        assert payload["options"]["num_ctx"] == 4096
