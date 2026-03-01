"""Integration tests against real Ollama and vLLM backends.

These tests make actual HTTP requests to live backend services.
They verify the full translation pipeline: inbound translation ->
canonical -> outbound translation -> real HTTP call -> response translation.

Configure backend URLs via environment variables:
  OLLAMA_TEST_URL  (default: http://localhost:8001)
  VLLM_TEST_URL    (default: http://localhost:8002)
"""

import json
import os
import pytest
import httpx

from backend.app.core.translators import (
    OpenAIInTranslator,
    OllamaInTranslator,
    OllamaOutTranslator,
    VLLMOutTranslator,
)
from backend.app.core.canonical_schemas import (
    CanonicalChatResponse,
    CanonicalStreamChunk,
)

# --- Constants ---

OLLAMA_URL = os.environ.get("OLLAMA_TEST_URL", "http://localhost:8001")
VLLM_URL = os.environ.get("VLLM_TEST_URL", "http://localhost:8002")
OLLAMA_MODEL = "phi4:14b"
VLLM_MODEL = "openai/gpt-oss-120b"
OLLAMA_TIMEOUT = 60
VLLM_TIMEOUT = 180

integration = pytest.mark.integration


# --- Connectivity helpers ---


def _ollama_available() -> bool:
    try:
        r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def _vllm_available() -> bool:
    try:
        r = httpx.get(f"{VLLM_URL}/v1/models", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def _skip_if_ollama_unavailable():
    if not _ollama_available():
        pytest.skip("Ollama backend not available")


def _skip_if_vllm_unavailable():
    if not _vllm_available():
        pytest.skip("vLLM backend not available")


# --- Async stream helper ---


async def _async_iter_bytes(response):
    """Wrap httpx streaming response as async byte iterator."""
    for chunk in response.iter_bytes():
        yield chunk


async def _collect_async(async_gen):
    """Collect all items from an async generator."""
    items = []
    async for item in async_gen:
        items.append(item)
    return items


# ============================================================
# Ollama Direct (native format)
# ============================================================


@integration
class TestOllamaDirectChat:
    """Ollama-format requests sent directly to Ollama backend."""

    def setup_method(self):
        _skip_if_ollama_unavailable()

    def test_nonstreaming_chat(self):
        """Translate Ollama request -> POST -> translate response."""
        ollama_data = {
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": "Say hello in 5 words."}],
            "stream": False,
            "options": {"num_predict": 50},
        }

        canonical = OllamaInTranslator.translate_chat_request(ollama_data)
        payload = OllamaOutTranslator.translate_chat_request(canonical)
        payload["stream"] = False

        r = httpx.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=OLLAMA_TIMEOUT)
        assert r.status_code == 200

        data = r.json()
        response = OllamaOutTranslator.translate_chat_response(
            data, "test-req", OLLAMA_MODEL
        )

        assert isinstance(response, CanonicalChatResponse)
        assert len(response.choices) == 1
        assert response.choices[0].message.content
        assert len(response.choices[0].message.content) > 0
        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0

    @pytest.mark.asyncio
    async def test_streaming_chat(self):
        """Translate Ollama request -> stream POST -> parse ndjson."""
        ollama_data = {
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": "Say hi."}],
            "stream": True,
            "options": {"num_predict": 20},
        }

        canonical = OllamaInTranslator.translate_chat_request(ollama_data)
        payload = OllamaOutTranslator.translate_chat_request(canonical)
        payload["stream"] = True

        with httpx.stream(
            "POST", f"{OLLAMA_URL}/api/chat", json=payload, timeout=OLLAMA_TIMEOUT
        ) as r:
            assert r.status_code == 200
            chunks = await _collect_async(
                OllamaOutTranslator.translate_chat_stream(
                    _async_iter_bytes(r), "test-req", OLLAMA_MODEL
                )
            )

        assert len(chunks) > 0
        assert all(isinstance(c, CanonicalStreamChunk) for c in chunks)

        # Last chunk should have finish_reason
        final = chunks[-1]
        assert final.choices[0].finish_reason == "stop"

        # Accumulate content
        content = "".join(
            c.choices[0].delta.content or "" for c in chunks
        )
        assert len(content) > 0

    def test_json_format(self):
        """Ollama format:"json" produces valid JSON response."""
        ollama_data = {
            "model": OLLAMA_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": "Return a JSON object with keys 'greeting' (string) and 'count' (integer).",
                }
            ],
            "stream": False,
            "format": "json",
            "options": {"num_predict": 50},
        }

        canonical = OllamaInTranslator.translate_chat_request(ollama_data)
        payload = OllamaOutTranslator.translate_chat_request(canonical)
        payload["stream"] = False

        r = httpx.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=OLLAMA_TIMEOUT)
        assert r.status_code == 200

        data = r.json()
        response = OllamaOutTranslator.translate_chat_response(
            data, "test-req", OLLAMA_MODEL
        )
        content = response.choices[0].message.content
        parsed = json.loads(content)
        assert isinstance(parsed, dict)

    def test_schema_format(self):
        """Ollama format:{schema} produces response matching the schema."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }

        ollama_data = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "user", "content": "Return a person named Alice who is 30."}
            ],
            "stream": False,
            "format": schema,
            "options": {"num_predict": 50},
        }

        canonical = OllamaInTranslator.translate_chat_request(ollama_data)
        payload = OllamaOutTranslator.translate_chat_request(canonical)
        payload["stream"] = False

        r = httpx.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=OLLAMA_TIMEOUT)
        assert r.status_code == 200

        data = r.json()
        response = OllamaOutTranslator.translate_chat_response(
            data, "test-req", OLLAMA_MODEL
        )
        content = response.choices[0].message.content
        parsed = json.loads(content)
        assert "name" in parsed
        assert "age" in parsed
        assert isinstance(parsed["name"], str)
        assert isinstance(parsed["age"], int)

    def test_all_parameters(self):
        """All Ollama options are accepted by the backend."""
        ollama_data = {
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": "Say hello."}],
            "stream": False,
            "options": {
                "temperature": 0.5,
                "top_p": 0.9,
                "num_predict": 20,
                "seed": 42,
            },
        }

        canonical = OllamaInTranslator.translate_chat_request(ollama_data)
        payload = OllamaOutTranslator.translate_chat_request(canonical)
        payload["stream"] = False

        r = httpx.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=OLLAMA_TIMEOUT)
        assert r.status_code == 200

        data = r.json()
        assert data.get("done") is True
        assert data.get("message", {}).get("content")


# ============================================================
# vLLM Direct (OpenAI format)
# ============================================================


@integration
class TestVLLMDirectChat:
    """OpenAI-format requests sent directly to vLLM backend."""

    def setup_method(self):
        _skip_if_vllm_unavailable()

    def test_nonstreaming_chat(self):
        """Translate OpenAI request -> POST -> translate response."""
        openai_data = {
            "model": VLLM_MODEL,
            "messages": [{"role": "user", "content": "Say hello in 5 words."}],
            "stream": False,
            "max_tokens": 200,
        }

        canonical = OpenAIInTranslator.translate_chat_request(openai_data)
        payload = VLLMOutTranslator.translate_chat_request(canonical)
        payload["stream"] = False

        r = httpx.post(
            f"{VLLM_URL}/v1/chat/completions", json=payload, timeout=VLLM_TIMEOUT
        )
        assert r.status_code == 200

        data = r.json()
        response = VLLMOutTranslator.translate_chat_response(data, "test-req")

        assert isinstance(response, CanonicalChatResponse)
        assert len(response.choices) == 1
        # vLLM reasoning model may return None content with short max_tokens
        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0

    @pytest.mark.asyncio
    async def test_streaming_chat(self):
        """Translate OpenAI request -> SSE stream -> parse chunks."""
        openai_data = {
            "model": VLLM_MODEL,
            "messages": [{"role": "user", "content": "Say hi."}],
            "stream": True,
            "max_tokens": 200,
        }

        canonical = OpenAIInTranslator.translate_chat_request(openai_data)
        payload = VLLMOutTranslator.translate_chat_request(canonical)
        payload["stream"] = True

        with httpx.stream(
            "POST",
            f"{VLLM_URL}/v1/chat/completions",
            json=payload,
            timeout=VLLM_TIMEOUT,
        ) as r:
            assert r.status_code == 200
            chunks = await _collect_async(
                VLLMOutTranslator.translate_chat_stream(
                    _async_iter_bytes(r), "test-req", VLLM_MODEL
                )
            )

        assert len(chunks) > 0
        assert all(isinstance(c, CanonicalStreamChunk) for c in chunks)

    def test_json_object_mode(self):
        """vLLM response_format:json_object produces valid JSON."""
        openai_data = {
            "model": VLLM_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": "Return a JSON object with keys 'greeting' (string) and 'count' (integer). Only output the JSON.",
                }
            ],
            "stream": False,
            "max_tokens": 200,
            "response_format": {"type": "json_object"},
        }

        canonical = OpenAIInTranslator.translate_chat_request(openai_data)
        payload = VLLMOutTranslator.translate_chat_request(canonical)
        payload["stream"] = False

        r = httpx.post(
            f"{VLLM_URL}/v1/chat/completions", json=payload, timeout=VLLM_TIMEOUT
        )
        assert r.status_code == 200

        data = r.json()
        response = VLLMOutTranslator.translate_chat_response(data, "test-req")
        content = response.choices[0].message.content
        assert content is not None
        parsed = json.loads(content)
        assert isinstance(parsed, dict)

    def test_with_parameters(self):
        """Temperature, max_tokens, seed accepted by vLLM."""
        openai_data = {
            "model": VLLM_MODEL,
            "messages": [{"role": "user", "content": "Say hello."}],
            "stream": False,
            "temperature": 0.5,
            "max_tokens": 200,
            "seed": 42,
        }

        canonical = OpenAIInTranslator.translate_chat_request(openai_data)
        payload = VLLMOutTranslator.translate_chat_request(canonical)
        payload["stream"] = False

        r = httpx.post(
            f"{VLLM_URL}/v1/chat/completions", json=payload, timeout=VLLM_TIMEOUT
        )
        assert r.status_code == 200

        data = r.json()
        assert "choices" in data
        assert "usage" in data


# ============================================================
# Cross-engine: Ollama request -> vLLM backend
# ============================================================


@integration
class TestOllamaToVLLMCrossEngine:
    """Ollama-format request translated and sent to vLLM."""

    def setup_method(self):
        _skip_if_vllm_unavailable()

    def test_chat_translated(self):
        """Ollama dict -> OllamaIn -> canonical -> VLLMOut -> POST vLLM."""
        ollama_data = {
            "model": VLLM_MODEL,
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "Say hello."},
            ],
            "stream": False,
            "options": {"num_predict": 200},
        }

        # Inbound: Ollama -> Canonical
        canonical = OllamaInTranslator.translate_chat_request(ollama_data)
        assert canonical.max_tokens == 200

        # Outbound: Canonical -> vLLM
        vllm_payload = VLLMOutTranslator.translate_chat_request(canonical)
        vllm_payload["stream"] = False
        assert vllm_payload["max_tokens"] == 200
        assert vllm_payload["messages"][0]["role"] == "system"

        # Real HTTP call to vLLM
        r = httpx.post(
            f"{VLLM_URL}/v1/chat/completions",
            json=vllm_payload,
            timeout=VLLM_TIMEOUT,
        )
        assert r.status_code == 200

        # Translate response back
        data = r.json()
        response = VLLMOutTranslator.translate_chat_response(data, "test-req")
        assert isinstance(response, CanonicalChatResponse)
        assert response.usage.prompt_tokens > 0

    def test_structured_output_translated(self):
        """Ollama format:"json" -> vLLM response_format:json_object."""
        ollama_data = {
            "model": VLLM_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": "Return a JSON object with a 'message' key. Only output the JSON.",
                }
            ],
            "stream": False,
            "format": "json",
            "options": {"num_predict": 200},
        }

        canonical = OllamaInTranslator.translate_chat_request(ollama_data)
        vllm_payload = VLLMOutTranslator.translate_chat_request(canonical)
        vllm_payload["stream"] = False

        # Verify format was translated
        assert vllm_payload["response_format"] == {"type": "json_object"}

        r = httpx.post(
            f"{VLLM_URL}/v1/chat/completions",
            json=vllm_payload,
            timeout=VLLM_TIMEOUT,
        )
        assert r.status_code == 200

        data = r.json()
        response = VLLMOutTranslator.translate_chat_response(data, "test-req")
        content = response.choices[0].message.content
        assert content is not None
        parsed = json.loads(content)
        assert isinstance(parsed, dict)

    def test_parameters_preserved(self):
        """Ollama options translated to vLLM top-level params."""
        ollama_data = {
            "model": VLLM_MODEL,
            "messages": [{"role": "user", "content": "Say hello."}],
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 200,
                "seed": 123,
            },
        }

        canonical = OllamaInTranslator.translate_chat_request(ollama_data)
        vllm_payload = VLLMOutTranslator.translate_chat_request(canonical)
        vllm_payload["stream"] = False

        assert vllm_payload["temperature"] == 0.7
        assert vllm_payload["top_p"] == 0.9
        assert vllm_payload["max_tokens"] == 200
        assert vllm_payload["seed"] == 123

        r = httpx.post(
            f"{VLLM_URL}/v1/chat/completions",
            json=vllm_payload,
            timeout=VLLM_TIMEOUT,
        )
        assert r.status_code == 200
        assert r.json()["usage"]["prompt_tokens"] > 0


# ============================================================
# Cross-engine: OpenAI request -> Ollama backend
# ============================================================


@integration
class TestVLLMToOllamaCrossEngine:
    """OpenAI-format request translated and sent to Ollama."""

    def setup_method(self):
        _skip_if_ollama_unavailable()

    def test_chat_translated(self):
        """OpenAI dict -> OpenAIIn -> canonical -> OllamaOut -> POST Ollama."""
        openai_data = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "Say hello."},
            ],
            "temperature": 0.7,
            "max_tokens": 50,
            "stream": False,
        }

        # Inbound: OpenAI -> Canonical
        canonical = OpenAIInTranslator.translate_chat_request(openai_data)
        assert canonical.temperature == 0.7
        assert canonical.max_tokens == 50

        # Outbound: Canonical -> Ollama
        ollama_payload = OllamaOutTranslator.translate_chat_request(canonical)
        ollama_payload["stream"] = False
        assert ollama_payload["options"]["temperature"] == 0.7
        assert ollama_payload["options"]["num_predict"] == 50

        # Real HTTP call to Ollama
        r = httpx.post(
            f"{OLLAMA_URL}/api/chat", json=ollama_payload, timeout=OLLAMA_TIMEOUT
        )
        assert r.status_code == 200

        # Translate response back to canonical
        data = r.json()
        response = OllamaOutTranslator.translate_chat_response(
            data, "test-req", OLLAMA_MODEL
        )
        assert isinstance(response, CanonicalChatResponse)
        assert response.choices[0].message.content
        assert response.usage.prompt_tokens > 0

    def test_structured_output_translated(self):
        """OpenAI response_format:json_object -> Ollama format:"json"."""
        openai_data = {
            "model": OLLAMA_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": "Return a JSON object with a 'message' key.",
                }
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": 50,
            "stream": False,
        }

        canonical = OpenAIInTranslator.translate_chat_request(openai_data)
        ollama_payload = OllamaOutTranslator.translate_chat_request(canonical)
        ollama_payload["stream"] = False

        # Verify format was translated
        assert ollama_payload["format"] == "json"

        r = httpx.post(
            f"{OLLAMA_URL}/api/chat", json=ollama_payload, timeout=OLLAMA_TIMEOUT
        )
        assert r.status_code == 200

        data = r.json()
        response = OllamaOutTranslator.translate_chat_response(
            data, "test-req", OLLAMA_MODEL
        )
        content = response.choices[0].message.content
        parsed = json.loads(content)
        assert isinstance(parsed, dict)

    @pytest.mark.asyncio
    async def test_streaming_cross_engine(self):
        """OpenAI streaming intent -> translated to Ollama streaming -> canonical chunks."""
        openai_data = {
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": "Say hi."}],
            "stream": True,
            "max_tokens": 20,
        }

        canonical = OpenAIInTranslator.translate_chat_request(openai_data)
        ollama_payload = OllamaOutTranslator.translate_chat_request(canonical)
        ollama_payload["stream"] = True

        with httpx.stream(
            "POST", f"{OLLAMA_URL}/api/chat", json=ollama_payload, timeout=OLLAMA_TIMEOUT
        ) as r:
            assert r.status_code == 200
            chunks = await _collect_async(
                OllamaOutTranslator.translate_chat_stream(
                    _async_iter_bytes(r), "test-req", OLLAMA_MODEL
                )
            )

        assert len(chunks) > 0
        content = "".join(c.choices[0].delta.content or "" for c in chunks)
        assert len(content) > 0
        assert chunks[-1].choices[0].finish_reason == "stop"


# ============================================================
# Response translation fidelity
# ============================================================


@integration
class TestResponseTranslationFidelity:
    """Verify canonical responses have expected structure and usage."""

    def test_ollama_response_has_usage(self):
        """Ollama response translates to canonical with usage > 0."""
        _skip_if_ollama_unavailable()

        payload = {
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": "Hello."}],
            "stream": False,
            "options": {"num_predict": 20},
        }

        r = httpx.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=OLLAMA_TIMEOUT)
        assert r.status_code == 200

        response = OllamaOutTranslator.translate_chat_response(
            r.json(), "test-req", OLLAMA_MODEL
        )
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens == (
            response.usage.prompt_tokens + response.usage.completion_tokens
        )

    def test_vllm_response_has_usage(self):
        """vLLM response translates to canonical with usage > 0."""
        _skip_if_vllm_unavailable()

        payload = {
            "model": VLLM_MODEL,
            "messages": [{"role": "user", "content": "Hello."}],
            "stream": False,
            "max_tokens": 200,
        }

        r = httpx.post(
            f"{VLLM_URL}/v1/chat/completions", json=payload, timeout=VLLM_TIMEOUT
        )
        assert r.status_code == 200

        response = VLLMOutTranslator.translate_chat_response(r.json(), "test-req")
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens == (
            response.usage.prompt_tokens + response.usage.completion_tokens
        )

    def test_canonical_response_shape(self):
        """Canonical model_dump() has expected OpenAI-compatible shape."""
        _skip_if_ollama_unavailable()

        payload = {
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": "Hello."}],
            "stream": False,
            "options": {"num_predict": 20},
        }

        r = httpx.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=OLLAMA_TIMEOUT)
        assert r.status_code == 200

        response = OllamaOutTranslator.translate_chat_response(
            r.json(), "test-req", OLLAMA_MODEL
        )
        d = response.model_dump()

        assert "id" in d
        assert "model" in d
        assert "choices" in d
        assert "usage" in d
        assert d["object"] == "chat.completion"
        assert isinstance(d["choices"], list)
        assert len(d["choices"]) > 0
        assert "message" in d["choices"][0]
        assert "content" in d["choices"][0]["message"]
        assert "role" in d["choices"][0]["message"]
