"""
Live structured output integration tests.

Hits the live MindRouter2 deployment to verify end-to-end structured output
behavior across all API surfaces (OpenAI, Ollama, Anthropic), multiple
schema types, streaming modes, and models.

Usage:
    python -m pytest backend/app/tests/integration/test_structured_outputs_live.py -v \
        --api-key <KEY> --base-url https://mindrouter.uidaho.edu
"""

import json

import httpx
import jsonschema
import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIMEOUT = httpx.Timeout(connect=10, read=180, write=10, pool=10)

MODELS = [
    "qwen/qwen3.5-400b",
    "openai/gpt-oss-20b",
    "phi4:14b",
    "qwen3-32k:32b",
    "qwen2.5-32k:7b",
]

integration = pytest.mark.integration

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

SIMPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
    },
    "required": ["name", "age"],
}

NESTED_SCHEMA = {
    "type": "object",
    "properties": {
        "person": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "address": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "zip": {"type": "string"},
                    },
                    "required": ["city", "zip"],
                },
            },
            "required": ["name", "address"],
        },
    },
    "required": ["person"],
}

ENUM_SCHEMA = {
    "type": "object",
    "properties": {
        "color": {"type": "string", "enum": ["red", "green", "blue"]},
        "value": {"type": "integer"},
    },
    "required": ["color", "value"],
}

ARRAY_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "quantity": {"type": "integer"},
                },
                "required": ["name", "quantity"],
            },
        },
    },
    "required": ["items"],
}

COMPLEX_SCHEMA = {
    "type": "object",
    "properties": {
        "countries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "capital": {"type": "string"},
                    "population": {"type": "integer"},
                    "languages": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["name", "capital", "population", "languages"],
            },
        },
    },
    "required": ["countries"],
}

# Prompts tailored to each schema
SCHEMA_PROMPTS = {
    "simple": (
        "Return a JSON object with keys 'name' (string) and 'age' (integer). "
        "The person is Alice, age 30. Only output the JSON."
    ),
    "nested": (
        "Return a JSON object with a 'person' key containing 'name' (string) "
        "and 'address' (object with 'city' and 'zip' as strings). "
        "Use Alice in New York, zip 10001. Only output the JSON."
    ),
    "enum": (
        "Return a JSON object with 'color' (one of: red, green, blue) and "
        "'value' (integer). Pick color blue and value 42. Only output the JSON."
    ),
    "array": (
        "Return a JSON object with an 'items' array containing 2 objects, "
        "each with 'name' (string) and 'quantity' (integer). "
        "Use apples:3 and bananas:5. Only output the JSON."
    ),
    "complex": (
        "Return a JSON object with a 'countries' array containing 2 country "
        "objects. Each has 'name' (string), 'capital' (string), 'population' "
        "(integer), and 'languages' (array of strings). "
        "Use France and Japan. Only output the JSON."
    ),
}

SCHEMA_MAP = {
    "simple": SIMPLE_SCHEMA,
    "nested": NESTED_SCHEMA,
    "enum": ENUM_SCHEMA,
    "array": ARRAY_SCHEMA,
    "complex": COMPLEX_SCHEMA,
}

# Transient HTTP status codes — skip, don't fail
SKIP_STATUS_CODES = {404, 502, 503}

# ---------------------------------------------------------------------------
# Helpers — strip markdown fences
# ---------------------------------------------------------------------------


def _strip_markdown_fences(text: str) -> str:
    """Strip ```json ... ``` wrappers that models sometimes add."""
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.index("\n") if "\n" in text else len(text)
        text = text[first_newline + 1:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


# ---------------------------------------------------------------------------
# Helpers — OpenAI endpoint
# ---------------------------------------------------------------------------


def openai_chat(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    stream: bool,
    response_format: dict | None = None,
    max_tokens: int = 512,
) -> str:
    """Call /v1/chat/completions and return the content string."""
    body: dict = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "max_tokens": max_tokens,
    }
    if response_format is not None:
        body["response_format"] = response_format

    headers = {"Authorization": f"Bearer {api_key}"}

    with httpx.Client(base_url=base_url, timeout=TIMEOUT) as client:
        try:
            if stream:
                with client.stream(
                    "POST", "/v1/chat/completions", headers=headers, json=body
                ) as r:
                    if r.status_code in SKIP_STATUS_CODES:
                        pytest.skip(f"HTTP {r.status_code} (transient)")
                    assert r.status_code == 200, f"HTTP {r.status_code}: {_safe_text(r)}"
                    return _collect_openai_stream(r)
            else:
                r = client.post("/v1/chat/completions", headers=headers, json=body)
                if r.status_code in SKIP_STATUS_CODES:
                    pytest.skip(f"HTTP {r.status_code} (transient): {r.text[:200]}")
                assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:500]}"
                data = r.json()
                return _extract_openai_content(data)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            pytest.skip(f"Connection failed: {exc}")
        except httpx.RemoteProtocolError as exc:
            pytest.fail(f"Backend crashed mid-response: {exc}")
        except httpx.ReadTimeout as exc:
            pytest.fail(f"Read timeout: {exc}")


# ---------------------------------------------------------------------------
# Helpers — Ollama endpoint
# ---------------------------------------------------------------------------


def ollama_chat(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    stream: bool,
    format: str | dict | None = None,
    max_tokens: int = 512,
) -> str:
    """Call /api/chat and return the content string."""
    body: dict = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "options": {"num_predict": max_tokens},
    }
    if format is not None:
        body["format"] = format

    headers = {"Authorization": f"Bearer {api_key}"}

    with httpx.Client(base_url=base_url, timeout=TIMEOUT) as client:
        try:
            if stream:
                with client.stream(
                    "POST", "/api/chat", headers=headers, json=body
                ) as r:
                    if r.status_code in SKIP_STATUS_CODES:
                        pytest.skip(f"HTTP {r.status_code} (transient)")
                    assert r.status_code == 200, f"HTTP {r.status_code}: {_safe_text(r)}"
                    return _collect_ollama_stream(r)
            else:
                r = client.post("/api/chat", headers=headers, json=body)
                if r.status_code in SKIP_STATUS_CODES:
                    pytest.skip(f"HTTP {r.status_code} (transient): {r.text[:200]}")
                assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:500]}"
                data = r.json()
                return _extract_ollama_content(data)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            pytest.skip(f"Connection failed: {exc}")
        except httpx.RemoteProtocolError as exc:
            pytest.fail(f"Backend crashed mid-response: {exc}")
        except httpx.ReadTimeout as exc:
            pytest.fail(f"Read timeout: {exc}")


# ---------------------------------------------------------------------------
# Helpers — Anthropic endpoint
# ---------------------------------------------------------------------------


def anthropic_chat(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    stream: bool,
    output_config: dict | None = None,
    max_tokens: int = 512,
) -> str:
    """Call /anthropic/v1/messages and return the content string."""
    body: dict = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "max_tokens": max_tokens,
    }
    if output_config is not None:
        body["output_config"] = output_config

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }

    with httpx.Client(base_url=base_url, timeout=TIMEOUT) as client:
        try:
            if stream:
                with client.stream(
                    "POST", "/anthropic/v1/messages", headers=headers, json=body
                ) as r:
                    if r.status_code in SKIP_STATUS_CODES:
                        pytest.skip(f"HTTP {r.status_code} (transient)")
                    assert r.status_code == 200, f"HTTP {r.status_code}: {_safe_text(r)}"
                    return _collect_anthropic_stream(r)
            else:
                r = client.post("/anthropic/v1/messages", headers=headers, json=body)
                if r.status_code in SKIP_STATUS_CODES:
                    pytest.skip(f"HTTP {r.status_code} (transient): {r.text[:200]}")
                assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:500]}"
                data = r.json()
                return _extract_anthropic_content(data)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            pytest.skip(f"Connection failed: {exc}")
        except httpx.RemoteProtocolError as exc:
            pytest.fail(f"Backend crashed mid-response: {exc}")
        except httpx.ReadTimeout as exc:
            pytest.fail(f"Read timeout: {exc}")


# ---------------------------------------------------------------------------
# Stream collectors
# ---------------------------------------------------------------------------


def _safe_text(r: httpx.Response) -> str:
    """Safely read up to 200 chars from a streaming response."""
    try:
        return r.text[:200]
    except Exception:
        return "(unreadable)"


def _collect_openai_stream(response: httpx.Response) -> str:
    """Collect content from OpenAI SSE stream (with reasoning_content fallback)."""
    content_parts = []
    reasoning_parts = []
    for line in response.iter_lines():
        if not line or not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        if payload.strip() == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            if delta.get("content"):
                content_parts.append(delta["content"])
            if delta.get("reasoning_content"):
                reasoning_parts.append(delta["reasoning_content"])
        except json.JSONDecodeError:
            continue
    content = "".join(content_parts)
    if not content:
        content = "".join(reasoning_parts)
    return content


def _collect_ollama_stream(response: httpx.Response) -> str:
    """Collect content from Ollama NDJSON stream (with thinking fallback)."""
    content_parts = []
    thinking_parts = []
    for line in response.iter_lines():
        if not line:
            continue
        try:
            chunk = json.loads(line)
            msg = chunk.get("message", {})
            if msg.get("content"):
                content_parts.append(msg["content"])
            if msg.get("thinking"):
                thinking_parts.append(msg["thinking"])
            if chunk.get("done"):
                break
        except json.JSONDecodeError:
            continue
    content = "".join(content_parts)
    if not content:
        content = "".join(thinking_parts)
    return content


def _collect_anthropic_stream(response: httpx.Response) -> str:
    """Collect content from Anthropic SSE stream (with thinking fallback)."""
    content_parts = []
    thinking_parts = []
    for line in response.iter_lines():
        if not line:
            continue
        if line.startswith("data: "):
            payload = line[len("data: "):]
            try:
                data = json.loads(payload)
                event_type = data.get("type", "")
                if event_type == "content_block_delta":
                    delta = data.get("delta", {})
                    if delta.get("text"):
                        content_parts.append(delta["text"])
                    if delta.get("thinking"):
                        thinking_parts.append(delta["thinking"])
                elif event_type == "message_stop":
                    break
            except json.JSONDecodeError:
                continue
    content = "".join(content_parts)
    if not content:
        content = "".join(thinking_parts)
    return content


# ---------------------------------------------------------------------------
# Content extractors (non-streaming)
# ---------------------------------------------------------------------------


def _extract_openai_content(data: dict) -> str:
    choices = data.get("choices", [])
    if not choices:
        return ""
    msg = choices[0].get("message", {})
    content = msg.get("content", "") or ""
    if not content:
        content = msg.get("reasoning_content", "") or ""
    return content


def _extract_ollama_content(data: dict) -> str:
    msg = data.get("message", {})
    content = msg.get("content", "") or ""
    if not content:
        content = msg.get("thinking", "") or ""
    return content


def _extract_anthropic_content(data: dict) -> str:
    text = ""
    thinking = ""
    for block in data.get("content", []):
        if block.get("type") == "text" and block.get("text"):
            text = block["text"]
        elif block.get("type") == "thinking" and block.get("thinking"):
            thinking = block["thinking"]
    return text or thinking


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_json(content: str, schema: dict | None = None) -> dict:
    """Parse JSON from content, validate against schema, return parsed dict."""
    assert content, f"Expected non-empty content, got: {content!r}"
    clean = _strip_markdown_fences(content)
    parsed = json.loads(clean)
    assert isinstance(parsed, dict), f"Expected dict, got {type(parsed)}: {clean[:200]}"
    if schema is not None:
        jsonschema.validate(instance=parsed, schema=schema)
    return parsed


# ===========================================================================
# Test classes
# ===========================================================================


@integration
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("stream", [False, True], ids=["sync", "stream"])
class TestOpenAIStructuredOutput:
    """Structured output via /v1/chat/completions."""

    def test_json_object(self, base_url, api_key, model, stream):
        """response_format: {"type": "json_object"} returns valid JSON dict."""
        content = openai_chat(
            base_url, api_key, model,
            messages=[{"role": "user", "content": SCHEMA_PROMPTS["simple"]}],
            stream=stream,
            response_format={"type": "json_object"},
        )
        _validate_json(content)

    def test_json_schema_simple(self, base_url, api_key, model, stream):
        """Simple {name, age} schema via json_schema response_format."""
        content = openai_chat(
            base_url, api_key, model,
            messages=[{"role": "user", "content": SCHEMA_PROMPTS["simple"]}],
            stream=stream,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "person", "schema": SIMPLE_SCHEMA},
            },
        )
        _validate_json(content, SIMPLE_SCHEMA)

    def test_json_schema_nested(self, base_url, api_key, model, stream):
        """Nested object schema via json_schema response_format."""
        content = openai_chat(
            base_url, api_key, model,
            messages=[{"role": "user", "content": SCHEMA_PROMPTS["nested"]}],
            stream=stream,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "nested_person", "schema": NESTED_SCHEMA},
            },
        )
        _validate_json(content, NESTED_SCHEMA)

    def test_json_schema_enum(self, base_url, api_key, model, stream):
        """Enum-constrained field via json_schema response_format."""
        content = openai_chat(
            base_url, api_key, model,
            messages=[{"role": "user", "content": SCHEMA_PROMPTS["enum"]}],
            stream=stream,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "color_value", "schema": ENUM_SCHEMA},
            },
        )
        _validate_json(content, ENUM_SCHEMA)

    def test_json_schema_array(self, base_url, api_key, model, stream):
        """Array of objects schema via json_schema response_format."""
        content = openai_chat(
            base_url, api_key, model,
            messages=[{"role": "user", "content": SCHEMA_PROMPTS["array"]}],
            stream=stream,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "item_list", "schema": ARRAY_SCHEMA},
            },
        )
        _validate_json(content, ARRAY_SCHEMA)

    def test_json_schema_complex(self, base_url, api_key, model, stream):
        """Complex countries schema via json_schema response_format."""
        content = openai_chat(
            base_url, api_key, model,
            messages=[{"role": "user", "content": SCHEMA_PROMPTS["complex"]}],
            stream=stream,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "countries", "schema": COMPLEX_SCHEMA},
            },
            max_tokens=1024,
        )
        _validate_json(content, COMPLEX_SCHEMA)


@integration
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("stream", [False, True], ids=["sync", "stream"])
class TestOllamaStructuredOutput:
    """Structured output via /api/chat."""

    def test_json_object(self, base_url, api_key, model, stream):
        """format: "json" returns valid JSON dict."""
        content = ollama_chat(
            base_url, api_key, model,
            messages=[{"role": "user", "content": SCHEMA_PROMPTS["simple"]}],
            stream=stream,
            format="json",
        )
        _validate_json(content)

    def test_json_schema_simple(self, base_url, api_key, model, stream):
        """Simple {name, age} schema via format:{schema}."""
        content = ollama_chat(
            base_url, api_key, model,
            messages=[{"role": "user", "content": SCHEMA_PROMPTS["simple"]}],
            stream=stream,
            format=SIMPLE_SCHEMA,
        )
        _validate_json(content, SIMPLE_SCHEMA)

    def test_json_schema_nested(self, base_url, api_key, model, stream):
        """Nested object schema via format:{schema}."""
        content = ollama_chat(
            base_url, api_key, model,
            messages=[{"role": "user", "content": SCHEMA_PROMPTS["nested"]}],
            stream=stream,
            format=NESTED_SCHEMA,
        )
        _validate_json(content, NESTED_SCHEMA)

    def test_json_schema_enum(self, base_url, api_key, model, stream):
        """Enum-constrained field via format:{schema}."""
        content = ollama_chat(
            base_url, api_key, model,
            messages=[{"role": "user", "content": SCHEMA_PROMPTS["enum"]}],
            stream=stream,
            format=ENUM_SCHEMA,
        )
        _validate_json(content, ENUM_SCHEMA)

    def test_json_schema_array(self, base_url, api_key, model, stream):
        """Array of objects schema via format:{schema}."""
        content = ollama_chat(
            base_url, api_key, model,
            messages=[{"role": "user", "content": SCHEMA_PROMPTS["array"]}],
            stream=stream,
            format=ARRAY_SCHEMA,
        )
        _validate_json(content, ARRAY_SCHEMA)

    def test_json_schema_complex(self, base_url, api_key, model, stream):
        """Complex countries schema via format:{schema}."""
        content = ollama_chat(
            base_url, api_key, model,
            messages=[{"role": "user", "content": SCHEMA_PROMPTS["complex"]}],
            stream=stream,
            format=COMPLEX_SCHEMA,
            max_tokens=1024,
        )
        _validate_json(content, COMPLEX_SCHEMA)


@integration
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("stream", [False, True], ids=["sync", "stream"])
class TestAnthropicStructuredOutput:
    """Structured output via /anthropic/v1/messages."""

    def test_json_object(self, base_url, api_key, model, stream):
        """JSON object output via system prompt instruction."""
        content = anthropic_chat(
            base_url, api_key, model,
            messages=[{"role": "user", "content": SCHEMA_PROMPTS["simple"]}],
            stream=stream,
        )
        _validate_json(content)

    def test_json_schema_simple(self, base_url, api_key, model, stream):
        """Simple schema via output_config json_schema."""
        content = anthropic_chat(
            base_url, api_key, model,
            messages=[{"role": "user", "content": SCHEMA_PROMPTS["simple"]}],
            stream=stream,
            output_config={
                "format": {
                    "type": "json_schema",
                    "name": "person",
                    "schema": SIMPLE_SCHEMA,
                }
            },
        )
        _validate_json(content, SIMPLE_SCHEMA)

    def test_json_schema_nested(self, base_url, api_key, model, stream):
        """Nested schema via output_config json_schema."""
        content = anthropic_chat(
            base_url, api_key, model,
            messages=[{"role": "user", "content": SCHEMA_PROMPTS["nested"]}],
            stream=stream,
            output_config={
                "format": {
                    "type": "json_schema",
                    "name": "nested_person",
                    "schema": NESTED_SCHEMA,
                }
            },
        )
        _validate_json(content, NESTED_SCHEMA)

    def test_json_schema_enum(self, base_url, api_key, model, stream):
        """Enum schema via output_config json_schema."""
        content = anthropic_chat(
            base_url, api_key, model,
            messages=[{"role": "user", "content": SCHEMA_PROMPTS["enum"]}],
            stream=stream,
            output_config={
                "format": {
                    "type": "json_schema",
                    "name": "color_value",
                    "schema": ENUM_SCHEMA,
                }
            },
        )
        _validate_json(content, ENUM_SCHEMA)

    def test_json_schema_array(self, base_url, api_key, model, stream):
        """Array schema via output_config json_schema."""
        content = anthropic_chat(
            base_url, api_key, model,
            messages=[{"role": "user", "content": SCHEMA_PROMPTS["array"]}],
            stream=stream,
            output_config={
                "format": {
                    "type": "json_schema",
                    "name": "item_list",
                    "schema": ARRAY_SCHEMA,
                }
            },
        )
        _validate_json(content, ARRAY_SCHEMA)

    def test_json_schema_complex(self, base_url, api_key, model, stream):
        """Complex countries schema via output_config json_schema."""
        content = anthropic_chat(
            base_url, api_key, model,
            messages=[{"role": "user", "content": SCHEMA_PROMPTS["complex"]}],
            stream=stream,
            output_config={
                "format": {
                    "type": "json_schema",
                    "name": "countries",
                    "schema": COMPLEX_SCHEMA,
                }
            },
            max_tokens=1024,
        )
        _validate_json(content, COMPLEX_SCHEMA)


# ---------------------------------------------------------------------------
# Cross-engine: verify schema compliance when routing through a
# different API surface than the model's native engine.
# ---------------------------------------------------------------------------

# Ollama-native models tested via OpenAI endpoint
OLLAMA_MODELS = ["phi4:14b", "qwen3-32k:32b", "qwen2.5-32k:7b"]
# vLLM-native models tested via Ollama endpoint
VLLM_MODELS = ["qwen/qwen3.5-400b", "openai/gpt-oss-20b"]


@integration
@pytest.mark.parametrize("stream", [False, True], ids=["sync", "stream"])
class TestCrossEngineStructuredOutput:
    """Cross-engine structured output: Ollama models via OpenAI endpoint
    and vLLM models via Ollama endpoint."""

    @pytest.mark.parametrize("model", OLLAMA_MODELS)
    def test_ollama_model_via_openai_json_schema(
        self, base_url, api_key, model, stream
    ):
        """Ollama model accessed through /v1/chat/completions with json_schema."""
        content = openai_chat(
            base_url, api_key, model,
            messages=[{"role": "user", "content": SCHEMA_PROMPTS["simple"]}],
            stream=stream,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "person", "schema": SIMPLE_SCHEMA},
            },
        )
        _validate_json(content, SIMPLE_SCHEMA)

    @pytest.mark.parametrize("model", OLLAMA_MODELS)
    def test_ollama_model_via_openai_json_object(
        self, base_url, api_key, model, stream
    ):
        """Ollama model accessed through /v1/chat/completions with json_object."""
        content = openai_chat(
            base_url, api_key, model,
            messages=[{"role": "user", "content": SCHEMA_PROMPTS["simple"]}],
            stream=stream,
            response_format={"type": "json_object"},
        )
        _validate_json(content)

    @pytest.mark.parametrize("model", VLLM_MODELS)
    def test_vllm_model_via_ollama_json_schema(
        self, base_url, api_key, model, stream
    ):
        """vLLM model accessed through /api/chat with format:{schema}."""
        content = ollama_chat(
            base_url, api_key, model,
            messages=[{"role": "user", "content": SCHEMA_PROMPTS["simple"]}],
            stream=stream,
            format=SIMPLE_SCHEMA,
        )
        _validate_json(content, SIMPLE_SCHEMA)

    @pytest.mark.parametrize("model", VLLM_MODELS)
    def test_vllm_model_via_ollama_json_object(
        self, base_url, api_key, model, stream
    ):
        """vLLM model accessed through /api/chat with format:"json"."""
        content = ollama_chat(
            base_url, api_key, model,
            messages=[{"role": "user", "content": SCHEMA_PROMPTS["simple"]}],
            stream=stream,
            format="json",
        )
        _validate_json(content)
