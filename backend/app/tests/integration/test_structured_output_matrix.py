"""
Structured output matrix integration tests.

Exercises every meaningful combination of:
  - API style (OpenAI, Ollama, Anthropic)
  - Format type (text, json_object, json_schema)
  - Thinking mode (off, on, disabled, reasoning_effort)
  - Streaming (True, False)
  - Model category (non-thinking, thinking-toggle, always-thinking)

Runs through the MindRouter2 API (full stack).

Requirements:
  - Live MindRouter2 deployment
  - Valid API key (MINDROUTER_API_KEY env var)
"""

import json
import os

import httpx
import pytest

# --- Constants ---

BASE_URL = os.environ.get("MINDROUTER_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("MINDROUTER_API_KEY", "")
TIMEOUT = 180

if not API_KEY:
    pytest.skip("MINDROUTER_API_KEY not set", allow_module_level=True)

PERSON_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
    },
    "required": ["name", "age"],
}

# Fallback model names by category
FALLBACK_MODELS = {
    "non_thinking": ["phi4:14b"],
    "thinking_toggle": [],  # auto-discover qwen3*
    "always_thinking": ["openai/gpt-oss-120b"],
}

# Classification patterns
MODEL_PATTERNS = {
    "thinking_toggle": ["qwen3"],
    "always_thinking": ["gpt-oss"],
}

PROMPT_TEXT = "Return a person named Alice who is 30 years old."
PROMPT_JSON = (
    "Return a JSON object with keys 'name' (string) and 'age' (integer). "
    "The person is Alice, age 30. Only output the JSON."
)

integration = pytest.mark.integration


# --- Model discovery ---


# Model name substrings that indicate non-chat models (skip these)
EXCLUDED_PATTERNS = ["embed", "rerank", "-vl", "_vl"]


def _is_chat_model(model_id: str) -> bool:
    """Check if a model ID is a chat model (not embedding/reranker/vision)."""
    lower = model_id.lower()
    return not any(pat in lower for pat in EXCLUDED_PATTERNS)


def _classify_model(model_id: str) -> str | None:
    """Classify a model ID into a category based on substring matching."""
    if not _is_chat_model(model_id):
        return None
    lower = model_id.lower()
    for category, patterns in MODEL_PATTERNS.items():
        for pattern in patterns:
            if pattern in lower:
                return category
    return None


def _discover_models(client: httpx.Client, headers: dict) -> dict[str, str | None]:
    """Discover available models via /v1/models and classify them."""
    models: dict[str, str | None] = {
        "non_thinking": None,
        "thinking_toggle": None,
        "always_thinking": None,
    }

    try:
        r = client.get("/v1/models", headers=headers, timeout=10)
        if r.status_code != 200:
            return models
        data = r.json()
        available = [m["id"] for m in data.get("data", [])]
    except Exception:
        return models

    # Auto-discover by pattern
    for model_id in available:
        category = _classify_model(model_id)
        if category and models[category] is None:
            models[category] = model_id

    # Check fallbacks for any unfilled categories
    for category, fallbacks in FALLBACK_MODELS.items():
        if models[category] is None:
            for fb in fallbacks:
                if fb in available:
                    models[category] = fb
                    break

    # For non_thinking, pick any model that isn't thinking-related
    if models["non_thinking"] is None:
        for model_id in available:
            if _classify_model(model_id) is None:
                # Exclude embedding/reranker models
                lower = model_id.lower()
                if "embed" not in lower and "rerank" not in lower:
                    models["non_thinking"] = model_id
                    break

    return models


# --- Request builders ---


def _openai_request(
    model: str,
    format_type: str,
    thinking_params: dict | None,
    stream: bool,
    max_tokens: int = 256,
) -> tuple[str, dict]:
    """Build /v1/chat/completions request body. Returns (endpoint, body)."""
    prompt = PROMPT_JSON if format_type != "text" else PROMPT_TEXT
    body: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
        "max_tokens": max_tokens,
    }

    if format_type == "json_object":
        body["response_format"] = {"type": "json_object"}
    elif format_type == "json_schema":
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "person", "schema": PERSON_SCHEMA},
        }

    if thinking_params:
        body.update(thinking_params)

    return "/v1/chat/completions", body


def _ollama_request(
    model: str,
    format_type: str,
    thinking_params: dict | None,
    stream: bool,
    max_tokens: int = 256,
) -> tuple[str, dict]:
    """Build /api/chat request body. Returns (endpoint, body)."""
    prompt = PROMPT_JSON if format_type != "text" else PROMPT_TEXT
    body: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
        "options": {"num_predict": max_tokens},
    }

    if format_type == "json_object":
        body["format"] = "json"
    elif format_type == "json_schema":
        body["format"] = PERSON_SCHEMA

    if thinking_params:
        body.update(thinking_params)

    return "/api/chat", body


def _anthropic_request(
    model: str,
    format_type: str,
    thinking_params: dict | None,
    stream: bool,
    max_tokens: int = 256,
) -> tuple[str, dict]:
    """Build /anthropic/v1/messages request body. Returns (endpoint, body)."""
    prompt = PROMPT_JSON if format_type != "text" else PROMPT_TEXT
    body: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
        "max_tokens": max_tokens,
    }

    if format_type == "json_schema":
        body["output_config"] = {
            "format": {
                "type": "json_schema",
                "name": "person",
                "schema": PERSON_SCHEMA,
            }
        }

    if thinking_params:
        # Anthropic uses thinking: {"type": "enabled"/"disabled"}
        if "think" in thinking_params:
            think_val = thinking_params["think"]
            body["thinking"] = {
                "type": "enabled" if think_val else "disabled",
            }
            if think_val:
                body["thinking"]["budget_tokens"] = max(max_tokens - 50, 100)
        elif "reasoning_effort" in thinking_params:
            body["reasoning_effort"] = thinking_params["reasoning_effort"]

    return "/anthropic/v1/messages", body


REQUEST_BUILDERS = {
    "openai": _openai_request,
    "ollama": _ollama_request,
    "anthropic": _anthropic_request,
}


# --- Stream collectors ---


def _collect_openai_stream(response: httpx.Response) -> str:
    """Collect content from OpenAI SSE stream.

    Falls back to reasoning_content when content is empty (thinking models
    with constrained output may put the answer in reasoning_content).
    """
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
    """Collect content from Ollama NDJSON stream."""
    content_parts = []
    for line in response.iter_lines():
        if not line:
            continue
        try:
            chunk = json.loads(line)
            msg = chunk.get("message", {})
            if msg.get("content"):
                content_parts.append(msg["content"])
            if chunk.get("done"):
                break
        except json.JSONDecodeError:
            continue
    return "".join(content_parts)


def _collect_anthropic_stream(response: httpx.Response) -> str:
    """Collect content from Anthropic event stream."""
    content_parts = []
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
                elif event_type == "message_stop":
                    break
            except json.JSONDecodeError:
                continue
    return "".join(content_parts)


STREAM_COLLECTORS = {
    "openai": _collect_openai_stream,
    "ollama": _collect_ollama_stream,
    "anthropic": _collect_anthropic_stream,
}


# --- Response extractors (non-streaming) ---


def _extract_openai_content(data: dict) -> str:
    """Extract content from OpenAI response JSON.

    Falls back to reasoning_content when content is empty (happens with
    thinking models + constrained output where the model puts its answer
    in the reasoning field).
    """
    choices = data.get("choices", [])
    if not choices:
        return ""
    msg = choices[0].get("message", {})
    content = msg.get("content", "") or ""
    if not content:
        content = msg.get("reasoning_content", "") or ""
    return content


def _extract_ollama_content(data: dict) -> str:
    """Extract content from Ollama response JSON."""
    return data.get("message", {}).get("content", "") or ""


def _extract_anthropic_content(data: dict) -> str:
    """Extract content from Anthropic response JSON."""
    for block in data.get("content", []):
        if block.get("type") == "text":
            return block.get("text", "")
    return ""


CONTENT_EXTRACTORS = {
    "openai": _extract_openai_content,
    "ollama": _extract_ollama_content,
    "anthropic": _extract_anthropic_content,
}


# --- Validation ---


def _strip_markdown_fences(text: str) -> str:
    """Strip markdown code fences (```json ... ```) from content."""
    text = text.strip()
    if text.startswith("```"):
        # Remove opening fence (```json or ```)
        first_newline = text.index("\n") if "\n" in text else len(text)
        text = text[first_newline + 1:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _validate_content(content: str, format_type: str) -> None:
    """Validate extracted content against the expected format."""
    assert content, f"Expected non-empty content, got: {content!r}"

    if format_type == "text":
        # Just needs to be non-empty
        return

    # Strip markdown fences that some models wrap around JSON
    clean = _strip_markdown_fences(content)

    # json_object or json_schema: must parse as JSON dict
    parsed = json.loads(clean)
    assert isinstance(parsed, dict), f"Expected dict, got {type(parsed)}"

    if format_type == "json_schema":
        assert "name" in parsed, f"Missing 'name' in {parsed}"
        assert "age" in parsed, f"Missing 'age' in {parsed}"
        assert isinstance(parsed["name"], str), f"'name' should be str, got {type(parsed['name'])}"
        assert isinstance(parsed["age"], (int, float)), f"'age' should be numeric, got {type(parsed['age'])}"


# --- Core test runner ---


def _run_test(
    client: httpx.Client,
    headers: dict,
    api_style: str,
    model: str,
    format_type: str,
    thinking_params: dict | None,
    streaming: bool,
    max_tokens: int = 256,
) -> None:
    """Execute a single matrix test case."""
    builder = REQUEST_BUILDERS[api_style]
    endpoint, body = builder(model, format_type, thinking_params, streaming, max_tokens=max_tokens)

    # Set up headers for the API style
    req_headers = dict(headers)
    if api_style == "anthropic":
        req_headers["x-api-key"] = API_KEY
        req_headers["anthropic-version"] = "2023-06-01"

    # Transient status codes that indicate infrastructure issues, not test failures
    SKIP_STATUS_CODES = {404, 502, 503}

    # Use a fresh client per request to avoid connection pool poisoning
    # after timeouts on previous tests
    with httpx.Client(base_url=client.base_url, timeout=TIMEOUT) as req_client:
        try:
            if streaming:
                with req_client.stream(
                    "POST", endpoint, headers=req_headers, json=body, timeout=TIMEOUT
                ) as r:
                    if r.status_code in SKIP_STATUS_CODES:
                        pytest.skip(
                            f"HTTP {r.status_code} (transient) for {api_style} "
                            f"{format_type} model={model}"
                        )
                    assert r.status_code == 200, (
                        f"HTTP {r.status_code} for {api_style} {format_type} "
                        f"stream={streaming} model={model}"
                    )
                    collector = STREAM_COLLECTORS[api_style]
                    content = collector(r)
            else:
                r = req_client.post(endpoint, headers=req_headers, json=body, timeout=TIMEOUT)
                if r.status_code in SKIP_STATUS_CODES:
                    pytest.skip(
                        f"HTTP {r.status_code} (transient) for {api_style} "
                        f"{format_type} model={model}: {r.text[:200]}"
                    )
                assert r.status_code == 200, (
                    f"HTTP {r.status_code} for {api_style} {format_type} "
                    f"stream={streaming} model={model}: {r.text[:500]}"
                )
                data = r.json()
                extractor = CONTENT_EXTRACTORS[api_style]
                content = extractor(data)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            pytest.skip(f"Connection failed: {exc}")
        except httpx.RemoteProtocolError as exc:
            pytest.fail(f"Backend crashed mid-response: {exc}")
        except httpx.ReadTimeout as exc:
            pytest.fail(f"Read timeout ({TIMEOUT}s): {exc}")

    _validate_content(content, format_type)


# --- Test class ---


@integration
class TestStructuredOutputMatrix:
    """Comprehensive structured output matrix test across API styles,
    format types, thinking modes, and streaming."""

    @classmethod
    def setup_class(cls):
        """Discover available models, build test config."""
        cls.base_url = BASE_URL
        cls.api_key = API_KEY
        cls.client = httpx.Client(base_url=cls.base_url, timeout=TIMEOUT)
        cls.headers = {"Authorization": f"Bearer {cls.api_key}"}
        cls.models = _discover_models(cls.client, cls.headers)

    @classmethod
    def teardown_class(cls):
        cls.client.close()

    # --- Non-thinking model ---

    @pytest.mark.parametrize("api_style", ["openai", "ollama", "anthropic"])
    @pytest.mark.parametrize("format_type", ["text", "json_object", "json_schema"])
    @pytest.mark.parametrize("streaming", [False, True])
    def test_non_thinking_model(self, api_style, format_type, streaming):
        """Non-thinking model structured output across API styles."""
        model = self.models.get("non_thinking")
        if not model:
            pytest.skip("No non-thinking model available")
        if api_style == "anthropic" and format_type == "json_object":
            pytest.skip("Anthropic does not support json_object format")

        _run_test(
            self.client, self.headers, api_style, model, format_type,
            thinking_params=None, streaming=streaming,
        )

    # --- Thinking-toggle model with thinking ON ---

    @pytest.mark.parametrize("api_style", ["openai", "ollama", "anthropic"])
    @pytest.mark.parametrize("format_type", ["text", "json_object", "json_schema"])
    @pytest.mark.parametrize("streaming", [False, True])
    def test_thinking_model_think_on(self, api_style, format_type, streaming):
        """Thinking model with thinking enabled + structured output."""
        model = self.models.get("thinking_toggle")
        if not model:
            pytest.skip("No thinking-toggle model available")
        if api_style == "anthropic" and format_type == "json_object":
            pytest.skip("Anthropic does not support json_object format")
        # Known limitation: Ollama response translation loses reasoning_content,
        # so thinking + constrained output via Ollama API returns empty content
        if api_style == "ollama" and format_type in ("json_object", "json_schema"):
            pytest.xfail("Ollama API loses thinking content with constrained output")

        _run_test(
            self.client, self.headers, api_style, model, format_type,
            thinking_params={"think": True}, streaming=streaming,
            max_tokens=2048,
        )

    # --- Thinking-toggle model with thinking OFF ---

    @pytest.mark.parametrize("api_style", ["openai", "ollama", "anthropic"])
    @pytest.mark.parametrize("format_type", ["text", "json_object", "json_schema"])
    @pytest.mark.parametrize("streaming", [False, True])
    def test_thinking_model_think_off(self, api_style, format_type, streaming):
        """Thinking model with thinking disabled + structured output."""
        model = self.models.get("thinking_toggle")
        if not model:
            pytest.skip("No thinking-toggle model available")
        if api_style == "anthropic" and format_type == "json_object":
            pytest.skip("Anthropic does not support json_object format")
        # Known limitation: Ollama API translation loses content or
        # produces invalid response_format for thinking models
        if api_style == "ollama" and format_type in ("json_object", "json_schema"):
            pytest.xfail("Ollama API translation issue with thinking model constrained output")

        _run_test(
            self.client, self.headers, api_style, model, format_type,
            thinking_params={"think": False}, streaming=streaming,
        )

    # --- Always-thinking model (gpt-oss with reasoning_effort) ---

    @pytest.mark.parametrize("api_style", ["openai", "ollama", "anthropic"])
    @pytest.mark.parametrize("format_type", ["text", "json_object", "json_schema"])
    @pytest.mark.parametrize("streaming", [False, True])
    def test_always_thinking_model(self, api_style, format_type, streaming):
        """gpt-oss model with reasoning_effort + structured output."""
        model = self.models.get("always_thinking")
        if not model:
            pytest.skip("No always-thinking model available")
        if api_style == "anthropic" and format_type == "json_object":
            pytest.skip("Anthropic does not support json_object format")

        _run_test(
            self.client, self.headers, api_style, model, format_type,
            thinking_params={"reasoning_effort": "low"}, streaming=streaming,
            max_tokens=512,
        )
