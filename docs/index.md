# MindRouter Documentation

MindRouter is a production-ready **LLM inference load balancer and translation layer** that fronts a heterogeneous cluster of **Ollama** and **vLLM** inference backends. It provides a unified OpenAI-compatible API surface with native Ollama compatibility, fair-share scheduling, per-user quotas, full audit logging, and real-time GPU telemetry.

**Developed by** Luke Sheneman, Research Computing and Data Services (RCDS), Institute for Interdisciplinary Data Sciences (IIDS), University of Idaho.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Getting Started](#getting-started)
4. [API Reference](#api-reference)
5. [Web Dashboard](#web-dashboard)
6. [Users, Groups & Quotas](#users-groups-quotas)
7. [Backend Management](#backend-management)
8. [Scheduling & Fair Share](#scheduling-fair-share)
9. [Translation Layer](#translation-layer)
10. [Telemetry & Monitoring](#telemetry-monitoring)
11. [Chat System](#chat-system)
12. [Voice API](#voice-api)
13. [Blog System](#blog-system)
14. [Configuration Reference](#configuration-reference)
15. [Implementation Notes](#implementation-notes)
16. [Deployment](#deployment)
17. [Testing](#testing)

---

## Overview

MindRouter sits between API consumers and GPU inference servers, providing:

- **Unified API Gateway** -- OpenAI-compatible `/v1/*`, Ollama-compatible `/api/*`, and Anthropic-compatible `/anthropic/v1/*` endpoints, all backed by the same pool of inference servers.
- **Cross-Engine Routing** -- A request arriving as OpenAI format can be served by an Ollama backend (and vice versa). The translation layer handles all protocol conversion transparently.
- **Fair-Share Scheduling** -- Weighted Deficit Round Robin (WDRR) ensures equitable GPU access across users with different roles and priorities.
- **Multi-Modal Support** -- Text chat, text completion, embeddings, vision-language models, structured JSON outputs, and tool calling (function calling).
- **Per-User Quotas** -- Token budgets, requests-per-minute limits, and concurrent request caps, all configurable by group.
- **Full Audit Logging** -- Every prompt, response, and token count is recorded for compliance and review.
- **Real-Time GPU Telemetry** -- Per-GPU utilization, memory, temperature, and power metrics via lightweight sidecar agents.
- **Web Dashboards** -- Public status page, user self-service dashboard, admin control panel, and built-in chat interface.

### Who It's For

- **Research computing centers** managing shared GPU clusters for multiple user groups
- **Universities** providing LLM access to students, staff, and faculty with differentiated quotas
- **Organizations** needing a unified API gateway across mixed Ollama/vLLM infrastructure

---

## Architecture

MindRouter follows a layered architecture:

```
Client Request (OpenAI, Ollama, or Anthropic format)
        │
        ▼
┌─────────────────────────────┐
│     API Gateway Layer       │  ← /v1/*, /api/*, /anthropic/*, /api/admin/*
├─────────────────────────────┤
│  Authentication & Quotas    │  ← API key verification, rate limiting
├─────────────────────────────┤
│    Translation Layer        │  ← OpenAI/Ollama/Anthropic ↔ Canonical ↔ Ollama/vLLM
├─────────────────────────────┤
│   Fair-Share Scheduler      │  ← WDRR with per-user deficit counters
├─────────────────────────────┤
│    Backend Registry         │  ← Health monitoring, model tracking
└─────────────────────────────┘
        │
        ▼
┌───────┴───────┬─────────────┐
│  GPU Node 1   │  GPU Node 2 │  ...
│  ┌─────────┐  │  ┌────────┐ │
│  │ Sidecar │  │  │Sidecar │ │  ← Per-node GPU metrics agent
│  ├─────────┤  │  ├────────┤ │
│  │ Ollama  │  │  │  vLLM  │ │  ← Inference engines
│  └─────────┘  │  └────────┘ │
└───────────────┴─────────────┘
```

**Key concepts:**

- A **Node** is a physical GPU server running a sidecar agent.
- A **Backend** is an inference endpoint (Ollama or vLLM instance) running on a node. Multiple backends can share a node, each assigned specific GPUs via `gpu_indices`.

For a deep dive into component interactions, data flow, and design decisions, see **[architecture.md](architecture.md)**.

---

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for local development)

### Quickstart with Docker Compose

```bash
# 1. Clone and configure
git clone <repository-url>
cd mindrouter
cp .env.example .env
nano .env  # Set DATABASE_URL, SECRET_KEY, etc.

# 2. Start all services
docker compose up --build

# 3. Run database migrations
docker compose exec app alembic upgrade head

# 4. Seed development data (creates users, quotas, API keys)
docker compose exec app python scripts/seed_dev_data.py
```

### Default Development Credentials

After running the seed script:

| Username | Password | Role | Scheduler Weight |
|----------|----------|------|-----------------|
| `admin` | `admin123` | admin | 10 |
| `faculty1` | `faculty123` | faculty | 3 |
| `staff1` | `staff123` | staff | 2 |
| `student1` | `student123` | student | 1 |

### Accessing the Application

| URL | Description |
|-----|-------------|
| `http://localhost:8000/` | Public status page |
| `http://localhost:8000/dashboard` | User dashboard (login required) |
| `http://localhost:8000/admin` | Admin dashboard (admin role required) |
| `http://localhost:8000/chat` | Chat interface (login required) |
| `http://localhost:8000/docs` | Interactive API docs (Swagger UI) |
| `http://localhost:8000/redoc` | API reference (ReDoc) |

---

## API Reference

### Interactive API Documentation

MindRouter includes built-in interactive API documentation powered by FastAPI:

- **Swagger UI** at [`/docs`](/docs) -- Interactive API explorer where you can try endpoints directly from your browser. Supports authentication via the "Authorize" button (enter your API key as a Bearer token).
- **ReDoc** at [`/redoc`](/redoc) -- Clean, readable API reference with request/response schemas and examples.

Both are auto-generated from the application's route definitions and Pydantic models, so they always reflect the current API surface.

### Authentication

All inference and admin endpoints require authentication. MindRouter supports two methods:

**API Key (Bearer Token):**
```bash
curl -H "Authorization: Bearer mr2_your-api-key" http://localhost:8000/v1/models
```

**API Key (Header):**
```bash
curl -H "X-API-Key: mr2_your-api-key" http://localhost:8000/v1/models
```

**Session Cookie** (dashboard/admin AJAX only):
Browser-based dashboard calls authenticate via the `mindrouter_session` cookie set at login. This is used internally by the web UI and is not intended for programmatic access.

**Authentication error messages (401):**

| Detail message | Cause |
|----------------|-------|
| "Missing API key. Provide via Authorization header or X-API-Key header." | No API key provided in the request |
| "Invalid API key" | API key not found in the database |
| "API key is {status}" | API key has been revoked |
| "API key has expired" | API key's `expires_at` timestamp has passed |
| "User account is inactive" | The user associated with the key is disabled |

### Request IDs

Every API response includes a unique request ID for tracing and audit correlation:

- **Auto-generated** in the format `chatcmpl-*`, `cmpl-*`, `emb-*`, or `msg-*` based on the endpoint type.
- **Client-provided** -- Clients can supply their own ID via the `X-Request-ID` header, which MindRouter will use instead of generating one.
- **Returned** in both the response body (`id` field) and response headers for easy correlation.
- **Audit trail** -- The request ID links the audit log entry to the API response for end-to-end traceability.

### Error Responses

Error response format varies by API style:

**OpenAI endpoints** (`/v1/*`):
```json
{"error": {"message": "...", "type": "invalid_request_error", "code": "model_not_found"}}
```

**Ollama endpoints** (`/api/*`):
```json
{"detail": "model 'xxx' not found"}
```

**Anthropic endpoints** (`/anthropic/v1/*`):
```json
{"type": "error", "error": {"type": "not_found_error", "message": "..."}}
```

Common HTTP status codes:

| Code | Meaning |
|------|---------|
| 400 | Invalid request body or parameters |
| 400 | "Model '{model}' does not support multimodal/image input" -- Sent images to a non-vision model |
| 400 | "Model '{model}' does not support structured output" -- Requested JSON schema on unsupported model |
| 401 | Missing or invalid API key |
| 403 | Insufficient permissions (e.g., non-admin accessing admin endpoint) |
| 404 | Resource not found |
| 409 | Conflict (duplicate name, URL, etc.) |
| 413 | Request payload exceeds `MAX_REQUEST_SIZE` (default 50 MB) |
| 422 | Request validation failed (malformed JSON, wrong types, missing required fields). Returned by FastAPI's built-in request validation. |
| 429 | Rate limit exceeded. MindRouter does not include a `Retry-After` header on 429 responses. Clients should implement exponential backoff. |
| 500 | Internal server error |
| 503 | "No suitable backend: {reason}" -- Model doesn't support required capability (vision, structured output) |
| 503 | "No backend capacity available (waited Ns)" -- All backends at max concurrent; timed out waiting |
| 503 | "No healthy backends available" -- All backends unhealthy or circuit-broken |

> **Backend pass-through:** Backend 4xx errors (e.g., invalid prompt format) are forwarded directly to the client and are not retried.

> **Model names are exact match only.** MindRouter does not support prefix matching, aliases, or fuzzy matching. The `model` field in requests must exactly match a model name as shown in `/v1/models` or `/api/tags`.

### OpenAI-Compatible Endpoints

These endpoints accept and return data in the OpenAI API format. Any OpenAI-compatible client or SDK can be pointed at MindRouter by changing the base URL.

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/v1/chat/completions` | API Key | Chat completions (streaming and non-streaming) |
| POST | `/v1/completions` | API Key | Text completions (legacy, internally converts to chat format) |
| POST | `/v1/embeddings` | API Key | Generate embeddings |
| POST | `/v1/rerank` | API Key | Rerank documents against a query |
| POST | `/v1/score` | API Key | Score similarity between text pairs |
| GET | `/v1/models` | API Key | List available models |

#### Chat Completions

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer mr2_your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"}
    ],
    "temperature": 0.7,
    "max_tokens": 500,
    "stream": false
  }'
```

**Response:**
```json
{
  "id": "chatcmpl-abc123...",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "llama3.2",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 10,
    "total_tokens": 35
  }
}
```

**Streaming** -- Set `"stream": true` to receive Server-Sent Events (SSE):

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer mr2_your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.2", "messages": [{"role": "user", "content": "Hi"}], "stream": true}'
```

**Thinking/Reasoning Mode:**

MindRouter supports multiple formats for controlling thinking/reasoning on models that support it (qwen3.5, qwen3, gpt-oss):

```json
// gpt-oss: control reasoning depth via reasoning_effort
{
  "model": "openai/gpt-oss-120b",
  "messages": [{"role": "user", "content": "Solve this step by step"}],
  "reasoning_effort": "high",
  "max_completion_tokens": 16384
}

// Qwen-style: toggle thinking on/off
{
  "model": "qwen/qwen3.5-400b",
  "messages": [{"role": "user", "content": "Explain quantum computing"}],
  "chat_template_kwargs": {"enable_thinking": true},
  "max_completion_tokens": 16384
}

// Also accepted: thinking object (OpenAI/Anthropic style)
{
  "model": "qwen/qwen3.5-400b",
  "thinking": {"type": "disabled"},
  "messages": [...]
}
```

When thinking is enabled, the response includes a `reasoning_content` field alongside `content`:
```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "The answer is 42.",
      "reasoning_content": "Let me work through this step by step..."
    }
  }]
}
```

> **Important:** Thinking models can consume large numbers of output tokens for reasoning. Use `max_completion_tokens` (or `max_tokens`) to set an adequate budget -- 16384 is recommended for qwen3.5-400b with thinking enabled. Without a limit, the model may use up to the full context window (131K tokens) on reasoning.

**Output Token Limits:**

MindRouter accepts both `max_completion_tokens` (preferred, current OpenAI standard) and `max_tokens` (legacy). If both are provided, `max_completion_tokens` takes priority.

**Structured Output (JSON Mode):**
```json
{
  "model": "llama3.2",
  "messages": [{"role": "user", "content": "List 3 colors as JSON"}],
  "response_format": {"type": "json_object"}
}
```

**Structured Output (JSON Schema):**
```json
{
  "model": "llama3.2",
  "messages": [{"role": "user", "content": "List 3 colors"}],
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "colors",
      "schema": {
        "type": "object",
        "properties": {
          "colors": {"type": "array", "items": {"type": "string"}}
        }
      }
    }
  }
}
```

**Vision (Multimodal):**
```json
{
  "model": "llava",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
      ]
    }
  ]
}
```

**Tool Calling (Function Calling):**
```json
{
  "model": "llama3.2",
  "messages": [{"role": "user", "content": "What's the weather in Seattle?"}],
  "tools": [{
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get the current weather",
      "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
      }
    }
  }],
  "tool_choice": "auto"
}
```

When the model decides to call a tool, the response includes `tool_calls` with `finish_reason: "tool_calls"`. Submit the tool result back as a `role: "tool"` message with the matching `tool_call_id`.

#### Tool Calling Details

MindRouter supports OpenAI-style tool/function calling across all API formats (OpenAI, Ollama, and Anthropic inbound):

- **Tool definitions** use `type: "function"` with a `function` object containing `name`, `description`, and `parameters` (JSON Schema).
- **`tool_choice`** controls tool selection: `"auto"` (model decides), `"none"` (no tools), or `{"type": "function", "function": {"name": "..."}}` (force a specific tool).
- **Tool results** are submitted as follow-up messages with `role: "tool"`, including the `tool_call_id` from the model's response.
- **Streaming** -- tool call data arrives as `tool_calls` deltas within SSE chunks, with each delta containing the function name and argument fragments.
- **Finish reason** is set to `"tool_calls"` when the model invokes one or more tools.
- **Backend requirement** -- the backend must support tool calling. For vLLM, this requires the `--enable-auto-tool-choice` and `--tool-call-parser <parser>` flags at serve time.

> **OpenAI spec compliance:** Chat completion responses always include `message.content` in each choice, even when the value is `null` (e.g., when `finish_reason` is `"tool_calls"`).

#### Completions Parameters

The `/v1/completions` endpoint supports additional parameters beyond the standard chat completions set:

- **`suffix`** -- Text appended after the completion.
- **`echo`** -- Echo the prompt back in the response.
- **`n`** -- Number of completions to generate (default 1).
- **`best_of`** -- Number of beam search candidates to consider.

> **Note:** Chat completions (`/v1/chat/completions`) also support `n` to generate multiple alternative responses (default 1).

#### Embeddings

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Authorization: Bearer mr2_your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "nomic-embed-text", "input": "Hello world"}'
```

Additional embedding parameters:

- **`encoding_format`** -- Response encoding (`"float"` or `"base64"`, default `"float"`).
- **`dimensions`** -- Desired output dimensionality (model-dependent).

#### List Models

```bash
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer mr2_your-api-key"
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "llama3.2",
      "object": "model",
      "created": 1700000000,
      "owned_by": "mindrouter",
      "capabilities": {"multimodal": false, "embeddings": false, "structured_output": true, "thinking": false},
      "backends": ["ollama-gpu1", "ollama-gpu2"],
      "context_length": 32768,
      "model_max_context": 131072,
      "parameter_count": "8.0B",
      "quantization": "Q4_K_M",
      "family": "llama"
    }
  ]
}
```

### Ollama-Compatible Endpoints

These endpoints accept and return data in Ollama's native format. Ollama clients can be pointed at MindRouter as a drop-in replacement.

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/api/chat` | API Key | Ollama chat (streaming by default) |
| POST | `/api/generate` | API Key | Ollama text generation |
| POST | `/api/embeddings` | API Key | Ollama embeddings |
| GET | `/api/tags` | API Key | List models (Ollama format) |

#### Ollama Chat

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Authorization: Bearer mr2_your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

> **Note:** Ollama defaults to `stream: true`. Set `"stream": false` explicitly for non-streaming responses.

**Thinking/Reasoning Mode (Ollama):**

For Ollama endpoints, use the `think` field at the top level:

```json
// Qwen-style: boolean toggle
{
  "model": "qwen3-32k:32b",
  "messages": [{"role": "user", "content": "Solve this step by step"}],
  "think": true,
  "stream": false
}

// gpt-oss: string effort level ("low", "medium", "high")
{
  "model": "gpt-oss-32k:120b",
  "messages": [{"role": "user", "content": "Explain quantum entanglement"}],
  "think": "high",
  "stream": false
}
```

When thinking is enabled, the response includes a `thinking` field in the message:
```json
{
  "message": {
    "role": "assistant",
    "content": "The answer is 42.",
    "thinking": "Let me reason through this..."
  }
}
```

> **Note:** For `/api/generate`, thinking content appears as a top-level `thinking` field alongside `response`.

> **Ollama `think` placement:** For Ollama backends, the `think` parameter is placed at the **top level** of the request payload, not inside the `options` dict. This matches Ollama's native API format.

#### Ollama Generate

```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Authorization: Bearer mr2_your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.2", "prompt": "Why is the sky blue?"}'
```

> **Ollama `/api/generate` system prompt:** When a `system` field is provided in an Ollama `/api/generate` request, it is prepended to the `prompt` field (separated by a blank line) before translation to canonical format. This differs from `/api/chat`, where system messages are preserved as separate message objects.

### Anthropic-Compatible Endpoint

This endpoint accepts and returns data in the Anthropic Messages API format. Anthropic SDK clients can be pointed at MindRouter by setting `base_url`.

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/anthropic/v1/messages` | API Key | Anthropic Messages API (streaming and non-streaming) |

#### Messages

```bash
curl -X POST http://localhost:8000/anthropic/v1/messages \
  -H "Authorization: Bearer mr2_your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2",
    "max_tokens": 500,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

**Response:**
```json
{
  "id": "msg_abc123...",
  "type": "message",
  "role": "assistant",
  "model": "llama3.2",
  "content": [
    {"type": "text", "text": "Hello! How can I help you today?"}
  ],
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 10,
    "output_tokens": 12
  }
}
```

**Streaming** -- Set `"stream": true` to receive Anthropic SSE events (`message_start`, `content_block_delta`, `message_stop`, etc.):

```bash
curl -X POST http://localhost:8000/anthropic/v1/messages \
  -H "Authorization: Bearer mr2_your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.2", "max_tokens": 500, "messages": [{"role": "user", "content": "Hi"}], "stream": true}'
```

**System Prompt:**
```json
{
  "model": "llama3.2",
  "max_tokens": 500,
  "system": "You are a helpful assistant.",
  "messages": [{"role": "user", "content": "Hello!"}]
}
```

**SDK Usage (Python):**
```python
import anthropic
client = anthropic.Anthropic(
    base_url="http://localhost:8000/anthropic",
    api_key="mr2_your-api-key",
)
message = client.messages.create(
    model="llama3.2",
    max_tokens=500,
    messages=[{"role": "user", "content": "Hello!"}],
)
```

**Supported features:**
- System prompts (string or content block array)
- Multimodal inputs (base64 and URL images)
- Tool calling -- `tools` with `input_schema`, `tool_choice` (`auto`/`any`/`tool`), `tool_use`/`tool_result` content blocks, streaming tool use with `input_json_delta`. Anthropic `tool_choice` values are mapped: `auto` to `auto`, `any` to `required`, `tool` (with name) to `{"type": "function", "function": {"name": "..."}}`.
- Thinking/reasoning mode (`thinking.type`: `enabled`, `adaptive`, `disabled`; set `budget_tokens` to control reasoning token allocation)
- Structured output via `output_config.format` with `type: "json_schema"`
- Parameters: `max_tokens` (required), `temperature`, `top_p`, `top_k`, `stop_sequences`, `stream`
- `metadata.user_id` mapping

> **Note:** This is inbound-only -- there are no Anthropic backends. Requests are translated to canonical format and routed to Ollama/vLLM backends like any other request.

### Voice API Endpoints

OpenAI-compatible text-to-speech and speech-to-text endpoints. These proxy to configured upstream TTS/STT services (e.g., Kokoro TTS, faster-whisper) and require API key authentication with quota tracking.

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/v1/audio/speech` | API Key | Text-to-speech (streaming audio response) |
| POST | `/v1/audio/transcriptions` | API Key | Speech-to-text (file upload) |

#### POST /v1/audio/speech

Converts text to speech audio. Returns a streaming audio response.

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kokoro",
    "input": "Hello, world!",
    "voice": "af_heart",
    "response_format": "mp3",
    "speed": 1.0
  }' \
  --output speech.mp3
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | `"kokoro"` | TTS model name |
| `input` | string | (required) | Text to synthesize |
| `voice` | string | `"af_heart"` | Voice identifier (see available voices in Voice API admin config) |
| `response_format` | string | `"mp3"` | Audio format: `mp3`, `wav`, `opus`, `flac` |
| `speed` | float | `1.0` | Speed multiplier (0.25 -- 4.0) |

Returns streaming audio with content type matching the requested format (e.g., `audio/mpeg` for mp3).

#### POST /v1/audio/transcriptions

Transcribes audio to text. Accepts multipart file upload.

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer your-api-key" \
  -F "file=@recording.mp3" \
  -F "model=whisper-large-v3-turbo" \
  -F "language=en"
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | file | (required) | Audio file to transcribe |
| `model` | string | (from config) | STT model name (defaults to admin-configured model) |
| `language` | string | (none) | ISO language code hint (e.g., `en`, `fr`) |

Response:
```json
{"text": "Hello, world!"}
```

#### STT Limitations & Long Audio

The transcription endpoint is designed for short-to-medium audio clips (up to ~10 minutes). Several constraints affect long-form audio:

| Constraint | Value | Impact |
|------------|-------|--------|
| Upload size limit | 50 MB (nginx `client_max_body_size`) | A 1-hour MP3 at 128 kbps is ~57 MB -- over the limit. WAV files are much larger. |
| Proxy timeout | 120 seconds (hardcoded in `voice_api.py`) | Whisper transcribing a long file can take 5--10+ minutes, causing a 502 timeout. |
| Memory buffering | Entire file read into RAM | Large uploads spike container memory since the file is fully buffered before forwarding. |
| No chunking | Single request per file | There is no server-side segmentation -- one file = one request to the upstream STT service. |
| Flat quota cost | Same cost regardless of duration | A 1-hour file costs the same 200 tokens as a 5-second clip. |

**Mitigation: client-side chunking.** Split long audio into segments before sending. This avoids all of the above limits and gives you per-segment error recovery.

**Python example -- split and transcribe a long file with pydub:**

```python
import httpx
from pydub import AudioSegment

API_KEY = "mr2_your-api-key"
BASE_URL = "https://mindrouter.example.com"
CHUNK_MINUTES = 5

audio = AudioSegment.from_file("lecture.mp3")
chunk_ms = CHUNK_MINUTES * 60 * 1000

transcript_parts = []
for i in range(0, len(audio), chunk_ms):
    chunk = audio[i : i + chunk_ms]
    buf = chunk.export(format="mp3")

    resp = httpx.post(
        f"{BASE_URL}/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        files={"file": (f"chunk_{i // chunk_ms}.mp3", buf, "audio/mpeg")},
        data={"model": "whisper-large-v3-turbo"},
        timeout=120.0,
    )
    resp.raise_for_status()
    transcript_parts.append(resp.json()["text"])

full_transcript = " ".join(transcript_parts)
print(full_transcript)
```

**Bash example -- split with ffmpeg and transcribe each chunk:**

```bash
# Split into 5-minute chunks
ffmpeg -i lecture.mp3 -f segment -segment_time 300 -c copy chunk_%03d.mp3

# Transcribe each chunk
for f in chunk_*.mp3; do
  curl -s -X POST https://mindrouter.example.com/v1/audio/transcriptions \
    -H "Authorization: Bearer mr2_your-api-key" \
    -F "file=@$f" \
    -F "model=whisper-large-v3-turbo" \
    | jq -r '.text'
done
```

> **Tip:** Each chunk is a separate API request, so each one deducts the configured STT quota cost. For a 1-hour file split into 12 chunks at the default 200 tokens/request, the total cost is 2,400 tokens.

#### Voice API Quota

Each Voice API request deducts a fixed token cost from the user's quota. The cost per request is configurable by admins in the Voice API Config page:

- **TTS**: default 100 tokens per request
- **STT**: default 200 tokens per request

These costs are stored in the database (`voice_api.tts_quota_tokens`, `voice_api.stt_quota_tokens`) and can be changed without redeploying.

### Health & Metrics Endpoints

These endpoints are unauthenticated and intended for monitoring infrastructure.

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/healthz` | None | Liveness probe (always 200 if app is running) |
| GET | `/readyz` | None | Readiness probe (checks DB + healthy backends) |
| GET | `/metrics` | None | Prometheus metrics (text format) |
| GET | `/status` | None | Cluster status summary (JSON) |
| GET | `/api/cluster/throughput` | None | Token throughput (last 10 seconds) |
| GET | `/api/cluster/total-tokens` | None | Total tokens ever served (cached 10s) |
| GET | `/api/cluster/trends` | None | Token and active-user trends over time (query param: `range=hour\|day\|week\|month\|year`) |

#### Example Responses

**GET /healthz** — Liveness probe:
```json
{"status": "alive", "timestamp": "2026-03-01T12:00:00+00:00"}
```

**GET /readyz** — Readiness probe:
```json
{
  "status": "ready",
  "checks": {"database": true, "backends": true},
  "timestamp": "2026-03-01T12:00:00+00:00"
}
```

**GET /status** — Cluster summary:
```json
{
  "service": "MindRouter",
  "version": "2.0.0",
  "timestamp": "2026-03-01T12:00:00+00:00",
  "backends": {"total": 6, "healthy": 5},
  "models": ["gpt-oss-120b", "llama3.2", "qwen3.5"],
  "queue": {"total": 3},
  "fair_share": {"total_users": 2},
  "active_users": 12
}
```

**GET /api/cluster/throughput** — Token throughput:
```json
{
  "tokens_per_second": 142.5,
  "requests_per_minute": 8,
  "active_requests": 3,
  "total_tokens_last_10s": 1425,
  "inflight_tokens": 200
}
```

**GET /api/cluster/total-tokens** — Total tokens served:
```json
{"total_tokens": 15234567}
```

**GET /api/cluster/trends** — Trends over time:
```json
{
  "tokens": [{"period": "2026-03-01T11:00:00Z", "total": 50000}, ...],
  "users": [{"period": "2026-03-01T11:00:00Z", "count": 5}, ...],
  "range": "day",
  "since": "2026-02-28T12:00:00Z",
  "now": "2026-03-01T12:00:00Z"
}
```

#### Prometheus Metrics

The `/metrics` endpoint exposes the following Prometheus metrics:

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `mindrouter_requests_total` | Counter | `endpoint`, `status` | Total requests processed |
| `mindrouter_request_latency_seconds` | Histogram | `endpoint` | Request latency |
| `mindrouter_queue_size` | Gauge | -- | Current scheduler queue depth |
| `mindrouter_active_backends` | Gauge | -- | Number of healthy backends |
| `mindrouter_tokens_total` | Counter | `type` (prompt/completion) | Total tokens processed |

### Admin API Endpoints

All admin endpoints require the `admin` role. They are mounted under `/api/admin/`.

#### Backend Management

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/admin/backends/register` | Register a new inference backend |
| GET | `/api/admin/backends` | List all backends |
| PATCH | `/api/admin/backends/{id}` | Update backend properties |
| POST | `/api/admin/backends/{id}/disable` | Disable a backend |
| POST | `/api/admin/backends/{id}/enable` | Enable a disabled backend |
| POST | `/api/admin/backends/{id}/refresh` | Force-refresh capabilities and models |
| POST | `/api/admin/backends/{id}/ollama/pull` | Pull a model on an Ollama backend |
| GET | `/api/admin/backends/{id}/ollama/pull/{job_id}` | Check pull job status |
| POST | `/api/admin/backends/{id}/ollama/delete` | Delete a model from an Ollama backend |

#### Node Management

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/admin/nodes/register` | Register a new GPU node |
| GET | `/api/admin/nodes` | List all nodes |
| PATCH | `/api/admin/nodes/{id}` | Update node properties |
| DELETE | `/api/admin/nodes/{id}` | Delete a node (fails if backends reference it) |
| POST | `/api/admin/nodes/{id}/refresh` | Force-refresh sidecar data |

#### User Management

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/admin/users` | List all users (filterable by group) |
| POST | `/api/admin/users` | Create a new user with group-based quota defaults |
| GET | `/api/admin/users/{id}` | Get user detail including quotas, API keys, and group |
| PATCH | `/api/admin/users/{id}` | Update user properties (group, quotas, etc.) |
| DELETE | `/api/admin/users/{id}` | Hard-delete a user and all associated data |
| POST | `/api/admin/users/{id}/api-keys` | Create an API key for a user |

#### Group Management

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/admin/groups` | List all groups with user counts |
| POST | `/api/admin/groups` | Create a new group |
| PATCH | `/api/admin/groups/{id}` | Update group defaults (token budget, RPM, etc.) |
| DELETE | `/api/admin/groups/{id}` | Delete a group (fails if users are assigned) |

#### API Key Management

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/admin/api-keys` | List all API keys with user info (filterable by status, searchable) |

#### Quota Management

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/admin/quota-requests` | List pending quota increase requests |
| POST | `/api/admin/quota-requests/{id}/review` | Approve or deny a quota request |

#### Queue & Audit

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/admin/queue` | Scheduler queue statistics |
| GET | `/api/admin/audit/search` | Search audit logs (filter by user, model, status, date, text) |
| GET | `/api/admin/audit/{id}` | Full audit detail including prompt and response content |

#### Conversations

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/admin/conversations/export` | Export conversations as CSV or JSON (filterable by user, date range, search) |

#### Telemetry

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/admin/telemetry/overview` | Cluster-wide telemetry (nodes, backends, GPUs) |
| GET | `/api/admin/telemetry/latest` | Lightweight polling endpoint for dashboard |
| GET | `/api/admin/telemetry/backends/{backend_id}/history` | Time-series telemetry for a backend |
| GET | `/api/admin/telemetry/gpus/{gpu_device_id}/history` | Time-series telemetry for a GPU device |
| GET | `/api/admin/telemetry/nodes/{node_id}/history` | Aggregated time-series for a node (all GPUs) |
| GET | `/api/admin/telemetry/export` | Export raw telemetry as JSON or CSV |

---

## Web Dashboard

MindRouter includes a full web dashboard built with Bootstrap 5. All pages extend a common base template with navigation and accessibility features (WCAG 2.1 Level AA).

### Public Pages

| Page | URL | Description |
|------|-----|-------------|
| Cluster Status | `/` | Shows healthy backend count, available models, queue size, and overall cluster status |
| Login | `/login` | Username/password authentication (and Azure AD SSO when configured) |
| Blog | `/blog` | Public blog with published posts |

The public landing page (`/`) includes a live token flow animation showing real-time cluster throughput, along with counters for healthy backends, available models, active users (24h), and total tokens served.

### User Dashboard

| Page | URL | Description |
|------|-----|-------------|
| Dashboard | `/dashboard` | Token usage progress bar, active API keys, quota usage history |
| Request Quota | `/dashboard/request-quota` | Submit a quota increase request with justification |
| Key Created | (after creation) | Displays the full API key once (copy-to-clipboard) |

The user dashboard includes the following features:

- **Dark Mode Toggle** -- The Preferences section includes a dark mode toggle. The preference is saved to browser localStorage and persists across sessions.
- **TTS Voice Preference** -- When TTS is enabled by an admin, a "TTS Voice" dropdown appears in the Preferences card. Users can pick their preferred voice for Read Aloud in chat, or leave it on "(System default)" to use the admin-configured default. Voice names are shown in friendly format (e.g. `af_heart` displays as "Heart", `am_michael` as "Michael"). The "(System default)" option shows which voice it maps to. The dropdown is filtered to only voices the admin has listed in Voice Config > Available Voices.
- **TTS Speed Preference** -- A playback speed slider (0.5--2.0) lets users set their preferred TTS speed. Changes auto-save with a brief confirmation. A "Reset to default" button clears the override to use the admin-configured speed.
- **Live Token Usage** -- Token usage statistics on the dashboard update in real-time via polling (every 1 second), providing live feedback without page refresh.
- **Lifetime vs Rolling Usage** -- The dashboard displays two token metrics: **Lifetime Token Usage** (all-time total tokens consumed) and **Current Period Usage** (tokens used in the current rolling budget period). These are distinct -- the lifetime counter never resets, while the period counter resets when the budget period rolls over.
- **Quota Details** -- Users can view their current quota limits (RPM limit and max concurrent requests) in the Quota Details card on the dashboard.
- **API Key Expiration Warnings** -- API keys nearing expiration (7 days or fewer remaining) display a yellow warning countdown. Expired keys show an "Expired" badge. The "Last Used" column shows when each key was last used for authentication.

### Admin Dashboard

The admin dashboard has a persistent sidebar with links to all admin pages:

| Page | URL | Description |
|------|-----|-------------|
| Overview | `/admin` | System metrics overview with pending request badges and health alerts |
| Backends | `/admin/backends` | Backend health, models, enable/disable/drain controls |
| Models | `/admin/models` | Model catalog with capability overrides, metadata editing, context length configuration (see below) |
| Nodes | `/admin/nodes` | GPU node management, sidecar status, take offline/bring online, force drain, active requests |
| GPU Metrics | `/admin/metrics` | Real-time GPU utilization, memory, temperature, power charts with time range controls |
| Users | `/admin/users` | User accounts, group assignment, quota management, masquerade |
| Groups | `/admin/groups` | Group management with quota defaults, scheduler weights, admin flag |
| API Keys | `/admin/api-keys` | All API keys across users, status filtering, search |
| Requests | `/admin/requests` | Pending API key and quota increase requests, approve/deny |
| Audit Log | `/admin/audit` | Inference request history with filtering and search |
| Conversations | `/admin/conversations` | Browse and search all user conversations, view messages, export |
| Chat Config | `/admin/chat-config` | Configure core models, default model, system prompt, max_tokens, temperature, thinking mode, voice TTS/STT settings |
| Voice API Config | `/admin/voice-config` | Configure TTS/STT backend connections, available voices, and API quota token costs |
| Blog | `/admin/blog` | Blog/CMS management -- create, edit, publish, delete posts |
| Settings | `/admin/settings` | Site-wide settings: timezone, enforce `num_ctx` override |

#### Admin Dashboard Actions

These are dashboard routes (not API endpoints) that require an admin session cookie:

| Method | Path | Description |
|--------|------|-------------|
| POST | `/admin/system/toggle-online` | Force the entire system offline or back online |
| POST | `/admin/backends/{id}/drain` | Start draining a backend (stop new requests, let in-flight finish) |
| POST | `/admin/nodes/{id}/take-offline` | Disable all backends on a node and mark it offline |
| POST | `/admin/nodes/{id}/bring-online` | Re-enable all backends on a node and mark it online |
| POST | `/admin/nodes/{id}/force-drain` | Force-cancel all active requests on a node's backends |
| GET | `/admin/nodes/{id}/active-requests` | Count of in-flight requests across all backends on a node |
| POST | `/admin/masquerade/{target_user_id}` | Start masquerading as a user (sets signed cookie, redirects to their dashboard) |
| POST | `/admin/masquerade/stop` | Stop masquerading and return to admin view |
| GET | `/admin/audit/export` | Export audit logs as CSV or JSON (filterable by user, model, status, date range) |

> **Export content option:** Both audit and conversation exports support an optional `include_content` checkbox. When enabled, exports include full prompt messages, request parameters, and response content. This is disabled by default; enabling it produces significantly larger export files.

### Model Metadata Editing

The model detail page (`/admin/models`) allows admins to override discovery-provided metadata for any model. Available override fields:

- **Context Length Override** -- Effective context window injected as `num_ctx` (overrides discovery value)
- **Model Max Context Override** -- Architectural maximum context (immutable from discovery, but overridable)
- **Embedding Dimension** -- Vector dimension for embedding models
- **Attention Heads**, **Layers/Depth**, **FFN Size** -- Architecture parameters
- **Model Format** -- Quantization or format label (e.g., `Q4_K_M`, `fp16`)
- **Parent Model** -- Base model identifier
- **Description** -- Admin-provided text description
- **Model URL** -- Link to the model card or documentation
- **Capabilities** -- Comma-separated list: `completion`, `vision`, `tools`, `thinking`, `embedding`

The **"Reset All Overrides"** button clears all metadata customizations, reverting the model to discovery-provided values.

### Re-pull All Ollama Models

The models page includes a **"Re-pull All Ollama Models"** button that triggers a bulk re-pull across all Ollama backends. Models are processed sequentially per node (respecting shared model storage folders). The UI provides per-backend progress tracking with success, failure, retry, and skip controls, along with an overall progress bar. Individual pulls can be cancelled or skipped.

### Health Alerts

The admin dashboard (`/admin`) displays a prominent warning banner when any backend is unhealthy/unknown or any node is offline/unknown. The alert includes counts and names of affected items with direct links to the backends or nodes management pages. Intentionally **disabled** backends are excluded from the alert -- only unexpected health issues are flagged.

### System Offline Toggle

Admins can force the entire MindRouter system offline or back online via `POST /admin/system/toggle-online`. When forced offline, the registry stops polling and marks all backends as unhealthy, causing all inference requests to be rejected. This is useful for planned maintenance windows.

### Masquerade

Admins can masquerade as any user to view the system from their perspective. Start masquerading via `POST /admin/masquerade/{target_user_id}` -- a signed cookie is set and the admin is redirected to the target user's dashboard. The admin sees the user's token usage, API keys, and conversations as if logged in as that user. Stop masquerading via `POST /admin/masquerade/stop` to return to the admin view.

### Chat Interface

| Page | URL | Description |
|------|-----|-------------|
| Chat | `/chat` | Full-featured chat UI with model selection, streaming, file upload, web search, vision support |

The chat interface supports:
- Collapsible conversation sidebar
- Model and backend selection
- Real-time streaming responses
- File upload via button or **drag-and-drop** anywhere in the chat window (images, PDFs, DOCX, XLSX, CSV, JSON, Markdown, etc.)
- Vision model support with automatic image handling
- **Web search toggle** -- when enabled, queries are sent to the Brave Search API and results are injected into the system message as context before the LLM generates its response (requires `BRAVE_SEARCH_API_KEY` configuration)
- Code syntax highlighting
- LaTeX rendering
- Message editing and deletion

---

## Users, Groups & Quotas

### Groups

MindRouter uses database-driven **groups** to control permissions, quotas, and scheduling priority. Each user belongs to exactly one group. Groups replace the earlier role-based system with more flexible, admin-configurable settings.

Each group has the following fields:

| Field | Description |
|-------|-------------|
| `name` | Unique identifier (e.g., `student`, `staff`, `faculty`, `admin`) |
| `display_name` | Human-readable name shown in the UI |
| `description` | Optional description |
| `is_admin` | Whether members have admin access |
| `scheduler_weight` | Scheduling priority weight for fair-share scheduling |
| `default_token_budget` | Default monthly token budget for new users in this group |
| `default_rpm` | Default requests-per-minute limit |
| `default_max_concurrent` | Default maximum concurrent requests |

Groups are managed via the admin dashboard (`/admin/groups`) or the admin API (`/api/admin/groups`).

### Default Quotas by Group

The seed script creates four default groups:

| Setting | Student | Staff | Faculty | Admin |
|---------|---------|-------|---------|-------|
| Token budget (monthly) | 100,000 | 500,000 | 1,000,000 | 10,000,000 |
| Requests per minute (RPM) | 30 | 60 | 120 | 1,000 |
| Max concurrent requests | 2 | 4 | 8 | 50 |
| Scheduler weight | 1 | 2 | 3 | 10 |
| Admin access | No | No | No | Yes |

Group defaults are configurable through the admin UI or API. The per-role environment variables (e.g., `DEFAULT_TOKEN_BUDGET_STUDENT`, `SCHEDULER_WEIGHT_STAFF`) are **deprecated** and serve only as fallbacks for environments that have not migrated to database-driven groups.

> **Per-user weight override:** Admins can override an individual user's scheduler weight via the user detail page (`/admin/users/{id}`). When set, the user's `weight_override` takes precedence over their group's `scheduler_weight`. An empty or null `weight_override` means the user inherits the group default. This allows fine-grained fair-share tuning for specific users without changing group-wide settings.

### Change Password

Users with local (non-SSO) accounts can change their password from the user dashboard (`/dashboard`). The form requires the current password, a new password (minimum 8 characters), and password confirmation.

### API Key Lifecycle

1. **Generation** -- Keys use the format `mr2_<random_urlsafe_base64>` (48+ characters total).
2. **Storage** -- The raw key is shown once at creation. Only the Argon2 hash and a prefix (`mr2_<first 8 chars>`) are stored in the database.
3. **Verification** -- Lookup by prefix (fast), then full Argon2 hash verification.
4. **Expiration** -- Optional `expires_at` timestamp.
5. **Revocation** -- Keys can be revoked (soft-delete) without deleting the audit trail.
6. **Usage tracking** -- `last_used_at` and `usage_count` updated atomically on each request.

> API keys can become unusable through two distinct mechanisms: **expiration** (automatic -- the key's `expires_at` timestamp has passed) or **revocation** (admin action -- the key's status is set to `REVOKED`). In both cases, authentication fails and the key's audit trail is preserved.

### Quota System

Each user has a quota record with:

- **Token budget** -- Total tokens allowed per period. Deducted on each completed request (prompt + completion tokens).
- **RPM limit** -- Maximum requests per minute.
- **Max concurrent** -- Maximum simultaneous in-flight requests.

When a quota is exceeded, the request is rejected with HTTP 429.

> **Rolling budget period:** Token budgets use a rolling window, not calendar months. Each user's quota tracks `budget_period_start` and `budget_period_days` (default: 30). When the current time exceeds the period end, `tokens_used` resets to zero and the period start advances. This means budget resets are per-user, not system-wide.

### Rate Limiting

> **Note:** RPM and concurrent request rate limiting are defined in the codebase but **not currently enforced**. The rate limiter middleware is not registered in the application. Only token quota (monthly budget) enforcement is active, returning HTTP 429 when the token budget is exceeded. The `rpm_limit` and `max_concurrent` fields are stored in group/quota configuration for future use.

### Quota Increase Requests

Users can submit quota increase requests via the dashboard (`/dashboard/request-quota`). The request includes:
- Desired token budget
- Written justification

Admins review requests at `/admin/requests` or via `POST /api/admin/quota-requests/{id}/review`, which can approve (with a custom granted token amount) or deny the request.

---

## Backend Management

### Node/Backend Model

MindRouter separates the concept of physical GPU servers (**Nodes**) from inference endpoints (**Backends**):

- A **Node** represents a physical server with GPUs and a sidecar agent.
- A **Backend** is an Ollama or vLLM instance running on a node.
- One node can host multiple backends, each assigned specific GPUs via `gpu_indices`.
- Backends without a `node_id` work as standalone endpoints (no GPU telemetry).

```
Node: gpu-server-1 (4x A100-80GB, sidecar at :8007)
├── Backend: vllm-large  (gpu_indices: [0, 1])  ← uses GPUs 0-1
├── Backend: vllm-small  (gpu_indices: [2])      ← uses GPU 2
└── Backend: ollama-misc (gpu_indices: [3])      ← uses GPU 3
```

### Supported Engines

| Engine | Health Check | Model Discovery | Telemetry Source |
|--------|-------------|-----------------|------------------|
| **Ollama** | `GET /api/tags` | `GET /api/tags` + `POST /api/show` (model details) + `POST /api/ps` (loaded models) | Sidecar agent |
| **vLLM** | `GET /health` (fallback: `GET /v1/models`) | `GET /v1/models` | `GET /metrics` (Prometheus format) |

### Registration

**Register a node:**
```bash
curl -X POST http://localhost:8000/api/admin/nodes/register \
  -H "Authorization: Bearer admin-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "gpu-server-1",
    "hostname": "gpu1.example.com",
    "sidecar_url": "http://gpu1.example.com:8007",
    "sidecar_key": "your-sidecar-secret-key"
  }'
```

**Register a backend on that node:**
```bash
curl -X POST http://localhost:8000/api/admin/backends/register \
  -H "Authorization: Bearer admin-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "ollama-gpu1",
    "url": "http://gpu1.example.com:11434",
    "engine": "ollama",
    "max_concurrent": 4,
    "node_id": 1,
    "gpu_indices": [0, 1]
  }'
```

### Enable/Disable/Drain/Refresh

- **Disable** a backend to take it out of rotation without deleting it: `POST /api/admin/backends/{id}/disable`
- **Enable** to bring it back: `POST /api/admin/backends/{id}/enable`
- **Drain** for graceful shutdown: `POST /admin/backends/{id}/drain` (dashboard route, not an API endpoint -- requires admin session)
- **Refresh** to force re-discovery of models and capabilities: `POST /api/admin/backends/{id}/refresh`

### Drain Mode

**Drain mode** provides graceful backend shutdown for maintenance. When a backend is set to draining:

1. The scheduler stops routing **new** requests to the backend.
2. Existing in-flight requests continue to completion.
3. When the backend's queue depth reaches 0, it automatically transitions to **disabled**.

This avoids abruptly killing active requests when you need to restart an inference engine, upgrade models, or perform node maintenance. Use it before upgrading vLLM, restarting Ollama, or taking a GPU node offline.

### Node Lifecycle

Nodes support the following lifecycle operations from the admin dashboard (`/admin/nodes`):

- **Take Offline** (`POST /admin/nodes/{id}/take-offline`) -- Disables all backends on the node and marks the node as offline.
- **Bring Online** (`POST /admin/nodes/{id}/bring-online`) -- Re-enables all backends on the node and marks it online.
- **Force Drain** (`POST /admin/nodes/{id}/force-drain`) -- Force-cancels all active (in-flight) requests on the node's backends. Use this when you need to immediately stop all processing on a node.
- **Active Requests** (`GET /admin/nodes/{id}/active-requests`) -- Returns the count of in-flight requests across all backends on the node.
- **Refresh** (`POST /api/admin/nodes/{id}/refresh`) -- Force-refreshes sidecar GPU data for the node.

---

## Scheduling & Fair Share

MindRouter implements **Weighted Deficit Round Robin (WDRR)** to ensure fair GPU access across users.

### How It Works

1. **Share weights** are assigned by group (e.g., student=1, staff=2, faculty=3, admin=10).
2. Each user has a **deficit counter** that tracks how much service debt they've accrued.
3. On each scheduling round, users with the highest deficit (most underserved) are served first.
4. **Burst credits** allow full cluster utilization when the cluster is idle.
5. **Heavy user deprioritization** kicks in when a user exceeds their fair share within the fairness window.

### Backend Scoring

When multiple backends can serve a request, the scheduler scores them on:

- **Model already loaded** (+100 points) -- avoids cold-loading the model
- **Low GPU utilization** (+50 points) -- prefers idle GPUs
- **Low latency** (+40 points) -- based on EMA of recent response times
- **Short queue** (+30 points) -- prefers backends with fewer queued requests
- **High throughput** (+20 points) -- based on recent tokens/second
- **Priority** (+N x 10 points) -- admin-configured backend priority (N = backend `priority` value)

Hard constraints (vision capability, embedding support, model availability) are checked before soft scoring.

> **Priority gating:** When multiple requests are waiting for backend capacity on the same model, the highest-priority waiter (based on fair-share deficit) proceeds first. Lower-priority waiters yield briefly (100ms) and retry, ensuring fair ordering even under contention.

> **Scheduling audit trail:** Each routing decision is recorded in a `SchedulerDecision` log entry containing the selected backend, all candidate backends with their scores, hard constraints passed/failed, and the user's deficit and weight at decision time. This data is available in the admin audit view for debugging fairness and routing issues.

For the complete algorithm specification, see **[scheduler.md](scheduler.md)**.

### Retry & Failover

MindRouter automatically retries failed inference requests with intelligent backend selection:

- **Max attempts:** Up to 3 total attempts (configurable via `BACKEND_RETRY_MAX_ATTEMPTS`).
- **Retryable errors:** 5xx responses, timeouts, and connection failures trigger retries. 4xx errors (bad request, auth failure, etc.) fail immediately and are **not** retried.
- **Fail-fast routing on retries:** The first attempt waits for backend capacity as normal. Subsequent retry attempts use fail-fast routing (`max_wait=0`) -- if no backend has immediate capacity, the retry is skipped.
- **Backend diversity:** Each retry attempt selects a different backend when one is available, avoiding a backend that just failed.
- **Streaming constraint:** Streaming requests can only retry **before** the first chunk is sent to the client. Once streaming has begun, a failure is terminal because partial data has already been delivered.

> **Mid-stream errors:** If a backend fails after streaming has begun (first SSE chunk already sent), the connection is terminated immediately. No `[DONE]` signal or error event is sent — the SSE stream simply ends, and the client must handle the incomplete response.

### Circuit Breaker

Per-backend circuit breakers prevent routing to repeatedly failing backends:

- **Threshold:** After **`BACKEND_CIRCUIT_BREAKER_THRESHOLD`** (default: 3) consecutive failures, the backend's circuit opens and it is excluded from routing.
- **Recovery:** After **`BACKEND_CIRCUIT_BREAKER_RECOVERY_SECONDS`** (default: 30) seconds, a probe request is allowed through to test recovery.
- **State transitions:** Closed (healthy) → Open (failing, excluded from routing) → Half-Open (probe allowed) → Closed (recovered)
- Circuit breakers work alongside retry logic -- broken backends are automatically skipped during failover selection.

---

## Translation Layer

MindRouter's translation layer enables cross-engine routing: a request arriving in OpenAI, Ollama, or Anthropic format can be served by any Ollama or vLLM backend. All translation passes through a **canonical internal schema**.

### Request Flow

```
OpenAI Request    ──→ OpenAIInTranslator    ──→ CanonicalChatRequest
                                                       │
Ollama Request    ──→ OllamaInTranslator    ──→ CanonicalChatRequest
                                                       │
Anthropic Request ──→ AnthropicInTranslator ──→ CanonicalChatRequest
                                                       │
                                                       ▼
                                                [Scheduler selects backend]
                                                       │
                           ┌───────────────────────────┴───────────────┐
                           ▼                                           ▼
                 OllamaOutTranslator                         VLLMOutTranslator
                 (Ollama backend)                            (vLLM backend, OpenAI format)
```

### Canonical Schemas

The canonical internal representation (`backend/app/core/canonical_schemas.py`) includes:

- **CanonicalChatRequest** -- model, messages, temperature, top_p, max_tokens, stream, tools, tool_choice, response_format, think (`Union[bool, str]`), reasoning_effort, etc.
- **CanonicalMessage** -- role (system/user/assistant/tool), content (text or multimodal content blocks, nullable), tool_calls, tool_call_id
- **ContentBlock** -- TextContent, ImageUrlContent, or ImageBase64Content
- **CanonicalToolCall** / **CanonicalFunctionCall** -- tool call with id, function name, and arguments (JSON string)
- **CanonicalToolDefinition** -- tool definition with function name, description, and parameters schema
- **CanonicalEmbeddingRequest** -- model, input, encoding_format, dimensions
- **CanonicalChatResponse** / **CanonicalStreamChunk** -- response and streaming types (including tool call deltas)

### Key Translation Mappings

| Concept | OpenAI Format | Ollama Format | Anthropic Format | Canonical |
|---------|---------------|---------------|------------------|-----------|
| Max tokens | `max_completion_tokens` or `max_tokens` | `options.num_predict` | `max_tokens` (required) | `max_tokens` |
| Streaming default | `false` | `true` | `false` | -- |
| System prompt | `messages` with `role: system` | `messages` with `role: system` | Top-level `system` field | `CanonicalMessage(role=SYSTEM)` |
| JSON mode | `response_format: {"type": "json_object"}` | `format: "json"` | -- | `response_format.type = JSON_OBJECT` |
| JSON schema | `response_format: {"type": "json_schema", ...}` | `format: {schema}` | `output_config.format: {"type": "json_schema", ...}` | `response_format.type = JSON_SCHEMA` |
| Parameters | Top-level fields | `options` dict | Top-level fields | Top-level fields |
| Stop sequences | `stop` | `options.stop` | `stop_sequences` | `stop` |
| Images | `image_url` content block | `images` array (base64) | `image` block with `source` | `ImageBase64Content` / `ImageUrlContent` |
| Tool definitions | `tools` | `tools` | `tools` (with `input_schema`) | `tools` (`CanonicalToolDefinition`) |
| Tool choice | `tool_choice` | -- | `tool_choice` (`auto`/`any`/`tool`) | `tool_choice` |
| Tool calls | `tool_calls` (JSON string args) | `tool_calls` (dict args) | `tool_use` content blocks | `CanonicalToolCall` (JSON string args) |
| Tool results | `role: "tool"` + `tool_call_id` | -- | `tool_result` content blocks | `CanonicalMessage(role=TOOL, tool_call_id)` |
| Thinking mode | `think` (bool), `thinking.type`, `chat_template_kwargs`, `reasoning_effort` | `think` (bool or `"low"`/`"medium"`/`"high"`) | `thinking.type` (enabled/adaptive/disabled) | `think` (`Union[bool, str]`) |
| User ID | `user` | -- | `metadata.user_id` | `user` |
| Structured output | `response_format` | `format` | `output_config` | `response_format` |
| Stream format | SSE (`data: {...}`) | NDJSON | SSE (Anthropic events) | `CanonicalStreamChunk` |

### Translators

| Translator | Direction | Purpose |
|------------|-----------|---------|
| `OpenAIInTranslator` | API → Canonical | Translates incoming OpenAI-format requests |
| `OllamaInTranslator` | API → Canonical | Translates incoming Ollama-format requests |
| `AnthropicInTranslator` | API → Canonical | Translates incoming Anthropic Messages API requests; also formats responses and SSE stream events back to Anthropic format |
| `OllamaOutTranslator` | Canonical → Backend | Translates outgoing requests to Ollama backends |
| `VLLMOutTranslator` | Canonical → Backend | Translates outgoing requests to vLLM backends |

All translators use static methods -- no instantiation needed.

### vLLM Thinking Translations

- When `think` is a **boolean** (Qwen-style), it translates to `chat_template_kwargs: {enable_thinking: bool}` for vLLM.
- When `think` is a **string** like `"low"`/`"medium"`/`"high"` (GPT-OSS style), it translates to `reasoning_effort` for vLLM.

### Model-Specific Behaviors

- **Qwen3-32B on vLLM:** Does not use the `reasoning_content` response field. Instead, thinking content appears as `<think>...</think>` tags embedded in the content field. MindRouter automatically extracts these tags into the canonical `reasoning` field for both streaming and non-streaming responses.
- **Qwen3.5 on vLLM with thinking disabled:** When thinking is explicitly disabled (`think: false`), vLLM may return all output in `reasoning_content` with an empty `content` field. MindRouter promotes the reasoning content to the `content` field in this case, ensuring clients receive the expected output.

### Backend-Specific Options

> The `backend_options` dict allows passing Ollama-specific parameters (e.g., `mirostat`, `tfs_z`, `repeat_penalty`) that are forwarded directly to Ollama backends. These options are ignored for vLLM backends. See [Implementation Notes](#implementation-notes) for details.

---

## Telemetry & Monitoring

### GPU Sidecar Agent

Each GPU node runs a lightweight FastAPI sidecar agent (`sidecar/gpu_agent.py`) that exposes per-GPU hardware metrics:

**Collected metrics per GPU:**
- Utilization (GPU % and memory %)
- Memory (used/free/total GB)
- Temperature (GPU and memory)
- Power draw and limit (watts)
- Fan speed, SM/memory clocks
- Running processes (PID + memory) — **Note:** Process information is collected by the sidecar but is not currently stored in MindRouter's database — it is available only in the raw `/gpu-info` response.
- Device identity (name, UUID, compute capability)
- Driver and CUDA versions

**Authentication:** Requires `SIDECAR_SECRET_KEY` env var. All requests must include `X-Sidecar-Key` header (constant-time comparison).

**Deployment options:**

1. **Docker Compose** -- `docker compose --profile gpu up gpu-sidecar`
2. **Standalone Docker** -- Build from `sidecar/Dockerfile.sidecar`, run with `--gpus all`
3. **Direct Python** -- `pip install fastapi uvicorn nvidia-ml-py && python sidecar/gpu_agent.py`

**Sidecar HTTP endpoints:**

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/health` | `X-Sidecar-Key` | Health check — returns `{"status": "ok", "gpu_count": N, "sidecar_version": "..."}` |
| GET | `/gpu-info` | `X-Sidecar-Key` | Full GPU metrics — returns hostname, timestamp, driver/CUDA versions, sidecar version, per-GPU metrics |
| POST | `/ollama/pull` | `X-Sidecar-Key` | Start model pull — body: `{"ollama_url": "...", "model": "..."}`, returns `{"job_id": "...", "status": "pulling"}` |
| GET | `/ollama/pull/{job_id}` | `X-Sidecar-Key` | Poll pull progress — returns job status with progress, error, timestamps |
| POST | `/ollama/delete` | `X-Sidecar-Key` | Delete a model from Ollama backend |

### Health Polling

The Backend Registry runs an adaptive polling loop that accelerates when problems are detected:

- **Normal interval:** `BACKEND_POLL_INTERVAL` (default: 30 seconds)
- **Fast interval:** When failures are detected, polling speeds up to `BACKEND_ADAPTIVE_POLL_FAST_INTERVAL` (default: 10 seconds)
- **Fast duration:** Fast polling lasts for `BACKEND_ADAPTIVE_POLL_FAST_DURATION` (default: 120 seconds) before returning to normal
- **Unhealthy threshold:** `BACKEND_UNHEALTHY_THRESHOLD` (default: 3) consecutive poll failures marks a backend as unhealthy

Each poll cycle has two phases:
1. Poll sidecar agents (one per physical node) for GPU snapshots
2. Poll each backend adapter for health, models, and engine-specific telemetry

### Startup Fast Polls

On container start, the registry runs **two immediate full poll cycles** with a 5-second gap between them. This ensures backends and nodes are marked healthy within seconds of a restart, rather than waiting for the first normal 30-second poll interval.

### Circuit Breaker

> See [Circuit Breaker](#circuit-breaker) under Scheduling & Fair Share for full details on thresholds, recovery, and state transitions.

### Latency Tracking

Exponential Moving Average (EMA) tracks per-backend latency:

- **Alpha:** 0.3 (30% current observation, 70% history)
- **Metrics:** Total latency EMA and TTFT (time-to-first-token) EMA
- **Throughput score:** `1.0 / (1.0 + latency_ms / 5000.0)` -- used in backend scoring
- **Persistence:** EMAs are periodically saved to the database for recovery after restart

### Redis

When a `REDIS_URL` is configured, MindRouter uses Redis for several purposes beyond rate limiting:

- **Inflight streaming token counting** -- During streaming responses, token counts are atomically incremented/decremented in Redis (`streaming:inflight_tokens` key) so the `/api/cluster/throughput` endpoint can include tokens from in-progress requests.
- **Per-user quota caching** -- Token usage counters are cached in Redis (`quota:tokens:{user_id}` keys) for fast atomic increment/read without hitting the database on every request.
- **Graceful degradation** -- All Redis operations are wrapped in try/except blocks. If Redis is unavailable (not configured, connection lost, or connection fails), MindRouter falls back silently: inflight token counts return 0, quota checks fall through to the database, and a `redis_disabled` or `redis_connect_failed` log entry is emitted. No requests are rejected due to Redis unavailability.

### Prometheus Metrics

Scrape `/metrics` for Prometheus-compatible metrics. See the [Health & Metrics Endpoints](#health--metrics-endpoints) section for the full list.

### Telemetry API

Admin users can access detailed telemetry via the API:

- **Cluster overview** -- All nodes, backends, GPUs with current metrics
- **Historical data** -- Time-series with configurable resolution (1m, 5m, 15m, 1h, 6h, 1d)
- **Per-GPU history** -- Individual GPU device telemetry over time
- **Export** -- Download telemetry data as JSON or CSV

See [Telemetry endpoints](#telemetry) for the full API reference.

---

## Chat System

MindRouter includes a built-in chat interface at `/chat` with full conversation management.

### Conversations

- Each user has their own conversation history
- Conversations store: title, selected model, creation/update timestamps
- Users can rename, switch models, or delete conversations
- Up to 50 conversations shown in the sidebar (most recent first)
- Conversations older than `CONVERSATION_RETENTION_DAYS` (default 730 days / 2 years) are automatically purged by a background cleanup task

### Messages

- Messages include role (user/assistant/system) and content
- Assistant messages are streamed in real-time
- Messages are immutable once sent — to revise a conversation, start a new one or continue from the current point
- Conversations are automatically titled based on the first user message
- Attachments are linked to individual messages

### File Upload

Supported file types and processing:

| Category | Extensions | Processing |
|----------|-----------|------------|
| Images | `.jpg`, `.jpeg`, `.png`, `.gif`, `.webp` | Resized to max 1536px, compressed JPEG q85, thumbnail generated |
| Documents | `.pdf` | Text extracted from all pages, first-page thumbnail generated |
| Documents | `.docx` | Text extracted from all paragraphs |
| Spreadsheets | `.xlsx` | All sheets read, formatted as tab-separated text |
| Text files | `.txt`, `.md`, `.csv`, `.json`, `.html`, `.htm`, `.log` | Read as-is |

**Limits:**
- Chat file uploads are limited to `CHAT_UPLOAD_MAX_SIZE_MB` (default 10 MB)
- System artifact uploads allow up to `ARTIFACT_MAX_SIZE_MB` (default 50 MB)
- Artifact storage path: `/data/artifacts` (configurable via `ARTIFACT_STORAGE_PATH`)
- Artifact retention: 365 days

**Storage layout:**
```
/artifacts/YYYY/MM/DD/<sha256_prefix>/<full_sha256>_<uuid>.<ext>
```

### Vision Model Support

- Models with vision capability are automatically detected by name patterns (e.g., `llava`, `-vl-`, `vision`)
- When images are sent to a vision model, they are included as base64-encoded content blocks
- When images are sent to a non-vision model, they are replaced with a placeholder: `[Image omitted -- model does not support vision: filename]`
- A warning modal is shown in the chat UI when uploading images to a non-vision model

### Streaming

Chat responses are streamed in real-time:
- Backend streaming uses NDJSON (Ollama) or SSE (vLLM/OpenAI)
- The chat UI renders tokens as they arrive
- TTFT (time-to-first-token) is tracked for latency monitoring
- If the client disconnects, the backend request is not cancelled (to prevent DB corruption)

### Chat Configuration

Admins can configure the chat interface defaults at `/admin/chat-config`:

- **Core models** -- Select which models appear in the chat model selector.
- **Default model** -- The model pre-selected when a user starts a new conversation.
- **System prompt** -- A default system prompt injected into all chat conversations (can be reset to empty).
- **Max tokens** -- Default `max_tokens` value for chat requests.
- **Temperature** -- Default temperature for chat requests.
- **Thinking mode** -- Enable or disable thinking/reasoning mode by default for chat.
- **TTS settings** -- Enable/disable "Read Aloud" in the chat UI, select TTS provider (Kokoro or OpenedAI), set default voice (dynamically populated dropdown from upstream TTS service) and playback speed.
- **STT settings** -- Enable/disable microphone input in the chat UI.

### Chat UI Features

**File drag-and-drop** -- Files can be uploaded by dragging and dropping anywhere in the chat window (not just the input area). A visual drop overlay appears during drag.

**Advanced models toggle** -- When core models are configured (via admin Chat Config), the model dropdown shows only core models by default. An "Advanced" checkbox reveals the full model list. This preference persists in browser localStorage.

**Per-request thinking controls** -- For thinking-capable models, the chat UI shows inline controls:

- A checkbox to enable/disable thinking mode (Qwen-style boolean)
- A dropdown to select reasoning effort level (low/medium/high for GPT-OSS-style models)

These controls only appear when the selected model supports thinking.

**Thinking block collapsing** -- When thinking/reasoning content is streamed, it appears in a collapsible block with a toggle button. Users can expand or collapse the reasoning to focus on the final response.

**Keyboard shortcuts** -- `Shift+Enter` inserts a newline in the message input. `Enter` alone sends the message.

**Sidebar collapse/expand** -- The conversation sidebar can be collapsed or expanded via a toggle button. The sidebar state persists across page reloads via browser localStorage.

**Copy buttons** -- Each assistant response includes a "Copy" button to copy the response text to the clipboard. Individual code blocks also have copy buttons that appear on hover.

**Image lightbox** -- Clicking an image thumbnail in a chat message opens a larger preview in a lightbox modal.

**LaTeX rendering** -- Mathematical expressions are rendered client-side. The system handles LaTeX placeholder extraction, dollar-sign notation, and bare environment wrapping for reliable rendering of equations in responses.

**Auto-conversation titling** -- New conversations are automatically titled from the first user message. The title can be updated by the user via the conversation settings.

**Model selection persistence** -- The last selected model is saved to browser localStorage and automatically restored when returning to the chat.

---

## Voice API

MindRouter provides public TTS and STT endpoints that proxy to self-hosted voice services. These are OpenAI-compatible and separate from the chat UI's voice features.

### Architecture

The Voice API acts as a proxy between API consumers and upstream voice services:

- **TTS**: Proxies to a self-hosted service (Kokoro TTS or OpenedAI Speech) exposing `/v1/audio/speech`
- **STT**: Proxies to a self-hosted service (faster-whisper / Speaches) exposing `/v1/audio/transcriptions`

Both endpoints require API key authentication and deduct a configurable fixed token cost from the user's quota per request.

### Admin Configuration

Voice API settings are managed on two admin pages:

**Voice API Config** (`/admin/voice-config`):
- TTS backend URL and API key
- Available TTS voices (one per line -- restricts which voices users can choose in their dashboard)
- Default System Voice (dynamically populated dropdown -- the voice assigned to users unless they choose their own)
- STT backend URL, API key, and default model
- Quota token costs per TTS/STT request

**Chat Config** (`/admin/chat-config`):
- TTS enable/disable toggle, provider, default voice, playback speed (chat UI only)
- STT enable/disable toggle (chat UI only)

The backend connection settings (URLs, API keys) are shared between the chat UI and the Voice API. The chat-specific settings (enable toggles, provider, voice, speed) only affect the chat interface and do not gate the Voice API endpoints.

**Voice discovery endpoint** (`GET /api/tts-voices`): Returns the list of available TTS voices and the current default voice. Tries the upstream TTS service first (`{tts_url}/v1/audio/voices`), then falls back to the `voice_api.tts_voices` config. Supports `?allowed_only=true` to filter to only admin-configured voices (used by the user dashboard). Response: `{"voices": [...], "source": "upstream"|"config", "default_voice": "af_heart"}`.

**Chat TTS voice resolution order:** When a user triggers Read Aloud in the chat UI, the voice is resolved with this priority:
1. Explicit `voice` in the request body (not currently exposed in the UI)
2. User's per-user preference (`user.{user_id}.tts_voice`, set via Dashboard > Preferences)
3. Default System Voice (`voice_api.default_voice`, set via Admin > Voice Config)
4. Chat Config fallback (`voice.tts_voice`, set via Admin > Chat Config)

**Chat TTS speed resolution order:**
1. User's per-user preference (`user.{user_id}.tts_speed`, set via Dashboard > Preferences)
2. Admin global default (`voice.tts_speed`, set via Admin > Chat Config)

### Limitations

The Voice API is a thin proxy layer. It does not perform any audio processing itself -- it forwards requests to the upstream TTS/STT service and relays the response.

**STT upload constraints:**
- **50 MB** maximum upload size (nginx limit)
- **120-second** proxy timeout to the upstream Whisper service
- The entire audio file is buffered in memory before forwarding
- No server-side audio segmentation or chunking

These limits are appropriate for short-to-medium clips (up to ~10 minutes). For longer audio, clients should split the file into chunks before sending -- see [STT Limitations & Long Audio](#stt-limitations--long-audio) in the API Reference for code examples.

**Quota model:** Both TTS and STT use a flat per-request token cost regardless of input size. Admins can adjust the cost via the Voice API Config page.

### DB Config Keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `voice.tts_url` | string | (none) | TTS service base URL |
| `voice.tts_api_key` | string | (none) | TTS service API key |
| `voice.stt_url` | string | (none) | STT service base URL |
| `voice.stt_api_key` | string | (none) | STT service API key |
| `voice.stt_model` | string | `"whisper-large-v3-turbo"` | Default STT model |
| `voice.tts_enabled` | boolean | `false` | Enable TTS in chat UI |
| `voice.tts_provider` | string | `"kokoro"` | Chat TTS provider (`kokoro` or `openedai`) |
| `voice.tts_voice` | string | `"af_heart"` | Default voice for chat TTS |
| `voice.tts_speed` | float | `1.0` | Default playback speed for chat TTS |
| `voice.stt_enabled` | boolean | `false` | Enable STT in chat UI |
| `voice_api.tts_voices` | string | `"af_heart\naf_bella\nam_adam\nam_michael"` | Available TTS voices (newline-separated, restricts user choices) |
| `voice_api.default_voice` | string | `"af_heart"` | Default System Voice — assigned to users unless they choose their own |
| `voice_api.tts_quota_tokens` | integer | `100` | Token cost per TTS API request |
| `voice_api.stt_quota_tokens` | integer | `200` | Token cost per STT API request |
| `user.{user_id}.tts_voice` | string | (none) | Per-user TTS voice preference (overrides system default) |
| `user.{user_id}.tts_speed` | string | (none) | Per-user TTS playback speed preference (overrides `voice.tts_speed`) |

---

## Blog System

MindRouter includes a built-in blog/CMS for publishing announcements, documentation, and updates.

### Public Pages

- **Blog listing** (`/blog`) -- Displays all published blog posts, most recent first.
- **Blog post** (`/blog/{slug}`) -- Displays a single blog post by its URL slug.

### Admin Management

Blog management is available at `/admin/blog` (admin access required):

- **Post listing** (`/admin/blog`) -- View all posts (published and draft) with edit/delete controls.
- **Create post** (`/admin/blog/new`) -- Create a new blog post with title, slug, content, and publish status.
- **Edit post** (`/admin/blog/{id}/edit`) -- Edit an existing post's title, slug, and content.
- **Publish post** (`POST /admin/blog/{id}/publish`) -- Publish or unpublish a post.
- **Delete post** (`POST /admin/blog/{id}/delete`) -- Soft-delete a post (not permanently removed from the database).

The blog editor includes:
- **Live split-screen markdown preview** with a "Show Preview"/"Hide Preview" toggle
- **Real-time HTML rendering** using marked.js with syntax highlighting
- **Auto-generated URL slugs** from post titles (editable before saving)

---

## Configuration Reference

All settings are loaded from environment variables or `.env` / `.env.prod` files. Variable names are case-insensitive.

### Application

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `APP_NAME` | str | `MindRouter` | Application name |
| `APP_VERSION` | str | (from `pyproject.toml`) | Application version |
| `DEBUG` | bool | `false` | Enable debug mode |
| `RELOAD` | bool | `false` | Auto-reload on code changes (development) |

### Database

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DATABASE_URL` | str | `mysql+pymysql://...` | MariaDB/MySQL connection string |
| `DATABASE_POOL_SIZE` | int | `30` | Connection pool size |
| `DATABASE_MAX_OVERFLOW` | int | `20` | Max overflow connections beyond pool |
| `DATABASE_ECHO` | bool | `false` | Log SQL queries |

### Cache

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `REDIS_URL` | str | `None` | Redis connection string (optional) |

### Security

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SECRET_KEY` | str | `dev-secret-key-...` | JWT/session signing key (**change in production**) |
| `JWT_ALGORITHM` | str | `HS256` | JWT signing algorithm |
| `JWT_EXPIRATION_HOURS` | int | `24` | JWT token lifetime |
| `SESSION_COOKIE_NAME` | str | `mindrouter_session` | Session cookie name |
| `SESSION_COOKIE_SECURE` | bool | `false` | HTTPS-only cookies |
| `SESSION_COOKIE_HTTPONLY` | bool | `true` | JavaScript-inaccessible cookies |
| `SESSION_COOKIE_SAMESITE` | str | `lax` | SameSite cookie policy |
| `API_KEY_HASH_ALGORITHM` | str | `argon2` | API key hashing algorithm |

### Azure AD SSO

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `AZURE_AD_CLIENT_ID` | str | `None` | Azure AD application (client) ID |
| `AZURE_AD_CLIENT_SECRET` | str | `None` | Azure AD client secret |
| `AZURE_AD_TENANT_ID` | str | `None` | Azure AD tenant ID |
| `AZURE_AD_REDIRECT_URI` | str | `https://your-domain.example.com/login/azure/authorized` | OAuth2 redirect URI |
| `AZURE_AD_DEFAULT_GROUP` | str | `other` | Default group for new SSO users |

Azure AD SSO is enabled automatically when both `AZURE_AD_CLIENT_ID` and `AZURE_AD_TENANT_ID` are set. Users authenticating via SSO for the first time are automatically created with JIT (just-in-time) group mapping based on the user's `jobTitle` claim from Azure AD. The mapping uses case-insensitive substring matching: if `jobTitle` contains "student", the user is assigned to the `students` group; if it contains "faculty" or "professor", the user is assigned to the `faculty` group; if it contains "staff", the user is assigned to the `staff` group. If `jobTitle` is missing or does not match any of these substrings, the user falls back to the group specified by `AZURE_AD_DEFAULT_GROUP`.

### Artifact Storage

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ARTIFACT_STORAGE_PATH` | str | `/data/artifacts` | File storage directory |
| `ARTIFACT_MAX_SIZE_MB` | int | `50` | Max artifact file size |
| `ARTIFACT_RETENTION_DAYS` | int | `365` | Artifact retention period |

### Quotas (Deprecated)

> **Note:** These per-role environment variables are **deprecated**. Use database-driven Groups instead (see [Groups](#groups)). These variables serve as fallbacks only.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DEFAULT_TOKEN_BUDGET_STUDENT` | int | `100000` | Student token budget |
| `DEFAULT_TOKEN_BUDGET_STAFF` | int | `500000` | Staff token budget |
| `DEFAULT_TOKEN_BUDGET_FACULTY` | int | `1000000` | Faculty token budget |
| `DEFAULT_TOKEN_BUDGET_ADMIN` | int | `10000000` | Admin token budget |
| `DEFAULT_RPM_STUDENT` | int | `30` | Student requests per minute |
| `DEFAULT_RPM_STAFF` | int | `60` | Staff requests per minute |
| `DEFAULT_RPM_FACULTY` | int | `120` | Faculty requests per minute |
| `DEFAULT_RPM_ADMIN` | int | `1000` | Admin requests per minute |
| `DEFAULT_MAX_CONCURRENT_STUDENT` | int | `2` | Student max concurrent requests |
| `DEFAULT_MAX_CONCURRENT_STAFF` | int | `4` | Staff max concurrent requests |
| `DEFAULT_MAX_CONCURRENT_FACULTY` | int | `8` | Faculty max concurrent requests |
| `DEFAULT_MAX_CONCURRENT_ADMIN` | int | `50` | Admin max concurrent requests |

### Scheduler

> **Note:** The per-role `SCHEDULER_WEIGHT_*` variables are **deprecated**. Use `Group.scheduler_weight` in the database instead.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SCHEDULER_WEIGHT_STUDENT` | int | `1` | Student scheduling priority weight (deprecated) |
| `SCHEDULER_WEIGHT_STAFF` | int | `2` | Staff scheduling priority weight (deprecated) |
| `SCHEDULER_WEIGHT_FACULTY` | int | `3` | Faculty scheduling priority weight (deprecated) |
| `SCHEDULER_WEIGHT_ADMIN` | int | `10` | Admin scheduling priority weight (deprecated) |
| `SCHEDULER_FAIRNESS_WINDOW` | int | `300` | Fairness tracking window (seconds) |
| `SCHEDULER_DEPRIORITIZE_THRESHOLD` | float | `0.5` | Usage threshold for deprioritization |
| `SCHEDULER_SCORE_MODEL_LOADED` | int | `100` | Score bonus for pre-loaded model |
| `SCHEDULER_SCORE_LOW_UTILIZATION` | int | `50` | Score bonus for low GPU utilization |
| `SCHEDULER_SCORE_LATENCY` | int | `40` | Score factor for low latency |
| `SCHEDULER_SCORE_SHORT_QUEUE` | int | `30` | Score factor for short queue |
| `SCHEDULER_SCORE_HIGH_THROUGHPUT` | int | `20` | Score factor for high throughput |

### Latency Tracking

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `LATENCY_EMA_ALPHA` | float | `0.3` | EMA smoothing factor |
| `LATENCY_EMA_PERSIST_INTERVAL` | int | `30` | EMA persistence interval (seconds) |

### Backend Registry

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `BACKEND_POLL_INTERVAL` | int | `30` | Health check interval (seconds) |
| `BACKEND_HEALTH_TIMEOUT` | int | `5` | Health check timeout (seconds) |
| `BACKEND_UNHEALTHY_THRESHOLD` | int | `3` | Failed checks before marking unhealthy |
| `BACKEND_CIRCUIT_BREAKER_THRESHOLD` | int | `3` | Failures before circuit opens |
| `BACKEND_CIRCUIT_BREAKER_RECOVERY_SECONDS` | int | `30` | Circuit breaker recovery time |
| `BACKEND_ADAPTIVE_POLL_FAST_INTERVAL` | int | `10` | Fast poll interval after unhealthy |
| `BACKEND_ADAPTIVE_POLL_FAST_DURATION` | int | `120` | Duration of fast polling (seconds) |

### Request Handling

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MAX_REQUEST_SIZE` | int | `52428800` | Max HTTP request body (50 MB) |
| `BACKEND_REQUEST_TIMEOUT` | int | `300` | Total request timeout (seconds) |
| `BACKEND_REQUEST_TIMEOUT_PER_ATTEMPT` | int | `180` | Per-attempt timeout (seconds) |
| `BACKEND_RETRY_MAX_ATTEMPTS` | int | `3` | Max total retry attempts |
| `BACKEND_RETRY_ATTEMPTS` | int | `2` | Default retry attempts (deprecated -- not currently used by retry logic; see `BACKEND_RETRY_MAX_ATTEMPTS`) |
| `BACKEND_RETRY_BACKOFF` | float | `1.0` | Retry backoff multiplier (deprecated -- not currently used by retry logic) |
| `STRUCTURED_OUTPUT_RETRY_ON_INVALID` | bool | `true` | When enabled, retries the request on a different backend if the response fails structured output JSON validation |

> *Note: `STRUCTURED_OUTPUT_RETRY_ON_INVALID` is defined but not currently implemented in the inference pipeline. It is reserved for future use.*

> **Timeout split behavior:** The total `BACKEND_REQUEST_TIMEOUT` is split in half -- the first half is allocated for routing and capacity wait (waiting for a backend with available capacity), and the remaining half for actual inference. Retry attempts after the first use immediate fail-fast routing (`max_wait=0`) to avoid wasting time waiting again. `BACKEND_REQUEST_TIMEOUT_PER_ATTEMPT` (default 180s) applies independently to each individual attempt, separate from the total timeout budget.

### Logging

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `LOG_LEVEL` | str | `INFO` | Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `LOG_FORMAT` | str | `json` | Log format (`json` or `text`) |
| `LOG_FILE` | str | `None` | Log file path (optional, stdout if not set) |

### Audit Logging

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `AUDIT_LOG_ENABLED` | bool | `true` | Enable audit logging |
| `AUDIT_LOG_PROMPTS` | bool | `true` | Log user prompts |
| `AUDIT_LOG_RESPONSES` | bool | `true` | Log LLM responses |

### Telemetry & GPU

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `TELEMETRY_RETENTION_DAYS` | int | `30` | Telemetry data retention period |
| `TELEMETRY_CLEANUP_INTERVAL` | int | `3600` | Cleanup interval (seconds) |
| `SIDECAR_TIMEOUT` | int | `5` | Sidecar HTTP call timeout (seconds) |
| `GPU_AGENT_HOST` | str | `0.0.0.0` | Bind address for sidecar HTTP server |
| `GPU_AGENT_PORT` | int | `8007` | Port for sidecar HTTP server |

### Observability

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `METRICS_ENABLED` | bool | `true` | Enable Prometheus metrics |
| `METRICS_PREFIX` | str | `mindrouter` | Metrics name prefix |
| `OTEL_ENABLED` | bool | `false` | Enable OpenTelemetry |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | str | `None` | OpenTelemetry exporter endpoint |

### CORS

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `CORS_ORIGINS` | list | `["http://localhost:3000", "http://localhost:8000"]` | Allowed origins (JSON array or comma-separated) |

### Chat UI

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `CHAT_FILES_PATH` | str | `/data/chat_files` | Chat file upload directory |
| `CHAT_UPLOAD_MAX_SIZE_MB` | int | `10` | Max upload file size (MB) |
| `CHAT_UPLOAD_ALLOWED_EXTENSIONS` | list | See below | Allowed upload file extensions |

Default allowed extensions: `.txt`, `.md`, `.csv`, `.json`, `.html`, `.htm`, `.log`, `.docx`, `.xlsx`, `.pdf`, `.jpg`, `.jpeg`, `.png`, `.gif`, `.webp`

### Conversation Retention

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `CONVERSATION_RETENTION_DAYS` | int | `730` | Conversation retention period (days, default 2 years) |
| `CONVERSATION_CLEANUP_INTERVAL` | int | `86400` | Cleanup interval in seconds (default 24 hours) |

### Web Search (Brave)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `BRAVE_SEARCH_API_KEY` | str | `None` | Brave Search API key (enables web search in chat) |
| `BRAVE_SEARCH_MAX_RESULTS` | int | `5` | Maximum number of search results to inject as context |

When configured, users can toggle web search in the chat interface. Search results from the Brave Search API are formatted and injected into the system message as context before the LLM generates its response.

### Tokenizer

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DEFAULT_TOKENIZER` | str | `cl100k_base` | Default tokenizer encoding |

### Runtime AppConfig (Database-Driven)

In addition to the environment variables above, MindRouter stores runtime configuration in the `app_config` database table. These settings are managed via the Admin Dashboard (Site Settings and Chat Config pages) and take effect immediately without restart.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `chat.core_models` | JSON array | `[]` | Models pinned to chat dropdown |
| `chat.default_model` | string | (none) | Default model for new conversations |
| `chat.system_prompt` | string | (none) | Global system prompt override for chat |
| `chat.max_tokens` | integer | `16384` | Default max_tokens for chat requests |
| `chat.temperature` | float | (none) | Default temperature override |
| `chat.think` | bool/string | (none) | Default thinking mode (`true`/`false`/`"low"`/`"medium"`/`"high"`) |
| `voice.tts_enabled` | boolean | `false` | Enable TTS "Read Aloud" in chat UI |
| `voice.tts_provider` | string | `"kokoro"` | Chat TTS provider (`kokoro` or `openedai`) |
| `voice.tts_voice` | string | `"af_heart"` | Default voice for chat TTS |
| `voice.tts_speed` | float | `1.0` | Default playback speed (0.5--2.0) |
| `voice.stt_enabled` | boolean | `false` | Enable microphone input in chat UI |
| `voice.tts_url` | string | (none) | TTS service base URL |
| `voice.tts_api_key` | string | (none) | TTS service API key |
| `voice.stt_url` | string | (none) | STT service base URL |
| `voice.stt_api_key` | string | (none) | STT service API key |
| `voice.stt_model` | string | `"whisper-large-v3-turbo"` | Default STT model |
| `voice_api.tts_voices` | string | (see below) | Available TTS voices (newline-separated, restricts user choices) |
| `voice_api.default_voice` | string | `"af_heart"` | Default System Voice assigned to users |
| `voice_api.tts_quota_tokens` | integer | `100` | Token cost per TTS API request |
| `voice_api.stt_quota_tokens` | integer | `200` | Token cost per STT API request |
| `user.{user_id}.tts_voice` | string | (none) | Per-user TTS voice preference |
| `user.{user_id}.tts_speed` | float | (none) | Per-user TTS playback speed preference |
| `app.timezone` | string | `"America/Los_Angeles"` | IANA timezone for date display in web UI |
| `ollama.enforce_num_ctx` | boolean | `true` | Override user-supplied `num_ctx` with model config `context_length` |

---

## Implementation Notes

This section documents internal behaviors useful for operators and developers.

### Inflight Token Estimation

During streaming responses, tokens are estimated at **1 token per 4 characters** for real-time quota and throughput tracking. Estimates are flushed to Redis every 10 chunks. When the response completes, estimated counts are replaced by accurate backend-reported token counts.

> **Token count fallback:** If a backend returns zero for both prompt and completion token counts, MindRouter falls back to the job's pre-estimated token counts (based on tiktoken encoding of the input).

### Redis Token Counter Sync

A background sync loop flushes Redis token usage counters to the database every **60 seconds**. On startup, counters are seeded from the database. A final flush runs on graceful shutdown to prevent token count drift.

### Conversation Cleanup

A background task automatically deletes expired conversations every **24 hours** (configurable via `CONVERSATION_CLEANUP_INTERVAL`). The default retention period is 2 years (`CONVERSATION_RETENTION_DAYS=730`).

### Backend Options Passthrough

The `backend_options` dict in requests allows passing Ollama-specific options (e.g., `mirostat`, `tfs_z`, `repeat_penalty`) directly to Ollama backends. These options are ignored when the request is routed to a vLLM backend.

### Thinking Input Format Priority

The system accepts four input formats for thinking/reasoning mode, resolved in priority order:

1. `think` field (bool or string) -- canonical format
2. `thinking: {type: "enabled"/"disabled"}` -- OpenAI/Anthropic style
3. `chat_template_kwargs: {enable_thinking: bool}` -- vLLM-specific
4. Ollama top-level `think` field

### Response Format Normalization

When an `/api/chat` request (Ollama format) is routed to a vLLM backend, responses are automatically converted back to Ollama format. The `reasoning_content` field from vLLM/OpenAI responses is promoted to the Ollama `thinking` field.

### Per-Backend Performance Tracking

The scheduler maintains an exponential moving average (EMA) of request latency (`latency_ema_ms`) and time-to-first-token (`ttft_ema_ms`) for each backend. These metrics inform the "Low Latency" and "High Throughput" scoring factors. Circuit breaker state (`live_failure_count`, `circuit_open_until`) is also persisted per-backend, surviving application restarts.

### Soft Delete

User accounts and blog posts use soft deletion -- a `deleted_at` timestamp is set rather than removing the row. Soft-deleted records are excluded from normal queries but retained in the database for audit purposes.

### Status Enums

**BackendStatus:** `HEALTHY` (available for routing), `UNHEALTHY` (failed health checks), `DISABLED` (admin-disabled), `DRAINING` (graceful shutdown -- no new requests, existing ones complete), `UNKNOWN` (initial state before first health check).

**NodeStatus:** `ONLINE` (reachable), `OFFLINE` (unreachable), `UNKNOWN` (initial state).

**RequestStatus:** `QUEUED` (waiting in scheduler), `PROCESSING` (executing on backend), `COMPLETED` (success), `FAILED` (error), `CANCELLED` (timeout or user-cancelled).

---

## Deployment

MindRouter is designed for deployment on Linux servers with NVIDIA GPUs. The full deployment guide covers:

- Rocky Linux 8 prerequisites and dependency installation
- SSL/TLS configuration (self-signed and Let's Encrypt)
- Apache reverse proxy setup
- Firewall and SELinux configuration
- Docker Compose production stack
- Database migrations
- GPU sidecar agent deployment
- Node and backend registration
- Verification and ongoing operations

For step-by-step production deployment instructions, see **[../deploy/DEPLOYMENT.md](../deploy/DEPLOYMENT.md)**.

> **Database migrations:** MindRouter uses Alembic for schema migrations. Run `alembic upgrade head` inside the app container after deployment. When writing new migrations, note that MariaDB DDL is non-transactional -- a failed migration leaves partial state requiring manual cleanup. Always drop foreign key constraints before dropping their backing indexes (MariaDB error 1553).

---

## Testing

MindRouter has a comprehensive test suite covering unit, integration, end-to-end, smoke, stress, and accessibility tests.

### Quick Reference

| Command | Description |
|---------|-------------|
| `make test-unit` | Run unit tests (525+ tests) |
| `make test-int` | Integration tests (requires live backends) |
| `make test-e2e` | End-to-end tests |
| `make test-smoke` | Smoke tests (full API surface) |
| `make test-stress` | Load/stress tests |
| `make test-a11y` | WCAG 2.1 accessibility tests |
| `make test-sidecar` | GPU sidecar agent tests |
| `make test-all` | Run all test suites |

For the complete test manifest including all test files, descriptions, and counts, see **[../TESTING.md](../TESTING.md)**.
