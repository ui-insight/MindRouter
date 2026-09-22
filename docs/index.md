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
6. [Users, Groups & Quotas](#users-groups--quotas)
7. [Backend Management](#backend-management)
8. [Scheduling & Fair Share](#scheduling--fair-share)
9. [Translation Layer](#translation-layer)
10. [Telemetry & Monitoring](#telemetry--monitoring)
11. [Chat System](#chat-system)
12. [Voice API](#voice-api)
13. [Blog System](#blog-system)
14. [Configuration Reference](#configuration-reference)
15. [Implementation Notes](#implementation-notes)
16. [Deployment](#deployment)
17. [Testing](#testing)

### Generative Media API references

Dedicated developer references for the generative endpoints (also summarized on
the in-app [`/documentation`](#api-reference) page):

- [Image generation & image-to-image](images-api.md) — `/v1/images/generations`, `/v1/images/edits`
- [Video generation](video-api.md) — `/v1/videos`, keyframe assets, async job model
- [Voice (TTS / STT)](voice-api.md) — `/v1/audio/speech`, `/v1/audio/transcriptions`
- [Media studio integration guide](media-studio-integration.md) — end-to-end recipe: images → video keyframes → clips → stitch → narration (for building a storyboarding / ad-mockup app)

### Operator / admin guides

- [UI branding & theming](branding.md) — rebrand a deployment for one institution: organization name, logos (navbar / footer / login), favicon, and accessible light/dark accent colors, from Admin → Branding
- [Single sign-on configuration](sso-configuration.md) — step-by-step IdP setup for the four SSO providers (Azure AD / Entra ID, Google, generic OIDC including InCommon via CILogon, and native SAML 2.0), plus JIT provisioning and security notes
- [Registered applications](registered-apps.md) — let a first-party app (e.g. VandalChat) exchange its users' Entra `id_token`s for their own MindRouter keys, so its users get per-user quota, telemetry and fair share without ever signing in here. No Entra administrator involvement required

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

### Development Credentials (first admin)

The seed script creates the default groups and **one** user — the initial
administrator. No other accounts are seeded; other users appear on first SSO
login or are created by an admin.

| Username | Password | Group | Scheduler Weight |
|----------|----------|-------|-----------------|
| `admin` | `admin123` (override with `ADMIN_PASSWORD`) | admin | 10 |

It also mints an API key for that user and prints it as one parseable line
(`ADMIN_API_KEY=mr2_…`); the full key is not recoverable afterwards. Optional
env overrides: `ADMIN_PASSWORD` (admin password), `ADMIN_API_KEY` (use a
supplied key instead of minting one), `MINT_ADMIN_KEY=1` (mint another key for
an existing admin).

!!! warning "Development credentials — not for production"
    `admin123` is a documented default and must never be left in place on a
    reachable deployment. Seed production with
    `ADMIN_PASSWORD='<strong secret>' python scripts/seed_dev_data.py`, then
    change the password from the dashboard. This first admin is what lets you
    reach Admin → Branding, SSO configuration, and node registration.

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
| "Missing API key. Provide via 'Authorization: Bearer `<key>`' or 'X-API-Key: `<key>`'" | No API key provided in the request |
| "Invalid API key" | API key not found in the database |
| "API key is {status}" | API key's status is not `active` -- `{status}` is the literal enum value, either `revoked` or `expired` |
| "API key has expired" | API key's `expires_at` timestamp has passed |
| "User account is inactive" | The user associated with the key is disabled |

### Request IDs

Every API response carries an `X-Request-ID` response header. The body `id` field is a **separate** value and is not the same thing:

- **`X-Request-ID` header** -- A request-ID middleware echoes a client-supplied `X-Request-ID` back verbatim, or generates a bare UUID4 if the client did not send one. This value is used for server-side log correlation.
- **Client-supplied IDs never appear in the response body.** Sending `X-Request-ID: my-id` gets `my-id` back in the header only; the body `id` is still minted server-side.
- **Body `id` on inference endpoints** -- `/v1/chat/completions`, `/v1/completions`, and `/anthropic/v1/messages` return the audit row's bare UUID4 (`requests.request_uuid`) -- **not** a prefixed id -- in the non-streaming body. For **streaming**, only Ollama-backed responses carry that UUID in their SSE chunks (`ollama_out.py`); vLLM-backed chunks pass the backend's own `chatcmpl-*` id through (`vllm_out.py`), so a streamed chunk id is not a reliable audit key -- use the `X-Request-ID` header or the non-streaming body id. `/v1/embeddings` returns no `id` field at all. `/v1/rerank` and `/v1/score` pass the backend's own id through. `/v1/responses` returns a real `resp_<hex>` id (the stored-response row), and `/v1/ocr` returns its own `ocr-<hex>` id.
- **`chatcmpl-*`, `cmpl-*`, `emb-*`, `rnk-*`, `scr-*`, `msg_*`, `img-*` prefixes exist only in server logs.** Each route mints one at entry to bind its log context, then the inference service replaces it with the audit UUID before the response is built.
- **Streaming responses return two `x-request-id` headers** -- the route's prefixed log id (e.g. `chatcmpl-…`) plus the middleware's value. Non-streaming responses return only the middleware's.
- **Audit trail** -- Because the body `id` on chat/completion endpoints *is* `requests.request_uuid`, it is the value to search on when tracing a response back to its audit row.

### Error Responses

Error response format varies by API style. MindRouter registers **no `HTTPException` handler** (the only handler in `main.py` catches bare `Exception`), so on most endpoints FastAPI's default `{"detail": ...}` wrapper is applied to every error.

**Most OpenAI endpoints** (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/rerank`, `/v1/score`, `/v1/tokenize`, `/v1/ocr`, `/v1/models`, `/v1/images/*`, `/v1/videos/*`, `/v1/search`) -- errors are wrapped in `detail`, whose value is either a nested OpenAI-style error object or a plain string:

```json
// 404 unknown model -- OpenAI-shaped object nested under "detail"
{"detail": {"error": {"message": "The model 'foo' does not exist", "type": "invalid_request_error", "code": "model_not_found"}}}
```
```json
// 400/401/429/503 -- flat string detail
{"detail": "Invalid JSON body"}
{"detail": "No healthy backends available for this model"}
```

**Responses and Conversations endpoints** (`/v1/responses*`, `/v1/conversations*`) -- the only endpoints that emit the true bare OpenAI envelope, with `param` and `code` always present. These routes build their own `JSONResponse` and re-shape service `HTTPException`s into this form:

```json
{"error": {"message": "The model 'foo' does not exist or you do not have access to it.", "type": "invalid_request_error", "param": null, "code": "model_not_found"}}
```

**Unhandled server errors (500)** -- the global `Exception` handler emits a bare envelope with no `detail` wrapper:

```json
{"error": {"message": "Internal server error", "type": "server_error"}}
```

**`/v1/ocrmd`** returns errors as `text/plain` bodies (the message only, no JSON at all), matching its plain-markdown success response.

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
| 413 | Request body exceeds the reverse proxy's `client_max_body_size` (`50m` in `nginx/nginx.conf`), or an uploaded OCR file exceeds the configured OCR max file size. The `MAX_REQUEST_SIZE` setting exists in `settings.py` but nothing in the application reads it -- the app itself enforces no global body limit. |
| 422 | Request validation failed (malformed JSON, wrong types, missing required fields). Returned by FastAPI's built-in request validation. |
| 429 | Rate limit exceeded. MindRouter does not include a `Retry-After` header on 429 responses. Clients should implement exponential backoff. |
| 500 | Internal server error |
| 503 | "No suitable backend: {reason}" -- Model doesn't support required capability (vision, structured output) |
| 503 | "No backend capacity available (waited Ns)" -- All backends at max concurrent; timed out waiting |
| 503 | "No healthy backends available" -- All backends unhealthy or circuit-broken |

> **Backend pass-through:** Backend 4xx errors (e.g., invalid prompt format) are forwarded directly to the client and are not retried.

> **Parsing errors:** branch on the HTTP status code, not on the body shape. Outside `/v1/responses` and `/v1/conversations`, `detail` may hold either a string or an `{"error": {...}}` object, so an OpenAI SDK that expects `error.message` at the top level will not find it and may surface the error as an opaque unknown-format failure.

> **Model names are exact match, except for aliases.** MindRouter does not support prefix matching or fuzzy matching -- the `model` field must exactly match either a model name or a configured alias as shown in `/v1/models` or `/api/tags`. Aliases are resolved on every inference endpoint (chat completions, completions, embeddings, rerank, score, tokenize, OCR, images, Ollama, Anthropic, Responses, and video). Alias entries are listed alongside real models in `/v1/models` and `/api/tags`, tagged with `"is_alias": true` and `"alias_target": "<real model name>"`. Aliases are administrator-defined (Admin > Models) and stored in the `model_aliases` table, so the set available on any given deployment is entirely a matter of local configuration. A common convention is to name them for an intent rather than a model version -- for example `default-llm`, `default-llm-large`, `default-agent`, `default-coder`, `default-vision`, `default-embedding` -- so that clients can pin a role and let the operator repoint it as models are upgraded.

### OpenAI-Compatible Endpoints

These endpoints accept and return data in the OpenAI API format. Any OpenAI-compatible client or SDK can be pointed at MindRouter by changing the base URL.

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/v1/chat/completions` | API Key | Chat completions (streaming and non-streaming) |
| POST | `/v1/responses` | API Key | OpenAI Responses API (typed items/SSE events; serves agent clients like Codex) |
| POST | `/v1/responses/input_tokens` | API Key | Count input tokens without generating |
| GET | `/v1/responses/{response_id}` | API Key | Retrieve a stored response (`?stream=true` replay is rejected with 400) |
| DELETE | `/v1/responses/{response_id}` | API Key | Delete a stored response and its offloaded artifacts |
| GET | `/v1/responses/{response_id}/input_items` | API Key | List the input items of a stored response (`limit`, `order`, `after`) |
| POST | `/v1/responses/{response_id}/cancel` | API Key | Cancel a background response (background mode is unsupported) |
| * | `/v1/conversations` | API Key | OpenAI Conversations API (conv_* objects, item CRUD) |
| POST | `/v1/completions` | API Key | Text completions (legacy; converts to chat and returns a `chat.completion` object) |
| POST | `/v1/embeddings` | API Key | Generate embeddings |
| POST | `/v1/rerank` | API Key | Rerank documents against a query |
| POST | `/v1/score` | API Key | Score similarity between text pairs |
| POST | `/v1/tokenize` | API Key | Count input tokens for a chat request (exact for vLLM, tiktoken estimate for Ollama) |
| POST | `/v1/ocr` | API Key | OCR images/PDFs/Office docs to markdown or JSON (multipart upload) |
| POST | `/v1/ocrmd` | API Key | Same OCR pipeline as `/v1/ocr`, returns raw `text/markdown` |
| POST | `/v1/search` | API Key | Web search via the configured provider (also served at `/api/search`); `"extra_snippets": true` asks Brave for up to five further excerpts per result, returned as `extra_snippets` on each item |
| POST | `/v1/images/generations` | API Key | Image generation (FLUX; requires per-account enablement) |
| POST | `/v1/images/edits` | API Key | Reference-image edit / img2img (multipart upload) |
| POST | `/v1/videos` | API Key | Submit an async video generation job (202 Accepted) |
| POST | `/v1/videos/assets` | API Key | Upload a keyframe asset for video generation (201 Created) |
| GET | `/v1/videos` | API Key | List your video jobs |
| GET | `/v1/videos/models` | API Key | List available video models |
| GET | `/v1/videos/{video_id}` | API Key | Get a video job's status |
| GET | `/v1/videos/{video_id}/content` | API Key | Stream the rendered MP4 (HTTP range requests supported) |
| DELETE | `/v1/videos/{video_id}` | API Key | Delete a video job and its artifacts |
| GET | `/v1/models` | API Key | List available models (including aliases) |

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
  "id": "19730079-3a32-4888-9c6c-9eac62ad0bcc",
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

> **Reasoning is OFF by default -- this is a gateway policy, not the model's own default.** When a request omits every thinking control, MindRouter forces thinking off on the outbound backend call, so a thinking-capable model such as `qwen/qwen3.6-35b` returns no `reasoning_content` unless you opt in. The policy is the `THINKING_OFF_BY_DEFAULT` setting (default `true`), applied in the inference service on both the streaming and non-streaming paths; it emits `chat_template_kwargs: {"enable_thinking": false}` to vLLM backends and `think: false` to Ollama backends. Models whose name contains `gpt-oss` are exempt -- they use `reasoning_effort` and are left untouched. Set `THINKING_OFF_BY_DEFAULT=false` to restore each backend's per-model launch default. (Ollama models that do not advertise thinking support have the field stripped entirely.)

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

`/v1/completions` accepts the legacy OpenAI completion parameters, but internally converts the request to a chat request before routing:

- **`n`** -- Number of completions to generate (default 1). Carried through the conversion.
- **`suffix`** -- Parsed, then **silently dropped**: the chat conversion does not copy it. No error is returned.
- **`echo`** -- Parsed, then **silently dropped**. The prompt is not echoed back.
- **`best_of`** -- Parsed, then **silently dropped**. No beam search is requested of the backend.

> **Response shape:** `/v1/completions` does **not** return a `text_completion` object with `choices[].text`. Because the request is converted with `to_chat_request()`, the response is a `chat.completion` object whose choices carry `message.role` / `message.content`, exactly like `/v1/chat/completions`. Clients using an OpenAI SDK's legacy completions helper will not find `choices[0].text`. Only the prompt is carried over -- a list-valued `prompt` uses only its first element, and it becomes a single `user` message.

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
      "capabilities": {"multimodal": false, "embeddings": false, "structured_output": true, "thinking": false, "tools": true},
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

Configured aliases are appended to the same `data` list. An alias entry inherits all of its target's metadata (capabilities, backends, context lengths, family) and adds two fields:

```json
{
  "id": "default-llm",
  "object": "model",
  "owned_by": "mindrouter",
  "is_alias": true,
  "alias_target": "openai/gpt-oss-120b",
  "capabilities": {"multimodal": false, "embeddings": false, "structured_output": true, "thinking": true, "tools": true}
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
| POST | `/v1/messages` | API Key | Same handler, also mounted at the OpenAI base path for clients that append `/v1/messages` to a bare host |
| GET | `/anthropic/v1/models` | API Key | Model list served at the Anthropic base path (delegates to `/v1/models`; identical response body) |

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
| Proxy timeout | 600 seconds (10 min, hardcoded in `voice_api.py`) | Generous, so the upload size cap usually binds first -- but a very long file can still exceed it. The bundled nginx allows `proxy_read_timeout 720s`, so this application timeout fires first. |
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
  "version": "2.9.2",
  "timestamp": "2026-03-01T12:00:00+00:00",
  "backends": {"total": 6, "healthy": 5},
  "models": ["gpt-oss-120b", "llama3.2", "qwen3.5"],
  "queue": {"total": 3, "by_user": {}, "by_model": {}, "average_wait_seconds": 0.0},
  "queue_health": {"status": "healthy", "queue_total": 3, "trend": "stable", "stale_jobs": 0, "backend_depths": {"4": 1, "9": 2}},
  "fair_share": {"total_users": 2},
  "active_users": 12
}
```

`version` is the running build (`settings.app_version`, taken from the package version -- `2.9.2` at the time of writing), so `/status` is the quickest way to confirm which release a deployment is serving. `models` lists every model exposed by a currently healthy backend, and `queue_health` carries the scheduler's self-assessment (`status`, `trend`, `stale_jobs`, per-backend depths, and last garbage-collection run).

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
| `mindrouter_queue_size` | Gauge | -- | Current scheduler queue depth. Set on every scrape from the scheduler's stats. |
| `mindrouter_active_backends` | Gauge | -- | Number of healthy backends. Set on every scrape from the registry. |
| `mindrouter_requests_total` | Counter | `endpoint`, `status` | Declared but **never incremented**. Because it is labeled and `.labels()` is never called, it exports **no time series at all** (not a zero). |
| `mindrouter_request_latency_seconds` | Histogram | `endpoint` | Declared but **never observed**. Labeled and never instantiated, so it exports **no time series at all** (not empty buckets). |
| `mindrouter_tokens_total` | Counter | `type` (prompt/completion) | Declared but **never incremented**. Labeled and never instantiated, so it exports **no time series at all** (not a zero). |

> **Only two of the five metrics carry data.** `mindrouter_queue_size` and `mindrouter_active_backends` are refreshed inside the `/metrics` handler itself (`api/health.py`). The other three collectors are declared in the same module but nothing anywhere in the backend calls `.inc()`, `.observe()`, or `.labels()` on them, so they are exported with no samples. Do not build alerts or dashboards on request counts, latency, or token totals from `/metrics` -- use the telemetry API (`/api/admin/telemetry/*`) or the audit log for those.

### Admin API Endpoints

These endpoints are mounted under `/api/admin/`. Authorization is not uniform:

- **Write operations** (`POST`, `PATCH`, `DELETE`) require a group with `is_admin`. Anything else gets 403 `"Admin access required"`.
- **Every `GET`** below uses the read-only gate, which accepts `is_admin` **or** `is_auditor` (`Group.has_admin_read`). An auditor group can therefore read the entire admin surface -- backends, nodes, users, groups, API keys, quota requests, queue, audit log including prompt and response content -- while being unable to change anything. The two exceptions are noted in place: the telemetry routes require full `is_admin`, and the conversations export is authenticated by session cookie only.
- **Some endpoints also accept the dashboard session cookie** in place of an API key, so admin pages can call them by AJAX: backend refresh, the Ollama pull/delete/pull-status routes, `/queue/monitor`, `/top-users`, and every `/api/admin/telemetry/*` route.

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
| DELETE | `/api/admin/users/{id}` | Hard-delete a user: removes their keys, requests/responses, chats, images, videos, and stored responses (including files on disk); blog posts, the email log, and admin-audit entries are kept with the user reference detached. Returns 409 (never deletes partially) if a dependent record cannot be removed. Self-deletion is blocked |
| POST | `/api/admin/users/{id}/reset-password` | Set a new password on a **local** account (`{"new_password": "..."}`, min 8 chars; SSO accounts are rejected — they have no local password) |
| POST | `/api/admin/users/{id}/api-keys` | Create an API key for a user |
| POST | `/api/admin/api-keys/{id}/revoke` | Revoke any user's API key (personal or service). Idempotent |
| DELETE | `/api/admin/api-keys/{id}` | Hard-delete an API key row. Refused (409) when audit rows reference the key — revoke instead; deletion is for keys that were never used |
| GET | `/api/admin/top-users` | Top 10 users by token usage in a window (`window` = `1m`, `1h`, `4h`, `12h`, or `24h`; default `1h`) |

#### Group Management

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/admin/groups` | List all groups with user counts |
| POST | `/api/admin/groups` | Create a new group |
| PATCH | `/api/admin/groups/{id}` | Update group defaults (token budget, RPM, etc.) |
| DELETE | `/api/admin/groups/{id}` | Delete a group. Returns 409 `Cannot delete group with active users. Reassign users first.` while any user is still assigned — reassign every member to another group (Admin → Users, filter by the group, edit each user), then delete |

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
| GET | `/api/admin/queue/monitor` | Queue capacity and completion stats by model and user (`window` = minutes, default 5) |
| GET | `/api/admin/audit/search` | Search audit logs (filter by user, model, status, date, text) |
| GET | `/api/admin/audit/{id}` | Full audit detail including prompt and response content |

#### Conversations

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/admin/conversations/export` | Export conversations as JSON (filters: `search`, `user_id_filter`, `model_filter`, `start_date`, `end_date`, `include_content` -- default `true`) |

> **This route is session-authenticated, not API-key authenticated.** Unlike every other endpoint in this section, `/api/admin/conversations/export` is served by the dashboard router and identifies the caller only from the signed `mindrouter_session` cookie. An `Authorization: Bearer` or `X-API-Key` header is ignored: an unauthenticated request is redirected (302) to `/login`, and a signed-in user whose group lacks admin-or-auditor read is redirected to `/dashboard` -- neither returns 401/403 JSON. The caller's group must satisfy the same admin-or-auditor read check as the rest of the section.
>
> **`format` is not honored here.** The handler hardcodes JSON output; passing `format=csv` has no effect. CSV export is available from the browser at `/admin/conversations/export?format=csv`, which is the same underlying handler reached through the admin dashboard page (also session-authenticated).

#### Telemetry

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/admin/telemetry/overview` | Cluster-wide telemetry (nodes, backends, GPUs) |
| GET | `/api/admin/telemetry/latest` | Lightweight polling endpoint for dashboard |
| GET | `/api/admin/telemetry/backends/{backend_id}/history` | Time-series telemetry for a backend |
| GET | `/api/admin/telemetry/gpus/{gpu_device_id}/history` | Time-series telemetry for a GPU device |
| GET | `/api/admin/telemetry/nodes/{node_id}/history` | Aggregated time-series for a node (all GPUs) |
| GET | `/api/admin/telemetry/export` | Export raw telemetry as JSON or CSV |
| GET | `/api/admin/telemetry/energy/nodes/{node_id}/history` | Power history for one node (`metric` = `server_power` or `gpu_power`; `range`, `resolution`) |
| GET | `/api/admin/telemetry/energy/cluster/history` | Cluster-wide power history (all nodes summed) |
| GET | `/api/admin/telemetry/energy/export` | Export energy/power data as CSV or JSON (`node_id`, `scope` = `all`/`server`/`gpu`, `range`, `resolution`) |

> **Telemetry routes require full admin.** Every route in this subsection uses the admin-or-session gate rather than the read-only one, so `is_auditor` alone is not sufficient -- the group must have `is_admin`. All of them accept either an admin API key or the dashboard session cookie.

---

## Web Dashboard

MindRouter includes a full web dashboard built with Bootstrap 5. All pages extend a common base template with navigation and accessibility features (WCAG 2.1 Level AA).

### Public Pages

| Page | URL | Description |
|------|-----|-------------|
| Cluster Status | `/` | Shows healthy backend count, available models, queue size, and overall cluster status |
| Login | `/login` | Local username/password authentication, plus one sign-in button per configured SSO provider (Azure AD, SAML, generic OIDC, Google) |
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
- **Quota Details** -- The Quota Details card shows the group token budget, tokens used in the current period, and the RPM limit. There is no per-user concurrency limit to display; that field was dropped from the quota table (see [Rate Limiting](#rate-limiting)).
- **API Key Expiration Warnings** -- API keys nearing expiration (7 days or fewer remaining) display a yellow warning countdown. Expired keys show an "Expired" badge. The "Last Used" column shows when each key was last used for authentication.

### Admin Dashboard

The admin dashboard has a persistent sidebar with links to all admin pages:

| Page | URL | Description |
|------|-----|-------------|
| Overview | `/admin` | System metrics overview with pending request badges and health alerts |
| Backends | `/admin/backends` | Backend health, models, enable/disable/drain controls |
| Models | `/admin/models` | Model catalog with capability overrides, metadata editing, context length configuration (see below) |
| Nodes | `/admin/nodes` | GPU node management, sidecar status, take offline/bring online, force drain, active requests |
| Queue | `/admin/queue` | Live queue monitor -- per-model capacity and per-user completion stats, polled from `GET /api/admin/queue/monitor` |
| GPU Metrics | `/admin/metrics` | Real-time GPU utilization, memory, temperature, power charts with time range controls |
| Energy | `/admin/energy` | Server and GPU power history per node and cluster-wide, with CSV/JSON export |
| Users | `/admin/users` | User accounts, group assignment, quota management, masquerade, account-type filter, Create Local User |
| Groups | `/admin/groups` | Group management with token budget, RPM, scheduler weight, admin/auditor flags, API key expiry and count limits |
| API Keys | `/admin/api-keys` | All API keys across users, status filtering, search |
| Requests | `/admin/requests` | Pending API key, service key, and quota increase requests, approve/deny |
| Audit Log | `/admin/audit` | Inference request history with filtering and search |
| Admin Audit | `/admin/admin-audit` | Log of administrative actions -- who changed what, when, and from which IP |
| DLP | `/admin/dlp` | Data-loss-prevention scanner configuration and alert review (filter by severity, scanner, text; acknowledge alerts). Three scanners: regex/keyword (always available), GLiNER NER (local model, downloaded on first scan), and an optional LLM contextual scanner that dispatches directly to a backend serving the model you name — no API key, and scans create no request records and consume no quota. Alert rows store **masked** snippets, never the verbatim match; open the linked request for full context. Alerts are governed by the `DLP Alerts (days)` retention setting (default 0 = keep forever). Requires content capture: see `AUDIT_LOG_ENABLED` below. |
| Conversations | `/admin/conversations` | Browse and search all user conversations, view messages, export |
| Chat Config | `/admin/chat-config` | Configure core models, default model, system prompt, max_tokens, temperature, thinking mode, voice TTS/STT settings |
| Voice Config | `/admin/voice-config` | Configure TTS/STT backend connections, available voices, and API quota token costs |
| Search | `/admin/search-config` | Web search provider (Brave or SearXNG), endpoint, API key, max results, per-search quota token cost, plus a test query |
| OCR | `/admin/ocr-config` | OCR enable toggle, default model, prompts, DPI, page/file limits, chunk size and overlap, chunk concurrency, retries |
| Images | `/admin/images-config` | Image generation enable toggle, default model, default/allowed sizes, steps and guidance defaults, safety judge models and policy, and user access as a **global default plus per-user allow/deny exceptions** (on by default; new SSO- and app-provisioned accounts inherit it) |
| Video | `/admin/video-config` | Video generation enable toggle, default model, allowed sizes and durations, per-user concurrent job cap and storage cap, token cost per second, per-user access grants |
| Blog Posts | `/admin/blog` | Blog/CMS management -- create, edit, publish, delete posts |
| Email | `/admin/email` | Compose and send announcement email to selected users or groups, with recipient count preview and test send |
| Data Retention | `/admin/retention` | Retention policies for request, conversation, telemetry, request-image, stored-response, and Conversations-API data; archive statistics; an archive browser; and a **Purge** tab for immediate per-category deletion older than a chosen cutoff (typed `PURGE` confirmation) |
| Backup & Restore | `/admin/backup` | Export or restore MindRouter configuration (nodes, backends, users, groups, API keys, quotas, models, settings, blog posts). The export is role-scoped since 2.9.9: full admins get a restorable dump including credential material, while read-only auditors get the same configuration with password hashes, API key hashes, and secret settings (SMTP password, provider API keys) nulled. A redacted file is marked in its metadata and is refused by Restore |
| Branding | `/admin/branding` | Institution / organization name, logos (navbar / footer / login), favicon, and accessible light/dark accent colors (see [branding.md](branding.md)) |
| Settings | `/admin/settings` | Site-wide settings: timezone, enforce `num_ctx` override |

> **Auditor groups can read every page in this table.** All of the admin page `GET` handlers gate on the admin-or-auditor read check (`Group.has_admin_read`, i.e. `is_admin OR is_auditor`), so a group with only `is_auditor` can browse the entire admin dashboard. The write actions below, and every mutating admin form post, require `is_admin`.

The Branding page includes an **Institution / organization name** field (config key `branding.org_name`, optional, max 120 characters). Besides labeling the UI, it supplies the institutional label for SSO login buttons -- see [Login button labels](#login-button-labels).

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
| POST | `/admin/users/create` | Create a local account (username/password) from the admin Users page |
| POST | `/admin/users/{id}/reset-password` | Set a new password on a local account (min 8 chars, confirmation required; SSO accounts rejected) |
| POST | `/admin/users/{id}/set-active` | Activate or deactivate an account. Deactivation immediately disables the user's API keys **and** their existing dashboard sessions |
| POST | `/admin/users/{id}/delete` | Hard-delete a user; requires typing the exact username to confirm (validated server-side). Self-deletion is blocked |
| POST | `/admin/api-keys/{id}/revoke` | Revoke any user's API key (personal or service) |
| POST | `/admin/retention` (`action=purge`) | Purge one retention category older than a cutoff; requires typing `PURGE` (validated server-side) |

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

### Account Types and Local Users

The user list (`/admin/users`) shows an **account-type badge** in a "Type" column. The three types are mutually exclusive and are resolved in this precedence order:

| Badge | Condition (`User.account_type`) | Meaning |
|-------|--------------------------------|---------|
| **Admin** | The user's group has `is_admin` | Member of an admin group (checked first, regardless of how they sign in) |
| **SSO** | Not admin, and `azure_oid` or `sso_subject` is set | Signs in through a configured identity provider |
| **Local** | Not admin, and neither identifier is set | Signs in with a local username and password |

The same three values drive the `account_type` query filter on `/admin/users` (`admin`, `sso`, or `local`; any other value is ignored), which composes with the existing `search`, `group_id`, `sort`, and `dir` parameters.

> **The filter is narrower than the badge.** The `account_type=sso` / `account_type=local` SQL filter tests `azure_oid` only, while the badge also accepts a generic `sso_subject`. A non-admin user provisioned through a non-Azure provider (SAML, generic OIDC, Google) therefore renders an **SSO** badge but is returned by `account_type=local`.

Admins can create a local account directly from the user list with the **Create Local User** button, which posts to `POST /admin/users/create` (admin session required; `is_admin`, not auditor). The form takes username, email, password, and group, plus optional full name, college, department, and intended use. The handler rejects a password shorter than 8 characters and a username or email that already exists, provisions the user's quota from the selected group's `rpm_limit`, writes a `user.create` entry to the admin audit log, and redirects to the new user's detail page. Local accounts coexist with SSO accounts -- see [Change Password](#change-password) for how they differ afterward.

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
- Conversation renaming and deletion (individual messages cannot be edited or deleted)

---

## Users, Groups & Quotas

### Groups

MindRouter uses database-driven **groups** to control permissions, quotas, and scheduling priority. Each user belongs to exactly one group. Groups replace the earlier role-based system with more flexible, admin-configurable settings.

Each group has the following fields:

| Field | Default | Description |
|-------|---------|-------------|
| `name` | -- | Unique identifier, lowercase (e.g., `students`, `staff`, `faculty`, `admin`) |
| `display_name` | -- | Human-readable name shown in the UI |
| `description` | `null` | Optional description |
| `token_budget` | `100000` | Token budget applied to every member of the group. `0` means unlimited |
| `rpm_limit` | `30` | Requests-per-minute limit, copied into each new member's quota record |
| `scheduler_weight` | `1` | Scheduling priority weight for fair-share scheduling |
| `is_admin` | `false` | Full admin access -- required for every mutating admin action |
| `is_auditor` | `false` | Read-only admin access. `is_admin OR is_auditor` is exposed as `Group.has_admin_read` and gates all admin `GET` routes |
| `api_key_expiry_days` | `45` | Lifetime applied to API keys created by members |
| `max_api_keys` | `16` | Maximum number of active API keys a member may hold |

> There is no `max_concurrent` field on a group. The column existed until migration `056`, which dropped `max_concurrent` from `groups`, `quotas`, and `api_keys`; per-user concurrency limits are not part of the data model any more. The unrelated per-**backend** `max_concurrent` (how many in-flight requests a single inference endpoint accepts) is still real -- see [Registration](#registration).

Groups are managed via the admin dashboard (`/admin/groups`) or the admin API (`/api/admin/groups`).

### Default Quotas by Group

Migration `009` creates **seven** groups when the schema is first built, and migration `041` adds an eighth (`auditor`) alongside the `is_auditor` column. `scripts/seed_dev_data.py` re-creates the same seven for a development database, skipping any that already exist:

| Group | Display name | Token budget | RPM | Scheduler weight | `is_admin` | `is_auditor` |
|-------|--------------|--------------|-----|------------------|------------|--------------|
| `students` | Students | 100,000 | 30 | 1 | No | No |
| `staff` | Staff | 500,000 | 60 | 2 | No | No |
| `faculty` | Faculty | 1,000,000 | 120 | 3 | No | No |
| `researchers` | Researchers | 1,000,000 | 120 | 3 | No | No |
| `admin` | Admin | 10,000,000 | 1,000 | 10 | **Yes** | No |
| `nerds` | Nerds | 500,000 | 60 | 2 | No | No |
| `other` | Other | 100,000 | 30 | 1 | No | No |
| `auditor` | Auditor | 100,000 | 30 | 1 | No | **Yes** |

These names are only a starting point -- they are ordinary rows, and a deployment is expected to rename, delete, or add to them. The token budget is a rolling-window budget, not a calendar-month one (see [Quota System](#quota-system)).

Seeded groups get `api_key_expiry_days = 45` and `max_api_keys = 8` (the column defaults added by migration `020`); groups created afterwards through the admin UI or API get `max_api_keys = 16` from the model default.

Group settings are editable through the admin UI or API. The per-role environment variables (e.g., `DEFAULT_TOKEN_BUDGET_STUDENT`, `SCHEDULER_WEIGHT_STAFF`) are **deprecated** and serve only as fallbacks for environments that have not migrated to database-driven groups.

> **Per-user weight override:** Admins can override an individual user's scheduler weight via the user detail page (`/admin/users/{id}`). When set, the user's `weight_override` takes precedence over their group's `scheduler_weight`. An empty or null `weight_override` means the user inherits the group default. This allows fine-grained fair-share tuning for specific users without changing group-wide settings.

### Change Password

Users with local (non-SSO) accounts can change their password from the user dashboard (`/dashboard`). The form requires the current password, a new password (minimum 8 characters), and password confirmation. Accounts provisioned through SSO have no local password, so `POST /dashboard/change-password` returns early for them; there is no admin UI or API for adding a local password to an existing SSO account.

An admin can set a new password on a **local** account without knowing the current one: `POST /api/admin/users/{id}/reset-password` (`{"new_password": "..."}`, minimum 8 characters), or **Reset Password** under Account Management on the user's admin detail page. SSO accounts are rejected — they have no local password to reset.

### API Key Lifecycle

1. **Generation** -- Keys use the format `mr2_<random_urlsafe_base64>` (48+ characters total).
2. **Storage** -- The raw key is shown once at creation. Only the Argon2 hash and a prefix (`mr2_<first 8 chars>`) are stored in the database.
3. **Verification** -- Lookup by prefix (fast), then full Argon2 hash verification.
4. **Expiration** -- Optional `expires_at` timestamp.
5. **Revocation** -- Keys can be revoked (soft-delete) without deleting the audit trail.
6. **Usage tracking** -- `last_used_at` and `usage_count` updated atomically on each request.

> API keys can become unusable through two distinct mechanisms: **expiration** (automatic -- the key's `expires_at` timestamp has passed) or **revocation** (the key's status is set to `REVOKED`). Personal keys are revoked by their owner from the dashboard (one step, immediate) or by an admin (`POST /api/admin/api-keys/{id}/revoke` or the Revoke button on Admin → API Keys). Service keys use the two-step flow: the owner submits a revocation request, an admin executes it. In all cases, authentication fails from the next request and the key's audit trail is preserved. Deactivating a user (`is_active=false`) immediately disables all of that user's keys and their dashboard sessions.

### Quota System

Each user has a quota record holding:

- **`tokens_used`** -- Tokens consumed in the current rolling period. Incremented on each completed request (prompt + completion tokens).
- **`lifetime_tokens_used`** -- All-time total, never reset.
- **`budget_period_start` / `budget_period_days`** -- The rolling window (see below).
- **`rpm_limit`** -- Maximum requests per minute for this user.
- **`weight_override`** -- Optional per-user scheduler weight (null = inherit the group's).

The **token budget itself lives on the group**, not on the quota row: enforcement compares `quota.tokens_used` against `user.group.token_budget`, and a budget of `0` (or no group) means unlimited. There is no per-user concurrency field -- migration `056` dropped `max_concurrent` from `quotas`.

When a quota is exceeded, the request is rejected with HTTP 429.

> **Rolling budget period:** Token budgets use a rolling window, not calendar months. Each user's quota tracks `budget_period_start` and `budget_period_days` (default: 30). When the current time exceeds the period end, `tokens_used` resets to zero and the period start advances. This means budget resets are per-user, not system-wide.

### Rate Limiting

**RPM limiting is enforced.** It runs inside the same pre-flight check as the token budget (`InferenceService._check_quota`, and the equivalent helper in the voice API), not as middleware -- so it applies to requests that reach an inference or voice endpoint, and endpoints that never call the check are not rate limited.

The limit that applies is the API key's `rpm_limit` when the key sets one, otherwise the user's quota `rpm_limit`. A limit of `0` disables the check. Counting is done in Redis (`check_rpm`) with an `INCR` plus a 60-second `EXPIRE` on a per-user key, so the window is shared across all application workers. Over-limit requests are rejected with **HTTP 429** and the message `Rate limit exceeded: N requests per minute (current: M)`; exceeding the token budget returns **HTTP 429** with `Token quota exceeded`.

> **Redis fail-open.** If Redis is unavailable, or the Redis call raises, `check_rpm` returns "allowed" -- RPM limiting silently stops being enforced rather than blocking traffic. Token budget enforcement is unaffected, since it reads from the database.

> **Concurrency limiting per user or group does not exist.** Migration `056` dropped `max_concurrent` from `groups`, `quotas`, and `api_keys`, and `backend/app/security/rate_limits.py` is now only a docstring pointing at the Redis implementation. The surviving concurrency controls are per-backend (`backends.max_concurrent`, used by the scheduler) and, for video, a per-user concurrent **job** cap configured at `/admin/video-config`.

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
- A **Backend** is an inference server running on a node -- an Ollama, vLLM, diffusion (image), or video-generation instance.
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
| **Ollama** (`ollama`) | `GET /api/tags` | `GET /api/tags` + `POST /api/show` (model details) + `POST /api/ps` (loaded models) | Sidecar agent |
| **vLLM** (`vllm`) | `GET /health` (fallback: `GET /v1/models`) | `GET /v1/models` | `GET /metrics` (Prometheus format) |
| **Diffusion** (`diffusion`) | `GET /health` (fallback: `GET /v1/models`) | `GET /v1/models` | `GET /metrics` when the server exposes it; otherwise sidecar GPU data only |
| **Video** (`video`) | `GET /health` (fallback: `GET /v1/models`) | `GET /v1/models` | `GET /metrics` when the server exposes it; otherwise sidecar GPU data only |
| **TTS** (`tts`) | `GET /health` (fallback: `GET /v1/models`) | `GET /v1/models` | Sidecar GPU data only |
| **STT** (`stt`) | `GET /health` (fallback: `GET /v1/models`) | `GET /v1/models` | Sidecar GPU data only |

The `BackendEngine` enum has exactly these six values. The registry builds an `OllamaAdapter` for `ollama` backends and a `VLLMAdapter` for every other engine, so diffusion, video and voice servers must expose the same OpenAI-compatible `/health` and `/v1/models` surface as vLLM. Discovery then assigns modality by engine before looking at the model name: every model found on a `diffusion` backend becomes `IMAGE_GENERATION`, every model on a `video` backend becomes `VIDEO_GENERATION`, every model on a `tts` backend becomes `TTS`, and every model on an `stt` backend becomes `STT`.

> **The model catalog lists text models only.** `GET /v1/models`, `GET /api/tags` and
> `GET /anthropic/v1/models` publish chat, completion, multimodal, embedding and
> reranking models. Image, video and speech models are excluded as of 2.9.10 —
> listing them there advertises them as LLMs, and a client that selects one for
> `/v1/chat/completions` gets a confusing failure. Each has its own discovery
> endpoint: `GET /v1/images/models`, `GET /videos/models`, and
> `GET /v1/audio/voices` for TTS voices. Aliases follow their target, so an alias
> pointing at a non-text model is likewise absent.


> **Registering a voice backend (2.9.10+).** `tts` and `stt` were added by migration 072 so speech services join the registry like any other backend, gaining health checks, circuit breakers and load balancing. Both reference implementations already speak the required surface — Kokoro advertises `kokoro`/`tts-1`/`tts-1-hd` and speaches advertises its whisper model — so no adapter work is needed. Register through the admin API (`POST /api/admin/backends/register` with `"engine": "tts"` or `"stt"`); the Admin → Backends form does not yet offer these engines. Until a voice backend is registered, requests continue to use the single `voice.tts_url` / `voice.stt_url` values on Admin → Voice Config, which remain the fallback. The upstream credential still comes from `voice.tts_api_key` / `voice.stt_api_key` — `backends` has no per-backend credential column.

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
- Chat retention is governed by the runtime-editable policies at **Admin → Retention** (`retention.chat.tier1_days`, seeded default 90 days in the live DB, then archived if an archive DB is configured; archive copies are purged after `retention.chat.tier2_days`, seeded default 730 days). Setting a tier to 0 disables that sweep. The retention cycle runs hourly. Users can also delete their own conversations at any time.

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
- System artifact writes have no application-level size cap (`ARTIFACT_MAX_SIZE_MB` was removed in 2.9.5 — nothing read it); request-body limits come from the reverse proxy
- Artifact storage path: `/data/artifacts` (configurable via `ARTIFACT_STORAGE_PATH`)
- Artifact DB rows are deleted together with their parent request by the retention cycle's **Requests** policy (there is no separate artifact retention clock)

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
- TTS enable/disable toggle (gates the chat UI **and** `POST /v1/audio/speech`), provider, default voice, playback speed (chat UI only)
- STT enable/disable toggle (gates the chat UI **and** `POST /v1/audio/transcriptions`)

The backend connection settings (URLs, API keys) are shared between the chat UI and the Voice API. So are the **enable toggles**: `POST /v1/audio/speech` reads `voice.tts_enabled` and `POST /v1/audio/transcriptions` reads `voice.stt_enabled` -- the same keys the Chat Config page writes -- and each returns **HTTP 404** (`TTS is not enabled` / `STT is not enabled`) when its toggle is off. Turning TTS or STT off for the chat UI therefore turns the corresponding public endpoint off as well. The remaining chat-specific settings (provider, default voice, playback speed) affect only the chat interface; API callers pass `voice` and `speed` in the request body.

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
- **50 MB** maximum upload size (the reverse proxy's `client_max_body_size`, set to `50m` in the bundled `nginx/nginx.conf`)
- **600-second** (10-minute) proxy timeout to the upstream Whisper service, hardcoded in `voice_api.py`. The bundled nginx sets `proxy_read_timeout 720s`, so the application timeout is the one that fires first.
- The entire audio file is buffered in memory before forwarding
- No server-side audio segmentation or chunking

The 50 MB upload cap, not the timeout, is what usually binds: 50 MB is roughly 52 minutes of 128 kbps MP3 (a one-hour file at that bitrate is about 57 MB), and higher bitrates or uncompressed WAV reach it far sooner. Within that size, a request survives up to 10 minutes of upstream transcription. Clients should still split long audio into chunks -- it keeps each upload well under both ceilings, avoids buffering a large file in RAM, and gives per-segment error recovery. See [STT Limitations & Long Audio](#stt-limitations--long-audio) in the API Reference for code examples.

**Quota model:** Both TTS and STT use a flat per-request token cost regardless of input size. Admins can adjust the cost via the Voice API Config page.

### DB Config Keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `voice.tts_url` | string | (none) | TTS service base URL |
| `voice.tts_api_key` | string | (none) | TTS service API key |
| `voice.stt_url` | string | (none) | STT service base URL |
| `voice.stt_api_key` | string | (none) | STT service API key |
| `voice.stt_model` | string | `"whisper-large-v3-turbo"` | Default STT model |
| `voice.tts_enabled` | boolean | `false` | Enable TTS -- gates both the chat UI and `POST /v1/audio/speech` (404 when false) |
| `voice.tts_provider` | string | `"kokoro"` | Chat TTS provider (`kokoro` or `openedai`) |
| `voice.tts_voice` | string | `"af_heart"` | Default voice for chat TTS |
| `voice.tts_speed` | float | `1.0` | Default playback speed for chat TTS |
| `voice.stt_enabled` | boolean | `false` | Enable STT -- gates both the chat UI and `POST /v1/audio/transcriptions` (404 when false) |
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

All settings are loaded from environment variables or `.env` / `.env.prod` files. Variable names are case-insensitive. Inside Docker, whether a variable actually reaches the process depends on which Compose file you run -- see [Docker Compose env var passthrough](#docker-compose-env-var-passthrough).

Per-model and per-feature runtime tunables (chat defaults, voice, video presets and quotas, branding, retention windows) are **not** environment variables -- they live as rows in the `app_config` table and are edited from the Admin Dashboard. See [Runtime AppConfig](#runtime-appconfig-database-driven).

### Application

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `APP_NAME` | str | `MindRouter` | Application name |
| `APP_VERSION` | str | (from `pyproject.toml`) | Application version |
| `APP_BASE_URL` | str | `https://your-domain.example.com` (placeholder -- **no usable default**) | Public HTTPS origin of *this* deployment, scheme + host, no trailing path. **Must be set per deployment.** SSO redirect URIs (Azure/Google/OIDC) and SAML `Destination`/`Recipient` validation are derived from it rather than from request headers, so a wrong value sends users to another host and fails as `redirect_uri_mismatch`. `docker-compose.yml` passes it through with an *empty* default (`${APP_BASE_URL:-}`), so leaving it out of `.env` reaches the app as an empty string and the code falls back to the request's own scheme and `Host` header |
| `MCP_SERVER_URL` | str | `http://127.0.0.1:8001` | Upstream address of the standalone single-worker MCP server that the mounted `/mcp/*` proxy forwards to |
| `RUN_MIGRATIONS` | bool | `false` | When true, the app runs `alembic upgrade head` at startup before serving, so a fresh or unmigrated database does not crash-loop. Opt-in; run single-worker on first boot |
| `DEBUG` | bool | `false` | Enable debug mode |
| `RELOAD` | bool | `false` | Auto-reload on code changes (development) |

### Database

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DATABASE_URL` | str | `mysql+pymysql://...` | MariaDB/MySQL connection string |
| `DATABASE_POOL_SIZE` | int | `30` | Connection pool size |
| `DATABASE_MAX_OVERFLOW` | int | `20` | Max overflow connections beyond pool |
| `DATABASE_ECHO` | bool | `false` | Log SQL queries |
| `ARCHIVE_DATABASE_URL` | str | `None` | Optional second database for tiered retention. Archival is skipped entirely when unset |

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

### Single Sign-On (SSO)

MindRouter supports four SSO providers: **Azure AD / Entra ID**, **Google**, **generic OIDC** (Okta, Keycloak, Auth0, CILogon, or any spec-compliant IdP), and **native SAML 2.0** (Shibboleth, ADFS).

- **Providers are enabled purely by environment variables.** There is no admin-UI toggle -- a provider is on when its required variables are set and off when they are unset.
- **Any subset can be enabled simultaneously.** The login page renders **one button per enabled provider**, in this fixed order: Azure AD, SAML, generic OIDC, Google.
- **Local username/password accounts are always available.** SSO never disables the local login form (it is collapsed behind a "Sign in with a local account" toggle when SSO buttons are present). With no provider configured, `/login` is just the local login form.
- **Restart after any change.** Settings are process-level and `lru_cache`d per worker; OIDC discovery documents and SAML IdP metadata are cached in-process for 1 hour. Recreate the container (`docker compose up -d`) after editing any SSO variable.
- **`APP_BASE_URL` must be your public HTTPS origin.** OIDC redirect URIs and the SAML `Destination`/`Recipient` checks are derived from it rather than from request headers; if it is left blank the code falls back to the request's own scheme and `Host` header -- and the two paths differ, since the OIDC driver honors `X-Forwarded-Proto` while the SAML adapter does not, so behind a TLS-terminating proxy a blank value yields `http://` SAML URLs. Keep it set.
- **SSO variables must actually reach the container**, and how that works differs between the two shipped stacks: `docker-compose.yml` forwards only the variables named in its `environment:` block (all SSO variables are already listed there), while `docker-compose.prod.yml` uses `env_file: .env.prod` and forwards every key in that file. See [Docker Compose env var passthrough](#docker-compose-env-var-passthrough). Secrets belong in the host `.env` / `.env.prod`, never in the repository.
- **Database:** migration `068` adds `users.sso_provider` and `users.sso_subject`. Run `alembic upgrade head` after upgrading.

For step-by-step IdP-side setup (Azure portal, Google Cloud console, Okta/Keycloak, Shibboleth, CILogon registration), see **[sso-configuration.md](sso-configuration.md)**.

#### Azure AD / Entra ID

Enabled when `AZURE_AD_CLIENT_ID` **and** `AZURE_AD_TENANT_ID` are set. Routes: `GET /login/azure`, `GET /login/azure/authorized`.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `AZURE_AD_CLIENT_ID` | str | `None` | Azure AD application (client) ID |
| `AZURE_AD_CLIENT_SECRET` | str | `None` | Azure AD client secret |
| `AZURE_AD_TENANT_ID` | str | `None` | Azure AD tenant ID |
| `AZURE_AD_REDIRECT_URI` | str | `https://your-domain.example.com/login/azure/authorized` | OAuth2 redirect URI -- an absolute URL that must match the app registration exactly (**not** derived from `APP_BASE_URL`) |
| `AZURE_AD_DEFAULT_GROUP` | str | `other` | Default group for new Azure AD users |

Azure AD keeps its own legacy identity column (`users.azure_oid`) and its own driver; the other three providers share `users.sso_provider` / `users.sso_subject`. The driver is separate, but its **account-linking behavior is not** -- it applies the same unclaimed-account-only email guard as the shared SSO path (see [JIT Provisioning and Account Linking](#jit-provisioning-and-account-linking)). Genuinely unclaimed accounts (no `azure_oid`, no `sso_provider` -- a local password account, for instance) still link exactly as before; the one operator-visible change is that an Azure login whose email matches an account carrying a *different* `azure_oid` now fails instead of silently rebinding. Entra object IDs are stable, so this is rare.

**Group mapping via job title** is unique to Azure AD. For a brand-new user, the Microsoft Graph `jobTitle` claim is matched case-insensitively: if `jobTitle` contains "student", the user is assigned to the `students` group; if it contains "faculty" or "professor", the user is assigned to the `faculty` group; if it contains "staff", the user is assigned to the `staff` group. If `jobTitle` is missing or does not match any of these substrings, the user falls back to the group specified by `AZURE_AD_DEFAULT_GROUP`.

#### Google

Enabled when `GOOGLE_SSO_CLIENT_ID` **and** `GOOGLE_SSO_CLIENT_SECRET` are set. Routes: `GET /login/google`, `GET /login/google/authorized`. The button is always labeled "Sign in with Google".

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `GOOGLE_SSO_CLIENT_ID` | str | `None` | Google OAuth client ID |
| `GOOGLE_SSO_CLIENT_SECRET` | str | `None` | Google OAuth client secret |
| `GOOGLE_SSO_REDIRECT_URI` | str | `<APP_BASE_URL>/login/google/authorized` | Optional override for the callback URL |
| `GOOGLE_SSO_HOSTED_DOMAIN` | str | `None` | Optional Google Workspace domain restriction |
| `GOOGLE_SSO_DEFAULT_GROUP` | str | `other` | Default group for new Google users |

`GOOGLE_SSO_HOSTED_DOMAIN` both passes `hd=<domain>` on the authorization request (Google pre-filters the account picker) **and** rejects any callback profile whose `hd` claim does not match, so the restriction is enforced server-side rather than cosmetically.

#### Generic OIDC

Enabled when `OIDC_SSO_ISSUER` **and** `OIDC_SSO_CLIENT_ID` **and** `OIDC_SSO_CLIENT_SECRET` are set. Routes: `GET /login/oidc`, `GET /login/oidc/authorized`.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `OIDC_SSO_ISSUER` | str | `None` | Issuer base URL; endpoints are read from `<issuer>/.well-known/openid-configuration` |
| `OIDC_SSO_CLIENT_ID` | str | `None` | OIDC client ID |
| `OIDC_SSO_CLIENT_SECRET` | str | `None` | OIDC client secret |
| `OIDC_SSO_REDIRECT_URI` | str | `<APP_BASE_URL>/login/oidc/authorized` | Optional override for the callback URL |
| `OIDC_SSO_DISPLAY_NAME` | str | `SSO` | Login button reads "Sign in with `<this>`" |
| `OIDC_SSO_SCOPES` | str | `openid profile email` | Requested scopes |
| `OIDC_SSO_DEFAULT_GROUP` | str | `other` | Default group for new OIDC users |

Discovery documents are fetched at first login and cached in-process for 1 hour -- authorize/token URLs are never configured by hand. Claims used: `sub` (stable subject), `email` (required), `name`, `preferred_username` (username hint). The IdP must publish a `userinfo` endpoint.

> **InCommon via CILogon (recommended InCommon path):** CILogon is an OIDC gateway fronting the entire InCommon federation, so **one** client registration replaces per-campus SAML metadata exchange. Configure the *generic OIDC* provider with `OIDC_SSO_ISSUER=https://cilogon.org`, a client registered at cilogon.org (callback `<APP_BASE_URL>/login/oidc/authorized`), and `OIDC_SSO_DISPLAY_NAME=InCommon`. Users pick their home institution on the CILogon page. Native SAML remains available for organizations that require direct federation with their own Shibboleth IdP.

#### SAML 2.0

Enabled when `SAML_SP_ENTITY_ID` is set **and** either `SAML_IDP_METADATA_URL` **or** all three of `SAML_IDP_ENTITY_ID` / `SAML_IDP_SSO_URL` / `SAML_IDP_X509_CERT` are set.

| Route | Purpose |
|-------|---------|
| `GET /login/saml` | SP-initiated AuthnRequest redirect to the IdP |
| `POST /login/saml/acs` | Assertion Consumer Service |
| `GET /saml/metadata` | SP metadata XML -- **this is the URL an admin registers with the IdP or federation**; returns 404 when SAML is not configured. A missing `python3-saml` reports 501 ("SAML support is not installed") *only* in the explicit entity-ID / SSO-URL / cert configuration -- with `SAML_IDP_METADATA_URL` set, the parser import fails first and the endpoint answers 404 instead, so the status code is not a reliable way to tell the two failure modes apart; check the logs |

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SAML_SP_ENTITY_ID` | str | `None` | SP entity ID (typically the metadata URL) |
| `SAML_SP_ACS_URL` | str | `<APP_BASE_URL>/login/saml/acs` | Optional override for the ACS URL |
| `SAML_IDP_METADATA_URL` | str | `None` | IdP metadata URL (**must be https**) |
| `SAML_IDP_ENTITY_ID` | str | `None` | IdP entity ID (explicit style, no metadata URL) |
| `SAML_IDP_SSO_URL` | str | `None` | IdP SSO endpoint (explicit style) |
| `SAML_IDP_X509_CERT` | str | `None` | IdP signing certificate, base64, no PEM headers (explicit style) |
| `SAML_DISPLAY_NAME` | str | `SSO` | Login button label |
| `SAML_DEFAULT_GROUP` | str | `other` | Default group for new SAML users |
| `SAML_ATTR_EMAIL` | str | `mail` | Attribute mapped to email |
| `SAML_ATTR_NAME` | str | `displayName` | Attribute mapped to display name |
| `SAML_ATTR_USERNAME` | str | `eduPersonPrincipalName` | Attribute mapped to the username hint |

The attribute-mapping defaults follow eduPerson / InCommon conventions.

**What the IdP must sign:** MindRouter's SP settings are `wantAssertionsSigned: true` and `wantMessagesSigned: false` -- the **assertion** must be signed; a message-level (SAML `Response` element) signature is not required.

**SAML requires HTTPS.** The IdP returns the assertion by *cross-site* HTTP-POST to the ACS, and browsers do not send `SameSite=Lax` cookies on a cross-site POST -- so the `saml_request_id` AuthnRequest-ID cookie is issued with `SameSite=None; Secure`. A `Secure` cookie is not stored over plain `http`, so **SAML cannot be exercised over a plain-http development URL**: the ACS would see no cookie and reject every login as unsolicited. Serve the dashboard over TLS (and point `APP_BASE_URL` at the `https://` origin) before enabling SAML.

**The IdP must release a persistent (or otherwise stable) NameID.** MindRouter requests `urn:oasis:names:tc:SAML:2.0:nameid-format:persistent` in its AuthnRequest but does **not** verify the format that comes back. With a transient or rotating NameID, a user signs in successfully once -- stamping `sso_provider` / `sso_subject` on the account -- and is then **permanently refused**: every later login misses on the new subject, falls back to the email match, finds `sso_provider` already set, and is rejected as `sso_email_link_refused`. Recovering such an account requires editing the `users` table directly; there is no admin UI or API for clearing a stale identity.

**Installing with SAML support:** the shipped Docker image already includes everything (the `Dockerfile` installs `libxmlsec1-dev` + `libxmlsec1-openssl` and runs `pip install -e .[saml]`; `xmlsec` and `lxml` resolve as prebuilt wheels on both amd64 and arm64, so nothing compiles). On bare metal, SAML is an optional extra -- install `libxmlsec1-dev` first, then `pip install -e .[saml]`. Without the extra, Azure/Google/OIDC work normally and the SAML routes degrade with a clear "SAML support is not installed" error rather than crashing.

#### Login button labels

Button labels tie into **Admin → Branding → "Institution / organization name"** (`branding.org_name`). Azure AD is treated as the primary/institutional provider: its button reads "Sign in with `<org name>`", falling back to "Sign in with SSO" when no org name is set. SAML and generic OIDC use the org name only when they are the primary provider *and* their `*_DISPLAY_NAME` is still at its default; otherwise they use their own display name. Google is always "Google". The helper text reads "Use your `<org name or 'organization'>` credentials to sign in."

#### JIT Provisioning and Account Linking

All providers share the same semantics. On login, MindRouter looks up `(provider, subject)` first -- the stable IdP identifier (OIDC `sub`, SAML persistent NameID, Azure object ID) -- then falls back to a lowercased email match.

- **Email matches adopt unclaimed accounts only.** If the matched account already carries *any* identity -- an `azure_oid`, or an `sso_provider` of any value -- the login is **refused** and logged as `sso_email_link_refused`. The test is on any value, not on a *different* one, so the **same** provider presenting a **new** subject is refused too. Email is an IdP-supplied attribute, not proof of ownership -- without this rule, any enabled IdP could assert an existing user's address (including an admin's) and inherit the account. The guard applies to the Azure driver as well as the shared Google / OIDC / SAML path.
- **A rotating IdP subject locks a user out after their first login.** Because the guard fires on any set `sso_provider`, an IdP that issues transient or rotating subjects (a SAML transient NameID, for example) admits a user once and refuses every attempt afterwards: the new subject misses, the email match hits, and the account is already claimed. **Configure the IdP to release a persistent (or otherwise stable) NameID / subject.**
- **Clearing a stale identity requires direct database access.** There is no admin UI or API for resetting `azure_oid` / `sso_provider` / `sso_subject`, so moving a user between providers -- or recovering an account stranded by a rotated subject -- means editing the `users` table by hand.
- **Unclaimed accounts -- notably local password accounts -- are linked**, and the local password is retained, so both login methods keep working. Display name is refreshed from the IdP on every login; **department and college are refreshed for Azure AD only** -- the OIDC/Google and SAML profile mappers never populate those fields, so they stay empty for those providers.
- **New users** are created with no password hash -- the account is SSO-only, cannot use the local login form, and there is currently no admin UI or API for adding a local password to it (that would mean a separate local account or direct database access) -- and land in the provider's `*_DEFAULT_GROUP`. **That group must already exist** -- create it on the admin Groups page first. Quota is seeded from the group's defaults. The username is the local part of the username hint (ePPN / `preferred_username`) or email, with `_<first 8 chars of subject>` appended on collision.
- **Deactivated accounts** (`is_active = false`) are refused at login regardless of provider.

#### SSO Security Posture

- **Account linking is restricted to unclaimed accounts** (see above) -- the single most important protection against IdP-asserted account takeover.
- **Emails are rejected when the IdP sends `email_verified` and it is not true** (OIDC/Google); IdPs that omit the claim entirely are trusted. The claim is normalized, so string forms like `"false"` or `"0"` cannot fail open.
- **OIDC CSRF protection** uses a signed, timed state token (10-minute lifetime) round-tripped through an HttpOnly cookie and validated on every callback.
- **SAML is SP-initiated only.** The ACS requires a signed `saml_request_id` cookie issued by `/login/saml` **and** requires the response's `InResponseTo` to echo that AuthnRequest ID. Stated plainly: **IdP-initiated login is not supported** -- launching MindRouter from a campus app-portal tile will not work; users must start at the MindRouter login page. (This is enforced in MindRouter because python3-saml has no unsolicited-response setting and skips its own `InResponseTo` comparison when the attribute is absent.)
- **SAML requires HTTPS.** That `saml_request_id` cookie is set `SameSite=None; Secure`, because the IdP delivers the assertion by cross-site HTTP-POST to the ACS and browsers withhold `SameSite=Lax` cookies on cross-site POSTs. A `Secure` cookie is never stored over plain `http`, so **SAML cannot be exercised over a plain-http dev URL** -- the ACS would see no cookie and reject every login as unsolicited.
- **A stable IdP subject is a security requirement, not just an ergonomic one.** MindRouter requests a persistent NameID but does not verify the returned format; with a transient or rotating subject a user is refused on every login after their first, and only direct database access can clear the stale identity.
- **SAML assertions must be signed** (strict mode), and `rejectDeprecatedAlgorithm` blocks SHA-1 signatures.
- **SAML IdP metadata must be served over HTTPS** -- it carries the signing certificate, the only trust anchor. A plain-`http` `SAML_IDP_METADATA_URL` disables the provider. For a stronger anchor, pin the certificate locally with the explicit `SAML_IDP_ENTITY_ID` / `SAML_IDP_SSO_URL` / `SAML_IDP_X509_CERT` trio instead of fetching metadata.
- **Public URLs are derived from `APP_BASE_URL` rather than from request headers** -- keep `APP_BASE_URL` set, and a spoofed `X-Forwarded-Host` / `X-Forwarded-Proto` cannot influence OIDC redirect URIs or SAML `Destination`/`Recipient` validation. If `APP_BASE_URL` is blank the code falls back to the request's own scheme and `Host` header (the OIDC path consults `X-Forwarded-Proto`; the SAML adapter does not), which is exactly why leaving it set matters.
- **`SECRET_KEY` underpins the SSO anti-forgery cookies.** The signed OIDC state cookie and the SAML `saml_request_id` cookie are both signed with it, so a weak or leaked `SECRET_KEY` weakens those protections; rotating it invalidates any login already in flight.

#### Scope and Limits

- One IdP per provider type -- a single generic-OIDC issuer and a single SAML IdP.
- SAML: SP-initiated only, and no Single Logout (SLO). Encrypted assertions and signed AuthnRequests require an **SP key pair** — set `SAML_SP_X509_CERT` / `SAML_SP_PRIVATE_KEY` (a self-signed, long-lived pair; *not* your TLS certificate) plus `SAML_WANT_ASSERTIONS_ENCRYPTED` / `SAML_AUTHN_REQUESTS_SIGNED`. With a cert configured, `/saml/metadata` publishes a `<KeyDescriptor>`. Without one, assertion encryption must be disabled for this SP at the IdP — stock Shibboleth encrypts by default, and a mismatch fails *at the IdP* with nothing in MindRouter's logs. **At an InCommon member institution, CILogon (OIDC) is the simpler path** — same campus IdP, no metadata exchange or key material. See [SSO configuration](sso-configuration.md).
- No SCIM / directory sync and no group-claim-driven authorization. Group assignment comes from the provider's `*_DEFAULT_GROUP` (except Azure's `jobTitle` mapping); ongoing group and role changes are managed in MindRouter, not pushed from the IdP.

### Artifact Storage

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ARTIFACT_STORAGE_PATH` | str | `/data/artifacts` | File storage directory. Also the root for audit-offloaded request images and Responses API image offload |

> `ARTIFACT_MAX_SIZE_MB` and `ARTIFACT_RETENTION_DAYS` were removed in 2.9.5 —
> they were never read by the application. Reaping is driven by the Admin →
> Retention policies (`retention.request_images_days`, default 180;
> `retention.responses_store_days`, default 30; artifacts die with their
> parent request under the requests policy).

### Video Generation

Video settings are process-level. Per-model tunables -- presets, allowed sizes, duration limits, per-user storage cap, the `vid.enabled` master switch -- are `vid.*` rows in `app_config`, managed from Admin > Video Config, not environment variables.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `VIDEO_STORAGE_PATH` | str | `/data/video` | Root directory for rendered clips and uploaded reference images |
| `VIDEO_RUNNER_ENABLED` | bool | `true` | Start the background video runner in this process |
| `VIDEO_RUNNER_POLL_INTERVAL_SECONDS` | int | `5` | Queue poll interval for the runner |
| `VIDEO_WORKER_TIMEOUT_SECONDS` | int | `60` | Timeout for control-plane calls to the worker (submit/poll/cancel) |
| `VIDEO_WORKER_FETCH_TIMEOUT_SECONDS` | int | `900` | Timeout for pulling the finished artifact back from the worker |
| `VIDEO_JOB_MAX_WALL_SECONDS` | int | `3600` | Hard wall-clock cap on a single render before it is failed |
| `VIDEO_JOB_STALE_HEARTBEAT_SECONDS` | int | `120` | Heartbeat age after which a rendering job is treated as stale |
| `VIDEO_RECONCILE_INTERVAL_SECONDS` | int | `20` | Interval of the ground-truth sweep that recovers orphaned renders |
| `VIDEO_RUNNER_LEASE_TTL_SECONDS` | int | `30` | Redis leader-lease TTL, so only **one** runner is active across workers/containers |
| `VIDEO_MAX_UPLOAD_MB` | int | `64` | Defined in `settings.py`, but **nothing in the application reads it** -- the reference-image upload cap comes from the `vid.max_image_upload_mb` config key (default 10) |
| `VIDEO_WEBHOOK_SIGNING_KEY` | str | `""` | Reserved for HMAC-signing worker callbacks. Defined and passed through Compose, but **nothing in the application reads it yet** -- the runner polls the worker rather than receiving webhooks. When used, the value belongs in the host `.env` only |

### UI Branding

Colors, organization name, and logo selections are `branding.*` rows in `app_config` (Admin > Branding). Only the on-disk storage of uploaded assets is configured here.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `BRANDING_STORAGE_PATH` | str | `/data/branding` | Directory for uploaded logo/favicon files. Must be a persistent volume, or uploads are lost on rebuild |
| `BRANDING_MAX_LOGO_MB` | int | `4` | Per-file cap for logo and favicon uploads |

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

> **There are no `DEFAULT_MAX_CONCURRENT_*` variables.** Migration `056` dropped the `max_concurrent` column from `groups`, `quotas`, and `api_keys`, and no per-user or per-group concurrency setting exists in `settings.py`. The surviving `max_concurrent` is a **per-backend** capability field on the `backends` table (default `4`), set through the admin backend API -- it caps in-flight requests to one backend, not to one user.

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
| `MAX_REQUEST_SIZE` | int | `52428800` | Defined in `settings.py`, but **nothing in the application reads it** -- the app enforces no global body limit. The real ceiling is the reverse proxy: `client_max_body_size 50m` in the bundled `nginx/nginx.conf`, which is what returns `413` |
| `BACKEND_REQUEST_TIMEOUT` | int | `300` | Total request timeout (seconds) |
| `BACKEND_REQUEST_TIMEOUT_PER_ATTEMPT` | int | `180` | Per-attempt timeout (seconds) |
| `BACKEND_RETRY_MAX_ATTEMPTS` | int | `3` | Max total retry attempts |
| `STRUCTURED_OUTPUT_RETRY_ON_INVALID` | bool | `true` | Intended to retry on a different backend when a response fails structured-output JSON validation |
| `THINKING_OFF_BY_DEFAULT` | bool | `true` | Gateway policy: reasoning/thinking is forced **off** unless the client explicitly opts in (`think: true`, `thinking: {type: "enabled"}`, or `reasoning_effort`). Applies to `enable_thinking`-style models (Qwen, Gemma, Nemotron); gpt-oss uses `reasoning_effort` and is left untouched. Set `false` to restore per-model launch defaults |
| `FIELD_VALIDATION` | str | `log` | Handling of unknown or vLLM-dialect request fields that would otherwise be silently dropped: `off`, `log` (record and continue), or `enforce` (reject with `400`). Deploy at `log` to observe real traffic, then flip to `enforce` |

> *Note: `STRUCTURED_OUTPUT_RETRY_ON_INVALID` is defined but not read anywhere in the inference pipeline. Setting it has no effect today; it is reserved for future use.*

### Responses & Conversations APIs

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `RESPONSES_API_ENABLED` | bool | `true` | Master switch for `/v1/responses` **and the entire `/v1/conversations` surface** -- every route in both routers checks it and returns `404` with `"The Responses API is not enabled on this server."` / `"The Conversations API is not enabled on this server."` when false |
| `RESPONSES_STORE_MAX_CHAIN_DEPTH` | int | `20` | Maximum `previous_response_id` hops walked when rebuilding a stored chain |
| `RESPONSES_STORE_MAX_PAYLOAD_BYTES` | int | `5242880` | Per-response stored-payload cap (5 MB). `0` = uncapped |
| `RESPONSES_STORE_MAX_ROWS_PER_USER` | int | `1000` | Stored responses retained per user; oldest are evicted. `0` = uncapped |
| `RESPONSES_WEB_SEARCH_ENABLED` | bool | `true` | Enable the hosted `{"type": "web_search"}` tool, executed server-side via the `/v1/search` provider stack |
| `RESPONSES_WEB_SEARCH_MAX_CALLS` | int | `4` | Search calls allowed per response. A request's `max_tool_calls` can lower it, not raise it |
| `RESPONSES_WEB_SEARCH_MAX_RESULTS` | int | `5` | Results fed back to the model per search |
| `CONVERSATIONS_MAX_PER_USER` | int | `1000` | Conversation objects per user; creation is rejected beyond it. `0` = uncapped |
| `CONVERSATIONS_MAX_ITEMS` | int | `10000` | Items per conversation; appends are rejected beyond it |
| `CONVERSATIONS_MAX_ITEM_BYTES` | int | `2097152` | Per-item size cap after image offload (2 MB). `0` = uncapped |

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
| `AUDIT_LOG_ENABLED` | bool | `true` | Master switch for prompt AND response content capture into the audit tables — metadata (model, tokens, timings) is always recorded. Disabling capture also disables DLP scanning of that content (DLP reads the stored copy) |
| `AUDIT_LOG_PROMPTS` | bool | `true` | Capture request messages/prompt text and offloaded request images |
| `AUDIT_LOG_RESPONSES` | bool | `true` | Capture response content |

### Telemetry & GPU

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `TELEMETRY_RETENTION_DAYS` | int | `30` | Telemetry data retention period |
| `TELEMETRY_CLEANUP_INTERVAL` | int | `3600` | Cleanup interval (seconds) |
| `SIDECAR_TIMEOUT` | int | `15` | Sidecar HTTP call timeout (seconds) |
| `GPU_AGENT_HOST` | str | `0.0.0.0` | Bind address for sidecar HTTP server |
| `GPU_AGENT_PORT` | int | `8007` | Port for sidecar HTTP server |

### Observability

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `METRICS_ENABLED` | bool | `true` | Enable Prometheus metrics |
| `METRICS_PREFIX` | str | `mindrouter` | Metrics name prefix |
| `OTEL_ENABLED` | bool | `false` | Enable OpenTelemetry |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | str | `None` | OpenTelemetry exporter endpoint |
| `OTEL_SERVICE_NAME` | str | `mindrouter` | Value reported as `service.name` on exported traces |

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

Default allowed extensions: `.txt`, `.md`, `.csv`, `.json`, `.html`, `.htm`, `.log`, `.docx`, `.xlsx`, `.pptx`, `.pdf`, `.jpg`, `.jpeg`, `.png`, `.gif`, `.webp`

### Data Retention & Audit Capture

Retention periods are **not** environment variables: they are runtime-editable
policies stored in the `app_config` table and managed at **Admin → Retention**
(categories: requests, chat, telemetry, request images, stored responses,
Conversations API; `0` = keep forever). The retention cycle runs hourly.

Audit content capture IS environment-configurable:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `AUDIT_LOG_ENABLED` | bool | `true` | Master switch for prompt AND response content capture into the audit tables (metadata is always recorded) |
| `AUDIT_LOG_PROMPTS` | bool | `true` | Capture request messages/prompt text (and offloaded request images) |
| `AUDIT_LOG_RESPONSES` | bool | `true` | Capture response content |

> **Note:** the DLP scanner reads the *stored* audit content — disabling
> capture also disables DLP scanning of that content. Web-chat conversation
> storage is user-facing state, not audit, and is unaffected by these flags.

**Never removed by retention or by a manual purge:** the `admin_audit_log`
(retained permanently by design, as tamper-evidence) and the email log, which
has no retention category. DLP alerts gained a retention category in 2.9.9
(`DLP Alerts (days)` on the Retention page, and a `dlp_alerts` purge
category); the default is `0` — keep forever — so existing alert history is
never deleted without an admin opting in. Purging is time-based per category —
there is no per-user or per-conversation deletion. To remove one person's
records, delete the user account (Admin → Users → the user →
Account Management).

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
| `app.base_url` | string | falls back to the `APP_BASE_URL` setting | Public base URL, written by Admin > Settings > **Site URL**. Used for the links MindRouter *generates* -- blog/RSS links and outgoing email. **SSO does not read this key**: OIDC redirect URIs and SAML `Destination`/`Recipient` come from the `APP_BASE_URL` environment variable, so setting Site URL alone does not fix an SSO mismatch |
| `app.timezone` | string | `"America/Los_Angeles"` | IANA timezone for date display in web UI |
| `ollama.enforce_num_ctx` | boolean | `true` | Override user-supplied `num_ctx` with model config `context_length` |
| `vid.*` | mixed | (see Admin > Video Config) | Video generation runtime policy -- `vid.enabled`, `vid.default_model`, `vid.allowed_sizes`, `vid.min_seconds`, `vid.max_total_seconds`, `vid.max_concurrent_jobs_per_user`, `vid.user_storage_cap_gb`, `vid.max_image_upload_mb` |
| `branding.*` | mixed | (see Admin > Branding) | Organization name, accent colors, and uploaded logo/favicon filenames |
| `retention.request_images_days` | integer | `180` | Retention window for audit-offloaded request images on disk |
| `retention.responses_store_days` | integer | `30` | Retention window for stored Responses API state. `0` disables the sweep |

---

## Implementation Notes

This section documents internal behaviors useful for operators and developers.

### Inflight Token Estimation

During streaming responses, tokens are estimated at **1 token per 4 characters** for real-time quota and throughput tracking. Estimates are flushed to Redis every 10 chunks. When the response completes, estimated counts are replaced by accurate backend-reported token counts.

> **Token count fallback:** If a backend returns zero for both prompt and completion token counts, MindRouter falls back to the job's pre-estimated token counts (based on tiktoken encoding of the input).

### Redis Token Counter Sync

A background sync loop flushes Redis token usage counters to the database every **60 seconds**. On startup, counters are seeded from the database. A final flush runs on graceful shutdown to prevent token count drift.

### Retention Cycle

A background task runs the data-retention cycle every **hour** (configurable via the `retention.cleanup_interval` policy at Admin → Retention). The requests, chat, and telemetry day values are seeded into `app_config` by migration 029; the request-images, stored-responses, and Conversations-API values fall back to built-in defaults (180 / 30 / 0 days) until first saved. All six are runtime-editable on the same page. An admin can also trigger a cycle immediately with **Run Retention Now**, or purge a single category on demand from the **Purge** tab (type-`PURGE` confirmation required).

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

Blog posts use soft deletion -- a `deleted_at` timestamp is set rather than removing the row, and soft-deleted posts are excluded from normal queries but retained in the database.

**User accounts are hard-deleted**, not soft-deleted (`DELETE /api/admin/users/{id}`, or Account Management on the admin user detail page). The account and its keys, requests/responses, chats, images, videos, and stored responses are removed, including files on disk; blog posts, the email log, and admin-audit entries are kept with the user reference detached, so authorship history survives the deletion.

### Status Enums

**BackendStatus:** `HEALTHY` (available for routing), `UNHEALTHY` (failed health checks), `DISABLED` (admin-disabled), `DRAINING` (graceful shutdown -- no new requests, existing ones complete), `UNKNOWN` (initial state before first health check).

**NodeStatus:** `ONLINE` (reachable), `OFFLINE` (unreachable), `UNKNOWN` (initial state).

**RequestStatus:** `QUEUED` (waiting in scheduler), `PROCESSING` (executing on backend), `COMPLETED` (success), `FAILED` (error), `CANCELLED` (timeout or user-cancelled).

---

## Deployment

MindRouter is designed for deployment on Linux servers with NVIDIA GPUs. The full deployment guide covers:

- Rocky Linux 8 prerequisites and dependency installation
- SSL/TLS configuration (self-signed and Let's Encrypt)
- Reverse proxy: the production stack ships its own **nginx** container for TLS termination on ports 80/443, so no host web server is required. An external Apache reverse proxy is documented as an alternative
- Firewall and SELinux configuration
- Docker Compose production stack, and bootstrapping the first admin account
- Database migrations
- GPU sidecar agent deployment (NVIDIA Container Toolkit, per-node nginx proxy)
- Node and backend registration
- Verification, tuning (uvicorn workers, MariaDB, nginx timeouts), Compose profiles, ongoing operations, and a security checklist

For step-by-step production deployment instructions, see **[../deploy/DEPLOYMENT.md](../deploy/DEPLOYMENT.md)**.

### Docker Compose env var passthrough

The repository ships **two different Compose stacks**, and they get their configuration in two different ways. `pydantic-settings` reads `.env` / `.env.prod` only from *inside* the container, and neither file is mounted into the image -- so how a variable reaches the process depends on which file you run:

| | `docker-compose.yml` (development / host-network stack) | `docker-compose.prod.yml` (guide's production stack) |
|---|---|---|
| Mechanism | `environment:` block, one `- VAR=${VAR:-default}` line per variable | `env_file: - .env.prod` |
| Passthrough | **Only variables explicitly listed** reach the container. Anything absent from the block is invisible to the app no matter what the host `.env` says | **Every key in `.env.prod`** reaches the container; no per-variable edit needed |
| Adding a new setting | Add the setting to `settings.py` **and** add a matching `- NEW_VAR=${NEW_VAR:-}` line here | Add the key to `.env.prod` on the host |

Consequences worth knowing:

- `docker-compose.yml` currently forwards the database/Redis/secret basics, all SSO variables (including the `SAML_SP_*` key pair), `APP_BASE_URL`, `BRAVE_SEARCH_API_KEY`, OTel, MCP, `RESPONSES_API_ENABLED`, `RUN_MIGRATIONS`, the `AUDIT_LOG_*` capture toggles, and the full `VIDEO_*` and `BRANDING_*` blocks. Settings **not** in that list -- `THINKING_OFF_BY_DEFAULT`, `FIELD_VALIDATION`, the `SCHEDULER_*` and `BACKEND_*` tunables, the `RESPONSES_STORE_*` / `CONVERSATIONS_*` caps -- run at their `settings.py` defaults under that stack until a passthrough line is added.
- `APP_BASE_URL` is forwarded with an **empty** default (`${APP_BASE_URL:-}`), so an unset value does not fall back to the `settings.py` placeholder; the app sees an empty string and derives URLs from request headers instead.
- A bare `docker compose up -d` on a host deployed from the guide starts the *other* stack alongside the running one. Pass `-f docker-compose.prod.yml` on every command, or export `COMPOSE_FILE=docker-compose.prod.yml` in the deployment shell.
- Secrets belong only in the host `.env` / `.env.prod`. Never commit them.

> **Database migrations:** MindRouter uses Alembic for schema migrations. Run `alembic upgrade head` inside the app container after deployment, or set `RUN_MIGRATIONS=1` to have the app apply them at startup before serving (run single-worker on first boot). When writing new migrations, note that MariaDB DDL is non-transactional -- a failed migration leaves partial state requiring manual cleanup. Always drop foreign key constraints before dropping their backing indexes (MariaDB error 1553).

---

## Testing

MindRouter has a comprehensive test suite covering unit, integration, end-to-end, smoke, stress, and accessibility tests.

### Quick Reference

| Command | Description |
|---------|-------------|
| `make test` | Every pytest suite under `backend/app/tests` (unit + integration + e2e) |
| `make test-unit` | Unit tests -- 1,000+ tests, no live services needed |
| `make test-int` | Integration tests (requires live backends) |
| `make test-e2e` | End-to-end tests |
| `make test-smoke` | Smoke tests against a live deployment (`API_KEY=` required; `BASE_URL=` defaults to `http://localhost:8000`) |
| `make test-stress` | Load/stress tests (`DURATION`, `CONCURRENCY` overridable) |
| `make test-matrix` | Structured-output matrix tests across all API styles (live stack) |
| `make test-thinking` | Structured output + thinking compliance (live stack) |
| `make test-tools` | Live tool-calling compliance across tool-capable models |
| `make test-a11y` | WCAG 2.1 accessibility tests (a subset of the unit suite) |
| `make test-sidecar` | GPU sidecar agent tests |
| `make test-all` | `backend/app/tests` plus `sidecar/tests` -- note this runs the pytest suites and the sidecar tests, **not** the smoke or stress targets |
| `make coverage` | Unit + integration with an HTML/terminal coverage report |

For the complete test manifest including all test files, descriptions, and counts, see **[../TESTING.md](../TESTING.md)** -- it is the single source of truth, and new test files must be registered there.
