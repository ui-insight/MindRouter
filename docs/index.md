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
12. [Blog System](#blog-system)
13. [Configuration Reference](#configuration-reference)
14. [Deployment](#deployment)
15. [Testing](#testing)

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

### Error Responses

All error responses follow a consistent format:

```json
{
  "detail": "Human-readable error message"
}
```

Common HTTP status codes:

| Code | Meaning |
|------|---------|
| 400 | Invalid request body or parameters |
| 401 | Missing or invalid API key |
| 403 | Insufficient permissions (e.g., non-admin accessing admin endpoint) |
| 404 | Resource not found |
| 409 | Conflict (duplicate name, URL, etc.) |
| 429 | Rate limit exceeded |
| 500 | Internal server error |

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

#### Embeddings

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Authorization: Bearer mr2_your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "nomic-embed-text", "input": "Hello world"}'
```

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
      "capabilities": {"vision": false, "embeddings": false, "structured_output": true},
      "backends": ["ollama-gpu1", "ollama-gpu2"]
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

#### Ollama Generate

```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Authorization: Bearer mr2_your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.2", "prompt": "Why is the sky blue?"}'
```

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
- Tool calling -- `tools` with `input_schema`, `tool_choice` (`auto`/`any`/`tool`), `tool_use`/`tool_result` content blocks, streaming tool use with `input_json_delta`
- Thinking/reasoning mode (`thinking.type`: `enabled`, `adaptive`, `disabled`; set `budget_tokens` to control reasoning token allocation)
- Structured output via `output_config.format` with `type: "json_schema"`
- Parameters: `max_tokens` (required), `temperature`, `top_p`, `top_k`, `stop_sequences`, `stream`
- `metadata.user_id` mapping

> **Note:** This is inbound-only -- there are no Anthropic backends. Requests are translated to canonical format and routed to Ollama/vLLM backends like any other request.

### Health & Metrics Endpoints

These endpoints are unauthenticated and intended for monitoring infrastructure.

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/healthz` | None | Liveness probe (always 200 if app is running) |
| GET | `/readyz` | None | Readiness probe (checks DB + healthy backends) |
| GET | `/metrics` | None | Prometheus metrics (text format) |
| GET | `/status` | None | Cluster status summary (JSON) |
| GET | `/api/cluster/throughput` | None | Token throughput (last 20 seconds) |
| GET | `/api/cluster/total-tokens` | None | Total tokens ever served (cached 10s) |
| GET | `/api/cluster/trends` | None | Token and active-user trends over time (query param: `range=day\|week\|month`) |

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
| POST | `/api/admin/backends/{id}/drain` | Start draining (stop new requests, let in-flight finish) |
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
| GET | `/api/admin/telemetry/backends/{id}/history` | Time-series telemetry for a backend |
| GET | `/api/admin/telemetry/gpus/{id}/history` | Time-series telemetry for a GPU device |
| GET | `/api/admin/telemetry/nodes/{id}/history` | Aggregated time-series for a node (all GPUs) |
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
| Request API Key | `/request-api-key` | Form for unauthenticated users to request an API key |

### User Dashboard

| Page | URL | Description |
|------|-----|-------------|
| Dashboard | `/dashboard` | Token usage progress bar, active API keys, quota usage history |
| Request Quota | `/dashboard/request-quota` | Submit a quota increase request with justification |
| Key Created | (after creation) | Displays the full API key once (copy-to-clipboard) |

### Admin Dashboard

The admin dashboard has a persistent sidebar with links to all admin pages:

| Page | URL | Description |
|------|-----|-------------|
| Overview | `/admin` | System metrics overview with pending request badges and health alerts |
| Backends | `/admin/backends` | Backend health, models, enable/disable/drain controls |
| Models | `/admin/models` | Model catalog with capability overrides (multimodal, thinking), metadata editing, context length configuration |
| Nodes | `/admin/nodes` | GPU node management, sidecar status, take offline/bring online, force drain, active requests |
| GPU Metrics | `/admin/metrics` | Real-time GPU utilization, memory, temperature, power charts with time range controls |
| Users | `/admin/users` | User accounts, group assignment, quota management, masquerade |
| Groups | `/admin/groups` | Group management with quota defaults, scheduler weights, admin flag |
| API Keys | `/admin/api-keys` | All API keys across users, status filtering, search |
| Requests | `/admin/requests` | Pending API key and quota increase requests, approve/deny |
| Audit Log | `/admin/audit` | Inference request history with filtering and search |
| Conversations | `/admin/conversations` | Browse and search all user conversations, view messages, export |
| Chat Config | `/admin/chat-config` | Configure core models, default model, system prompt, max_tokens, temperature, thinking mode |
| Blog | `/admin/blog` | Blog/CMS management -- create, edit, publish, delete posts |
| Settings | `/admin/settings` | Site-wide settings: timezone, enforce `num_ctx` override |

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

### Change Password

Users with local (non-SSO) accounts can change their password from the user dashboard (`/dashboard`). The form requires the current password, a new password (minimum 8 characters), and password confirmation.

### API Key Lifecycle

1. **Generation** -- Keys use the format `mr2_<random_urlsafe_base64>` (48+ characters total).
2. **Storage** -- The raw key is shown once at creation. Only the Argon2 hash and a prefix (`mr2_<first 8 chars>`) are stored in the database.
3. **Verification** -- Lookup by prefix (fast), then full Argon2 hash verification.
4. **Expiration** -- Optional `expires_at` timestamp.
5. **Revocation** -- Keys can be revoked (soft-delete) without deleting the audit trail.
6. **Usage tracking** -- `last_used_at` and `usage_count` updated atomically on each request.

### Quota System

Each user has a quota record with:

- **Token budget** -- Total tokens allowed per period. Deducted on each completed request (prompt + completion tokens).
- **RPM limit** -- Maximum requests per minute.
- **Max concurrent** -- Maximum simultaneous in-flight requests.

When a quota is exceeded, the request is rejected with HTTP 429.

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
| **Ollama** | `GET /api/tags` | `GET /api/tags` + `POST /api/ps` (loaded models) | Sidecar agent |
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
- **Drain** for graceful shutdown: `POST /api/admin/backends/{id}/drain`
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

Hard constraints (vision capability, embedding support, model availability) are checked before soft scoring.

For the complete algorithm specification, see **[scheduler.md](scheduler.md)**.

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
- Running processes (PID + memory)
- Device identity (name, UUID, compute capability)
- Driver and CUDA versions

**Authentication:** Requires `SIDECAR_SECRET_KEY` env var. All requests must include `X-Sidecar-Key` header (constant-time comparison).

**Deployment options:**

1. **Docker Compose** -- `docker compose --profile gpu up gpu-sidecar`
2. **Standalone Docker** -- Build from `sidecar/Dockerfile.sidecar`, run with `--gpus all`
3. **Direct Python** -- `pip install fastapi uvicorn nvidia-ml-py && python sidecar/gpu_agent.py`

### Health Polling

The Backend Registry runs an adaptive polling loop:

- **Normal interval:** 30 seconds (configurable via `BACKEND_POLL_INTERVAL`)
- **Fast interval:** 10 seconds after a backend becomes unhealthy (configurable)
- **Fast duration:** 120 seconds before returning to normal polling

Each poll cycle has two phases:
1. Poll sidecar agents (one per physical node) for GPU snapshots
2. Poll each backend adapter for health, models, and engine-specific telemetry

### Startup Fast Polls

On container start, the registry runs **two immediate full poll cycles** with a 5-second gap between them. This ensures backends and nodes are marked healthy within seconds of a restart, rather than waiting for the first normal 30-second poll interval.

### Circuit Breaker

Per-backend circuit breaker protects against cascading failures:

- **Threshold:** 3 consecutive failures before opening (configurable via `BACKEND_CIRCUIT_BREAKER_THRESHOLD`)
- **Recovery:** 30 seconds before allowing a probe request (`BACKEND_CIRCUIT_BREAKER_RECOVERY_SECONDS`)
- **States:** Closed (healthy) → Open (failing) → Half-Open (probe) → Closed (recovered)

### Latency Tracking

Exponential Moving Average (EMA) tracks per-backend latency:

- **Alpha:** 0.3 (30% current observation, 70% history)
- **Metrics:** Total latency EMA and TTFT (time-to-first-token) EMA
- **Throughput score:** `1.0 / (1.0 + latency_ms / 5000.0)` -- used in backend scoring
- **Persistence:** EMAs are periodically saved to the database for recovery after restart

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

### Messages

- Messages include role (user/assistant/system) and content
- Assistant messages are streamed in real-time
- Messages can be edited or deleted after creation
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
- Max upload size: 10 MB (configurable via `CHAT_UPLOAD_MAX_SIZE_MB`)
- Artifact storage path: `/data/artifacts` (configurable via `ARTIFACT_STORAGE_PATH`)
- Artifact max size: 50 MB (configurable via `ARTIFACT_MAX_SIZE_MB`)
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
- **Delete post** (`POST /admin/blog/{id}/delete`) -- Permanently delete a post.

---

## Configuration Reference

All settings are loaded from environment variables or `.env` / `.env.prod` files. Variable names are case-insensitive.

### Application

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `APP_NAME` | str | `MindRouter` | Application name |
| `APP_VERSION` | str | `1.0.0` | Application version |
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

Azure AD SSO is enabled automatically when both `AZURE_AD_CLIENT_ID` and `AZURE_AD_TENANT_ID` are set. Users authenticating via SSO for the first time are automatically created and assigned to the group specified by `AZURE_AD_DEFAULT_GROUP`.

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
| `BACKEND_RETRY_ATTEMPTS` | int | `2` | Default retry attempts |
| `BACKEND_RETRY_BACKOFF` | float | `1.0` | Retry backoff multiplier |
| `STRUCTURED_OUTPUT_RETRY_ON_INVALID` | bool | `true` | Retry on invalid structured output |

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

---

## Testing

MindRouter has a comprehensive test suite covering unit, integration, end-to-end, smoke, stress, and accessibility tests.

### Quick Reference

| Command | Description |
|---------|-------------|
| `make test-unit` | Run unit tests (238 tests) |
| `make test-int` | Integration tests (requires live backends) |
| `make test-e2e` | End-to-end tests |
| `make test-smoke` | Smoke tests (full API surface) |
| `make test-stress` | Load/stress tests |
| `make test-a11y` | WCAG 2.1 accessibility tests |
| `make test-sidecar` | GPU sidecar agent tests |
| `make test-all` | Run all test suites |

For the complete test manifest including all test files, descriptions, and counts, see **[../TESTING.md](../TESTING.md)**.
