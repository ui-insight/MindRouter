# MindRouter2 Architecture

## Overview

MindRouter2 is a production-ready LLM inference load balancer designed to front heterogeneous backend clusters of Ollama and vLLM inference nodes. It provides a unified OpenAI-compatible API surface while implementing fair-share scheduling, quota management, and comprehensive audit logging.

## System Components

### 1. API Gateway

The API Gateway is the entry point for all client requests.

**Key Responsibilities:**
- Request authentication via API keys
- Request validation and normalization
- Protocol translation (OpenAI ↔ Ollama ↔ Anthropic ↔ vLLM)
- Response streaming with low latency
- Error handling and retry logic

**Endpoints:**
- `/v1/chat/completions` - OpenAI-compatible chat
- `/v1/embeddings` - OpenAI-compatible embeddings
- `/v1/models` - List available models
- `/api/chat` - Ollama-compatible chat
- `/api/generate` - Ollama-compatible generation
- `/api/tags` - Ollama-compatible model list
- `/anthropic/v1/messages` - Anthropic Messages API compatible (inbound-only)

### 2. Translation Layer

The translation layer converts between different API formats.

```
┌─────────────────┐
│ Client Request  │
└────────┬────────┘
         │
    ┌────▼────┐  ┌─────────┐  ┌────────────┐
    │OpenAI In│  │Ollama In│  │Anthropic In│
    └────┬────┘  └────┬────┘  └──────┬─────┘
         │            │              │
         └────────────┼──────────────┘
                      │
              ┌───────▼───────┐
              │   Canonical   │
              │    Schema     │
              └───────┬───────┘
                      │
              ┌───────┴────────┐
              │                │
         ┌────▼────┐      ┌────▼─────┐
         │vLLM Out │      │Ollama Out│
         └────┬────┘      └────┬─────┘
              │                │
              └───────┬────────┘
                      │
         ┌────────────▼────────────┐
         │     Backend Request     │
         └─────────────────────────┘
```

> **Note:** The Anthropic translator is inbound-only. Responses from backends (in OpenAI/Ollama format) are converted back to Anthropic Messages format by `AnthropicInTranslator.format_response()` and `format_stream_event()`.

**Canonical Schemas:**
- `CanonicalChatRequest` - Unified chat format (messages, tools, tool_choice, response_format, etc.)
- `CanonicalMessage` - Message with role, content, tool_calls, tool_call_id
- `CanonicalToolCall` / `CanonicalToolDefinition` - Tool calling primitives (arguments stored as JSON strings)
- `CanonicalEmbeddingRequest` - Unified embedding format
- `CanonicalStreamChunk` - Unified streaming format (including tool call deltas)

### 3. Fair-Share Scheduler

Implements Weighted Deficit Round Robin (WDRR) for fair resource allocation.

**Key Concepts:**
- **Share Weights**: faculty=3, staff=2, student=1, admin=10
- **Deficit Counters**: Track service debt per user
- **Burst Credits**: Allow full cluster use when idle
- **Deprioritization**: Reduce priority for heavy users

**Scheduling Flow:**
1. Job arrives, compute initial priority
2. Add to per-user queue
3. When backend available:
   - Select user with highest (deficit + burst_credits) / weight
   - Score eligible backends
   - Route to best backend
4. On completion, update deficit counter

### 4. Backend Registry

Manages backend discovery, health, and telemetry. The registry operates on a **Node/Backend separation**: a Node represents a physical GPU server running a sidecar agent, while a Backend is an inference endpoint running on a node.

**Node/Backend Model:**
- **Node**: Physical server with a sidecar agent, owns GPU devices
- **Backend**: Inference endpoint (Ollama/vLLM) running on a node, assigned specific GPUs via `gpu_indices`
- One sidecar poll per node (deduplicated across backends)
- Per-backend GPU aggregation uses only assigned GPU subset

**Two-Phase Telemetry Collection:**
1. **Phase A — Per-node sidecar poll**: Calls each node's sidecar once, upserts GPU device records, stores GPU telemetry, updates node hardware info
2. **Phase B — Per-backend health poll**: Calls each backend's inference API for health/model/request metrics, computes per-backend GPU utilization from cached node data filtered by `gpu_indices`

**Backend Adapters:**
- `OllamaAdapter` - Polls `/api/tags`, `/api/ps`, `/api/version`
- `VLLMAdapter` - Polls `/v1/models`, `/health`, `/metrics`
- `SidecarClient` - Polls node's `/gpu-info` for GPU hardware metrics

### 4a. GPU Sidecar Agent

A lightweight FastAPI service (`sidecar/gpu_agent.py`) that runs on each physical GPU server. Uses NVIDIA Management Library (pynvml) to expose per-GPU metrics.

**Endpoints:**
- `GET /health` - Liveness check, returns GPU count
- `GET /gpu-info` - Full GPU metrics (utilization, memory, temperature, power, clocks, processes)

**Deployment:** Runs as a Docker container with `--gpus all` or directly via Python on each GPU node. Exposes port 8007 by default.

### 5. Backend Scorer

Multi-factor scoring for backend selection.

**Hard Constraints (must pass):**
- Model availability
- Modality support (vision, embeddings)
- Memory fit
- Capacity available

**Soft Scores (higher = better):**
- Model already loaded: +100
- Low GPU utilization: +50
- Short queue: +30
- High throughput GPU: +20

### 6. Quota Management

Token-based quota system with role-based defaults.

**Features:**
- Monthly token budgets
- Requests per minute (RPM) limits
- Max concurrent requests
- Automatic quota reset

### 7. Audit Logging

Complete request/response logging for compliance.

**Logged Data:**
- Full request content (prompts, messages)
- Full response content
- Token usage (actual or estimated)
- Timing metrics (queue delay, processing time)
- Scheduling decisions

## Data Model

```
┌─────────────┐     ┌─────────────┐
│    User     │────<│   ApiKey    │
└──────┬──────┘     └─────────────┘
       │
       │     ┌─────────────┐
       ├────<│    Quota    │
       │     └─────────────┘
       │
       │     ┌─────────────┐     ┌─────────────┐
       └────<│   Request   │────<│  Response   │
             └──────┬──────┘     └─────────────┘
                    │
                    │     ┌─────────────┐
                    ├────<│  Artifact   │
                    │     └─────────────┘
                    │
                    │     ┌─────────────┐
                    └────<│ Scheduler   │
                          │  Decision   │
                          └─────────────┘

┌─────────────┐     ┌─────────────┐
│    Node     │────<│  GPUDevice  │
└──────┬──────┘     └──────┬──────┘
       │                   │
       │     ┌─────────────┤
       │     │  GPUDevice   │
       │     │  Telemetry   │
       │     └─────────────┘
       │
┌──────┴──────┐     ┌─────────────┐
│   Backend   │────<│    Model    │
└──────┬──────┘     └─────────────┘
       │
       │     ┌─────────────┐
       └────<│  Telemetry  │
             └─────────────┘

Node → Backend: one-to-many (a node can host multiple backends)
Backend.gpu_indices: JSON list of GPU device indices assigned to this backend
GPUDevice belongs to Node (not Backend)
```

## Request Flow

```
Client Request
       │
       ▼
┌─────────────────┐
│  Authentication │
│   (API Key)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Quota Check     │
│ Rate Limit      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Translation    │
│  (→ Canonical)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Create Job     │
│  Compute Priority│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Route to       │
│  Backend        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Proxy Request  │
│  (Stream if     │
│   applicable)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Record Audit   │
│  Update Usage   │
└────────┬────────┘
         │
         ▼
    Response
```

## Deployment

### Development
```bash
docker compose up --build
```

### Production Considerations
- Use external MariaDB cluster
- Configure Redis for distributed rate limiting
- Mount persistent volume for artifacts
- Set up monitoring (Prometheus/Grafana)
- Configure TLS termination
- Set secure SECRET_KEY
