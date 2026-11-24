# MindRouter

MindRouter is a production-ready, self-hosted **AI inference gateway** that fronts a heterogeneous cluster of **vLLM** and **Ollama** GPU nodes and load-balances requests across them. It exposes a single API that speaks the **OpenAI**, **Anthropic**, and **Ollama** dialects, translating between them so existing client code works unchanged.

![MindRouter cluster status dashboard: live token-flow throughput, backend health, and usage at a glance](docs/images/cluster-status.jpg)

*The public Cluster Status page (shown here with University of Idaho branding): real-time token-flow throughput across the backend cluster, healthy-backend and active-user counts, queue depth, and total tokens served.*

The platform is multi-modal, spanning chat and text completion, embeddings, vision, **image, video and 3D model generation**, **speech (TTS/STT)**, and document OCR. It also provides the governance and operations layers a shared deployment needs: **fair-share scheduling** across users and groups, **per-user token and request quotas**, **role-based access with optional SSO (Azure AD/Entra, Google, any OIDC provider, or SAML 2.0)**, **full audit logging**, and real-time **GPU, health, and energy monitoring**.

Because all inference runs on hardware the operator owns, MindRouter is built to support **institutional and organizational AI sovereignty**: scalable on-premises hosting that keeps prompts, documents, and generated media on local infrastructure instead of third-party clouds.

> **Citation.** If you use MindRouter in your work, please cite:
>
> Sheneman, L. (2026). MindRouter: Open-source LLM inference gateway for institutional AI sovereignty. In *Proceedings of the Practice and Experience in Advanced Research Computing Conference 2026 (PEARC '26)*. Association for Computing Machinery. [https://doi.org/10.1145/3785462.3815861](https://doi.org/10.1145/3785462.3815861)

## Documentation

For comprehensive documentation covering the full application, API reference, configuration, and more, see **[docs/index.md](docs/index.md)**.

Interactive API docs are also available at `/docs` (Swagger UI) and `/redoc` (ReDoc) when the application is running.

## Related Projects

- **[dgx-stack](https://github.com/ui-insight/dgx-stack)**: Running MindRouter and vLLM on NVIDIA DGX Spark systems.

## Features

- **Unified API Gateway**: OpenAI-compatible `/v1/*` (including the Responses API `/v1/responses`), Ollama `/api/*`, and Anthropic `/anthropic/v1/*` endpoints
- **API Dialect Translation**: Automatic translation between Ollama and vLLM formats
- **Fair-Share Scheduling**: Weighted Deficit Round Robin with burst credits
- **Multi-Modal Support**: Text, embeddings, and vision-language models
- **Image Generation**: Text-to-image and image-to-image / reference edit via `/v1/images/generations` and `/v1/images/edits` (diffusion backends; per-account enablement)
- **Video Generation**: Asynchronous text-to-video with keyframe conditioning via `/v1/videos` (per-account enablement, per-user storage quota)
- **Image to 3D**: Asynchronous image-to-mesh via `/v1/meshes`: one picture in, a textured GLB (PBR) and a preview out, with a 3D viewer tab. Models: Hunyuan3D 2.1 (default), TRELLIS.2 and Pixal3D, each on its own GPU worker (`mesh-worker/`). Admin → Meshes sets the service switch, default model, face/texture ranges, per-model caps and settings, and license acceptance (per-account enablement, per-user storage quota). See [docs/mesh-api.md](docs/mesh-api.md)
- **Document OCR**: Text extraction from PDFs and images via `/v1/ocr`
- **Structured Outputs**: JSON schema validation across all backends
- **Quota Management**: Per-user token budgets with role-based weights
- **Node/Backend Architecture**: Separate physical GPU nodes from inference endpoints — one sidecar poll per node, GPU-to-backend assignment
- **GPU Sidecar Agent**: Lightweight per-node agent for real-time GPU metrics (utilization, memory, temperature, power)
- **Real-Time Telemetry**: GPU/memory/utilization monitoring per node and per backend
- **Energy Monitoring**: Per-node power draw and per-request energy accounting, surfaced in the admin dashboard
- **Drain Mode**: Gracefully take backends offline — stop new requests while in-flight requests finish, then auto-disable
- **Health Alerts**: Admin dashboard shows a prominent warning banner when any backend is unhealthy or any node is offline
- **Tool Calling**: Function calling support across all API surfaces with cross-engine translation
- **Thinking/Reasoning Mode**: Control reasoning depth on supported models (qwen3.5, qwen3, gpt-oss)
- **Web Search**: Standalone search API (`/v1/search`) with pluggable providers (Brave Search, SearXNG)
- **Agentic AI Integrations**: MCP servers and agent skills for Claude Code, ForgeCode, OpenCode, Codex, Cursor, and others (`agentic_ai/`)
- **Single Sign-On**: Optional SSO via Azure AD / Entra ID, Google, any OIDC provider (Okta, Keycloak, Auth0, or CILogon for InCommon federation), and SAML 2.0 (Shibboleth, ADFS) — enable any subset from environment variables, with JIT user provisioning; local accounts keep working alongside it ([guide](docs/sso-configuration.md))
- **Voice API**: Public TTS and STT endpoints (`/v1/audio/speech`, `/v1/audio/transcriptions`) with API key auth and quota tracking
- **Full Audit Logging**: All prompts, responses, and artifacts stored for review
- **Dual Dashboards**: Public status + authenticated user/admin interfaces with dark mode
- **Custom Branding & Theming**: Rebrand a deployment for one institution — organization name, logos (navbar/footer/login), favicon, and accessible light/dark accent colors, from Admin → Branding ([guide](docs/branding.md))

## Quickstart

### Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for local development)

### Running with Docker Compose

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings (database passwords, secret key, etc.)
nano .env

# Start all services
docker compose up --build

# In another terminal, run migrations
docker compose exec app alembic upgrade head

# Seed development data
docker compose exec app python scripts/seed_dev_data.py
```

### API Access

The gateway runs on `http://localhost:8000` by default.

#### OpenAI-Compatible Endpoints

```bash
# Chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'

# Embeddings
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nomic-embed-text",
    "input": "Hello world"
  }'

# List models
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer your-api-key"
```

#### Anthropic-Compatible Endpoint

```bash
# Chat via Anthropic Messages API
curl -X POST http://localhost:8000/anthropic/v1/messages \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2",
    "max_tokens": 500,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Anthropic SDK clients (Python, TypeScript) can point directly at MindRouter:

```python
import anthropic
client = anthropic.Anthropic(
    base_url="http://localhost:8000/anthropic",
    api_key="your-api-key",
)
message = client.messages.create(
    model="llama3.2",
    max_tokens=500,
    messages=[{"role": "user", "content": "Hello!"}],
)
```

#### Ollama-Compatible Endpoints

```bash
# Chat via Ollama API
curl -X POST http://localhost:8000/api/chat \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'

# Generate via Ollama API
curl -X POST http://localhost:8000/api/generate \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2",
    "prompt": "Why is the sky blue?"
  }'
```

### Dashboards

- **Public Dashboard**: `http://localhost:8000/` - Cluster status, request API key
- **User Dashboard**: `http://localhost:8000/dashboard` - Usage, keys, quota requests
- **Chat Interface**: `http://localhost:8000/chat` - Full-featured chat with file upload, web search, streaming
- **Admin Dashboard**: `http://localhost:8000/admin` - Full system control

### Development Credentials (first admin)

`scripts/seed_dev_data.py` creates the default groups and **one** user — the
initial administrator. No other accounts are seeded; everyone else is created on
first SSO login (or by an admin).

| User | Password | Group |
|------|----------|-------|
| `admin` | `admin123` (override with `ADMIN_PASSWORD`) | admin |

The script also mints an API key for that user and prints it as a single
parseable line, `ADMIN_API_KEY=mr2_…`. It cannot be shown again. Automation
overrides (all optional env vars):

| Env var | Effect |
|---------|--------|
| `ADMIN_PASSWORD` | Password for the seeded admin (default `admin123`) |
| `ADMIN_API_KEY` | Use this exact key instead of minting a random one |
| `MINT_ADMIN_KEY=1` | Mint an additional key for an already-existing admin |

> **Development only.** `admin123` is a well-known password — never run the
> seeder with defaults on an internet-reachable deployment. For production, run
> it once with `ADMIN_PASSWORD` set to a strong secret (and `ADMIN_API_KEY` if
> you manage keys elsewhere), then change the password from the dashboard.
> You need this first admin to reach Admin → Branding, SSO, and node
> configuration.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MindRouter Gateway                       │
├─────────────────────────────────────────────────────────────────┤
│  ┌────────┐ ┌────────┐ ┌───────────┐ ┌───────┐ ┌──────────────┐ │
│  │ OpenAI │ │ Ollama │ │ Anthropic │ │ Admin │ │  Dashboard   │ │
│  │ /v1/*  │ │ /api/* │ │/anthropic/│ │  API  │ │ (Bootstrap)  │ │
│  └───┬────┘ └───┬────┘ └─────┬─────┘ └─-──┬──┘ └───-───┬──────┘ │
│      │          │            │            │            │        │
│      └──────────┴────────────┴────────────┴────────────┘        │
│                              │                                  │
│  ┌───────────────────────────┴───────────────────────────────┐  │
│  │                    Translation Layer                      │  │
│  │  OpenAI/Ollama/Anthropic ←→ Canonical ←→ Ollama/vLLM      │  │
│  └───────────────────────────┬───────────────────────────────┘  │
│                              │                                  │
│  ┌───────────────────────────┴───────────────────────────────┐  │
│  │              Fair-Share Scheduler (WDRR)                  │  │
│  │  • Per-user queues with deficit counters                  │  │
│  │  • Role-based weights (faculty > staff > student)         │  │
│  │  • Burst credits for idle cluster utilization             │  │
│  └───────────────────────────┬───────────────────────────────┘  │
│                              │                                  │
│  ┌───────────────────────────┴───────────────────────────────┐  │
│  │                  Backend Registry                         │  │
│  │  • Node + Backend separation (1 node → N backends)        │  │
│  │  • Per-node sidecar polling (deduplicated)                │  │
│  │  • GPU-to-backend assignment via gpu_indices              │  │
│  │  • Health monitoring & model residency tracking           │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
   │   Node 1    │      │   Node 2    │      │   Node 3    │
   │  4x A100    │      │  2x L40S    │      │  2x RTX4090 │
   │ ┌─────────┐ │      │ ┌─────────┐ │      │ ┌─────────┐ │
   │ │ Sidecar │ │      │ │ Sidecar │ │      │ │ Sidecar │ │
   │ │ :8007   │ │      │ │ :8007   │ │      │ │ :8007   │ │
   │ └─────────┘ │      │ └─────────┘ │      │ └─────────┘ │
   │ ┌────┬────┐ │      │ ┌─────────┐ │      │ ┌─────────┐ │
   │ │vLLM│vLLM│ │      │ │  Ollama │ │      │ │  Ollama │ │
   │ │0,1 │2,3 │ │      │ │  (all)  │ │      │ │  (all)  │ │
   │ └────┴────┘ │      │ └─────────┘ │      │ └─────────┘ │
   └─────────────┘      └─────────────┘      └─────────────┘
```

**Key concept**: A **Node** represents a physical GPU server running a sidecar agent. A **Backend** is an inference endpoint (Ollama or vLLM instance) running on a node. Multiple backends can share a node, each assigned specific GPUs via `gpu_indices`.

## Configuration

### Environment Variables

See `.env.example` for all available configuration options.

Key settings:

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | MariaDB connection string | Required |
| `SECRET_KEY` | Session/JWT signing key | Required |
| `APP_BASE_URL` | Public HTTPS origin of this deployment — required for correct SSO redirect URIs and SAML validation | `https://mindrouter.uidaho.edu` |
| `REDIS_URL` | Redis for rate limiting (optional) | None |
| `ARTIFACT_STORAGE_PATH` | Path for uploaded files | `/data/artifacts` |
| `SCHEDULER_FAIRNESS_WINDOW` | Rolling window for usage tracking | 300 (5 min) |
| `GPU_AGENT_PORT` | Sidecar agent listen port (sidecar-side env var, not a MindRouter setting) | 8007 |

### Single Sign-On (optional)

MindRouter supports **Azure AD / Entra ID**, **Google**, **generic OIDC** (Okta, Keycloak, Auth0, or CILogon for InCommon), and **SAML 2.0** (Shibboleth, ADFS). Each provider is enabled purely by environment variables — any subset can run at once, and the login page renders one button per enabled provider. Local username/password login is always available. The Docker image ships with SAML support included; bare-metal installs need `pip install -e ".[saml]"`. Set `APP_BASE_URL` to your public HTTPS origin, since redirect URIs and SAML validation derive from it.

See [docs/sso-configuration.md](docs/sso-configuration.md) for per-provider setup, JIT provisioning, and security notes.

### Node and Backend Registration

MindRouter separates **nodes** (physical GPU servers) from **backends** (inference endpoints). Register a node first, then attach backends to it.

```bash
# Step 1: Register a GPU node (physical server running the sidecar agent)
curl -X POST http://localhost:8000/api/admin/nodes/register \
  -H "Authorization: Bearer admin-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "gpu-server-1",
    "hostname": "gpu1.example.com",
    "sidecar_url": "http://gpu1.example.com:8007"
  }'

# Step 2: Register a backend on that node (all GPUs)
curl -X POST http://localhost:8000/api/admin/backends/register \
  -H "Authorization: Bearer admin-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "ollama-gpu1",
    "url": "http://gpu1.example.com:11434",
    "engine": "ollama",
    "max_concurrent": 4,
    "node_id": 1
  }'

# Or assign specific GPUs to a backend (multi-backend node)
curl -X POST http://localhost:8000/api/admin/backends/register \
  -H "Authorization: Bearer admin-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "vllm-gpu1-01",
    "url": "http://gpu1.example.com:8000",
    "engine": "vllm",
    "max_concurrent": 16,
    "node_id": 1,
    "gpu_indices": [0, 1]
  }'
```

Backends without a `node_id` still work as standalone endpoints (no GPU telemetry).

### Concurrency Alignment

The `max_concurrent` value registered in MindRouter **must match** the concurrency limit configured on the inference engine itself:

| Engine | Engine Setting | MindRouter Setting |
|--------|---------------|-------------------|
| vLLM | `--max-num-seqs N` | `"max_concurrent": N` |
| Ollama | `OLLAMA_NUM_PARALLEL=N` | `"max_concurrent": N` |

**Why this matters**: MindRouter uses `max_concurrent` to decide how many requests to route to a backend. If MindRouter thinks a backend can handle 8 concurrent requests but vLLM is configured with `--max-num-seqs 4`, the extra requests queue silently inside vLLM. MindRouter can't see this hidden queue, so it keeps routing requests there instead of spreading load to other backends. The result is uneven load distribution and unpredictable latency — the fair-share scheduler is effectively bypassed for those excess requests.

### Context Length (num_ctx)

MindRouter auto-discovers each model's context length and caps it at `min(model_max_context, 32768)` to prevent small models from consuming excessive VRAM. For every Ollama request, MindRouter injects `num_ctx` matching the configured `context_length`. By default, user-supplied `num_ctx` values are overridden to prevent GPU memory oversubscription — this can be toggled in Site Settings. Admins can also set `context_length_override` per model in the admin UI. Ollama 0.17+ automatically adjusts downward if the requested context doesn't fit in GPU memory.

## Development

### Local Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Start MariaDB (via docker)
docker compose up -d mariadb redis

# Run migrations
alembic upgrade head

# Seed data
python scripts/seed_dev_data.py

# Start development server
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

### Running Tests

```bash
# Unit tests
pytest backend/app/tests/unit -v

# Integration tests (requires docker)
pytest backend/app/tests/integration -v

# End-to-end tests
pytest backend/app/tests/e2e -v

# All tests with coverage
pytest --cov=backend/app --cov-report=html
```

### Makefile Commands

```bash
make dev          # Start development server
make test         # Run all tests
make test-unit    # Run unit tests only
make lint         # Run linters
make format       # Format code
make migrate      # Run database migrations
make seed         # Seed development data
make docker-up    # Start docker compose stack
make docker-down  # Stop docker compose stack
make migrate-down   # Rollback one Alembic migration
make demo           # Run fairness demo script
make docker-shell   # Open bash shell in the app container
make docker-seed    # Seed development data via docker-compose
make docker-migrate # Run Alembic migrations via docker-compose
```

## API Documentation

### Endpoints

#### Inference Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | OpenAI-compatible chat |
| POST | `/v1/responses` | OpenAI Responses API (agents, Codex) |
| POST | `/v1/responses/input_tokens` | Count input tokens without generating |
| * | `/v1/conversations` | OpenAI Conversations API (durable conversation state) |
| POST | `/v1/completions` | OpenAI-compatible completion |
| POST | `/v1/embeddings` | OpenAI-compatible embeddings |
| POST | `/v1/rerank` | Rerank documents against a query |
| POST | `/v1/score` | Score similarity between text pairs |
| POST | `/v1/images/generations` | Image generation (FLUX; per-account enablement) |
| GET | `/v1/models` | List available models |
| POST | `/api/chat` | Ollama-compatible chat |
| POST | `/api/generate` | Ollama-compatible generate |
| GET | `/api/tags` | Ollama-compatible model list |
| POST | `/anthropic/v1/messages` | Anthropic Messages API compatible |

#### Voice API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/audio/speech` | OpenAI-compatible text-to-speech |
| POST | `/v1/audio/transcriptions` | OpenAI-compatible speech-to-text |

#### Image to 3D Endpoints (`/v1/meshes`)

Asynchronous: one picture in, a textured 3D model (binary glTF, PBR) out.
Requires per-account enablement and the service switch on Admin → Meshes.

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/meshes` | Create a job (multipart `image`, or JSON with base64 `image`) → `202` + job object |
| GET | `/v1/meshes/{id}` | Poll a job (status, stage, progress, queue position) |
| GET | `/v1/meshes/{id}/content` | Download the GLB; `?variant=preview` for the PNG preview |
| GET | `/v1/meshes` | List your jobs, newest first |
| DELETE | `/v1/meshes/{id}` | Cancel a running job (refunded) or delete a finished one |
| GET | `/v1/meshes/models` | Models with their own limits, settings, license and availability |

Request fields: `model` (default `default-mesh` → Hunyuan3D 2.1), `texture`
(default true), `texture_size` (1024 / 2048 / 4096, default 2048),
`target_faces` (default 40,000; 5,000–300,000, Hunyuan3D capped at 100,000),
`remove_background` (`auto` / `on` / `off`), `seed`. Results use glTF's
convention (+Y up, front facing +Z) for every model.

```bash
MESH=$(curl -s -X POST https://mindrouter.example.edu/v1/meshes \
  -H "Authorization: Bearer $MINDROUTER_API_KEY" -F image=@chair.jpg | jq -r .id)
curl -s https://mindrouter.example.edu/v1/meshes/$MESH -H "Authorization: Bearer $MINDROUTER_API_KEY" | jq .status
curl -L https://mindrouter.example.edu/v1/meshes/$MESH/content -H "Authorization: Bearer $MINDROUTER_API_KEY" -o chair.glb
```

Hunyuan3D 2.1 is used under the Tencent Hunyuan 3D 2.1 Community License
Agreement: not licensed in the EU, the UK or South Korea; outputs may not
be used to improve other AI models; Tencent is not affiliated with or
endorsing the service. It takes no jobs until an admin accepts its license
on Admin → Meshes. Full reference: [docs/mesh-api.md](docs/mesh-api.md);
the GPU worker: [mesh-worker/README.md](mesh-worker/README.md).

#### Health & Metrics

| Method | Path | Description |
|--------|------|-------------|
| GET | `/healthz` | Liveness probe |
| GET | `/readyz` | Readiness probe |
| GET | `/metrics` | Prometheus metrics |
| GET | `/status` | Cluster status summary (JSON) |
| GET | `/api/cluster/total-tokens` | Total tokens served |
| GET | `/api/cluster/trends` | Token and user trends |
| GET | `/api/cluster/throughput` | Real-time token throughput |

#### Admin Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/admin/nodes/register` | Register a GPU node |
| GET | `/api/admin/nodes` | List all nodes |
| DELETE | `/api/admin/nodes/{id}` | Remove a node |
| POST | `/api/admin/nodes/{id}/refresh` | Refresh node sidecar |
| POST | `/api/admin/backends/register` | Register new backend |
| POST | `/api/admin/backends/{id}/disable` | Disable backend |
| POST | `/api/admin/backends/{id}/enable` | Enable backend |
| POST | `/admin/backends/{id}/drain` | Drain backend (dashboard-only) |
| POST | `/api/admin/backends/{id}/refresh` | Refresh capabilities |
| POST | `/api/admin/backends/{id}/ollama/pull` | Pull a model to Ollama backend |
| POST | `/api/admin/backends/{id}/ollama/delete` | Delete a model from Ollama backend |
| GET | `/api/admin/telemetry/overview` | Cluster telemetry overview |
| GET | `/api/admin/telemetry/nodes/{id}/history` | Node GPU history |
| GET | `/api/admin/telemetry/export` | Export telemetry as JSON or CSV |
| GET | `/api/admin/queue` | View scheduler queue |
| GET | `/api/admin/audit/search` | Search audit logs |
| GET | `/api/admin/conversations/export` | Export conversations |

## Scheduler Algorithm

MindRouter implements **Weighted Deficit Round Robin (WDRR)** for fair resource allocation:

1. **Share Weights**: faculty=3, staff=2, student=1, admin=10
2. **Deficit Counters**: Track service debt per user
3. **Burst Credits**: Allow full cluster use when idle
4. **Backend Scoring**: Model residency, GPU utilization, queue depth

See [docs/scheduler.md](docs/scheduler.md) for detailed algorithm specification.

## GPU Sidecar Agent

Each GPU node runs a lightweight **sidecar agent** (`sidecar/gpu_agent.py`) that exposes per-GPU hardware metrics via HTTP. MindRouter's backend registry polls the sidecar once per node (not per backend) to collect telemetry.

### What it collects

- GPU utilization and memory utilization (%)
- Memory usage (used/free/total GB)
- Temperature (GPU and memory)
- Power draw and power limit (watts)
- Fan speed, SM/memory clocks
- Running processes per GPU
- Driver version and CUDA version

### Deploying the sidecar

The sidecar must run on each physical GPU server. It requires NVIDIA drivers and the NVIDIA Container Toolkit. A `SIDECAR_SECRET_KEY` environment variable is **required** — the sidecar will refuse to start without it. Generate one with: `python -c "import secrets; print(secrets.token_hex(32))"`

#### NVIDIA Container Toolkit

The sidecar container requires GPU access via `--gpus all`. Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on each GPU node:

```bash
# Debian/Ubuntu
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

# RHEL/Rocky Linux
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
  | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
sudo dnf install -y nvidia-container-toolkit

# Configure and restart Docker
sudo nvidia-ctk runtime configure --driver=docker
sudo systemctl restart docker

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

#### Docker network prerequisites

Docker's default bridge network uses `172.17.0.0/16`, which can collide with campus or institutional routing. Configure each GPU node's Docker daemon to use `10.x.x.x` address space before deploying the sidecar:

```bash
# /etc/docker/daemon.json — create or merge on each GPU node
{
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    },
    "bip": "10.77.0.1/16",
    "default-address-pools": [
        { "base": "10.78.0.0/16", "size": 24 }
    ]
}
```

After creating or updating `daemon.json`, restart Docker: `sudo systemctl restart docker`

#### Deployment options

**Option A: Docker Compose (development)**

```bash
# Set the key in .env or export it
export SIDECAR_SECRET_KEY=your-generated-key
# Start with the gpu profile (requires NVIDIA GPU on this machine)
docker compose --profile gpu up gpu-sidecar
```

**Option B: Build directly from GitHub (production — run on each GPU node)**

Build and deploy a specific version directly from the repository without cloning.

First, create the sidecar configuration directory and env file (once per node):

```bash
sudo mkdir -p /etc/mindrouter
python3 -c "import secrets; print('SIDECAR_SECRET_KEY=' + secrets.token_hex(32))" | sudo tee /etc/mindrouter/sidecar.env
sudo chmod 600 /etc/mindrouter/sidecar.env
```

Then build and run:

```bash
# Build a specific release tag
docker build -t mindrouter-sidecar:v2.0.0 \
  -f Dockerfile.sidecar \
  https://github.com/ui-insight/MindRouter.git#v2.0.0:sidecar

# Or build latest from master
docker build -t mindrouter-sidecar:latest \
  -f Dockerfile.sidecar \
  https://github.com/ui-insight/MindRouter.git:sidecar

# Run bound to localhost only (nginx will proxy external traffic)
docker run -d --name gpu-sidecar \
  --gpus all \
  -p 127.0.0.1:18007:8007 \
  --env-file /etc/mindrouter/sidecar.env \
  --restart unless-stopped \
  mindrouter-sidecar:v2.0.0
```

To upgrade an existing sidecar to a new version:

```bash
docker build -t mindrouter-sidecar:v2.0.0 \
  -f Dockerfile.sidecar \
  https://github.com/ui-insight/MindRouter.git#v2.0.0:sidecar
docker stop gpu-sidecar && docker rm gpu-sidecar
docker run -d --name gpu-sidecar \
  --gpus all \
  -p 127.0.0.1:18007:8007 \
  --env-file /etc/mindrouter/sidecar.env \
  --restart unless-stopped \
  mindrouter-sidecar:v2.0.0
```

Then configure nginx to proxy external port 8007 to the container:

```nginx
# /etc/nginx/conf.d/sidecar-proxy.conf
server {
    listen 8007;
    listen [::]:8007;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:18007;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Sidecar-Key $http_x_sidecar_key;
        proxy_connect_timeout 5s;
        proxy_read_timeout 10s;
    }
}
```

```bash
sudo systemctl enable --now nginx
```

**Option C: Direct Python (no Docker)**

```bash
# On each GPU server:
pip install fastapi uvicorn nvidia-ml-py
cd sidecar/
SIDECAR_SECRET_KEY=your-generated-key GPU_AGENT_PORT=8007 python gpu_agent.py
```

### Registering nodes with sidecars

After starting the sidecar on a GPU server, register the node in MindRouter. Include the same `sidecar_key` that was set as `SIDECAR_SECRET_KEY` on the sidecar:

```bash
curl -X POST http://mindrouter:8000/api/admin/nodes/register \
  -H "Authorization: Bearer admin-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "gpu-server-1",
    "hostname": "gpu1.example.com",
    "sidecar_url": "http://gpu1.example.com:8007",
    "sidecar_key": "your-generated-key"
  }'
```

Or use the admin dashboard at `/admin/nodes` to register nodes and view GPU telemetry.

### Multi-backend nodes

A single GPU server can host multiple inference endpoints. Assign specific GPUs to each backend using `gpu_indices`:

```
Node: gpu-server-1 (4x A100-80GB, sidecar at :8007)
├── Backend: vllm-large  (gpu_indices: [0, 1])  ← uses GPUs 0-1
├── Backend: vllm-small  (gpu_indices: [2])      ← uses GPU 2
└── Backend: ollama-misc (gpu_indices: [3])      ← uses GPU 3
```

Each backend's telemetry (utilization, memory) is aggregated only from its assigned GPUs. Omit `gpu_indices` to use all GPUs on the node.

## Security

- API keys are stored hashed (Argon2)
- Role-based access control (RBAC)
- Rate limiting per key (RPM) and per user (concurrency)
- All admin actions are audited
- Request/response content logged for compliance review
- GPU sidecar endpoints are authenticated via shared secret key (`X-Sidecar-Key` header, constant-time comparison)
- Sidecar containers bind to localhost only; nginx reverse proxy handles external access on port 8007
- Docker daemon on GPU nodes uses 10.x.x.x address space to avoid routing collisions with campus/institutional networks

## License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

Please follow conventional commit messages.

## Citing

If you use MindRouter in your work, please cite:

> Sheneman, L. (2026). MindRouter: Open-source LLM inference gateway for institutional AI sovereignty. In *Proceedings of the Practice and Experience in Advanced Research Computing Conference 2026 (PEARC '26)*. Association for Computing Machinery. [https://doi.org/10.1145/3785462.3815861](https://doi.org/10.1145/3785462.3815861)

## Acknowledgments

<img src="https://github.com/user-attachments/assets/d2b43c22-84f6-4912-91dc-8b081d9e2c6f" alt="NSF Logo" width="157">

This project was developed with support from the National Science Foundation ([NSF Award #2427549](https://www.nsf.gov/awardsearch/show-award?AWD_ID=2427549)).
