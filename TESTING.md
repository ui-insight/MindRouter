# MindRouter — Test Manifest

> **Single source of truth** for every test in the project.
> When adding new tests, **add an entry here** so `run all tests` stays accurate.

---

## Quick Reference

| Shorthand             | Makefile target    | What it runs                                         |
|-----------------------|--------------------|------------------------------------------------------|
| `run unit tests`      | `make test-unit`   | All pytest unit tests                                |
| `run integration tests` | `make test-int` | Live-backend integration tests                       |
| `run e2e tests`       | `make test-e2e`    | E2E chat subsystem tests                             |
| `run smoke tests`     | `make test-smoke`  | API smoke test against live deployment               |
| `run stress tests`    | `make test-stress` | Multi-user load/stress test                          |
| `run matrix tests`  | `make test-matrix` | Structured output matrix tests (all API styles)    |
| `run thinking tests` | `make test-thinking` | Structured output + thinking compliance (live stack) |
| `run accessibility tests` | `make test-a11y` | WCAG 2.1 accessibility tests (subset of unit)      |
| `run sidecar tests`   | `make test-sidecar`| GPU sidecar agent tests                              |
| `run all tests`       | `make test-all`    | Unit + integration + E2E + smoke + sidecar tests     |
| `run coverage`        | `make coverage`    | Unit + integration tests with coverage report        |

---

## 1. Unit Tests

**Runner:** `pytest backend/app/tests/unit/ -v`
**Makefile:** `make test-unit`
**Requirements:** No live services needed. Some tests (test_quota, test_scheduler) need pymysql.

| File | Tests | What it covers |
|------|-------|----------------|
| `backend/app/tests/unit/test_translators.py` | 12 | OpenAIIn, OllamaIn, OllamaOut, VLLMOut translator static methods |
| `backend/app/tests/unit/test_translation_roundtrip.py` | 33 | Round-trip: OpenAI ↔ Canonical ↔ Ollama parameter mapping, vision edge cases, embedding gaps, edge cases |
| `backend/app/tests/unit/test_hyperparameter_fidelity.py` | 33 | Extended sampling params (top_k, repeat_penalty, min_p), backend_options passthrough, thinking mode |
| `backend/app/tests/unit/test_completions.py` | 22 | /v1/completions and /api/generate translation, to_chat_request conversion, completion output |
| `backend/app/tests/unit/test_structured_outputs.py` | 22 | JSON schema validation, structured output for Ollama & OpenAI |
| `backend/app/tests/unit/test_streaming.py` | 20 | ndjson (Ollama) and SSE (vLLM/OpenAI) stream parsing |
| `backend/app/tests/unit/test_validators.py` | 28 | Input validation: request params, message schemas, constraints |
| `backend/app/tests/unit/test_cross_engine_routing.py` | 15 | Ollama ↔ vLLM routing with parameter translation |
| `backend/app/tests/unit/test_circuit_breaker.py` | 5 | CircuitBreakerState: open/half-open, failure counts, reset |
| `backend/app/tests/unit/test_latency_tracker.py` | 8 | Latency tracking, p50/p99 percentile calculations |
| `backend/app/tests/unit/test_retry_failover.py` | 12 | Retry logic, exponential backoff, backend failover |
| `backend/app/tests/unit/test_scheduler.py` | 30 | Fair-share WDRR scheduler, job queue, BackendScorer, HardConstraints |
| `backend/app/tests/unit/test_quota.py` | 24 | Quota management, token accounting, RPM/concurrent limits, group-based defaults |
| `backend/app/tests/unit/test_anthropic_translator.py` | 19 | AnthropicIn translator: Messages API request/response, multimodal, thinking, streaming format |
| `backend/app/tests/unit/test_tool_calling.py` | 33 | Tool calling: schemas, OpenAI/Ollama/Anthropic inbound, vLLM/Ollama outbound, round-trips |
| `backend/app/tests/unit/test_sidecar_client.py` | 16 | GPU sidecar client: auth, GPU info retrieval, communication |
| `backend/app/tests/unit/test_version_alignment.py` | 6 | Version alignment: pyproject.toml reading, sidecar VERSION file consistency |
| `backend/app/tests/unit/test_accessibility.py` | 99 | WCAG 2.1 Level A/AA: ARIA, semantic HTML, heading hierarchy, forms, sidebar include |
| `backend/app/tests/unit/test_chat_mobile.py` | 37 | Chat mobile responsiveness: sidebar collapse/backdrop, thinking block collapse, compact layout CSS |
| `backend/app/tests/unit/test_rerank_translators.py` | 22 | Rerank/score translators: OpenAIIn, VLLMOut rerank & score methods, canonical schema validation |
| `backend/app/tests/unit/test_model_enrichment.py` | 28 | Model auto-enrichment: brave_web_search api_key param, LLM call helper, enrichment pipeline, CRUD helpers, config gating |
| `backend/app/tests/unit/test_voice_api.py` | 34 | Public voice API: TTSRequest validation, quota check, request recording, TTS endpoint (happy path, errors, content-type), STT endpoint (happy path, errors, timeout, language, model), Modality enum |
| `backend/app/tests/unit/test_dlp.py` | 22 | DLP scanner: regex (SSN, CC, email, keywords, custom patterns), severity classification, text extraction (messages, images, response), LLM prompt construction, ScanResult/ScanFinding dataclasses |
| `backend/app/tests/unit/test_latex_normalize.py` | 29 | LaTeX normalization: $$-block preservation (v2.4.2 regression), bare command/operator wrapping, display math promotion, inline preservation, mixed content, code block immunity, \\begin/\\end environments |
| `backend/app/tests/unit/test_ocr.py` | 27 | OCR pipeline: chunking logic, fence stripping, prompt building, overlap detection, deterministic merge, image conversion, PDF fixture |

**Shared fixtures:** `backend/app/tests/conftest.py`

---

## 2. Integration Tests

**Runner:** `pytest backend/app/tests/integration/ -v`
**Makefile:** `make test-int`
**Requirements:** Live Ollama and vLLM backends (configure URLs in test constants).

| File | What it covers |
|------|----------------|
| `backend/app/tests/integration/test_live_backends.py` | Full translation pipeline against real Ollama and vLLM backends — streaming and non-streaming chat |
| `backend/app/tests/integration/test_rag_pipeline.py` | RAG pipeline: embedding, reranking, scoring endpoints through MindRouter proxy, end-to-end RAG test |
| `backend/app/tests/integration/test_structured_output_matrix.py` | Structured output matrix: all combos of API style (OpenAI/Ollama/Anthropic) × format (text/json_object/json_schema) × thinking mode × streaming across model categories |
| `backend/app/tests/integration/test_structured_outputs_live.py` | Live structured output: 5 models × 6 schema types × 3 API surfaces × 2 streaming modes + cross-engine routing (`--api-key`, `--base-url` CLI args) |

---

## 3. End-to-End Tests

**Runner:** `python tests/e2e_chat.py <args>`
**Makefile:** `make test-e2e`
**Requirements:** Live Docker stack, valid user credentials.

| File | What it covers |
|------|----------------|
| `tests/e2e_chat.py` | Chat subsystem: persistence, image preprocessing, storage, multi-turn context, vision model Q&A, cross-user isolation, CRUD operations |

**CLI arguments:**
```
--base-url       http://localhost:8000
--username       Primary user username
--password       Primary user password
--text-model     Text model ID (required, e.g. phi4:14b)
--vision-model   Vision model ID (e.g. qwen2.5-VL-32k:7b)
--username2      Second user for cross-user isolation tests
--password2      Second user password
--cookie-file    Session cookie file path
--cookie-file2   Second user cookie file
--skip-vision    Skip vision model tests
--docker-container  Container name for direct inspection
```

---

## 4. Smoke Tests (API)

**Runner:** `python test.py <args>`
**Makefile:** `make test-smoke`
**Requirements:** Live deployment, valid API key.

Exercises every API surface. Sections: `health`, `auth`, `openai`, `ollama`, `anthropic`, `cross`, `errors`, `admin`, `rerank`.

| File | What it covers |
|------|----------------|
| `test.py` | Health endpoints, authentication, OpenAI-compatible API, Ollama-compatible API, Anthropic-compatible API, cross-engine routing, error handling, admin API, reranker (basic, top_n, return_documents) |

**CLI arguments:**
```
--api-key        API key (required)
--base-url       http://localhost:8000
--admin-key      Admin API key (enables admin section)
--ollama-model   phi4:14b
--vllm-model     openai/gpt-oss-120b
--embedding-model EMBED/all-minilm:33m
--rerank-model   Qwen/Qwen3-Reranker-8B
--timeout        Request timeout in seconds (180)
--section        Run specific section(s) only
```

---

## 5. Stress / Load Tests

**Runner:** `python stress.py <args>`
**Makefile:** `make test-stress`
**Requirements:** Live deployment, admin API key for user provisioning.

| File | What it covers |
|------|----------------|
| `stress.py` | Multi-user concurrent load: fair-share WDRR scheduler, throughput, latency percentiles, error rates |

**CLI arguments:**
```
--api-key        Admin API key (required)
--base-url       http://localhost:8000
--duration       Test duration in seconds (300)
--concurrency    Concurrent request workers (10)
--users          Number of test users to provision (6)
--ollama-model   phi4:14b
--vllm-model     openai/gpt-oss-120b
--embedding-model EMBED/all-minilm:33m
--max-tokens     Max tokens per request (32)
--timeout        Per-request timeout in seconds (180)
--chat-only      Only send chat requests (no embeddings)
--verbose        Print individual request results
```

---

## 6. Structured Output + Thinking Compliance Tests

**Runner:** `python tests/test_structured_thinking.py`
**Makefile:** `make test-thinking`
**Requirements:** Live deployment, `MINDROUTER_API_KEY` env var set.

Comprehensive matrix test covering structured output (JSON schema validation) and thinking/reasoning mode control across all 4 API surfaces, 6 models, and all reasoning modes (ON/OFF/low/medium/high/N/A). Runs 3 replicates per combination (144 total requests at 10 concurrency).

| File | What it covers |
|------|----------------|
| `tests/test_structured_thinking.py` | 12 model×reasoning combos × 4 endpoints × 3 replicates: JSON schema validation, thinking detection, thinking mode match (ON/OFF/effort levels) across OpenAI, Ollama chat, Ollama generate, and Anthropic endpoints |

**Models tested:** openai/gpt-oss-120b, gpt-oss-32k:120b, qwen/qwen3.5-400b, qwen3-32k:32b, qwen2.5-8k:7b, phi4:14b

---

## 7. Accessibility Tests

**Runner:** `pytest backend/app/tests/unit/test_accessibility.py -v`
**Makefile:** `make test-a11y`
**Requirements:** None (parses template files directly).

Subset of unit tests, broken out for convenience. 99 tests validating WCAG 2.1 Level A and AA compliance across all 19 Jinja2 HTML templates (including sidebar include, user detail, groups, API keys, and data retention).

---

## 8. GPU Sidecar Tests

**Runner:** `pytest sidecar/tests/ -v`
**Makefile:** `make test-sidecar`
**Requirements:** None (mocks pynvml).

| File | What it covers |
|------|----------------|
| `sidecar/tests/test_gpu_agent.py` | GPU agent unit tests: pynvml mocking, GPU info, auth, health |
| `sidecar/tests/test_gpu_agent_stress.py` | 60-second concurrent auth stress test for sidecar |

---

## 9. Security / Vulnerability Tests

> **Status:** Not yet implemented.

Planned coverage:
- [ ] SQL injection on API endpoints
- [ ] XSS in dashboard templates
- [ ] CSRF token validation
- [ ] API key brute-force rate limiting
- [ ] Header injection
- [ ] Path traversal via file upload
- [ ] Auth bypass / privilege escalation
- [ ] Dependency vulnerability scan (`pip-audit`)

---

## Configuration Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | pytest paths, asyncio mode, markers, coverage config |
| `backend/app/tests/conftest.py` | Shared fixtures: mock settings, backends, users, API keys, streaming data |
| `Makefile` | All `make test-*` targets |

---

## Adding New Tests

When you create a new test file:

1. **Add the file** to the appropriate section in this manifest.
2. **Update the test count** in the table.
3. If it's a new *category*, add a Makefile target and a row to the Quick Reference table.
4. Ensure `conftest.py` has any new shared fixtures.
