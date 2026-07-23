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
| `run tool tests`    | `make test-tools`  | Live tool calling compliance (all tool-capable models) |
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
| `backend/app/tests/unit/test_accessibility.py` | 117 | WCAG 2.1 Level A/AA: ARIA, semantic HTML, heading hierarchy, forms, sidebar include, Video tab + admin Video config (sidebar include, table scope/caption) |
| `backend/app/tests/unit/test_chat_mobile.py` | 37 | Chat mobile responsiveness: sidebar collapse/backdrop, thinking block collapse, compact layout CSS |
| `backend/app/tests/unit/test_rerank_translators.py` | 22 | Rerank/score translators: OpenAIIn, VLLMOut rerank & score methods, canonical schema validation |
| `backend/app/tests/unit/test_model_enrichment.py` | 28 | Model auto-enrichment: brave_web_search api_key param, LLM call helper, enrichment pipeline, CRUD helpers, config gating |
| `backend/app/tests/unit/test_voice_api.py` | 34 | Public voice API: TTSRequest validation, quota check, request recording, TTS endpoint (happy path, errors, content-type), STT endpoint (happy path, errors, timeout, language, model), Modality enum |
| `backend/app/tests/unit/test_dlp.py` | 22 | DLP scanner: regex (SSN, CC, email, keywords, custom patterns), severity classification, text extraction (messages, images, response), LLM prompt construction, ScanResult/ScanFinding dataclasses |
| `backend/app/tests/unit/test_latex_normalize.py` | 29 | LaTeX normalization: $$-block preservation (v2.4.2 regression), bare command/operator wrapping, display math promotion, inline preservation, mixed content, code block immunity, \\begin/\\end environments |
| `backend/app/tests/unit/test_ocr.py` | 27 | OCR pipeline: chunking logic, fence stripping, prompt building, overlap detection, deterministic merge, image conversion, PDF fixture |
| `backend/app/tests/unit/test_responses_in.py` | 46 | ResponsesIn translator: input polymorphism (string/items/typeless), function-call round trip via call_id, flat tool re-nesting + non-function strip, text.format, reasoning.effort→think, truncation flag, format_response/build_snapshot, vLLM round trip |
| `backend/app/tests/unit/test_responses_stream.py` | 12 | Chat-SSE→Responses-SSE adapter: canonical event sequences (text/reasoning/tools), deferred terminal + usage harvesting, incomplete/failed terminals, exception hardening, drain contract |
| `backend/app/tests/unit/test_responses_api.py` | 42 | /v1/responses routes: feature flag, validation errors, OpenAI error envelopes, quota pre-flight (skip_quota_check), streaming/non-streaming dispatch, alias resolution, store persistence, previous_response_id chains, GET/DELETE/input_items/cancel, hosted web_search dispatch, count input_tokens, conversation integration |
| `backend/app/tests/unit/test_conversations_api.py` | 13 | Conversations API routes: conversation CRUD (create/seed/cap/update/delete), item endpoints (create/list/get/delete envelopes), owner-scoped 404s, flag gating |
| `backend/app/tests/unit/test_blog_email_render.py` | 8 | Blog-email HTML: [TOC] stripping, inline black table borders (alignment merge, thead guard), bordered code containers |
| `backend/app/tests/unit/test_website_publisher.py` | 9 | mindrouter.ai GitHub publisher (fake GitHub client + storage): enabled gating (config + repo allowlist), hard repo allowlist enforcement (rejects non-mindrouter-website), publish/unpublish filesets (page/index/feed/css/images; missing-image skip), atomic Git-Data commit sequence (ref→commit→blobs→tree→commit→ref), null-sha deletions, API-error propagation |
| `backend/app/tests/unit/test_field_validation.py` | 6 | Request-field validation (off/log/enforce): dialect fields (structured_outputs→response_format hint) + unknown fields 400 in enforce, accepted/ignored fields pass, log/off never raise, default reads setting, stream_options.include_usage parsed into canonical |
| `backend/app/tests/unit/test_install_friction_fixes.py` | 8 | Install-friction fixes from field feedback: multimodal heuristic covers qwen3.6/dots (#6), sync-script drops nonexistent supports_embeddings (#10a), vLLM streaming injects include_usage + real-usage accounting (#10b), opt-in RUN_MIGRATIONS at startup before init_registry (#1), automation-friendly admin seed (ADMIN_PASSWORD/ADMIN_API_KEY/MINT_ADMIN_KEY + single-line key) (#2a), OCR degrade-to-single-page on image-limit 400 (#3) |
| `backend/app/tests/unit/test_blog_website_publish.py` | 4 | Selective mindrouter.ai publish contract (source checks): migration 064 add/drop of website_published/website_published_at/website_commit_sha, BlogPost fields, CRUD get_website_published filters (selected+published+undeleted), website-publish/unpublish routes (admin guard, is_published gate, state transitions) |
| `backend/app/tests/unit/test_blog_export.py` | 16 | Blog static-HTML export for mindrouter.ai: markdown/codehilite/tables render, image path collection (dedup/order/external-skip), description derivation (excerpt vs markdown-stripped), post page (canonical, OG article, JSON-LD, site shell, byline, title escaping + JSON-LD script-breakout guard), index/RSS render, async image fetch (present/missing), export_post combine |
| `backend/app/tests/unit/test_responses_store.py` | 20 | Responses store service: item id stamping, image offload/re-inflate + path containment, chain rebuild + item_reference, payload/row caps, persist contract, crud/migration/retention source checks |
| `backend/app/tests/unit/test_responses_websearch.py` | 11 | Hosted web_search: tool detection/synthetic tool, non-streaming loop (threading, budget, client passthrough), streaming loop (suppression, ws events, cross-round sequencing), error terminal |
| `backend/app/tests/unit/test_context_trim.py` | 6 | truncation:"auto" trimming: turn grouping, tool-pair atomicity, oldest-first drops, system/final-turn protection |
| `backend/app/tests/unit/test_video_schema_contract.py` | 9 | Video-gen v1 foundation (source-inspection + spec-load, pollution-proof): CanonicalVideoRequest defaults + text-to-video-only shape, CanonicalVideoJob OpenAI shape (in_progress status set), models.py video enums (BackendEngine.VIDEO, Modality.VIDEO_GENERATION, users.video_generation_enabled), JobModality.VIDEO_GENERATION, migration 065 ENUM widening (ALGORITHM=INSTANT) + user flag, migration 066 four tables + claim/heartbeat indexes + source_clip provisions, migration 067 config seed (vid.enabled False by default, cap 50, retention), 065→066→067 chain linear, every video_* setting has docker-compose passthrough |
| `backend/app/tests/unit/test_video_field_validation.py` | 8 | Video request-field validation dialect (spec-loaded): enforce accepts all v1 fields, rejects duration/width/height typos with hints pointing to seconds/size, rejects v1-unsupported conditioning (image/input_reference/first_frame/storyboard), rejects unknown fields, ignored fields pass, log/off never raise, VIDEO_ACCEPTED drift-guard vs CanonicalVideoRequest |
| `backend/app/tests/unit/test_video_runner.py` | 19 | VideoRunner state machine (in-memory fake repo + scriptable fake worker, spec-loaded): happy path claim→submit→poll→fetch→complete with token/duration accounting + shot rendering/rendered transitions; no-backend requeues (not fails); cancel before submit; cancel during poll (worker cancel + shot skipped); worker-reported failure fails without retry; transient submit error retries under cap / fails over cap; non-retryable submit fails immediately; tick() empty-queue False; tick() processes claimed job; run_forever reconciles then stops on cancel. **Reconciliation (no stuck jobs):** ground-truth poll of the worker for stale RENDERING jobs — completed→recover output, failed→fail, still-rendering→resume to completion, worker-lost→requeue under cap / fail over cap, wall-timeout→fail, pre-submit(no backend_job_id)→requeue, readopt-lost-to-peer→no-op |
| `backend/app/tests/unit/test_video_api.py` | 22 | /v1/videos routes (v1 text-to-video, spec-loaded with save/restore sys.modules hygiene): _job_to_dict status mapping (rendering→in_progress) + content_url gating; create_video gates — disabled 503, user-flag-off 403, missing-prompt 400, disallowed size/duration 400, bad quality 400, model-not-found 404, over-concurrency 429, over-storage-cap 507 (+under-cap proceeds, cap=0 disables); happy path returns 'queued' + persists + non-blocking; get_video 404 (no existence leak); cancel flags cancel; /videos/models capability shape (t2v only, max_shots 1); GET /content — 404 missing job, 409 not-ready, 404 file-missing, FileResponse stream (Range/206 via starlette) |
| `backend/app/tests/unit/test_diffusion_img2img.py` | 5 | img2img (reference-edit) request translation: DiffusionOutTranslator omits image/strength for txt2img, passes base64 reference list + strength for edits, image-without-strength case, empty image list stays txt2img (no key), CanonicalImageRequest image/strength default None |
| `backend/app/tests/unit/test_image_policy_edit.py` | 5 | Edit-aware content-policy judging: edit user-template carries the reference-image/anti-ambiguity note (plain template does not), evaluate_prompt forwards is_edit True/False to _call_judge, no-policy short-circuits PASS even for edits — regression for "put glasses on this man" being FAILED as ambiguous |

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

Exercises every API surface. Sections: `health`, `auth`, `openai`, `ollama`, `anthropic`, `cross`, `errors`, `admin`, `rerank`, `responses`.

| File | What it covers |
|------|----------------|
| `test.py` | Health endpoints, authentication, OpenAI-compatible API, Ollama-compatible API, Anthropic-compatible API, cross-engine routing, error handling, admin API, reranker (basic, top_n, return_documents), Responses API (non-streaming, typed-SSE streaming, function-call round trip, auth, store/retrieve/chain/delete, input_tokens count, Conversations lifecycle) |

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

## 7. Live Tool Calling Compliance Tests

**Runner:** `python tests/test_tool_calling_live.py`
**Makefile:** `make test-tools`
**Requirements:** Live deployment, `MINDROUTER_API_KEY` env var set.

Auto-discovers all tool-capable models via `/v1/models` (filtering by `capabilities.tools`), then tests each model across OpenAI, Ollama, and Anthropic API styles with both streaming and non-streaming requests. Validates that models return proper `tool_calls` with correct function names and parseable arguments.

| File | What it covers |
|------|----------------|
| `tests/test_tool_calling_live.py` | N models x 3 API styles x 2 stream modes: tool call generation, argument parsing, function name correctness, streaming tool call accumulation across OpenAI `/v1/chat/completions`, Ollama `/api/chat`, and Anthropic `/anthropic/v1/messages` |

**Models tested:** All models with `capabilities.tools == true` (auto-discovered, excludes embeddings/rerankers)

---

## 8. Accessibility Tests

**Runner:** `pytest backend/app/tests/unit/test_accessibility.py -v`
**Makefile:** `make test-a11y`
**Requirements:** None (parses template files directly).

Subset of unit tests, broken out for convenience. 117 tests validating WCAG 2.1 Level A and AA compliance across all Jinja2 HTML templates (including the Video tab and admin Video config, sidebar include, user detail, groups, API keys, and data retention).

---

## 9. GPU Sidecar Tests

**Runner:** `pytest sidecar/tests/ -v`
**Makefile:** `make test-sidecar`
**Requirements:** None (mocks pynvml).

| File | What it covers |
|------|----------------|
| `sidecar/tests/test_gpu_agent.py` | GPU agent unit tests: pynvml mocking, GPU info, auth, health |
| `sidecar/tests/test_gpu_agent_stress.py` | 60-second concurrent auth stress test for sidecar |

---

## 9b. Video Worker Service Tests

**Runner:** `cd video-worker && pytest tests/ -q`
**Requirements:** `video-worker/requirements.txt` only (fastapi/uvicorn/httpx/pytest). Runs in **mock mode — no GPU, no torch, no ltx_pipelines**; the worker's GPU deps are a separate venv on the H200 node and are NOT needed for these tests. Do not fold this into `make test-unit` (separate venv, own deps).

| File | What it covers |
|------|----------------|
| `video-worker/tests/test_worker.py` | LTX-2.3 worker async contract (10 tests, FastAPI TestClient, mock engine): capabilities/models/version; submit→poll→completed→fetch full lifecycle; content 409 before complete; **/health responsive (<1s) while a render occupies the executor** (off-event-loop invariant); Range request → 206 + Accept-Ranges; disallowed size/duration/missing-prompt → 400; unknown job → 404 (poll + cancel); cancel a still-queued job |

---

## 10. Security / Vulnerability Tests

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

## 11. vLLM MTP Speculative-Decoding Benchmarks (ops — on GPU nodes)

> **Not part of the repo test suite.** Manual throughput/latency benchmark run on the
> vLLM GPU hosts; harness lives under `/data/vllm/` (lynx) or `/zdata/data/vllm/`
> (aspen NFS), not in this repo. No Makefile target.

**Harness:** `vllm_bench_spec.py URL MODEL LABEL [conc_csv]` — concurrency sweep
(default `1,2,4,8,16`), temp=0, `ignore_eos`, 256 out-tokens, ~650-tok prompt; reports
median decode tok/s **per stream** (single-stream latency) and **aggregate** tok/s (server
throughput). `launch-gpu3-qwen-bench.sh <num_spec>` boots a serve on a non-prod port
(`0`=baseline, `N`=MTP depth); `run-qwen-full.sh` drives the baseline→mtp sweep with
orphan-safe GPU teardown. Enable built-in MTP via
`--speculative-config '{"method":"mtp","num_speculative_tokens":N}'`.

**qwen3.6-27b MTP depth sweep** (2026-07-20, lynx-gpu3 H200, vLLM 0.25.1, gpu-mem 0.95, 16K ctx, exclusive GPU):

| depth | conc1 single-stream (tok/s) | conc1 AGG (×base) | conc16 AGG | acceptance |
|-------|-----------------------------|-------------------|-----------|------------|
| baseline | 87 | 85 (1.00×) | 1003 | — |
| mtp-1 | 131 | 127 (1.49×) | 1310 | len 1.93 / 93% |
| mtp-2 | 179 | 171 (2.01×) | 1623 | len 2.68 / 84% |
| **mtp-3** | **211** | **200 (2.35×)** | **1729** | len 3.18 / 73% |
| mtp-4 | 220 | 208 (2.45×) | 1692 | len 3.69 / 67% |

**Finding:** MTP is monotonically faster at every concurrency (1→16) through depth-3, then
plateaus (mtp-4 ≈ mtp-3, worse per-token efficiency). No regression. **mtp-3 is the deployed
optimum** on both qwen3.6-27b backends (lynx gpu1/gpu3). Caveat: an idle stray vLLM unit
sharing the GPU poisoned an earlier run — always bench on an **exclusive** GPU.

**qwen3.5-122b MTP depth sweep** (2026-07-21, aspen1-gpu0 H200, vLLM 0.25.1, 16K ctx, max-num-seqs 8):

| depth | mean accept length (tok / target-forward) | avg accept | AGG conc8 |
|-------|------------------------------------------|-----------|-----------|
| 0.23 baseline | — | — | 673 |
| mtp-1 | 1.83 | 82.8% | 921 |
| **mtp-2** | **2.43** | 71.3% | **1170** |
| mtp-3 | 1.88 | **29.4%** | 669 |

**Finding — optimal depth is per-model, always sweep it:** the 122b's MTP head is healthy at
depth 1–2 (83%/71%) but **collapses at depth 3 (29%)**, where it drafts 7,104 tokens to accept
2,086 and yields *no more* tokens/forward than mtp-1 — landing back at the 0.23 baseline. **mtp-2
is the deployed optimum** for the 122b, while qwen3.6-27b/35b hold 73–85% at depth 3 and use mtp-3.

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
