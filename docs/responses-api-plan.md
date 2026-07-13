# MindRouter `/v1/responses` (OpenAI Responses API) — Implementation Plan

**Goal:** a native Responses API dialect so the ChatGPT desktop app's Codex agent (and OpenAI SDKs) can use MindRouter as a custom provider (`wire_api = "responses"`).

**Tier 1** = stateless `/v1/responses` (request/response/stream translation, function-tool passthrough, reasoning items). **Tier 2** = server-side state (`store`, `previous_response_id`, GET/DELETE/input_items/cancel, retention).

*Provenance: grounded in the Codex Rust source (tag rust-v0.144.3), the official `openai/openai-openapi` spec (2026-05-13), raw wire captures of api.openai.com/v1/responses, and a full read of the MindRouter dialect/inference/DB/test subsystems; then adversarially reviewed against all three. Line references are against branch `fix/ocr-memory-bound-and-pdf-400`.*

---

## 0. Ground truth that shapes this design

1. **Codex never sends `previous_response_id` over HTTP.** The field doesn't exist in its `ResponsesApiRequest` struct. It resends the **full transcript every request** with `store:false` (store is `true` only for auto-detected Azure endpoints). → **Tier 1 alone makes Codex work.** Tier 2 is for OpenAI-SDK ecosystem compatibility (SDK default `store:true`, `client.responses.retrieve(...)`, chaining).
2. **Codex's minimal event needs:** it accumulates items from `response.output_item.done` and terminates on `response.completed` (which **must** carry `response.id`; `usage` optional). Deltas are UI-only. Unknown events ignored. **EOF without a terminal event = error → up to 5 full re-POST retries** (retry-storm risk).
3. **Strict-client invariants** (wire captures + a real proxy-breaks-Codex bug): `response.created` (seq 0) then `response.in_progress` (seq 1) first; `content_part.added` **before** any `output_text.delta`; `sequence_number` on every event, monotonic from 0; `item_id` stable per item; items never interleave; usage only in the terminal snapshot; **no `data: [DONE]`** — the terminal event is the last frame, then close. Framing: `event: <type>\ndata: <compact-json>\n\n`.
4. **Codex sends tools we must tolerate:** `function` tools (`shell_command`, `update_plan`, `view_image`, MCP) in the **flat** shape (`{type:"function", name, description, parameters, strict}` — no nested `function:{}`), plus **`{"type":"web_search","external_web_access":false}` by default** (config default `web_search="cached"`). Non-function tools must be **stripped, not errored**. Also: `tool_choice:"auto"`, `parallel_tool_calls:false`, `stream:true`, `include:["reasoning.encrypted_content"]` when reasoning summaries are on, `prompt_cache_key:<thread-id>` always, and a non-standard `client_metadata` object to ignore gracefully.
5. **Codex echoes back whatever items we emit** — `reasoning` items (with `encrypted_content:null`), `function_call` items — with `id` stripped (store=false default) but `call_id` preserved. When store=true (Azure) or the `item_ids` feature is on, items arrive **with** `rs_/fc_/msg_` ids. SDK users also send the bare `{"role","content"}` form with **no `type` key** (EasyInputMessage — `type` is optional in the spec and the SDK's most common form).
6. **MindRouter internals:** `InferenceService.chat_completion()` returns an OpenAI `chat.completion` dict; `stream_chat_completion()` yields OpenAI-chunk SSE bytes ending `data: [DONE]`. The Anthropic dialect (`anthropic_api.py` + `anthropic_in.py`) is the template — including the **drain contract** (anthropic_api.py:169-174): after finishing your own protocol, keep consuming the inner stream to exhaustion or `_complete_streaming_request` never runs and backend capacity leaks. The audit `endpoint` string is hardcoded `"/v1/chat/completions"` inside both service methods (inference.py:285, :332). `request.request_id` is overwritten with the DB `request_uuid` (inference.py:289), so `resp_*` public ids are applied by the dialect layer. `_check_quota` is **not** a pure check — `check_rpm` INCRs the Redis counter on every call (redis_client.py:259-271). vLLM's `include_usage` final chunk has **empty choices and arrives after the finish_reason chunk**.

---

## Tier 1 — Stateless `/v1/responses`

### 1.1 New file: `backend/app/core/translators/responses_in.py`

`ResponsesInTranslator` — static methods, imports **only stdlib + canonical_schemas** (plain-import unit tests, like `anthropic_in.py`; no fastapi imports here — error-response helpers live in the route).

**`ResponsesRequestContext`** (dataclass, same module): everything needed to echo request params in Response snapshots: raw `model` string, `tools` (as received), `tool_choice`, `temperature`, `top_p`, `max_output_tokens`, `parallel_tool_calls`, `reasoning`, `text`, `store`, `previous_response_id`, `truncation`, `metadata`, `instructions`, `prompt_cache_key`, public `response_id` (`resp_<uuid4().hex>`), `created_at`. Snapshot echo uses **spec defaults, not nulls, for omitted fields**: `parallel_tool_calls→true`, `store→true`, `truncation→"disabled"`, `text→{"format":{"type":"text"}}`, `temperature→1.0`, `top_p→1.0`.

**`translate_responses_request(body: dict) -> CanonicalChatRequest`:**

| Responses field | Canonical mapping |
|---|---|
| `model` | `model` (missing → 400) |
| `instructions` | prepended `system` message |
| `input` (string) | one `user` message |
| `input` (item array) | item table below |
| `tools[]` type `function` (flat) | `CanonicalToolDefinition(type="function", function={"name","description","parameters"})` — re-nest |
| `tools[]` other types (`web_search`, `custom`, `mcp`, …) | **strip + `logger.info("responses_tool_stripped", type=...)`** — never error |
| `tool_choice` `"auto"/"none"/"required"` | passthrough |
| `tool_choice` `{"type":"function","name":N}` | `{"type":"function","function":{"name":N}}` |
| `tool_choice` other objects | `"auto"` (+ log) |
| `text.format` text / json_object / json_schema | `ResponseFormat` TEXT / JSON_OBJECT / JSON_SCHEMA with `json_schema={"name","schema","strict"}` (same nesting as `openai_in._translate_response_format`) |
| `reasoning.effort` | `"none"` → `think=False`; else `think=<effort string>` clamped (`minimal→low`, `xhigh→high`). **String-think is the proven cross-engine path**: vllm_out.py:172-174 converts string `think` → `reasoning_effort` for vLLM; ollama_out forwards the string for gpt-oss-on-Ollama. Do **not** set `reasoning_effort` + `think=True` together (no current dialect produces that combination; Ollama would receive a bool where it needs a string). |
| `max_output_tokens` | `max_tokens` |
| `temperature`, `top_p`, `user`/`safety_identifier` | passthrough (`safety_identifier`→`user` if `user` absent) |
| `stream` | `stream` |
| `parallel_tool_calls`, `store`, `previous_response_id`, `include`, `metadata`, `truncation`, `background`, `prompt_cache_key`, `service_tier`, `stream_options`, `client_metadata`, unknown fields | captured into ctx or ignored — **never an error** |

Tier-1 state semantics: `store` accepted + echoed, nothing persists; `previous_response_id` → 400 `{"code":"previous_response_not_found"}` (the legitimate error for an unknown id; Codex never sends the field over HTTP); `background:true` → 400; `truncation:"auto"` accepted + echoed verbatim, but performs no server-side truncation (context overflow behaves as `"disabled"` — document).

**Input item table** (`_translate_item`). Every item type **ignores `id` and `status` keys if present** (Codex-on-Azure and SDK replays send them):

| Item | Canonical |
|---|---|
| `message` role `user`/`system` — **also any item with `role`+`content` and NO `type` key** (EasyInputMessage; the SDK's most common form) | same role; `developer` → `system` |
| `message` role `assistant` (OutputMessage form) | assistant message; `output_text` parts concatenated; `refusal` appended as text |
| part `input_text` | `TextContent` |
| part `input_image` (`data:` URI) | `ImageBase64Content` (parse like `openai_in._translate_content_block`) |
| part `input_image` (`http(s)` URL) | `ImageUrlContent` |
| part `input_image` with `file:` URL scheme | **400** (never accept client-supplied file refs — security) |
| part `input_image` with `file_id`, or `input_file` | 400 "file inputs not supported" |
| `function_call` | assistant message with `tool_calls=[CanonicalToolCall(id=call_id, …)]` — **`call_id`, not `id`, is the correlation key**; consecutive calls merge into one assistant message |
| `function_call_output` (string output) | `tool` message, `tool_call_id=call_id` |
| `function_call_output` (part-array output) | text parts → tool message; `input_image` parts → **append a follow-up `user` message carrying the image** (canonical user messages support images; relevant to Codex's `view_image` tool) + log |
| `reasoning` | **dropped** (encrypted_content is opaque OpenAI data; backends can't replay CoT; matches LiteLLM/hermes shims). Debug log. |
| `item_reference` | 400 in Tier 1; resolved in Tier 2 |
| unknown `type` | skipped + debug log |

**`format_response(chat_response: dict, ctx) -> dict`** — non-streaming Response object from `choices[0].message`:
- `reasoning_content` → `{"id":"rs_<hex>","type":"reasoning","summary":[],"content":[{"type":"reasoning_text","text":…}],"status":"completed"}`
- `content` → `{"id":"msg_<hex>","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":…,"annotations":[],"logprobs":[]}]}` (`annotations`/`logprobs` are required arrays)
- `tool_calls[i]` → `{"id":"fc_<hex>","type":"function_call","status":"completed","call_id":<tc.id>,"name":…,"arguments":<JSON string, not parsed>}`
- `finish_reason "length"` → `status:"incomplete"` + `incomplete_details:{"reason":"max_output_tokens"}`; else completed
- `usage` → `{"input_tokens","input_tokens_details":{"cached_tokens":0},"output_tokens","output_tokens_details":{"reasoning_tokens":0},"total_tokens"}` (drop `is_estimated`)
- Envelope via shared `_build_snapshot(ctx, status, output, usage, error=None, incomplete_details=None)`: all spec-required snapshot fields (`id`, `object:"response"`, `created_at`, `error`, `incomplete_details`, `instructions`, `model` **raw client string**, `tools` as received, `output`, `parallel_tool_calls`, `metadata`, `tool_choice`, `temperature`, `top_p`) plus `completed_at`, `background:false`, `max_output_tokens`, `previous_response_id`, `prompt_cache_key`, `reasoning`, `service_tier:"default"`, `store`, `text`, `top_logprobs:0`, `truncation`, `usage`, `user:null`.

**`_map_finish_reason(reason)`**: `stop|tool_calls|None`→completed; `length`→incomplete/max_output_tokens; unknown→completed.

### 1.2 New file: `backend/app/core/translators/responses_stream.py`

`stream_responses_events(inner: AsyncIterator[bytes], ctx) -> AsyncIterator[str]` — stdlib-only SSE re-framer (unit-testable with plain imports; deliberate improvement over the untested in-route `_stream_anthropic_events`). `_event(seq, type, **fields)` emits `event: {type}\ndata: {compact json, "type" first}\n\n`.

1. **Immediately** (before consuming inner): `response.created` (seq 0) + `response.in_progress` (seq 1) with `_build_snapshot(ctx,"in_progress",[],None)`.
2. Parse inner like anthropic_api.py:176-197 (bytes/str tolerant, `data: ` lines only, skip `[DONE]`, skip bad JSON) — but **harvest `usage` from empty-`choices` chunks before skipping them** (that's where vLLM's `include_usage` rides).
3. **State machine** — one open item at a time; transitions close current then open next; `output_index` per item, `sequence_number` global:
   - `delta.reasoning_content` → reasoning item: `output_item.added` (`{"id":"rs_…","type":"reasoning","summary":[]}`) → `response.reasoning_text.delta` per delta (`{item_id, output_index, content_index:0, delta}`) → close: `reasoning_text.done` → `output_item.done` (+`content:[{"type":"reasoning_text","text":…}]`, `status:"completed"`).
   - `delta.content` → message item: `output_item.added` → `content_part.added` (part `{"type":"output_text","annotations":[],"logprobs":[],"text":""}`) → `output_text.delta` per delta (`logprobs:[]`, omit `obfuscation`) → close: `output_text.done` → `content_part.done` → `output_item.done`.
   - `delta.tool_calls[]` keyed by `tc.index` (buffer id/name/arg fragments per index, like the Anthropic adapter): `output_item.added` with best-effort `{"id":"fc_<hex>","type":"function_call","status":"in_progress","arguments":"","call_id":<tc.id or "call_<hex>">,"name":<name>}` (Codex reads name/call_id from `output_item.done`, so `.added` best-effort is safe) → `function_call_arguments.delta` (`{item_id, output_index, delta}`, no content_index) → close: `function_call_arguments.done` (+`name`,`arguments`) → `output_item.done`.
   - Arbitrary interleavings handled by close-current-open-next; text-after-tools opens a second message item.
4. **`finish_reason` chunk: record pending terminal state — do NOT emit the terminal event yet.** Close open items, stash `(status, incomplete_details)` (`length`→incomplete/max_output_tokens), keep consuming. vLLM's usage chunk arrives **after** finish_reason; emitting at finish would permanently discard it.
5. **Emit the terminal snapshot at inner exhaustion** (natural `StopAsyncIteration`): accumulated `output`, pending status (default completed), `completed_at`, `usage` = harvested real usage else `{input_tokens:0, output_tokens:ceil(chars/4), total_tokens:…}` estimate. This unifies the happy path with the fallback and preserves the **drain contract** by construction — the loop always runs the inner generator dry, so `_complete_streaming_request` always executes.
6. **Inner error frames** (`{"error":{message,type,code}}` from pre-first-chunk backend 4xx, inference.py:400-412): stash terminal `response.failed` with `error:{"code": <mapped>, "message"}` — map 429/quota→`insufficient_quota` or `rate_limit_exceeded`, context messages→`context_length_exceeded`, else `server_error` (codes Codex special-cases); emit at exhaustion; keep draining.
7. **Exception hardening (required — the Anthropic template lacks it):** wrap the consume loop in try/except.
   - `except HTTPException/Exception`: close open items, emit terminal `response.failed` (`server_error` or mapped code), return. This covers the in-generator `_check_quota` HTTPException (inference.py:329 sits **outside** the service's try at :351 and would otherwise propagate after 200 headers → EOF-without-terminal → Codex 5× re-POST storm) and mid-stream re-raises from inference.py:413-424.
   - `except (CancelledError, GeneratorExit)`: best-effort emit terminal `response.failed` (worker shutdown/client disconnect), then **re-raise** so inner cleanup still runs.
   - Exhaustion-cause discrimination: if the inner stream died after content with no finish_reason **due to an error path**, emit `response.failed` (`server_error`) — a truncated answer reported as `completed` would silently corrupt the agent transcript; bounded Codex retries (≤5, backed off) are the lesser evil. Reserve the `completed` fallback for benign EOF-after-content.

### 1.3 New file: `backend/app/api/responses_api.py`

`router = APIRouter(tags=["responses"])`; `error_json(status, message, err_type="invalid_request_error", code=None, param=None) -> JSONResponse({"error":{…}})` lives **here** (route layer — keeps the translator fastapi-free). `POST /v1/responses`:

1. `auth: Tuple[User, ApiKey] = Depends(authenticate_request)`, `db: AsyncSession = Depends(get_async_db)`, raw `Request`.
2. Feature flag: `if not get_settings().responses_api_enabled: return error_json(404, …)`. **Default `False`** until live validation completes (no staging exists — validation runs on prod behind this flag, flipped via env).
3. `body = await request.json()` → `error_json(400, "Invalid JSON body")`. (OpenAI envelope for route-level errors; dependency-raised 401s keep FastAPI's `{"detail":…}` — Codex/SDKs key 401 handling on status, not body. Optional polish: router-scoped exception handler to reshape those too.)
4. `ctx = ResponsesRequestContext.from_body(body)`; `bind_request_context(request_id=ctx.response_id, user_id=user.id)`.
5. Translate → stamp `request_id/user_id/api_key_id`; translator errors → `error_json(400, f"Invalid request: {e}")`.
6. `canonical.model, _ = registry.resolve_alias(canonical.model)`; not `await registry.model_exists(...)` → `error_json(404, f"The model '{raw}' does not exist or you do not have access to it.", code="model_not_found")`.
7. **Pre-flight quota exactly once**: route calls `await service._check_quota(user, api_key)` and passes **`skip_quota_check=True`** to the service methods (see §1.4 — `check_rpm` INCRs on every call; without the skip flag every request burns two RPM slots and Codex users get throttled at half their limit). 429s → `error_json(429, "… Please try again in {n}s.", err_type="requests", code="rate_limit_exceeded")` / `code="insufficient_quota"` — the message phrasing engages Codex's retry-delay parser.
8. Dispatch:
   - streaming → `StreamingResponse(stream_responses_events(service.stream_chat_completion(canonical, user, api_key, request, endpoint="/v1/responses", skip_quota_check=True), ctx), media_type="text/event-stream", headers={"Cache-Control":"no-cache","Connection":"keep-alive","X-Request-ID":ctx.response_id})`
   - non-streaming → `format_response(await service.chat_completion(..., endpoint="/v1/responses", skip_quota_check=True), ctx)`; catch service HTTPExceptions and reshape via `error_json`.
9. **Audit correlation**: include `ctx.response_id` in the request `parameters` JSON (via a small `extra_parameters` hook on `_create_request_record` or by stamping it on the canonical request) so a user-reported `resp_…` id is findable in the admin audit page; container logs rotate.
10. **Observability**: structured log per request at terminal: endpoint, terminal event type (completed/incomplete/failed), real-vs-estimated usage, and `prompt_cache_key` (Codex's thread id — makes duplicate re-POST storms trivially grepable).

### 1.4 Edits to existing files

- **`backend/app/services/inference.py`** — add `endpoint: str = "/v1/chat/completions"` **and** `skip_quota_check: bool = False` kwargs to `chat_completion()` (:261) and `stream_chat_completion()` (:317); thread endpoint into `_create_request_record` (:285, :332); guard the internal `_check_quota` calls (:281, :329). Defaults preserve all nine existing callers (verified all positional).
- **`backend/app/api/__init__.py`** — import + `include_router` (blocks :19-27 / :33-41).
- **`backend/app/settings.py`** — `responses_api_enabled: bool = False` (flip to True after validation, or via env).
- **Docs (required for ship, ~0.5–1 day)** — `backend/app/dashboard/templates/public/documentation.html` (new "OpenAI Responses API" section with endpoint row, error envelope, streaming events, and a Codex `config.toml` example — the Anthropic section at :613-668 is the template; also gateway bullet :112 and endpoint table :303); `README.md` (:13 feature bullet, :355-364 endpoint table); `docs/index.md` (endpoint table ~:237; request-ID format list :184 gains `resp_*`); `docs/architecture.md` (:21-30 endpoint list).
- **Release mechanics** — bump `[project].version` in `pyproject.toml` **and** `sidecar/VERSION` together (test_version_alignment.py enforces the pair). Note: git tags have drifted ahead of pyproject (latest tag v2.5.14 vs pyproject 2.4.4) — ship Tier 1 as **v2.6.0** (new API surface), realigning pyproject in the same commit; release notes in the annotated tag per house convention.

**Separately-committed pre-work (sequenced BEFORE Tier 1 ships):**
- **`vllm_out.py` `stream_options: {"include_usage": true}`** on streaming requests, so real usage reaches the terminal snapshot (Codex uses it for context tracking/auto-compaction). This flips behavior for every vLLM streaming consumer (/v1/chat/completions, Ollama dialect, Anthropic dialect, dashboard chat) — own PR with unit tests for usage-only empty-choices chunks across all four consumers, full `python test.py` across dialects, and a `stress.py` run.
- **Dockerfile `--graceful-timeout`** raise from 60s → ≥300s (matching `backend_request_timeout_per_attempt=180` headroom). With `--max-requests 2000` recycling, 60s graceful means routine mid-stream SIGKILLs on long Codex turns → EOF-without-terminal → retry storms. Cheap, orthogonal commit.

### 1.5 Tier 1 tests

- **`backend/app/tests/unit/test_responses_in.py`** (plain imports): basics (string input, param map, unknown-field tolerance incl. `client_metadata`/`prompt_cache_key`/`include`); **typeless `{"role","content"}` items** (both string and part-array content); items **with** `id`/`status` keys (Azure/SDK replay) accepted and ignored; every item-table row incl. `developer`→system, function_call/function_call_output round-trip via `call_id`, consecutive-call merge, reasoning drop, image-in-tool-output → follow-up user message, `file:` URL rejection, unknown-item skip; tools (flat→nested, `web_search` stripped not raised, tool_choice variants); text.format ×3; reasoning-effort matrix (`none`→think False; `xhigh`→think "high"; no reasoning_effort+think combo); `format_response` (message/reasoning/function_call/mixed, `length`→incomplete, usage mapping, snapshot required fields + spec defaults for omitted params); round-trip through `VLLMOutTranslator.translate_chat_request`.
- **`backend/app/tests/unit/test_responses_stream.py`**: `_async_iter`/`_collect_stream` helpers (test_streaming.py:21-32); scripted chat-SSE inputs (reuse `vllm_sse_chunks` fixture + hand-built tool/reasoning scripts); assert exact ordered event-type lists for the three canonical sequences; per-frame invariants (event line == json `type`, seq monotonic from 0, `content_part.added` before first delta, stable item_ids, no `[DONE]`); **terminal-after-exhaustion timing** (usage-only chunk after finish_reason lands in `response.completed.usage`); `length`→`response.incomplete`; inner-error-frame→`response.failed` with mapped code; **raised-exception→terminal `response.failed`** (fake inner that raises HTTPException / generic Exception); benign-EOF fallback→completed vs error-EOF→failed; drain contract (sentinel flag on fake generator proves full consumption).
- **`backend/app/tests/unit/test_responses_api.py`** (Pattern C bootstrap per test_voice_api.py:30-67; stub `backend.app.db*`, `api.auth`, `services.inference`, `core.telemetry.registry`, `logging_config`, `settings`): happy paths, 400 invalid JSON, 404 model envelope, 400 previous_response_id, 429 pre-flight with `skip_quota_check=True` asserted on the service call, error envelope shape, feature-flag 404.
- **`test.py`** — `test_responses(client, cfg)` section (SECTIONS dict :904-914): non-streaming shape check, streaming event collection (≥1 `output_text.delta`, terminal `response.completed`, **no `data: [DONE]`**), function-call round trip, 401 (assert **status only** — body is FastAPI-default).
- **TESTING.md**: §1 table +3 rows (Tier 1), §4 Sections line + coverage row gain `responses`.

### 1.6 Tier 1 acceptance

1. `make test-unit` green; TESTING.md counts updated.
2. `python test.py --api-key … --section responses` green (local, then prod-with-flag).
3. **Live Codex desktop test** (prod, flag enabled via env, admin key):
   ```toml
   [model_providers.mindrouter]
   name = "MindRouter"
   base_url = "https://mindrouter.uidaho.edu/v1"
   env_key = "MINDROUTER_API_KEY"
   wire_api = "responses"
   requires_openai_auth = false
   [profiles.mindrouter]
   model_provider = "mindrouter"
   model = "qwen/qwen3.5-122b"
   ```
   Exercise multi-turn chat, a real file edit via `shell_command`, reasoning rendering (qwen thinking → `reasoning_text.delta`), tool-output images via `view_image`. Grep logs by `prompt_cache_key` for duplicate re-POSTs (retry storms); confirm zero after the exception-hardening. Measure TTFT growth across turns (prefix-cache miss cost — see risks). Try `web_search = "disabled"` vs default to confirm the strip path.
4. Audit rows show `endpoint="/v1/responses"` with `resp_…` in parameters; capacity gauges return to baseline after streams.

**Tier 1 estimate:** translator ~450 lines, stream adapter ~400, route ~180, service diff ~20, tests ~1000, docs ~0.5–1 day, include_usage + gunicorn pre-work ~0.5–1 day. **6–8 working days** including live Codex debugging and release mechanics.

---

## Tier 2 — Server-side state (`store` / `previous_response_id` / retrieval)

For OpenAI-SDK clients; not needed by Codex-over-HTTP.

### 2.1 DB: `StoredResponse` + migration

Per house conventions (models.py; **revision id assigned at implementation time from `alembic heads`** — 061 is head today but interim migrations are likely; verify single head before prod upgrade):

- `StoredResponseStatus(str, PyEnum)`: in_progress/completed/failed/incomplete/cancelled.
- `StoredResponse(Base, TimestampMixin)`, table `stored_responses`: BigInteger PK; `response_id String(80) unique index` default `resp_<uuid4().hex>`; `user_id`/`api_key_id` FKs NOT NULL; optional `request_id → requests.id` audit link; `model String(200)`; `status` Enum via `values_callable=_enum_values`; `previous_response_id String(80) index` (**not a self-FK** — retention deletes oldest-first; app-level 404 matches OpenAI semantics); JSON `input_items` (delta only — see §2.2), `output_items`, `parameters`, `usage`, `error`, **`offloaded_images`** (server-side map of item-path → artifact file, see security note); `instructions MEDIUMTEXT`; single `__table_args__` (avoid the models.py:515/:581 double-assignment bug): `ix_stored_responses_user_created(user_id, created_at)`, standalone `ix_stored_responses_created_at`, `ix_stored_responses_prev`.
- Migration file per the `YYYYMMDD_HHMMSS_<rev>_<slug>.py` scheme, shape of migration 061; docstring notes MariaDB non-transactional DDL. **Rehearse on a local compose stack restored from a prod dump before running in the prod container.**
- Add model to crud import block (crud.py:27-64).

### 2.2 CRUD + store service

- **crud.py**: `create_stored_response` (add+flush); `get_stored_response(db, response_id, user_id)` (owner-scoped); `delete_stored_response`; `get_stored_response_chain(db, response_id, user_id, max_depth)` (iterative walk, cycle/depth errors).
- **`backend/app/services/responses_store.py`**:
  - **Semantics:** absent `store` → **true** (OpenAI parity). Each row stores only the request's **own delta** input items + its output items (chain rebuild = concat root→leaf; storing rebuilt input would duplicate ancestors per generation — O(N²)).
  - **Caps (required):** max stored payload bytes per response (reject with 400 or persist-truncated with a flag; default ~5 MB); per-user max stored rows with oldest-first eviction at insert; surface row/size counts on the admin retention page.
  - `persist_response(...)`: own `get_async_db_context()` session; streaming persists **from a `finally` block** under `asyncio.shield` (dashboard/chat.py:1112-1126 pattern) so abnormal ends (disconnect, failure) still write a row with status failed/incomplete + accumulated output — a billed request must leave a trace and not silently break a planned chain. Document the tiny commit-vs-retrieve race after the terminal event.
  - **Image offload — security-critical design:** >1 KB `data:` URIs in stored input are offloaded to `{artifact_storage_path}/responses_store/<response_id>/<n>.<ext>`, but the JSON is **never** rewritten to `file://` strings a client could forge. The mapping lives out-of-band in the `offloaded_images` column (item-path → filename); re-inflation reads **only** from that column and validates `realpath(file)` is under this row's own directory. Client-supplied `file:` URLs are already rejected at translation (§1.1). Wire output (GET response / input_items / chain replay) either re-inflates to `data:` URIs or omits the image content — **never emits `file://`**. Files are deleted row-driven (with the row), not mtime-driven.
  - `rebuild_input_from_chain(chain, new_input_items)`: concat each response's delta input + output items root→leaf + new items → existing `_translate_item` path. Per spec, `instructions`/`tools`/params are **not** inherited — current request's values apply; reasoning items drop at translation.
- **Route changes**: `previous_response_id` → owner-scoped chain load (miss/depth/cycle → 400 `previous_response_not_found`) → rebuild before translation; `store` (defaulted true) → persist; `item_reference` resolves against the store.

### 2.3 Companion endpoints (all owner-scoped, OpenAI envelopes)

- `GET /v1/responses/{response_id}` → rebuild Response via `_build_snapshot`; 404 `{"code":"response_not_found"}`; `stream=true` replay → 400 (unsupported v1).
- `DELETE /v1/responses/{response_id}` → delete row + files → `{"id":…,"object":"response","deleted":true}`.
- `GET /v1/responses/{response_id}/input_items` → `{"object":"list","data":[…],"first_id","last_id","has_more"}`; `limit` 1–100 default 20, `order` default `desc`, `after` cursor; items get stable ids stamped at persist time.
- `POST /v1/responses/{response_id}/cancel` → 400 (background unsupported).

### 2.4 Retention

- `retention.py` `_DEFAULTS` (:38-48): `"retention.responses_store_days": 30` (0 = keep forever).
- `cleanup_expired_stored_responses(app_db, cutoff, batch_size)` — batched `DELETE … LIMIT :batch` loop (:705-733 pattern) + per-batch offloaded-file removal before row delete; hook into `run_retention_cycle()` (~:1253) behind `if days > 0`.
- **Dashboard needs BOTH edits**: the allow-list (`dashboard/routes.py:4669-4679`) **and** a form input in `templates/admin/retention.html` (the form hardcodes each key; allow-list alone yields an uneditable, invisible knob) + the GET handler's template config dict.
- `settings.py`: `responses_store_max_chain_depth: int = 20`.

### 2.5 Tier 2 tests

- **`test_responses_store.py`** (Pattern C): chain rebuild ordering + params-not-inherited; depth/cycle errors; owner-scope (source-text assertion for the crud SQL per test_model_enrichment.py style); offload map round-trip incl. **realpath containment rejection** (attempted `../` / foreign-row path); caps (payload reject, per-user eviction); pagination math.
- **`test_responses_api.py` additions**: GET 200/404, DELETE envelope, input_items envelope (+ no `file://` in wire output), cancel 400, store-default-true persists, previous_response_id happy + not-found.
- **`test.py` smoke**: create(store default) → retrieve → chained follow-up answers with context → delete → 404.
- TESTING.md: +1 §1 row; migration rehearsal noted in the deploy checklist.

### 2.6 Tier 2 acceptance

OpenAI **python SDK** against MindRouter: `responses.create` (default store) → `retrieve` → two-hop `previous_response_id` chain with a function call mid-chain → `delete` → `input_items.list` — all green. Retention sweep on a seeded DB deletes rows + files. Admin retention page shows and edits the new knob.

**Tier 2 estimate:** model/migration/crud ~280 lines, store service ~350 (caps + offload security), endpoints ~220, retention ~100, tests ~700, docs ~0.5 day. **5–6 working days.**

---

## Sequencing

1. Pre-work PRs: vllm_out `include_usage` (with cross-dialect regression matrix + stress run) and Dockerfile `--graceful-timeout` raise.
2. Tier 1 translator + stream adapter with unit tests (pure, no infra risk) → route + service kwargs → docs → smoke.
3. Deploy to prod with `responses_api_enabled=False`; flip via env for the live Codex validation window; then default-on and **ship v2.6.0** (pyproject + sidecar/VERSION realigned, annotated-tag release notes).
4. Tier 2 DB/store/endpoints/retention (+ dashboard form) → SDK validation → migration rehearsal → **ship v2.6.1**.

## Risks / open questions

- **Prefix-cache affinity (biggest UX lever):** Codex resends a growing full transcript every step, and MindRouter routes each request independently across (e.g.) 4 qwen3.5-122b replicas — near-guaranteed vLLM prefix-cache misses, full re-prefill per step, TTFT ballooning toward the 180s per-attempt timeout on long tasks. Codex hands us the affinity key for free (`prompt_cache_key` = thread id). **Recommended fast-follow:** soft session affinity in the scheduler (consistent-hash `prompt_cache_key` to a healthy backend, fall back to normal scoring when saturated). Measure TTFT-per-turn in the live test either way; this likely decides whether Codex-on-MindRouter feels usable on long tasks.
- **Model tool-following quality:** Codex drives `shell_command` hard; qwen3.5-122b/gpt-oss-120b fidelity under Codex's prompts is the real unknown. Validate early; MindRouter can't fix malformed shell calls.
- **`apply_patch`/`custom` tools:** stripped in Tier 1 (not sent by default for unknown models). If a user forces `apply_patch_tool_type`, edits degrade to shell heredocs. A `custom`-tool→function shim is possible later.
- **Reasoning replay:** input reasoning items are dropped (OpenAI replays via encrypted_content). Correctness-safe; slight quality cost on long tool chains.
- **`web_search` stripped silently:** the model never sees the tool so it can't call it; document that users should set `web_search = "disabled"` in config.toml.
- **Usage accuracy:** without the include_usage pre-work, Codex sees estimates (input 0) and its auto-compaction timing skews — which is why that PR is sequenced first, not "whenever".
