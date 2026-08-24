############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# inference.py: Core inference service for request routing and proxying
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Inference service - handles request routing and backend proxying."""

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, Optional, Set, Tuple

import httpx
import orjson
import tiktoken
from fastapi import HTTPException, Request, status
from opentelemetry import trace
from sqlalchemy.ext.asyncio import AsyncSession

_tracer = trace.get_tracer(__name__)

from backend.app.core.redis_client import incr_inflight_tokens, decr_inflight_tokens
from backend.app.core.canonical_schemas import (
    CanonicalChatRequest,
    CanonicalChatResponse,
    CanonicalChoice,
    CanonicalEmbeddingRequest,
    CanonicalEmbeddingResponse,
    CanonicalImageRequest,
    CanonicalMessage,
    CanonicalRerankRequest,
    CanonicalScoreRequest,
    MessageRole,
    ResponseFormatType,
    UsageInfo,
)
from backend.app.core.scheduler.policy import get_scheduler
from backend.app.core.scheduler.queue import Job, JobModality
from backend.app.core.stream_coalesce import StreamCoalescer
from backend.app.core.telemetry.registry import get_registry
from backend.app.core.translators import DiffusionOutTranslator, OllamaOutTranslator, VLLMOutTranslator
from backend.app.core.translators.vllm_out import (
    reasoning_promotion_applies as _reasoning_promotion_applies,
)
from backend.app.db import crud
from backend.app.db.models import ApiKey, Backend, BackendEngine, Modality, User
from backend.app.logging_config import get_logger
from backend.app.settings import get_settings

logger = get_logger(__name__)


# Module-level tiktoken encoder (lazy singleton)
_tiktoken_encoder = None


def _ensure_encoder():
    """Lazily initialize (and return) the module-level tiktoken encoder."""
    global _tiktoken_encoder
    if _tiktoken_encoder is None:
        try:
            _tiktoken_encoder = tiktoken.get_encoding(get_settings().default_tokenizer)
        except Exception:
            _tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
    return _tiktoken_encoder


def count_request_text_chars(request: "CanonicalChatRequest") -> int:
    """Sum the input text characters in a chat request.

    Used as a cheap admission-control size gate before the (potentially
    expensive) tokenizer round-trip on the /tokenize endpoints.
    """
    total = 0
    for msg in request.messages or []:
        content = msg.content
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for block in content:
                text = getattr(block, "text", None)
                if text:
                    total += len(text)
    if request.tools:
        total += len(json.dumps([t.model_dump() for t in request.tools]))
    return total


def _tiktoken_estimate_sync(request: "CanonicalChatRequest", encoder, max_chars: int) -> int:
    """CPU-bound tiktoken estimate (runs off the event loop via to_thread).

    Caps the total number of characters fed to the encoder at ``max_chars``
    so a pathologically large prompt cannot pin a worker in a synchronous
    encode. The cap only bounds the *counting* cost — the actual prompt sent
    to the model is untouched. Per-message overhead is always accounted for.
    """
    total = 0
    budget = max_chars

    def _encode_capped(text) -> int:
        nonlocal budget
        if not text or budget <= 0:
            return 0
        chunk = text[:budget]
        budget -= len(chunk)
        return len(encoder.encode(chunk))

    for msg in request.messages or []:
        content = msg.content
        if isinstance(content, str):
            total += _encode_capped(content)
        elif isinstance(content, list):
            for block in content:
                total += _encode_capped(getattr(block, "text", None))
        # Per-message overhead (role, separators)
        total += 4

    # Tool definitions contribute tokens too
    if request.tools:
        tool_text = json.dumps([t.model_dump() for t in request.tools])
        total += _encode_capped(tool_text)

    return total


async def _tiktoken_estimate(request: "CanonicalChatRequest") -> int:
    """Estimate input tokens using tiktoken (fallback for Ollama / vLLM errors).

    The synchronous encode is offloaded to a thread so it never blocks the
    event loop, and the text fed to it is capped at
    ``settings.tokenize_max_input_chars``.
    """
    encoder = _ensure_encoder()
    # getattr fallback keeps this robust if the shared setting has not yet been
    # added (concurrent settings work); default matches the contract. Coerce to
    # a positive int so a missing/misconfigured (non-numeric) value falls back
    # to the safe default instead of crashing this hot path.
    max_chars = getattr(get_settings(), "tokenize_max_input_chars", 2_000_000)
    if not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars <= 0:
        max_chars = 2_000_000
    return await asyncio.to_thread(_tiktoken_estimate_sync, request, encoder, max_chars)


# Hard cap on max_tokens to prevent any single response from being excessively large
_MAX_OUTPUT_TOKENS = 65536

# Buffer reserved for internal overhead (template tokens, special tokens, etc.)
_TOKEN_BUFFER = 1024


async def _estimate_input_tokens(request: "CanonicalChatRequest") -> int:
    """Tiktoken estimate, memoized on the request so retries don't re-encode."""
    est = getattr(request, "_est_input_tokens", None)
    if est is None:
        est = await _tiktoken_estimate(request)
        request._est_input_tokens = est
    return est


def _invalidate_token_memos(request: "CanonicalChatRequest") -> None:
    """Drop memoized token counts (after messages change or a recount is forced)."""
    request._est_input_tokens = None
    request._exact_input_tokens = None


def _is_context_length_error(detail: Any) -> bool:
    """Detect a vLLM 400 caused by context-window overflow."""
    text = detail if isinstance(detail, str) else json.dumps(detail, default=str)
    text = text.lower()
    return "context length" in text or "max_model_len" in text


# ollama.enforce_num_ctx is read on every Ollama attempt — cache the DB config
# value for a short TTL instead of opening a fresh session on the hot path.
_ENFORCE_NUM_CTX_TTL_S = 30.0
_enforce_num_ctx_cache: Optional[Tuple[bool, float]] = None


async def _get_enforce_num_ctx() -> bool:
    """Read the ollama.enforce_num_ctx config with a 30s TTL cache."""
    global _enforce_num_ctx_cache
    cached = _enforce_num_ctx_cache
    now = time.monotonic()
    if cached is not None and now - cached[1] < _ENFORCE_NUM_CTX_TTL_S:
        return cached[0]
    from backend.app.db.session import get_async_db_context
    async with get_async_db_context() as cfg_db:
        value = bool(await crud.get_config_json(cfg_db, "ollama.enforce_num_ctx", True))
    _enforce_num_ctx_cache = (value, now)
    return value


class InferenceService:
    """
    Handles inference request processing.

    Responsibilities:
    - Create and submit jobs to scheduler
    - Route requests to backends
    - Proxy requests and stream responses
    - Record audit logs
    - Track token usage
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self._settings = get_settings()
        self._scheduler = get_scheduler()
        self._registry = get_registry()
        self._latency_tracker = self._registry.latency_tracker
        self._http_client: Optional[httpx.AsyncClient] = None
        # Prompt redactions decided by the inline DLP gate, applied to the
        # outbound request by _create_request_record after the ORIGINAL is
        # stored (so the audit trail + async alert path see the real text).
        self._pending_prompt_redactions: list = []

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with per-attempt timeout."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=float(self._settings.backend_request_timeout_per_attempt),
                    write=10.0,
                    pool=10.0,
                ),
                limits=httpx.Limits(
                    max_connections=200,
                    max_keepalive_connections=40,
                    keepalive_expiry=30,
                ),
            )
        return self._http_client

    def _make_inference_client(self) -> httpx.AsyncClient:
        """Create a dedicated HTTP client for a single inference request.

        Unlike the shared pool client, this client is meant to be closed
        when the request completes or times out.  Closing it tears down
        the underlying TCP connection, which signals vLLM to abort any
        in-progress generation — preventing orphaned requests from
        occupying backend slots after MindRouter gives up.
        """
        return httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=10.0,
                read=float(self._settings.backend_request_timeout_per_attempt),
                write=10.0,
                pool=10.0,
            ),
            limits=httpx.Limits(
                max_connections=1,
                max_keepalive_connections=0,
            ),
        )

    async def _close_inference_client(
        self,
        client: httpx.AsyncClient,
        backend: Backend,
        request_id: str,
    ) -> None:
        """Close a per-request inference client, aborting any in-flight work.

        Tearing down the TCP connection signals vLLM to abort generation,
        freeing the execution slot.
        """
        try:
            await client.aclose()
            logger.info(
                "backend_request_aborted",
                backend_id=backend.id,
                request_id=request_id,
            )
        except Exception:
            pass

    async def _count_input_tokens(
        self,
        request: CanonicalChatRequest,
        backend: Backend,
    ) -> Tuple[int, bool]:
        """Count input tokens for a request.

        For vLLM backends, calls the /tokenize endpoint which applies the full
        chat template (including tool-calling overhead). Falls back to tiktoken
        estimation on failure. For Ollama backends, uses tiktoken directly.

        Counts are memoized on the request object so retry attempts don't
        re-tokenize identical messages.

        Returns:
            (token_count, is_estimate) — is_estimate is False only when vLLM
            /tokenize returned an authoritative count.
        """
        if backend.engine == BackendEngine.VLLM:
            cached = getattr(request, "_exact_input_tokens", None)
            if cached is not None:
                return cached, False
            try:
                payload = VLLMOutTranslator.translate_chat_request(request)
                payload.pop("stream", None)
                payload.pop("max_tokens", None)
                client = await self._get_http_client()
                resp = await asyncio.wait_for(
                    client.post(f"{backend.url}/tokenize", json=payload),
                    timeout=10.0,
                )
                resp.raise_for_status()
                count = resp.json().get("count", 0)
                request._exact_input_tokens = count
                return count, False
            except Exception as e:
                logger.warning(
                    "tokenize_fallback",
                    backend_id=backend.id,
                    error=str(e),
                )
                # Fall through to tiktoken estimation

        return await _estimate_input_tokens(request), True

    async def cap_max_tokens(
        self,
        request: CanonicalChatRequest,
        backend: Backend,
        context_length: int,
        force_exact: bool = False,
    ) -> None:
        """Cap max_tokens to fit within the model's context window.

        The backend /tokenize round-trip is skipped only when the
        conservative tiktoken bound (estimate * 1.3 + per-message overhead
        + buffer) proves the shortcut is exactly equivalent to exact
        counting: an explicit max_tokens that provably fits, or an unset
        max_tokens whose room reaches the hard cap either way.  Otherwise —
        near the boundary, unset max_tokens without full-hard-cap room, or
        when exactness is required (auto_truncate, or force_exact after a
        context-length 400) — uses exact counting for vLLM (via /tokenize)
        or tiktoken estimation for Ollama, then applies:
            remaining = context_length - input_tokens - buffer
            max_tokens = min(requested, remaining, hard_cap)
        """
        if force_exact:
            # A context-length 400 slipped past the gate — recount from scratch.
            _invalidate_token_memos(request)
        elif not request.auto_truncate:
            est = await _estimate_input_tokens(request)
            bound = int(est * 1.3) + 16 * len(request.messages or []) + _TOKEN_BUFFER
            room = context_length - bound
            if request.max_tokens is not None:
                # Explicit budget: if even the conservative bound leaves room
                # for it, exact counting would return min(requested,
                # remaining) = requested — skipping /tokenize is provably
                # harmless.
                if room >= 2048 and room >= request.max_tokens:
                    request.max_tokens = min(request.max_tokens, _MAX_OUTPUT_TOKENS)
                    return
            elif room >= _MAX_OUTPUT_TOKENS:
                # Unset budget: the exact path would also cap at the hard
                # limit, so the shortcut yields an identical result.
                request.max_tokens = _MAX_OUTPUT_TOKENS
                return
            # Near the boundary, or unset max_tokens without full-hard-cap
            # room: fall through to exact counting so the default completion
            # budget stays context_length - exact_input - _TOKEN_BUFFER.

        input_tokens, is_estimate = await self._count_input_tokens(request, backend)

        remaining = context_length - input_tokens - _TOKEN_BUFFER
        if remaining <= 0 and request.auto_truncate:
            # Responses API truncation:"auto" — drop oldest turns until
            # the input fits, reserving room for a useful response.
            from backend.app.core.context_trim import trim_messages_to_fit

            _ensure_encoder()  # ensure the module-level encoder is ready
            reserve = min(request.max_tokens or 1024, 4096)
            budget = context_length - _TOKEN_BUFFER - reserve
            request.messages, dropped = trim_messages_to_fit(
                request.messages, budget,
                lambda s: len(_tiktoken_encoder.encode(s)),
            )
            if dropped:
                _invalidate_token_memos(request)
                input_tokens, is_estimate = await self._count_input_tokens(
                    request, backend
                )
                remaining = context_length - input_tokens - _TOKEN_BUFFER
                logger.info(
                    "auto_truncate_applied",
                    dropped_messages=dropped,
                    input_tokens=input_tokens,
                    context_length=context_length,
                    model=request.model,
                )
        if remaining <= 0:
            logger.warning(
                "context_overflow",
                input_tokens=input_tokens,
                context_length=context_length,
                is_estimate=is_estimate,
                model=request.model,
            )
            # Allow a minimal output — the backend may still reject it,
            # but at least we give it a chance rather than failing here.
            remaining = 256

        requested = request.max_tokens if request.max_tokens is not None else _MAX_OUTPUT_TOKENS
        request.max_tokens = min(requested, remaining, _MAX_OUTPUT_TOKENS)

        logger.debug(
            "max_tokens_capped",
            input_tokens=input_tokens,
            is_estimate=is_estimate,
            context_length=context_length,
            requested=requested,
            capped=request.max_tokens,
        )

    @staticmethod
    def _dlp_scan_text(request: Any) -> Optional[str]:
        """Assemble the scannable prompt text from a canonical request.

        Uses the LIVE request (not stored audit content), so inline blocking
        works even when prompt capture is disabled.
        """
        from backend.app.services.dlp_scanner import extract_scannable_text

        messages = None
        raw = getattr(request, "messages", None)
        if raw:
            try:
                messages = [m.model_dump() if hasattr(m, "model_dump") else m for m in raw]
            except Exception:
                messages = None
        prompt = getattr(request, "prompt", None)
        if prompt is not None and not isinstance(prompt, str):
            prompt = str(prompt)
        return extract_scannable_text(messages=messages, prompt=prompt)

    async def enforce_prompt_dlp(
        self,
        request: Any,
        user: User,
        api_key: ApiKey,
        http_request: Request,
        endpoint: str,
        extra_parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Inline DLP prompt gate — block and/or redact, per policy.

        Called by the API handlers BEFORE the stream/non-stream branch so a
        block returns a clean 4xx rather than a broken SSE stream. Returns
        immediately (no scan, no text extraction) unless an inline prompt action
        is active, so alert-only deployments keep zero added latency.

        Block: the request is recorded (marked failed) and enqueued for the
        async worker (audit + email alert if that severity's Alert is on), then
        DlpBlockedError is raised (→ 422). Redact: the offending values are
        stashed and applied to the OUTBOUND request by _create_request_record,
        after the original text is stored — so the audit trail and the async
        alert path still see the real content.
        """
        from backend.app.services.dlp_enforcement import (
            prompt_inline_active,
            evaluate_prompt_inline,
        )

        self._pending_prompt_redactions = []
        if not await prompt_inline_active(self.db):
            return

        text = self._dlp_scan_text(request)
        action = await evaluate_prompt_inline(self.db, text)
        if action is None:
            return

        if action.redactions:
            # Applied downstream, after the original is stored for audit.
            self._pending_prompt_redactions = action.redactions
            return

        blocked = action.block
        # Record the blocked attempt for audit + async alert/email.
        db_request = await self._create_request_record(
            request, user, api_key, http_request, endpoint,
            extra_parameters=extra_parameters,
        )
        try:
            await crud.update_request_failed(
                self.db, db_request.id,
                error_message=(
                    f"Blocked by DLP policy ({blocked.severity}): "
                    f"{', '.join(blocked.categories)}"
                ),
            )
            await self.db.commit()
            from backend.app.services.dlp_worker import enqueue_for_dlp
            await enqueue_for_dlp(db_request.id)
        except Exception:
            logger.warning("dlp_block_audit_failed", request_id=getattr(db_request, "id", None))
        raise blocked

    @staticmethod
    def _apply_prompt_redactions(request: Any, redactions: list) -> None:
        """Redact offending values in the OUTBOUND canonical request in place.

        Walks message contents (str or multimodal text parts) and the raw
        prompt. The audit record is built from the original before this runs, so
        only what is dispatched to the backend is altered.
        """
        from backend.app.services.dlp_enforcement import redact_text

        for msg in getattr(request, "messages", None) or []:
            content = getattr(msg, "content", None)
            if isinstance(content, str):
                msg.content = redact_text(content, redactions)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        part["text"] = redact_text(part["text"], redactions)
                    elif getattr(part, "text", None) and isinstance(part.text, str):
                        part.text = redact_text(part.text, redactions)
        if isinstance(getattr(request, "prompt", None), str):
            request.prompt = redact_text(request.prompt, redactions)

    @staticmethod
    def _apply_response_redactions(response: Any, redactions: list) -> None:
        """Redact offending values in the OUTBOUND response dict in place.

        Mirrors _response_scan_text's shape coverage. Applied AFTER the original
        response is persisted, so only what the caller receives is altered.
        """
        from backend.app.services.dlp_enforcement import redact_text

        if not isinstance(response, dict):
            return
        for ch in response.get("choices", []) or []:
            if not isinstance(ch, dict):
                continue
            msg = ch.get("message")
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                msg["content"] = redact_text(msg["content"], redactions)
            if isinstance(ch.get("text"), str):
                ch["text"] = redact_text(ch["text"], redactions)
        msg = response.get("message")
        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            msg["content"] = redact_text(msg["content"], redactions)
        if isinstance(response.get("response"), str):
            response["response"] = redact_text(response["response"], redactions)

    @staticmethod
    def _response_scan_text(response: Any) -> Optional[str]:
        """Assemble scannable text from a completion response (any dialect).

        Covers OpenAI chat (choices[].message.content), OpenAI/text completion
        (choices[].text), Ollama chat (message.content) and Ollama generate
        (response). Returns None when there's nothing textual to scan.
        """
        if not isinstance(response, dict):
            return None
        parts: List[str] = []
        for ch in response.get("choices", []) or []:
            if not isinstance(ch, dict):
                continue
            msg = ch.get("message") or {}
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                parts.append(msg["content"])
            if isinstance(ch.get("text"), str):
                parts.append(ch["text"])
        msg = response.get("message") or {}
        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            parts.append(msg["content"])
        if isinstance(response.get("response"), str):
            parts.append(response["response"])
        combined = "\n".join(p for p in parts if p)
        return combined or None

    async def _response_inline_plan(self, response: Any):
        """Return the InlineAction for a response (block or redactions), else None.

        Fast no-op unless an inline response action is active (cached gate).
        """
        from backend.app.services.dlp_enforcement import (
            response_inline_active,
            evaluate_response_inline,
        )

        if not await response_inline_active(self.db):
            return None
        return await evaluate_response_inline(self.db, self._response_scan_text(response))

    async def chat_completion(
        self,
        request: CanonicalChatRequest,
        user: User,
        api_key: ApiKey,
        http_request: Request,
        endpoint: str = "/v1/chat/completions",
        skip_quota_check: bool = False,
        extra_parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Handle non-streaming chat completion.

        Args:
            request: Canonical chat request
            user: Authenticated user
            api_key: API key used
            http_request: Original HTTP request
            endpoint: Endpoint string recorded on the audit row
            skip_quota_check: Caller already ran _check_quota (the RPM
                check increments a Redis counter, so it must run
                exactly once per request)

        Returns:
            OpenAI-compatible chat completion response
        """
        # Check quota
        if not skip_quota_check:
            await self._check_quota(user, api_key)

        # Create audit record
        db_request = await self._create_request_record(
            request, user, api_key, http_request, endpoint,
            extra_parameters=extra_parameters,
        )

        # Propagate request UUID so translators can use it as chunk/response ID
        request.request_id = db_request.request_uuid

        # Create job
        job = self._scheduler.create_job_from_chat_request(
            request, user.id, api_key.id
        )
        job.request_id = db_request.request_uuid

        completion_started = False
        try:
            # Route and proxy with retry
            response, backend = await self._proxy_with_retry(
                request, job, user, proxy_fn="_proxy_chat_request"
            )

            # Update records. Once the shielded completion starts,
            # _fail_request must not run: cancellation here leaves the
            # shielded write finishing in the background, and a concurrent
            # _fail_request would use the same session mid-transaction
            # (and double-release the scheduler slot).
            completion_started = True
            await self._complete_request(
                db_request, backend.id, response, job
            )

            return response

        except BaseException as e:
            if not completion_started:
                try:
                    await self._fail_request(db_request, None, str(e), job)
                except Exception:
                    pass
            raise

    async def stream_chat_completion(
        self,
        request: CanonicalChatRequest,
        user: User,
        api_key: ApiKey,
        http_request: Request,
        endpoint: str = "/v1/chat/completions",
        skip_quota_check: bool = False,
        extra_parameters: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[bytes]:
        """
        Handle streaming chat completion.

        Yields SSE-formatted chunks.

        ``skip_quota_check`` lets dialect routes pre-flight the quota
        before HTTP 200 headers are committed — this generator only
        runs on first iteration, after the status line is already sent.
        """
        if not skip_quota_check:
            await self._check_quota(user, api_key)

        db_request = await self._create_request_record(
            request, user, api_key, http_request, endpoint,
            extra_parameters=extra_parameters,
        )

        # Propagate request UUID so translators can use it as chunk ID
        request.request_id = db_request.request_uuid

        job = self._scheduler.create_job_from_chat_request(
            request, user.id, api_key.id
        )
        job.request_id = db_request.request_uuid

        full_content = ""
        chunk_count = 0
        routed_backend = None
        last_finish_reason = None
        inflight_chars = 0
        inflight_total_tokens = 0
        real_usage = None  # populated from the backend's usage chunk, if present
        completed = False
        completion_started = False
        coalescer = StreamCoalescer(
            self._settings.stream_coalesce_events,
            self._settings.stream_coalesce_ms,
        )

        try:
            async for chunk, backend in self._proxy_stream_with_retry(
                request, job, user, proxy_fn="_proxy_stream_request"
            ):
                routed_backend = backend

                # The vLLM translator yields wire-shaped dicts (hot path);
                # the Ollama translator still yields Pydantic chunks.
                if not isinstance(chunk, dict):
                    chunk = chunk.model_dump(exclude_none=True, by_alias=True)

                usage = chunk.get("usage")
                choices = chunk.get("choices") or []

                # vLLM's include_usage chunk carries real token counts with
                # empty choices; Ollama's final chunk has usage AND choices.
                # Always capture usage for accounting; forward the usage-only
                # chunk to the client only if they asked via
                # stream_options.include_usage (otherwise the client-visible
                # stream is unchanged).
                if usage is not None:
                    real_usage = UsageInfo(**usage)
                    if not choices and not getattr(request, "include_usage", False):
                        continue  # internal-only; suppress from the client

                chunk_count += 1

                # Accumulate content/reasoning and track finish reason
                force_flush = usage is not None
                for choice in choices:
                    delta = choice.get("delta") or {}
                    content = delta.get("content")
                    if content:
                        full_content += content
                        inflight_chars += len(content)
                    reasoning = delta.get("reasoning_content")
                    if reasoning:
                        inflight_chars += len(reasoning)
                    if choice.get("finish_reason"):
                        last_finish_reason = choice["finish_reason"]
                        force_flush = True

                # Format as SSE (None keys already omitted, matching the old
                # exclude_none serialization) and coalesce socket writes
                event = b"data: " + orjson.dumps(chunk) + b"\n\n"
                out = coalescer.add(event, force=force_flush)
                if out is not None:
                    yield out

                # Flush estimated tokens to Redis at coalesce-flush
                # boundaries (every 10 chunks when coalescing is disabled)
                if (
                    (out is not None if coalescer.enabled else chunk_count % 10 == 0)
                    and inflight_chars > 0
                ):
                    estimated = inflight_chars // 4
                    if estimated > 0:
                        await incr_inflight_tokens(estimated)
                        inflight_total_tokens += estimated
                        inflight_chars -= estimated * 4

            # Flush remaining chars
            if inflight_chars > 0:
                estimated = inflight_chars // 4
                if estimated > 0:
                    await incr_inflight_tokens(estimated)
                    inflight_total_tokens += estimated

            # Send done signal (drains any coalesced events ahead of it)
            out = coalescer.add(b"data: [DONE]\n\n", force=True)
            if out is not None:
                yield out

            # Update records — shield the entire completion path so that
            # CancelledError from client disconnect after [DONE] cannot
            # interrupt unshielded scheduler awaits before the DB write.
            if routed_backend:
                # Once the shielded completion starts, _fail_request must not
                # run: cancellation of THIS generator leaves the shielded task
                # finishing in the background, and a concurrent _fail_request
                # would use the same session mid-transaction.
                completion_started = True
                await asyncio.shield(self._complete_streaming_request(
                    db_request, routed_backend.id, full_content, chunk_count, job,
                    finish_reason=last_finish_reason,
                    usage=real_usage,
                ))

            completed = True
        except HTTPException as e:
            # Backend returned a 4xx error. If no chunks have been yielded,
            # we can emit the error as an SSE event so the client gets a
            # clean error message instead of a truncated chunked response.
            backend_id = routed_backend.id if routed_backend else None
            try:
                await asyncio.shield(
                    self._fail_request(db_request, backend_id, str(e.detail), job)
                )
            except BaseException:
                pass
            # Drain coalesced events so they precede the error event
            pending = coalescer.flush()
            if pending is not None:
                yield pending
            error_body = json.dumps({"error": {"message": str(e.detail), "type": "backend_error", "code": e.status_code}})
            yield f"data: {error_body}\n\ndata: [DONE]\n\n".encode()
        except BaseException as e:
            # BaseException catches CancelledError (not a subclass of Exception
            # in Python 3.9+) and GeneratorExit from client disconnects — both
            # of which would otherwise leak the scheduler queue depth counter.
            backend_id = routed_backend.id if routed_backend else None
            if not completion_started:
                try:
                    await asyncio.shield(
                        self._fail_request(db_request, backend_id, str(e), job)
                    )
                except BaseException:
                    pass
            # Mid-stream backend failure: drain token events the backend
            # already delivered so the client's truncated stream matches
            # main's per-event write behavior. Never yield for
            # CancelledError/GeneratorExit — the client is gone, and
            # yielding after GeneratorExit raises RuntimeError.
            if not isinstance(e, (asyncio.CancelledError, GeneratorExit)):
                pending = coalescer.flush()
                if pending is not None:
                    yield pending
            raise
        finally:
            # Always clean up inflight counter — handles normal completion,
            # exceptions, and client disconnects (GeneratorExit)
            if inflight_total_tokens > 0:
                await decr_inflight_tokens(inflight_total_tokens)

    async def embedding(
        self,
        request: CanonicalEmbeddingRequest,
        user: User,
        api_key: ApiKey,
        http_request: Request,
    ) -> Dict[str, Any]:
        """Handle embedding request."""
        await self._check_quota(user, api_key)

        db_request = await self._create_request_record(
            request, user, api_key, http_request, "/v1/embeddings",
            modality=Modality.EMBEDDING
        )

        job = self._scheduler.create_job_from_embedding_request(
            request, user.id, api_key.id
        )
        job.request_id = db_request.request_uuid

        completion_started = False
        try:
            response, backend = await self._proxy_with_retry(
                request, job, user,
                modality=Modality.EMBEDDING,
                proxy_fn="_proxy_embedding_request",
            )

            # Gate _fail_request once the shielded completion starts
            # (same session-race rationale as chat_completion).
            completion_started = True
            await self._complete_request(
                db_request, backend.id, response, job,
                modality=Modality.EMBEDDING
            )

            return response

        except BaseException as e:
            if not completion_started:
                try:
                    await self._fail_request(db_request, None, str(e), job)
                except Exception:
                    pass
            raise

    async def rerank(
        self,
        request: CanonicalRerankRequest,
        user: User,
        api_key: ApiKey,
        http_request: Request,
    ) -> Dict[str, Any]:
        """Handle rerank request."""
        await self._check_quota(user, api_key)

        db_request = await self._create_request_record(
            request, user, api_key, http_request, "/v1/rerank",
            modality=Modality.RERANKING
        )

        job = self._scheduler.create_job_from_rerank_request(
            request, user.id, api_key.id
        )
        job.request_id = db_request.request_uuid

        completion_started = False
        try:
            response, backend = await self._proxy_with_retry(
                request, job, user,
                modality=Modality.RERANKING,
                proxy_fn="_proxy_rerank_request",
            )

            # Gate _fail_request once the shielded completion starts
            # (same session-race rationale as chat_completion).
            completion_started = True
            await self._complete_request(
                db_request, backend.id, response, job,
                modality=Modality.RERANKING
            )

            return response

        except BaseException as e:
            if not completion_started:
                try:
                    await self._fail_request(db_request, None, str(e), job)
                except Exception:
                    pass
            raise

    async def score(
        self,
        request: CanonicalScoreRequest,
        user: User,
        api_key: ApiKey,
        http_request: Request,
    ) -> Dict[str, Any]:
        """Handle score request."""
        await self._check_quota(user, api_key)

        db_request = await self._create_request_record(
            request, user, api_key, http_request, "/v1/score",
            modality=Modality.RERANKING
        )

        job = self._scheduler.create_job_from_score_request(
            request, user.id, api_key.id
        )
        job.request_id = db_request.request_uuid

        completion_started = False
        try:
            response, backend = await self._proxy_with_retry(
                request, job, user,
                modality=Modality.RERANKING,
                proxy_fn="_proxy_score_request",
            )

            # Gate _fail_request once the shielded completion starts
            # (same session-race rationale as chat_completion).
            completion_started = True
            await self._complete_request(
                db_request, backend.id, response, job,
                modality=Modality.RERANKING
            )

            return response

        except BaseException as e:
            if not completion_started:
                try:
                    await self._fail_request(db_request, None, str(e), job)
                except Exception:
                    pass
            raise

    async def image_generation(
        self,
        request: CanonicalImageRequest,
        user: User,
        api_key: ApiKey,
        http_request: Request,
    ) -> Dict[str, Any]:
        """Handle image generation request."""
        await self._check_quota(user, api_key)

        db_request = await self._create_request_record(
            request, user, api_key, http_request, "/v1/images/generations",
            modality=Modality.IMAGE_GENERATION
        )

        job = self._scheduler.create_job_from_image_request(
            request, user.id, api_key.id
        )
        job.request_id = db_request.request_uuid

        completion_started = False
        try:
            response, backend = await self._proxy_with_retry(
                request, job, user,
                modality=Modality.IMAGE_GENERATION,
                proxy_fn="_proxy_image_request",
            )

            # Gate _fail_request once the shielded completion starts
            # (same session-race rationale as chat_completion).
            completion_started = True
            await self._complete_request(
                db_request, backend.id, response, job,
                modality=Modality.IMAGE_GENERATION
            )

            return response

        except BaseException as e:
            if not completion_started:
                try:
                    await self._fail_request(db_request, None, str(e), job)
                except Exception:
                    pass
            raise

    async def ollama_chat(
        self,
        request: CanonicalChatRequest,
        user: User,
        api_key: ApiKey,
        http_request: Request,
    ) -> Dict[str, Any]:
        """Handle Ollama chat request (non-streaming)."""
        await self._check_quota(user, api_key)

        db_request = await self._create_request_record(
            request, user, api_key, http_request, "/api/chat"
        )

        job = self._scheduler.create_job_from_chat_request(
            request, user.id, api_key.id
        )
        job.request_id = db_request.request_uuid

        completion_started = False
        try:
            response, backend = await self._proxy_with_retry(
                request, job, user, proxy_fn="_proxy_ollama_chat"
            )

            # Gate _fail_request once the shielded completion starts
            # (same session-race rationale as chat_completion).
            completion_started = True
            await self._complete_request(
                db_request, backend.id, response, job
            )

            return response

        except BaseException as e:
            if not completion_started:
                try:
                    await self._fail_request(db_request, None, str(e), job)
                except Exception:
                    pass
            raise

    async def stream_ollama_chat(
        self,
        request: CanonicalChatRequest,
        user: User,
        api_key: ApiKey,
        http_request: Request,
    ) -> AsyncIterator[bytes]:
        """Handle streaming Ollama chat request."""
        await self._check_quota(user, api_key)

        db_request = await self._create_request_record(
            request, user, api_key, http_request, "/api/chat"
        )

        job = self._scheduler.create_job_from_chat_request(
            request, user.id, api_key.id
        )
        job.request_id = db_request.request_uuid

        full_content = ""
        chunk_count = 0
        routed_backend = None
        inflight_chars = 0
        inflight_total_tokens = 0
        last_finish_reason = None
        completion_started = False
        coalescer = StreamCoalescer(
            self._settings.stream_coalesce_events,
            self._settings.stream_coalesce_ms,
        )

        try:
            async for chunk_data, backend in self._proxy_stream_with_retry(
                request, job, user, proxy_fn="_proxy_ollama_stream"
            ):
                routed_backend = backend
                chunk_count += 1

                if "message" in chunk_data:
                    content = chunk_data["message"].get("content", "")
                    full_content += content
                    inflight_chars += len(content)
                    thinking = chunk_data["message"].get("thinking", "")
                    if thinking:
                        inflight_chars += len(thinking)

                done = bool(chunk_data.get("done"))
                if done:
                    last_finish_reason = chunk_data.get("done_reason", "stop")

                # Coalesce NDJSON line writes; the final (done) chunk
                # always flushes
                out = coalescer.add(
                    (json.dumps(chunk_data) + "\n").encode(), force=done
                )
                if out is not None:
                    yield out

                # Flush estimated tokens to Redis at coalesce-flush
                # boundaries (every 10 chunks when coalescing is disabled)
                if (
                    (out is not None if coalescer.enabled else chunk_count % 10 == 0)
                    and inflight_chars > 0
                ):
                    estimated = inflight_chars // 4
                    if estimated > 0:
                        await incr_inflight_tokens(estimated)
                        inflight_total_tokens += estimated
                        inflight_chars -= estimated * 4

            # Drain any events still buffered (backend stream ended
            # without a done chunk)
            out = coalescer.flush()
            if out is not None:
                yield out

            # Flush remaining chars
            if inflight_chars > 0:
                estimated = inflight_chars // 4
                if estimated > 0:
                    await incr_inflight_tokens(estimated)
                    inflight_total_tokens += estimated

            if routed_backend:
                # Once the shielded completion starts, _fail_request must not
                # run concurrently on the same session (see stream_chat_completion).
                completion_started = True
                await asyncio.shield(self._complete_streaming_request(
                    db_request, routed_backend.id, full_content, chunk_count, job,
                    finish_reason=last_finish_reason,
                ))
        except HTTPException as e:
            backend_id = routed_backend.id if routed_backend else None
            try:
                await asyncio.shield(
                    self._fail_request(db_request, backend_id, str(e.detail), job)
                )
            except BaseException:
                pass
            # Drain coalesced events so they precede the error line
            pending = coalescer.flush()
            if pending is not None:
                yield pending
            error_body = json.dumps({"error": str(e.detail)})
            yield (error_body + "\n").encode()
        except BaseException as e:
            # BaseException catches CancelledError and GeneratorExit from
            # client disconnects that would otherwise leak queue depth.
            backend_id = routed_backend.id if routed_backend else None
            if not completion_started:
                try:
                    await asyncio.shield(
                        self._fail_request(db_request, backend_id, str(e), job)
                    )
                except BaseException:
                    pass
            # Mid-stream backend failure: drain token events the backend
            # already delivered so the client's truncated stream matches
            # main's per-event write behavior. Never yield for
            # CancelledError/GeneratorExit — the client is gone, and
            # yielding after GeneratorExit raises RuntimeError.
            if not isinstance(e, (asyncio.CancelledError, GeneratorExit)):
                pending = coalescer.flush()
                if pending is not None:
                    yield pending
            raise
        finally:
            # Always clean up inflight counter — handles normal completion,
            # exceptions, and client disconnects (GeneratorExit)
            if inflight_total_tokens > 0:
                await decr_inflight_tokens(inflight_total_tokens)

    async def stream_ollama_generate(
        self,
        request: CanonicalChatRequest,
        user: User,
        api_key: ApiKey,
        http_request: Request,
    ) -> AsyncIterator[bytes]:
        """Handle streaming Ollama generate request."""
        # Reuse chat streaming with different endpoint recorded
        async for chunk in self.stream_ollama_chat(request, user, api_key, http_request):
            yield chunk

    async def ollama_generate(
        self,
        request: CanonicalChatRequest,
        user: User,
        api_key: ApiKey,
        http_request: Request,
    ) -> Dict[str, Any]:
        """Handle Ollama generate request (non-streaming)."""
        result = await self.ollama_chat(request, user, api_key, http_request)
        # Convert chat format (message) to generate format (response)
        msg = result.pop("message", {})
        result["response"] = msg.get("content", "")
        # Preserve thinking/reasoning in generate response
        thinking = msg.get("thinking") or msg.get("reasoning") or msg.get("reasoning_content")
        if thinking:
            result["thinking"] = thinking
        return result

    async def ollama_embedding(
        self,
        request: CanonicalEmbeddingRequest,
        user: User,
        api_key: ApiKey,
        http_request: Request,
    ) -> Dict[str, Any]:
        """Handle Ollama embedding request."""
        return await self.embedding(request, user, api_key, http_request)

    # Flat quota cost per generated image. 0 disables the charge entirely
    # (images then cost nothing), mirroring search.quota_tokens_per_request.
    # The default is one average chat exchange on this cluster — measured at
    # ~10,663 total tokens — so "an image costs about as much as a
    # conversation turn" is a defensible, explainable rule.
    DEFAULT_IMAGE_TOKEN_COST = 10000

    async def _image_token_cost(self) -> int:
        """Configured quota cost per image (img.quota_tokens_per_image)."""
        try:
            from backend.app.db import crud as _crud

            raw = await _crud.get_config_json(
                self.db, "img.quota_tokens_per_image", self.DEFAULT_IMAGE_TOKEN_COST
            )
            cost = int(raw)
        except (TypeError, ValueError):
            return self.DEFAULT_IMAGE_TOKEN_COST
        except Exception:
            # A config read failure must not fail the request that just
            # succeeded; fall back to the default rather than charging nothing.
            logger.warning("image_token_cost_unreadable", exc_info=True)
            return self.DEFAULT_IMAGE_TOKEN_COST
        return max(0, cost)

    async def _check_quota(self, user: User, api_key: ApiKey) -> None:
        """Check if user has sufficient quota."""
        # Reset quota if period expired
        await crud.reset_quota_if_needed(self.db, user.id)

        quota = await crud.get_user_quota(self.db, user.id)
        group_budget = user.group.token_budget if user.group else 0
        if quota and group_budget > 0 and quota.tokens_used >= group_budget:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Token quota exceeded",
            )

        # RPM rate limit (Redis-backed, shared across all workers)
        rpm_limit = 0
        if api_key and api_key.rpm_limit:
            rpm_limit = api_key.rpm_limit
        elif quota:
            rpm_limit = quota.rpm_limit
        if rpm_limit > 0:
            from backend.app.core.redis_client import check_rpm
            allowed, current = await check_rpm(f"user:{user.id}", rpm_limit)
            if not allowed:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded: {rpm_limit} requests per minute (current: {current})",
                )

    async def _create_request_record(
        self,
        request: Any,
        user: User,
        api_key: ApiKey,
        http_request: Request,
        endpoint: str,
        modality: Modality = Modality.CHAT,
        extra_parameters: Optional[Dict[str, Any]] = None,
    ):
        """Create audit record for the request.

        Commits the pre-inference transaction immediately so no DB locks
        are held during the long-running inference phase.
        """
        messages = None
        prompt = None
        parameters = {}

        # Content capture is policy-gated: with prompt capture off, the
        # audit row keeps metadata (model, tokens, timings) but no
        # message/prompt text and no offloaded image files.  Note the
        # DLP worker scans the STORED content, so disabling capture also
        # disables DLP scanning of that content.
        capture_prompts = (
            self._settings.audit_log_enabled and self._settings.audit_log_prompts
        )
        if capture_prompts:
            if hasattr(request, "messages"):
                messages = [m.model_dump() for m in request.messages]
                # Extract inline base64 images to filesystem to avoid exceeding
                # DB packet limits and bloating audit logs.  Each image is saved
                # under /data/artifacts/request_images/<request_uuid>/ and the
                # audit record stores a lightweight reference instead.
                messages = self._extract_images_from_messages(messages)
            if hasattr(request, "prompt"):
                prompt = request.prompt if isinstance(request.prompt, str) else str(request.prompt)

        # Extract parameters
        for param in ["temperature", "top_p", "max_tokens", "stop"]:
            if hasattr(request, param) and getattr(request, param) is not None:
                parameters[param] = getattr(request, param)

        # Image-specific parameters
        for param in ["n", "size", "num_inference_steps", "guidance_scale", "seed"]:
            if hasattr(request, param) and getattr(request, param) is not None:
                parameters[param] = getattr(request, param)
        if hasattr(request, "policy_verdict") and request.policy_verdict:
            parameters["policy_verdict"] = request.policy_verdict
        # Dialect-supplied extras (e.g. the public resp_* id of a
        # /v1/responses call, so audit rows are searchable by the id
        # the client saw).
        if extra_parameters:
            parameters.update(extra_parameters)

        response_format = None
        if hasattr(request, "response_format") and request.response_format:
            rf = request.response_format
            if hasattr(rf, "model_dump"):
                response_format = rf.model_dump()
            else:
                response_format = {"type": str(rf)}

        db_request = await crud.create_request(
            db=self.db,
            user_id=user.id,
            api_key_id=api_key.id,
            endpoint=endpoint,
            model=request.model,
            modality=modality,
            is_streaming=getattr(request, "stream", False),
            messages=messages,
            prompt=prompt,
            parameters=parameters,
            response_format=response_format,
            client_ip=self._get_client_ip(http_request),
            user_agent=http_request.headers.get("user-agent"),
        )

        # Commit pre-inference writes (request INSERT, any quota changes)
        # so no row locks are held during the long-running inference phase.
        # expire_on_commit=False ensures db_request.id remains accessible.
        await self.db.commit()

        # DLP prompt redaction: the audit row above captured the ORIGINAL text;
        # now strip the offending values from the request that is dispatched to
        # the backend, so the model never sees them. The async worker still
        # scans the stored original, so alerting is unaffected.
        if self._pending_prompt_redactions:
            try:
                self._apply_prompt_redactions(request, self._pending_prompt_redactions)
            except Exception:
                logger.warning("dlp_prompt_redaction_failed",
                               request_id=getattr(db_request, "id", None))
            finally:
                self._pending_prompt_redactions = []

        return db_request

    def _extract_images_from_messages(self, messages: list) -> list:
        """
        Extract inline base64 images from serialized messages and save them
        to the filesystem.  Replaces the image data in the messages dict with
        a lightweight reference containing the storage path and size.

        This prevents large multimodal requests (OCR, vision) from exceeding
        MariaDB's max_allowed_packet and from bloating the audit log.
        """
        import base64 as _b64
        from pathlib import Path

        images_dir = None  # lazily created
        img_idx = 0

        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue

                # Handle image_url blocks with data: URIs
                iu = block.get("image_url")
                if isinstance(iu, dict) and isinstance(iu.get("url"), str):
                    url = iu["url"]
                    if url.startswith("data:") and len(url) > 1024:
                        if images_dir is None:
                            images_dir = self._make_request_images_dir()
                        path = self._save_audit_image(
                            images_dir, img_idx, url
                        )
                        img_idx += 1
                        iu["url"] = f"file://{path}"
                        iu["_stored_size_bytes"] = path.stat().st_size if path.exists() else 0

                # Handle image_base64 blocks
                if block.get("type") == "image_base64" and isinstance(block.get("data"), str):
                    data = block["data"]
                    if len(data) > 1024:
                        if images_dir is None:
                            images_dir = self._make_request_images_dir()
                        media_type = block.get("media_type", "image/png")
                        ext = media_type.split("/")[-1].split(";")[0]
                        path = images_dir / f"{img_idx}.{ext}"
                        try:
                            path.write_bytes(_b64.b64decode(data))
                        except Exception:
                            pass
                        img_idx += 1
                        block["data"] = f"file://{path}"
                        block["_stored_size_bytes"] = path.stat().st_size if path.exists() else 0

        return messages

    def _make_request_images_dir(self):
        """Create and return a directory for storing audit images."""
        import uuid as _uuid
        from pathlib import Path

        base = Path(self._settings.artifact_storage_path) / "request_images"
        dirname = _uuid.uuid4().hex[:16]
        img_dir = base / dirname
        img_dir.mkdir(parents=True, exist_ok=True)
        return img_dir

    @staticmethod
    def _save_audit_image(images_dir, idx: int, data_url: str):
        """Save a data: URI image to the filesystem. Returns the Path."""
        import base64 as _b64

        # Parse data:image/png;base64,AAAA...
        header, _, b64_data = data_url.partition(",")
        # Extract extension from media type
        media = header.split(":")[1].split(";")[0] if ":" in header else "image/png"
        ext = media.split("/")[-1]
        if ext not in ("png", "jpeg", "jpg", "webp", "gif", "tiff", "bmp"):
            ext = "png"
        path = images_dir / f"{idx}.{ext}"
        try:
            path.write_bytes(_b64.b64decode(b64_data))
        except Exception:
            # If decode fails, write raw truncated reference
            path.write_bytes(b"(decode failed)")
        return path

    @staticmethod
    def _get_client_ip(http_request: Request) -> Optional[str]:
        """Extract the real client IP from proxy headers, falling back to direct connection."""
        forwarded_for = http_request.headers.get("x-forwarded-for")
        if forwarded_for:
            # X-Forwarded-For is comma-separated; first entry is the original client
            return forwarded_for.split(",")[0].strip()
        real_ip = http_request.headers.get("x-real-ip")
        if real_ip:
            return real_ip.strip()
        return http_request.client.host if http_request.client else None

    async def _route_request(
        self,
        job: Job,
        user: User,
        modality: Optional[Modality] = None,
        exclude_backend_ids: Optional[Set[int]] = None,
        max_wait: Optional[float] = None,
    ):
        """Route request to a backend, waiting for capacity if needed.

        Args:
            max_wait: Maximum seconds to wait for capacity.  ``None`` (default)
                uses ``backend_request_timeout / 2``.  Pass ``0`` to fail
                immediately if no backend is available right now.
        """
        with _tracer.start_as_current_span(
            "mindrouter.route_request",
            attributes={
                "mindrouter.model": job.model,
                "mindrouter.request_id": job.request_id or "",
            },
        ):
            return await self._route_request_inner(
                job, user, modality, exclude_backend_ids, max_wait,
            )

    async def _route_request_inner(
        self,
        job: Job,
        user: User,
        modality: Optional[Modality] = None,
        exclude_backend_ids: Optional[Set[int]] = None,
        max_wait: Optional[float] = None,
    ):
        # Get backends that support the model
        backends = await self._registry.get_backends_with_model(
            job.model, modality
        )

        if not backends:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{job.model}' not found",
            )

        # Filter out circuit-broken backends
        available = []
        for b in backends:
            if await self._registry.is_backend_available(b.id):
                available.append(b)
        if available:
            backends = available

        # Get models for each backend
        backend_models = {}
        for backend in backends:
            backend_models[backend.id] = await self._registry.get_backend_models(
                backend.id
            )

        # Get GPU utilizations and latency EMAs
        gpu_utilizations = await self._registry.get_gpu_utilizations()
        latency_emas = await self._latency_tracker.get_all_latencies()

        # Compute user weight (needed for submit and priority recomputation)
        user_weight = 1.0
        if hasattr(user, 'group') and user.group:
            user_weight = float(user.group.scheduler_weight)
        sched_quota = await crud.get_user_quota(self.db, user.id)
        if sched_quota and sched_quota.weight_override:
            user_weight = float(sched_quota.weight_override)

        # Submit to scheduler (only on first attempt — not on retries)
        if not exclude_backend_ids:
            await self._scheduler.submit_job(job, user_role=user.role.value, user_weight=user_weight)

        # Retry loop: wait for capacity instead of immediately 503-ing
        # Use half the backend timeout for routing, leaving the rest for inference
        route_timeout = max_wait if max_wait is not None else self._settings.backend_request_timeout / 2
        deadline = time.monotonic() + route_timeout
        last_reason = ""
        waiter_registered = False

        while True:
            try:
                decision = await self._scheduler.route_job(
                    job, backends, backend_models, gpu_utilizations,
                    exclude_backend_ids=exclude_backend_ids,
                    latency_emas=latency_emas,
                )
            except Exception:
                if waiter_registered:
                    self._scheduler.unregister_waiter(job)
                await self._scheduler.cancel_job(job.request_id)
                raise

            if decision.success:
                if waiter_registered:
                    self._scheduler.unregister_waiter(job)
                return decision.backend, backend_models.get(decision.backend.id, [])

            last_reason = decision.reason

            # Determine if this is a transient capacity issue (worth retrying)
            # or a permanent failure (fail immediately).
            # ONLY an explicit no_capacity constraint — where a backend could
            # serve the request once a slot opens — is transient.  Everything
            # else (missing model, wrong modality, no backends at all) is
            # permanent and must not enter the wait loop.
            capacity_issue = False
            if decision.all_scores:
                for s in decision.all_scores:
                    if s.failed_constraints == ["no_capacity"]:
                        capacity_issue = True
                        break

            if not capacity_issue:
                if waiter_registered:
                    self._scheduler.unregister_waiter(job)
                await self._scheduler.cancel_job(job.request_id)

                # Collect failed constraints for a user-friendly error message
                all_failed = set()
                for s in (decision.all_scores or []):
                    if s.failed_constraints:
                        all_failed.update(s.failed_constraints)

                if "modality_not_supported" in all_failed:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Model '{job.model}' does not support multimodal/image input",
                    )
                elif "structured_output_not_supported" in all_failed:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Model '{job.model}' does not support structured output",
                    )
                else:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail=f"No suitable backend: {last_reason}",
                    )

            # Register as waiter on first capacity failure
            if not waiter_registered:
                self._scheduler.register_waiter(job)
                waiter_registered = True

            # Check deadline
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._scheduler.unregister_waiter(job)
                await self._scheduler.cancel_job(job.request_id)
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"No backend capacity available (waited {route_timeout}s)",
                )

            # Wait for a slot to open (capped at remaining time)
            wait_time = min(5.0, remaining)
            await self._scheduler.wait_for_capacity(wait_time)

            # Recompute priority with current fair-share state
            await self._scheduler.recompute_priority(
                job, role=user.role.value, weight=user_weight,
            )

            # Priority gate: only the highest-priority waiter for this model
            # proceeds to route_job(); lower-priority waiters yield briefly.
            if not self._scheduler.is_highest_priority_waiter(job):
                await asyncio.sleep(0.1)
                continue

            # Refresh backend state for next attempt
            backends = await self._registry.get_backends_with_model(
                job.model, modality
            )
            if not backends:
                backends = await self._registry.get_healthy_backends()
            if not backends:
                self._scheduler.unregister_waiter(job)
                await self._scheduler.cancel_job(job.request_id)
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="No healthy backends available",
                )

            # Filter out circuit-broken backends
            available = []
            for b in backends:
                if await self._registry.is_backend_available(b.id):
                    available.append(b)
            if available:
                backends = available

            backend_models = {}
            for backend in backends:
                backend_models[backend.id] = await self._registry.get_backend_models(
                    backend.id
                )
            gpu_utilizations = await self._registry.get_gpu_utilizations()
            latency_emas = await self._latency_tracker.get_all_latencies()

    # ------------------------------------------------------------------
    # Retry-with-failover wrappers
    # ------------------------------------------------------------------

    async def _proxy_with_retry(
        self,
        request,
        job: Job,
        user: User,
        modality: Optional[Modality] = None,
        proxy_fn: str = "_proxy_chat_request",
    ) -> Tuple[Any, Backend]:
        """Execute a proxy call with retry on different backends.

        On timeout, 5xx, or connection error the request is retried on a
        different backend (up to ``backend_retry_max_attempts`` total).
        4xx errors are NOT retried (client errors).

        Returns:
            (response, backend) on success.
        """
        span = trace.get_current_span()
        span.set_attribute("mindrouter.model", job.model)
        span.set_attribute("mindrouter.request_id", job.request_id or "")
        max_attempts = self._settings.backend_retry_max_attempts
        tried_backends: Set[int] = set()
        last_error: Optional[Exception] = None
        context_recap_done = False

        for attempt in range(max_attempts):
            # First attempt waits normally for capacity; retries fail fast.
            retry_wait = None if attempt == 0 else 0
            try:
                backend, models = await self._route_request(
                    job, user, modality,
                    exclude_backend_ids=tried_backends or None,
                    max_wait=retry_wait,
                )
            except HTTPException:
                # No routable backends with exclusions — clear exclusions so we
                # can retry a previously-tried backend (e.g. single-backend model
                # with a transient 5xx).
                if tried_backends:
                    tried_backends.clear()
                    try:
                        backend, models = await self._route_request(
                            job, user, modality, max_wait=retry_wait,
                        )
                    except HTTPException:
                        # Still no backends — all are circuit-broken.  Fail fast.
                        break
                else:
                    raise

            # Cap max_tokens using actual token count (vLLM /tokenize) or
            # tiktoken estimate (Ollama / fallback).
            cap_context_length = None
            if models and hasattr(request, 'max_tokens'):
                _target = next((m for m in models if m.name == job.model), models[0])
                if _target.context_length:
                    cap_context_length = _target.context_length
                    await self.cap_max_tokens(request, backend, cap_context_length)

            # Strip thinking mode for OLLAMA models that don't support it
            # (Ollama 400s if think is set on a non-thinking model). vLLM ignores
            # enable_thinking harmlessly, so never strip there: supports_thinking is
            # discovery-populated and can be stale/inconsistent across replicas,
            # which would otherwise wrongly block a client's opt-in.
            if (backend.engine == BackendEngine.OLLAMA
                    and models and hasattr(request, 'think') and request.think):
                _target = next((m for m in models if m.name == job.model), models[0])
                if not getattr(_target, 'supports_thinking', False):
                    request.think = None

            # Gateway policy: reasoning/thinking is OFF by default unless the
            # client opts in. Applies to all engines except gpt-oss (which uses
            # reasoning_effort and ignores enable_thinking; leaving think None
            # preserves its reasoning-promotion handling). For Ollama, think:false
            # is accepted by both thinking and non-thinking models (verified), and
            # the Ollama-only strip above still guards think:true on non-thinking
            # models.
            if (
                self._settings.thinking_off_by_default
                and hasattr(request, 'think')
                and request.think is None
                and "gpt-oss" not in (job.model or "").lower()
            ):
                request.think = False

            # Inject num_ctx for Ollama backends from model config
            if backend.engine == BackendEngine.OLLAMA and models and hasattr(request, 'backend_options'):
                _ctx_target = next((m for m in models if m.name == job.model), models[0])
                if _ctx_target.context_length:
                    if request.backend_options is None:
                        request.backend_options = {}
                    enforce = await _get_enforce_num_ctx()
                    if enforce:
                        request.backend_options["num_ctx"] = _ctx_target.context_length
                    else:
                        request.backend_options.setdefault("num_ctx", _ctx_target.context_length)

            tried_backends.add(backend.id)
            start_time = time.monotonic()

            try:
                response = await asyncio.wait_for(
                    getattr(self, proxy_fn)(request, backend),
                    timeout=float(self._settings.backend_request_timeout_per_attempt),
                )
                # Success — record latency
                elapsed_ms = (time.monotonic() - start_time) * 1000
                await self._registry.report_live_success(backend.id)
                await self._latency_tracker.record_latency(backend.id, elapsed_ms)
                span.set_attribute("mindrouter.backend_id", backend.id)
                span.set_attribute("mindrouter.backend_engine", str(backend.engine))
                span.set_attribute("mindrouter.latency_ms", elapsed_ms)
                span.set_attribute("mindrouter.attempt", attempt + 1)
                return response, backend

            except (asyncio.TimeoutError, httpx.TimeoutException) as e:
                # The per-request httpx client's async-with block closes the
                # TCP connection on cancellation, signalling vLLM to abort
                # the in-progress generation and free the execution slot.
                logger.warning(
                    "backend_timeout",
                    backend_id=backend.id,
                    attempt=attempt + 1,
                    max_attempts=max_attempts,
                    elapsed_ms=(time.monotonic() - start_time) * 1000,
                )
                await self._scheduler.on_job_failed(job, backend.id)
                await self._registry.report_live_failure(backend.id)
                last_error = e

            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500:
                    logger.warning(
                        "backend_5xx",
                        backend_id=backend.id,
                        status=e.response.status_code,
                        attempt=attempt + 1,
                    )
                    await self._scheduler.on_job_failed(job, backend.id)
                    await self._registry.report_live_failure(backend.id)
                    last_error = e
                else:
                    # 4xx — not retryable, but must release slot
                    await self._scheduler.on_job_failed(job, backend.id)
                    try:
                        detail = e.response.json()
                    except Exception:
                        detail = e.response.text or str(e)

                    # Safety net for the /tokenize gate: a vLLM 400 about
                    # context length means the conservative estimate was too
                    # low — recount exactly and retry once on the same backend.
                    if (
                        not context_recap_done
                        and attempt < max_attempts - 1
                        and cap_context_length
                        and backend.engine == BackendEngine.VLLM
                        and e.response.status_code == 400
                        and _is_context_length_error(detail)
                    ):
                        context_recap_done = True
                        await self.cap_max_tokens(
                            request, backend, cap_context_length, force_exact=True,
                        )
                        tried_backends.discard(backend.id)
                        last_error = e
                        continue

                    logger.warning(
                        "backend_4xx",
                        backend_id=backend.id,
                        status=e.response.status_code,
                        detail=detail,
                    )
                    raise HTTPException(
                        status_code=e.response.status_code,
                        detail=detail,
                    )

            except (httpx.ConnectError, httpx.RemoteProtocolError, ConnectionError) as e:
                logger.warning(
                    "backend_connection_error",
                    backend_id=backend.id,
                    attempt=attempt + 1,
                    error=str(e),
                )
                await self._scheduler.on_job_failed(job, backend.id)
                await self._registry.report_live_failure(backend.id)
                last_error = e

        # All attempts exhausted
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"All {max_attempts} backend attempts failed. Last error: {last_error}",
        )

    async def _proxy_stream_with_retry(
        self,
        request,
        job: Job,
        user: User,
        proxy_fn: str = "_proxy_stream_request",
        modality: Optional[Modality] = None,
    ) -> AsyncIterator[Tuple[Any, Backend]]:
        """Stream with retry support.

        Retries are only possible BEFORE the first chunk is yielded.
        Once streaming begins, failures are terminal (the client already
        has partial data).

        Yields:
            (chunk, backend) tuples.
        """
        span = trace.get_current_span()
        span.set_attribute("mindrouter.model", job.model)
        span.set_attribute("mindrouter.request_id", job.request_id or "")
        span.set_attribute("mindrouter.streaming", True)
        max_attempts = self._settings.backend_retry_max_attempts
        tried_backends: Set[int] = set()
        last_error: Optional[Exception] = None
        context_recap_done = False

        for attempt in range(max_attempts):
            # First attempt waits normally for capacity; retries fail fast.
            retry_wait = None if attempt == 0 else 0
            try:
                backend, _models = await self._route_request(
                    job, user, modality,
                    exclude_backend_ids=tried_backends or None,
                    max_wait=retry_wait,
                )
            except HTTPException:
                if tried_backends:
                    tried_backends.clear()
                    try:
                        backend, _models = await self._route_request(
                            job, user, modality, max_wait=retry_wait,
                        )
                    except HTTPException:
                        break
                else:
                    raise

            # Cap max_tokens (same logic as non-streaming path).
            cap_context_length = None
            if _models and hasattr(request, 'max_tokens'):
                _target = next((m for m in _models if m.name == job.model), _models[0])
                if _target.context_length:
                    cap_context_length = _target.context_length
                    await self.cap_max_tokens(request, backend, cap_context_length)

            # Strip thinking mode for OLLAMA models that don't support it
            # (see non-streaming path for rationale; never strip on vLLM).
            if (backend.engine == BackendEngine.OLLAMA
                    and _models and hasattr(request, 'think') and request.think):
                _target = next((m for m in _models if m.name == job.model), _models[0])
                if not getattr(_target, 'supports_thinking', False):
                    request.think = None

            # Gateway policy: reasoning/thinking is OFF by default unless the
            # client opts in (see non-streaming path). All engines except gpt-oss.
            if (
                self._settings.thinking_off_by_default
                and hasattr(request, 'think')
                and request.think is None
                and "gpt-oss" not in (job.model or "").lower()
            ):
                request.think = False

            # Inject num_ctx for Ollama backends from model config
            if backend.engine == BackendEngine.OLLAMA and _models and hasattr(request, 'backend_options'):
                _ctx_target = next((m for m in _models if m.name == job.model), _models[0])
                if _ctx_target.context_length:
                    if request.backend_options is None:
                        request.backend_options = {}
                    enforce = await _get_enforce_num_ctx()
                    if enforce:
                        request.backend_options["num_ctx"] = _ctx_target.context_length
                    else:
                        request.backend_options.setdefault("num_ctx", _ctx_target.context_length)

            tried_backends.add(backend.id)
            start_time = time.monotonic()
            first_chunk_received = False

            try:
                async for chunk in getattr(self, proxy_fn)(request, backend):
                    if not first_chunk_received:
                        first_chunk_received = True
                        ttft_ms = (time.monotonic() - start_time) * 1000
                        await self._registry.report_live_success(backend.id)
                        await self._latency_tracker.record_ttft(backend.id, ttft_ms)
                        span.set_attribute("mindrouter.backend_id", backend.id)
                        span.set_attribute("mindrouter.ttft_ms", ttft_ms)
                        span.set_attribute("mindrouter.attempt", attempt + 1)

                    yield chunk, backend

                # Stream completed successfully — record total latency
                total_ms = (time.monotonic() - start_time) * 1000
                await self._latency_tracker.record_latency(backend.id, total_ms)
                return

            except (
                asyncio.TimeoutError,
                httpx.TimeoutException,
                httpx.ConnectError,
                httpx.RemoteProtocolError,
                ConnectionError,
            ) as e:
                if first_chunk_received:
                    # Can't retry after streaming started
                    raise

                logger.warning(
                    "stream_backend_failure",
                    backend_id=backend.id,
                    attempt=attempt + 1,
                    error=str(e),
                )
                await self._scheduler.on_job_failed(job, backend.id)
                await self._registry.report_live_failure(backend.id)
                last_error = e

            except httpx.HTTPStatusError as e:
                if e.response.status_code < 500:
                    # 4xx — not retryable, but must release slot
                    await self._scheduler.on_job_failed(job, backend.id)
                    try:
                        detail = e.response.json()
                    except Exception:
                        detail = e.response.text or str(e)

                    # Safety net for the /tokenize gate (see _proxy_with_retry).
                    # 4xx always precedes the first chunk, so retry is safe.
                    if (
                        not context_recap_done
                        and not first_chunk_received
                        and attempt < max_attempts - 1
                        and cap_context_length
                        and backend.engine == BackendEngine.VLLM
                        and e.response.status_code == 400
                        and _is_context_length_error(detail)
                    ):
                        context_recap_done = True
                        await self.cap_max_tokens(
                            request, backend, cap_context_length, force_exact=True,
                        )
                        tried_backends.discard(backend.id)
                        last_error = e
                        continue

                    logger.warning(
                        "stream_backend_4xx",
                        backend_id=backend.id,
                        status=e.response.status_code,
                        detail=detail,
                    )
                    raise HTTPException(
                        status_code=e.response.status_code,
                        detail=detail,
                    )
                if first_chunk_received:
                    raise
                logger.warning(
                    "stream_backend_5xx",
                    backend_id=backend.id,
                    status=e.response.status_code,
                    attempt=attempt + 1,
                )
                await self._scheduler.on_job_failed(job, backend.id)
                await self._registry.report_live_failure(backend.id)
                last_error = e

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"All {max_attempts} streaming attempts failed. Last error: {last_error}",
        )

    # ------------------------------------------------------------------
    # Proxy methods (unchanged — called by retry wrappers above)
    # ------------------------------------------------------------------

    async def _proxy_chat_request(
        self,
        request: CanonicalChatRequest,
        backend: Backend,
    ) -> Dict[str, Any]:
        """Proxy chat request to backend."""
        if backend.engine == BackendEngine.OLLAMA:
            payload = OllamaOutTranslator.translate_chat_request(request)
            url = f"{backend.url}/api/chat"
        else:
            payload = VLLMOutTranslator.translate_chat_request(request)
            url = f"{backend.url}/v1/chat/completions"

        payload["stream"] = False

        async with self._make_inference_client() as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

        # Translate response to OpenAI format
        if backend.engine == BackendEngine.OLLAMA:
            canonical = OllamaOutTranslator.translate_chat_response(
                data, request.request_id, request.model
            )
        else:
            thinking_enabled = request.think if request.think is not None else True
            canonical = VLLMOutTranslator.translate_chat_response(
                data, request.request_id,
                thinking_enabled=thinking_enabled,
            )

        result = canonical.model_dump(exclude_none=True, by_alias=True)

        # OpenAI spec requires "content" in message even when null
        for choice in result.get("choices", []):
            msg = choice.get("message")
            if msg is not None and "content" not in msg:
                msg["content"] = None

        # Detect structured output requests where reasoning exhausted the
        # token budget before any content was generated.  Returning a 200
        # with content=None is misleading — the client asked for structured
        # output and got nothing usable.
        if (
            request.response_format
            and request.response_format.type in (ResponseFormatType.JSON_OBJECT, ResponseFormatType.JSON_SCHEMA)
        ):
            choices = result.get("choices", [])
            if choices:
                c = choices[0]
                content = (c.get("message") or {}).get("content")
                finish = c.get("finish_reason")
                if content is None:
                    if finish == "length":
                        raise HTTPException(
                            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail=(
                                "Structured output was requested but the model's reasoning/thinking "
                                "consumed the entire token budget before generating content. "
                                "Increase max_tokens or use a lower reasoning_effort."
                            ),
                        )
                    elif finish == "stop":
                        raise HTTPException(
                            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail=(
                                "Structured output was requested but the model completed without "
                                "generating any content. The model may not support structured "
                                "output with thinking enabled. Try disabling thinking or use a "
                                "different model."
                            ),
                        )

        return result

    async def _proxy_stream_request(
        self,
        request: CanonicalChatRequest,
        backend: Backend,
    ) -> AsyncIterator[Any]:
        """Proxy streaming request to backend.

        Yields wire-shaped chunk dicts for vLLM backends and
        CanonicalStreamChunk models for Ollama backends; the consumer
        normalizes.
        """
        if backend.engine == BackendEngine.OLLAMA:
            payload = OllamaOutTranslator.translate_chat_request(request)
            url = f"{backend.url}/api/chat"
        else:
            payload = VLLMOutTranslator.translate_chat_request(request)
            url = f"{backend.url}/v1/chat/completions"

        payload["stream"] = True

        async with self._make_inference_client() as client:
            async with client.stream("POST", url, json=payload) as response:
                if response.status_code >= 400:
                    await response.aread()
                    response.raise_for_status()

                if backend.engine == BackendEngine.OLLAMA:
                    async for chunk in OllamaOutTranslator.translate_chat_stream(
                        response.aiter_bytes(), request.request_id, request.model
                    ):
                        yield chunk
                else:
                    thinking_enabled = request.think if request.think is not None else True
                    async for chunk in VLLMOutTranslator.translate_chat_stream(
                        response.aiter_bytes(), request.request_id, request.model,
                        thinking_enabled=thinking_enabled,
                    ):
                        yield chunk

    async def _proxy_embedding_request(
        self,
        request: CanonicalEmbeddingRequest,
        backend: Backend,
    ) -> Dict[str, Any]:
        """Proxy embedding request to backend."""
        if backend.engine == BackendEngine.OLLAMA:
            payload = OllamaOutTranslator.translate_embedding_request(request)
            url = f"{backend.url}/api/embeddings"
        else:
            payload = VLLMOutTranslator.translate_embedding_request(request)
            url = f"{backend.url}/v1/embeddings"

        async with self._make_inference_client() as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

        if backend.engine == BackendEngine.OLLAMA:
            canonical = OllamaOutTranslator.translate_embedding_response(
                data, request.model
            )
        else:
            canonical = VLLMOutTranslator.translate_embedding_response(data)

        return canonical.model_dump(exclude_none=True, by_alias=True)

    async def _proxy_rerank_request(
        self,
        request: CanonicalRerankRequest,
        backend: Backend,
    ) -> Dict[str, Any]:
        """Proxy rerank request to backend (vLLM only)."""
        if backend.engine == BackendEngine.OLLAMA:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Reranking is not supported on Ollama backends",
            )

        async with self._make_inference_client() as client:
            payload = VLLMOutTranslator.translate_rerank_request(request)
            url = f"{backend.url}/v1/rerank"

            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

        canonical = VLLMOutTranslator.translate_rerank_response(data)
        result = canonical.model_dump(exclude_none=True, by_alias=True)

        # vLLM always returns documents; strip them if client asked not to
        if not request.return_documents:
            for r in result.get("results", []):
                r.pop("document", None)

        return result

    async def _proxy_score_request(
        self,
        request: CanonicalScoreRequest,
        backend: Backend,
    ) -> Dict[str, Any]:
        """Proxy score request to backend (vLLM only)."""
        if backend.engine == BackendEngine.OLLAMA:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Scoring is not supported on Ollama backends",
            )

        async with self._make_inference_client() as client:
            payload = VLLMOutTranslator.translate_score_request(request)
            url = f"{backend.url}/v1/score"

            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

        canonical = VLLMOutTranslator.translate_score_response(data)
        return canonical.model_dump(exclude_none=True, by_alias=True)

    async def _proxy_image_request(
        self,
        request: CanonicalImageRequest,
        backend: Backend,
    ) -> Dict[str, Any]:
        """Proxy image generation (or reference-edit) to a diffusion backend.

        The backend exposes OpenAI-compatible ``/v1/images/generations`` and, for
        img2img, ``/v1/images/edits``. When the request carries reference image(s)
        it is routed to the edits route.
        """
        payload = DiffusionOutTranslator.translate_image_request(request)
        route = "edits" if request.image else "generations"
        url = f"{backend.url}/v1/images/{route}"

        # Invisible provenance watermark (TrustMark) config. Read BEFORE
        # dispatch, and fail-safe: the image path must never 500 on a config
        # read — the GPU seconds are already spent by the time marking runs,
        # so a DB blip falls back to marking with the defaults.
        from backend.app.services import image_watermark

        wm_text = image_watermark.WATERMARK_DEFAULT_TEXT
        wm_enabled = True
        try:
            from backend.app.db.session import get_async_db_context

            async with get_async_db_context() as wm_db:
                wm_enabled = bool(
                    await crud.get_config_json(wm_db, "img.watermark_enabled", True)
                )
                wm_text = str(
                    await crud.get_config_json(
                        wm_db, "img.watermark_text", image_watermark.WATERMARK_DEFAULT_TEXT
                    )
                )
        except Exception:
            logger.warning("watermark_config_read_failed_using_defaults")
        if image_watermark.validate_watermark_text(wm_text) is not None:
            wm_text = image_watermark.WATERMARK_DEFAULT_TEXT

        # Watermarking needs the bytes, so force b64 from the backend. This
        # also closes the response_format="url" bypass: url mode returned a
        # RELATIVE link into the backend node's public static mount — dead
        # through the gateway AND an unmarked copy left on the node — so
        # clients get b64_json entries whenever marking is on.
        if wm_enabled:
            payload["response_format"] = "b64_json"

        # Image generation can be slow — use a longer timeout
        timeout = max(
            float(self._settings.backend_request_timeout_per_attempt),
            300.0,
        )

        async with self._make_inference_client() as client:
            response = await client.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()

        canonical = DiffusionOutTranslator.translate_image_response(data)

        # Apply the watermark at the gateway so every diffusion backend and
        # both routes (generations/edits) return marked bytes — and the
        # gallery stores exactly what the client received.
        # apply_watermark_b64 fails open: unmarked beats failed.
        if wm_enabled:
            for item in canonical.data:
                if item.b64_json:
                    item.b64_json = await image_watermark.apply_watermark_b64(
                        item.b64_json, wm_text
                    )

        return canonical.model_dump(exclude_none=True, by_alias=True)

    async def _proxy_ollama_chat(
        self,
        request: CanonicalChatRequest,
        backend: Backend,
    ) -> Dict[str, Any]:
        """Proxy chat request, return in Ollama format."""
        if backend.engine == BackendEngine.OLLAMA:
            payload = OllamaOutTranslator.translate_chat_request(request)
            payload["stream"] = False
            url = f"{backend.url}/api/chat"
        else:
            payload = VLLMOutTranslator.translate_chat_request(request)
            payload["stream"] = False
            url = f"{backend.url}/v1/chat/completions"

        async with self._make_inference_client() as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

        # Convert OpenAI format to Ollama format when backend is not Ollama
        if backend.engine != BackendEngine.OLLAMA:
            thinking_enabled = request.think if request.think is not None else True
            data = self._openai_response_to_ollama(data, thinking_enabled=thinking_enabled)

        # Detect structured output requests where reasoning exhausted the
        # token budget before any content was generated.
        if (
            request.response_format
            and request.response_format.type in (ResponseFormatType.JSON_OBJECT, ResponseFormatType.JSON_SCHEMA)
        ):
            msg = data.get("message", {})
            content = msg.get("content")
            done_reason = data.get("done_reason")
            # Ollama uses done_reason; vLLM-converted uses done_reason from _openai_response_to_ollama
            if not content:
                if done_reason != "stop":
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail=(
                            "Structured output was requested but the model's reasoning/thinking "
                            "consumed the entire token budget before generating content. "
                            "Increase max_tokens (num_predict) or use a lower reasoning_effort."
                        ),
                    )
                elif msg.get("thinking") or msg.get("reasoning") or msg.get("reasoning_content"):
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail=(
                            "Structured output was requested but the model completed without "
                            "generating any content. The model may not support structured "
                            "output with thinking enabled. Try disabling thinking or use a "
                            "different model."
                        ),
                    )

        return data

    def _openai_response_to_ollama(
        self, openai_response: Dict, thinking_enabled: bool = True,
    ) -> Dict:
        """Convert a non-streaming OpenAI response to Ollama format."""
        choices = openai_response.get("choices", [])
        message = {"role": "assistant", "content": ""}
        finish_reason = "stop"
        if choices:
            msg = choices[0].get("message", {})
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or msg.get("reasoning")

            # vLLM/Qwen3.5 bug: when thinking is disabled the model may
            # put all output into reasoning_content with content empty.
            # Promote reasoning to content in that case.
            if (
                not thinking_enabled and not content and reasoning
                and _reasoning_promotion_applies(openai_response.get("model", ""))
            ):
                content = reasoning
                reasoning = None

            message = {
                "role": msg.get("role", "assistant"),
                "content": content,
            }
            # Pass through thinking/reasoning content
            if reasoning:
                message["thinking"] = reasoning
            finish_reason = choices[0].get("finish_reason", "stop")

            # Pass through tool_calls, converting arguments string → dict
            if "tool_calls" in msg and msg["tool_calls"]:
                ollama_tool_calls = []
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    args = func.get("arguments", "")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except (json.JSONDecodeError, TypeError):
                            args = {}
                    ollama_tool_calls.append({
                        "function": {
                            "name": func.get("name", ""),
                            "arguments": args,
                        },
                    })
                message["tool_calls"] = ollama_tool_calls

        usage = openai_response.get("usage", {})
        return {
            "model": openai_response.get("model", ""),
            "message": message,
            "done": True,
            "done_reason": finish_reason,
            "total_duration": 0,
            "prompt_eval_count": usage.get("prompt_tokens", 0),
            "eval_count": usage.get("completion_tokens", 0),
        }

    async def _proxy_ollama_stream(
        self,
        request: CanonicalChatRequest,
        backend: Backend,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Proxy streaming request, yield Ollama format chunks."""
        if backend.engine == BackendEngine.OLLAMA:
            payload = OllamaOutTranslator.translate_chat_request(request)
            payload["stream"] = True
            url = f"{backend.url}/api/chat"
        else:
            payload = VLLMOutTranslator.translate_chat_request(request)
            payload["stream"] = True
            url = f"{backend.url}/v1/chat/completions"

        async with self._make_inference_client() as client:
            async with client.stream("POST", url, json=payload) as response:
                if response.status_code >= 400:
                    await response.aread()
                    response.raise_for_status()

                buffer = ""
                async for chunk_bytes in response.aiter_bytes():
                    buffer += chunk_bytes.decode()

                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if not line:
                            continue

                        # Handle SSE format from vLLM
                        if line.startswith("data:"):
                            data_str = line[5:].strip()
                            if data_str == "[DONE]":
                                return
                            try:
                                data = json.loads(data_str)
                                # Convert OpenAI chunk to Ollama format
                                thinking_enabled = request.think if request.think is not None else True
                                ollama_chunk = self._openai_chunk_to_ollama(data, thinking_enabled=thinking_enabled)
                                yield ollama_chunk
                            except json.JSONDecodeError:
                                continue
                        else:
                            # Native Ollama format
                            try:
                                yield json.loads(line)
                            except json.JSONDecodeError:
                                continue

    def _openai_chunk_to_ollama(
        self, openai_chunk: Dict, thinking_enabled: bool = True,
    ) -> Dict:
        """Convert OpenAI streaming chunk to Ollama format."""
        choices = openai_chunk.get("choices", [])
        if not choices:
            return {"done": True}

        delta = choices[0].get("delta", {})
        finish = choices[0].get("finish_reason")

        content = delta.get("content", "")
        reasoning = delta.get("reasoning_content") or delta.get("reasoning")

        # vLLM/Qwen3.5 bug: when thinking is disabled the model may
        # put all output into reasoning_content with content empty.
        # Promote reasoning to content in that case — but only for models
        # whose template honors the disable switch (see
        # reasoning_promotion_applies); an always-reasoning model's
        # reasoning is genuine and must stay separate.
        if (
            not thinking_enabled and not content and reasoning
            and _reasoning_promotion_applies(openai_chunk.get("model", ""))
        ):
            content = reasoning
            reasoning = None

        message: Dict[str, Any] = {
            "role": delta.get("role", "assistant"),
            "content": content,
        }

        # Pass through thinking/reasoning content
        if reasoning:
            message["thinking"] = reasoning

        # Pass through tool_calls, converting arguments string → dict
        if "tool_calls" in delta and delta["tool_calls"]:
            ollama_tool_calls = []
            for tc in delta["tool_calls"]:
                func = tc.get("function", {})
                args = func.get("arguments", "")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                ollama_tool_calls.append({
                    "function": {
                        "name": func.get("name", ""),
                        "arguments": args,
                    },
                })
            message["tool_calls"] = ollama_tool_calls

        return {
            "model": openai_chunk.get("model", ""),
            "message": message,
            "done": finish is not None,
        }

    async def _complete_request(
        self,
        db_request,
        backend_id: int,
        response: Dict,
        job: Job,
        modality: Modality = Modality.CHAT,
    ) -> None:
        """Complete a request with response data."""
        # Inline DLP response gate — NON-STREAMING only: a streamed response is
        # already on the wire and cannot be recalled, so it is left to the async
        # alert path. No-op unless an inline response action is active. Block:
        # the generation already finished, so release the slot as completed (no
        # capacity leak), record the block for audit + alert, and raise a
        # DlpBlockedError (→ 422). Redact: strip the offending values from the
        # returned response AFTER the original is persisted below.
        response_redactions: list = []
        if not getattr(db_request, "is_streaming", False):
            plan = await self._response_inline_plan(response)
            if plan is not None and plan.block is not None:
                blocked = plan.block
                usage = response.get("usage", {}) or {}
                total = (usage.get("prompt_tokens", 0) or 0) + (usage.get("completion_tokens", 0) or 0)
                await self._scheduler.on_job_completed(job, backend_id, total)
                try:
                    await crud.update_request_failed(
                        self.db, db_request.id,
                        error_message=(
                            f"Response blocked by DLP policy ({blocked.severity}): "
                            f"{', '.join(blocked.categories)}"
                        ),
                    )
                    await self.db.commit()
                    from backend.app.services.dlp_worker import enqueue_for_dlp
                    await enqueue_for_dlp(db_request.id)
                except Exception:
                    logger.warning(
                        "dlp_response_block_audit_failed",
                        request_id=getattr(db_request, "id", None),
                    )
                raise blocked
            if plan is not None and plan.redactions:
                response_redactions = plan.redactions

        # Extract token counts
        usage = response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        # Estimate if not provided
        tokens_estimated = prompt_tokens == 0 and completion_tokens == 0
        if tokens_estimated:
            prompt_tokens = job.estimated_prompt_tokens
            completion_tokens = job.estimated_completion_tokens

        # Image generation is billed at a FLAT configured rate, not by the
        # length of the prompt text.
        #
        # Diffusion backends return no `usage`, so the fallback above charged a
        # token count of the prompt string: a 24-character prompt cost 6 tokens,
        # the same as typing that sentence into chat. But an image occupies one
        # of a handful of max_concurrent=1 diffusion workers for seconds of
        # exclusive GPU time — the scarcest capacity on the cluster — so the
        # metering was inverted: the most expensive request to serve was the
        # cheapest to charge for. See img.quota_tokens_per_image.
        if modality == Modality.IMAGE_GENERATION:
            per_image = await self._image_token_cost()
            if per_image > 0:
                # Charge per image actually returned: an n>1 request costs n
                # times as much, because it occupied the worker n times over.
                images = len(response.get("data") or []) or 1
                prompt_tokens = per_image * images
                completion_tokens = 0
                tokens_estimated = True

        total_tokens = prompt_tokens + completion_tokens

        # Release backend capacity FIRST, before DB writes.
        # This prevents the scheduler queue depth counter from getting stuck
        # if subsequent DB operations fail or hang.
        await self._scheduler.on_job_completed(job, backend_id, total_tokens)

        # Increment live cluster token counter in Redis
        from backend.app.core import redis_client
        await redis_client.incr_cluster_tokens(
            prompt_tokens, completion_tokens, total_tokens
        )

        # Mark the model as loaded so the scheduler prefers this backend
        # for subsequent requests (bridges the 30s discovery poll gap).
        try:
            from backend.app.db.session import get_async_db_context
            async with get_async_db_context() as mark_db:
                await crud.mark_model_loaded(mark_db, backend_id, job.model)
                await mark_db.commit()
        except Exception:
            pass  # Best-effort, don't break the completion path

        # DB bookkeeping — shielded from task cancellation so the commit
        # can't be interrupted mid-flush (which corrupts the session and
        # leaks the connection from the pool). Stores the ORIGINAL response.
        await asyncio.shield(self._do_complete_db(
            db_request, backend_id, response, prompt_tokens,
            completion_tokens, tokens_estimated, total_tokens,
        ))

        # DLP response redaction: original is now persisted (and will be scanned
        # by the async worker for alerting), so strip the offending values from
        # the response object the caller receives.
        if response_redactions:
            try:
                self._apply_response_redactions(response, response_redactions)
            except Exception:
                logger.warning("dlp_response_redaction_failed",
                               request_id=getattr(db_request, "id", None))

    # Attempts for the completion DB transaction. Hot-row deadlocks (quotas /
    # api_keys under concurrent same-user traffic, MariaDB error 1213) are
    # transient; more than a few retries means something structural (the
    # durable fix is atomic increments in update_quota_usage/update_api_key_usage).
    _COMPLETION_DB_ATTEMPTS = 5

    async def _run_completion_db(self, request_id: int, write_once, path: str) -> None:
        """Run a completion DB transaction; retry deadlocks; never raise.

        The whole completion write (status flip, response row, quota and key
        usage) is one transaction. A quotas hot-row deadlock aborts it, and
        swallowing that silently loses the response row, the accounting, AND
        the DLP scan while the client already got its 200 — so terminal
        failures are logged loudly and the request is still enqueued for DLP:
        its prompt was committed at creation and can be scanned regardless.

        Takes the scalar request id, not the ORM instance: rollback() expires
        ORM objects, and touching an expired attribute on an async session
        raises MissingGreenlet. write_once must likewise close over scalars
        captured before the first attempt.
        """
        for attempt in range(1, self._COMPLETION_DB_ATTEMPTS + 1):
            try:
                await write_once()
                return
            except asyncio.CancelledError:
                # Shutdown/cancellation: keep the loud-failure + DLP contract,
                # then propagate — callers must still observe cancellation.
                logger.error(
                    "request_completion_db_failed",
                    request_id=request_id,
                    path=path,
                    error="CancelledError",
                    db_error_code=None,
                    attempts=attempt,
                )
                try:
                    from backend.app.services.dlp_worker import enqueue_for_dlp
                    await enqueue_for_dlp(request_id)
                except Exception:
                    pass
                raise
            except Exception as e:
                try:
                    await self.db.rollback()
                except Exception:
                    pass
                code = getattr(getattr(e, "orig", None), "args", (None,))[0]
                # Deadlocks (1213) fail fast and retry cheaply. Lock-wait
                # timeouts (1205) pin a pooled connection for the whole wait —
                # retrying them multiplies pool occupancy, so at most one retry.
                retryable = (code == 1213 and attempt < self._COMPLETION_DB_ATTEMPTS) or (
                    code == 1205 and attempt < 2
                )
                if retryable:
                    await asyncio.sleep(0.05 * (2 ** (attempt - 1)))
                    continue
                logger.error(
                    "request_completion_db_failed",
                    request_id=request_id,
                    path=path,
                    error=type(e).__name__,
                    db_error_code=code,
                    attempts=attempt,
                )
                try:
                    from backend.app.services.dlp_worker import enqueue_for_dlp
                    await enqueue_for_dlp(request_id)
                except Exception:
                    pass
                return

    async def _do_complete_db(
        self, db_request, backend_id, response, prompt_tokens,
        completion_tokens, tokens_estimated, total_tokens,
    ) -> None:
        """DB writes for request completion (run inside asyncio.shield)."""
        # Scalars only past this point: a retry's rollback expires the ORM
        # instance and touching it afterwards raises MissingGreenlet.
        request_id = db_request.id
        user_id = db_request.user_id
        api_key_id = db_request.api_key_id

        async def write_once():
            await crud.update_request_completed(
                self.db, request_id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                tokens_estimated=tokens_estimated,
                backend_id=backend_id,
            )

            content = None
            if "choices" in response and response["choices"]:
                msg = response["choices"][0].get("message", {})
                content = msg.get("content")
            if not (
                self._settings.audit_log_enabled
                and self._settings.audit_log_responses
            ):
                content = None

            await crud.create_response(
                self.db, request_id,
                content=content,
                finish_reason=response.get("choices", [{}])[0].get("finish_reason"),
            )

            await crud.update_quota_usage(self.db, user_id, total_tokens)
            await crud.update_api_key_usage(self.db, api_key_id)

            await self.db.commit()
            # Increment Redis quota only after DB commit succeeds to prevent drift
            await crud.incr_quota_redis(user_id, total_tokens)
            # Enqueue for DLP scanning (best-effort, never affects inference)
            try:
                from backend.app.services.dlp_worker import enqueue_for_dlp
                await enqueue_for_dlp(request_id)
            except Exception:
                pass

        await self._run_completion_db(request_id, write_once, "complete")

    async def _complete_streaming_request(
        self,
        db_request,
        backend_id: int,
        content: str,
        chunk_count: int,
        job: Job,
        finish_reason: Optional[str] = None,
        usage=None,
    ) -> None:
        """Complete a streaming request."""
        if usage is not None:
            # Real token counts harvested from vLLM's include_usage chunk.
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens or (prompt_tokens + completion_tokens)
        else:
            # Fallback: estimate from prompt sizing + generated content length.
            prompt_tokens = job.estimated_prompt_tokens
            completion_tokens = self._scheduler.estimate_tokens(content)
            total_tokens = prompt_tokens + completion_tokens

        # Release backend capacity FIRST
        await self._scheduler.on_job_completed(job, backend_id, total_tokens)

        # Increment live cluster token counter in Redis
        from backend.app.core import redis_client
        await redis_client.incr_cluster_tokens(
            prompt_tokens, completion_tokens, total_tokens
        )

        # Mark the model as loaded so the scheduler prefers this backend
        try:
            from backend.app.db.session import get_async_db_context
            async with get_async_db_context() as mark_db:
                await crud.mark_model_loaded(mark_db, backend_id, job.model)
                await mark_db.commit()
        except Exception:
            pass

        await asyncio.shield(self._do_complete_streaming_db(
            db_request, backend_id, content, chunk_count,
            prompt_tokens, completion_tokens, total_tokens,
            finish_reason=finish_reason or "stop",
        ))

    async def _do_complete_streaming_db(
        self, db_request, backend_id, content, chunk_count,
        prompt_tokens, completion_tokens, total_tokens,
        finish_reason: str = "stop",
    ) -> None:
        """DB writes for streaming completion (run inside asyncio.shield)."""
        # Scalars only past this point: a retry's rollback expires the ORM
        # instance and touching it afterwards raises MissingGreenlet.
        request_id = db_request.id
        user_id = db_request.user_id
        api_key_id = db_request.api_key_id

        async def write_once():
            stored_content = content
            await crud.update_request_completed(
                self.db, request_id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                tokens_estimated=True,
                backend_id=backend_id,
            )

            if not (
                self._settings.audit_log_enabled
                and self._settings.audit_log_responses
            ):
                stored_content = None
            await crud.create_response(
                self.db, request_id,
                content=stored_content,
                chunk_count=chunk_count,
                finish_reason=finish_reason,
            )

            await crud.update_quota_usage(self.db, user_id, total_tokens)
            await crud.update_api_key_usage(self.db, api_key_id)

            await self.db.commit()
            # Increment Redis quota only after DB commit succeeds to prevent drift
            await crud.incr_quota_redis(user_id, total_tokens)
            # Enqueue for DLP scanning (best-effort, never affects inference)
            try:
                from backend.app.services.dlp_worker import enqueue_for_dlp
                await enqueue_for_dlp(request_id)
            except Exception:
                pass

        await self._run_completion_db(request_id, write_once, "complete_streaming")

    async def _fail_request(
        self,
        db_request,
        backend_id: Optional[int],
        error_message: str,
        job: Job,
    ) -> None:
        """Record a failed request.

        Always releases the scheduler slot properly.  If ``backend_id`` is not
        provided explicitly, falls back to ``job.assigned_backend_id`` which is
        set by ``route_job()`` when the slot is acquired.  This prevents slot
        leaks when exceptions occur after routing but before the caller has
        captured the backend reference.
        """
        # Resolve backend_id from job if not provided
        effective_backend_id = backend_id or getattr(job, "assigned_backend_id", None)

        # Release backend capacity FIRST
        if effective_backend_id:
            await self._scheduler.on_job_failed(job, effective_backend_id)
        else:
            # Job was never routed — just remove from queue
            await self._scheduler.cancel_job(job.request_id)

        await asyncio.shield(self._do_fail_db(db_request, error_message))

    async def _do_fail_db(self, db_request, error_message: str) -> None:
        """DB writes for failed request (run inside asyncio.shield)."""
        # Scalars up front: if any earlier rollback expired this ORM instance,
        # touching its attributes mid-transaction raises MissingGreenlet.
        try:
            request_id = db_request.id
            api_key_id = db_request.api_key_id
        except Exception:
            logger.error("request_fail_db_skipped", error="expired_orm_instance")
            return
        try:
            await crud.update_request_failed(
                self.db, request_id,
                error_message=error_message,
            )

            await crud.update_api_key_usage(self.db, api_key_id)

            await self.db.commit()
        except Exception as e:
            try:
                await self.db.rollback()
            except Exception:
                pass
            logger.error(
                "request_fail_db_failed",
                request_id=request_id,
                error=type(e).__name__,
            )
