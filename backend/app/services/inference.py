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
from fastapi import HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.redis_client import incr_inflight_tokens, decr_inflight_tokens
from backend.app.core.canonical_schemas import (
    CanonicalChatRequest,
    CanonicalChatResponse,
    CanonicalChoice,
    CanonicalEmbeddingRequest,
    CanonicalEmbeddingResponse,
    CanonicalMessage,
    CanonicalRerankRequest,
    CanonicalScoreRequest,
    CanonicalStreamChunk,
    CanonicalStreamChoice,
    CanonicalStreamDelta,
    MessageRole,
    UsageInfo,
)
from backend.app.core.scheduler.policy import get_scheduler
from backend.app.core.scheduler.queue import Job, JobModality
from backend.app.core.telemetry.registry import get_registry
from backend.app.core.translators import OllamaOutTranslator, VLLMOutTranslator
from backend.app.db import crud
from backend.app.db.models import ApiKey, Backend, BackendEngine, Modality, User
from backend.app.logging_config import get_logger
from backend.app.settings import get_settings

logger = get_logger(__name__)


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

    async def chat_completion(
        self,
        request: CanonicalChatRequest,
        user: User,
        api_key: ApiKey,
        http_request: Request,
    ) -> Dict[str, Any]:
        """
        Handle non-streaming chat completion.

        Args:
            request: Canonical chat request
            user: Authenticated user
            api_key: API key used
            http_request: Original HTTP request

        Returns:
            OpenAI-compatible chat completion response
        """
        # Check quota
        await self._check_quota(user, api_key)

        # Create audit record
        db_request = await self._create_request_record(
            request, user, api_key, http_request, "/v1/chat/completions"
        )

        # Propagate request UUID so translators can use it as chunk/response ID
        request.request_id = db_request.request_uuid

        # Create job
        job = self._scheduler.create_job_from_chat_request(
            request, user.id, api_key.id
        )
        job.request_id = db_request.request_uuid

        try:
            # Route and proxy with retry
            response, backend = await self._proxy_with_retry(
                request, job, user, proxy_fn="_proxy_chat_request"
            )

            # Update records
            await self._complete_request(
                db_request, backend.id, response, job
            )

            return response

        except BaseException as e:
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
    ) -> AsyncIterator[bytes]:
        """
        Handle streaming chat completion.

        Yields SSE-formatted chunks.
        """
        await self._check_quota(user, api_key)

        db_request = await self._create_request_record(
            request, user, api_key, http_request, "/v1/chat/completions"
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
        completed = False

        try:
            async for chunk, backend in self._proxy_stream_with_retry(
                request, job, user, proxy_fn="_proxy_stream_request"
            ):
                routed_backend = backend

                # Format as SSE (exclude_none to avoid tool_calls:null in chunks)
                yield f"data: {chunk.model_dump_json(exclude_none=True, by_alias=True)}\n\n".encode()

                chunk_count += 1

                # Accumulate content/reasoning and track finish reason
                for choice in chunk.choices:
                    if choice.delta.content:
                        full_content += choice.delta.content
                        inflight_chars += len(choice.delta.content)
                    if choice.delta.reasoning:
                        inflight_chars += len(choice.delta.reasoning)
                    if choice.finish_reason:
                        last_finish_reason = choice.finish_reason

                # Flush estimated tokens to Redis every 10 chunks
                if chunk_count % 10 == 0 and inflight_chars > 0:
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

            # Send done signal
            yield b"data: [DONE]\n\n"

            # Update records
            if routed_backend:
                await self._complete_streaming_request(
                    db_request, routed_backend.id, full_content, chunk_count, job,
                    finish_reason=last_finish_reason,
                )

            completed = True
        except BaseException as e:
            # BaseException catches CancelledError (not a subclass of Exception
            # in Python 3.9+) and GeneratorExit from client disconnects — both
            # of which would otherwise leak the scheduler queue depth counter.
            backend_id = routed_backend.id if routed_backend else None
            try:
                await self._fail_request(db_request, backend_id, str(e), job)
            except Exception:
                pass
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

        try:
            response, backend = await self._proxy_with_retry(
                request, job, user,
                modality=Modality.EMBEDDING,
                proxy_fn="_proxy_embedding_request",
            )

            await self._complete_request(
                db_request, backend.id, response, job,
                modality=Modality.EMBEDDING
            )

            return response

        except BaseException as e:
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

        try:
            response, backend = await self._proxy_with_retry(
                request, job, user,
                modality=Modality.RERANKING,
                proxy_fn="_proxy_rerank_request",
            )

            await self._complete_request(
                db_request, backend.id, response, job,
                modality=Modality.RERANKING
            )

            return response

        except BaseException as e:
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

        try:
            response, backend = await self._proxy_with_retry(
                request, job, user,
                modality=Modality.RERANKING,
                proxy_fn="_proxy_score_request",
            )

            await self._complete_request(
                db_request, backend.id, response, job,
                modality=Modality.RERANKING
            )

            return response

        except BaseException as e:
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

        try:
            response, backend = await self._proxy_with_retry(
                request, job, user, proxy_fn="_proxy_ollama_chat"
            )

            await self._complete_request(
                db_request, backend.id, response, job
            )

            return response

        except BaseException as e:
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

        try:
            async for chunk_data, backend in self._proxy_stream_with_retry(
                request, job, user, proxy_fn="_proxy_ollama_stream"
            ):
                routed_backend = backend
                yield (json.dumps(chunk_data) + "\n").encode()
                chunk_count += 1

                if "message" in chunk_data:
                    content = chunk_data["message"].get("content", "")
                    full_content += content
                    inflight_chars += len(content)
                    thinking = chunk_data["message"].get("thinking", "")
                    if thinking:
                        inflight_chars += len(thinking)

                # Flush estimated tokens to Redis every 10 chunks
                if chunk_count % 10 == 0 and inflight_chars > 0:
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

            if routed_backend:
                await self._complete_streaming_request(
                    db_request, routed_backend.id, full_content, chunk_count, job
                )
        except BaseException as e:
            # BaseException catches CancelledError and GeneratorExit from
            # client disconnects that would otherwise leak queue depth.
            backend_id = routed_backend.id if routed_backend else None
            try:
                await self._fail_request(db_request, backend_id, str(e), job)
            except Exception:
                pass
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

    async def _create_request_record(
        self,
        request: Any,
        user: User,
        api_key: ApiKey,
        http_request: Request,
        endpoint: str,
        modality: Modality = Modality.CHAT,
    ):
        """Create audit record for the request.

        Commits the pre-inference transaction immediately so no DB locks
        are held during the long-running inference phase.
        """
        messages = None
        prompt = None
        parameters = {}

        if hasattr(request, "messages"):
            messages = [m.model_dump() for m in request.messages]
        if hasattr(request, "prompt"):
            prompt = request.prompt if isinstance(request.prompt, str) else str(request.prompt)

        # Extract parameters
        for param in ["temperature", "top_p", "max_tokens", "stop"]:
            if hasattr(request, param) and getattr(request, param) is not None:
                parameters[param] = getattr(request, param)

        response_format = None
        if hasattr(request, "response_format") and request.response_format:
            response_format = request.response_format.model_dump()

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

        return db_request

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
        max_attempts = self._settings.backend_retry_max_attempts
        tried_backends: Set[int] = set()
        last_error: Optional[Exception] = None

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

            # Cap max_tokens so it doesn't exceed model context_length
            if models and request.max_tokens:
                for m in models:
                    if m.context_length and request.max_tokens >= m.context_length:
                        # Reserve at least half the context for input
                        request.max_tokens = m.context_length // 2
                    break

            # Inject num_ctx for Ollama backends from model config
            if backend.engine == BackendEngine.OLLAMA and models:
                for m in models:
                    if m.context_length:
                        if request.backend_options is None:
                            request.backend_options = {}
                        # Check if admin wants to enforce context length
                        from backend.app.db.session import get_async_db_context
                        async with get_async_db_context() as cfg_db:
                            enforce = await crud.get_config_json(cfg_db, "ollama.enforce_num_ctx", True)
                        if enforce:
                            request.backend_options["num_ctx"] = m.context_length
                        else:
                            request.backend_options.setdefault("num_ctx", m.context_length)
                        break

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
                return response, backend

            except (asyncio.TimeoutError, httpx.TimeoutException) as e:
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
                    # 4xx = client error — don't retry, but convert to HTTPException
                    try:
                        detail = e.response.json()
                    except Exception:
                        detail = e.response.text or str(e)
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
        max_attempts = self._settings.backend_retry_max_attempts
        tried_backends: Set[int] = set()
        last_error: Optional[Exception] = None

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

            # Cap max_tokens so it doesn't exceed model context_length
            if _models and hasattr(request, 'max_tokens') and request.max_tokens:
                for m in _models:
                    if m.context_length and request.max_tokens >= m.context_length:
                        request.max_tokens = m.context_length // 2
                    break

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
                    # 4xx = client error — convert to HTTPException
                    try:
                        detail = e.response.json()
                    except Exception:
                        detail = e.response.text or str(e)
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
        client = await self._get_http_client()

        if backend.engine == BackendEngine.OLLAMA:
            payload = OllamaOutTranslator.translate_chat_request(request)
            url = f"{backend.url}/api/chat"
        else:
            payload = VLLMOutTranslator.translate_chat_request(request)
            url = f"{backend.url}/v1/chat/completions"

        payload["stream"] = False

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

        return result

    async def _proxy_stream_request(
        self,
        request: CanonicalChatRequest,
        backend: Backend,
    ) -> AsyncIterator[CanonicalStreamChunk]:
        """Proxy streaming request to backend."""
        client = await self._get_http_client()

        if backend.engine == BackendEngine.OLLAMA:
            payload = OllamaOutTranslator.translate_chat_request(request)
            url = f"{backend.url}/api/chat"
        else:
            payload = VLLMOutTranslator.translate_chat_request(request)
            url = f"{backend.url}/v1/chat/completions"

        payload["stream"] = True

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
        client = await self._get_http_client()

        if backend.engine == BackendEngine.OLLAMA:
            payload = OllamaOutTranslator.translate_embedding_request(request)
            url = f"{backend.url}/api/embeddings"
        else:
            payload = VLLMOutTranslator.translate_embedding_request(request)
            url = f"{backend.url}/v1/embeddings"

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

        client = await self._get_http_client()
        payload = VLLMOutTranslator.translate_rerank_request(request)
        url = f"{backend.url}/v1/rerank"

        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

        canonical = VLLMOutTranslator.translate_rerank_response(data)
        return canonical.model_dump(exclude_none=True, by_alias=True)

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

        client = await self._get_http_client()
        payload = VLLMOutTranslator.translate_score_request(request)
        url = f"{backend.url}/v1/score"

        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

        canonical = VLLMOutTranslator.translate_score_response(data)
        return canonical.model_dump(exclude_none=True, by_alias=True)

    async def _proxy_ollama_chat(
        self,
        request: CanonicalChatRequest,
        backend: Backend,
    ) -> Dict[str, Any]:
        """Proxy chat request, return in Ollama format."""
        client = await self._get_http_client()

        payload = OllamaOutTranslator.translate_chat_request(request)
        payload["stream"] = False

        if backend.engine == BackendEngine.OLLAMA:
            url = f"{backend.url}/api/chat"
        else:
            # Need to translate through OpenAI and back
            payload = VLLMOutTranslator.translate_chat_request(request)
            url = f"{backend.url}/v1/chat/completions"

        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

        # Convert OpenAI format to Ollama format when backend is not Ollama
        if backend.engine != BackendEngine.OLLAMA:
            thinking_enabled = request.think if request.think is not None else True
            data = self._openai_response_to_ollama(data, thinking_enabled=thinking_enabled)

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
            if not thinking_enabled and not content and reasoning:
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
        client = await self._get_http_client()

        payload = OllamaOutTranslator.translate_chat_request(request)
        payload["stream"] = True

        if backend.engine == BackendEngine.OLLAMA:
            url = f"{backend.url}/api/chat"
        else:
            url = f"{backend.url}/v1/chat/completions"

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
        # Promote reasoning to content in that case.
        if not thinking_enabled and not content and reasoning:
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
        # Extract token counts
        usage = response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        # Estimate if not provided
        tokens_estimated = prompt_tokens == 0 and completion_tokens == 0
        if tokens_estimated:
            prompt_tokens = job.estimated_prompt_tokens
            completion_tokens = job.estimated_completion_tokens

        total_tokens = prompt_tokens + completion_tokens

        # Release backend capacity FIRST, before DB writes.
        # This prevents the scheduler queue depth counter from getting stuck
        # if subsequent DB operations fail or hang.
        await self._scheduler.on_job_completed(job, backend_id, total_tokens)

        # DB bookkeeping — shielded from task cancellation so the commit
        # can't be interrupted mid-flush (which corrupts the session and
        # leaks the connection from the pool).
        await asyncio.shield(self._do_complete_db(
            db_request, backend_id, response, prompt_tokens,
            completion_tokens, tokens_estimated, total_tokens,
        ))

    async def _do_complete_db(
        self, db_request, backend_id, response, prompt_tokens,
        completion_tokens, tokens_estimated, total_tokens,
    ) -> None:
        """DB writes for request completion (run inside asyncio.shield)."""
        try:
            await crud.update_request_completed(
                self.db, db_request.id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                tokens_estimated=tokens_estimated,
            )

            content = None
            if "choices" in response and response["choices"]:
                msg = response["choices"][0].get("message", {})
                content = msg.get("content")

            await crud.create_response(
                self.db, db_request.id,
                content=content,
                finish_reason=response.get("choices", [{}])[0].get("finish_reason"),
            )

            await crud.create_usage_entry(
                self.db,
                user_id=db_request.user_id,
                api_key_id=db_request.api_key_id,
                request_id=db_request.id,
                model=db_request.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                is_estimated=tokens_estimated,
                backend_id=backend_id,
            )

            await crud.update_quota_usage(self.db, db_request.user_id, total_tokens)
            await crud.update_api_key_usage(self.db, db_request.api_key_id)

            await self.db.commit()
            # Increment Redis quota only after DB commit succeeds to prevent drift
            await crud.incr_quota_redis(db_request.user_id, total_tokens)
        except Exception:
            try:
                await self.db.rollback()
            except Exception:
                pass

    async def _complete_streaming_request(
        self,
        db_request,
        backend_id: int,
        content: str,
        chunk_count: int,
        job: Job,
        finish_reason: Optional[str] = None,
    ) -> None:
        """Complete a streaming request."""
        prompt_tokens = job.estimated_prompt_tokens
        completion_tokens = self._scheduler.estimate_tokens(content)
        total_tokens = prompt_tokens + completion_tokens

        # Release backend capacity FIRST
        await self._scheduler.on_job_completed(job, backend_id, total_tokens)

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
        try:
            await crud.update_request_completed(
                self.db, db_request.id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                tokens_estimated=True,
            )

            await crud.create_response(
                self.db, db_request.id,
                content=content,
                chunk_count=chunk_count,
                finish_reason=finish_reason,
            )

            await crud.create_usage_entry(
                self.db,
                user_id=db_request.user_id,
                api_key_id=db_request.api_key_id,
                request_id=db_request.id,
                model=db_request.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                is_estimated=True,
                backend_id=backend_id,
            )

            await crud.update_quota_usage(self.db, db_request.user_id, total_tokens)
            await crud.update_api_key_usage(self.db, db_request.api_key_id)

            await self.db.commit()
            # Increment Redis quota only after DB commit succeeds to prevent drift
            await crud.incr_quota_redis(db_request.user_id, total_tokens)
        except Exception:
            try:
                await self.db.rollback()
            except Exception:
                pass

    async def _fail_request(
        self,
        db_request,
        backend_id: Optional[int],
        error_message: str,
        job: Job,
    ) -> None:
        """Record a failed request."""
        # Release backend capacity FIRST
        if backend_id:
            await self._scheduler.on_job_failed(job, backend_id)
        else:
            # Even without a backend_id, the job may have been submitted to the
            # queue (submit_job) or routed (route_job increments queue depth).
            # Cancel it from the queue to prevent phantom entries.
            await self._scheduler.cancel_job(job.request_id)

        await asyncio.shield(self._do_fail_db(db_request, error_message))

    async def _do_fail_db(self, db_request, error_message: str) -> None:
        """DB writes for failed request (run inside asyncio.shield)."""
        try:
            await crud.update_request_failed(
                self.db, db_request.id,
                error_message=error_message,
            )

            await crud.update_api_key_usage(self.db, db_request.api_key_id)

            await self.db.commit()
        except Exception:
            try:
                await self.db.rollback()
            except Exception:
                pass
