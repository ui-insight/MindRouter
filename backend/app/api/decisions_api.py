############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# api/decisions_api.py: POST /v1/decisions — typed decisions
# ("System One") over existing vLLM backends. EXPERIMENTAL.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""POST /v1/systemone (alias /v1/decisions) — EXPERIMENTAL, transitional.

TypeSafe's System One API (Jev's wire format): one shared ``state`` plus a map
of typed questions (``noul`` / ``choice`` / ``score``) in; one typed answer per
question out, with probabilities and confidence. No text is generated. A Jev
client — including TypeSafe's SDK with ``TYPESAFE_BASE_URL`` set to MindRouter
— works unchanged with a MindRouter API key. The numbers come from the vLLM
model MindRouter routes to, not from Jev (see services/decisions/systemone.py).

The route owns everything MindRouter-specific — auth, the enable switch, the
model allow-list, quota + RPM, the audited ``requests`` row, metrics and logs.
How answers are obtained lives behind ``services.decisions.DecisionBackend``.
The state is never logged or stored.

See docs/decisions-api.md.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from prometheus_client import Counter, Histogram
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.auth import authenticate_request
from backend.app.api.model_availability import AVAILABLE, model_availability, openai_error
from backend.app.api.voice_api import _check_quota
from backend.app.core.telemetry.registry import get_registry
from backend.app.db import crud
from backend.app.db.models import ApiKey, BackendEngine, Modality, User
from backend.app.db.session import get_async_db
from backend.app.logging_config import bind_request_context, get_logger
from backend.app.services.decisions import (
    DecisionBackendError,
    DecisionOutcome,
    get_decision_backend,
    get_decisions_config,
    get_upstream_backend,
)
from backend.app.services.decisions.systemone import (
    JEV_MODEL_ALIASES,
    MAX_FORWARDED_QUESTIONS_CHARS,
    SystemOneValidationError,
    compile_plan,
    format_response,
    render,
    validate_wire,
)
from backend.app.services.decisions.upstream import SCORE_SEMANTICS_UPSTREAM

logger = get_logger(__name__)

router = APIRouter(tags=["decisions"])

DECISION_REQUESTS = Counter(
    "mindrouter_decisions_requests_total",
    "Decision requests by model, backend implementation and outcome",
    ["model", "backend", "status"],
)
DECISION_QUESTIONS = Counter(
    "mindrouter_decisions_questions_total",
    "Questions answered by model and question type",
    ["model", "type"],
)
DECISION_LATENCY = Histogram(
    "mindrouter_decisions_latency_seconds",
    "End-to-end latency of a decision request (all questions)",
    ["model"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
)
DECISION_TOKENS = Counter(
    "mindrouter_decisions_tokens_total",
    "Prompt, scoring and prefix-cached tokens consumed by decision requests",
    ["model", "type"],
)


DECISION_FALLBACKS = Counter(
    "mindrouter_decisions_fallbacks_total",
    "Decision requests answered by the configured fallback instead of the requested model",
    ["requested", "answered_by"],
)

REQUEST_ID_HEADER = "x-typesafe-request-id"  # what TypeSafe's SDK reads as request_id

# A model that answers one of these is "not working" for this request: the
# configured fallback (decisions.fallbacks) takes over. 4xx are the caller's.
_FALLBACK_STATUSES = frozenset({500, 502, 503, 504})
# Of those, the ones that count against a monitored server's circuit breaker.
# 503 is "busy", which is load, not sickness.
_SICK_STATUSES = frozenset({500, 502, 504})


@dataclass
class _Target:
    """One model that could answer the request, checked and ready to call."""

    name: str                      # the name it goes by in the settings
    model: str                     # the name recorded and returned
    backend: Any                   # the implementation that calls it
    upstream: Any = None           # set for a System One server
    plan: Any = None               # set for letter scoring on a vLLM model
    monitor_id: Optional[int] = None   # its registered backend, when monitored


class _Unavailable(Exception):
    """The model cannot take a request right now (known before any work)."""

    def __init__(self, status_code: int, detail: str, headers: Optional[dict] = None, reason: str = "unavailable"):
        super().__init__(detail)
        self.status_code, self.detail, self.headers, self.reason = status_code, detail, headers, reason


class _AttemptFailed(Exception):
    """A model was tried and failed; ``error`` is the HTTP error to send."""

    def __init__(self, error: HTTPException, retryable: bool = False, reason: str = "failed"):
        super().__init__(reason)
        self.error, self.retryable, self.reason = error, retryable, reason


async def _report(report, backend_id: int) -> None:
    """Tell the registry how a live request to a monitored server went. Never
    lets bookkeeping fail the request."""
    try:
        await report(backend_id)
    except Exception:
        logger.warning("decision_circuit_report_failed", backend_id=backend_id)


async def _prepare(name: str, *, wire, body, cfg, requested, registry, db) -> _Target:
    """Check that ``name`` can answer this request and return how to call it.

    Raises HTTPException(422) when the request itself is the problem for this
    model, and _Unavailable when the model is down.
    """
    upstream = cfg["upstreams"].get(name)
    if upstream is not None:
        if len(json.dumps(body["questions"], ensure_ascii=False)) > MAX_FORWARDED_QUESTIONS_CHARS:
            raise _invalid(["questions"], f"questions exceed {MAX_FORWARDED_QUESTIONS_CHARS} characters in total")
        if wire.images and not upstream.images:
            # Forwarding to a server that ignores the field would get an answer
            # about the text alone, with nothing to say the image went unseen.
            raise _invalid(["images"], f"model '{name}' does not accept images")
        # A decision server registered as a backend (engine "decision") is
        # health-polled; when it is known to be down, do not dial it.
        try:
            monitor_id, problem = await registry.decision_server_state(upstream.url, db)
        except Exception as e:
            # Monitoring is an optimisation. If the lookup itself fails, dial
            # the server as if it were not registered rather than fail the request.
            logger.warning("decision_server_lookup_failed", model=name, error_type=type(e).__name__)
            monitor_id, problem = None, None
        if problem:
            raise _Unavailable(
                status.HTTP_503_SERVICE_UNAVAILABLE, f"model '{name}' is unavailable ({problem})",
                {"Retry-After": "15"}, reason=problem,
            )
        return _Target(name=name, model=name, backend=get_upstream_backend(), upstream=upstream, monitor_id=monitor_id)

    model, _alias = registry.resolve_alias(name)
    # Resolve the admin's list too: an alias typed into decisions.allowed_models
    # must admit the model it points at, not reject every request.
    allowed = {registry.resolve_alias(m)[0] for m in cfg["allowed_models"]}
    if model not in allowed:
        offered = [*JEV_MODEL_ALIASES, *cfg["allowed_models"], *cfg["upstreams"]]
        raise _invalid(
            ["model"],
            f"model '{str(requested)[:100]}' is not available for decisions here; use one of: {', '.join(offered)}",
        )
    try:
        plan = compile_plan(wire, cfg["permutations"])
    except SystemOneValidationError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=e.detail) from None
    availability = await model_availability(registry, model)
    if availability != AVAILABLE:
        code, detail, headers = openai_error(model, availability)
        raise _Unavailable(code, detail["error"]["message"], headers, reason="no healthy replica")
    if wire.images and await registry.pick_available_backend(
        model, engine=BackendEngine.VLLM, multimodal=True
    ) is None:
        # Known before any work, like the upstream check above. As a fallback
        # target this means "cannot take this request", not an error about a
        # model the caller never named.
        raise _invalid(["images"], f"model '{name}' does not accept images")
    return _Target(name=name, model=model, backend=get_decision_backend(), plan=plan)


async def _prepare_fallback(name: str, **ctx) -> Optional[_Target]:
    """The configured alternative for ``name``, ready to call, or None when
    there is none or it cannot take this request either (it is down too, or
    the request does not fit it: images to a model that cannot see, more
    options than letter scoring allows)."""
    fallbacks = ctx["cfg"].get("fallbacks", {})
    # The setting may name the model by its catalog name while the caller used an alias.
    alternative = fallbacks.get(name) or fallbacks.get(ctx["registry"].resolve_alias(name)[0])
    if not alternative:
        return None
    try:
        return await _prepare(alternative, **ctx)
    except (HTTPException, _Unavailable):
        return None
    except Exception as e:
        # A fallback is best effort: whatever goes wrong preparing it, the
        # caller gets the original model's error, not this one.
        logger.warning("decision_fallback_unusable", model=alternative, error_type=type(e).__name__)
        return None


def _invalid(loc: list, msg: str, kind: str = "value_error") -> HTTPException:
    """A 422 in the FastAPI shape TypeSafe's API (and SDK) use."""
    return HTTPException(
        status.HTTP_422_UNPROCESSABLE_ENTITY, detail=[{"loc": ["body", *loc], "msg": msg, "type": kind}]
    )


@router.post("/v1/systemone")
@router.post("/v1/decisions")
async def systemone(
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_async_db),
    auth: tuple[User, ApiKey] = Depends(authenticate_request),
):
    user, api_key = auth
    endpoint = request.url.path

    try:
        body = await request.json()
    except Exception:
        raise _invalid([], "Request body is not valid JSON", "json_invalid") from None

    request_id = f"dec-{uuid.uuid4().hex[:24]}"
    bind_request_context(request_id=request_id, user_id=user.id)

    cfg = await get_decisions_config(db)
    if not cfg["enabled"]:
        # 404, not 503: TypeSafe's SDK retries 5xx, and retrying cannot help.
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="The decisions API is not enabled on this server")
    try:
        wire = validate_wire(body)
    except SystemOneValidationError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=e.detail) from None
    state_text = render(wire.state)
    if len(state_text) > cfg["max_state_chars"]:
        raise _invalid(["state"], f"state exceeds {cfg['max_state_chars']} characters on this server")

    # ``model`` picks how the answer is produced: an upstream decision server
    # (decisions.upstreams, e.g. Clef) or letter scoring on a vLLM chat model
    # (decisions.allowed_models). Jev's aliases mean "this server's default".
    requested = wire.model
    name = cfg["default_model"] if requested is None or requested in JEV_MODEL_ALIASES else requested
    registry = get_registry()
    ctx = {"wire": wire, "body": body, "cfg": cfg, "requested": requested, "registry": registry, "db": db}
    request_started = time.perf_counter()

    # ``fallback`` is set once a configured alternative (decisions.fallbacks)
    # is answering in place of the model that was asked for.
    fallback: dict | None = None
    try:
        target = await _prepare(name, **ctx)
    except _Unavailable as unavailable:
        # Known to be down before any work is done: go straight to the alternative.
        target = await _prepare_fallback(name, **ctx)
        if target is None:
            raise HTTPException(unavailable.status_code, detail=unavailable.detail, headers=unavailable.headers) from None
        fallback = {"requested": name, "reason": unavailable.reason}

    # Quota + RPM BEFORE any GPU work, like every endpoint that dispatches
    # outside InferenceService (voice, moderations).
    await _check_quota(db, user, api_key)

    async def attempt(target: _Target, fallback: dict | None):
        """One model's try at the request: its audit row, the call, and the
        failure bookkeeping. Returns (row, answered, outcome, started) or
        raises _AttemptFailed with the HTTP error to send."""
        model, backend = target.model, target.backend
        dreq = target.plan.decision_request if target.plan else None
        db_request = await crud.create_request(
            db=db,
            user_id=user.id,
            api_key_id=api_key.id,
            endpoint=endpoint,
            model=model,
            modality=Modality.CHAT,
            # Shape only — never the state or the question text.
            parameters={
                "backend": backend.name,
                "questions": len(wire.questions),
                # A number when the caller named it; otherwise the server's per-type defaults applied.
                "permutations": wire.permutations if wire.permutations is not None else "default",
                "types": sorted({q.type for q in wire.questions.values()}),
                "state_chars": len(state_text),
                "images": len(wire.images or []),
                "model_requested": requested,
                **({"fallback_from": fallback["requested"], "fallback_reason": fallback["reason"]} if fallback else {}),
            },
            client_ip=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
        )

        # Commit the audit row BEFORE dialing out. Held open, this transaction
        # would pin a pooled connection for the whole fan-out and keep the
        # requests-row FK lock on api_keys, which the completion writers take
        # exclusively (the 2.9.81 deadlock order).
        await crud.update_request_started(db, db_request.id, backend_id=target.monitor_id)
        await db.commit()

        started = time.perf_counter()
        try:
            if target.upstream is not None:
                # Forward the caller's own state and questions, untouched.
                answered = await backend.answer(
                    target.upstream, body["state"], body["questions"], concurrency=cfg["backend_concurrency"],
                    images=wire.images or None,
                )
                outcome = None
            elif dreq is None:
                # Every question had a single option: answered without the model.
                answered, outcome = None, DecisionOutcome(results=[], usage=None, backend_id=None, backend_name=None)
            else:
                answered = None
                outcome = await backend.decide(
                    dreq, model, fanout=cfg["fanout"], backend_concurrency=cfg["backend_concurrency"],
                )
        except SystemOneValidationError as e:
            # The upstream refused the request's content. Its wording may quote
            # the request, so the audit row gets the fixed summary instead.
            await _record_failure(db, db_request.id, e.audit_message, "422")
            DECISION_REQUESTS.labels(model, backend.name, "error").inc()
            raise _AttemptFailed(HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=e.detail)) from None
        except DecisionBackendError as e:
            await _record_failure(db, db_request.id, str(e), str(e.status_code))
            DECISION_REQUESTS.labels(model, backend.name, "error").inc()
            logger.warning(
                "decision_request_failed",
                model=model, backend=backend.name, status=e.status_code, error=str(e),
                questions=len(wire.questions), latency_ms=int((time.perf_counter() - started) * 1000),
            )
            if target.monitor_id is not None and e.status_code in _SICK_STATUSES:
                # A monitored decision server that fails live requests is marked
                # down by its circuit breaker, so the next callers skip it at once.
                await _report(registry.report_live_failure, target.monitor_id)
            raise _AttemptFailed(
                HTTPException(e.status_code, detail=str(e)),
                retryable=e.status_code in _FALLBACK_STATUSES, reason=str(e)[:160],
            ) from e
        except asyncio.CancelledError:
            # The client went away or the server is shutting down. Nothing sweeps
            # a row left in PROCESSING, so close it before the cancellation continues.
            await asyncio.shield(_record_failure(db, db_request.id, "cancelled before the model answered", "499"))
            DECISION_REQUESTS.labels(model, backend.name, "error").inc()
            raise
        except Exception as e:
            # Anything the backend did not map: keep the audit row and answer 500
            # instead of letting the request vanish with an unhandled error. Only
            # the exception's type is recorded; its text could quote the request.
            logger.error("decision_request_crashed", model=model, backend=backend.name, error_type=type(e).__name__)
            await _record_failure(db, db_request.id, f"internal error: {type(e).__name__}", "500")
            DECISION_REQUESTS.labels(model, backend.name, "error").inc()
            raise _AttemptFailed(
                HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="decision request failed")) from None
        if target.monitor_id is not None:
            await _report(registry.report_live_success, target.monitor_id)
        return db_request, answered, outcome, started

    try:
        db_request, answered, outcome, started = await attempt(target, fallback)
    except _AttemptFailed as failed:
        # The model failed this request. If an alternative is configured and
        # can take the request, it answers; its row is separate from the
        # failed one, so each model's record stays true.
        alternative = await _prepare_fallback(name, **ctx) if failed.retryable and fallback is None else None
        if alternative is None:
            raise failed.error from None
        fallback = {"requested": name, "reason": failed.reason}
        target = alternative
        try:
            db_request, answered, outcome, started = await attempt(target, fallback)
        except _AttemptFailed as failed_again:
            raise failed_again.error from None
    model, backend, plan = target.model, target.backend, target.plan
    # What the caller waited, including a failed attempt before a fallback.
    started = request_started
    if fallback:
        DECISION_FALLBACKS.labels(fallback["requested"], model).inc()
        logger.warning("decision_fallback", requested=fallback["requested"], answered_by=model, reason=fallback["reason"])
    latency_ms = int((time.perf_counter() - started) * 1000)

    try:
        payload, token_cost, counts = await _complete(
            db, db_request.id, user.id, plan, outcome, answered,
            model=model, request_id=request_id, backend_name=backend.name, temperature=cfg["temperature"],
            monitor_id=target.monitor_id, fallback=fallback,
        )
    except Exception as e:
        # The model answered but the result could not be formatted or recorded.
        # Close the row as failed rather than leaving it in PROCESSING.
        logger.error("decision_completion_failed", model=model, backend=backend.name, error_type=type(e).__name__)
        await _record_failure(db, db_request.id, f"internal error: {type(e).__name__}", "500")
        DECISION_REQUESTS.labels(model, backend.name, "error").inc()
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="decision request failed") from None
    try:
        await crud.incr_quota_redis(user.id, token_cost)
    except Exception:
        # The durable quota row is already committed; the Redis counter is a cache of it.
        logger.warning("decision_quota_redis_failed", user_id=user.id)

    prompt_tokens, scoring_tokens, cached_tokens = counts["prompt"], counts["scoring"], counts["cached"]
    DECISION_REQUESTS.labels(model, backend.name, "ok").inc()
    DECISION_LATENCY.labels(model).observe(latency_ms / 1000.0)
    DECISION_TOKENS.labels(model, "prompt").inc(prompt_tokens)
    DECISION_TOKENS.labels(model, "scoring").inc(scoring_tokens)
    if cached_tokens:
        DECISION_TOKENS.labels(model, "cached").inc(cached_tokens)
    for q in wire.questions.values():
        DECISION_QUESTIONS.labels(model, q.type).inc()

    logger.info(
        "decision_request",
        endpoint=endpoint, model=model, model_requested=requested, backend=backend.name,
        backend_id=counts["backend_id"], questions=len(wire.questions), backend_calls=counts["backend_calls"],
        prompt_tokens=prompt_tokens, scoring_tokens=scoring_tokens, cached_tokens=cached_tokens,
        charged_tokens=token_cost, latency_ms=latency_ms, incomplete=counts["incomplete"],
    )

    response.headers[REQUEST_ID_HEADER] = request_id
    return payload


async def _complete(db, row_id, user_id, plan, outcome, answered, *, model, request_id, backend_name, temperature,
                    monitor_id=None, fallback=None):
    """Build the response and record the completed request + quota in one
    transaction. Returns (payload, tokens charged, counts for metrics)."""
    if answered is not None:
        # An upstream reports its own token counts; charge what it says it read.
        prompt_tokens, scoring_tokens, cached_tokens = answered.input_tokens, answered.output_tokens, 0
        # The registered backend for this decision server, when it is monitored.
        backend_id, backend_calls, incomplete = monitor_id, 1, 0
        payload = {
            "model": model,
            "answers": answered.answers,
            "usage": {"input_tokens": prompt_tokens, "output_tokens": scoring_tokens},
            "id": request_id,
            "metadata": {
                "score_semantics": SCORE_SEMANTICS_UPSTREAM,
                "backend": backend_name,
                "upstream_model": answered.upstream_model,
                **answered.extras,
            },
        }
    else:
        # The audit row records what the server processed; quota charges what it
        # had to compute. Every view repeats the state, and after the first the
        # server serves it from its prefix cache, so cached prompt tokens are not
        # charged again (a 5-question request would otherwise pay for the state 5x).
        usage = outcome.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        scoring_tokens = usage.scoring_tokens if usage else 0
        cached_tokens = (usage.cached_tokens or 0) if usage else 0
        backend_id, backend_calls = outcome.backend_id, (usage.backend_calls if usage else 0)
        incomplete = sum(1 for r in outcome.results if not r.complete)
        payload = format_response(
            plan, outcome.results, usage, model=model, request_id=request_id, backend_name=backend_name,
            temperature=temperature,
        )
    token_cost = max(0, prompt_tokens - cached_tokens) + scoring_tokens
    if fallback:
        # Tell the caller another model answered, and why. ``model`` above is
        # already the one that did.
        payload.setdefault("metadata", {})["fallback"] = dict(fallback)

    await crud.update_request_completed(
        db, row_id,
        prompt_tokens=prompt_tokens,
        completion_tokens=scoring_tokens,
        tokens_estimated=False,
        backend_id=backend_id,
    )
    await crud.update_quota_usage(db, user_id, token_cost)
    await db.commit()
    counts = {"prompt": prompt_tokens, "scoring": scoring_tokens, "cached": cached_tokens,
              "backend_id": backend_id, "backend_calls": backend_calls, "incomplete": incomplete}
    return payload, token_cost, counts


async def _record_failure(db: AsyncSession, request_id: int, message: str, code: str) -> None:
    """Mark the audit row failed. Never masks the error being reported."""
    try:
        await db.rollback()
        await crud.update_request_failed(db, request_id, message, error_code=code)
        await db.commit()
    except Exception:
        logger.exception("decision_failure_record_failed", request_id=request_id)
