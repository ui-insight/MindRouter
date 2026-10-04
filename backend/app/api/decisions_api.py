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

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from prometheus_client import Counter, Histogram
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.auth import authenticate_request
from backend.app.api.model_availability import AVAILABLE, model_availability, openai_error
from backend.app.api.voice_api import _check_quota
from backend.app.core.telemetry.registry import get_registry
from backend.app.db import crud
from backend.app.db.models import ApiKey, Modality, User
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


REQUEST_ID_HEADER = "x-typesafe-request-id"  # what TypeSafe's SDK reads as request_id


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
    # (decisions.upstreams, e.g. Laya) or letter scoring on a vLLM chat model
    # (decisions.allowed_models). Jev's aliases mean "this server's default".
    requested = wire.model
    name = cfg["default_model"] if requested is None or requested in JEV_MODEL_ALIASES else requested
    upstream = cfg["upstreams"].get(name)
    plan = None
    if upstream is not None:
        model, backend = name, get_upstream_backend()
        if len(json.dumps(body["questions"], ensure_ascii=False)) > MAX_FORWARDED_QUESTIONS_CHARS:
            raise _invalid(["questions"], f"questions exceed {MAX_FORWARDED_QUESTIONS_CHARS} characters in total")
        if wire.images and not upstream.images:
            # Forwarding to a server that ignores the field would get an answer
            # about the text alone, with nothing to say the image went unseen.
            raise _invalid(["images"], f"model '{name}' does not accept images")
    else:
        registry = get_registry()
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
            raise HTTPException(code, detail=detail["error"]["message"], headers=headers)
        backend = get_decision_backend()

    # Quota + RPM BEFORE any GPU work, like every endpoint that dispatches
    # outside InferenceService (voice, moderations).
    await _check_quota(db, user, api_key)

    dreq = plan.decision_request if plan else None
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
        },
        client_ip=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
    )

    # Commit the audit row BEFORE dialing out. Held open, this transaction
    # would pin a pooled connection for the whole fan-out and keep the
    # requests-row FK lock on api_keys, which the completion writers take
    # exclusively (the 2.9.81 deadlock order).
    await crud.update_request_started(db, db_request.id, backend_id=None)
    await db.commit()

    started = time.perf_counter()
    try:
        if upstream is not None:
            # Forward the caller's own state and questions, untouched.
            answered = await backend.answer(
                upstream, body["state"], body["questions"], concurrency=cfg["backend_concurrency"],
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
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=e.detail) from None
    except DecisionBackendError as e:
        await _record_failure(db, db_request.id, str(e), str(e.status_code))
        DECISION_REQUESTS.labels(model, backend.name, "error").inc()
        logger.warning(
            "decision_request_failed",
            model=model, backend=backend.name, status=e.status_code, error=str(e),
            questions=len(wire.questions), latency_ms=int((time.perf_counter() - started) * 1000),
        )
        raise HTTPException(e.status_code, detail=str(e)) from e
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
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="decision request failed") from None
    latency_ms = int((time.perf_counter() - started) * 1000)

    try:
        payload, token_cost, counts = await _complete(
            db, db_request.id, user.id, plan, outcome, answered,
            model=model, request_id=request_id, backend_name=backend.name, temperature=cfg["temperature"],
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


async def _complete(db, row_id, user_id, plan, outcome, answered, *, model, request_id, backend_name, temperature):
    """Build the response and record the completed request + quota in one
    transaction. Returns (payload, tokens charged, counts for metrics)."""
    if answered is not None:
        # An upstream reports its own token counts; charge what it says it read.
        prompt_tokens, scoring_tokens, cached_tokens = answered.input_tokens, answered.output_tokens, 0
        backend_id, backend_calls, incomplete = None, 1, 0
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
