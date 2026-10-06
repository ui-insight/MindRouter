############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# services/decisions/vllm_logprobs.py: decision backend that scores
# option labels over vLLM's OpenAI HTTP API (EXPERIMENTAL, transitional)
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Score typed questions on an existing MindRouter-managed vLLM server.

How a score is obtained
-----------------------
For every (question, option order) we POST one ``/v1/chat/completions`` with

* one user turn ``INSTRUCTION + Context + Question + "A. ..." lines``,
  rendered by the server's own chat template with ``enable_thinking=false``
  (Qwen3.x then ends the prompt with ``<think>\\n\\n</think>\\n\\n``);
* ``max_tokens=1, temperature=0`` — one forward pass, no text generated;
* ``allowed_token_ids=[A, B, ...]`` — the sampler may only pick a label, so
  the returned token IS the exact argmax over the options;
* ``logprobs=true, top_logprobs=20`` — the raw log-probability of the 20
  likeliest next tokens. The labels are read out of that list. The prompt
  asks for a letter, so the labels that matter are in it: measured on
  qwen3.8-27b, the labels left out of the top 20 (it happens from about 8
  options up) together held at most 0.0001 of the probability. A label that
  is not in the list is given a value one nat below the lowest that is (its
  true value cannot be higher) and the decision is flagged
  ``complete=false``. How much that approximation matters depends on
  ``label_mass``: near 1 it is negligible. When the model did not want to
  answer with a letter (low ``label_mass``) the floored labels carry real
  weight: the probabilities are rough, and what is computed from them (a
  ``score``'s expected value, a ``choice`` averaged over two option orders)
  can differ from the exact result. The label the sampler picked is still
  the likeliest in its own view.

``logprob_token_ids`` IS NOT SENT, although it returns exactly the labels and
was used until 2.9.90. vLLM does not handle it under speculative decoding
(MTP or a draft model, which every Qwen3.x replica here runs): as soon as any
sequence in the same step carries draft tokens, i.e. whenever the replica is
serving anything else, the reply fails with HTTP 500 (``IndexError`` in
``_create_chat_logprobs``). Found 2026-10-06 on 0.29.0, where System One on
qwen3.8-27b failed for nearly every request while chat traffic was running;
the 0.31.0rc2 sampler has the same gap. Do not bring the field back without
testing under concurrent chat load (``tests/decisions_under_load.py``).

This is exactly what open-alternative-jev's vLLM backend does in its
recommended ``separate`` mode (``label_scores_last``), minus the in-process
``vllm.LLM`` engine: that library's ``allowed_token_ids`` + ``logprobs``
calls map one-to-one onto the HTTP fields above, and vLLM's prefix cache
shares the state across the questions automatically because the state is
the first thing in every prompt.

What is NOT available over HTTP (and why we do not need it)
-----------------------------------------------------------
* ``logprobs_mode=processed_logprobs`` is an engine flag, not a request
  field. We read RAW logprobs and renormalize over the labels ourselves,
  which is the same number the processed path yields after masking. (Raw is
  also why a label can fall outside the top 20: other tokens compete for the
  places, even though the sampler may not pick them.)
* The ``packed`` mode (all questions in one sequence, read via
  ``prompt_logprobs``) is possible over ``/v1/completions`` with token-id
  prompts, but the library itself measures ``separate`` as faster AND exact
  on vLLM, so packed is not implemented.

Label tokens are looked up once per (backend, model) through the server's
``/tokenize`` endpoint and each must be a single token, as in so1.
"""
from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Sequence
from typing import Any

import httpx

from backend.app.core.telemetry.registry import get_registry
from backend.app.db.models import BackendEngine
from backend.app.logging_config import get_logger
from backend.app.settings import get_settings

from . import DecisionBackendError, DecisionOutcome
from .schema import (
    DecisionRequest,
    DecisionResult,
    DecisionUsage,
    Question,
)
from .scoring import LETTERS, LabelReadout, combine, option_orders, render_turn

logger = get_logger(__name__)

# vLLM's default --max-logprobs: the most a server returns, and what is
# always asked for (the labels are read out of this list).
_TOP_LOGPROBS_CAP = 20
_TOKEN_ID_PREFIX = "token_id:"
# Replicas tried for one request: the first, and one other if the first is
# sick. The request-level fallback to another MODEL (decisions.fallbacks)
# comes after this, in the API layer.
_MAX_REPLICAS = 2
# A replica answering one of these is not working for this request; another
# replica may be. 4xx are about the request and would fail anywhere.
_RETRY_STATUSES = frozenset({500, 502, 503, 504})
# Transport failures that mean the replica itself is gone or going: it could
# not be reached, or it dropped the connection (restarted, killed). A read or
# write TIMEOUT is not here: that request already waited its minute on this
# replica and is not sent round again.
_REPLICA_GONE = (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadError, httpx.WriteError,
                 httpx.RemoteProtocolError)


class _ReplicaFailed(DecisionBackendError):
    """One replica could not score the request; another one might."""


def _error_type(response: Any) -> str | None:
    """The error's class as the engine names it (``error.type`` in vLLM's JSON
    error body), or None. A short identifier only: an engine's error MESSAGE
    can quote the prompt and is never logged."""
    try:
        body = response.json()
    except Exception:
        return None          # an unhandled engine exception is a plain-text 500
    if not isinstance(body, dict):
        return None
    error = body.get("error") if isinstance(body.get("error"), dict) else body
    kind = error.get("type")
    if isinstance(kind, str) and 0 < len(kind) <= 64 and kind.replace("_", "").replace(".", "").isalnum():
        return kind
    return None


class VLLMLogprobsBackend:
    name = "vllm_logprobs"

    def __init__(self) -> None:
        # (backend url, model) -> token id of each letter A.. (single tokens)
        self._label_ids: dict[tuple[str, str], list[int]] = {}
        # backend id -> (limit, semaphore). Shared by every request in this
        # process so a burst of decision traffic cannot pile onto one chat
        # replica; rebuilt when the admin changes the limit.
        self._backend_gates: dict[int, tuple[int, asyncio.Semaphore]] = {}

    # ------------------------------------------------------------------ public

    async def decide(
        self, request: DecisionRequest, model: str, *, fanout: int = 8, backend_concurrency: int = 4
    ) -> DecisionOutcome:
        images = list(request.images)

        # One "view" per (question, option order).
        views: list[tuple[int, list[int]]] = []
        for qi, q in enumerate(request.questions):
            for order in option_orders(len(q.options), q.permutations or request.permutations):
                views.append((qi, order))

        # A replica that answers 5xx or cannot be reached is not the whole
        # model: try one other replica before failing the request.
        backend = await self._pick_backend(model, needs_vision=bool(images))
        tried = {backend.id}
        while True:
            try:
                scored = await self._score_views(
                    backend, request, model, views, images, fanout, backend_concurrency)
                break
            except _ReplicaFailed as failure:
                other = None
                if len(tried) < _MAX_REPLICAS:
                    other = await self._another_replica(model, bool(images), tried)
                if other is None:
                    # The first replica's failure is the answer (502), so the
                    # API layer can still hand the request to a fallback model.
                    raise DecisionBackendError(str(failure), failure.status_code) from failure
                logger.warning("decision_backend_retry_on_another_replica", failed_backend_id=backend.id)
                backend = other
                tried.add(backend.id)

        # Regroup per question and fold the views back into the original order.
        per_q: dict[int, list[tuple[list[int], LabelReadout]]] = {}
        prompt_tokens = scoring_tokens = 0
        cached: int | None = None
        for (qi, order), (readout, usage) in zip(views, scored, strict=True):
            per_q.setdefault(qi, []).append((order, readout))
            # The answer is already in hand: a usage block of the wrong shape
            # costs the token counts, not the request.
            usage = usage if isinstance(usage, dict) else {}
            prompt_tokens += _count(usage.get("prompt_tokens")) or 0
            scoring_tokens += _count(usage.get("completion_tokens")) or 0
            details = usage.get("prompt_tokens_details")
            c = _count(details.get("cached_tokens")) if isinstance(details, dict) else None
            if c is not None:
                cached = (cached or 0) + c

        results = [
            _to_result(q, combine(len(q.options), [o for o, _ in per_q[qi]], [r for _, r in per_q[qi]]))
            for qi, q in enumerate(request.questions)
        ]
        usage = DecisionUsage(
            prompt_tokens=prompt_tokens,
            scoring_tokens=scoring_tokens,
            total_tokens=prompt_tokens + scoring_tokens,
            cached_tokens=cached,
            backend_calls=len(views),
        )
        return DecisionOutcome(results=results, usage=usage, backend_id=backend.id, backend_name=backend.name)

    # ---------------------------------------------------------------- internals

    async def _score_views(
        self, backend: Any, request: DecisionRequest, model: str, views: list[tuple[int, list[int]]],
        images: list[str], fanout: int, backend_concurrency: int,
    ) -> list[tuple[LabelReadout, dict]]:
        """Score every view on one replica. Raises _ReplicaFailed when this
        replica is sick (another may answer) and DecisionBackendError when
        the request would fail anywhere."""
        settings = get_settings()
        timeout = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)
        verify = bool(getattr(settings, "internal_tls_verify", True))
        max_n = max(len(q.options) for q in request.questions)
        backend_gate = self._backend_gate(backend.id, backend_concurrency)

        async with httpx.AsyncClient(timeout=timeout, verify=verify) as client:
            try:
                label_ids = await self._get_label_ids(client, backend.url, model, max_n)
                request_gate = asyncio.Semaphore(max(1, fanout))

                async def score(qi: int, order: list[int]) -> tuple[LabelReadout, dict]:
                    q = request.questions[qi]
                    shown = [q.options[k] for k in order]
                    text = render_turn(q.question, shown, request.state)
                    async with request_gate, backend_gate:
                        return await self._score_one(
                            client, backend.url, model, text, label_ids[: len(shown)], images)

                # Every view starts with the same state. Score the first one on
                # its own so it fills vLLM's prefix cache; the rest then hit the
                # cache instead of each prefilling the whole state in parallel.
                if (request.state or images) and len(views) > 1:
                    first = await score(*views[0])
                    rest = await _gather_or_cancel([score(qi, order) for qi, order in views[1:]])
                    return [first, *rest]
                return await _gather_or_cancel([score(qi, order) for qi, order in views])
            except DecisionBackendError:
                raise
            except httpx.HTTPStatusError as e:
                # Status and the engine's own error class only: an engine's
                # error text can quote the prompt.
                status = e.response.status_code
                logger.warning("decision_backend_http_error", backend_id=backend.id, status=status,
                               error_type=_error_type(e.response))
                if images and status == 400:
                    # The gateway reads only an image's header; pixels the model
                    # cannot decode (a cut-off file) are the caller's to fix.
                    raise DecisionBackendError("the model could not read the request's images", 422) from e
                kind = _ReplicaFailed if status in _RETRY_STATUSES else DecisionBackendError
                raise kind(f"decision backend returned HTTP {status}", 502) from e
            except httpx.HTTPError as e:
                # A transport error's text names the cause (refused, DNS, an
                # expired certificate) and never quotes the request.
                logger.warning("decision_backend_unreachable", backend_id=backend.id, error=type(e).__name__,
                               detail=str(e)[:200])
                kind = _ReplicaFailed if isinstance(e, _REPLICA_GONE) else DecisionBackendError
                raise kind("decision backend unreachable", 502) from e
            except (ValueError, KeyError, TypeError, AttributeError) as e:
                # A reply that is not the JSON shape we read (resp.json() raises
                # ValueError on a non-JSON body, e.g. a proxy error page).
                logger.warning("decision_backend_bad_reply", backend_id=backend.id, error=type(e).__name__)
                raise DecisionBackendError("decision backend returned a malformed reply", 502) from e

    async def _another_replica(self, model: str, needs_vision: bool, tried: set[int]):
        """A replica to retry on, or None when there is none. Never raises: a
        failed lookup must not turn the first replica's 502 into a crash."""
        try:
            return await get_registry().pick_available_backend(
                model, engine=BackendEngine.VLLM, multimodal=needs_vision, exclude=tried)
        except Exception as e:
            logger.warning("decision_backend_retry_lookup_failed", error=type(e).__name__)
            return None

    async def _pick_backend(self, model: str, needs_vision: bool = False):
        """A random healthy, circuit-closed vLLM backend serving ``model``;
        with ``needs_vision``, one whose copy of the model takes images.

        Mirrors the direct-to-backend precedents (image_policy, dlp_worker):
        no scheduler slot is taken; ``_backend_gate`` bounds the load instead.
        See docs/decisions-api.md "Limitations".
        """
        registry = get_registry()
        backend = await registry.pick_available_backend(
            model, engine=BackendEngine.VLLM, multimodal=needs_vision)
        if backend is None and needs_vision and await registry.pick_available_backend(model, engine=BackendEngine.VLLM):
            # The model is up; it just cannot see. The caller's request to fix.
            raise DecisionBackendError(f"model '{model}' does not accept images", 422)
        if backend is None:
            raise DecisionBackendError(f"no healthy vLLM backend serves '{model}'", 503)
        return backend

    def _backend_gate(self, backend_id: int, limit: int) -> asyncio.Semaphore:
        limit = max(1, int(limit))
        current = self._backend_gates.get(backend_id)
        if current is None or current[0] != limit:
            # A changed limit gets a fresh semaphore; calls already holding the
            # old one finish under it, so the cap is briefly approximate.
            current = (limit, asyncio.Semaphore(limit))
            self._backend_gates[backend_id] = current
        return current[1]

    async def _get_label_ids(self, client: httpx.AsyncClient, url: str, model: str, n: int) -> list[int]:
        key = (url, model)
        ids = self._label_ids.get(key)
        if ids is not None and len(ids) >= n:
            return ids
        ids = []
        try:
            for letter in LETTERS[:n]:
                resp = await client.post(
                    f"{url}/tokenize",
                    json={"model": model, "prompt": letter, "add_special_tokens": False},
                )
                resp.raise_for_status()
                toks = resp.json().get("tokens") or []
                if len(toks) != 1:
                    raise DecisionBackendError(
                        f"label {letter!r} is not a single token for model '{model}'; "
                        "this model cannot be scored with letter labels",
                        400,
                    )
                ids.append(int(toks[0]))
        except httpx.HTTPError as e:
            # This is the first thing asked of a replica, so a replica that is
            # down fails HERE. Down or answering 5xx: another replica may do.
            status = e.response.status_code if isinstance(e, httpx.HTTPStatusError) else None
            sick = status in _RETRY_STATUSES or isinstance(e, _REPLICA_GONE)
            logger.warning("decision_backend_tokenizer_failed", url_host=httpx.URL(url).host, status=status,
                           error=type(e).__name__, detail=None if status else str(e)[:200])
            kind = _ReplicaFailed if sick else DecisionBackendError
            raise kind("decision backend tokenizer unreachable", 502) from e
        self._label_ids[key] = ids
        return ids

    async def _score_one(
        self, client: httpx.AsyncClient, url: str, model: str, text: str, label_ids: Sequence[int],
        images: Sequence[str] = (),
    ) -> tuple[LabelReadout, dict]:
        ids = list(label_ids)
        # Images go first, before the state, identically in every view, so they
        # are part of the prefix the server caches across the questions.
        content: Any = text if not images else [
            *({"type": "image_url", "image_url": {"url": url_}} for url_ in images),
            {"type": "text", "text": text},
        ]
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": False,
            "logprobs": True,
            # Always the server's full list, and never `logprob_token_ids`:
            # see the module docstring (HTTP 500 under speculative decoding).
            "top_logprobs": _TOP_LOGPROBS_CAP,
            "allowed_token_ids": ids,
            "return_tokens_as_token_ids": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        resp = await client.post(f"{url}/v1/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return parse_readout(data, ids), (data.get("usage") or {})


async def _gather_or_cancel(aws: list[Awaitable]) -> list:
    """``asyncio.gather`` that cancels the remaining calls when one fails.

    Plain gather leaves the siblings running after the first exception; they
    would keep using the HTTP client after ``decide`` closes it and keep a
    GPU busy for an answer nobody reads.
    """
    tasks = [asyncio.ensure_future(a) for a in aws]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


def _token_id(entry: dict, label_ids: Sequence[int]) -> int | None:
    """Token id of one logprob entry: from ``token_id:N`` when the server
    honoured return_tokens_as_token_ids, else from a bare letter string."""
    tok = entry.get("token")
    if isinstance(tok, str):
        if tok.startswith(_TOKEN_ID_PREFIX):
            try:
                return int(tok[len(_TOKEN_ID_PREFIX):])
            except ValueError:
                return None
        idx = LETTERS.find(tok)
        if len(tok) == 1 and 0 <= idx < len(label_ids):
            return label_ids[idx]
    return None


def _count(value: Any) -> int | None:
    """A token count from a usage block, or None when it is not one."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        return None
    return int(value)


def _number(value: Any) -> float | None:
    """A finite log-probability, or None (not a number, a bool, NaN, inf)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def parse_readout(data: dict, label_ids: Sequence[int]) -> LabelReadout:
    """Turn one chat-completion response into per-label raw logprobs."""
    try:
        choice = data["choices"][0]
        content = (choice.get("logprobs") or {}).get("content") or []
        entry = content[0]
    except (KeyError, IndexError, TypeError) as e:
        raise DecisionBackendError("decision backend returned no logprobs", 502) from e

    if not isinstance(entry, dict):
        raise DecisionBackendError("decision backend returned no logprobs", 502)

    values: dict[int, float] = {}
    returned: list[float] = []       # every token in the list, label or not
    for t in entry.get("top_logprobs") or []:
        value = _number(t.get("logprob")) if isinstance(t, dict) else None
        if value is None:
            continue
        returned.append(value)
        tid = _token_id(t, label_ids)
        if tid is not None and tid in label_ids:
            # Two tokens can decode to the same letter; the likelier one counts.
            values[tid] = max(value, values.get(tid, value))
    sampled_id = _token_id(entry, label_ids)
    sampled_value = _number(entry.get("logprob"))
    if sampled_id in label_ids and sampled_value is not None:
        values.setdefault(sampled_id, sampled_value)
    if not values:
        raise DecisionBackendError("decision backend returned no label logprobs", 502)

    # A label that is not in the list is no likelier than the least likely
    # token that is, nor than the label the sampler picked (the sampler takes
    # the likeliest allowed label). Its value is set one nat below that
    # bound: tiny when the labels hold the probability (the usual case), and
    # always strictly below every label that was returned, so a missing
    # label never ties with the answer, even when the answer itself is the
    # only label seen (the model did not want to reply with a letter).
    floor = min([*returned, *values.values()]) - 1.0
    logprobs, complete = [], True
    for tid in label_ids:
        if tid in values:
            logprobs.append(values[tid])
        else:
            logprobs.append(floor)
            complete = False
    sampled = label_ids.index(sampled_id) if sampled_id in label_ids else None
    return LabelReadout(logprobs=logprobs, sampled=sampled, complete=complete)


def _to_result(q: Question, c) -> DecisionResult:
    options = list(q.options)
    likelihoods = dict(zip(options, c.likelihoods, strict=True))
    logprobs = dict(zip(options, c.logprobs, strict=True))
    chosen = options[c.answer_index]
    common: dict[str, Any] = {
        "id": q.id, "type": q.type, "likelihoods": likelihoods, "logprobs": logprobs,
        "label_mass": c.label_mass, "complete": c.complete,
    }
    if q.type == "boolean":
        return DecisionResult(answer=(chosen == "yes"), likelihood_true=likelihoods["yes"], **common)
    if q.type == "scale":
        expected = sum(int(o) * p for o, p in likelihoods.items())
        return DecisionResult(answer=int(chosen), expected_value=expected, likelihood=likelihoods[chosen], **common)
    return DecisionResult(answer=chosen, likelihood=likelihoods[chosen], **common)
