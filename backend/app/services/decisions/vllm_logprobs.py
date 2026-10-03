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
* ``logprobs=true, logprob_token_ids=[A, B, ...]`` (vLLM >= 0.29) — the raw
  log-probability of every label regardless of the server's ``--max-logprobs``
  cap. ``top_logprobs`` is sent too, so an older server that ignores
  ``logprob_token_ids`` still returns its top-20; labels missing from that
  list are floored and the decision is flagged ``complete=false``.

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
  which is the same number the processed path yields after masking.
* The ``packed`` mode (all questions in one sequence, read via
  ``prompt_logprobs``) is possible over ``/v1/completions`` with token-id
  prompts, but the library itself measures ``separate`` as faster AND exact
  on vLLM, so packed is not implemented.

Label tokens are looked up once per (backend, model) through the server's
``/tokenize`` endpoint and each must be a single token, as in so1.
"""
from __future__ import annotations

import asyncio
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

# vLLM's default --max-logprobs; the fallback path cannot see past it.
_TOP_LOGPROBS_CAP = 20
_TOKEN_ID_PREFIX = "token_id:"


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
        backend = await self._pick_backend(model)
        settings = get_settings()
        timeout = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)
        verify = bool(getattr(settings, "internal_tls_verify", True))
        max_n = max(len(q.options) for q in request.questions)
        backend_gate = self._backend_gate(backend.id, backend_concurrency)

        async with httpx.AsyncClient(timeout=timeout, verify=verify) as client:
            try:
                label_ids = await self._get_label_ids(client, backend.url, model, max_n)

                # One "view" per (question, option order).
                views: list[tuple[int, list[int]]] = []
                for qi, q in enumerate(request.questions):
                    for order in option_orders(len(q.options), request.permutations):
                        views.append((qi, order))

                request_gate = asyncio.Semaphore(max(1, fanout))

                async def score(qi: int, order: list[int]) -> tuple[LabelReadout, dict]:
                    q = request.questions[qi]
                    shown = [q.options[k] for k in order]
                    text = render_turn(q.question, shown, request.state)
                    async with request_gate, backend_gate:
                        return await self._score_one(client, backend.url, model, text, label_ids[: len(shown)])

                # Every view starts with the same state. Score the first one on
                # its own so it fills vLLM's prefix cache; the rest then hit the
                # cache instead of each prefilling the whole state in parallel.
                if request.state and len(views) > 1:
                    first = await score(*views[0])
                    rest = await _gather_or_cancel([score(qi, order) for qi, order in views[1:]])
                    scored = [first, *rest]
                else:
                    scored = await _gather_or_cancel([score(qi, order) for qi, order in views])
            except DecisionBackendError:
                raise
            except httpx.HTTPStatusError as e:
                # Status only: an engine's error text can quote the prompt.
                logger.warning("decision_backend_http_error", backend_id=backend.id, status=e.response.status_code)
                raise DecisionBackendError(f"decision backend returned HTTP {e.response.status_code}", 502) from e
            except httpx.HTTPError as e:
                logger.warning("decision_backend_unreachable", backend_id=backend.id, error=str(e))
                raise DecisionBackendError("decision backend unreachable", 502) from e
            except (ValueError, KeyError, TypeError) as e:
                # A reply that is not the JSON shape we read (resp.json() raises
                # ValueError on a non-JSON body, e.g. a proxy error page).
                logger.warning("decision_backend_bad_reply", backend_id=backend.id, error=str(e)[:300])
                raise DecisionBackendError("decision backend returned a malformed reply", 502) from e

        # Regroup per question and fold the views back into the original order.
        per_q: dict[int, list[tuple[list[int], LabelReadout]]] = {}
        prompt_tokens = scoring_tokens = 0
        cached: int | None = None
        for (qi, order), (readout, usage) in zip(views, scored, strict=True):
            per_q.setdefault(qi, []).append((order, readout))
            prompt_tokens += int(usage.get("prompt_tokens") or 0)
            scoring_tokens += int(usage.get("completion_tokens") or 0)
            c = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
            if c is not None:
                cached = (cached or 0) + int(c)

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

    async def _pick_backend(self, model: str):
        """A random healthy, circuit-closed vLLM backend serving ``model``.

        Mirrors the direct-to-backend precedents (image_policy, dlp_worker):
        no scheduler slot is taken; ``_backend_gate`` bounds the load instead.
        See docs/decisions-api.md "Limitations".
        """
        backend = await get_registry().pick_available_backend(model, engine=BackendEngine.VLLM)
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
            raise DecisionBackendError("decision backend tokenizer unreachable", 502) from e
        self._label_ids[key] = ids
        return ids

    async def _score_one(
        self, client: httpx.AsyncClient, url: str, model: str, text: str, label_ids: Sequence[int]
    ) -> tuple[LabelReadout, dict]:
        ids = list(label_ids)
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": text}],
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": False,
            "logprobs": True,
            "top_logprobs": min(len(ids), _TOP_LOGPROBS_CAP),
            "logprob_token_ids": ids,
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


def parse_readout(data: dict, label_ids: Sequence[int]) -> LabelReadout:
    """Turn one chat-completion response into per-label raw logprobs."""
    try:
        choice = data["choices"][0]
        content = (choice.get("logprobs") or {}).get("content") or []
        entry = content[0]
    except (KeyError, IndexError, TypeError) as e:
        raise DecisionBackendError("decision backend returned no logprobs", 502) from e

    values: dict[int, float] = {}
    for t in entry.get("top_logprobs") or []:
        tid = _token_id(t, label_ids)
        if tid is not None and isinstance(t.get("logprob"), (int, float)):
            values[tid] = float(t["logprob"])
    sampled_id = _token_id(entry, label_ids)
    if sampled_id is not None and isinstance(entry.get("logprob"), (int, float)):
        values.setdefault(sampled_id, float(entry["logprob"]))
    if not values:
        raise DecisionBackendError("decision backend returned no label logprobs", 502)

    floor = min(values.values()) - 1.0
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
