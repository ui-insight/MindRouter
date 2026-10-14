############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# services/decisions/upstream.py: forward a System One request
# to a server that already speaks it (Laya's laya-serve, Open-Jev)
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Decision models that are their own System One servers.

A purpose-built decision model (Laya, Open-Jev) ships a server that already
answers ``POST /v1/systemone``. For those there is nothing to score: MindRouter
forwards the caller's ``state`` and ``questions`` and returns the answers.
Which caller-facing model name goes where is the admin setting
``decisions.upstreams``::

    {"laya": {"url": "https://host:8010", "api_key": "...", "model": null}}

``model`` is the name sent upstream (``null`` omits it, which lets Laya's
router pick a checkpoint by language); ``api_key`` is the upstream's bearer
token, if it requires one.

An upstream may also be registered as a backend with engine ``decision`` and
the same URL (Clef is): it is then health-polled, has a circuit breaker and is
skipped while down (``registry.decision_server_state``). An unregistered
upstream is dialed blind and answers 502 on every request until fixed.

The reply is not trusted: every answer is checked against the question that
was asked and reduced to TypeSafe's fields before it reaches the caller, so a
client never sees a shape its SDK would reject.
"""
from __future__ import annotations

import asyncio
import math
import re
from dataclasses import dataclass
from typing import Any

import httpx

from backend.app.logging_config import get_logger
from backend.app.settings import get_settings

from . import DecisionBackendError
from .systemone import SystemOneValidationError, choice_confidence, score_confidence

logger = get_logger(__name__)

SCORE_SEMANTICS_UPSTREAM = "upstream_model_probability"

# No System One request reads more than this; a larger reported count is a
# malfunctioning upstream, not something to bill.
MAX_PLAUSIBLE_TOKENS = 2_000_000

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/\-]{0,99}$")
_DEFAULT_TIMEOUT = 30.0
_MAX_TIMEOUT = 120.0


@dataclass(frozen=True)
class Upstream:
    name: str               # what callers send in ``model``
    url: str                # server root; /v1/systemone is appended
    api_key: str | None = None
    model: str | None = None  # sent upstream as ``model``; None omits the field
    timeout: float = _DEFAULT_TIMEOUT
    images: bool = False      # the upstream accepts the ``images`` extension (Clef does; Laya does not)


def parse_upstreams(raw: Any) -> tuple[dict[str, Upstream], list[str]]:
    """Validate ``decisions.upstreams``. Returns (upstreams, problems); an
    entry with a problem is left out rather than half-applied."""
    if raw in (None, "", {}):
        return {}, []
    if not isinstance(raw, dict):
        return {}, ["upstreams must be a JSON object keyed by model name"]
    out: dict[str, Upstream] = {}
    problems: list[str] = []
    for name, spec in raw.items():
        if not isinstance(name, str) or not _NAME_RE.match(name):
            problems.append(f"{name!r}: not a valid model name")
            continue
        if not isinstance(spec, dict):
            problems.append(f"{name}: must be an object with a 'url'")
            continue
        unknown = set(spec) - {"url", "api_key", "model", "timeout", "images"}
        if unknown:
            problems.append(f"{name}: unknown keys {sorted(unknown)}")
            continue
        url = spec.get("url")
        if not isinstance(url, str) or not re.match(r"^https?://[^\s/]+", url):
            problems.append(f"{name}: 'url' must be an http(s) URL")
            continue
        api_key, model, timeout = spec.get("api_key"), spec.get("model"), spec.get("timeout", _DEFAULT_TIMEOUT)
        if api_key is not None and (not isinstance(api_key, str) or not api_key.strip()):
            problems.append(f"{name}: 'api_key' must be a non-empty string or null")
            continue
        if model is not None and (not isinstance(model, str) or not model.strip()):
            problems.append(f"{name}: 'model' must be a non-empty string or null")
            continue
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= _MAX_TIMEOUT:
            problems.append(f"{name}: 'timeout' must be a number of seconds up to {int(_MAX_TIMEOUT)}")
            continue
        accepts_images = spec.get("images", False)
        if not isinstance(accepts_images, bool):
            problems.append(f"{name}: 'images' must be true or false")
            continue
        out[name] = Upstream(name=name, url=url.rstrip("/"), api_key=api_key, model=model, timeout=float(timeout),
                             images=accepts_images)
    return out, problems


# Shown in the admin form in place of a stored api_key, and accepted back as
# "keep the key already stored for this name".
KEY_PLACEHOLDER = "(stored)"


def mask_keys(raw: Any) -> Any:
    """The upstreams setting with every api_key replaced by a placeholder, for
    display. A bearer token is never rendered back into a page."""
    if not isinstance(raw, dict):
        return raw
    return {
        name: ({**spec, "api_key": KEY_PLACEHOLDER} if isinstance(spec, dict) and spec.get("api_key") else spec)
        for name, spec in raw.items()
    }


def restore_keys(submitted: Any, stored: Any) -> tuple[Any, list[str]]:
    """Replace placeholders in a submitted setting with the stored keys.
    Returns (value, problems); a placeholder with no stored key is a problem."""
    if not isinstance(submitted, dict):
        return submitted, []
    stored = stored if isinstance(stored, dict) else {}
    out, problems = {}, []
    for name, spec in submitted.items():
        if isinstance(spec, dict) and spec.get("api_key") == KEY_PLACEHOLDER:
            previous = stored.get(name)
            key = previous.get("api_key") if isinstance(previous, dict) else None
            if not key:
                problems.append(f"{name}: no stored api_key to keep; enter the key")
                continue
            spec = {**spec, "api_key": key}
        out[name] = spec
    return out, problems


@dataclass
class UpstreamAnswer:
    answers: dict[str, dict[str, Any]]
    input_tokens: int
    output_tokens: int
    upstream_model: str | None
    extras: dict[str, Any]   # non-Jev fields worth surfacing (e.g. Laya's routing)


class SystemOneUpstreamBackend:
    name = "systemone_upstream"

    def __init__(self) -> None:
        # upstream name -> (limit, semaphore), shared by every request in this process
        self._gates: dict[str, tuple[int, asyncio.Semaphore]] = {}

    def _gate(self, name: str, limit: int) -> asyncio.Semaphore:
        limit = max(1, int(limit))
        current = self._gates.get(name)
        if current is None or current[0] != limit:
            current = (limit, asyncio.Semaphore(limit))
            self._gates[name] = current
        return current[1]

    async def answer(
        self, upstream: Upstream, state: Any, questions: dict[str, Any], *, concurrency: int = 4,
        images: list[str] | None = None,
    ) -> UpstreamAnswer:
        # Only TypeSafe's question fields go upstream. Anything else a caller
        # put on a question is not sent under MindRouter's credential.
        questions = {
            qid: {k: q[k] for k in ("type", "instructions", "criteria") if k in q}
            for qid, q in questions.items()
        }
        payload: dict[str, Any] = {"state": state, "questions": questions}
        if upstream.model is not None:
            payload["model"] = upstream.model
        if images:
            # Already validated and normalized to data URLs by the gateway.
            payload["images"] = images
        headers = {"Content-Type": "application/json"}
        if upstream.api_key:
            headers["Authorization"] = f"Bearer {upstream.api_key}"
        verify = bool(getattr(get_settings(), "internal_tls_verify", True))
        timeout = httpx.Timeout(connect=10.0, read=upstream.timeout, write=10.0, pool=10.0)

        try:
            async with self._gate(upstream.name, concurrency), httpx.AsyncClient(timeout=timeout, verify=verify) as client:
                resp = await client.post(f"{upstream.url}/v1/systemone", json=payload, headers=headers)
        except httpx.HTTPError as e:
            logger.warning("decision_upstream_unreachable", upstream=upstream.name, error=str(e)[:200])
            raise DecisionBackendError(f"decision model '{upstream.name}' is unreachable", 502) from e

        if resp.status_code in (400, 413, 422):
            # The upstream refused the request's content: the caller's to fix.
            # The upstream's wording goes back to the caller (it is about their
            # own request) but not into the audit row or logs, since it may
            # quote the state or a question: ``audit_message`` is what is stored.
            raise SystemOneValidationError(
                [{"loc": ["body"], "type": "value_error",
                  "msg": f"rejected by decision model '{upstream.name}': {_detail_text(resp)}"}],
                audit_message=f"rejected by decision model '{upstream.name}' (HTTP {resp.status_code})",
            )
        if resp.status_code in (429, 503, 529):
            raise DecisionBackendError(f"decision model '{upstream.name}' is busy", 503)
        if resp.status_code != 200:
            # 401/403 here means MindRouter's configured key is wrong, not the caller's.
            # Status only: an error body from a server we do not control may quote the request.
            logger.warning("decision_upstream_http_error", upstream=upstream.name, status=resp.status_code)
            raise DecisionBackendError(f"decision model '{upstream.name}' returned HTTP {resp.status_code}", 502)
        try:
            data = resp.json()
        except ValueError as e:
            raise DecisionBackendError(f"decision model '{upstream.name}' returned a malformed reply", 502) from e
        return _checked(upstream.name, data, questions)


def _detail_text(resp: httpx.Response) -> str:
    try:
        body = resp.json()
    except ValueError:
        return resp.text[:300]
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, list):
        return "; ".join(str(d.get("msg", d)) if isinstance(d, dict) else str(d) for d in detail)[:300]
    if isinstance(detail, str):
        return detail[:300]
    error = body.get("error") if isinstance(body, dict) else None
    return (error if isinstance(error, str) else str(body))[:300]


def _prob(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    v = float(value)
    return v if math.isfinite(v) and -1e-6 <= v <= 1 + 1e-6 else None


def _checked(name: str, data: Any, questions: dict[str, Any]) -> UpstreamAnswer:
    """Verify the upstream's reply against what was asked and keep only
    TypeSafe's answer fields."""
    def bad(why: str) -> DecisionBackendError:
        logger.warning("decision_upstream_bad_reply", upstream=name, reason=why)
        return DecisionBackendError(f"decision model '{name}' returned an invalid reply ({why})", 502)

    if not isinstance(data, dict) or not isinstance(data.get("answers"), dict):
        raise bad("no answers")
    raw_answers = data["answers"]
    if set(raw_answers) != set(questions):
        raise bad("answer ids do not match the questions")

    answers: dict[str, dict[str, Any]] = {}
    for qid, q in questions.items():
        kind = q.get("type") if isinstance(q, dict) else None
        a = raw_answers[qid]
        if not isinstance(a, dict) or a.get("type") != kind:
            raise bad(f"answer '{qid}' is not a {kind}")
        if kind == "noul":
            p = _prob(a.get("noul"))
            if p is None:
                raise bad(f"'{qid}' has no noul probability")
            answers[qid] = {"type": "noul", "noul": min(1.0, max(0.0, p))}
            continue
        probs = a.get("probabilities")
        if not isinstance(probs, dict) or not probs:
            raise bad(f"'{qid}' lacks probabilities")
        clean = {}
        for key, value in probs.items():
            p = _prob(value)
            if p is None:
                raise bad(f"'{qid}' has a non-probability value")
            clean[str(key)] = min(1.0, max(0.0, p))
        criteria = q.get("criteria")
        # ``confidence`` is recomputed from the probabilities with TypeSafe's
        # formulas rather than taken from the upstream. Servers disagree on what
        # the field means (Clef reports its top probability; Laya and Jev use
        # the formula), and a caller thresholding on it must get one meaning
        # whichever model answered.
        ordered = list(clean.values())
        if kind == "choice":
            choice = a.get("choice")
            if not isinstance(choice, str) or choice not in clean:
                raise bad(f"'{qid}' chose an option it did not list")
            if isinstance(criteria, dict) and set(clean) != set(criteria):
                raise bad(f"'{qid}' answered a different set of options than was asked")
            answers[qid] = {"type": "choice", "choice": choice, "probabilities": clean,
                            "confidence": choice_confidence(ordered)}
        elif kind == "score":
            score, legend = a.get("score"), a.get("legend")
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                raise bad(f"'{qid}' has no numeric score")
            if isinstance(criteria, list) and set(clean) != {str(i) for i in range(len(criteria))}:
                raise bad(f"'{qid}' answered a different set of levels than was asked")
            if not isinstance(legend, dict):
                # Laya and Jev both echo the levels; rebuild from the request if a server does not.
                levels = q.get("criteria") if isinstance(q.get("criteria"), list) else []
                legend = {str(i): level for i, level in enumerate(levels)}
            # Levels in index order, whatever order the upstream listed them in.
            by_level = [clean[str(i)] for i in range(len(clean))] if all(str(i) in clean for i in range(len(clean))) else ordered
            answers[qid] = {"type": "score", "score": float(score), "legend": {str(k): v for k, v in legend.items()},
                            "probabilities": clean, "confidence": score_confidence(by_level)}
        else:
            raise bad(f"'{qid}' has unknown type")

    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}

    def tokens(key: str) -> int:
        # These counts are charged to the caller's quota and stored in an INT
        # column, so a reply claiming an impossible number is a bad reply.
        v = usage.get(key)
        if v is None:
            return 0
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            raise bad(f"usage.{key} is not a token count")
        if v > MAX_PLAUSIBLE_TOKENS:
            raise bad(f"usage.{key} is implausibly large")
        return int(v)

    extras = {k: data[k] for k in ("routing",) if k in data}
    # A decision model with a short context cuts the state to fit. Servers that
    # say so (Laya, clef_service) report it in usage; pass it on, because an
    # answer about half the state is not the answer the caller asked for.
    if isinstance(usage.get("truncated"), bool):
        extras["truncated"] = usage["truncated"]
    dropped = usage.get("state_tokens_dropped")
    if isinstance(dropped, int) and not isinstance(dropped, bool) and dropped >= 0:
        extras["state_tokens_dropped"] = dropped
    upstream_model = data.get("model") if isinstance(data.get("model"), str) else None
    return UpstreamAnswer(answers=answers, input_tokens=tokens("input_tokens"), output_tokens=tokens("output_tokens"),
                          upstream_model=upstream_model, extras=extras)
