############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# services/decisions/__init__.py: typed decisions ("System One")
# capability — EXPERIMENTAL, transitional
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Typed decisions: state + typed questions in, typed answers out.

STATUS: EXPERIMENTAL / TRANSITIONAL. The HTTP surface is TypeSafe's System One
API (Jev's ``POST /v1/systemone``; see ``systemone.py``). The caller's ``model``
field picks one of two ways an answer is produced:

* ``vllm_logprobs.VLLMLogprobsBackend`` — one-token letter scoring on a vLLM
  chat model MindRouter already serves (``decisions.allowed_models``). No
  dedicated model, no extra GPU.
* ``upstream.SystemOneUpstreamBackend`` — forward to a purpose-built decision
  model that is its own System One server, e.g. Laya (``decisions.upstreams``).

Config
------
All knobs live in ``app_config`` under ``decisions.*`` (see
``get_decisions_config``) and on the Admin -> Settings card; no migration, no
environment variable. The capability is OFF unless ``decisions.enabled``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .schema import MAX_STATE_CHARS, DecisionRequest, DecisionResult, DecisionUsage
from .systemone import parse_permutations, parse_temperature

# The catalog name, exactly as GET /v1/models lists it. (2.9.83 shipped the
# short name "qwen3.8-27b", which no backend serves: jev-latest answered 404.)
DEFAULT_MODEL = "qwen/qwen3.8-27b"


@dataclass
class DecisionOutcome:
    results: list[DecisionResult]
    usage: DecisionUsage | None
    backend_id: int | None
    backend_name: str | None


class DecisionBackendError(Exception):
    """Raised by a backend; ``status_code`` is what the API should answer."""

    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code


class DecisionBackend(Protocol):
    name: str

    async def decide(
        self, request: DecisionRequest, model: str, *, fanout: int, backend_concurrency: int
    ) -> DecisionOutcome:
        """Answer every question in ``request`` with ``model``. ``model`` is
        already alias-resolved and allow-listed. ``fanout`` bounds this
        request's concurrent calls; ``backend_concurrency`` bounds calls to
        one backend across all requests in this process. Raise
        DecisionBackendError for anything the caller should see."""
        ...


async def _load_settings(db) -> dict:
    """Every ``decisions.*`` row in one query. This runs on each request, and
    decision traffic is high-volume by design; eight separate lookups would
    be most of a short request's database work."""
    import json

    from sqlalchemy import select

    from backend.app.db.models import AppConfig

    rows = await db.execute(select(AppConfig.key, AppConfig.value).where(AppConfig.key.like("decisions.%")))
    settings = {}
    for key, raw in rows.all():
        try:
            settings[key] = json.loads(raw)
        except (ValueError, TypeError):
            continue  # an unreadable row falls back to its default
    return settings


async def get_decisions_config(db) -> dict:
    """Admin-configurable knobs (``app_config`` keys ``decisions.*``)."""
    raw = await _load_settings(db)

    def get(name: str, default):
        value = raw.get(f"decisions.{name}")
        return default if value is None else value

    return {
        "enabled": bool(get("enabled", False)),
        "default_model": get("default_model", DEFAULT_MODEL),
        # Only models whose chat template and tokenizer this scoring recipe has
        # been validated on. Others are refused, not silently scored.
        "allowed_models": list(get("allowed_models", [DEFAULT_MODEL])),
        # Lower operational ceiling on state size (schema.MAX_STATE_CHARS is the hard cap).
        "max_state_chars": min(MAX_STATE_CHARS, int(get("max_state_chars", 32_000))),
        # Concurrent scoring calls issued per request.
        "fanout": int(get("fanout", 8)),
        # Concurrent scoring calls per backend across ALL requests, per app
        # worker process. These calls take no scheduler slot, so this is what
        # keeps one busy key from saturating a chat replica unseen; the fleet
        # worst case on one backend is (uvicorn workers x this).
        "backend_concurrency": int(get("backend_concurrency", 4)),
        # Model names served by a System One server of their own (Laya,
        # Open-Jev) instead of by letter scoring on vLLM: name -> Upstream.
        "upstreams": _upstreams(get("upstreams", {})),
        # Letter scoring only (never an upstream): option orders averaged and
        # temperature applied, per question type. See systemone.DEFAULT_*.
        "permutations": _checked_setting("permutations", parse_permutations(get("permutations", {}))),
        "temperature": _checked_setting("temperature", parse_temperature(get("temperature", {}))),
    }


def _checked_setting(name: str, parsed: tuple) -> dict:
    value, problems = parsed
    if problems:
        # Only reachable with a hand-edited app_config row; bad entries keep their default.
        from backend.app.logging_config import get_logger

        get_logger(__name__).warning("decisions_setting_invalid", setting=name, problems=problems)
    return value


def _upstreams(raw) -> dict:
    from .upstream import parse_upstreams

    upstreams, problems = parse_upstreams(raw)
    if problems:
        # Saved through the admin form these are already valid; this only
        # fires for a hand-edited app_config row. Bad entries are skipped.
        from backend.app.logging_config import get_logger

        get_logger(__name__).warning("decisions_upstreams_invalid", problems=problems)
    return upstreams


_backend: DecisionBackend | None = None


_upstream_backend = None


def get_upstream_backend():
    """The forwarder for models that are their own System One servers."""
    global _upstream_backend
    if _upstream_backend is None:
        from .upstream import SystemOneUpstreamBackend

        _upstream_backend = SystemOneUpstreamBackend()
    return _upstream_backend


def get_decision_backend() -> DecisionBackend:
    """The configured backend. There is exactly one today; when a second one
    exists, select it here (and only here) from ``decisions.backend``."""
    global _backend
    if _backend is None:
        from .vllm_logprobs import VLLMLogprobsBackend

        _backend = VLLMLogprobsBackend()
    return _backend


__all__ = [
    "DecisionBackend",
    "DecisionBackendError",
    "DecisionOutcome",
    "DecisionRequest",
    "DecisionResult",
    "DecisionUsage",
    "DEFAULT_MODEL",
    "get_decision_backend",
    "get_upstream_backend",
    "get_decisions_config",
]
