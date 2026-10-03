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

from backend.app.db import crud

from .schema import DecisionRequest, DecisionResult, DecisionUsage

DEFAULT_MODEL = "qwen3.8-27b"


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


async def get_decisions_config(db) -> dict:
    """Admin-configurable knobs (``app_config`` keys ``decisions.*``)."""
    return {
        "enabled": bool(await crud.get_config_json(db, "decisions.enabled", False)),
        "default_model": await crud.get_config_json(db, "decisions.default_model", DEFAULT_MODEL),
        # Only models whose chat template and tokenizer this scoring recipe has
        # been validated on. Others are refused with 400, not silently scored.
        "allowed_models": list(await crud.get_config_json(db, "decisions.allowed_models", [DEFAULT_MODEL])),
        # Lower operational ceiling on state size (schema.MAX_STATE_CHARS is the hard cap).
        "max_state_chars": int(await crud.get_config_json(db, "decisions.max_state_chars", 32_000)),
        # Concurrent scoring calls issued per request.
        "fanout": int(await crud.get_config_json(db, "decisions.fanout", 8)),
        # Concurrent scoring calls per backend across ALL requests, per app
        # worker process. These calls take no scheduler slot, so this is what
        # keeps one busy key from saturating a chat replica unseen; the fleet
        # worst case on one backend is (uvicorn workers x this).
        "backend_concurrency": int(await crud.get_config_json(db, "decisions.backend_concurrency", 4)),
        # Model names served by a System One server of their own (Laya,
        # Open-Jev) instead of by letter scoring on vLLM: name -> Upstream.
        "upstreams": await get_upstreams(db),
    }


async def get_upstreams(db) -> dict:
    from .upstream import parse_upstreams

    upstreams, problems = parse_upstreams(await crud.get_config_json(db, "decisions.upstreams", {}))
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
