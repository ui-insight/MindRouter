############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# core/sampling_policy.py: per-model sampling floor and token cap
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Per-model guard rails on sampling: a temperature floor and a max_tokens cap.

Why: several RL-trained models (GLM-5.3-Flash, MiMo-V2.6) degenerate under
greedy decoding — at ``temperature: 0`` they loop inside a think block or a
tool call until ``max_tokens``. Their vendors publish ``temperature 1.0 /
top_p 0.95``. A benchmark harness sending ``temperature: 0`` and
``max_tokens: 65536`` produced hundreds of 65k-token empty responses on
GLM-5.3-Flash in one night (2026-09-30), each holding four B300s for
minutes. The client's request is clamped here, at the gateway, after routing
and before translation, so every dialect is covered.

Config: ``app_config`` key ``sampling.policies`` — a JSON object mapping a
catalog model name (or ``"*"`` for every model) to
``{"min_temperature": <0..2>, "max_tokens": <int>=1>}``; either field may be
omitted. Exact model name wins over ``"*"``. Admin → Settings edits it.

No database or settings imports: pure functions, unit-tested in isolation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

CONFIG_KEY = "sampling.policies"
WILDCARD = "*"


@dataclass(frozen=True)
class SamplingPolicy:
    min_temperature: float | None = None
    max_tokens: int | None = None

    @property
    def is_empty(self) -> bool:
        return self.min_temperature is None and self.max_tokens is None


def validate_policies(raw: Any) -> tuple[dict[str, SamplingPolicy], list[str]]:
    """Parse the config value; return (valid policies, human-readable errors).

    ``raw`` may be the decoded JSON object or its string form. Invalid entries
    are reported and skipped so one bad line never disables the others.
    """
    errors: list[str] = []
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return {}, errors
        try:
            raw = json.loads(raw)
        except ValueError as exc:
            return {}, [f"not valid JSON: {exc}"]
    if raw is None:
        return {}, errors
    if not isinstance(raw, dict):
        return {}, ["must be a JSON object keyed by model name"]
    policies: dict[str, SamplingPolicy] = {}
    for name, spec in raw.items():
        if not isinstance(name, str) or not name.strip():
            errors.append("model names must be non-empty strings")
            continue
        if not isinstance(spec, dict):
            errors.append(f"{name}: policy must be an object")
            continue
        unknown = set(spec) - {"min_temperature", "max_tokens"}
        if unknown:
            errors.append(f"{name}: unknown keys {sorted(unknown)}")
            continue
        min_t = spec.get("min_temperature")
        cap = spec.get("max_tokens")
        if min_t is not None and (isinstance(min_t, bool) or not isinstance(min_t, (int, float)) or not 0 <= min_t <= 2):
            errors.append(f"{name}: min_temperature must be a number between 0 and 2")
            continue
        if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or cap < 1):
            errors.append(f"{name}: max_tokens must be an integer >= 1")
            continue
        policy = SamplingPolicy(
            min_temperature=float(min_t) if min_t is not None else None,
            max_tokens=cap,
        )
        if policy.is_empty:
            errors.append(f"{name}: policy sets nothing")
            continue
        policies[name.strip()] = policy
    return policies, errors


def parse_policies(raw: Any) -> dict[str, SamplingPolicy]:
    """Lenient form of :func:`validate_policies` for the request path."""
    return validate_policies(raw)[0]


def policy_for(policies: dict[str, SamplingPolicy], model_name: str | None) -> SamplingPolicy | None:
    """Exact name, then case-insensitive name, then the ``"*"`` wildcard."""
    if not policies or not model_name:
        return policies.get(WILDCARD) if policies else None
    if model_name in policies:
        return policies[model_name]
    lowered = model_name.lower()
    for name, policy in policies.items():
        if name != WILDCARD and name.lower() == lowered:
            return policy
    return policies.get(WILDCARD)


def apply_policy(request: Any, policy: SamplingPolicy | None) -> dict[str, tuple[Any, Any]]:
    """Clamp ``request.temperature`` / ``request.max_tokens`` in place.

    Returns ``{field: (before, after)}`` for every value that changed.
    A temperature the client did not send is left alone (the backend's own
    default applies); a missing ``max_tokens`` IS capped, because "no limit"
    is exactly how a looping generation runs to the end of the context.
    """
    changes: dict[str, tuple[Any, Any]] = {}
    if policy is None:
        return changes
    if policy.min_temperature is not None and hasattr(request, "temperature"):
        current = request.temperature
        if current is not None and current < policy.min_temperature:
            request.temperature = policy.min_temperature
            changes["temperature"] = (current, policy.min_temperature)
    if policy.max_tokens is not None and hasattr(request, "max_tokens"):
        current = request.max_tokens
        if current is None or current > policy.max_tokens:
            request.max_tokens = policy.max_tokens
            changes["max_tokens"] = (current, policy.max_tokens)
    return changes
