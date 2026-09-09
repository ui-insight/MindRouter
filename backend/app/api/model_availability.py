############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# model_availability.py: 404-vs-503 decision for model routing
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Tell "no such model" apart from "every replica is down right now".

``registry.model_exists()`` means *routable* — it only counts HEALTHY
backends. So when all replicas of a configured model go down at once, every
API surface reported 404 ``model_not_found``, which reads as "this model does
not exist here" and sends callers hunting for a configuration or catalog
problem instead of a capacity one.

That is not hypothetical: on 2026-09-09 all five ``qwen/qwen3.8-27b``
replicas exited within one second of each other, and clients saw 404 for the
~2 minutes it took them to restart.

A configured model with no healthy backend is now **503** with
``Retry-After``, which is both truthful and actionable. A genuinely unknown
model is still 404. The decision lives here so every surface agrees.
"""

from typing import Any, Dict, Optional, Tuple

AVAILABLE = "available"
UNAVAILABLE = "unavailable"  # configured, but no healthy backend right now
UNKNOWN = "unknown"  # no such model anywhere in the fleet

# Advisory retry delay. Backend health checks run on a short cycle, so a
# transient all-replicas-down window typically clears in well under a minute.
RETRY_AFTER_SECONDS = 30


async def model_availability(registry: Any, model_name: str) -> str:
    """Classify a model as AVAILABLE, UNAVAILABLE, or UNKNOWN."""
    if await registry.model_exists(model_name):
        return AVAILABLE
    if await registry.model_is_configured(model_name):
        return UNAVAILABLE
    return UNKNOWN


def unavailable_message(model_name: str, availability: str) -> str:
    if availability == UNAVAILABLE:
        return (
            f"The model '{model_name}' is temporarily unavailable — no healthy "
            f"backend is currently serving it. Please retry shortly."
        )
    return f"The model '{model_name}' does not exist"


def openai_error(model_name: str, availability: str) -> Tuple[int, Dict[str, Any], Optional[Dict[str, str]]]:
    """(status_code, detail, headers) in the OpenAI error envelope."""
    if availability == UNAVAILABLE:
        return (
            503,
            {
                "error": {
                    "message": unavailable_message(model_name, availability),
                    "type": "service_unavailable",
                    "code": "model_unavailable",
                }
            },
            {"Retry-After": str(RETRY_AFTER_SECONDS)},
        )
    return (
        404,
        {
            "error": {
                "message": unavailable_message(model_name, availability),
                "type": "invalid_request_error",
                "code": "model_not_found",
            }
        },
        None,
    )
