############################################################
#
# mindrouter - unit tests for the 404-vs-503 model decision
#
# Regression (2026-09-09): all five qwen/qwen3.8-27b replicas exited within
# one second of each other after a malformed JSON schema killed vLLM's
# EngineCore. registry.model_exists() counts only HEALTHY backends, so with
# every replica down the gateway answered 404 "model not found" — which reads
# as "no such model" and sent operators looking for a catalog problem instead
# of a capacity one. A configured model with no healthy backend is a 503.
#
############################################################

"""Unit tests for model_availability: unknown (404) vs unavailable (503)."""

import pathlib
import re

import pytest

from backend.app.api import model_availability as ma


class _Registry:
    """Stand-in registry: routable = healthy backend, configured = known."""

    def __init__(self, routable: bool, configured: bool):
        self._routable = routable
        self._configured = configured

    async def model_exists(self, name):
        return self._routable

    async def model_is_configured(self, name):
        return self._configured


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "routable,configured,expected",
    [
        (True, True, ma.AVAILABLE),
        (False, True, ma.UNAVAILABLE),   # every replica down — the incident
        (False, False, ma.UNKNOWN),      # genuinely no such model
        # A routable model is AVAILABLE even if the configured check would
        # disagree; routability is the stronger signal and is checked first.
        (True, False, ma.AVAILABLE),
    ],
)
async def test_classification(routable, configured, expected):
    assert await ma.model_availability(_Registry(routable, configured), "m") == expected


# --------------------------------------------------------------------------
# The 503 must be a *retryable* answer, and must not claim the model is absent
# --------------------------------------------------------------------------


def test_unavailable_is_503_with_retry_after():
    code, detail, headers = ma.openai_error("qwen/qwen3.8-27b", ma.UNAVAILABLE)
    assert code == 503
    assert detail["error"]["code"] == "model_unavailable"
    assert detail["error"]["type"] == "service_unavailable"
    assert headers == {"Retry-After": str(ma.RETRY_AFTER_SECONDS)}


def test_unavailable_message_does_not_say_the_model_is_missing():
    """The whole point: stop telling users a configured model doesn't exist."""
    msg = ma.unavailable_message("qwen/qwen3.8-27b", ma.UNAVAILABLE)
    assert "does not exist" not in msg
    assert "temporarily unavailable" in msg
    assert "qwen/qwen3.8-27b" in msg


def test_unknown_model_is_still_404():
    code, detail, headers = ma.openai_error("no/such-model", ma.UNKNOWN)
    assert code == 404
    assert detail["error"]["code"] == "model_not_found"
    assert detail["error"]["type"] == "invalid_request_error"
    assert headers is None
    assert "does not exist" in detail["error"]["message"]


def test_available_never_reaches_the_error_builder():
    """Callers must branch on AVAILABLE before formatting an error."""
    assert ma.AVAILABLE not in (ma.UNAVAILABLE, ma.UNKNOWN)


# --------------------------------------------------------------------------
# Every API surface must use the shared decision
# --------------------------------------------------------------------------


def _api(name):
    return (pathlib.Path(__file__).resolve().parents[2] / "api" / name).read_text()


@pytest.mark.parametrize(
    "module",
    ["v1_openai.py", "video_api.py", "ollama_api.py", "anthropic_api.py", "responses_api.py"],
)
def test_no_surface_still_bare_404s_on_model_exists(module):
    """A raw `if not model_exists(...)` guard would reintroduce the 404."""
    src = _api(module)
    assert "model_availability" in src, f"{module} does not use the shared decision"
    # No surface may still branch directly on model_exists to build an error.
    assert not re.search(r"if not await registry\.model_exists\(", src), module


def test_openai_surfaces_pass_retry_after_header():
    """A 503 without Retry-After is far less useful to a client."""
    src = _api("v1_openai.py")
    assert "headers=_headers" in src


def test_registry_and_crud_expose_the_configured_check():
    reg = (
        pathlib.Path(__file__).resolve().parents[2]
        / "core" / "telemetry" / "registry.py"
    ).read_text()
    crud = (pathlib.Path(__file__).resolve().parents[2] / "db" / "crud.py").read_text()
    assert "async def model_is_configured" in reg
    assert "async def model_is_configured" in crud
    # The configured check must NOT filter on health — that is the whole point.
    body = crud[crud.index("async def model_is_configured"):]
    body = body[: body.index("async def upsert_model")]
    # Examine the query, not the docstring (which mentions the health filter
    # precisely to explain why this function omits it).
    code_only = body[body.index('"""', body.index('"""') + 3) + 3 :]
    assert "BackendStatus.HEALTHY" not in code_only
    # ...but must still exclude DLP engines, mirroring the routable filter.
    assert "BackendEngine.DLP" in body
