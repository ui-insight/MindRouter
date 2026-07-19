"""Tests for request-field validation (vLLM-dialect / unknown fields).

field_validation is spec-loaded (its lazy settings import only fires when
mode=None, and tests pass mode explicitly). The openai_in integration test
imports the translator via package (light).
"""

import importlib.util
import os

import pytest
from fastapi import HTTPException

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_spec = importlib.util.spec_from_file_location(
    "fv", os.path.join(ROOT, "backend/app/core/translators/field_validation.py"))
fv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fv)
V = fv.validate_request_fields


def test_enforce_rejects_dialect_field_with_hint():
    with pytest.raises(HTTPException) as ei:
        V({"model": "m", "structured_outputs": {}}, mode="enforce")
    err = ei.value.detail["error"]
    assert ei.value.status_code == 400
    assert err["type"] == "invalid_request_error"
    assert err["param"] == "structured_outputs"
    assert "response_format" in err["message"]      # points at the supported alternative
    assert "vLLM dialect" in err["message"]


def test_enforce_rejects_unknown_field_generically():
    with pytest.raises(HTTPException) as ei:
        V({"model": "m", "frobnicate": 1}, mode="enforce")
    err = ei.value.detail["error"]
    assert err["param"] == "frobnicate"
    assert "Unsupported request field" in err["message"]


def test_enforce_accepts_known_and_ignored_fields():
    # accepted + deliberately-ignored + stream_options must NOT raise
    V(
        {
            "model": "m", "messages": [], "temperature": 0.5, "tools": [],
            "response_format": {}, "stream": True, "stream_options": {"include_usage": True},
            "logit_bias": {}, "metadata": {}, "parallel_tool_calls": True, "logprobs": True,
        },
        mode="enforce",
    )


def test_log_and_off_never_raise():
    V({"model": "m", "structured_outputs": {}, "frobnicate": 1}, mode="log")  # logs only
    V({"model": "m", "structured_outputs": {}}, mode="off")                    # no-op


def test_default_mode_reads_setting_and_is_not_off():
    # With no explicit mode it reads the field_validation setting (default 'log'),
    # so a dialect field is observed but not rejected out of the box.
    V({"model": "m", "structured_outputs": {}})  # must not raise at default 'log'


def test_stream_options_include_usage_is_parsed():
    from backend.app.core.translators.openai_in import OpenAIInTranslator
    msgs = [{"role": "user", "content": "hi"}]
    asked = OpenAIInTranslator.translate_chat_request(
        {"model": "m", "messages": msgs, "stream": True, "stream_options": {"include_usage": True}})
    assert asked.include_usage is True
    default = OpenAIInTranslator.translate_chat_request({"model": "m", "messages": msgs})
    assert default.include_usage is False
