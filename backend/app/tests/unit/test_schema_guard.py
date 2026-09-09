############################################################
#
# mindrouter - unit tests for structured-output schema guarding
#
# Incident 2026-09-09: a response_format carrying "enum": [] crashed vLLM's
# grammar compiler hard enough to kill EngineCore. The gateway then retried
# the request on the next replica, and the next — all five qwen3.8-27b
# workers died within one second, and the model went 404 for ~2 minutes.
#
# Two defences, both tested here:
#   1. reject the schema before dispatch (primary)
#   2. never re-dispatch a request that already killed a backend (backstop)
#
############################################################

"""Unit tests for schema_guard and request-fault retry classification."""

import importlib.util
import pathlib

import pytest

_CORE = pathlib.Path(__file__).resolve().parents[2] / "core"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sg = _load("schema_guard", _CORE / "schema_guard.py")


# The exact schema from production request 16936602.
INCIDENT_SCHEMA = {
    "type": "object",
    "properties": {
        "fields": {
            "type": "array",
            "description": "One entry per form field, same names as given",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "enum": []},
                    "label": {"type": "string", "maxLength": 200},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["name", "label", "confidence"],
                "additionalProperties": False,
            },
            "minItems": 0,
            "maxItems": 0,
        }
    },
    "required": ["fields"],
    "additionalProperties": False,
}


# --------------------------------------------------------------------------
# The incident schema must be rejected
# --------------------------------------------------------------------------


def test_incident_schema_is_rejected():
    with pytest.raises(sg.SchemaRejection) as exc:
        sg.validate_json_schema(INCIDENT_SCHEMA)
    assert "enum" in exc.value.reason


def test_rejection_message_locates_the_offending_field():
    """A caller has to be able to find it — the schema was machine-generated."""
    with pytest.raises(sg.SchemaRejection) as exc:
        sg.validate_json_schema(INCIDENT_SCHEMA)
    assert "enum" in exc.value.path
    assert "response_format" in exc.value.message


def test_incident_schema_rejected_through_the_response_format_wrapper():
    rf = {"type": "json_schema",
          "json_schema": {"name": "field_labels", "schema": INCIDENT_SCHEMA}}
    with pytest.raises(sg.SchemaRejection):
        sg.validate_response_format(rf)
    with pytest.raises(sg.SchemaRejection):
        sg.validate_canonical_response_format(rf)


# --------------------------------------------------------------------------
# Other schemas that admit no value / crash compilers
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "string", "enum": []},
        {"oneOf": []},
        {"anyOf": []},
        {"allOf": []},
        {"type": "array", "minItems": 5, "maxItems": 2},
        {"type": "array", "maxItems": -1},
        {"type": "object", "properties": {"a": {"type": "string"}},
         "required": ["b"], "additionalProperties": False},
        {"properties": {"deep": {"items": {"enum": []}}}},   # nested
    ],
)
def test_impossible_schemas_are_rejected(schema):
    with pytest.raises(sg.SchemaRejection):
        sg.validate_json_schema(schema)


# --------------------------------------------------------------------------
# Valid schemas must keep working — a false rejection breaks a real caller
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]},
        {"type": "string", "enum": ["x", "y"]},
        {"type": "array", "items": {"type": "string"}, "minItems": 0, "maxItems": 10},
        {"type": "array", "minItems": 0, "maxItems": 0},   # odd but compilable
        # required without additionalProperties:false is permissive, not fatal
        {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["b"]},
        {"oneOf": [{"type": "string"}, {"type": "number"}]},
        {},
    ],
)
def test_valid_schemas_pass(schema):
    sg.validate_json_schema(schema)


@pytest.mark.parametrize(
    "rf",
    [None, {"type": "text"}, {"type": "json_object"},
     {"type": "json_schema"}, {"type": "json_schema", "json_schema": None}],
)
def test_non_schema_requests_are_untouched(rf):
    sg.validate_response_format(rf)
    sg.validate_canonical_response_format(rf)


def test_pydantic_style_response_format_is_handled():
    """Canonical requests carry an object with .type/.json_schema, not a dict."""

    class _RF:
        type = "json_schema"
        json_schema = {"name": "x", "schema": INCIDENT_SCHEMA}

    with pytest.raises(sg.SchemaRejection):
        sg.validate_canonical_response_format(_RF())


def test_schema_inline_without_a_schema_key():
    """Some clients put the schema directly under json_schema."""
    rf = {"type": "json_schema", "json_schema": {"type": "string", "enum": []}}
    with pytest.raises(sg.SchemaRejection):
        sg.validate_canonical_response_format(rf)


# --------------------------------------------------------------------------
# The walk must be bounded — the schema is caller-supplied
# --------------------------------------------------------------------------


def test_deeply_nested_schema_is_bounded():
    node = {"type": "string"}
    for _ in range(sg.MAX_DEPTH + 10):
        node = {"properties": {"n": node}}
    with pytest.raises(sg.SchemaRejection) as exc:
        sg.validate_json_schema(node)
    assert "deep" in exc.value.reason or "nodes" in exc.value.reason


def test_guard_is_wired_into_both_retry_paths():
    """A fix on only the non-streaming path would leave the hole open."""
    src = (
        pathlib.Path(__file__).resolve().parents[2] / "services" / "inference.py"
    ).read_text()
    assert src.count("self._guard_structured_output(request)") == 2


# --------------------------------------------------------------------------
# Backstop: a request that killed one backend must not be handed to another
# --------------------------------------------------------------------------


class _Resp:
    def __init__(self, text, status_code=500):
        self.text = text
        self.status_code = status_code


class _Exc(Exception):
    def __init__(self, text, status_code=500):
        self.response = _Resp(text, status_code)


def _svc(detection=True):
    """A bare InferenceService-like object exposing only the classifier."""
    from types import SimpleNamespace
    import importlib.util as ilu

    spec = ilu.spec_from_file_location(
        "inference_src",
        pathlib.Path(__file__).resolve().parents[2] / "services" / "inference.py",
    )
    # Executing inference.py pulls the whole app; read the class attributes off
    # the source instead by rebuilding the two pure helpers under test.
    src = spec.origin
    text = pathlib.Path(src).read_text()
    ns = {}
    start = text.index("    _MIN_FAULT_BODY_CHARS = ")
    end = text.index("    def _guard_structured_output")
    body = "class S:\n" + text[start:end]
    from typing import Any, Optional

    exec(compile(body, "inference_helpers", "exec"),
         {"Any": Any, "Optional": Optional}, ns)
    s = ns["S"]()
    s._settings = SimpleNamespace(backend_request_fault_detection=detection)
    return s


ENGINE_DEAD = (
    "vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue"
)


def test_known_fatal_signature_is_request_fault_on_first_failure():
    """One replica lost, not five."""
    s = _svc()
    exc = _Exc(ENGINE_DEAD)
    fp = s._error_fingerprint(exc)
    assert s._is_request_fault(exc, fp, None, None, backend_id=4) is True


@pytest.mark.parametrize(
    "body",
    ["enum array must not be empty", "xgrammar compile failure",
     "json_schema_converter.cc:3363", "EngineDeadError"],
)
def test_all_incident_signatures_detected(body):
    s = _svc()
    exc = _Exc(body)
    assert s._is_request_fault(exc, s._error_fingerprint(exc), None, None, 9) is True


def test_ordinary_5xx_is_still_a_backend_fault_and_still_retried():
    """Must not over-trigger: a plain 500 should keep failing over."""
    s = _svc()
    exc = _Exc("500 Internal Server Error: upstream connection reset")
    assert s._is_request_fault(exc, s._error_fingerprint(exc), None, None, 4) is False


def test_same_error_on_a_different_backend_is_a_request_fault():
    """The general rule: two GPUs don't fail identically on the same input."""
    s = _svc()
    # Body must be distinctive (>= _MIN_FAULT_BODY_CHARS) — see
    # test_bare_500s_are_not_fingerprint_matched for why.
    exc = _Exc("some novel fatal error with a distinctive traceback body")
    fp = s._error_fingerprint(exc)
    # first failure on backend 4 -> backend fault, retried
    assert s._is_request_fault(exc, fp, None, None, 4) is False
    # identical failure on backend 9 -> request fault, stop
    assert s._is_request_fault(exc, fp, fp, 4, 9) is True


def test_same_error_on_the_SAME_backend_is_not_a_request_fault():
    """A flapping single backend must still be retried elsewhere."""
    s = _svc()
    exc = _Exc("some novel fatal error with a distinctive traceback body")
    fp = s._error_fingerprint(exc)
    assert s._is_request_fault(exc, fp, fp, 4, 4) is False


def test_fingerprint_ignores_ids_and_timestamps():
    """pids/timestamps differ per backend; the signature must still match."""
    s = _svc()
    a = _Exc("EngineCore pid=12345 died at 10:20:29")
    b = _Exc("EngineCore pid=98765 died at 10:20:31")
    assert s._error_fingerprint(a) == s._error_fingerprint(b)


def test_detection_can_be_disabled():
    s = _svc(detection=False)
    exc = _Exc(ENGINE_DEAD)
    assert s._is_request_fault(exc, s._error_fingerprint(exc), None, None, 4) is False


def test_request_fault_does_not_charge_the_circuit_breaker():
    """The backend didn't misbehave — charging it is what opened all five."""
    src = (
        pathlib.Path(__file__).resolve().parents[2] / "services" / "inference.py"
    ).read_text()
    for marker in ("backend_5xx_request_fault", "stream_backend_5xx_request_fault"):
        i = src.index(marker)
        block = src[i : src.index("raise HTTPException", i)]
        assert "report_live_failure" not in block, marker
        assert "on_job_failed" in block, marker  # the slot must still be released


def test_both_retry_paths_classify():
    src = (
        pathlib.Path(__file__).resolve().parents[2] / "services" / "inference.py"
    ).read_text()
    assert src.count("self._is_request_fault(") == 2


def test_bare_500s_are_not_fingerprint_matched():
    """An empty 500 is the generic failure shape, not a distinctive signature.

    Matching on it would turn an ordinary fleet-wide outage into a bogus
    "your request is invalid" and would stop the breaker from ever opening.
    """
    s = _svc()
    exc = _Exc("")
    fp = s._error_fingerprint(exc)
    assert s._is_request_fault(exc, fp, fp, 4, 9) is False


def test_short_generic_bodies_are_not_fingerprint_matched():
    s = _svc()
    exc = _Exc("Internal Server Error")   # < _MIN_FAULT_BODY_CHARS
    fp = s._error_fingerprint(exc)
    assert s._is_request_fault(exc, fp, fp, 4, 9) is False


def test_distinctive_body_still_matches_across_backends():
    s = _svc()
    exc = _Exc("Traceback: some long and highly distinctive fatal error text here")
    fp = s._error_fingerprint(exc)
    assert s._is_request_fault(exc, fp, fp, 4, 9) is True
