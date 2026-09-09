############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# schema_guard.py: reject JSON schemas that crash the inference engine
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Pre-flight validation for structured-output JSON schemas.

Guided decoding compiles the caller's JSON schema into a grammar inside the
inference engine. Some malformed-but-parseable schemas make that compiler
abort, and in vLLM the abort kills ``EngineCore`` — taking the whole worker
down, not just the request.

On 2026-09-09 a schema carrying ``"enum": []`` did exactly that::

    RuntimeError: json_schema_converter.cc:3363: enum array must not be empty
      -> xgrammar compile_json_schema -> EngineCore dies -> EngineDeadError
      -> vLLM shuts down -> systemd restarts it (~90s)

The gateway then retried the request on the next replica, and the next, so a
single caller took all five ``qwen/qwen3.8-27b`` workers down within one
second. The schema had been generated per-document, and an empty ``enum`` is
what a "list of this PDF's form fields" becomes when the PDF has none.

Validating costs microseconds and turns an engine kill into a 400. This is
the primary defence; refusing to re-dispatch a request that already killed a
backend (see InferenceService._proxy_with_retry) is the backstop.
"""

from typing import Any, List, Optional

# Bound the walk. Schemas are caller-supplied, so a pathological one must not
# be able to spend real CPU here or blow the Python stack.
MAX_DEPTH = 64
MAX_NODES = 20_000


class SchemaRejection(Exception):
    """A schema that must not be sent to a backend.

    ``path`` is a JSON-pointer-ish location so the caller can find the field.
    """

    def __init__(self, reason: str, path: str):
        self.reason = reason
        self.path = path
        super().__init__(f"{reason} (at {path or 'schema root'})")

    @property
    def message(self) -> str:
        loc = self.path or "the schema root"
        return (
            f"Invalid JSON schema in response_format: {self.reason} at {loc}. "
            f"This schema cannot be compiled into a decoding grammar."
        )


def _fail(reason: str, path: str) -> None:
    raise SchemaRejection(reason, path)


def validate_json_schema(schema: Any) -> None:
    """Raise SchemaRejection if the schema would crash grammar compilation.

    Deliberately conservative: it rejects only constructs that are known to be
    fatal or to admit no valid output at all. Anything merely unusual is left
    alone — a false rejection breaks a working caller, which is worse than
    letting a novel-but-harmless schema through to the (now non-propagating)
    retry path.
    """
    nodes = 0

    def walk(node: Any, path: str, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if depth > MAX_DEPTH:
            _fail(f"schema nests deeper than {MAX_DEPTH} levels", path)
        if nodes > MAX_NODES:
            _fail(f"schema exceeds {MAX_NODES} nodes", path)

        if isinstance(node, list):
            for i, item in enumerate(node):
                walk(item, f"{path}[{i}]", depth + 1)
            return
        if not isinstance(node, dict):
            return

        # --- the crash from 2026-09-09 -----------------------------------
        # xgrammar: "enum array must not be empty" -> EngineCore dies.
        if "enum" in node:
            enum = node["enum"]
            if isinstance(enum, list) and len(enum) == 0:
                _fail(
                    "'enum' is an empty array, which matches no value "
                    "(omit the key entirely if there are no choices)",
                    f"{path}.enum",
                )

        # Same class of defect: a constraint set that admits nothing.
        for key in ("oneOf", "anyOf", "allOf"):
            if key in node and isinstance(node[key], list) and len(node[key]) == 0:
                _fail(f"'{key}' is an empty array, which matches no value", f"{path}.{key}")

        # An array pinned to zero items alongside a required item schema is
        # contradictory, and pairs with the empty enum in the observed case.
        if node.get("type") == "array":
            mx, mn = node.get("maxItems"), node.get("minItems")
            if isinstance(mx, int) and mx < 0:
                _fail("'maxItems' is negative", f"{path}.maxItems")
            if isinstance(mn, int) and isinstance(mx, int) and mn > mx:
                _fail(f"'minItems' ({mn}) exceeds 'maxItems' ({mx})", path)

        if isinstance(node.get("required"), list) and node.get("properties") is not None:
            props = node["properties"]
            if isinstance(props, dict):
                missing = [
                    r for r in node["required"]
                    if isinstance(r, str) and r not in props
                ]
                if missing and node.get("additionalProperties") is False:
                    _fail(
                        "'required' names "
                        + ", ".join(repr(m) for m in missing[:3])
                        + " which are absent from 'properties' while "
                        "'additionalProperties' is false, so no value can match",
                        path,
                    )

        for key, value in node.items():
            if isinstance(value, (dict, list)):
                walk(value, f"{path}.{key}" if path else key, depth + 1)

    walk(schema, "", 0)


def extract_json_schema(response_format: Any) -> Optional[Any]:
    """Pull the schema out of an OpenAI-style ``response_format``.

    Returns None when the request is not doing schema-guided decoding.
    """
    if not isinstance(response_format, dict):
        return None
    if response_format.get("type") != "json_schema":
        return None
    js = response_format.get("json_schema")
    if not isinstance(js, dict):
        return None
    return js.get("schema")


def validate_response_format(response_format: Any) -> None:
    """Validate a request's ``response_format``; no-op when absent."""
    schema = extract_json_schema(response_format)
    if schema is not None:
        validate_json_schema(schema)


def validate_canonical_response_format(response_format: Any) -> None:
    """Validate a canonical request's ``response_format`` (pydantic or dict).

    Accepts the pydantic ``ResponseFormat`` (``.type`` / ``.json_schema``) as
    well as a plain dict, and tolerates both OpenAI shapes: ``json_schema``
    holding ``{"name": ..., "schema": {...}}`` or being the schema itself.
    No-op when the request is not doing schema-guided decoding.
    """
    if response_format is None:
        return

    if isinstance(response_format, dict):
        rf_type = response_format.get("type")
        js = response_format.get("json_schema")
    else:
        rf_type = getattr(response_format, "type", None)
        js = getattr(response_format, "json_schema", None)

    rf_type = getattr(rf_type, "value", rf_type)
    if rf_type != "json_schema" or not isinstance(js, dict):
        return

    # {"name": ..., "schema": {...}} vs the schema inline.
    schema = js.get("schema") if isinstance(js.get("schema"), (dict, list)) else js
    validate_json_schema(schema)
