############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# services/decisions/systemone.py: TypeSafe System One wire
# format (Jev's POST /v1/systemone) onto the decisions pipeline
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""TypeSafe's System One request/response shape, mapped onto decisions.

The wire contract is TypeSafe's own: https://docs.typesafe.ai/api.md and the
OpenAPI-generated models in their Python SDK (``typesafe-sdk`` 0.7.2,
``typesafe_sdk/_schemas/models.py``). A client written for Jev — including the
official SDK with ``TYPESAFE_BASE_URL`` pointed at MindRouter and a MindRouter
key as ``TYPESAFE_API_KEY`` — sends the same body and reads the same answers.

What is the same as Jev:
  * request ``{state, model, questions: {id: question}}``; question types
    ``noul`` / ``choice`` / ``score`` with ``instructions`` and ``criteria``;
    ``instructions``, option descriptions and score levels may be strings,
    objects or arrays;
  * response ``{model, answers: {id: answer}, usage: {input_tokens,
    output_tokens}}`` with Jev's answer fields, and ``confidence`` computed
    with the formulas TypeSafe documents (https://docs.typesafe.ai/confidence.md);
  * 422 bodies in the FastAPI ``{"detail": [{"loc": ["body", ...], ...}]}``
    form the SDK parses.

What is NOT the same (a model is not Jev just because the wire is):
  * the numbers come from one-token letter scoring on a vLLM chat model, not
    from Jev's calibrated decision model; ``metadata.score_semantics`` says so;
  * a ``choice`` takes at most ``MAX_OPTIONS`` (20) options here, not 255,
    because each option needs its own single-token letter;
  * ``model`` must be ``jev-latest`` / ``jev-preview`` (mapped to the admin's
    default decisions model) or a model on the decisions allow-list; the
    response's ``model`` names the model that actually answered.

Extensions (ignored by Jev clients): optional request fields ``permutations``
(1 or 2; omitted = the server's per-type defaults) and ``images`` (Cloudflare
Clef's extension: up to 4 base64 PNG/JPEG/WebP images shown before the state,
for models that can see), and response fields ``id`` and ``metadata``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .images import ImageError, normalize_images
from .scoring import apply_temperature
from .schema import (
    MAX_OPTION_CHARS,
    MAX_OPTIONS,
    MAX_PERMUTATIONS,
    MAX_QUESTION_CHARS,
    MAX_QUESTIONS,
    SCORE_SEMANTICS,
    BooleanQuestion,
    ChoiceQuestion,
    DecisionRequest,
    DecisionResult,
    DecisionUsage,
)

# Names TypeSafe's SDK and docs send in ``model`` (the SDK default is
# ``jev-latest``). They mean "this server's default decisions model".
JEV_MODEL_ALIASES = ("jev-latest", "jev-preview")

# TypeSafe caps a Score at 10 levels.
MAX_SCORE_LEVELS = 10

# Defaults for letter scoring on a vLLM chat model, per question type. Both are
# admin settings (decisions.permutations, decisions.temperature); these values
# were chosen on the training split of the LocalLLaMA/typed-decisions benchmark
# with qwen3.8-27b and checked once on its test split (accuracy 0.687 -> 0.710,
# calibration error 0.090 -> 0.022). See docs/decisions-api.md "Benchmarks".
#
# permutations: how many option orders are scored and averaged. Reversing a
# choice's options and averaging cancels position bias and is where the
# accuracy gain is (0.643 -> 0.718); it does not help yes/no or score questions,
# so they stay at one call.
DEFAULT_PERMUTATIONS = {"noul": 1, "choice": 2, "score": 1}
# temperature: softens the model's overconfident label distribution (> 1).
# It never changes which answer is chosen, only the probabilities reported.
DEFAULT_TEMPERATURE = {"noul": 1.35, "choice": 1.05, "score": 1.45}
QUESTION_TYPES = ("noul", "choice", "score")
MIN_TEMPERATURE, MAX_TEMPERATURE = 0.2, 5.0


def parse_permutations(raw: Any) -> tuple[dict[str, int], list[str]]:
    """decisions.permutations -> {type: 1..MAX_PERMUTATIONS}; unset types keep their default."""
    out, problems = dict(DEFAULT_PERMUTATIONS), []
    if raw in (None, "", {}):
        return out, problems
    if not isinstance(raw, dict):
        return out, ["permutations must be a JSON object such as {\"choice\": 2}"]
    for key, value in raw.items():
        if key not in QUESTION_TYPES:
            problems.append(f"permutations: unknown question type {key!r} (use noul, choice, score)")
        elif isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_PERMUTATIONS:
            problems.append(f"permutations.{key} must be a whole number from 1 to {MAX_PERMUTATIONS}")
        else:
            out[key] = value
    return out, problems


def parse_temperature(raw: Any) -> tuple[dict[str, float], list[str]]:
    """decisions.temperature -> {type: float}; unset types keep their default."""
    out, problems = dict(DEFAULT_TEMPERATURE), []
    if raw in (None, "", {}):
        return out, problems
    if not isinstance(raw, dict):
        return out, ["temperature must be a JSON object such as {\"noul\": 1.35}"]
    for key, value in raw.items():
        if key not in QUESTION_TYPES:
            problems.append(f"temperature: unknown question type {key!r} (use noul, choice, score)")
        elif isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not MIN_TEMPERATURE <= value <= MAX_TEMPERATURE:
            problems.append(f"temperature.{key} must be a number from {MIN_TEMPERATURE} to {MAX_TEMPERATURE}")
        else:
            out[key] = float(value)
    return out, problems


# Ceiling on the questions block forwarded to an upstream server, as JSON
# characters. Letter scoring bounds each question and option separately; an
# upstream gets the caller's questions as written, so the total is bounded.
MAX_FORWARDED_QUESTIONS_CHARS = 256_000

# Used only when a question arrives without instructions, which TypeSafe's
# OpenAPI schema allows (``instructions`` is nullable).
_DEFAULT_INSTRUCTIONS = {
    "noul": "Is this true?",
    "choice": "Which option applies?",
    "score": "Which level applies?",
}

JSONContent = str | dict[str, Any] | list[Any]


# ---------------------------------------------------------------------------
# Wire models: field names, types and nullability follow the SDK's
# OpenAPI-generated models exactly. Validation errors become FastAPI-style 422s.
# ---------------------------------------------------------------------------

class _Wire(BaseModel):
    model_config = ConfigDict(extra="ignore")


class NoulCriteria(_Wire):
    true: JSONContent | None = None
    false: JSONContent | None = None


class NoulQuestionIn(_Wire):
    type: Literal["noul"]
    instructions: JSONContent | None = None
    criteria: NoulCriteria | None = None


class ChoiceQuestionIn(_Wire):
    type: Literal["choice"]
    instructions: JSONContent | None = None
    criteria: dict[str, JSONContent | None] = Field(min_length=1)


class ScoreQuestionIn(_Wire):
    type: Literal["score"]
    instructions: JSONContent | None = None
    criteria: list[JSONContent] = Field(min_length=1)


# Discriminated on ``type`` like TypeSafe's schema, so a 422 names the question
# type in its path: ["body", "questions", "urgency", "score", "criteria"].
QuestionIn = Annotated[NoulQuestionIn | ChoiceQuestionIn | ScoreQuestionIn, Field(discriminator="type")]


class SystemOneRequestIn(_Wire):
    state: JSONContent
    # Required by Jev; tolerated when missing here (the default model answers).
    model: str | None = None
    questions: dict[str, QuestionIn] = Field(min_length=1, max_length=MAX_QUESTIONS)
    # Extension. Omitted: the server's per-type defaults (decisions.permutations).
    # Given: that many option orders for every question in the request.
    permutations: int | None = Field(default=None, ge=1, le=MAX_PERMUTATIONS)
    # Cloudflare Clef's extension: base64 images placed before the state. Checked
    # and normalized to data URLs by validate_wire (see images.py).
    images: list[Any] | None = None

    model_config = ConfigDict(extra="ignore")


class SystemOneValidationError(Exception):
    """A request MindRouter cannot serve; ``detail`` is FastAPI's 422 list."""

    def __init__(self, detail: list[dict[str, Any]], audit_message: str | None = None):
        super().__init__(detail[0]["msg"] if detail else "invalid request")
        self.detail = detail
        # What may be written to the audit row; never text that can quote the request.
        self.audit_message = audit_message or "request validation failed"


def _err(loc: list[str | int], msg: str, kind: str = "value_error") -> SystemOneValidationError:
    return SystemOneValidationError([{"loc": ["body", *loc], "msg": msg, "type": kind}])


# ---------------------------------------------------------------------------
# Request: wire -> DecisionRequest + a plan for formatting the answers
# ---------------------------------------------------------------------------

def render(value: JSONContent) -> str:
    """Text the model sees for a state, instruction, option or level.
    Strings pass through; structure is shown as compact JSON in the order the
    caller wrote it."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _one_line(text: str) -> str:
    # Options are listed one per line after their letter; keep each on one.
    return " ".join(text.split())


@dataclass
class PlannedQuestion:
    jev_id: str
    kind: Literal["noul", "choice", "score"]
    internal_id: str | None            # None: answered without the model (one option)
    keys: list[str] = field(default_factory=list)       # answer keys in option order
    permutations: int = 1                               # option orders scored for this question
    options: list[str] = field(default_factory=list)    # option texts shown, same order
    legend: dict[str, Any] | None = None                # score only: "0" -> level as sent


@dataclass
class Plan:
    model_requested: str | None
    state: str
    questions: list[PlannedQuestion]
    decision_request: DecisionRequest | None   # None when nothing needs the model
    images: list[str] = field(default_factory=list)


def validate_wire(body: Any) -> SystemOneRequestIn:
    """Check a body against the System One wire shape. This is all an
    upstream decision server needs; ``compile_plan`` adds what letter scoring
    on a chat model needs. Raises SystemOneValidationError (422)."""
    if not isinstance(body, dict):
        raise _err([], "Input should be a valid dictionary", "dict_type")
    # Python's JSON parser admits two things that cannot be sent on: lone
    # surrogates (``"\ud800"``), which have no UTF-8 encoding, and NaN /
    # Infinity. Refuse them here, where it is the caller's 422, instead of
    # letting them fail later as a backend error or an unhandled exception.
    try:
        json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except UnicodeEncodeError:
        raise _err([], "Request contains text that is not valid Unicode (an unpaired surrogate)") from None
    except ValueError:
        raise _err([], "Request contains a number that is not finite (NaN or Infinity)") from None
    try:
        wire = SystemOneRequestIn.model_validate(body)
    except ValidationError as e:
        raise _from_pydantic(e) from None
    try:
        wire.images = normalize_images(wire.images)
    except ImageError as e:
        raise _err(["images"] if e.index is None else ["images", e.index], str(e)) from None
    return wire


def _from_pydantic(e: ValidationError, prefix: tuple = ()) -> SystemOneValidationError:
    """FastAPI-shaped details from a pydantic error: location, message and
    type only. The offending input value is deliberately left out; it is the
    caller's state or question text."""
    return SystemOneValidationError([
        {"loc": ["body", *prefix, *err["loc"]], "msg": err["msg"], "type": err["type"]} for err in e.errors()
    ])


def parse_request(body: Any, permutations: dict[str, int] | None = None) -> Plan:
    """validate_wire + compile_plan."""
    return compile_plan(validate_wire(body), permutations)


def compile_plan(wire: SystemOneRequestIn, permutations: dict[str, int] | None = None) -> Plan:
    """Compile a validated request for one-token letter scoring on a chat
    model. ``permutations`` is the per-type default (decisions.permutations);
    a request that names ``permutations`` itself overrides it for every
    question. Raises SystemOneValidationError (422) for what letter scoring
    cannot express (more than MAX_OPTIONS choices, duplicate or oversized
    options)."""
    try:
        return _compile_plan(wire, permutations or DEFAULT_PERMUTATIONS)
    except ValidationError as e:
        # The internal models have limits of their own (lengths, option rules).
        # Anything they refuse is a 422, never an unhandled error whose text
        # would carry the request into the logs.
        raise _from_pydantic(e) from None


def _compile_plan(wire: SystemOneRequestIn, default_permutations: dict[str, int]) -> Plan:
    state = render(wire.state)
    planned: list[PlannedQuestion] = []
    internal: list[BooleanQuestion | ChoiceQuestion] = []

    for jev_id, q in wire.questions.items():
        loc = ["questions", jev_id]
        instructions = render(q.instructions) if q.instructions is not None else _DEFAULT_INSTRUCTIONS[q.type]
        if not instructions.strip():
            raise _err([*loc, "instructions"], "instructions must not be empty")
        internal_id = f"q{len(internal)}"
        perms = wire.permutations if wire.permutations is not None else default_permutations.get(q.type, 1)

        if isinstance(q, NoulQuestionIn):
            question = instructions
            if q.criteria is not None:
                if q.criteria.true is not None:
                    question += f"\nYes means: {render(q.criteria.true)}"
                if q.criteria.false is not None:
                    question += f"\nNo means: {render(q.criteria.false)}"
            _check_len(question, [*loc, "instructions"])
            internal.append(BooleanQuestion(id=internal_id, type="boolean", question=question, permutations=perms))
            planned.append(PlannedQuestion(jev_id, "noul", internal_id, keys=["true", "false"],
                                           options=["yes", "no"], permutations=perms))
            continue

        if isinstance(q, ChoiceQuestionIn):
            keys = list(q.criteria)
            if len(keys) > MAX_OPTIONS:
                raise _err([*loc, "criteria"],
                           f"this server scores at most {MAX_OPTIONS} choice options (got {len(keys)})")
            options = [
                _one_line(name if desc is None else f"{name}: {render(desc)}")
                for name, desc in q.criteria.items()
            ]
            legend = None
        else:  # score
            if len(q.criteria) > MAX_SCORE_LEVELS:
                raise _err([*loc, "criteria"], f"a score accepts at most {MAX_SCORE_LEVELS} levels")
            keys = [str(i) for i in range(len(q.criteria))]
            options = [_one_line(render(level)) for level in q.criteria]
            legend = dict(zip(keys, q.criteria, strict=True))

        for i, text in enumerate(options):
            if not text:
                raise _err([*loc, "criteria", keys[i]], "options must not be empty")
            if len(text) > MAX_OPTION_CHARS:
                raise _err([*loc, "criteria", keys[i]],
                           f"option text must be at most {MAX_OPTION_CHARS} characters")
        if len(set(options)) != len(options):
            raise _err([*loc, "criteria"], "options must be distinct")

        if len(options) == 1:
            # Nothing to decide: probability 1 without a model call.
            planned.append(PlannedQuestion(jev_id, q.type, None, keys=keys, options=options, legend=legend))
            continue
        _check_len(instructions, [*loc, "instructions"])
        internal.append(ChoiceQuestion(id=internal_id, type="choice", question=instructions, options=options,
                                       permutations=perms))
        planned.append(PlannedQuestion(jev_id, q.type, internal_id, keys=keys, options=options, legend=legend,
                                       permutations=perms))

    request = (
        DecisionRequest(model=wire.model, state=state, questions=internal,  # permutations are per question
                        images=list(wire.images or []))
        if internal else None
    )
    return Plan(model_requested=wire.model, state=state, questions=planned, decision_request=request,
                images=list(wire.images or []))


def _check_len(text: str, loc: list[str | int]) -> None:
    if len(text) > MAX_QUESTION_CHARS:
        raise _err(loc, f"question text must be at most {MAX_QUESTION_CHARS} characters")


# ---------------------------------------------------------------------------
# GET /v1/models for TypeSafe's SDK (client.models.list())
# ---------------------------------------------------------------------------

# TypeSafe's ModelMetadata requires a YYYY-MM-DD release_date; this is the date
# the System One surface first shipped in MindRouter.
LISTED_SINCE = "2026-10-03"


def typesafe_model_list(cfg: dict) -> dict[str, Any]:
    """The names a System One request may send, in TypeSafe's list shape
    ``{"models": [{name, description, release_date}]}``. Empty when the
    decisions API is off."""
    if not cfg.get("enabled"):
        return {"models": []}
    default = cfg["default_model"]
    entries = [
        {"name": alias, "description": f"Alias for {default} (MindRouter decisions; not TypeSafe Jev)",
         "release_date": LISTED_SINCE}
        for alias in JEV_MODEL_ALIASES
    ]
    entries += [
        {"name": m, "description": "vLLM chat model scored by one-token label likelihood",
         "release_date": LISTED_SINCE}
        for m in cfg["allowed_models"]
    ]
    entries += [
        {"name": name, "description": "Decision model served by its own System One server",
         "release_date": LISTED_SINCE}
        for name in cfg.get("upstreams", {})
    ]
    return {"models": entries}


# ---------------------------------------------------------------------------
# Response: decisions results -> Jev answers
# ---------------------------------------------------------------------------

def choice_confidence(probs: list[float]) -> float:
    """TypeSafe's Choice confidence: how far the top probability sits above
    uniform, rescaled to [0, 1] (https://docs.typesafe.ai/confidence.md)."""
    n = len(probs)
    if n <= 1:
        return 1.0
    total = sum(probs)
    peak = max(probs) / total if total else 1.0 / n
    return max(0.0, min(1.0, (n * peak - 1) / (n - 1)))


def score_confidence(probs: list[float]) -> float:
    """TypeSafe's Score confidence: 1 minus the expected distance from the
    modal level, relative to that of a uniform distribution."""
    n = len(probs)
    if n <= 1:
        return 1.0
    total = sum(probs)
    p = [v / total for v in probs] if total else [1.0 / n] * n
    peak = max(range(n), key=p.__getitem__)
    spread = sum(v * abs(i - peak) for i, v in enumerate(p))
    even = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
    return max(0.0, min(1.0, 1 - spread / even))


def format_response(
    plan: Plan,
    results: list[DecisionResult],
    usage: DecisionUsage | None,
    *,
    model: str,
    request_id: str,
    backend_name: str,
    temperature: dict[str, float] | None = None,
) -> dict[str, Any]:
    """``temperature`` is the per-type setting (decisions.temperature); None
    reports the model's label distribution as scored."""
    by_id = {r.id: r for r in results}
    answers: dict[str, Any] = {}
    quality: dict[str, Any] = {}
    applied: dict[str, float] = {}
    for pq in plan.questions:
        if pq.internal_id is None:
            probs = [1.0]
        else:
            r = by_id[pq.internal_id]
            probs = [float(r.likelihoods[o]) for o in pq.options]
            if temperature:
                t = float(temperature.get(pq.kind, 1.0))
                probs = apply_temperature(probs, t)
                applied[pq.kind] = t
            quality[pq.jev_id] = {"complete": r.complete, "label_mass": r.label_mass,
                                  "permutations": pq.permutations}

        if pq.kind == "noul":
            answers[pq.jev_id] = {"type": "noul", "noul": probs[0]}  # options are ("yes", "no")
            continue
        distribution = dict(zip(pq.keys, probs, strict=True))
        if pq.kind == "choice":
            best = max(range(len(probs)), key=probs.__getitem__)
            answers[pq.jev_id] = {
                "type": "choice",
                "choice": pq.keys[best],
                "probabilities": distribution,
                "confidence": choice_confidence(probs),
            }
        else:
            answers[pq.jev_id] = {
                "type": "score",
                "score": float(sum(i * p for i, p in enumerate(probs))),
                "legend": pq.legend,
                "probabilities": distribution,
                "confidence": score_confidence(probs),
            }

    prompt = usage.prompt_tokens if usage else 0
    cached = (usage.cached_tokens or 0) if usage else 0
    return {
        "model": model,
        "answers": answers,
        # Jev bills input tokens; here that is what the server had to compute
        # (prompt tokens not served from its prefix cache), which is also what
        # MindRouter charges against the caller's quota.
        "usage": {
            "input_tokens": max(0, prompt - cached),
            "output_tokens": usage.scoring_tokens if usage else 0,
        },
        "id": request_id,
        "metadata": {
            "score_semantics": SCORE_SEMANTICS,
            "backend": backend_name,
            "backend_calls": usage.backend_calls if usage else 0,
            "prompt_tokens": prompt,
            "cached_tokens": usage.cached_tokens if usage else None,
            # Temperature applied to each question type's probabilities (absent: none).
            "temperature": applied,
            "questions": quality,
        },
    }
