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

Extensions (ignored by Jev clients): optional request field ``permutations``
(1 or 2), and response fields ``id`` and ``metadata``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

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
    permutations: int = Field(default=1, ge=1, le=MAX_PERMUTATIONS)  # extension

    model_config = ConfigDict(extra="ignore")


class SystemOneValidationError(Exception):
    """A request MindRouter cannot serve; ``detail`` is FastAPI's 422 list."""

    def __init__(self, detail: list[dict[str, Any]]):
        super().__init__(detail[0]["msg"] if detail else "invalid request")
        self.detail = detail


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
    options: list[str] = field(default_factory=list)    # option texts shown, same order
    legend: dict[str, Any] | None = None                # score only: "0" -> level as sent


@dataclass
class Plan:
    model_requested: str | None
    state: str
    questions: list[PlannedQuestion]
    decision_request: DecisionRequest | None   # None when nothing needs the model


def validate_wire(body: Any) -> SystemOneRequestIn:
    """Check a body against the System One wire shape. This is all an
    upstream decision server needs; ``compile_plan`` adds what letter scoring
    on a chat model needs. Raises SystemOneValidationError (422)."""
    if not isinstance(body, dict):
        raise _err([], "Input should be a valid dictionary", "dict_type")
    try:
        return SystemOneRequestIn.model_validate(body)
    except ValidationError as e:
        raise SystemOneValidationError([
            {"loc": ["body", *err["loc"]], "msg": err["msg"], "type": err["type"]} for err in e.errors()
        ]) from None


def parse_request(body: Any) -> Plan:
    """validate_wire + compile_plan."""
    return compile_plan(validate_wire(body))


def compile_plan(wire: SystemOneRequestIn) -> Plan:
    """Compile a validated request for one-token letter scoring on a chat
    model. Raises SystemOneValidationError (422) for what that method cannot
    express (more than MAX_OPTIONS choices, duplicate or oversized options)."""
    state = render(wire.state)
    planned: list[PlannedQuestion] = []
    internal: list[BooleanQuestion | ChoiceQuestion] = []

    for jev_id, q in wire.questions.items():
        loc = ["questions", jev_id]
        instructions = render(q.instructions) if q.instructions is not None else _DEFAULT_INSTRUCTIONS[q.type]
        if not instructions.strip():
            raise _err([*loc, "instructions"], "instructions must not be empty")
        internal_id = f"q{len(internal)}"

        if isinstance(q, NoulQuestionIn):
            question = instructions
            if q.criteria is not None:
                if q.criteria.true is not None:
                    question += f"\nYes means: {render(q.criteria.true)}"
                if q.criteria.false is not None:
                    question += f"\nNo means: {render(q.criteria.false)}"
            _check_len(question, [*loc, "instructions"])
            internal.append(BooleanQuestion(id=internal_id, type="boolean", question=question))
            planned.append(PlannedQuestion(jev_id, "noul", internal_id, keys=["true", "false"],
                                           options=["yes", "no"]))
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
        internal.append(ChoiceQuestion(id=internal_id, type="choice", question=instructions, options=options))
        planned.append(PlannedQuestion(jev_id, q.type, internal_id, keys=keys, options=options, legend=legend))

    request = (
        DecisionRequest(model=wire.model, state=state, questions=internal, permutations=wire.permutations)
        if internal else None
    )
    return Plan(model_requested=wire.model, state=state, questions=planned, decision_request=request)


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
) -> dict[str, Any]:
    by_id = {r.id: r for r in results}
    answers: dict[str, Any] = {}
    quality: dict[str, Any] = {}
    for pq in plan.questions:
        if pq.internal_id is None:
            probs = [1.0]
        else:
            r = by_id[pq.internal_id]
            probs = [float(r.likelihoods[o]) for o in pq.options]
            quality[pq.jev_id] = {"complete": r.complete, "label_mass": r.label_mass}

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
            "questions": quality,
        },
    }
