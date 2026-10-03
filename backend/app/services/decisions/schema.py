############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# services/decisions/schema.py: request/response contract for
# POST /v1/decisions (EXPERIMENTAL, transitional)
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Typed decision requests and answers.

This is the only module the rest of MindRouter should import from the
decisions package besides ``get_decision_backend`` / ``get_decisions_config``.
It deliberately knows nothing about how answers are produced.

Every limit here is a hard ceiling that no admin setting can raise. They bound
the GPU work a single request can trigger: at most MAX_QUESTIONS x
MAX_PERMUTATIONS one-token scoring calls, each over at most
MAX_STATE_CHARS + MAX_QUESTION_CHARS characters of prompt.
"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

MAX_QUESTIONS = 32
MAX_OPTIONS = 20          # one single-token letter label per option (A..T)
MAX_PERMUTATIONS = 2      # 1 = options as given, 2 = also reversed and averaged
MAX_STATE_CHARS = 64_000  # absolute ceiling; decisions.max_state_chars lowers it
MAX_QUESTION_CHARS = 4_000   # Noul criteria are appended to the question text
MAX_OPTION_CHARS = 1_000     # a Choice option is "name: description"
MAX_SCALE_LEVELS = MAX_OPTIONS

BOOLEAN_OPTIONS = ("yes", "no")

_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")

# What the numbers in a decision mean. Returned verbatim on every response so a
# client never has to guess. See docs/decisions-api.md "Score semantics".
SCORE_SEMANTICS = "normalized_label_likelihood"


class BooleanQuestion(BaseModel):
    id: str
    type: Literal["boolean"]
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    # Option orders scored for THIS question; None uses the request's value.
    permutations: int | None = Field(default=None, ge=1, le=MAX_PERMUTATIONS)

    @property
    def options(self) -> tuple[str, ...]:
        return BOOLEAN_OPTIONS


class ChoiceQuestion(BaseModel):
    id: str
    type: Literal["choice"]
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    options: list[str] = Field(min_length=2, max_length=MAX_OPTIONS)
    permutations: int | None = Field(default=None, ge=1, le=MAX_PERMUTATIONS)

    @field_validator("options")
    @classmethod
    def _options_ok(cls, options: list[str]) -> list[str]:
        cleaned = [o.strip() for o in options]
        if any(not o for o in cleaned):
            raise ValueError("options must be non-empty strings")
        if any(len(o) > MAX_OPTION_CHARS for o in cleaned):
            raise ValueError(f"options must be at most {MAX_OPTION_CHARS} characters")
        if any("\n" in o for o in cleaned):
            raise ValueError("options must not contain newlines")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("options must be distinct")
        return cleaned


class ScaleQuestion(BaseModel):
    """An ordered integer rubric, e.g. severity 1..5. Scored as a choice over
    the integer labels; the response adds the likelihood-weighted mean."""

    id: str
    type: Literal["scale"]
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    min: int
    max: int
    permutations: int | None = Field(default=None, ge=1, le=MAX_PERMUTATIONS)

    @model_validator(mode="after")
    def _range_ok(self) -> ScaleQuestion:
        if self.max <= self.min:
            raise ValueError("scale max must be greater than min")
        if self.max - self.min + 1 > MAX_SCALE_LEVELS:
            raise ValueError(f"scale spans at most {MAX_SCALE_LEVELS} levels")
        return self

    @property
    def options(self) -> tuple[str, ...]:
        return tuple(str(i) for i in range(self.min, self.max + 1))


Question = BooleanQuestion | ChoiceQuestion | ScaleQuestion


class DecisionRequest(BaseModel):
    model: str | None = None
    state: str | None = Field(default=None, max_length=MAX_STATE_CHARS)
    questions: list[Question] = Field(min_length=1, max_length=MAX_QUESTIONS)
    permutations: int = Field(default=1, ge=1, le=MAX_PERMUTATIONS)

    @field_validator("questions")
    @classmethod
    def _ids_ok(cls, questions: list[Question]) -> list[Question]:
        seen: set[str] = set()
        for q in questions:
            if not _ID_RE.match(q.id):
                raise ValueError(
                    f"question id {q.id!r} must match [A-Za-z0-9_.-]{{1,64}}"
                )
            if q.id in seen:
                raise ValueError(f"duplicate question id {q.id!r}")
            seen.add(q.id)
        return questions


class DecisionResult(BaseModel):
    """One answered question.

    ``likelihoods`` are the model's next-token log-probabilities of the option
    labels, exponentiated and renormalized to sum to 1 over the options. They
    are NOT calibrated probabilities. ``logprobs`` are the raw per-label values
    (natural log, over the model's full vocabulary) and ``label_mass`` is their
    exponentiated sum: how much of the model's next-token mass fell on any
    option label at all. A low ``label_mass`` means the model did not want to
    answer with a letter and the decision is weakly grounded.

    ``answer`` is the label the constrained sampler actually selected, which is
    the exact argmax among the options. When ``complete`` is false at least one
    label's log-probability was not returned by the backend and was floored, so
    ``likelihoods`` are approximate (``answer`` is still exact).
    """

    id: str
    type: Literal["boolean", "choice", "scale"]
    answer: bool | str | int
    likelihoods: dict[str, float]
    logprobs: dict[str, float]
    label_mass: float
    complete: bool = True
    # type-specific conveniences
    likelihood: float | None = None       # choice: likelihood of the answer
    likelihood_true: float | None = None  # boolean
    expected_value: float | None = None   # scale: likelihood-weighted mean


class DecisionUsage(BaseModel):
    prompt_tokens: int
    scoring_tokens: int
    total_tokens: int
    cached_tokens: int | None = None  # prefix-cache hits reported by vLLM
    backend_calls: int
