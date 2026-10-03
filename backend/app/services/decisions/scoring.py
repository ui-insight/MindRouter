############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# services/decisions/scoring.py: prompt layout and label arithmetic
# for typed decisions (EXPERIMENTAL, transitional)
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Pure functions shared by every decision backend.

The prompt layout and the option-order averaging follow the "separate" mode of
open-alternative-jev (github.com/ikermoel/open-alternative-jev, Apache-2.0):
one ordinary user turn per question, the shared state written first so a
serving engine's prefix cache reuses it, options addressed by single-token
letters, and the answer read from the next-token distribution restricted to
those letters. The instruction strings are kept byte-identical to that
library's so that its published benchmarks stay comparable.

Nothing here touches a model, a database or the network.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Byte-identical to so1.prompting.DEFAULT_INSTRUCTION.
INSTRUCTION = "Choose the correct option. Reply with only its letter."


def render_turn(question: str, options: Sequence[str], state: str | None) -> str:
    """The single user turn that scores one question (so1 ``render_turn``,
    first=True). The state precedes the question so every question about the
    same state shares the longest possible prefix."""
    lines = [INSTRUCTION]
    if state is not None:
        lines += ["", "Context:", state]
    lines += ["", f"Question: {question}"]
    lines += [f"{LETTERS[i]}. {option}" for i, option in enumerate(options)]
    return "\n".join(lines)


def option_orders(n: int, permutations: int) -> list[list[int]]:
    """Deterministic option orderings: the original, then the reversal, then
    cyclic shifts (so1 ``option_orders``). ``orders[p][pos]`` is the original
    option index shown at position ``pos`` in view ``p``."""
    orders = [list(range(n))]
    if permutations >= 2 and n >= 2:
        orders.append(list(reversed(range(n))))
    shift = 1
    while len(orders) < permutations and shift < n:
        order = [(i + shift) % n for i in range(n)]
        if order not in orders:
            orders.append(order)
        shift += 1
    return orders


def softmax(scores: Sequence[float]) -> list[float]:
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    z = sum(exps)
    return [e / z for e in exps]


def apply_temperature(probs: Sequence[float], temperature: float) -> list[float]:
    """Sharpen (T < 1) or soften (T > 1) a label distribution: p_i ** (1/T),
    renormalized. This is the usual temperature scaling applied to the
    distribution itself, so it also works on one averaged over option orders.
    The ranking never changes; only how confident the numbers are."""
    if temperature == 1.0 or len(probs) < 2:
        return [float(p) for p in probs]
    powered = [p ** (1.0 / temperature) if p > 0 else 0.0 for p in probs]
    z = sum(powered)
    return [p / z for p in powered] if z > 0 else [float(p) for p in probs]


@dataclass
class LabelReadout:
    """What one scoring call returned for one presented option order.

    ``logprobs[pos]`` is the raw log-probability of the label at presented
    position ``pos``; ``sampled`` is the presented position the constrained
    sampler selected (None if the backend did not report it); ``complete`` is
    False when any label value had to be floored because the backend did not
    return it.
    """

    logprobs: list[float]
    sampled: int | None
    complete: bool = True


@dataclass
class Combined:
    answer_index: int
    likelihoods: list[float]  # original option order, sums to 1
    logprobs: list[float]     # original option order, from the first view
    label_mass: float
    complete: bool


def combine(n: int, orders: Sequence[Sequence[int]], readouts: Sequence[LabelReadout]) -> Combined:
    """Average the per-view label distributions back into the original option
    order (so1 ``Decider.decide_many``).

    With one view the answer is the label the sampler selected, which is exact
    even when a floored value made the distribution approximate. With several
    views the answer is the argmax of the averaged likelihoods, as in so1.
    """
    if len(orders) != len(readouts) or not readouts:
        raise ValueError("one readout per option order is required")
    acc = [0.0] * n
    for order, readout in zip(orders, readouts, strict=True):
        if len(readout.logprobs) != n:
            raise ValueError("readout length does not match the option count")
        probs = softmax(readout.logprobs)
        for pos, orig in enumerate(order):
            acc[orig] += probs[pos]
    likelihoods = [v / len(readouts) for v in acc]

    first_order, first = orders[0], readouts[0]
    logprobs = [0.0] * n
    for pos, orig in enumerate(first_order):
        logprobs[orig] = first.logprobs[pos]
    label_mass = min(1.0, sum(math.exp(v) for v in logprobs))

    if len(readouts) == 1 and first.sampled is not None:
        answer_index = first_order[first.sampled]
    else:
        answer_index = max(range(n), key=likelihoods.__getitem__)

    return Combined(
        answer_index=answer_index,
        likelihoods=likelihoods,
        logprobs=logprobs,
        label_mass=label_mass,
        complete=all(r.complete for r in readouts),
    )
