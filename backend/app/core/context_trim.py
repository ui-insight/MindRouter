############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# context_trim.py: Drop oldest conversation turns to fit a
# model's context window (Responses API truncation:"auto").
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Server-side context truncation.

Implements the OpenAI Responses API ``truncation: "auto"`` semantics:
when the input exceeds the model's context window, drop items from the
start of the conversation (after any leading system messages) until it
fits, instead of failing with a context-length error.

Messages are dropped in *turn groups* so tool-call pairing is never
broken (an orphaned ``tool`` message without its assistant
``tool_calls`` message is rejected by chat backends):

- a ``user`` message starts a group,
- an ``assistant`` message plus any immediately-following ``tool``
  messages form one group.

The most recent group is never dropped.

Before any group is dropped, the reasoning carried by earlier assistant
turns (kept for models whose templates replay it) is removed from turns
before the latest user message — most templates never render it anyway.
Then whole groups go, oldest first; the current tool loop's reasoning,
which gpt-oss / Qwen / Gemma-style templates do replay, is stripped only
as a last resort. ``cap_max_tokens`` uses the reasoning step
(``strip_reasoning_to_fit``, both stages) on its own for every client API,
so a forwarded-reasoning session that has outgrown the context loses old
reasoning rather than failing.
"""

from typing import Callable, List, Tuple

from backend.app.core.canonical_schemas import CanonicalMessage, MessageRole


def _group_messages(
    messages: List[CanonicalMessage],
) -> Tuple[List[CanonicalMessage], List[List[CanonicalMessage]]]:
    """Split into (leading system messages, list of turn groups)."""
    head: List[CanonicalMessage] = []
    idx = 0
    while idx < len(messages) and messages[idx].role == MessageRole.SYSTEM:
        head.append(messages[idx])
        idx += 1

    groups: List[List[CanonicalMessage]] = []
    for msg in messages[idx:]:
        if msg.role == MessageRole.TOOL and groups:
            groups[-1].append(msg)
        elif msg.role == MessageRole.ASSISTANT and groups and (
            groups[-1][-1].role == MessageRole.ASSISTANT
        ):
            # Consecutive assistant messages stay together
            groups[-1].append(msg)
        else:
            groups.append([msg])
    return head, groups


def _message_tokens(
    msg: CanonicalMessage, estimator: Callable[[str], int]
) -> int:
    total = 4  # per-message overhead (role, separators)
    content = msg.content
    if isinstance(content, str):
        total += estimator(content)
    elif isinstance(content, list):
        for block in content:
            text = getattr(block, "text", None)
            if text:
                total += estimator(text)
    for tc in msg.tool_calls or []:
        total += estimator(tc.function.arguments or "")
        total += estimator(tc.function.name or "")
    if msg.reasoning:
        total += estimator(msg.reasoning)
    return total


def _total(messages: List[CanonicalMessage], estimator: Callable[[str], int]) -> int:
    return sum(_message_tokens(m, estimator) for m in messages)


def strip_reasoning_to_fit(
    messages: List[CanonicalMessage],
    budget_tokens: int,
    estimator: Callable[[str], int],
    include_current_loop: bool = True,
) -> List[CanonicalMessage]:
    """Remove assistant reasoning until the messages fit, oldest context first.

    Stage 1 strips reasoning from messages before the latest user message
    (earlier turns); stage 2, only if still over budget and
    ``include_current_loop``, strips the rest (the current tool loop, which
    gpt-oss / Qwen / Gemma-style templates replay). Never drops a message. Returns a NEW list when
    anything was stripped, the same object otherwise; the caller's messages
    are never mutated (copies via model_copy).
    """
    if not any(m.reasoning for m in messages):
        return messages
    total = _total(messages, estimator)
    if total <= budget_tokens:
        return messages
    last_user = max(
        (i for i, m in enumerate(messages) if m.role == MessageRole.USER), default=-1,
    )
    out = list(messages)
    stages = (last_user, len(out)) if include_current_loop else (last_user,)
    for stage_end in stages:
        for i in range(stage_end):
            m = out[i]
            if m.reasoning:
                total -= _message_tokens(m, estimator)
                out[i] = m.model_copy(update={"reasoning": None})
                total += _message_tokens(out[i], estimator)
        if total <= budget_tokens:
            break
    if all(a is b for a, b in zip(out, messages)):
        return messages
    return out


def trim_messages_to_fit(
    messages: List[CanonicalMessage],
    budget_tokens: int,
    estimator: Callable[[str], int],
) -> Tuple[List[CanonicalMessage], int]:
    """Drop oldest turn groups until the estimated total fits the budget.

    Earlier assistant reasoning goes first (``strip_reasoning_to_fit``: older
    turns, then the current loop); then whole groups, oldest first.  Returns (trimmed_messages,
    dropped_message_count); the list is a new object whenever anything
    changed, the same object otherwise.  Leading system messages and the
    final turn group are always kept, so the result may still exceed the
    budget in the degenerate case — callers should treat this as best-effort.
    """
    # Older turns' reasoning first; whole older turns next; the current
    # loop's reasoning only as a last resort (below).
    stripped = strip_reasoning_to_fit(
        messages, budget_tokens, estimator, include_current_loop=False,
    )
    if stripped is not messages and _total(stripped, estimator) <= budget_tokens:
        return stripped, 0
    original = messages
    messages = stripped

    head, groups = _group_messages(messages)
    if not groups:
        return messages, 0

    head_cost = sum(_message_tokens(m, estimator) for m in head)
    group_costs = [
        sum(_message_tokens(m, estimator) for m in g) for g in groups
    ]

    total = head_cost + sum(group_costs)
    dropped = 0
    start = 0
    # Never drop the final group
    while total > budget_tokens and start < len(groups) - 1:
        total -= group_costs[start]
        dropped += len(groups[start])
        start += 1

    trimmed = head + [m for g in groups[start:] for m in g]
    if total > budget_tokens:
        trimmed = strip_reasoning_to_fit(trimmed, budget_tokens, estimator)
    if dropped == 0 and len(trimmed) == len(original) and all(a is b for a, b in zip(trimmed, original)):
        return original, 0
    return trimmed, dropped
