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
turns (kept for models whose templates replay it) is removed from every
group but the last: it is the cheapest context to lose.
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


def trim_messages_to_fit(
    messages: List[CanonicalMessage],
    budget_tokens: int,
    estimator: Callable[[str], int],
) -> Tuple[List[CanonicalMessage], int]:
    """Drop oldest turn groups until the estimated total fits the budget.

    Earlier assistant reasoning goes first (from every group but the last);
    then whole groups, oldest first.  Returns (trimmed_messages,
    dropped_message_count); the list is a new object whenever anything
    changed, the same object otherwise.  Leading system messages and the
    final turn group are always kept, so the result may still exceed the
    budget in the degenerate case — callers should treat this as best-effort.
    """
    head, groups = _group_messages(messages)
    if not groups:
        return messages, 0

    head_cost = sum(_message_tokens(m, estimator) for m in head)
    group_costs = [
        sum(_message_tokens(m, estimator) for m in g) for g in groups
    ]

    total = head_cost + sum(group_costs)
    if total > budget_tokens:
        stripped = False
        for i, group in enumerate(groups[:-1]):
            if any(m.reasoning for m in group):
                groups[i] = [m.model_copy(update={"reasoning": None}) if m.reasoning else m for m in group]
                group_costs[i] = sum(_message_tokens(m, estimator) for m in groups[i])
                stripped = True
        if stripped:
            total = head_cost + sum(group_costs)
            if total <= budget_tokens:
                return head + [m for g in groups for m in g], 0
    dropped = 0
    start = 0
    # Never drop the final group
    while total > budget_tokens and start < len(groups) - 1:
        total -= group_costs[start]
        dropped += len(groups[start])
        start += 1

    trimmed = head + [m for g in groups[start:] for m in g]
    if dropped == 0 and len(trimmed) == len(messages) and all(a is b for a, b in zip(trimmed, messages)):
        return messages, 0
    return trimmed, dropped
