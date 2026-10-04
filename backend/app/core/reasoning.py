"""Reasoning (thinking) control: one gateway vocabulary, translated per model family.

Every inbound dialect folds its own spelling of "should the model think, and
how hard" into two canonical fields, ``think`` (bool) and ``reasoning_effort``
(one of :data:`GATEWAY_LEVELS`).  This module owns the second half of the
contract: what those two fields mean for a given model, and what the backend
must actually be told.

Why a translation table exists at all: the models do not agree on names.
gpt-oss takes ``low``/``medium``/``high`` and cannot switch thinking off;
Qwen3.8 takes ``low``/``medium``/``xhigh`` and defaults to ``xhigh``, the most
expensive level, when thinking is on and no level is given; GLM-5.3 takes
``low``/``high``/``max`` and, like gpt-oss, cannot switch off; Kimi K3 takes
the same three names but does have an off switch; Qwen3.5/3.6,
Gemma 4, Nemotron and MiMo have an on/off switch and no levels.  Forwarding a client's
level name verbatim therefore produced a vLLM 400 whenever the client and the
model disagreed (``high`` on Qwen3.8, ``xhigh`` on gpt-oss), and a bare
``think: true`` silently bought the priciest reasoning Qwen3.8 has.

This module has no imports from the database or settings layers so it can be
unit-tested in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

# The gateway vocabulary.  It is a superset of every family's native names, so
# a client may also send a native name and it round-trips unchanged when the
# target family knows it.  ``none`` means "do not think"; on a family that
# cannot switch thinking off it becomes the lowest level.
GATEWAY_LEVELS: Tuple[str, ...] = ("none", "minimal", "low", "medium", "high", "xhigh")

# Fallback when thinking is on, the family has levels, and neither the client
# nor the admin setting (``reasoning.default_effort``) chose one.
FALLBACK_DEFAULT_EFFORT = "medium"

# Anthropic-style ``budget_tokens`` bucketed into gateway levels.  The
# thresholds follow the shape of Claude budgets in the wild: a few thousand
# tokens is a quick think, tens of thousands is a deep one.
_BUDGET_TIERS: Tuple[Tuple[int, str], ...] = (
    (2048, "low"),
    (8192, "medium"),
    (32768, "high"),
)


# Native level names that fall OUTSIDE the gateway vocabulary, and the
# gateway level they mean. The catalog publishes each family's native names
# and the chat page sends them back verbatim, so every native name must be
# accepted here. Qwen3.8's and gpt-oss's names all coincide with gateway
# names; GLM-5.3's and Kimi K3's top level is ``max``.
LEVEL_ALIASES: dict[str, str] = {"max": "xhigh"}


class InvalidReasoningLevel(ValueError):
    """A reasoning level outside :data:`GATEWAY_LEVELS`."""

    def __init__(self, value: object) -> None:
        self.value = value
        super().__init__(
            f"Unknown reasoning level {value!r}. Accepted values: "
            + ", ".join(GATEWAY_LEVELS)
            + " (also " + ", ".join(sorted(LEVEL_ALIASES)) + ")."
        )


def normalize_level(value: object) -> str:
    """Return the canonical spelling of a gateway level or raise."""
    if not isinstance(value, str):
        raise InvalidReasoningLevel(value)
    level = value.strip().lower()
    level = LEVEL_ALIASES.get(level, level)
    if level not in GATEWAY_LEVELS:
        raise InvalidReasoningLevel(value)
    return level


def budget_to_level(budget_tokens: object) -> Optional[str]:
    """Map an Anthropic-style token budget onto a gateway level.

    Returns None for a missing or non-numeric budget so the caller can fall
    back to "thinking on, family default".
    """
    try:
        budget = int(budget_tokens)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if budget <= 0:
        return "none"
    for ceiling, level in _BUDGET_TIERS:
        if budget < ceiling:
            return level
    return "xhigh"


@dataclass(frozen=True)
class ReasoningProfile:
    """What a model family can do with reasoning controls.

    ``levels`` holds the family's NATIVE level names in ascending cost order;
    an empty tuple means the family has an on/off switch only.  ``level_map``
    translates every gateway level to a native one (empty when there are no
    levels).  ``default_level`` is what the MODEL does when thinking is on and
    no level is sent, which is what the gateway default setting overrides.
    """

    family: str
    toggleable: bool
    levels: Tuple[str, ...] = ()
    default_level: Optional[str] = None
    level_map: Dict[str, str] = field(default_factory=dict)

    @property
    def has_levels(self) -> bool:
        return bool(self.levels)

    def native_level(self, gateway_level: str) -> Optional[str]:
        """Translate a gateway level to this family's name for it, if any."""
        if not self.levels:
            return None
        return self.level_map.get(gateway_level)

    def describe(self) -> Dict[str, object]:
        """The catalog descriptor exposed on /v1/models."""
        return {
            "family": self.family,
            "toggleable": self.toggleable,
            "levels": list(self.levels),
            "default_level": self.default_level,
            "accepts": list(GATEWAY_LEVELS) + sorted(LEVEL_ALIASES),
        }


# --- family table ----------------------------------------------------------

_GPT_OSS = ReasoningProfile(
    family="gpt-oss",
    toggleable=False,
    levels=("low", "medium", "high"),
    default_level="medium",
    level_map={
        "none": "low",
        "minimal": "low",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "xhigh": "high",
    },
)

_QWEN38 = ReasoningProfile(
    family="qwen3.8",
    toggleable=True,
    levels=("low", "medium", "xhigh"),
    default_level="xhigh",
    level_map={
        "minimal": "low",
        "low": "low",
        "medium": "medium",
        "high": "xhigh",
        "xhigh": "xhigh",
    },
)


# GLM-5.3 (zai-org): its chat template has NO off switch — the generation
# prompt always opens a think block — and takes ``reasoning_effort`` in
# ``low``/``high``/``max``, defaulting to ``max`` (the deepest, priciest) when
# nothing is sent.  An unknown level name is silently treated as ``max`` by the
# template, so the gateway must never forward ``medium``.  "Off" is its lowest
# level, like gpt-oss.  (Discovered 2026-09-29 on GLM-5.3-Flash NVFP4; the
# level reaches the template through vLLM's top-level ``reasoning_effort``.)
_GLM5 = ReasoningProfile(
    family="glm-5.3",
    toggleable=False,
    levels=("low", "high", "max"),
    default_level="max",
    level_map={
        "none": "low",
        "minimal": "low",
        "low": "low",
        "medium": "high",
        "high": "high",
        "xhigh": "max",
    },
)


# Kimi K3 (Moonshot): an on/off switch (``enable_thinking``) AND levels. Its
# template takes ``thinking_effort`` in ``low``/``high``/``max``, defaulting to
# ``max``; vLLM's Kimi K3 support fills it from the top-level
# ``reasoning_effort``. Any other name is a 400 from vLLM ("Kimi K3 supports
# thinking_effort values: low, high, max"), so ``medium`` must never be
# forwarded. (Measured 2026-10-04 on Kimi-K3 NVFP4, vLLM 0.31.0rc2.)
_KIMI_K3 = ReasoningProfile(
    family="kimi-k3",
    toggleable=True,
    levels=("low", "high", "max"),
    default_level="max",
    level_map={
        "minimal": "low",
        "low": "low",
        "medium": "high",
        "high": "high",
        "xhigh": "max",
    },
)


def _toggle_only(family: str) -> ReasoningProfile:
    return ReasoningProfile(family=family, toggleable=True)


_PLAIN = ReasoningProfile(family="plain", toggleable=False)

# Ordered: the first substring that matches wins, so the more specific
# generation ("qwen3.8") must precede its family ("qwen").
_NAME_RULES: Tuple[Tuple[str, ReasoningProfile], ...] = (
    ("gpt-oss", _GPT_OSS),
    ("qwen3.8", _QWEN38),
    ("qwen3", _toggle_only("qwen3")),
    ("deepseek-r1", _toggle_only("deepseek-r1")),
    ("gemma-4", _toggle_only("gemma4")),
    ("gemma4", _toggle_only("gemma4")),
    ("nemotron", _toggle_only("nemotron")),
    ("magistral", _toggle_only("magistral")),
    ("glm-5", _GLM5),
    ("glm5", _GLM5),
    # MiMo-V2.6 (Xiaomi): ChatML-style ``enable_thinking`` switch, no levels.
    ("mimo", _toggle_only("mimo")),
    # Only K3: earlier Kimi models have different templates.
    ("kimi-k3", _KIMI_K3),
    ("kimi_k3", _KIMI_K3),
)


def profile_for(
    model_name: Optional[str],
    family: Optional[str] = None,
    supports_thinking: Optional[bool] = None,
) -> ReasoningProfile:
    """Pick the profile for a model.

    The model name decides for the families we know.  For anything else the
    discovery flag ``supports_thinking`` decides between a generic on/off
    profile and a plain model with no reasoning controls.
    """
    haystack = " ".join(x for x in (model_name, family) if x).lower()
    for needle, profile in _NAME_RULES:
        if needle in haystack:
            return profile
    if supports_thinking:
        return _toggle_only("generic")
    return _PLAIN


# --- resolution ---------------------------------------------------------------

@dataclass(frozen=True)
class ResolvedReasoning:
    """What the backend must be told.

    ``enabled`` is the value for the family's on/off switch: True/False, or
    None to leave the backend's own default alone (always None for a family
    without a switch).  ``effort`` is the NATIVE level to send, or None.
    ``ignored_effort`` records a gateway level the family could not express,
    for logging; it is never an error.
    """

    enabled: Optional[bool]
    effort: Optional[str] = None
    ignored_effort: Optional[str] = None


_TRUE_WORDS = ("true", "1", "yes", "on")
_FALSE_WORDS = ("false", "0", "no", "off")


def split_legacy_think(
    think: Union[bool, str, None], effort: Optional[str]
) -> Tuple[Optional[bool], Optional[str]]:
    """Fold the legacy string form of ``think`` into (bool, effort).

    ``think: "low"`` was the gpt-oss spelling before levels became a first
    class field; ``"true"``/``"false"`` show up from form-backed admin config.
    An explicit ``reasoning_effort`` wins over a level smuggled in ``think``.
    """
    if not isinstance(think, str):
        return think, effort
    word = think.strip().lower()
    if word in _TRUE_WORDS:
        return True, effort
    if word in _FALSE_WORDS:
        return False, effort
    return True, effort or word


def resolve_reasoning(
    think: Union[bool, str, None],
    effort: Optional[str],
    profile: ReasoningProfile,
    *,
    off_by_default: bool = True,
    default_effort: Optional[str] = FALLBACK_DEFAULT_EFFORT,
) -> ResolvedReasoning:
    """Decide the backend-facing reasoning settings for one request.

    Rules, in order:

    1. Any level in ``effort`` (or a string ``think``) is an opt-in: thinking
       is on.  ``none`` is the opposite: thinking off.
    2. ``think: false`` wins over any level.
    3. With nothing said, ``off_by_default`` switches thinking off on a family
       that has a switch; a family without one is left to its own default.
    4. When thinking is on and the family has levels, the level sent is the
       client's, else ``default_effort``, translated to the family's name.
       A family without levels ignores the level (recorded, not rejected).
    5. A family that cannot switch off answers an "off" request with its
       lowest level.

    ``effort`` must already be a gateway level (see :func:`normalize_level`);
    ``default_effort`` may be None to fall back to the model's own default.
    """
    think, effort = split_legacy_think(think, effort)
    level = normalize_level(effort) if effort is not None else None

    wants_off = think is False or level == "none"
    if wants_off:
        if profile.toggleable:
            return ResolvedReasoning(enabled=False)
        # Cannot switch off: ask for the cheapest thinking instead.
        return ResolvedReasoning(enabled=None, effort=profile.native_level("none"))

    wants_on = think is True or level is not None
    if not wants_on:
        if off_by_default and profile.toggleable:
            return ResolvedReasoning(enabled=False)
        return ResolvedReasoning(enabled=None)

    enabled: Optional[bool] = True if profile.toggleable else None
    if not profile.has_levels:
        return ResolvedReasoning(enabled=enabled, ignored_effort=level)

    chosen = level
    if chosen is None and default_effort:
        chosen = normalize_level(default_effort)
    native = profile.native_level(chosen) if chosen else None
    return ResolvedReasoning(enabled=enabled, effort=native)
