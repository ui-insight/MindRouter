"""Reasoning on/off + levels: one gateway vocabulary, translated per family.

Covers core/reasoning.py (profiles, vocabulary, resolution), the inbound
translators' folding of every client spelling into the canonical pair, the
outbound translators' emission, the inference policy that ties them together,
and the catalog descriptor.

Background: Qwen3.8 takes reasoning_effort low/medium/xhigh and defaults to
xhigh; gpt-oss takes low/medium/high and cannot switch thinking off; the
other thinking families have a switch and no levels.  Before this feature a
bare reasoning_effort was ignored (thinking stayed off), a bare think:true
bought xhigh, and a level name the model did not know was a vLLM 400.
"""

import asyncio
from types import SimpleNamespace

import pytest

from backend.app.core import reasoning as R
from backend.app.core.canonical_schemas import (
    CanonicalChatRequest,
    CanonicalMessage,
    MessageRole,
)
from backend.app.core.translators import (
    AnthropicInTranslator,
    OllamaOutTranslator,
    OpenAIInTranslator,
    ResponsesInTranslator,
    VLLMOutTranslator,
)


def _canon(**overrides) -> CanonicalChatRequest:
    defaults = dict(
        model="qwen/qwen3.8-27b",
        messages=[CanonicalMessage(role=MessageRole.USER, content="Hi")],
    )
    defaults.update(overrides)
    return CanonicalChatRequest(**defaults)


def _openai(**extra):
    body = {"model": "qwen/qwen3.8-27b", "messages": [{"role": "user", "content": "Hi"}]}
    body.update(extra)
    return body


Q38 = R.profile_for("qwen/qwen3.8-27b")
GPT = R.profile_for("openai/gpt-oss-120b")
Q35 = R.profile_for("qwen/qwen3.5-122b")
PLAIN = R.profile_for("microsoft/phi-4")


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

class TestProfiles:
    def test_families_by_name(self):
        assert Q38.family == "qwen3.8" and Q38.toggleable and Q38.levels == ("low", "medium", "xhigh")
        assert GPT.family == "gpt-oss" and not GPT.toggleable and GPT.levels == ("low", "medium", "high")
        assert Q35.family == "qwen3" and Q35.toggleable and not Q35.has_levels
        assert R.profile_for("qwen/qwen3.6-35b").family == "qwen3"
        assert R.profile_for("google/gemma-4-31b").toggleable
        assert R.profile_for("nvidia/nemotron-3-super").toggleable
        assert R.profile_for("deepseek-r1:70b").toggleable

    def test_qwen38_precedes_qwen3(self):
        # "qwen3" is a substring of "qwen3.8"; the specific rule must win.
        assert R.profile_for("Qwen/Qwen3.8-27B-FP8").family == "qwen3.8"

    def test_unknown_models_fall_back_on_discovery_flag(self):
        assert PLAIN.family == "plain" and not PLAIN.toggleable and not PLAIN.has_levels
        generic = R.profile_for("some/new-model", supports_thinking=True)
        assert generic.toggleable and not generic.has_levels

    def test_glm53_and_mimo_families(self):
        # GLM-5.3: no switch, three native levels, model default is the deepest.
        glm = R.profile_for("zai-org/glm-5.3-flash")
        assert glm.family == "glm-5.3" and not glm.toggleable
        assert glm.levels == ("low", "high", "max") and glm.default_level == "max"
        assert R.profile_for("RedHatAI/GLM-5.3-Flash-NVFP4").family == "glm-5.3"
        assert R.profile_for("zai-org/GLM5-Air").family == "glm-5.3"
        # MiMo-V2.6: an enable_thinking switch and nothing else.
        mimo = R.profile_for("xiaomimimo/mimo-v2.6-flash")
        assert mimo.family == "mimo" and mimo.toggleable and not mimo.has_levels
        assert R.profile_for("XiaomiMiMo/MiMo-V2.6-Flash-MOPD").family == "mimo"

    def test_kimi_k3_has_a_switch_and_three_levels(self):
        kimi = R.profile_for("moonshotai/kimi-k3")
        assert kimi.family == "kimi-k3" and kimi.toggleable
        assert kimi.levels == ("low", "high", "max") and kimi.default_level == "max"
        for name in ("RedHatAI/Kimi-K3-NVFP4", "moonshotai/Kimi_K3", "kimi-k3:latest"):
            assert R.profile_for(name).family == "kimi-k3", name
        # The name decides, whatever the discovery flag says (it is name-guessed false for Kimi).
        assert R.profile_for("moonshotai/kimi-k3", supports_thinking=False).family == "kimi-k3"
        # Earlier Kimi models have other templates and are not claimed.
        assert R.profile_for("moonshotai/kimi-k2", supports_thinking=True).family == "generic"

    def test_kimi_k3_never_forwards_a_level_vllm_would_reject(self):
        # vLLM answers 400 to any thinking_effort other than low/high/max.
        kimi = R.profile_for("moonshotai/kimi-k3")
        for lvl in R.GATEWAY_LEVELS:
            if lvl == "none":
                continue                      # "none" is the switch on a family that has one
            assert kimi.native_level(lvl) in kimi.levels, lvl
        assert kimi.native_level("medium") == "high" and kimi.native_level("minimal") == "low"
        assert kimi.native_level("xhigh") == "max"

    def test_kimi_k3_resolution(self):
        kimi = R.profile_for("moonshotai/kimi-k3")
        # Nothing said: off (gateway policy), and no level sent.
        assert R.resolve_reasoning(None, None, kimi) == R.ResolvedReasoning(enabled=False)
        assert R.resolve_reasoning(False, "high", kimi) == R.ResolvedReasoning(enabled=False)   # off wins
        assert R.resolve_reasoning(None, "none", kimi) == R.ResolvedReasoning(enabled=False)
        # On: the switch plus a level Kimi knows.
        assert R.resolve_reasoning(True, None, kimi) == R.ResolvedReasoning(enabled=True, effort="high")   # admin default medium
        assert R.resolve_reasoning(None, "low", kimi) == R.ResolvedReasoning(enabled=True, effort="low")
        assert R.resolve_reasoning(None, "medium", kimi) == R.ResolvedReasoning(enabled=True, effort="high")
        assert R.resolve_reasoning(None, "max", kimi) == R.ResolvedReasoning(enabled=True, effort="max")
        assert R.resolve_reasoning(None, "xhigh", kimi) == R.ResolvedReasoning(enabled=True, effort="max")
        # No gateway default: thinking on at the model's own default (max), nothing sent.
        assert R.resolve_reasoning(True, None, kimi, default_effort=None) == R.ResolvedReasoning(enabled=True, effort=None)
        # Policy off: the model's own default (thinking on) is left alone.
        assert R.resolve_reasoning(None, None, kimi, off_by_default=False) == R.ResolvedReasoning(enabled=None)

    def test_kimi_k3_request_as_vllm_reads_it(self):
        from backend.app.core.translators.vllm_out import VLLMOutTranslator

        def payload(think, effort):
            req = CanonicalChatRequest(model="moonshotai/kimi-k3",
                                       messages=[CanonicalMessage(role=MessageRole.USER, content="hi")],
                                       think=think, reasoning_effort=effort)
            return VLLMOutTranslator.translate_chat_request(req)

        on = payload(True, "high")
        assert on["reasoning_effort"] == "high" and on["chat_template_kwargs"] == {"enable_thinking": True}
        off = payload(False, None)
        assert "reasoning_effort" not in off and off["chat_template_kwargs"] == {"enable_thinking": False}

    def test_native_max_is_accepted_everywhere(self):
        # The chat page sends GLM's native "max" back; it must validate and
        # mean the top gateway level on every family (2.9.82 hotfix).
        assert R.normalize_level("max") == "xhigh" and R.normalize_level(" MAX ") == "xhigh"
        glm = R.profile_for("zai-org/glm-5.3-flash")
        assert R.resolve_reasoning(None, "max", glm).effort == "max"
        assert R.resolve_reasoning(None, "max", Q38).effort == "xhigh"
        assert R.resolve_reasoning(None, "max", GPT).effort == "high"
        assert "max" in glm.describe()["accepts"]
        req = CanonicalChatRequest(model="zai-org/glm-5.3-flash", messages=[CanonicalMessage(role=MessageRole.USER, content="hi")], reasoning_effort="max")
        assert req.reasoning_effort == "xhigh"
        with pytest.raises(R.InvalidReasoningLevel):
            R.normalize_level("ultra")

    def test_glm53_never_forwards_a_level_its_template_would_misread(self):
        # The GLM template treats any name outside low/high/max as max, so
        # every gateway level must land on one of the three.
        glm = R.profile_for("zai-org/glm-5.3-flash")
        for lvl in R.GATEWAY_LEVELS:
            assert glm.native_level(lvl) in glm.levels, lvl
        assert glm.native_level("medium") == "high"
        assert glm.native_level("xhigh") == "max"

    def test_family_hint_is_consulted(self):
        assert R.profile_for("alias-name", family="gpt-oss").family == "gpt-oss"

    def test_descriptor_shape(self):
        d = Q38.describe()
        assert d == {
            "family": "qwen3.8",
            "toggleable": True,
            "levels": ["low", "medium", "xhigh"],
            "default_level": "xhigh",
            "accepts": list(R.GATEWAY_LEVELS) + ["max"],  # native aliases follow the vocabulary
        }

    def test_every_gateway_level_maps_for_families_with_levels(self):
        for prof in (Q38, GPT, R.profile_for("moonshotai/kimi-k3")):
            for lvl in R.GATEWAY_LEVELS:
                if lvl == "none" and prof.toggleable:
                    continue  # "none" is the switch, not a level, when a switch exists
                native = prof.native_level(lvl)
                assert native in prof.levels, (prof.family, lvl, native)


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

class TestVocabulary:
    def test_normalize_accepts_case_and_whitespace(self):
        assert R.normalize_level(" XHigh ") == "xhigh"

    def test_normalize_rejects_unknown_with_helpful_message(self):
        with pytest.raises(R.InvalidReasoningLevel) as ei:
            R.normalize_level("turbo")
        assert "turbo" in str(ei.value) and "xhigh" in str(ei.value)

    def test_normalize_rejects_non_strings(self):
        with pytest.raises(R.InvalidReasoningLevel):
            R.normalize_level(3)

    def test_budget_tiers(self):
        assert R.budget_to_level(0) == "none"
        assert R.budget_to_level(1024) == "low"
        assert R.budget_to_level(4096) == "medium"
        assert R.budget_to_level(16000) == "high"
        assert R.budget_to_level(64000) == "xhigh"
        assert R.budget_to_level("nope") is None

    def test_split_legacy_think(self):
        assert R.split_legacy_think("low", None) == (True, "low")
        assert R.split_legacy_think("low", "high") == (True, "high")  # explicit level wins
        assert R.split_legacy_think("true", None) == (True, None)
        assert R.split_legacy_think("False", None) == (False, None)
        assert R.split_legacy_think(True, "medium") == (True, "medium")


# ---------------------------------------------------------------------------
# Resolution rules
# ---------------------------------------------------------------------------

class TestResolve:
    def test_nothing_said_is_off_on_a_switch_family(self):
        assert R.resolve_reasoning(None, None, Q38) == R.ResolvedReasoning(enabled=False)
        assert R.resolve_reasoning(None, None, Q35) == R.ResolvedReasoning(enabled=False)

    def test_nothing_said_with_policy_off_leaves_model_default(self):
        assert R.resolve_reasoning(None, None, Q38, off_by_default=False).enabled is None

    def test_nothing_said_on_gpt_oss_is_left_alone(self):
        assert R.resolve_reasoning(None, None, GPT) == R.ResolvedReasoning(enabled=None)

    def test_think_true_gets_the_gateway_default_level(self):
        r = R.resolve_reasoning(True, None, Q38, default_effort="medium")
        assert r == R.ResolvedReasoning(enabled=True, effort="medium")

    def test_think_true_without_a_default_leaves_the_model_default(self):
        r = R.resolve_reasoning(True, None, Q38, default_effort=None)
        assert r == R.ResolvedReasoning(enabled=True, effort=None)

    def test_level_alone_is_an_opt_in(self):
        r = R.resolve_reasoning(None, "low", Q38)
        assert r == R.ResolvedReasoning(enabled=True, effort="low")

    def test_levels_translate_per_family(self):
        assert R.resolve_reasoning(None, "high", Q38).effort == "xhigh"
        assert R.resolve_reasoning(None, "minimal", Q38).effort == "low"
        assert R.resolve_reasoning(None, "xhigh", GPT).effort == "high"
        assert R.resolve_reasoning(None, "minimal", GPT).effort == "low"

    def test_native_names_round_trip(self):
        assert R.resolve_reasoning(None, "xhigh", Q38).effort == "xhigh"
        assert R.resolve_reasoning(None, "medium", GPT).effort == "medium"

    def test_think_false_wins_over_a_level(self):
        assert R.resolve_reasoning(False, "xhigh", Q38) == R.ResolvedReasoning(enabled=False)

    def test_none_means_off(self):
        assert R.resolve_reasoning(None, "none", Q38) == R.ResolvedReasoning(enabled=False)
        assert R.resolve_reasoning(True, "none", Q38) == R.ResolvedReasoning(enabled=False)

    def test_off_on_a_family_that_cannot_switch_off_is_its_lowest_level(self):
        assert R.resolve_reasoning(False, None, GPT) == R.ResolvedReasoning(enabled=None, effort="low")
        assert R.resolve_reasoning(None, "none", GPT) == R.ResolvedReasoning(enabled=None, effort="low")

    def test_gpt_oss_never_gets_the_switch(self):
        assert R.resolve_reasoning(True, "high", GPT) == R.ResolvedReasoning(enabled=None, effort="high")

    def test_glm53_off_is_low_and_on_translates(self):
        glm = R.profile_for("zai-org/glm-5.3-flash")
        # Nothing said: left to the server (the unit's default kwargs choose low).
        assert R.resolve_reasoning(None, None, glm) == R.ResolvedReasoning(enabled=None)
        # Off requests become the cheapest level; there is no switch to send.
        assert R.resolve_reasoning(False, None, glm) == R.ResolvedReasoning(enabled=None, effort="low")
        assert R.resolve_reasoning(None, "none", glm) == R.ResolvedReasoning(enabled=None, effort="low")
        # On with the gateway default (medium) is GLM's "high"; xhigh is "max".
        assert R.resolve_reasoning(True, None, glm, default_effort="medium") == R.ResolvedReasoning(enabled=None, effort="high")
        assert R.resolve_reasoning(None, "xhigh", glm) == R.ResolvedReasoning(enabled=None, effort="max")
        assert R.resolve_reasoning(None, "high", glm).effort == "high"
        assert R.resolve_reasoning(None, "low", glm).effort == "low"
        # A native name round-trips.
        assert R.resolve_reasoning(None, "xhigh", glm).effort == "max"

    def test_mimo_is_a_plain_switch(self):
        mimo = R.profile_for("xiaomimimo/mimo-v2.6-flash")
        assert R.resolve_reasoning(None, None, mimo) == R.ResolvedReasoning(enabled=False)
        assert R.resolve_reasoning(True, None, mimo) == R.ResolvedReasoning(enabled=True)
        assert R.resolve_reasoning(None, "high", mimo) == R.ResolvedReasoning(enabled=True, effort=None, ignored_effort="high")
        assert R.resolve_reasoning(False, "high", mimo) == R.ResolvedReasoning(enabled=False)

    def test_families_without_levels_ignore_but_record_the_level(self):
        r = R.resolve_reasoning(None, "high", Q35)
        assert r == R.ResolvedReasoning(enabled=True, effort=None, ignored_effort="high")

    def test_legacy_string_think(self):
        assert R.resolve_reasoning("low", None, GPT).effort == "low"
        assert R.resolve_reasoning("xhigh", None, Q38).effort == "xhigh"
        assert R.resolve_reasoning("false", "high", Q38).enabled is False

    def test_invalid_level_raises(self):
        with pytest.raises(R.InvalidReasoningLevel):
            R.resolve_reasoning(None, "ultra", Q38)


# ---------------------------------------------------------------------------
# Canonical model folds and validates
# ---------------------------------------------------------------------------

class TestCanonicalNormalization:
    def test_string_think_becomes_switch_plus_level(self):
        c = _canon(think="low")
        assert c.think is True and c.reasoning_effort == "low"

    def test_level_is_normalized(self):
        assert _canon(reasoning_effort=" XHIGH ").reasoning_effort == "xhigh"

    def test_unknown_level_is_a_validation_error(self):
        with pytest.raises(ValueError):
            _canon(reasoning_effort="turbo")

    def test_bool_think_untouched(self):
        c = _canon(think=False, reasoning_effort="high")
        assert c.think is False and c.reasoning_effort == "high"


# ---------------------------------------------------------------------------
# Inbound translators
# ---------------------------------------------------------------------------

class TestOpenAIIn:
    def test_reasoning_effort_alone_is_kept(self):
        c = OpenAIInTranslator.translate_chat_request(_openai(reasoning_effort="low"))
        assert c.think is None and c.reasoning_effort == "low"

    def test_reasoning_object_effort(self):
        c = OpenAIInTranslator.translate_chat_request(_openai(reasoning={"effort": "xhigh"}))
        assert c.reasoning_effort == "xhigh"

    def test_chat_template_kwargs_effort_and_passthrough(self):
        c = OpenAIInTranslator.translate_chat_request(_openai(
            chat_template_kwargs={"enable_thinking": True, "reasoning_effort": "medium", "preserve_thinking": False}
        ))
        assert c.think is True
        assert c.reasoning_effort == "medium"
        assert c.chat_template_kwargs == {"preserve_thinking": False}

    def test_passthrough_drops_non_scalars_and_is_none_when_empty(self):
        c = OpenAIInTranslator.translate_chat_request(_openai(
            chat_template_kwargs={"enable_thinking": False, "tools_spec": {"x": 1}}
        ))
        assert c.think is False and c.chat_template_kwargs is None

    def test_thinking_object_budget_buckets_into_a_level(self):
        c = OpenAIInTranslator.translate_chat_request(_openai(
            thinking={"type": "enabled", "budget_tokens": 16000}
        ))
        assert c.think is True and c.reasoning_effort == "high"

    def test_legacy_string_think(self):
        c = OpenAIInTranslator.translate_chat_request(_openai(think="high"))
        assert c.think is True and c.reasoning_effort == "high"

    def test_top_level_effort_wins_over_nested(self):
        c = OpenAIInTranslator.translate_chat_request(_openai(
            reasoning_effort="low", reasoning={"effort": "xhigh"},
            chat_template_kwargs={"reasoning_effort": "medium"},
        ))
        assert c.reasoning_effort == "low"

    def test_unknown_level_rejected_at_translation(self):
        with pytest.raises(ValueError):
            OpenAIInTranslator.translate_chat_request(_openai(reasoning_effort="turbo"))


class TestAnthropicIn:
    def _msg(self, **extra):
        body = {"model": "qwen/qwen3.8-27b", "max_tokens": 100,
                "messages": [{"role": "user", "content": "Hi"}]}
        body.update(extra)
        return AnthropicInTranslator.translate_messages_request(body)

    def test_budget_tokens_becomes_a_level(self):
        c = self._msg(thinking={"type": "enabled", "budget_tokens": 1024})
        assert c.think is True and c.reasoning_effort == "low"

    def test_enabled_without_budget_has_no_level(self):
        c = self._msg(thinking={"type": "enabled"})
        assert c.think is True and c.reasoning_effort is None

    def test_disabled_ignores_budget(self):
        c = self._msg(thinking={"type": "disabled", "budget_tokens": 50000})
        assert c.think is False and c.reasoning_effort is None

    def test_output_config_effort_max_is_xhigh(self):
        c = self._msg(thinking={"type": "enabled", "budget_tokens": 1024},
                      output_config={"effort": "max"})
        assert c.reasoning_effort == "xhigh"


class TestResponsesIn:
    def test_effort_is_switch_on_plus_level(self):
        c = ResponsesInTranslator.translate_responses_request(
            {"model": "m", "input": "hi", "reasoning": {"effort": "xhigh"}}
        )
        assert c.think is True and c.reasoning_effort == "xhigh"

    def test_none_is_off(self):
        c = ResponsesInTranslator.translate_responses_request(
            {"model": "m", "input": "hi", "reasoning": {"effort": "none"}}
        )
        assert c.think is False and c.reasoning_effort is None


# ---------------------------------------------------------------------------
# Outbound translators (values are post-policy: family-native)
# ---------------------------------------------------------------------------

class TestVLLMOut:
    def test_switch_and_level(self):
        p = VLLMOutTranslator.translate_chat_request(_canon(think=True, reasoning_effort="xhigh"))
        assert p["chat_template_kwargs"] == {"enable_thinking": True}
        assert p["reasoning_effort"] == "xhigh"

    def test_switch_only(self):
        p = VLLMOutTranslator.translate_chat_request(_canon(think=False))
        assert p["chat_template_kwargs"] == {"enable_thinking": False}
        assert "reasoning_effort" not in p

    def test_level_only_no_switch_for_gpt_oss(self):
        p = VLLMOutTranslator.translate_chat_request(
            _canon(model="openai/gpt-oss-120b", think=None, reasoning_effort="high")
        )
        assert p["reasoning_effort"] == "high" and "chat_template_kwargs" not in p

    def test_passthrough_kwargs_ride_beside_the_switch(self):
        p = VLLMOutTranslator.translate_chat_request(
            _canon(think=True, chat_template_kwargs={"preserve_thinking": False})
        )
        assert p["chat_template_kwargs"] == {"preserve_thinking": False, "enable_thinking": True}

    def test_nothing_emits_nothing(self):
        p = VLLMOutTranslator.translate_chat_request(_canon())
        assert "chat_template_kwargs" not in p and "reasoning_effort" not in p


class TestOllamaOut:
    def test_switch_family_gets_bool(self):
        p = OllamaOutTranslator.translate_chat_request(_canon(model="qwen3:32b", think=True, reasoning_effort=None))
        assert p["think"] is True

    def test_gpt_oss_gets_the_level_as_think(self):
        p = OllamaOutTranslator.translate_chat_request(
            _canon(model="gpt-oss:120b", think=None, reasoning_effort="low")
        )
        assert p["think"] == "low"

    def test_nothing_emits_nothing(self):
        assert "think" not in OllamaOutTranslator.translate_chat_request(_canon())


# ---------------------------------------------------------------------------
# Inference policy (pure function on the request; DB read stubbed)
# ---------------------------------------------------------------------------

@pytest.fixture
def policy(monkeypatch):
    """apply_reasoning_policy with the default-effort config stubbed."""
    from backend.app.services import inference as inf

    async def _default():
        return "medium"

    monkeypatch.setattr(inf, "_get_default_effort", _default)
    settings = SimpleNamespace(thinking_off_by_default=True)

    def run(request, model, row=None, s=settings):
        asyncio.run(inf.apply_reasoning_policy(request, model, row, s))
        return request

    return run


class TestInferencePolicy:
    def test_bare_effort_opts_in_and_is_translated(self, policy):
        req = policy(_canon(reasoning_effort="high"), "qwen/qwen3.8-27b")
        assert req.think is True and req.reasoning_effort == "xhigh"

    def test_bare_think_true_gets_admin_default(self, policy):
        req = policy(_canon(think=True), "qwen/qwen3.8-27b")
        assert req.think is True and req.reasoning_effort == "medium"

    def test_nothing_said_is_off(self, policy):
        req = policy(_canon(), "qwen/qwen3.5-122b")
        assert req.think is False and req.reasoning_effort is None

    def test_gpt_oss_xhigh_becomes_high_without_a_switch(self, policy):
        req = policy(_canon(model="openai/gpt-oss-120b", reasoning_effort="xhigh"), "openai/gpt-oss-120b")
        assert req.think is None and req.reasoning_effort == "high"

    def test_gpt_oss_untouched_when_silent(self, policy):
        req = policy(_canon(model="openai/gpt-oss-120b"), "openai/gpt-oss-120b")
        assert req.think is None and req.reasoning_effort is None

    def test_level_dropped_on_switch_only_family(self, policy):
        req = policy(_canon(model="qwen/qwen3.6-35b", reasoning_effort="high"), "qwen/qwen3.6-35b")
        assert req.think is True and req.reasoning_effort is None

    def test_policy_off_leaves_model_default(self, policy):
        req = policy(_canon(), "qwen/qwen3.8-27b", s=SimpleNamespace(thinking_off_by_default=False))
        assert req.think is None

    def test_settings_without_the_flag_default_to_off(self, policy):
        # Test fixtures elsewhere build settings as a bare SimpleNamespace.
        req = policy(_canon(), "qwen/qwen3.8-27b", s=SimpleNamespace())
        assert req.think is False

    def test_model_row_family_hint(self, policy):
        row = SimpleNamespace(family="gpt-oss", supports_thinking=True)
        req = policy(_canon(model="alias", reasoning_effort="xhigh"), "alias", row)
        assert req.reasoning_effort == "high" and req.think is None

    def test_requests_without_think_are_ignored(self, policy):
        req = SimpleNamespace(prompt="x")
        policy(req, "qwen/qwen3.8-27b")  # no AttributeError
        assert not hasattr(req, "think")


# ---------------------------------------------------------------------------
# Catalog descriptor
# ---------------------------------------------------------------------------

class TestCatalogDescriptor:
    def test_model_info_carries_reasoning(self):
        from backend.app.core.canonical_schemas import CanonicalModelInfo

        info = CanonicalModelInfo(id="qwen/qwen3.8-27b", created=0, reasoning=Q38.describe())
        assert info.reasoning["levels"] == ["low", "medium", "xhigh"]
        assert info.reasoning["toggleable"] is True

    def test_chat_ui_contract(self):
        # The chat page decides: switch if toggleable, level menu if levels.
        for name, switch, levels in (
            ("qwen/qwen3.8-27b", True, ["low", "medium", "xhigh"]),
            ("openai/gpt-oss-120b", False, ["low", "medium", "high"]),
            ("qwen/qwen3.6-35b", True, []),
            ("zai-org/glm-5.3-flash", False, ["low", "high", "max"]),
            ("xiaomimimo/mimo-v2.6-flash", True, []),
        ):
            d = R.profile_for(name).describe()
            assert (d["toggleable"], d["levels"]) == (switch, levels), name
