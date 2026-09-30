############################################################
#
# mindrouter - unit tests for per-model sampling guard rails
# (core/sampling_policy.py + the inference hook)
#
# GLM-5.3-Flash and MiMo-V2.6 loop until max_tokens under greedy decoding;
# a benchmark sending temperature 0 / max_tokens 65536 burned a night of
# B300 time producing empty 65k-token responses (2026-09-30). The admin
# policy clamps such requests at the gateway, on the canonical request, so
# every dialect is covered, and logs rather than rejects.
#
############################################################

"""Unit tests for sampling.policies."""

import asyncio
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.app.core import sampling_policy as sp
from backend.app.core.canonical_schemas import CanonicalChatRequest, CanonicalMessage, MessageRole


def _req(**kw):
    return CanonicalChatRequest(model="zai-org/glm-5.3-flash", messages=[CanonicalMessage(role=MessageRole.USER, content="hi")], **kw)


class TestValidate:
    def test_object_and_json_string_forms(self):
        obj = {"zai-org/glm-5.3-flash": {"min_temperature": 0.7, "max_tokens": 32768}}
        for raw in (obj, __import__("json").dumps(obj)):
            pol, errs = sp.validate_policies(raw)
            assert errs == [] and pol["zai-org/glm-5.3-flash"] == sp.SamplingPolicy(0.7, 32768)

    def test_empty_and_none_are_no_policy(self):
        assert sp.validate_policies("") == ({}, [])
        assert sp.validate_policies(None) == ({}, [])
        assert sp.validate_policies({}) == ({}, [])

    def test_partial_policies_allowed(self):
        pol, errs = sp.validate_policies({"m": {"max_tokens": 100}, "n": {"min_temperature": 1}})
        assert errs == [] and pol["m"] == sp.SamplingPolicy(None, 100) and pol["n"] == sp.SamplingPolicy(1.0, None)

    @pytest.mark.parametrize("raw, fragment", [
        ("{not json", "not valid JSON"),
        (["a"], "JSON object"),
        ({"m": "hot"}, "must be an object"),
        ({"m": {"temperature": 0.5}}, "unknown keys"),
        ({"m": {"min_temperature": 3}}, "between 0 and 2"),
        ({"m": {"min_temperature": "0.5"}}, "between 0 and 2"),
        ({"m": {"min_temperature": True}}, "between 0 and 2"),
        ({"m": {"max_tokens": 0}}, ">= 1"),
        ({"m": {"max_tokens": 10.5}}, ">= 1"),
        ({"m": {}}, "sets nothing"),
        ({"": {"max_tokens": 5}}, "non-empty"),
    ])
    def test_bad_entries_are_reported(self, raw, fragment):
        pol, errs = sp.validate_policies(raw)
        assert pol == {} and any(fragment in e for e in errs), errs

    def test_one_bad_entry_does_not_sink_the_others(self):
        pol, errs = sp.validate_policies({"good": {"max_tokens": 5}, "bad": {"max_tokens": -1}})
        assert list(pol) == ["good"] and len(errs) == 1

    def test_parse_is_lenient(self):
        assert sp.parse_policies({"good": {"max_tokens": 5}, "bad": "x"}) == {"good": sp.SamplingPolicy(None, 5)}


class TestLookup:
    POL = {"zai-org/glm-5.3-flash": sp.SamplingPolicy(0.7, None), "*": sp.SamplingPolicy(None, 8192)}

    def test_exact_then_case_insensitive_then_wildcard(self):
        assert sp.policy_for(self.POL, "zai-org/glm-5.3-flash").min_temperature == 0.7
        assert sp.policy_for(self.POL, "ZAI-ORG/GLM-5.3-Flash").min_temperature == 0.7
        assert sp.policy_for(self.POL, "some/other") == sp.SamplingPolicy(None, 8192)

    def test_no_wildcard_means_unknown_models_untouched(self):
        assert sp.policy_for({"m": sp.SamplingPolicy(0.5, None)}, "other") is None
        assert sp.policy_for({}, "m") is None
        assert sp.policy_for({"m": sp.SamplingPolicy(0.5, None)}, None) is None


class TestApply:
    def test_temperature_floor_raises_only_a_low_explicit_value(self):
        pol = sp.SamplingPolicy(min_temperature=0.7)
        r = _req(temperature=0.0)
        assert sp.apply_policy(r, pol) == {"temperature": (0.0, 0.7)} and r.temperature == 0.7
        r = _req(temperature=1.0)
        assert sp.apply_policy(r, pol) == {} and r.temperature == 1.0
        r = _req(temperature=0.7)
        assert sp.apply_policy(r, pol) == {}
        r = _req()
        assert sp.apply_policy(r, pol) == {} and r.temperature is None  # backend default stays

    def test_max_tokens_cap_applies_when_over_or_unset(self):
        pol = sp.SamplingPolicy(max_tokens=4096)
        r = _req(max_tokens=65536)
        assert sp.apply_policy(r, pol) == {"max_tokens": (65536, 4096)} and r.max_tokens == 4096
        r = _req()
        assert sp.apply_policy(r, pol) == {"max_tokens": (None, 4096)} and r.max_tokens == 4096
        r = _req(max_tokens=100)
        assert sp.apply_policy(r, pol) == {} and r.max_tokens == 100

    def test_no_policy_is_a_no_op(self):
        r = _req(temperature=0.0, max_tokens=65536)
        assert sp.apply_policy(r, None) == {} and (r.temperature, r.max_tokens) == (0.0, 65536)

    def test_objects_without_the_fields_are_ignored(self):
        assert sp.apply_policy(SimpleNamespace(), sp.SamplingPolicy(0.5, 10)) == {}


class TestInferenceHook:
    def _run(self, monkeypatch, policies, request, model="zai-org/glm-5.3-flash"):
        from backend.app.services import inference as inf
        async def _policies():
            return policies
        monkeypatch.setattr(inf, "_get_sampling_policies", _policies)
        events = []
        monkeypatch.setattr(inf.logger, "info", lambda ev, **kw: events.append((ev, kw)))
        asyncio.run(inf.apply_sampling_policy(request, model))
        return request, events

    def test_clamps_and_logs(self, monkeypatch):
        r, events = self._run(monkeypatch, {"zai-org/glm-5.3-flash": sp.SamplingPolicy(0.7, 32768)}, _req(temperature=0.0, max_tokens=65536))
        assert (r.temperature, r.max_tokens) == (0.7, 32768)
        assert events == [("sampling_policy_applied", {"model": "zai-org/glm-5.3-flash", "temperature": {"from": 0.0, "to": 0.7}, "max_tokens": {"from": 65536, "to": 32768}})]

    def test_no_policy_no_log(self, monkeypatch):
        r, events = self._run(monkeypatch, {}, _req(temperature=0.0))
        assert r.temperature == 0.0 and events == []

    def test_other_model_untouched(self, monkeypatch):
        r, events = self._run(monkeypatch, {"other": sp.SamplingPolicy(0.7, None)}, _req(temperature=0.0))
        assert r.temperature == 0.0 and events == []

    def test_cache_reads_config_key(self, monkeypatch):
        from backend.app.services import inference as inf
        inf._sampling_policy_cache = None
        seen = {}
        async def fake_cfg(db, key, default=None):
            seen["key"] = key
            return {"m": {"max_tokens": 7}}
        class _Ctx:
            async def __aenter__(self): return None
            async def __aexit__(self, *a): return False
        monkeypatch.setattr(inf.crud, "get_config_json", fake_cfg)
        monkeypatch.setattr("backend.app.db.session.get_async_db_context", lambda: _Ctx())
        pols = asyncio.run(inf._get_sampling_policies())
        assert seen["key"] == sp.CONFIG_KEY and pols == {"m": sp.SamplingPolicy(None, 7)}
        inf._sampling_policy_cache = None

    def test_hook_follows_every_reasoning_policy_call(self):
        # Both attempt paths (streaming and non-streaming) must clamp right
        # after resolving reasoning, so no dialect escapes the guard rails.
        src = (Path(__file__).resolve().parents[2] / "services" / "inference.py").read_text()
        pairs = re.findall(r"await apply_reasoning_policy\(request, job\.model, _policy_target, self\._settings\)\n\s+await apply_sampling_policy\(request, job\.model\)", src)
        assert len(pairs) == 2 == src.count("await apply_reasoning_policy(request, job.model")
