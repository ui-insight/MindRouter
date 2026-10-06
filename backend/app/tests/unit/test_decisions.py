############################################################
#
# mindrouter - unit tests for the EXPERIMENTAL /v1/decisions
# capability (services/decisions + api/decisions_api)
#
# The capability is a transitional Jev-style "System One" layer: typed
# questions scored by one-token constrained sampling on an existing vLLM
# server. These tests pin (1) the request contract and its hard limits,
# (2) the prompt layout and order-averaging arithmetic borrowed from
# open-alternative-jev, (3) the exact vLLM HTTP fields the adapter sends
# and how it reads the reply, (4) TypeSafe's System One wire format (Jev's
# POST /v1/systemone) in and out, and (5) the route's gating/metering. No
# network, no model.
#
############################################################

"""Unit tests for the decisions capability."""

import json
import math
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

import backend.app.api.decisions_api as api
from backend.app.services import decisions as pkg
from backend.app.services.decisions import DecisionBackendError, DecisionOutcome
from backend.app.services.decisions import systemone as so
from backend.app.services.decisions import upstream as up
from backend.app.services.decisions import vllm_logprobs as vl
from backend.app.services.decisions.schema import (
    MAX_OPTIONS,
    MAX_QUESTIONS,
    MAX_STATE_CHARS,
    SCORE_SEMANTICS,
    DecisionRequest,
    DecisionResult,
    DecisionUsage,
)
from backend.app.services.decisions.scoring import (
    INSTRUCTION,
    LabelReadout,
    combine,
    option_orders,
    render_turn,
    softmax,
)

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _req(**over):
    body = {
        "model": "qwen3.8-27b",
        "state": "Ticket: export button does nothing since the release.",
        "questions": [
            {"id": "escalate", "type": "boolean", "question": "Escalate?"},
            {"id": "cat", "type": "choice", "question": "Category?", "options": ["bug", "billing", "question"]},
        ],
    }
    body.update(over)
    return DecisionRequest.model_validate(body)


def _chat_reply(label_ids, logprobs, sampled_idx, *, as_ids=True, prompt_tokens=100, cached=None, drop=(),
                others=()):
    """A vLLM chat completion carrying label logprobs like 0.29 returns them.
    ``others`` are (token id, logprob) of tokens in the top list that are not
    labels: the list is the model's raw top 20, whatever those tokens are."""
    def tok(i):
        return f"token_id:{label_ids[i]}" if as_ids else "ABCDEFGHIJKLMNOPQRST"[i]

    top = [
        {"token": tok(i), "logprob": lp, "bytes": None}
        for i, lp in enumerate(logprobs)
        if i not in drop
    ] + [{"token": f"token_id:{tid}", "logprob": lp, "bytes": None} for tid, lp in others]
    top.sort(key=lambda t: -t["logprob"])
    usage = {"prompt_tokens": prompt_tokens, "completion_tokens": 1, "total_tokens": prompt_tokens + 1}
    if cached is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached}
    return {
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "A"},
            "logprobs": {"content": [{
                "token": tok(sampled_idx), "logprob": logprobs[sampled_idx], "bytes": None, "top_logprobs": top,
            }]},
            "finish_reason": "length",
        }],
        "usage": usage,
    }


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx
            raise httpx.HTTPStatusError("boom", request=MagicMock(), response=self)


def _encode_like_httpx(payload):
    """httpx serializes a `json=` body as UTF-8 with allow_nan=False. The fakes
    do the same so input that cannot be sent fails in tests as it would live."""
    json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")


class _FakeClient:
    """Stands in for httpx.AsyncClient; `handler(url, json)` returns a payload."""

    calls: list = []
    handler = None

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None):
        _encode_like_httpx(json)
        _FakeClient.calls.append((url, json))
        return _FakeClient.handler(url, json)


def _label_ids(n=26):
    return [32 + i for i in range(n)]  # Qwen byte-level BPE: 'A' == 32


def _default_handler(scores_by_n=None):
    """Tokenize → single ids; chat → logprobs by option count (A best)."""
    ids = _label_ids()

    def handler(url, body):
        if url.endswith("/tokenize"):
            letter = body["prompt"]
            return _FakeResponse({"count": 1, "max_model_len": 32768, "tokens": [ids["ABCDEFGHIJKLMNOPQRSTUVWXYZ".index(letter)]]})
        n = len(body["allowed_token_ids"])
        scores = (scores_by_n or {}).get(n) or [-0.2 - i for i in range(n)]
        best = max(range(n), key=scores.__getitem__)
        return _FakeResponse(_chat_reply(ids[:n], scores, best, cached=64))

    return handler


def _fake_registry(backends, aliases=None, open_circuits=()):
    """Registry stub. ``get_backends_with_model`` drops unhealthy backends
    like the SQL does; ``pick_available_backend`` is the REAL method run
    against these stubs, so the selection rules are tested, not mocked."""
    from backend.app.core.telemetry.registry import BackendRegistry

    reg = MagicMock()
    # Like the SQL: healthy backends only. Whether a copy of the model sees is
    # NOT decided here; the real picker reads it from each backend's model rows.
    reg.get_backends_with_model = AsyncMock(
        side_effect=lambda model_name: [b for b in backends if getattr(b.status, "value", b.status) == "healthy"])
    reg.is_backend_available = AsyncMock(side_effect=lambda bid: bid not in open_circuits)
    # No decision server is registered as a backend unless a test says so.
    reg.decision_server_state = AsyncMock(return_value=(None, None))
    reg.report_live_failure, reg.report_live_success = AsyncMock(), AsyncMock()
    reg.resolve_alias = MagicMock(side_effect=lambda m: ((aliases or {}).get(m, m), None))

    async def _pick(model_name, *, engine=None, multimodal=False, exclude=()):
        return await BackendRegistry.pick_available_backend(
            reg, model_name, engine=engine, multimodal=multimodal, exclude=exclude)

    reg.pick_available_backend = _pick
    return reg


def _vllm_backend(id=7, name="aspen5-gpu2-qwen3.8-27b", healthy=True, engine=None, sees_images=False,
                  model="qwen/qwen3.8-27b", other_models=()):
    """A backend stub. Its model rows carry ``supports_multimodal`` like the
    real ones; blind unless a test says otherwise, so no image test passes by default."""
    from types import SimpleNamespace

    from backend.app.db.models import BackendEngine
    b = MagicMock()
    b.id, b.name, b.url = id, name, f"https://node{id}:8002"
    b.models = [SimpleNamespace(name=model, supports_multimodal=sees_images),
                *(SimpleNamespace(name=n, supports_multimodal=sees) for n, sees in other_models)]
    b.engine = engine or BackendEngine.VLLM
    b.status = MagicMock(value="healthy" if healthy else "unhealthy")
    return b


@pytest.fixture
def fake_http(monkeypatch):
    _FakeClient.calls = []
    _FakeClient.handler = _default_handler()
    monkeypatch.setattr(vl.httpx, "AsyncClient", _FakeClient)
    return _FakeClient


# --------------------------------------------------------------------------
# 1. request contract
# --------------------------------------------------------------------------

class TestSchema:
    def test_three_question_types_parse(self):
        r = _req(questions=[
            {"id": "b", "type": "boolean", "question": "Yes?"},
            {"id": "c", "type": "choice", "question": "Which?", "options": [" x ", "y"]},
            {"id": "s", "type": "scale", "question": "Severity?", "min": 1, "max": 5},
        ])
        assert r.questions[0].options == ("yes", "no")
        assert r.questions[1].options == ["x", "y"]  # stripped
        assert r.questions[2].options == ("1", "2", "3", "4", "5")
        assert r.permutations == 1 and r.model == "qwen3.8-27b"

    def test_model_and_state_optional(self):
        r = DecisionRequest.model_validate({"questions": [{"id": "q", "type": "boolean", "question": "?"}]})
        assert r.model is None and r.state is None

    @pytest.mark.parametrize("bad", [
        {"questions": []},
        {"questions": [{"id": "q", "type": "boolean", "question": ""}]},
        {"questions": [{"id": "bad id", "type": "boolean", "question": "?"}]},
        {"questions": [{"id": "q", "type": "boolean", "question": "?"}] * 2},          # duplicate id
        {"questions": [{"id": "q", "type": "choice", "question": "?", "options": ["a"]}]},
        {"questions": [{"id": "q", "type": "choice", "question": "?", "options": ["a", "a"]}]},
        {"questions": [{"id": "q", "type": "choice", "question": "?", "options": ["a", ""]}]},
        {"questions": [{"id": "q", "type": "choice", "question": "?", "options": ["a", "b\nc"]}]},
        {"questions": [{"id": "q", "type": "choice", "question": "?", "options": [str(i) for i in range(MAX_OPTIONS + 1)]}]},
        {"questions": [{"id": "q", "type": "scale", "question": "?", "min": 3, "max": 3}]},
        {"questions": [{"id": "q", "type": "scale", "question": "?", "min": 0, "max": MAX_OPTIONS}]},
        {"questions": [{"id": "q", "type": "essay", "question": "?"}]},
        {"questions": [{"id": "q", "type": "boolean", "question": "?", "options": ["a", "b"]}], "permutations": 3},
        {"questions": [{"id": f"q{i}", "type": "boolean", "question": "?"} for i in range(MAX_QUESTIONS + 1)]},
        {"state": "x" * (MAX_STATE_CHARS + 1), "questions": [{"id": "q", "type": "boolean", "question": "?"}]},
    ])
    def test_hard_limits_reject(self, bad):
        with pytest.raises(ValidationError):
            DecisionRequest.model_validate(bad)

    def test_response_semantics_marker_is_fixed(self):
        assert SCORE_SEMANTICS == "normalized_label_likelihood"


# --------------------------------------------------------------------------
# 2. prompt layout + arithmetic (kept identical to open-alternative-jev)
# --------------------------------------------------------------------------

class TestScoring:
    def test_render_turn_matches_so1_layout(self):
        text = render_turn("Urgency?", ["low", "high"], "The site is down.")
        assert text == (
            "Choose the correct option. Reply with only its letter.\n"
            "\nContext:\nThe site is down.\n"
            "\nQuestion: Urgency?\nA. low\nB. high"
        )
        assert INSTRUCTION == "Choose the correct option. Reply with only its letter."

    def test_render_turn_without_state(self):
        assert render_turn("Q?", ["a", "b"], None) == f"{INSTRUCTION}\n\nQuestion: Q?\nA. a\nB. b"

    def test_state_is_a_shared_prefix_across_questions(self):
        s = "long shared state " * 50
        a = render_turn("first?", ["a", "b"], s)
        b = render_turn("second?", ["x", "y", "z"], s)
        prefix = f"{INSTRUCTION}\n\nContext:\n{s}\n\nQuestion: "
        assert a.startswith(prefix) and b.startswith(prefix)

    def test_option_orders(self):
        assert option_orders(3, 1) == [[0, 1, 2]]
        assert option_orders(3, 2) == [[0, 1, 2], [2, 1, 0]]
        assert option_orders(4, 3) == [[0, 1, 2, 3], [3, 2, 1, 0], [1, 2, 3, 0]]
        assert option_orders(2, 3) == [[0, 1], [1, 0]]  # only 2 distinct orders exist

    def test_softmax(self):
        p = softmax([0.0, math.log(3.0)])
        assert p == pytest.approx([0.25, 0.75])

    def test_combine_single_view_uses_sampled_label(self):
        # Floored second label makes the distribution approximate, but the
        # sampler's pick (position 1) is still the exact answer.
        c = combine(2, [[0, 1]], [LabelReadout([-0.5, -1.5], sampled=1, complete=False)])
        assert c.answer_index == 1 and c.complete is False
        assert c.likelihoods == pytest.approx(softmax([-0.5, -1.5]))
        assert c.logprobs == [-0.5, -1.5]
        assert c.label_mass == pytest.approx(math.exp(-0.5) + math.exp(-1.5))

    def test_combine_reversed_view_maps_back_and_averages(self):
        # View 1 shows [x, y]; view 2 shows [y, x]. Both prefer "y".
        r1 = LabelReadout([math.log(0.2), math.log(0.8)], sampled=1)
        r2 = LabelReadout([math.log(0.6), math.log(0.4)], sampled=0)
        c = combine(2, [[0, 1], [1, 0]], [r1, r2])
        assert c.likelihoods == pytest.approx([(0.2 + 0.4) / 2, (0.8 + 0.6) / 2])
        assert c.answer_index == 1
        assert c.logprobs == pytest.approx([math.log(0.2), math.log(0.8)])  # first view only

    def test_combine_label_mass_capped_at_one(self):
        c = combine(2, [[0, 1]], [LabelReadout([0.0, 0.0], sampled=0)])
        assert c.label_mass == 1.0


# --------------------------------------------------------------------------
# 3. reading vLLM's reply
# --------------------------------------------------------------------------

class TestParseReadout:
    def test_token_id_form(self):
        ids = _label_ids(3)
        r = vl.parse_readout(_chat_reply(ids, [-0.1, -2.0, -3.0], 0), ids)
        assert r.logprobs == [-0.1, -2.0, -3.0] and r.sampled == 0 and r.complete

    def test_bare_letter_form(self):
        ids = _label_ids(2)
        r = vl.parse_readout(_chat_reply(ids, [-1.0, -0.4], 1, as_ids=False), ids)
        assert r.logprobs == [-1.0, -0.4] and r.sampled == 1

    def test_missing_label_is_floored_and_flagged(self):
        ids = _label_ids(3)
        r = vl.parse_readout(_chat_reply(ids, [-0.1, -2.0, -9.0], 0, drop=(2,)), ids)
        assert r.complete is False
        assert r.logprobs == [-0.1, -2.0, -2.0]      # no likelier than the least likely token returned

    def test_a_missing_label_takes_the_lowest_value_in_the_list_label_or_not(self):
        # The list is the raw top 20: other tokens sit in it too. A label that
        # is not there is below ALL of them, not just below the other labels.
        ids = _label_ids(4)
        reply = _chat_reply(ids, [-0.05, -4.0, -99.0, -99.0], 0, drop=(2, 3),
                            others=[(9001, -3.0), (9002, -7.5), (9003, -11.25)])
        r = vl.parse_readout(reply, ids)
        assert r.logprobs == [-0.05, -4.0, -11.25, -11.25] and r.complete is False and r.sampled == 0

    def test_tokens_that_are_not_labels_are_never_read_as_labels(self):
        ids = _label_ids(2)
        # A stray token outranks both labels; it must not become anyone's value.
        r = vl.parse_readout(_chat_reply(ids, [-1.5, -2.5], 0, others=[(9001, -0.4)]), ids)
        assert r.logprobs == [-1.5, -2.5] and r.complete is True

    def test_a_reply_with_no_label_at_all_is_an_error_not_a_uniform_guess(self):
        # A server that ignored allowed_token_ids: the list holds other tokens
        # only and the sampled token is not a label. There is nothing to read.
        ids = _label_ids(2)
        reply = _chat_reply(ids, [-1.0, -2.0], 0, drop=(0, 1), others=[(9001, -0.1), (9002, -3.0)])
        reply["choices"][0]["logprobs"]["content"][0]["token"] = "token_id:9001"
        with pytest.raises(DecisionBackendError) as e:
            vl.parse_readout(reply, ids)
        assert e.value.status_code == 502 and "no label logprobs" in str(e.value)

    def test_all_labels_present_among_other_tokens_is_complete(self):
        ids = _label_ids(3)
        r = vl.parse_readout(_chat_reply(ids, [-0.2, -3.0, -6.0], 0, others=[(9000 + i, -8.0 - i) for i in range(17)]), ids)
        assert r.logprobs == [-0.2, -3.0, -6.0] and r.complete is True

    def test_a_malformed_entry_in_the_list_is_skipped(self):
        ids = _label_ids(2)
        reply = _chat_reply(ids, [-0.3, -1.4], 0)
        reply["choices"][0]["logprobs"]["content"][0]["top_logprobs"] += [
            "junk", {"token": "token_id:9001"}, {"token": "token_id:9002", "logprob": True}, {"logprob": -50.0}]
        r = vl.parse_readout(reply, ids)
        assert r.logprobs == [-0.3, -1.4] and r.complete is True

    def test_no_logprobs_is_a_backend_error(self):
        with pytest.raises(DecisionBackendError) as e:
            vl.parse_readout({"choices": [{"message": {"content": "A"}}]}, _label_ids(2))
        assert e.value.status_code == 502


# --------------------------------------------------------------------------
# 4. the adapter against a fake vLLM HTTP server
# --------------------------------------------------------------------------

class TestVLLMLogprobsBackend:
    async def _decide(self, req, backends=None, fanout=8):
        backend = vl.VLLMLogprobsBackend()
        with patch.object(vl, "get_registry", return_value=_fake_registry(backends or [_vllm_backend()])):
            return backend, await backend.decide(req, "qwen3.8-27b", fanout=fanout)

    async def test_sends_one_token_constrained_scoring_calls(self, fake_http):
        _, out = await self._decide(_req())
        chat = [(u, b) for u, b in fake_http.calls if u.endswith("/v1/chat/completions")]
        assert len(chat) == 2 and out.usage.backend_calls == 2
        url, body = chat[1]
        assert url == "https://node7:8002/v1/chat/completions"
        assert body["max_tokens"] == 1 and body["temperature"] == 0.0 and body["stream"] is False
        assert body["logprobs"] is True and body["allowed_token_ids"] == _label_ids(3)
        # The server's whole top list, whatever the option count, and NOT
        # logprob_token_ids: vLLM answers 500 to that field whenever another
        # sequence in the batch carries draft tokens (speculative decoding),
        # i.e. whenever the replica is doing anything else.
        assert body["top_logprobs"] == 20 and "logprob_token_ids" not in body
        assert body["return_tokens_as_token_ids"] is True
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        assert body["messages"] == [{"role": "user", "content": render_turn(
            "Category?", ["bug", "billing", "question"], "Ticket: export button does nothing since the release.")}]

    async def test_results_usage_and_backend_attribution(self, fake_http):
        _, out = await self._decide(_req())
        assert out.backend_id == 7 and out.backend_name == "aspen5-gpu2-qwen3.8-27b"
        esc, cat = out.results
        assert esc.type == "boolean" and esc.answer is True and esc.likelihood_true == esc.likelihoods["yes"]
        assert cat.type == "choice" and cat.answer == "bug" and cat.likelihood == cat.likelihoods["bug"]
        assert set(cat.likelihoods) == {"bug", "billing", "question"} and sum(cat.likelihoods.values()) == pytest.approx(1.0)
        assert cat.logprobs["bug"] == -0.2 and cat.complete
        assert out.usage.prompt_tokens == 200 and out.usage.scoring_tokens == 2 and out.usage.total_tokens == 202
        assert out.usage.cached_tokens == 128

    async def test_scale_question_reports_expected_value(self, fake_http):
        fake_http.handler = _default_handler({3: [math.log(0.2), math.log(0.3), math.log(0.5)]})
        _, out = await self._decide(_req(questions=[{"id": "sev", "type": "scale", "question": "Severity?", "min": 1, "max": 3}]))
        (sev,) = out.results
        assert sev.answer == 3 and sev.expected_value == pytest.approx(0.2 * 1 + 0.3 * 2 + 0.5 * 3)

    async def test_permutations_two_reverses_options_and_averages(self, fake_http):
        # The fake always scores the FIRST presented option highest, so the
        # reversed view votes for the other option; averaging must still be
        # reported in the caller's option order and sum to 1.
        _, out = await self._decide(_req(permutations=2, questions=[
            {"id": "c", "type": "choice", "question": "?", "options": ["p", "q"]}]))
        chat = [b for u, b in fake_http.calls if u.endswith("/v1/chat/completions")]
        assert len(chat) == 2
        assert chat[0]["messages"][0]["content"].endswith("A. p\nB. q")
        assert chat[1]["messages"][0]["content"].endswith("A. q\nB. p")
        (c,) = out.results
        assert list(c.likelihoods) == ["p", "q"]
        assert sum(c.likelihoods.values()) == pytest.approx(1.0)
        assert c.likelihoods["p"] == pytest.approx(c.likelihoods["q"])  # symmetric fake → tie
        assert out.usage.backend_calls == 2

    async def test_label_ids_are_tokenized_once_per_backend(self, fake_http):
        backend = vl.VLLMLogprobsBackend()
        with patch.object(vl, "get_registry", return_value=_fake_registry([_vllm_backend()])):
            await backend.decide(_req(), "qwen3.8-27b", fanout=4)
            n1 = sum(1 for u, _ in fake_http.calls if u.endswith("/tokenize"))
            await backend.decide(_req(), "qwen3.8-27b", fanout=4)
            n2 = sum(1 for u, _ in fake_http.calls if u.endswith("/tokenize"))
        assert n1 == 3 and n2 == 3  # max(2 boolean, 3 choice) letters, cached after

    async def test_multi_token_label_is_rejected_400(self, fake_http):
        def handler(url, body):
            if url.endswith("/tokenize"):
                return _FakeResponse({"tokens": [1, 2]})
            raise AssertionError("must not score")
        fake_http.handler = handler
        with pytest.raises(DecisionBackendError) as e:
            await self._decide(_req())
        assert e.value.status_code == 400

    async def test_no_request_ever_carries_logprob_token_ids(self, fake_http):
        qs = [{"id": "b", "type": "boolean", "question": "?"},
              {"id": "c", "type": "choice", "question": "?", "options": [f"o{i}" for i in range(20)]},
              {"id": "s", "type": "scale", "question": "?", "min": 1, "max": 10}]
        await self._decide(_req(questions=qs, permutations=2))
        chat = [b for u, b in fake_http.calls if u.endswith("/v1/chat/completions")]
        assert len(chat) == 6
        assert all("logprob_token_ids" not in b and b["top_logprobs"] == 20 and b["max_tokens"] == 1 for b in chat)

    async def test_twenty_options_with_labels_outside_the_top_list(self, fake_http):
        # What a real replica returns for a 20-way choice: the likely labels,
        # some other tokens, and the unlikely labels missing altogether.
        ids = _label_ids(20)
        scores = [-0.05, -3.5, -4.0, -6.0, -7.0] + [-30.0] * 15

        def handler(url, body):
            if url.endswith("/tokenize"):
                return _default_handler()(url, body)
            assert len(body["allowed_token_ids"]) == 20
            return _FakeResponse(_chat_reply(ids, scores, 0, drop=range(5, 20),
                                             others=[(9000 + i, -8.0 - i) for i in range(15)]))

        fake_http.handler = handler
        _, out = await self._decide(_req(questions=[
            {"id": "c", "type": "choice", "question": "?", "options": [f"o{i}" for i in range(20)]}]))
        (c,) = out.results
        assert c.answer == "o0" and c.complete is False
        assert sum(c.likelihoods.values()) == pytest.approx(1.0)
        assert c.likelihoods["o0"] > 0.9 and c.likelihoods["o1"] == pytest.approx(math.exp(-3.5) / sum(
            math.exp(v) for v in [-0.05, -3.5, -4.0, -6.0, -7.0] + [-22.0] * 15))
        assert max(c.likelihoods[f"o{i}"] for i in range(5, 20)) < 1e-8      # the bound, not a guess

    async def test_no_vllm_backend_is_503(self, fake_http):
        from backend.app.db.models import BackendEngine
        with pytest.raises(DecisionBackendError) as e:
            await self._decide(_req(), backends=[_vllm_backend(engine=BackendEngine.OLLAMA), _vllm_backend(healthy=False)])
        assert e.value.status_code == 503

    async def test_backend_http_failure_is_502(self, fake_http):
        good = _default_handler()

        def handler(url, body):
            if url.endswith("/tokenize"):
                return good(url, body)
            return _FakeResponse({"error": "cuda"}, status_code=500)
        fake_http.handler = handler
        with pytest.raises(DecisionBackendError) as e:
            await self._decide(_req())
        assert e.value.status_code == 502

    # ---- one sick replica is not the whole model

    def _replicas(self, n=3):
        return [_vllm_backend(id=10 + i, name=f"replica-{i}") for i in range(n)]

    def _failing(self, fake_http, fail, *, only_first=True):
        """Make the first replica that is asked (or every one) answer ``fail``:
        an HTTP status, or an exception to raise on the scoring call."""
        good, seen = _default_handler(), []

        def handler(url, body):
            host = url.split("/v1/")[0].split("/tokenize")[0]
            if host not in seen:
                seen.append(host)
            if url.endswith("/tokenize"):
                return good(url, body)
            if not only_first or host == seen[0]:
                if isinstance(fail, BaseException):
                    raise fail
                return _FakeResponse({"error": {"message": "SECRET PROMPT TEXT", "type": "InternalServerError",
                                                "code": fail}}, status_code=fail)
            return good(url, body)

        fake_http.handler = handler
        return seen

    @pytest.mark.parametrize("fail", [500, 502, 503, 504, httpx.ConnectError("refused"),
                                      httpx.ConnectTimeout("no route")])
    async def test_a_sick_replica_is_retried_once_on_another(self, fake_http, fail):
        seen = self._failing(fake_http, fail)
        _, out = await self._decide(_req(), backends=self._replicas())
        assert len(seen) == 2 and seen[0] != seen[1]                    # a different replica, never the same again
        assert out.backend_id == int(seen[1].split("node")[1].split(":")[0])    # the answer is credited to the one that gave it
        assert out.results[1].answer == "bug" and out.usage.backend_calls == 2
        scored_on = {u.split("/v1/")[0] for u, _ in fake_http.calls if u.endswith("/v1/chat/completions")}
        assert scored_on == set(seen)
        # The labels are looked up on the replica that is asked, not carried over.
        assert {u.split("/tokenize")[0] for u, _ in fake_http.calls if u.endswith("/tokenize")} == set(seen)

    @pytest.mark.parametrize("fail", [503, 500, httpx.ConnectError("refused"), httpx.ConnectTimeout("no route")])
    async def test_a_replica_that_is_down_fails_at_the_tokenizer_and_is_retried_too(self, fake_http, fail):
        # Looking up the letter tokens is the first thing asked of a replica,
        # so a dead one never reaches the scoring call. (Found by running the
        # real code against a dead address: the scoring-only retry missed it.)
        good, seen = _default_handler(), []

        def handler(url, body):
            host = url.split("/v1/")[0].split("/tokenize")[0]
            if host not in seen:
                seen.append(host)
            if host == seen[0]:
                assert url.endswith("/tokenize"), "a replica whose tokenizer failed must not be scored on"
                if isinstance(fail, BaseException):
                    raise fail
                return _FakeResponse({"error": "x"}, status_code=fail)
            return good(url, body)

        fake_http.handler = handler
        _, out = await self._decide(_req(), backends=self._replicas())
        assert len(seen) == 2 and out.backend_id == int(seen[1].split("node")[1].split(":")[0])
        assert out.results[1].answer == "bug"

    @pytest.mark.parametrize("fail", [400, 404, httpx.ReadTimeout("slow")])
    async def test_a_tokenizer_failure_that_is_not_sickness_is_not_retried(self, fake_http, fail):
        seen = []

        def handler(url, body):
            seen.append(url)
            if isinstance(fail, BaseException):
                raise fail
            return _FakeResponse({"error": "x"}, status_code=fail)

        fake_http.handler = handler
        with pytest.raises(DecisionBackendError) as e:
            await self._decide(_req(), backends=self._replicas())
        assert len(seen) == 1 and e.value.status_code == 502
        assert str(e.value) == "decision backend tokenizer unreachable"

    async def test_at_most_two_replicas_are_tried(self, fake_http):
        seen = self._failing(fake_http, 500, only_first=False)
        with pytest.raises(DecisionBackendError) as e:
            await self._decide(_req(), backends=self._replicas(5))
        assert len(seen) == 2 and e.value.status_code == 502
        assert str(e.value) == "decision backend returned HTTP 500"
        assert type(e.value) is DecisionBackendError                    # the private retry type does not leak

    async def test_a_single_replica_fails_without_a_second_attempt(self, fake_http):
        seen = self._failing(fake_http, 500, only_first=False)
        with pytest.raises(DecisionBackendError) as e:
            await self._decide(_req(), backends=self._replicas(1))
        assert len(seen) == 1 and e.value.status_code == 502
        assert sum(1 for u, _ in fake_http.calls if u.endswith("/v1/chat/completions")) == 1

    async def test_a_replica_with_an_open_circuit_is_not_the_second_choice(self, fake_http):
        seen = self._failing(fake_http, 500, only_first=False)
        replicas = self._replicas(2)
        backend = vl.VLLMLogprobsBackend()
        with patch.object(vl, "get_registry", return_value=_fake_registry(replicas, open_circuits={11})):
            with pytest.raises(DecisionBackendError):
                await backend.decide(_req(), "qwen3.8-27b", fanout=8)
        assert seen == ["https://node10:8002"]

    @pytest.mark.parametrize("fail", [400, 404, 422, httpx.ReadTimeout("slow"), httpx.RemoteProtocolError("cut")])
    async def test_what_would_fail_anywhere_or_already_waited_is_not_sent_round_again(self, fake_http, fail):
        seen = self._failing(fake_http, fail, only_first=False)
        with pytest.raises(DecisionBackendError) as e:
            await self._decide(_req(), backends=self._replicas())
        assert len(seen) == 1 and e.value.status_code == 502

    async def test_the_engines_error_class_is_logged_and_its_text_is_not(self, fake_http, monkeypatch):
        log = MagicMock()
        monkeypatch.setattr(vl, "logger", log)
        seen = self._failing(fake_http, 500)
        await self._decide(_req(), backends=self._replicas())
        failed_id = int(seen[0].split("node")[1].split(":")[0])
        log.warning.assert_any_call("decision_backend_http_error", backend_id=failed_id, status=500,
                                    error_type="InternalServerError")
        log.warning.assert_any_call("decision_backend_retry_on_another_replica", failed_backend_id=failed_id)
        assert "SECRET" not in repr(log.mock_calls)

    @pytest.mark.parametrize("body,expected", [
        ({"error": {"message": "x", "type": "BadRequestError", "code": 400}}, "BadRequestError"),
        ({"object": "error", "message": "x", "type": "invalid_request_error"}, "invalid_request_error"),
        ({"error": {"message": "x"}}, None),
        ({"error": {"type": "a prompt quoted here, with spaces"}}, None),       # not an identifier: not logged
        ({"error": {"type": "x" * 65}}, None),
        ({"error": {"type": 500}}, None),
        ({"error": "cuda"}, None), ([1, 2], None), ("Internal Server Error", None),
    ])
    def test_error_type_is_a_short_identifier_or_nothing(self, body, expected):
        assert vl._error_type(_FakeResponse(body)) == expected

    def test_error_type_of_a_plain_text_500_is_nothing(self):
        response = MagicMock()
        response.json.side_effect = ValueError("not json")       # what an unhandled engine exception returns
        assert vl._error_type(response) is None

    async def test_fanout_bounds_concurrent_scoring_calls(self, monkeypatch, fake_http):
        import asyncio
        inflight, peak = 0, 0
        base = _default_handler()

        class _SlowClient(_FakeClient):
            async def post(self, url, json=None):
                nonlocal inflight, peak
                if url.endswith("/tokenize"):
                    return base(url, json)
                inflight += 1
                peak = max(peak, inflight)
                await asyncio.sleep(0.005)
                inflight -= 1
                return base(url, json)
        monkeypatch.setattr(vl.httpx, "AsyncClient", _SlowClient)
        qs = [{"id": f"q{i}", "type": "boolean", "question": "?"} for i in range(12)]
        await self._decide(_req(questions=qs), fanout=3)
        assert peak <= 3

    async def test_backend_gate_bounds_calls_across_concurrent_requests(self, monkeypatch, fake_http):
        # Review fix 4: fanout is per request; the per-backend gate must hold
        # across requests, or several keys together saturate one replica.
        import asyncio
        inflight, peak = 0, 0
        base = _default_handler()

        class _SlowClient(_FakeClient):
            async def post(self, url, json=None):
                nonlocal inflight, peak
                if url.endswith("/tokenize"):
                    return base(url, json)
                inflight += 1
                peak = max(peak, inflight)
                await asyncio.sleep(0.005)
                inflight -= 1
                return base(url, json)
        monkeypatch.setattr(vl.httpx, "AsyncClient", _SlowClient)
        backend = vl.VLLMLogprobsBackend()
        qs = [{"id": f"q{i}", "type": "boolean", "question": "?"} for i in range(6)]
        with patch.object(vl, "get_registry", return_value=_fake_registry([_vllm_backend()])):
            await asyncio.gather(*(
                backend.decide(_req(questions=qs), "qwen3.8-27b", fanout=8, backend_concurrency=2)
                for _ in range(4)
            ))
        assert peak <= 2

    async def test_first_view_warms_the_prefix_cache_before_the_rest(self, monkeypatch, fake_http):
        # Review fix 5: with a state, view 1 must finish before views 2..n start,
        # so they hit the prefix cache instead of all prefilling the state.
        import asyncio
        events = []
        base = _default_handler()

        class _Client(_FakeClient):
            async def post(self, url, json=None):
                if url.endswith("/tokenize"):
                    return base(url, json)
                q = json["messages"][0]["content"].split("Question: ")[1].split("\n")[0]
                events.append(("start", q))
                await asyncio.sleep(0.002)
                events.append(("end", q))
                return base(url, json)
        monkeypatch.setattr(vl.httpx, "AsyncClient", _Client)
        qs = [{"id": f"q{i}", "type": "boolean", "question": f"Q{i}?"} for i in range(4)]
        await self._decide(_req(questions=qs), fanout=8)
        assert events[0] == ("start", "Q0?") and events[1] == ("end", "Q0?")
        assert {e for e in events[2:] if e[0] == "start"} == {("start", f"Q{i}?") for i in (1, 2, 3)}

    async def test_failed_view_cancels_its_siblings(self, monkeypatch, fake_http):
        # Review fix 8: the first failure must cancel the in-flight siblings,
        # not leave them running against a closed client.
        import asyncio
        finished = []
        base = _default_handler()

        class _Client(_FakeClient):
            async def post(self, url, json=None):
                if url.endswith("/tokenize"):
                    return base(url, json)
                q = json["messages"][0]["content"].split("Question: ")[1].split("\n")[0]
                if q == "bad?":
                    await asyncio.sleep(0.001)
                    return _FakeResponse({"error": "cuda"}, status_code=500)
                await asyncio.sleep(0.05)
                finished.append(q)
                return base(url, json)
        monkeypatch.setattr(vl.httpx, "AsyncClient", _Client)
        qs = [{"id": "bad", "type": "boolean", "question": "bad?"}] + [
            {"id": f"s{i}", "type": "boolean", "question": f"slow{i}?"} for i in range(4)]
        with pytest.raises(DecisionBackendError) as e:
            await self._decide(_req(state=None, questions=qs), fanout=8)
        assert e.value.status_code == 502
        await asyncio.sleep(0.08)
        assert finished == []

    async def test_non_json_reply_is_502_not_500(self, fake_http):
        # Review fix 6: a proxy error page (resp.json() ValueError) is a backend error.
        good = _default_handler()

        class _HtmlResponse(_FakeResponse):
            def json(self):
                raise ValueError("Expecting value: line 1 column 1")

        def handler(url, body):
            if url.endswith("/tokenize"):
                return good(url, body)
            return _HtmlResponse({})
        fake_http.handler = handler
        with pytest.raises(DecisionBackendError) as e:
            await self._decide(_req())
        assert e.value.status_code == 502

    async def test_open_circuit_backend_is_never_picked(self, fake_http):
        # Review fix 9: selection goes through registry.pick_available_backend.
        reg = _fake_registry([_vllm_backend(id=7), _vllm_backend(id=8)], open_circuits={7})
        backend = vl.VLLMLogprobsBackend()
        with patch.object(vl, "get_registry", return_value=reg):
            for _ in range(10):
                out = await backend.decide(_req(), "qwen3.8-27b", fanout=4)
                assert out.backend_id == 8


# --------------------------------------------------------------------------
# 5. the System One wire format (TypeSafe Jev's POST /v1/systemone)
# --------------------------------------------------------------------------

# The exact body TypeSafe's Python SDK (typesafe-sdk 0.7.2) sent for
# client.system_one(state, questions), captured from its HTTP transport. If
# parse_request stops accepting this, a Jev client stops working.
_SDK_BODY = json.loads(
    '{"state": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel.",'
    ' "model": "jev-latest", "questions": {'
    '"department": {"type": "choice", "instructions": "Which team should handle this?",'
    ' "criteria": {"billing": "Payments, invoicing, refunds", "technical": "Bugs, outages", "sales": null}},'
    ' "urgency": {"type": "score", "instructions": "How urgent is this?",'
    ' "criteria": ["Can wait", "Needs attention this week", {"level": "today", "note": "blocking"}]},'
    ' "churn_risk": {"type": "noul", "instructions": "Does the user threaten to leave?",'
    ' "criteria": {"true": "Explicit threat to cancel", "false": "No such threat"}}}}'
)


def _results_for(plan, weights=None):
    """Synthetic scoring results: first option most likely unless told otherwise."""
    out = []
    for q in plan.decision_request.questions:
        n = len(q.options)
        raw = (weights or {}).get(q.id) or [math.exp(-0.3 * i) for i in range(n)]
        z = sum(raw)
        lk = {o: v / z for o, v in zip(q.options, raw)}
        out.append(DecisionResult(id=q.id, type=q.type, answer=q.options[0], likelihoods=lk,
                                  logprobs={o: math.log(v) for o, v in lk.items()}, label_mass=0.9))
    return out


class TestSystemOneWire:
    def test_sdk_request_compiles_to_the_decisions_pipeline(self):
        plan = so.parse_request(_SDK_BODY)
        assert plan.model_requested == "jev-latest" and plan.state.startswith("Hi, we were billed twice")
        assert [(q.jev_id, q.kind) for q in plan.questions] == [
            ("department", "choice"), ("urgency", "score"), ("churn_risk", "noul")]
        dept, urg, churn = plan.decision_request.questions
        # choice: "name: description", a null description leaves the bare name
        assert dept.type == "choice" and dept.options == [
            "billing: Payments, invoicing, refunds", "technical: Bugs, outages", "sales"]
        # score: levels in order; a structured level is shown as JSON
        assert urg.options == ["Can wait", "Needs attention this week", '{"level": "today", "note": "blocking"}']
        # noul: a yes/no question with both criteria spelled out
        assert churn.type == "boolean" and churn.question == (
            "Does the user threaten to leave?\nYes means: Explicit threat to cancel\nNo means: No such threat")

    def test_response_has_jevs_answer_shapes(self):
        plan = so.parse_request(_SDK_BODY)
        usage = DecisionUsage(prompt_tokens=300, scoring_tokens=3, total_tokens=303, cached_tokens=120, backend_calls=3)
        r = so.format_response(plan, _results_for(plan), usage, model="qwen3.8-27b", request_id="dec-1",
                               backend_name="vllm_logprobs")
        assert r["model"] == "qwen3.8-27b" and set(r["answers"]) == {"department", "urgency", "churn_risk"}
        dept, urg, churn = (r["answers"][k] for k in ("department", "urgency", "churn_risk"))
        assert set(dept) == {"type", "choice", "probabilities", "confidence"} and dept["choice"] == "billing"
        assert list(dept["probabilities"]) == ["billing", "technical", "sales"]
        assert sum(dept["probabilities"].values()) == pytest.approx(1.0)
        assert set(urg) == {"type", "score", "legend", "probabilities", "confidence"}
        assert urg["legend"] == {"0": "Can wait", "1": "Needs attention this week",
                                 "2": {"level": "today", "note": "blocking"}}  # levels echoed as sent
        assert list(urg["probabilities"]) == ["0", "1", "2"]
        assert urg["score"] == pytest.approx(sum(i * p for i, p in enumerate(urg["probabilities"].values())))
        assert churn == {"type": "noul", "noul": pytest.approx(1 / (1 + math.exp(-0.3)))}  # P(yes)
        # usage: billable input (prompt minus prefix-cached) and scoring tokens, both ints
        assert r["usage"] == {"input_tokens": 180, "output_tokens": 3}
        assert r["id"] == "dec-1" and r["metadata"]["score_semantics"] == SCORE_SEMANTICS
        assert r["metadata"]["questions"]["department"] == {"complete": True, "label_mass": 0.9, "permutations": 2}

    def test_confidence_follows_typesafes_formulas(self):
        # choice: (n * max - 1) / (n - 1); docs.typesafe.ai/confidence.md
        assert so.choice_confidence([0.88, 0.12, 0.0]) == pytest.approx((3 * 0.88 - 1) / 2)
        assert so.choice_confidence([1 / 3] * 3) == pytest.approx(0.0) and so.choice_confidence([1.0]) == 1.0
        # score: 1 - E|level - mode| / (same for a uniform spread), clamped to [0, 1]
        assert so.score_confidence([0.0, 0.95, 0.05]) == pytest.approx(1 - 0.05 / (2 / 3))
        assert so.score_confidence([0.44, 0.32, 0.24]) == 0.0
        assert so.score_confidence([0, 0, 1, 0, 0]) == 1.0

    def test_choice_reports_the_highest_probability_option(self):
        plan = so.parse_request({"state": "s", "model": "jev-latest", "questions": {
            "c": {"type": "choice", "instructions": "?", "criteria": {"a": None, "b": None, "c": None}}}})
        r = so.format_response(plan, _results_for(plan, {"q0": [0.1, 0.7, 0.2]}), None, model="m",
                               request_id="x", backend_name="b")
        assert r["answers"]["c"]["choice"] == "b" and r["answers"]["c"]["confidence"] == pytest.approx((3 * 0.7 - 1) / 2)

    def test_structured_state_and_instructions_are_rendered_as_json(self):
        plan = so.parse_request({"state": {"body": "Login fails", "n": 2}, "model": "jev-latest", "questions": {
            "u": {"type": "noul", "instructions": {"question": "Does `body` need action?"}}}})
        assert plan.state == '{"body": "Login fails", "n": 2}'
        assert plan.decision_request.questions[0].question == '{"question": "Does `body` need action?"}'

    def test_missing_instructions_are_allowed_like_typesafes_schema(self):
        plan = so.parse_request({"state": "s", "model": "jev-latest", "questions": {"u": {"type": "noul"}}})
        assert plan.decision_request.questions[0].question

    def test_single_option_needs_no_model_call(self):
        plan = so.parse_request({"state": "s", "model": "jev-latest", "questions": {
            "only": {"type": "choice", "instructions": "?", "criteria": {"x": "the only one"}},
            "lvl": {"type": "score", "instructions": "?", "criteria": ["just this"]}}})
        assert plan.decision_request is None
        r = so.format_response(plan, [], None, model="m", request_id="x", backend_name="b")
        assert r["answers"]["only"] == {"type": "choice", "choice": "x", "probabilities": {"x": 1.0}, "confidence": 1.0}
        assert r["answers"]["lvl"]["score"] == 0.0 and r["answers"]["lvl"]["confidence"] == 1.0
        assert r["usage"] == {"input_tokens": 0, "output_tokens": 0}

    @pytest.mark.parametrize("body,loc_tail", [
        ({"model": "jev-latest", "questions": {"q": {"type": "noul"}}}, ["state"]),
        ({"state": "s", "model": "jev-latest"}, ["questions"]),
        ({"state": "s", "model": "jev-latest", "questions": {}}, ["questions"]),
        ({"state": "s", "model": "jev-latest", "questions": {"q": {"type": "essay"}}}, ["questions", "q"]),
        ({"state": "s", "model": "jev-latest", "questions": {"q": {"type": "choice", "criteria": {}}}},
         ["questions", "q", "choice", "criteria"]),
        ({"state": "s", "model": "jev-latest", "questions": {
            "q": {"type": "choice", "criteria": {f"o{i}": None for i in range(MAX_OPTIONS + 1)}}}},
         ["questions", "q", "criteria"]),
        ({"state": "s", "model": "jev-latest", "questions": {"q": {"type": "score", "criteria": ["a"] * 11}}},
         ["questions", "q", "criteria"]),
        ({"state": "s", "model": "jev-latest", "questions": {"q": {"type": "score", "criteria": ["same", "same"]}}},
         ["questions", "q", "criteria"]),
    ])
    def test_invalid_requests_are_422_details_under_body(self, body, loc_tail):
        with pytest.raises(so.SystemOneValidationError) as e:
            so.parse_request(body)
        first = e.value.detail[0]
        assert first["loc"][0] == "body" and first["loc"][1:1 + len(loc_tail)] == loc_tail
        assert isinstance(first["msg"], str) and isinstance(first["type"], str)

    def test_too_many_questions_is_422(self):
        qs = {f"q{i}": {"type": "noul", "instructions": "?"} for i in range(MAX_QUESTIONS + 1)}
        with pytest.raises(so.SystemOneValidationError):
            so.parse_request({"state": "s", "model": "jev-latest", "questions": qs})

    def test_typesafe_model_list_shape(self):
        cfg = {"enabled": True, "default_model": "qwen3.8-27b", "allowed_models": ["qwen3.8-27b"]}
        models = so.typesafe_model_list(cfg)["models"]
        assert [m["name"] for m in models] == ["jev-latest", "jev-preview", "qwen3.8-27b"]
        assert all(set(m) == {"name", "description", "release_date"} for m in models)
        assert so.typesafe_model_list({**cfg, "enabled": False}) == {"models": []}


# --------------------------------------------------------------------------
# 6. the route: gating, metering, never storing the state
# --------------------------------------------------------------------------

class _FakeRequest:
    client = MagicMock(host="10.0.0.1")
    headers = {"user-agent": "typesafe-sdk/0.7.2"}

    def __init__(self, body, path="/v1/systemone"):
        self._body = body
        self.url = MagicMock(path=path)

    async def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


def _auth():
    u, k = MagicMock(id=1), MagicMock(id=2)
    return u, k


_BODY = {"state": "secret ticket text", "model": "jev-latest",
         "questions": {"escalate": {"type": "noul", "instructions": "Escalate?"}}}
_NEUTRAL = {"permutations": {"noul": 1, "choice": 1, "score": 1},
            "temperature": {"noul": 1.0, "choice": 1.0, "score": 1.0}}
_CFG = {"enabled": True, "default_model": "qwen3.8-27b", "allowed_models": ["qwen3.8-27b"],
        "max_state_chars": 1000, "fanout": 4, "backend_concurrency": 4, "upstreams": {}, **_NEUTRAL}


def _outcome():
    return DecisionOutcome(
        results=[DecisionResult(id="q0", type="boolean", answer=True, likelihoods={"yes": 0.8, "no": 0.2},
                                logprobs={"yes": -0.3, "no": -1.7}, label_mass=0.9, likelihood_true=0.8)],
        usage=DecisionUsage(prompt_tokens=120, scoring_tokens=1, total_tokens=121, cached_tokens=64, backend_calls=1),
        backend_id=7, backend_name="b7",
    )


def _crud():
    crud = MagicMock()
    crud.create_request = AsyncMock(return_value=MagicMock(id=55))
    for n in ("update_request_started", "update_request_completed", "update_request_failed",
              "update_quota_usage", "incr_quota_redis"):
        setattr(crud, n, AsyncMock())
    return crud


async def _call(body, cfg=None, availability="available", outcome=None, registry=None, path="/v1/systemone",
                crud=None, db=None, decide=None, upstream_backend=None):
    backend = MagicMock(name="vllm_logprobs")
    backend.name = "vllm_logprobs"
    if decide is not None:
        backend.decide = AsyncMock(side_effect=decide)
    elif isinstance(outcome, Exception):
        backend.decide = AsyncMock(side_effect=outcome)
    else:
        backend.decide = AsyncMock(return_value=outcome or _outcome())
    crud = crud or _crud()
    db = db or MagicMock(commit=AsyncMock(), rollback=AsyncMock())
    quota = AsyncMock()
    response = MagicMock(headers={})
    with (
        patch.object(api, "get_decisions_config", AsyncMock(return_value={**_CFG, **(cfg or {})})),
        patch.object(api, "get_registry", return_value=registry or _fake_registry([])),
        patch.object(api, "model_availability", AsyncMock(return_value=availability)),
        patch.object(api, "_check_quota", quota),
        patch.object(api, "get_decision_backend", return_value=backend),
        patch.object(api, "get_upstream_backend", return_value=upstream_backend),
        patch.object(api, "crud", crud),
        patch.object(api, "bind_request_context"),
    ):
        try:
            result = await api.systemone(_FakeRequest(body, path), response, db=db, auth=_auth())
        except HTTPException as e:
            e.mocks = (backend, crud, quota, db)  # let failure tests inspect the audit trail
            raise
    return result, backend, crud, quota, db, response


class TestRoute:
    async def test_success_is_a_jev_response_and_is_metered(self):
        result, backend, crud, quota, db, response = await _call(_BODY)
        assert result["model"] == "qwen3.8-27b"
        assert result["answers"] == {"escalate": {"type": "noul", "noul": 0.8}}
        assert result["usage"] == {"input_tokens": 56, "output_tokens": 1}
        assert result["id"].startswith("dec-") and response.headers["x-typesafe-request-id"] == result["id"]
        quota.assert_awaited_once()
        backend.decide.assert_awaited_once()
        assert backend.decide.call_args.kwargs == {"fanout": 4, "backend_concurrency": 4}
        kw = crud.update_request_completed.call_args.kwargs
        assert kw == {"prompt_tokens": 120, "completion_tokens": 1, "tokens_estimated": False, "backend_id": 7}
        # 120 prompt - 64 prefix-cached + 1 scoring: the cached state is not charged twice.
        crud.update_quota_usage.assert_awaited_once_with(db, 1, 57)
        crud.incr_quota_redis.assert_awaited_once_with(1, 57)
        db.commit.assert_awaited()

    async def test_state_is_never_stored(self):
        _, _, crud, _, _, _ = await _call(_BODY)
        kw = crud.create_request.call_args.kwargs
        assert kw["endpoint"] == "/v1/systemone" and kw["model"] == "qwen3.8-27b"
        assert "messages" not in kw and "prompt" not in kw
        assert "secret" not in json.dumps(kw["parameters"]) and "Escalate" not in json.dumps(kw["parameters"])
        assert kw["parameters"]["questions"] == 1 and kw["parameters"]["state_chars"] == len("secret ticket text")
        assert kw["parameters"]["types"] == ["noul"] and kw["parameters"]["model_requested"] == "jev-latest"
        assert kw["parameters"]["permutations"] == "default"   # the caller did not name it

    async def test_decisions_path_is_an_alias_with_the_same_shape(self):
        result, _, crud, _, _, _ = await _call(_BODY, path="/v1/decisions")
        assert result["answers"]["escalate"]["type"] == "noul"
        assert crud.create_request.call_args.kwargs["endpoint"] == "/v1/decisions"

    @pytest.mark.parametrize("model", ["jev-latest", "jev-preview", None, "qwen3.8-27b"])
    async def test_jev_aliases_and_allowed_names_reach_the_default_model(self, model):
        body = {k: v for k, v in {**_BODY, "model": model}.items() if v is not None}
        result, backend, _, _, _, _ = await _call(body)
        assert backend.decide.call_args.args[1] == "qwen3.8-27b" and result["model"] == "qwen3.8-27b"

    @pytest.mark.parametrize("model", ["gpt-oss-120b", "jev-1.13.0"])
    async def test_other_models_are_422_on_the_model_field(self, model):
        # A pinned TypeSafe version is refused rather than silently answered by another model.
        with pytest.raises(HTTPException) as e:
            await _call({**_BODY, "model": model})
        assert e.value.status_code == 422 and e.value.detail[0]["loc"] == ["body", "model"]
        assert "jev-latest" in e.value.detail[0]["msg"]

    async def test_alias_in_allow_list_admits_the_resolved_model(self):
        # Review fix 3: both sides of the allow-list check are alias-resolved.
        reg = _fake_registry([], aliases={"default-decisions": "qwen3.8-27b"})
        _, backend, _, _, _, _ = await _call(
            {**_BODY, "model": "qwen3.8-27b"}, cfg={"allowed_models": ["default-decisions"]}, registry=reg)
        assert backend.decide.call_args.args[1] == "qwen3.8-27b"

    async def test_disabled_is_404_before_any_work(self):
        # 404, not 503: TypeSafe's SDK retries 5xx and retrying cannot help.
        with pytest.raises(HTTPException) as e:
            await _call(_BODY, cfg={"enabled": False})
        assert e.value.status_code == 404
        e.value.mocks[0].decide.assert_not_awaited()
        e.value.mocks[1].create_request.assert_not_awaited()

    async def test_unavailable_model_is_503_with_retry_after_and_a_string_detail(self):
        with pytest.raises(HTTPException) as e:
            await _call(_BODY, availability="unavailable")
        assert e.value.status_code == 503 and e.value.headers.get("Retry-After")
        assert isinstance(e.value.detail, str)  # the SDK shows `detail` strings as the error message

    async def test_state_over_admin_ceiling_is_422_on_state(self):
        with pytest.raises(HTTPException) as e:
            await _call({**_BODY, "state": "x" * 1001})
        assert e.value.status_code == 422 and e.value.detail[0]["loc"] == ["body", "state"]

    async def test_validation_errors_are_422_in_fastapi_shape(self):
        with pytest.raises(HTTPException) as e:
            await _call({"state": "s", "model": "jev-latest", "questions": {"q": {"type": "choice"}}})
        assert e.value.status_code == 422
        assert e.value.detail[0]["loc"][:3] == ["body", "questions", "q"]

    async def test_invalid_json_is_422(self):
        with pytest.raises(HTTPException) as e:
            await _call(ValueError("nope"))
        assert e.value.status_code == 422 and e.value.detail[0]["type"] == "json_invalid"

    async def test_audit_row_is_started_and_committed_before_dialing_out(self):
        # Review fixes 2 + 10: started_at is set and the transaction is closed
        # before the fan-out, so no connection or api_keys lock is held across it.
        order = []

        async def decide(*a, **k):
            order.append("decide")
            return _outcome()

        crud = _crud()
        crud.update_request_started = AsyncMock(side_effect=lambda *a, **k: order.append("started"))
        db = MagicMock(commit=AsyncMock(side_effect=lambda: order.append("commit")), rollback=AsyncMock())
        await _call(_BODY, crud=crud, db=db, decide=decide)
        assert order[:3] == ["started", "commit", "decide"]
        assert crud.update_request_started.call_args.kwargs == {"backend_id": None}

    async def test_backend_error_status_is_propagated_and_row_marked_failed(self):
        with pytest.raises(HTTPException) as e:
            await _call(_BODY, outcome=DecisionBackendError("unreachable", 502))
        assert e.value.status_code == 502 and isinstance(e.value.detail, str)
        _, crud, _, _ = e.value.mocks
        crud.update_request_failed.assert_awaited_once()
        crud.update_request_completed.assert_not_awaited()
        crud.update_quota_usage.assert_not_awaited()

    async def test_unexpected_error_is_500_and_still_audited(self):
        # Review fix 6: an unmapped exception keeps the audit row (failed) and answers 500.
        with pytest.raises(HTTPException) as e:
            await _call(_BODY, outcome=RuntimeError("bug"))
        assert e.value.status_code == 500
        _, crud, _, _ = e.value.mocks
        crud.update_request_failed.assert_awaited_once()
        assert crud.update_request_failed.call_args.kwargs["error_code"] == "500"
        crud.update_quota_usage.assert_not_awaited()

    async def test_quota_charges_full_prompt_when_server_reports_no_cache(self):
        out = _outcome()
        out.usage = DecisionUsage(prompt_tokens=120, scoring_tokens=1, total_tokens=121, cached_tokens=None, backend_calls=1)
        result, _, crud, _, db, _ = await _call(_BODY, outcome=out)
        crud.update_quota_usage.assert_awaited_once_with(db, 1, 121)
        assert result["usage"] == {"input_tokens": 120, "output_tokens": 1}

    async def test_single_option_request_answers_without_a_backend_call(self):
        body = {"state": "s", "model": "jev-latest",
                "questions": {"only": {"type": "choice", "instructions": "?", "criteria": {"x": None}}}}
        result, backend, crud, _, db, _ = await _call(body)
        backend.decide.assert_not_awaited()
        assert result["answers"]["only"]["choice"] == "x"
        crud.update_quota_usage.assert_awaited_once_with(db, 1, 0)

    def test_both_paths_are_registered(self):
        from backend.app.api import api_router
        paths = {getattr(r, "path", "") for r in api_router.routes}
        assert {"/v1/systemone", "/v1/decisions"} <= paths



# --------------------------------------------------------------------------
# 7. choosing the underlying model: vLLM letter scoring vs an upstream
#    System One server (Laya)
# --------------------------------------------------------------------------

_LAYA = up.Upstream(name="laya", url="https://laya.example.edu:8010", api_key="laya-secret", model=None)
_Q3 = _SDK_BODY["questions"]

# What laya-serve returns for _SDK_BODY, including the fields it adds beyond
# Jev's (a root `routing`, per-answer `action`, `confidence` on a noul).
_LAYA_REPLY = {
    "model": "convaiinnovations/laya",
    "answers": {
        "department": {"type": "choice", "choice": "billing", "confidence": 0.9, "action": "accept",
                       "probabilities": {"billing": 0.93, "technical": 0.05, "sales": 0.02}},
        "urgency": {"type": "score", "score": 1.8, "confidence": 0.7,
                    "legend": {"0": "Can wait", "1": "Needs attention this week",
                               "2": {"level": "today", "note": "blocking"}},
                    "probabilities": {"0": 0.05, "1": 0.1, "2": 0.85}},
        "churn_risk": {"type": "noul", "noul": 0.97, "confidence": 0.94},
    },
    "usage": {"input_tokens": 412, "output_tokens": 0, "truncated": False},
    "routing": {"model": "english", "reason": "Latin script, English"},
}


class _UpClient:
    """Stands in for httpx.AsyncClient in the upstream module."""
    calls: list = []
    reply = None

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None, headers=None):
        _encode_like_httpx(json)
        _UpClient.calls.append((url, json, headers))
        if isinstance(_UpClient.reply, Exception):
            raise _UpClient.reply
        return _UpClient.reply


class _UpResponse:
    def __init__(self, payload, status_code=200, text=None):
        self._payload, self.status_code = payload, status_code
        self.text = text if text is not None else json.dumps(payload)

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


@pytest.fixture
def up_http(monkeypatch):
    _UpClient.calls, _UpClient.reply = [], _UpResponse(_LAYA_REPLY)
    monkeypatch.setattr(up.httpx, "AsyncClient", _UpClient)
    return _UpClient


class TestUpstreamConfig:
    def test_valid_entry(self):
        ups, problems = up.parse_upstreams(
            {"laya": {"url": "https://h:8010/", "api_key": "k", "model": "multilingual", "timeout": 20}})
        assert problems == []
        assert ups["laya"] == up.Upstream(name="laya", url="https://h:8010", api_key="k", model="multilingual", timeout=20.0)

    def test_empty_means_none(self):
        assert up.parse_upstreams(None) == ({}, []) and up.parse_upstreams({}) == ({}, [])

    @pytest.mark.parametrize("raw", [
        ["laya"],
        {"laya": "https://h"},
        {"laya": {"url": "ftp://h"}},
        {"laya": {}},
        {"bad name!": {"url": "https://h"}},
        {"laya": {"url": "https://h", "api_key": ""}},
        {"laya": {"url": "https://h", "timeout": 0}},
        {"laya": {"url": "https://h", "timeout": 999}},
        {"laya": {"url": "https://h", "surprise": 1}},
    ])
    def test_bad_entries_are_reported_and_left_out(self, raw):
        ups, problems = up.parse_upstreams(raw)
        assert ups == {} and problems


class TestUpstreamBackend:
    async def test_forwards_state_and_questions_with_the_upstream_key(self, up_http):
        out = await up.SystemOneUpstreamBackend().answer(_LAYA, _SDK_BODY["state"], _Q3)
        (url, body, headers), = up_http.calls
        assert url == "https://laya.example.edu:8010/v1/systemone"
        assert body == {"state": _SDK_BODY["state"], "questions": _Q3}      # model omitted: Laya routes by language
        assert headers["Authorization"] == "Bearer laya-secret"
        assert out.input_tokens == 412 and out.output_tokens == 0
        assert out.upstream_model == "convaiinnovations/laya" and out.extras["routing"]["model"] == "english"

    async def test_configured_upstream_model_is_sent(self, up_http):
        typed = up.Upstream(name="laya-typed", url="https://h", model="typed-decisions")
        await up.SystemOneUpstreamBackend().answer(typed, "s", _Q3)
        (_, body, headers), = up_http.calls
        assert body["model"] == "typed-decisions" and "Authorization" not in headers

    async def test_reply_is_reduced_to_typesafes_fields(self, up_http):
        out = await up.SystemOneUpstreamBackend().answer(_LAYA, "s", _Q3)
        assert set(out.answers["department"]) == {"type", "choice", "probabilities", "confidence"}  # no `action`
        assert out.answers["churn_risk"] == {"type": "noul", "noul": 0.97}                           # no `confidence`
        assert set(out.answers["urgency"]) == {"type", "score", "legend", "probabilities", "confidence"}

    @pytest.mark.parametrize("mutate", [
        lambda r: r.pop("answers"),
        lambda r: r["answers"].pop("urgency"),                                   # missing an answer
        lambda r: r["answers"].update(extra={"type": "noul", "noul": 0.5}),      # answer nobody asked for
        lambda r: r["answers"]["churn_risk"].update(type="choice"),              # wrong type for the question
        lambda r: r["answers"]["churn_risk"].update(noul=1.7),                   # not a probability
        lambda r: r["answers"]["churn_risk"].update(noul="high"),
        lambda r: r["answers"]["department"].update(choice="legal"),             # option that was not offered
        lambda r: r["answers"]["urgency"].update(score=float("nan")),
    ])
    async def test_invalid_replies_are_502_not_passed_on(self, up_http, mutate):
        reply = json.loads(json.dumps(_LAYA_REPLY))
        mutate(reply)
        up_http.reply = _UpResponse(reply)
        with pytest.raises(DecisionBackendError) as e:
            await up.SystemOneUpstreamBackend().answer(_LAYA, "s", _Q3)
        assert e.value.status_code == 502

    async def test_upstream_422_is_the_callers_422(self, up_http):
        up_http.reply = _UpResponse({"detail": "state exceeds 50000 characters"}, status_code=422)
        with pytest.raises(so.SystemOneValidationError) as e:
            await up.SystemOneUpstreamBackend().answer(_LAYA, "s", _Q3)
        assert "state exceeds 50000" in e.value.detail[0]["msg"] and e.value.detail[0]["loc"] == ["body"]

    @pytest.mark.parametrize("status,expected", [(503, 503), (429, 503), (401, 502), (500, 502)])
    async def test_upstream_failures_map_to_gateway_statuses(self, up_http, status, expected):
        # A 401 means MindRouter's configured key is wrong; the caller must not see 401.
        up_http.reply = _UpResponse({"detail": "x"}, status_code=status)
        with pytest.raises(DecisionBackendError) as e:
            await up.SystemOneUpstreamBackend().answer(_LAYA, "s", _Q3)
        assert e.value.status_code == expected

    async def test_unreachable_and_non_json_are_502(self, up_http):
        import httpx
        for reply in (httpx.ConnectError("refused"), _UpResponse(ValueError("not json"), text="<html>")):
            up_http.reply = reply
            with pytest.raises(DecisionBackendError) as e:
                await up.SystemOneUpstreamBackend().answer(_LAYA, "s", _Q3)
            assert e.value.status_code == 502


def _upstream_backend(answer=None, error=None):
    b = MagicMock()
    b.name = "systemone_upstream"
    b.answer = AsyncMock(side_effect=error) if error else AsyncMock(return_value=answer or up._checked("laya", json.loads(json.dumps(_LAYA_REPLY)), _Q3))
    return b


class TestModelSelection:
    _cfg = {"upstreams": {"laya": _LAYA}}

    async def test_model_laya_is_forwarded_not_scored_on_vllm(self):
        ub = _upstream_backend()
        result, vllm, crud, _, db, _ = await _call({**_SDK_BODY, "model": "laya"}, cfg=self._cfg, upstream_backend=ub)
        vllm.decide.assert_not_awaited()
        ub.answer.assert_awaited_once()
        assert ub.answer.call_args.args == (_LAYA, _SDK_BODY["state"], _SDK_BODY["questions"])
        assert result["model"] == "laya" and result["answers"]["department"]["choice"] == "billing"
        assert result["usage"] == {"input_tokens": 412, "output_tokens": 0}
        assert result["metadata"]["backend"] == "systemone_upstream"
        assert result["metadata"]["score_semantics"] == "upstream_model_probability"
        assert result["metadata"]["upstream_model"] == "convaiinnovations/laya"
        assert result["metadata"]["routing"]["model"] == "english"
        kw = crud.create_request.call_args.kwargs
        assert kw["model"] == "laya" and kw["parameters"]["backend"] == "systemone_upstream"
        crud.update_quota_usage.assert_awaited_once_with(db, 1, 412)
        assert crud.update_request_completed.call_args.kwargs["backend_id"] is None

    async def test_model_qwen_still_scores_on_vllm_when_laya_is_configured(self):
        ub = _upstream_backend()
        result, vllm, _, _, _, _ = await _call({**_BODY, "model": "qwen3.8-27b"}, cfg=self._cfg, upstream_backend=ub)
        vllm.decide.assert_awaited_once()
        ub.answer.assert_not_awaited()
        assert result["model"] == "qwen3.8-27b" and result["metadata"]["backend"] == "vllm_logprobs"

    async def test_jev_latest_follows_the_admin_default(self):
        ub = _upstream_backend(answer=up._checked("laya", {"answers": {"escalate": {"type": "noul", "noul": 0.6}}},
                                                  _BODY["questions"]))
        result, vllm, _, _, _, _ = await _call(_BODY, cfg={**self._cfg, "default_model": "laya"}, upstream_backend=ub)
        vllm.decide.assert_not_awaited()
        assert result["model"] == "laya" and result["answers"]["escalate"] == {"type": "noul", "noul": 0.6}

    async def test_upstream_takes_more_choice_options_than_letter_scoring(self):
        # The 20-option ceiling belongs to letter scoring, not to the wire format.
        criteria = {f"o{i}": None for i in range(40)}
        body = {"state": "s", "model": "laya", "questions": {"q": {"type": "choice", "criteria": criteria}}}
        reply = {"answers": {"q": {"type": "choice", "choice": "o3", "confidence": 0.5,
                                   "probabilities": {k: 1 / 40 for k in criteria}}}}
        ub = _upstream_backend(answer=up._checked("laya", reply, body["questions"]))
        result, _, _, _, _, _ = await _call(body, cfg=self._cfg, upstream_backend=ub)
        assert result["answers"]["q"]["choice"] == "o3"
        with pytest.raises(HTTPException) as e:
            await _call({**body, "model": "qwen3.8-27b"}, cfg=self._cfg, upstream_backend=ub)
        assert e.value.status_code == 422

    async def test_unknown_model_error_lists_every_choice(self):
        with pytest.raises(HTTPException) as e:
            await _call({**_BODY, "model": "nope"}, cfg=self._cfg, upstream_backend=_upstream_backend())
        msg = e.value.detail[0]["msg"]
        assert all(n in msg for n in ("jev-latest", "qwen3.8-27b", "laya"))

    async def test_upstream_failure_is_audited_and_not_charged(self):
        ub = _upstream_backend(error=DecisionBackendError("decision model 'laya' is unreachable", 502))
        with pytest.raises(HTTPException) as e:
            await _call({**_BODY, "model": "laya"}, cfg=self._cfg, upstream_backend=ub)
        assert e.value.status_code == 502
        _, crud, _, _ = e.value.mocks
        crud.update_request_failed.assert_awaited_once()
        crud.update_quota_usage.assert_not_awaited()

    async def test_upstream_rejection_is_422_and_audited(self):
        ub = _upstream_backend(error=so.SystemOneValidationError(
            [{"loc": ["body"], "msg": "rejected by decision model 'laya': too long", "type": "value_error"}]))
        with pytest.raises(HTTPException) as e:
            await _call({**_BODY, "model": "laya"}, cfg=self._cfg, upstream_backend=ub)
        assert e.value.status_code == 422
        e.value.mocks[1].update_request_failed.assert_awaited_once()

    def test_model_list_names_both_kinds(self):
        cfg = {"enabled": True, "default_model": "qwen3.8-27b", "allowed_models": ["qwen3.8-27b"],
               "upstreams": {"laya": _LAYA}}
        assert [m["name"] for m in so.typesafe_model_list(cfg)["models"]] == [
            "jev-latest", "jev-preview", "qwen3.8-27b", "laya"]


# --------------------------------------------------------------------------
# 8. settings load, and keeping request content out of the audit trail
# --------------------------------------------------------------------------

class _Rows:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class TestSettings:
    async def _cfg(self, rows):
        db = MagicMock()
        db.execute = AsyncMock(return_value=_Rows(rows))
        cfg = await pkg.get_decisions_config(db)
        return cfg, db

    async def test_defaults_when_nothing_is_configured(self):
        cfg, db = await self._cfg([])
        assert cfg == {"enabled": False, "default_model": "qwen/qwen3.8-27b", "allowed_models": ["qwen/qwen3.8-27b"],
                       "max_state_chars": 32_000, "fanout": 8, "backend_concurrency": 4, "upstreams": {},
                       "permutations": {"noul": 1, "choice": 2, "score": 1},
                       "temperature": {"noul": 1.35, "choice": 1.05, "score": 1.45}, "fallbacks": {}}
        db.execute.assert_awaited_once()   # one query for every decisions.* key, not one per key

    async def test_rows_are_json_decoded_and_upstreams_parsed(self):
        cfg, _ = await self._cfg([
            ("decisions.enabled", "true"), ("decisions.default_model", '"laya"'),
            ("decisions.allowed_models", '["qwen3.8-27b"]'), ("decisions.fanout", "3"),
            ("decisions.upstreams", '{"laya": {"url": "https://h:8010", "api_key": "k"}}'),
        ])
        assert cfg["enabled"] is True and cfg["default_model"] == "laya" and cfg["fanout"] == 3
        assert cfg["upstreams"]["laya"].url == "https://h:8010" and cfg["upstreams"]["laya"].api_key == "k"

    async def test_unreadable_or_invalid_rows_fall_back(self):
        cfg, _ = await self._cfg([
            ("decisions.fanout", "not json"), ("decisions.max_state_chars", "null"),
            ("decisions.upstreams", '{"laya": {"url": "ftp://nope"}}'),
        ])
        assert cfg["fanout"] == 8 and cfg["max_state_chars"] == 32_000 and cfg["upstreams"] == {}


class TestNothingAboutTheRequestIsStored:
    _cfg = {"upstreams": {"laya": _LAYA}}

    async def test_upstream_rejection_text_goes_to_the_caller_not_the_audit_row(self, up_http):
        # The upstream's 422 may quote the state; the caller sees it, the audit row does not.
        up_http.reply = _UpResponse({"detail": "bad value near 'secret ticket text'"}, status_code=422)
        with pytest.raises(HTTPException) as e:
            await _call({**_BODY, "model": "laya"}, cfg=self._cfg, upstream_backend=up.SystemOneUpstreamBackend())
        assert e.value.status_code == 422 and "secret ticket text" in e.value.detail[0]["msg"]
        stored = e.value.mocks[1].update_request_failed.call_args.args[2]
        assert stored == "rejected by decision model 'laya' (HTTP 422)" and "secret" not in stored

    async def test_upstream_error_body_is_not_logged(self, up_http):
        up_http.reply = _UpResponse({"detail": "echo: secret ticket text"}, status_code=500)
        with patch.object(up, "logger") as log:
            with pytest.raises(DecisionBackendError):
                await up.SystemOneUpstreamBackend().answer(_LAYA, "secret ticket text", _BODY["questions"])
        assert "secret" not in repr(log.mock_calls)

    async def test_forwarded_questions_are_size_capped(self):
        big = {"q": {"type": "choice", "instructions": "?", "criteria": {"a": "x" * 300_000, "b": None}}}
        ub = _upstream_backend()
        with pytest.raises(HTTPException) as e:
            await _call({"state": "s", "model": "laya", "questions": big}, cfg=self._cfg, upstream_backend=ub)
        assert e.value.status_code == 422 and e.value.detail[0]["loc"] == ["body", "questions"]
        ub.answer.assert_not_awaited()

    async def test_unknown_model_name_is_truncated_in_the_error(self):
        with pytest.raises(HTTPException) as e:
            await _call({**_BODY, "model": "m" * 5000})
        assert len(e.value.detail[0]["msg"]) < 400


# --------------------------------------------------------------------------
# 9. findings from the pre-merge review of PR #21
# --------------------------------------------------------------------------

_SURROGATE = "SECRET \ud800 text"   # a lone surrogate: valid to Python's JSON parser, not encodable as UTF-8


class TestUnsendableInputIsTheCallers422:
    @pytest.mark.parametrize("body", [
        {"state": _SURROGATE, "model": "jev-latest", "questions": {"q": {"type": "noul", "instructions": "?"}}},
        {"state": "s", "model": "jev-latest", "questions": {"q": {"type": "noul", "instructions": _SURROGATE}}},
        {"state": "s", "model": "jev-latest", "questions": {
            "q": {"type": "choice", "instructions": "?", "criteria": {"a": _SURROGATE, "b": None}}}},
        {"state": "s", "model": "jev-latest", "questions": {_SURROGATE: {"type": "noul", "instructions": "?"}}},
        {"state": "s", "model": "jev-latest", "questions": {"q": {"type": "score", "criteria": ["a", _SURROGATE]}}},
        {"state": {"x": float("nan")}, "model": "jev-latest", "questions": {"q": {"type": "noul", "instructions": "?"}}},
        {"state": [float("inf")], "model": "jev-latest", "questions": {"q": {"type": "noul", "instructions": "?"}}},
    ])
    def test_surrogates_and_non_finite_numbers_are_refused_up_front(self, body):
        with pytest.raises(so.SystemOneValidationError) as e:
            so.validate_wire(body)
        assert e.value.detail[0]["loc"] == ["body"]
        assert "SECRET" not in json.dumps(e.value.detail, ensure_ascii=True)

    async def test_route_answers_422_before_any_row_or_backend_call(self):
        body = {"state": _SURROGATE, "model": "jev-latest", "questions": {"q": {"type": "noul", "instructions": "?"}}}
        with pytest.raises(HTTPException) as e:
            await _call(body)
        assert e.value.status_code == 422
        e.value.mocks[0].decide.assert_not_awaited()
        e.value.mocks[1].create_request.assert_not_awaited()

    def test_internal_model_errors_become_422_without_the_input_value(self):
        # Reaching the internal models' own limits must not surface as an
        # unhandled pydantic error, whose text carries the offending input.
        wire = so.validate_wire({"state": "SECRET " + "x" * MAX_STATE_CHARS, "model": "jev-latest",
                                 "questions": {"q": {"type": "noul", "instructions": "?"}}})
        with pytest.raises(so.SystemOneValidationError) as e:
            so.compile_plan(wire)
        assert e.value.detail[0]["loc"][0] == "body" and "SECRET" not in json.dumps(e.value.detail)
        assert "input" not in e.value.detail[0]

    async def test_admin_ceiling_cannot_exceed_the_hard_cap(self):
        db = MagicMock()
        db.execute = AsyncMock(return_value=_Rows([("decisions.max_state_chars", "9000000")]))
        assert (await pkg.get_decisions_config(db))["max_state_chars"] == MAX_STATE_CHARS


class TestAuditRowNeverSticksInProcessing:
    async def test_cancellation_closes_the_row_and_propagates(self):
        import asyncio

        async def decide(*a, **k):
            raise asyncio.CancelledError()

        crud = _crud()
        with pytest.raises(asyncio.CancelledError):
            await _call(_BODY, crud=crud, decide=decide)
        crud.update_request_failed.assert_awaited_once()
        assert crud.update_request_failed.call_args.kwargs["error_code"] == "499"
        crud.update_quota_usage.assert_not_awaited()

    async def test_failure_while_recording_completion_marks_the_row_failed(self):
        crud = _crud()
        crud.update_request_completed = AsyncMock(side_effect=RuntimeError("Out of range value for column"))
        with pytest.raises(HTTPException) as e:
            await _call(_BODY, crud=crud)
        assert e.value.status_code == 500
        crud.update_request_failed.assert_awaited_once()
        crud.incr_quota_redis.assert_not_awaited()

    async def test_redis_counter_failure_does_not_fail_a_committed_request(self):
        crud = _crud()
        crud.incr_quota_redis = AsyncMock(side_effect=ConnectionError("redis down"))
        result, _, crud, _, _, _ = await _call(_BODY, crud=crud)
        assert result["answers"]["escalate"]["noul"] == 0.8
        crud.update_request_failed.assert_not_awaited()

    async def test_crash_log_carries_the_error_type_not_its_text(self):
        with patch.object(api, "logger") as log:
            with pytest.raises(HTTPException):
                await _call(_BODY, outcome=RuntimeError("input_value='secret ticket text'"))
        assert "secret" not in repr(log.mock_calls) and "RuntimeError" in repr(log.mock_calls)


class TestUpstreamIsNotTrusted:
    async def test_only_typesafes_question_fields_are_forwarded(self, up_http):
        qs = {"q": {"type": "noul", "instructions": "?", "max_len": 8192, "x-inject": {"a": 1}}}
        up_http.reply = _UpResponse({"answers": {"q": {"type": "noul", "noul": 0.5}}})
        await up.SystemOneUpstreamBackend().answer(_LAYA, "s", qs)
        (_, body, _), = up_http.calls
        assert body["questions"] == {"q": {"type": "noul", "instructions": "?"}}

    @pytest.mark.parametrize("mutate", [
        lambda r: r["answers"]["department"].update(
            choice="legal", probabilities={"legal": 0.9, "billing": 0.1}),     # options nobody asked about
        lambda r: r["answers"]["department"]["probabilities"].pop("sales"),    # an asked option missing
        lambda r: r["answers"]["urgency"].update(probabilities={"0": 0.5, "1": 0.5}),   # a level missing
        lambda r: r["usage"].update(input_tokens=3_000_000_000),               # would overflow INT / drain quota
        lambda r: r["usage"].update(input_tokens=-5),
        lambda r: r["usage"].update(output_tokens="many"),
    ])
    async def test_mismatched_options_and_absurd_usage_are_502(self, up_http, mutate):
        reply = json.loads(json.dumps(_LAYA_REPLY))
        mutate(reply)
        up_http.reply = _UpResponse(reply)
        with pytest.raises(DecisionBackendError) as e:
            await up.SystemOneUpstreamBackend().answer(_LAYA, "s", _Q3)
        assert e.value.status_code == 502

    async def test_vllm_error_body_is_not_logged(self, fake_http):
        good = _default_handler()

        def handler(url, body):
            if url.endswith("/tokenize"):
                return good(url, body)
            return _FakeResponse({"error": "prompt was: secret ticket text"}, status_code=400)
        fake_http.handler = handler
        with patch.object(vl, "logger") as log, patch.object(vl, "get_registry", return_value=_fake_registry([_vllm_backend()])):
            with pytest.raises(DecisionBackendError):
                await vl.VLLMLogprobsBackend().decide(_req(), "qwen3.8-27b", fanout=2)
        assert "secret" not in repr(log.mock_calls)


class TestUpstreamKeysAreNotRenderedBack:
    _stored = {"laya": {"url": "https://h:8010", "api_key": "real-secret", "model": None},
               "open": {"url": "https://o:8791"}}

    def test_display_masks_every_key(self):
        shown = up.mask_keys(self._stored)
        assert "real-secret" not in json.dumps(shown)
        assert shown["laya"]["api_key"] == up.KEY_PLACEHOLDER and shown["laya"]["url"] == "https://h:8010"
        assert shown["open"] == {"url": "https://o:8791"}          # nothing to mask
        assert self._stored["laya"]["api_key"] == "real-secret"      # not mutated

    def test_saving_the_placeholder_keeps_the_stored_key(self):
        submitted = {"laya": {"url": "https://h:9999", "api_key": up.KEY_PLACEHOLDER}}
        value, problems = up.restore_keys(submitted, self._stored)
        assert problems == [] and value == {"laya": {"url": "https://h:9999", "api_key": "real-secret"}}

    def test_a_new_key_replaces_and_a_placeholder_without_a_stored_key_is_refused(self):
        value, problems = up.restore_keys({"laya": {"url": "https://h", "api_key": "rotated"}}, self._stored)
        assert problems == [] and value["laya"]["api_key"] == "rotated"
        value, problems = up.restore_keys({"new": {"url": "https://n", "api_key": up.KEY_PLACEHOLDER}}, self._stored)
        assert value == {} and problems and "new" in problems[0]


# --------------------------------------------------------------------------
# 10. letter-scoring tuning (2.9.84): option-order averaging per question
#     type, and temperature
# --------------------------------------------------------------------------

from backend.app.services.decisions.scoring import apply_temperature  # noqa: E402

_THREE = {"state": "s", "model": "jev-latest", "questions": {
    "yn": {"type": "noul", "instructions": "?"},
    "ch": {"type": "choice", "instructions": "?", "criteria": {"a": None, "b": None, "c": None}},
    "sc": {"type": "score", "instructions": "?", "criteria": ["low", "mid", "high"]},
}}


class TestDefaultModelName:
    def test_default_is_the_catalog_name(self):
        # 2.9.83 shipped "qwen3.8-27b"; the catalog lists "qwen/qwen3.8-27b", so
        # jev-latest answered 404 "model does not exist" in production.
        assert pkg.DEFAULT_MODEL == "qwen/qwen3.8-27b"


class TestOptionOrderDefaults:
    def test_only_choice_is_averaged_over_two_orders_by_default(self):
        plan = so.parse_request(_THREE)
        assert {q.jev_id: q.permutations for q in plan.questions} == {"yn": 1, "ch": 2, "sc": 1}
        assert [q.permutations for q in plan.decision_request.questions] == [1, 2, 1]

    @pytest.mark.parametrize("n", [1, 2])
    def test_a_request_that_names_permutations_overrides_every_type(self, n):
        plan = so.parse_request({**_THREE, "permutations": n})
        assert {q.permutations for q in plan.questions} == {n}

    def test_admin_setting_replaces_the_defaults(self):
        plan = so.parse_request(_THREE, {"noul": 2, "choice": 1, "score": 2})
        assert {q.jev_id: q.permutations for q in plan.questions} == {"yn": 2, "ch": 1, "sc": 2}

    async def test_adapter_scores_each_question_with_its_own_number_of_orders(self, fake_http):
        plan = so.parse_request(_THREE)
        backend = vl.VLLMLogprobsBackend()
        with patch.object(vl, "get_registry", return_value=_fake_registry([_vllm_backend()])):
            out = await backend.decide(plan.decision_request, "qwen/qwen3.8-27b", fanout=4)
        chat = [b for u, b in fake_http.calls if u.endswith("/v1/chat/completions")]
        assert len(chat) == 4 and out.usage.backend_calls == 4          # 1 (noul) + 2 (choice) + 1 (score)
        texts = [b["messages"][0]["content"] for b in chat]
        assert sum(t.endswith("A. a\nB. b\nC. c") for t in texts) == 1
        assert sum(t.endswith("A. c\nB. b\nC. a") for t in texts) == 1   # the reversed view of the choice
        assert sum(t.endswith("A. low\nB. mid\nC. high") for t in texts) == 1

    async def test_route_scores_with_the_admins_per_type_setting(self):
        # What reaches the scoring backend carries the configured number of orders.
        _, backend, _, _, _, _ = await _call(_THREE, cfg={"permutations": {"noul": 2, "choice": 1, "score": 2}},
                                             outcome=DecisionOutcome(results=[
                                                 DecisionResult(id="q0", type="boolean", answer=True, likelihoods={"yes": 0.6, "no": 0.4}, logprobs={"yes": -0.5, "no": -0.9}, label_mass=1.0),
                                                 DecisionResult(id="q1", type="choice", answer="a", likelihoods={"a": 0.6, "b": 0.3, "c": 0.1}, logprobs={"a": -0.5, "b": -1.2, "c": -2.3}, label_mass=1.0),
                                                 DecisionResult(id="q2", type="choice", answer="low", likelihoods={"low": 0.6, "mid": 0.3, "high": 0.1}, logprobs={"low": -0.5, "mid": -1.2, "high": -2.3}, label_mass=1.0)],
                                                 usage=DecisionUsage(prompt_tokens=90, scoring_tokens=5, total_tokens=95, backend_calls=5), backend_id=7, backend_name="b7"))
        sent = backend.decide.call_args.args[0]
        assert [q.permutations for q in sent.questions] == [2, 1, 2]

    async def test_named_permutations_are_recorded_in_the_audit_shape(self):
        _, _, crud, _, _, _ = await _call({**_BODY, "permutations": 2})
        assert crud.create_request.call_args.kwargs["parameters"]["permutations"] == 2


class TestTemperature:
    def test_softens_without_changing_the_ranking(self):
        p = [0.90, 0.07, 0.03]
        q = apply_temperature(p, 1.45)
        assert sum(q) == pytest.approx(1.0) and q[0] < p[0] and q[0] > q[1] > q[2]
        assert apply_temperature(p, 1.0) == p                      # 1 is "off"
        sharp = apply_temperature(p, 0.5)
        assert sharp[0] > p[0]                                     # below 1 sharpens
        assert apply_temperature([1.0, 0.0], 1.35) == [1.0, 0.0]   # certainty stays certain
        assert apply_temperature([1.0], 2.0) == [1.0]

    def test_matches_scaling_the_logits(self):
        logits = [2.0, 0.5, -1.0]
        t = 1.35
        z = [math.exp(v / t) for v in logits]
        expect = [v / sum(z) for v in z]
        assert apply_temperature(softmax(logits), t) == pytest.approx(expect)

    def _answers(self, temperature):
        plan = so.parse_request(_THREE)
        results = _results_for(plan, {"q0": [0.9, 0.1], "q1": [0.8, 0.15, 0.05], "q2": [0.05, 0.15, 0.8]})
        return so.format_response(plan, results, None, model="m", request_id="x", backend_name="b",
                                  temperature=temperature)

    def test_response_reports_softened_probabilities_and_the_same_answers(self):
        raw = self._answers(None)
        cal = self._answers({"noul": 1.35, "choice": 1.05, "score": 1.45})
        assert raw["answers"]["yn"]["noul"] == pytest.approx(0.9)
        assert 0.5 < cal["answers"]["yn"]["noul"] < 0.9
        assert cal["answers"]["ch"]["choice"] == raw["answers"]["ch"]["choice"] == "a"
        assert cal["answers"]["ch"]["probabilities"]["a"] < 0.8
        assert sum(cal["answers"]["ch"]["probabilities"].values()) == pytest.approx(1.0)
        assert cal["answers"]["ch"]["confidence"] < raw["answers"]["ch"]["confidence"]
        # a softer distribution pulls the expected score toward the middle
        assert raw["answers"]["sc"]["score"] == pytest.approx(1.75) and cal["answers"]["sc"]["score"] < 1.75
        assert cal["metadata"]["temperature"] == {"noul": 1.35, "choice": 1.05, "score": 1.45}
        assert raw["metadata"]["temperature"] == {}

    def test_single_option_answers_stay_certain(self):
        plan = so.parse_request({"state": "s", "model": "jev-latest", "questions": {
            "only": {"type": "choice", "instructions": "?", "criteria": {"x": None}}}})
        r = so.format_response(plan, [], None, model="m", request_id="x", backend_name="b",
                               temperature={"noul": 3.0, "choice": 3.0, "score": 3.0})
        assert r["answers"]["only"]["probabilities"] == {"x": 1.0} and r["metadata"]["temperature"] == {}

    async def test_route_applies_the_configured_temperature(self):
        result, _, _, _, _, _ = await _call(_BODY, cfg={"temperature": {"noul": 1.35, "choice": 1.05, "score": 1.45}})
        assert 0.5 < result["answers"]["escalate"]["noul"] < 0.8          # scored 0.8, softened
        assert result["metadata"]["temperature"] == {"noul": 1.35}

    async def test_temperature_is_never_applied_to_an_upstream_models_answers(self):
        ub = _upstream_backend()
        result, _, _, _, _, _ = await _call({**_SDK_BODY, "model": "laya"}, upstream_backend=ub, cfg={
            "upstreams": {"laya": _LAYA}, "temperature": {"noul": 3.0, "choice": 3.0, "score": 3.0}})
        assert result["answers"]["churn_risk"]["noul"] == 0.97
        assert "temperature" not in result["metadata"]


class TestTuningSettings:
    def test_empty_means_the_fitted_defaults(self):
        assert so.parse_permutations(None) == ({"noul": 1, "choice": 2, "score": 1}, [])
        assert so.parse_temperature({}) == ({"noul": 1.35, "choice": 1.05, "score": 1.45}, [])

    def test_partial_settings_keep_the_other_defaults(self):
        assert so.parse_permutations({"choice": 1}) == ({"noul": 1, "choice": 1, "score": 1}, [])
        value, problems = so.parse_temperature({"noul": 1})
        assert problems == [] and value == {"noul": 1.0, "choice": 1.05, "score": 1.45}

    @pytest.mark.parametrize("raw", [{"choice": 3}, {"choice": 0}, {"choice": 1.5}, {"choice": True},
                                     {"essay": 2}, [2], "2"])
    def test_bad_permutations_are_reported_and_ignored(self, raw):
        value, problems = so.parse_permutations(raw)
        assert problems and value == so.DEFAULT_PERMUTATIONS

    @pytest.mark.parametrize("raw", [{"noul": 0}, {"noul": 9}, {"noul": "hot"}, {"noul": True}, {"essay": 1.2}, 1.3])
    def test_bad_temperatures_are_reported_and_ignored(self, raw):
        value, problems = so.parse_temperature(raw)
        assert problems and value == so.DEFAULT_TEMPERATURE

    async def test_settings_are_loaded_with_the_rest(self):
        db = MagicMock()
        db.execute = AsyncMock(return_value=_Rows([
            ("decisions.permutations", '{"choice": 1}'), ("decisions.temperature", '{"score": 2}')]))
        cfg = await pkg.get_decisions_config(db)
        assert cfg["permutations"] == {"noul": 1, "choice": 1, "score": 1}
        assert cfg["temperature"] == {"noul": 1.35, "choice": 1.05, "score": 2.0}


class TestUpstreamAnswersMeanTheSameThing:
    """An upstream's reply is normalized so fields mean one thing for every model."""

    async def test_confidence_is_recomputed_with_typesafes_formula(self, up_http):
        # Clef reports its top probability as confidence (0.93 here); TypeSafe's
        # formula for three options at 0.93 is (3 * 0.93 - 1) / 2 = 0.895.
        out = await up.SystemOneUpstreamBackend().answer(_LAYA, "s", _Q3)
        assert out.answers["department"]["confidence"] == pytest.approx((3 * 0.93 - 1) / 2)
        assert out.answers["urgency"]["confidence"] == pytest.approx(so.score_confidence([0.05, 0.1, 0.85]))

    async def test_a_reply_without_confidence_is_still_usable(self, up_http):
        reply = json.loads(json.dumps(_LAYA_REPLY))
        reply["answers"]["department"].pop("confidence")
        reply["answers"]["urgency"].pop("confidence")
        up_http.reply = _UpResponse(reply)
        out = await up.SystemOneUpstreamBackend().answer(_LAYA, "s", _Q3)
        assert 0 <= out.answers["department"]["confidence"] <= 1

    async def test_score_confidence_uses_level_order_not_reply_order(self, up_http):
        reply = json.loads(json.dumps(_LAYA_REPLY))
        reply["answers"]["urgency"]["probabilities"] = {"2": 0.85, "0": 0.05, "1": 0.1}   # shuffled keys
        up_http.reply = _UpResponse(reply)
        out = await up.SystemOneUpstreamBackend().answer(_LAYA, "s", _Q3)
        assert out.answers["urgency"]["confidence"] == pytest.approx(so.score_confidence([0.05, 0.1, 0.85]))

    async def test_a_cut_state_is_passed_on(self, up_http):
        reply = json.loads(json.dumps(_LAYA_REPLY))
        reply["usage"].update(truncated=True, state_tokens_dropped=1488)
        up_http.reply = _UpResponse(reply)
        out = await up.SystemOneUpstreamBackend().answer(_LAYA, "s", _Q3)
        assert out.extras["truncated"] is True and out.extras["state_tokens_dropped"] == 1488
        ub = MagicMock()
        ub.name = "systemone_upstream"
        ub.answer = AsyncMock(return_value=out)
        result, _, _, _, _, _ = await _call({**_SDK_BODY, "model": "laya"}, cfg={"upstreams": {"laya": _LAYA}},
                                            upstream_backend=ub)
        assert result["metadata"]["truncated"] is True and result["metadata"]["state_tokens_dropped"] == 1488

    async def test_no_truncation_fields_when_the_upstream_reports_none(self, up_http):
        reply = json.loads(json.dumps(_LAYA_REPLY))
        reply["usage"] = {"input_tokens": 10, "output_tokens": 0}
        up_http.reply = _UpResponse(reply)
        out = await up.SystemOneUpstreamBackend().answer(_LAYA, "s", _Q3)
        assert "truncated" not in out.extras and "state_tokens_dropped" not in out.extras


# --------------------------------------------------------------------------
# 11. images (Cloudflare Clef's extension to the System One request)
# --------------------------------------------------------------------------

import base64  # noqa: E402
import io  # noqa: E402

from backend.app.services.decisions import images as img  # noqa: E402


def _picture(fmt="PNG", size=(8, 8), color=(200, 30, 30)):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format=fmt)
    return buf.getvalue()


def _data_url(fmt="PNG", **kw):
    kind = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}[fmt]
    return f"data:{kind};base64,{base64.b64encode(_picture(fmt, **kw)).decode()}"


_WITH_IMAGE = {"state": "Review the attached receipt.", "model": "jev-latest", "images": [_data_url("PNG")],
               "questions": {"legible": {"type": "noul", "instructions": "Is the total legible?"}}}


class TestImageValidation:
    @pytest.mark.parametrize("fmt", ["PNG", "JPEG", "WEBP"])
    def test_data_urls_and_objects_are_both_accepted(self, fmt):
        url = _data_url(fmt)
        kind, b64 = url[len("data:"):].split(";base64,")
        assert img.normalize_images([url]) == [url]
        assert img.normalize_images([{"content_type": kind, "base64": b64}]) == [url]   # same canonical form

    def test_no_images_is_fine(self):
        assert img.normalize_images(None) == [] and img.normalize_images([]) == []

    def test_whitespace_in_base64_and_jpg_alias_are_tolerated(self):
        url = _data_url("JPEG")
        b64 = url.split(",", 1)[1]
        wrapped = "\n".join(b64[i:i + 40] for i in range(0, len(b64), 40))
        assert img.normalize_images([{"content_type": "image/jpg", "base64": wrapped}]) == [url]

    @pytest.mark.parametrize("bad,index", [
        ("https://example.com/cat.png", 0),                                      # remote URLs are never fetched
        ("data:image/gif;base64,R0lGODlhAQABAAAAACw=", 0),                        # not an accepted type
        ("data:image/png;base64,!!!not-base64!!!", 0),
        ("data:image/png;base64," + base64.b64encode(b"not an image at all").decode(), 0),
        ({"content_type": "image/png"}, 0),                                      # no data
        (42, 0),
    ])
    def test_bad_items_name_their_position(self, bad, index):
        with pytest.raises(img.ImageError) as e:
            img.normalize_images([bad])
        assert e.value.index == index

    def test_declared_type_must_match_the_bytes(self):
        jpeg = base64.b64encode(_picture("JPEG")).decode()
        with pytest.raises(img.ImageError) as e:
            img.normalize_images([f"data:image/png;base64,{jpeg}"])
        assert "image/jpeg" in str(e.value)

    def test_count_size_and_pixel_limits(self, monkeypatch):
        with pytest.raises(img.ImageError) as e:
            img.normalize_images([_data_url()] * (img.MAX_IMAGES + 1))
        assert e.value.index is None
        monkeypatch.setattr(img, "MAX_IMAGE_BYTES", 50)
        with pytest.raises(img.ImageError):
            img.normalize_images([_data_url()])
        monkeypatch.undo()
        monkeypatch.setattr(img, "MAX_TOTAL_IMAGE_BYTES", len(_picture()) + 10)
        with pytest.raises(img.ImageError) as e:
            img.normalize_images([_data_url(), _data_url()])
        assert "in total" in str(e.value)
        monkeypatch.undo()
        monkeypatch.setattr(img, "MAX_IMAGE_PIXELS", 50)
        with pytest.raises(img.ImageError) as e:
            img.normalize_images([_data_url(size=(8, 8))])       # 64 pixels
        assert "megapixels" in str(e.value)

    def test_oversized_base64_is_refused_before_it_is_stripped_or_decoded(self, monkeypatch):
        # 50 MB of "AA AA ..." once cost 1 s and 1 GB here: the length check comes first.
        calls = []
        monkeypatch.setattr(img.base64, "b64decode", lambda *a, **k: calls.append(1) or b"")
        padded = "AA " * (img.MAX_IMAGE_CHARS // 3 + 10)
        for item in (f"data:image/png;base64,{padded}", {"content_type": "image/png", "base64": padded}):
            with pytest.raises(img.ImageError) as e:
                img.normalize_images([item])
            assert "MiB" in str(e.value) and e.value.index == 0
        assert calls == []

    def test_a_full_size_image_with_line_breaks_fits_the_length_limit(self):
        encoded = (img.MAX_IMAGE_BYTES + 2) // 3 * 4
        assert encoded + encoded // 60 * 2 <= img.MAX_IMAGE_CHARS      # wrapped at 60 columns with CRLF

    def test_a_phone_cameras_multi_frame_jpeg_is_a_jpeg(self):
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (8, 8), "red").save(buf, format="MPO", save_all=True,
                                              append_images=[Image.new("RGB", (8, 8), "blue")])
        (url,) = img.normalize_images([{"content_type": "image/jpeg", "base64": base64.b64encode(buf.getvalue()).decode()}])
        assert url.startswith("data:image/jpeg;base64,")

    def test_a_damaged_png_is_refused(self):
        whole = _picture("PNG", size=(64, 64))
        cut = base64.b64encode(whole[: len(whole) // 2]).decode()
        with pytest.raises(img.ImageError) as e:
            img.normalize_images([f"data:image/png;base64,{cut}"])
        assert "could not be read" in str(e.value)

    def test_only_the_three_decoders_ever_open_caller_bytes(self, monkeypatch):
        from PIL import Image
        seen = []
        real = Image.open
        monkeypatch.setattr(Image, "open", lambda fp, **kw: seen.append(kw.get("formats")) or real(fp, **kw))
        gif = io.BytesIO(); Image.new("RGB", (4, 4)).save(gif, format="GIF")
        with pytest.raises(img.ImageError):
            img.normalize_images([f"data:image/png;base64,{base64.b64encode(gif.getvalue()).decode()}"])
        assert seen == [["PNG", "JPEG", "WEBP"]]

    def test_wire_validation_reports_images_in_fastapi_shape(self):
        with pytest.raises(so.SystemOneValidationError) as e:
            so.validate_wire({**_WITH_IMAGE, "images": [_data_url(), "https://example.com/x.png"]})
        assert e.value.detail[0]["loc"] == ["body", "images", 1]
        with pytest.raises(so.SystemOneValidationError) as e:
            so.validate_wire({**_WITH_IMAGE, "images": "one.png"})
        assert e.value.detail[0]["loc"][:2] == ["body", "images"]

    def test_plan_carries_the_normalized_images(self):
        plan = so.parse_request(_WITH_IMAGE)
        assert plan.images == _WITH_IMAGE["images"] == plan.decision_request.images
        assert so.parse_request(_SDK_BODY).images == []


class TestImagesOnVLLMModels:
    async def test_images_precede_the_text_in_every_scoring_call(self, fake_http):
        body = {**_WITH_IMAGE, "questions": {
            "legible": {"type": "noul", "instructions": "Is the total legible?"},
            "kind": {"type": "choice", "instructions": "What is it?", "criteria": {"receipt": None, "invoice": None}}}}
        plan = so.parse_request(body)
        backend = vl.VLLMLogprobsBackend()
        with patch.object(vl, "get_registry", return_value=_fake_registry([_vllm_backend(sees_images=True)])):
            await backend.decide(plan.decision_request, "qwen/qwen3.8-27b", fanout=4)
        chat = [b for u, b in fake_http.calls if u.endswith("/v1/chat/completions")]
        assert len(chat) == 3                                           # noul 1 + choice 2 orders
        for call in chat:
            content = call["messages"][0]["content"]
            assert [part["type"] for part in content] == ["image_url", "text"]
            assert content[0]["image_url"] == {"url": _WITH_IMAGE["images"][0]}
            assert content[1]["text"].startswith(INSTRUCTION)

    async def test_text_only_requests_keep_a_plain_string_turn(self, fake_http):
        backend = vl.VLLMLogprobsBackend()
        with patch.object(vl, "get_registry", return_value=_fake_registry([_vllm_backend()])):
            await backend.decide(_req(), "qwen/qwen3.8-27b", fanout=4)
        chat = [b for u, b in fake_http.calls if u.endswith("/v1/chat/completions")]
        assert all(isinstance(b["messages"][0]["content"], str) for b in chat)

    async def test_only_a_copy_of_the_model_that_sees_is_picked(self, fake_http):
        blind, sighted = _vllm_backend(id=7, sees_images=False), _vllm_backend(id=8, sees_images=True)
        plan = so.parse_request(_WITH_IMAGE)
        backend = vl.VLLMLogprobsBackend()
        with patch.object(vl, "get_registry", return_value=_fake_registry([blind, sighted])):
            for _ in range(8):
                out = await backend.decide(plan.decision_request, "qwen/qwen3.8-27b", fanout=4)
                assert out.backend_id == 8

    async def test_a_model_that_cannot_see_is_the_callers_422(self, fake_http):
        plan = so.parse_request(_WITH_IMAGE)
        backend = vl.VLLMLogprobsBackend()
        with patch.object(vl, "get_registry", return_value=_fake_registry([_vllm_backend(sees_images=False)])):
            with pytest.raises(DecisionBackendError) as e:
                await backend.decide(plan.decision_request, "qwen/qwen3.8-27b", fanout=4)
        assert e.value.status_code == 422 and "does not accept images" in str(e.value)
        assert not [u for u, _ in fake_http.calls if u.endswith("/v1/chat/completions")]

    async def test_first_view_still_warms_the_cache_when_only_images_are_shared(self, monkeypatch, fake_http):
        import asyncio
        events, base = [], _default_handler()

        class _Client(_FakeClient):
            async def post(self, url, json=None):
                if url.endswith("/tokenize"):
                    return base(url, json)
                events.append("start")
                await asyncio.sleep(0.002)
                events.append("end")
                return base(url, json)
        monkeypatch.setattr(vl.httpx, "AsyncClient", _Client)
        body = {"state": "", "model": "jev-latest", "images": [_data_url()], "questions": {
            f"q{i}": {"type": "noul", "instructions": f"Q{i}?"} for i in range(3)}}
        plan = so.parse_request(body)
        backend = vl.VLLMLogprobsBackend()
        with patch.object(vl, "get_registry", return_value=_fake_registry([_vllm_backend(sees_images=True)])):
            await backend.decide(plan.decision_request, "qwen/qwen3.8-27b", fanout=8)
        assert events[:2] == ["start", "end"]

    async def test_seeing_is_the_requested_models_own_flag(self, fake_http):
        # Another vision model on the same backend does not make this one see.
        backend_ = _vllm_backend(sees_images=False, other_models=[("google/gemma-4-31b", True)])
        plan = so.parse_request(_WITH_IMAGE)
        with patch.object(vl, "get_registry", return_value=_fake_registry([backend_])):
            with pytest.raises(DecisionBackendError) as e:
                await vl.VLLMLogprobsBackend().decide(plan.decision_request, "qwen/qwen3.8-27b", fanout=4)
        assert e.value.status_code == 422

    async def test_text_requests_do_not_need_a_model_that_sees(self, fake_http):
        with patch.object(vl, "get_registry", return_value=_fake_registry([_vllm_backend(sees_images=False)])):
            out = await vl.VLLMLogprobsBackend().decide(_req(), "qwen/qwen3.8-27b", fanout=4)
        assert out.backend_id == 7

    async def test_no_healthy_replica_is_503_even_with_images(self, fake_http):
        plan = so.parse_request(_WITH_IMAGE)
        with patch.object(vl, "get_registry", return_value=_fake_registry([_vllm_backend(healthy=False, sees_images=True)])):
            with pytest.raises(DecisionBackendError) as e:
                await vl.VLLMLogprobsBackend().decide(plan.decision_request, "qwen/qwen3.8-27b", fanout=4)
        assert e.value.status_code == 503

    @pytest.mark.parametrize("with_image,status,expected", [(True, 400, 422), (True, 500, 502), (False, 400, 502)])
    async def test_an_image_the_model_rejects_is_the_callers_422(self, fake_http, with_image, status, expected):
        good = _default_handler()
        fake_http.handler = lambda url, body: (
            good(url, body) if url.endswith("/tokenize") else _FakeResponse({"error": "x"}, status_code=status))
        request = so.parse_request(_WITH_IMAGE).decision_request if with_image else _req()
        with patch.object(vl, "get_registry", return_value=_fake_registry([_vllm_backend(sees_images=True)])):
            with pytest.raises(DecisionBackendError) as e:
                await vl.VLLMLogprobsBackend().decide(request, "qwen/qwen3.8-27b", fanout=4)
        assert e.value.status_code == expected


class TestImagesOnUpstreams:
    _clef = up.Upstream(name="clef", url="https://clef.example.edu:8004", api_key="k", model="clef", images=True)

    def test_images_capability_is_a_setting(self):
        ups, problems = up.parse_upstreams({"clef": {"url": "https://h", "images": True}, "laya": {"url": "https://l"}})
        assert problems == [] and ups["clef"].images is True and ups["laya"].images is False
        assert up.parse_upstreams({"clef": {"url": "https://h", "images": "yes"}})[1]

    async def test_images_are_forwarded_to_an_upstream_that_sees(self, up_http):
        up_http.reply = _UpResponse({"answers": {"legible": {"type": "noul", "noul": 0.9}}})
        await up.SystemOneUpstreamBackend().answer(
            self._clef, _WITH_IMAGE["state"], _WITH_IMAGE["questions"], images=_WITH_IMAGE["images"])
        (_, body, _), = up_http.calls
        assert body["images"] == _WITH_IMAGE["images"] and body["model"] == "clef"

    async def test_no_images_key_when_there_are_none(self, up_http):
        up_http.reply = _UpResponse({"answers": {"legible": {"type": "noul", "noul": 0.9}}})
        await up.SystemOneUpstreamBackend().answer(self._clef, "s", _WITH_IMAGE["questions"])
        assert "images" not in up_http.calls[0][1]

    async def test_route_forwards_images_to_clef_and_records_only_their_count(self):
        ub = _upstream_backend(answer=up._checked("clef", {"answers": {"legible": {"type": "noul", "noul": 0.9}}},
                                                  _WITH_IMAGE["questions"]))
        result, _, crud, _, _, _ = await _call({**_WITH_IMAGE, "model": "clef"},
                                               cfg={"upstreams": {"clef": self._clef}}, upstream_backend=ub)
        assert ub.answer.call_args.kwargs["images"] == _WITH_IMAGE["images"]
        params = crud.create_request.call_args.kwargs["parameters"]
        assert params["images"] == 1 and "base64" not in json.dumps(params)
        assert result["answers"]["legible"]["noul"] == 0.9

    async def test_an_upstream_that_cannot_see_refuses_images(self):
        # Laya ignores unknown fields: forwarding would return an answer about the text alone.
        ub = _upstream_backend()
        with pytest.raises(HTTPException) as e:
            await _call({**_WITH_IMAGE, "model": "laya"}, cfg={"upstreams": {"laya": _LAYA}}, upstream_backend=ub)
        assert e.value.status_code == 422 and e.value.detail[0]["loc"] == ["body", "images"]
        ub.answer.assert_not_awaited()
        e.value.mocks[1].create_request.assert_not_awaited()

    async def test_bad_image_is_422_before_any_row_or_backend_call(self):
        with pytest.raises(HTTPException) as e:
            await _call({**_WITH_IMAGE, "images": ["https://example.com/x.png"]})
        assert e.value.status_code == 422 and e.value.detail[0]["loc"] == ["body", "images", 0]
        e.value.mocks[1].create_request.assert_not_awaited()


# --------------------------------------------------------------------------
# 12. fallback to a configured alternative, and monitored decision servers
# --------------------------------------------------------------------------

from backend.app.services.decisions import parse_fallbacks  # noqa: E402

_CLEF = up.Upstream(name="clef", url="https://aspen4.example.edu:8001/", api_key="k", model="clef", images=True)
_CLEF_ANSWER = {"answers": {"escalate": {"type": "noul", "noul": 0.9}}, "usage": {"input_tokens": 40, "output_tokens": 0}}
_FB_CFG = {"default_model": "clef", "upstreams": {"clef": _CLEF}, "fallbacks": {"clef": "qwen3.8-27b"}}


def _clef_backend(error=None):
    ub = _upstream_backend(answer=up._checked("clef", _CLEF_ANSWER, _BODY["questions"]))
    if error is not None:
        ub.answer = AsyncMock(side_effect=error)
    return ub


def _monitored(state):
    reg = _fake_registry([])
    reg.decision_server_state = AsyncMock(return_value=state)
    return reg


class TestFallbackSetting:
    def test_valid_entries_and_each_kind_of_bad_one(self):
        known = ["clef", "qwen/qwen3.8-27b"]
        assert parse_fallbacks({"clef": "qwen/qwen3.8-27b"}, known) == ({"clef": "qwen/qwen3.8-27b"}, [])
        assert parse_fallbacks({}, known) == ({}, []) and parse_fallbacks(None, known) == ({}, [])
        for bad in ({"laya": "clef"}, {"clef": "gpt-4"}, {"clef": "clef"}, {"clef": 3}, ["clef"]):
            good, problems = parse_fallbacks(bad, known)
            assert good == {} and problems, bad

    def test_a_bad_entry_does_not_drop_the_good_ones(self):
        good, problems = parse_fallbacks({"clef": "q", "q": "nope"}, ["clef", "q"])
        assert good == {"clef": "q"} and len(problems) == 1


class TestFallback:
    async def test_working_model_answers_and_nothing_mentions_a_fallback(self):
        ub = _clef_backend()
        result, backend, crud, *_ = await _call(_BODY, cfg=_FB_CFG, upstream_backend=ub)
        assert result["model"] == "clef" and "fallback" not in result["metadata"]
        backend.decide.assert_not_awaited()
        assert crud.create_request.await_count == 1

    @pytest.mark.parametrize("status_code", [502, 503, 504])
    async def test_a_failing_model_is_answered_by_its_alternative(self, status_code):
        ub = _clef_backend(DecisionBackendError("decision upstream unreachable", status_code))
        result, backend, crud, quota, *_ = await _call(_BODY, cfg=_FB_CFG, upstream_backend=ub)
        assert result["model"] == "qwen3.8-27b"                       # the model that actually answered
        assert result["metadata"]["fallback"] == {"requested": "clef", "reason": "decision upstream unreachable"}
        assert result["answers"] == {"escalate": {"type": "noul", "noul": 0.8}}
        ub.answer.assert_awaited_once(); backend.decide.assert_awaited_once()
        # Two audit rows: the failed try on clef, then the answer, each under its own model.
        models = [c.kwargs["model"] for c in crud.create_request.call_args_list]
        assert models == ["clef", "qwen3.8-27b"]
        assert crud.update_request_failed.await_count == 1 and crud.update_request_completed.await_count == 1
        second = crud.create_request.call_args_list[1].kwargs["parameters"]
        assert second["fallback_from"] == "clef" and "fallback_from" not in crud.create_request.call_args_list[0].kwargs["parameters"]
        quota.assert_awaited_once()                                    # one request, one quota check

    async def test_the_callers_own_mistake_is_not_retried_elsewhere(self):
        refused = so.SystemOneValidationError([{"loc": ["body", "questions"], "msg": "no", "type": "value_error"}], "refused")
        ub = _clef_backend(refused)
        with pytest.raises(HTTPException) as e:
            await _call(_BODY, cfg=_FB_CFG, upstream_backend=ub)
        assert e.value.status_code == 422
        e.value.mocks[0].decide.assert_not_awaited()

    async def test_a_crash_in_our_own_code_is_not_hidden_by_a_fallback(self):
        ub = _clef_backend(RuntimeError("bug"))
        with pytest.raises(HTTPException) as e:
            await _call(_BODY, cfg=_FB_CFG, upstream_backend=ub)
        assert e.value.status_code == 500
        e.value.mocks[0].decide.assert_not_awaited()

    async def test_without_a_configured_alternative_the_error_stands(self):
        ub = _clef_backend(DecisionBackendError("decision upstream unreachable", 502))
        with pytest.raises(HTTPException) as e:
            await _call(_BODY, cfg={**_FB_CFG, "fallbacks": {}}, upstream_backend=ub)
        assert e.value.status_code == 502

    async def test_when_the_alternative_fails_too_its_error_is_returned(self):
        ub = _clef_backend(DecisionBackendError("decision upstream unreachable", 502))
        with pytest.raises(HTTPException) as e:
            await _call(_BODY, cfg=_FB_CFG, upstream_backend=ub, outcome=DecisionBackendError("no healthy vLLM backend", 503))
        assert e.value.status_code == 503
        crud = e.value.mocks[1]
        assert crud.create_request.await_count == 2 and crud.update_request_failed.await_count == 2   # no third try

    async def test_an_alternative_that_cannot_take_the_request_is_not_used(self):
        # 21 options: fine for Clef, more than letter scoring allows. The fallback cannot answer this.
        body = {**_BODY, "questions": {"pick": {"type": "choice", "instructions": "Which?",
                                                 "criteria": {f"o{i}": None for i in range(21)}}}}
        ub = _clef_backend(DecisionBackendError("decision upstream unreachable", 502))
        with pytest.raises(HTTPException) as e:
            await _call(body, cfg=_FB_CFG, upstream_backend=ub)
        assert e.value.status_code == 502                               # Clef's error, not a 422 about options
        e.value.mocks[0].decide.assert_not_awaited()

    async def test_whatever_goes_wrong_preparing_the_alternative_the_original_error_stands(self):
        # Primary is the vLLM model and fails; preparing the alternative (clef) blows up unexpectedly.
        cfg = {**_FB_CFG, "default_model": "qwen3.8-27b", "fallbacks": {"qwen3.8-27b": "clef"}}
        with patch.object(api, "_invalid", side_effect=RuntimeError("unexpected")):
            body = {**_BODY, "images": [_data_url()]}
            blind_clef = up.Upstream(name="clef", url="https://h:8001", api_key="k", model="clef", images=False)
            with pytest.raises(HTTPException) as e:
                await _call(body, cfg={**cfg, "upstreams": {"clef": blind_clef}}, upstream_backend=_clef_backend(),
                            registry=_fake_registry([_vllm_backend(model="qwen3.8-27b", sees_images=True)]),
                            outcome=DecisionBackendError("backend 502", 502))
        assert e.value.status_code == 502                                # the vLLM model's error, not a raw RuntimeError

    async def test_images_are_never_sent_to_an_alternative_that_cannot_see(self):
        # Clef is down and the request has an image; the alternative is a blind chat model.
        # The caller must get Clef's error, not a 422 about a model they never named.
        body = {**_BODY, "images": [_data_url()]}
        ub = _clef_backend(DecisionBackendError("decision upstream unreachable", 502))
        with pytest.raises(HTTPException) as e:
            await _call(body, cfg=_FB_CFG, upstream_backend=ub,
                        registry=_fake_registry([_vllm_backend(model="qwen3.8-27b", sees_images=False)]))
        assert e.value.status_code == 502
        e.value.mocks[0].decide.assert_not_awaited()

    async def test_images_go_to_an_alternative_that_can_see(self):
        body = {**_BODY, "images": [_data_url()]}
        ub = _clef_backend(DecisionBackendError("decision upstream unreachable", 502))
        result, backend, *_ = await _call(body, cfg=_FB_CFG, upstream_backend=ub,
                                          registry=_fake_registry([_vllm_backend(model="qwen3.8-27b", sees_images=True)]))
        assert result["model"] == "qwen3.8-27b" and backend.decide.call_args.args[0].images == body["images"]

    async def test_a_blind_model_asked_directly_is_422_before_any_row(self):
        body = {**_BODY, "images": [_data_url()]}
        with pytest.raises(HTTPException) as e:
            await _call(body, registry=_fake_registry([_vllm_backend(model="qwen3.8-27b", sees_images=False)]))
        assert e.value.status_code == 422 and e.value.detail[0]["loc"] == ["body", "images"]
        e.value.mocks[1].create_request.assert_not_awaited()

    async def test_the_setting_may_name_a_model_an_alias_points_at(self):
        cfg = {**_FB_CFG, "default_model": "decider", "fallbacks": {"qwen3.8-27b": "clef"}}
        reg = _fake_registry([], aliases={"decider": "qwen3.8-27b"})
        result, *_ = await _call(_BODY, cfg=cfg, upstream_backend=_clef_backend(), registry=reg,
                                 outcome=DecisionBackendError("backend 502", 502))
        assert result["model"] == "clef" and result["metadata"]["fallback"]["requested"] == "decider"

    async def test_fallback_works_in_the_other_direction_too(self):
        cfg = {**_FB_CFG, "default_model": "qwen3.8-27b", "fallbacks": {"qwen3.8-27b": "clef"}}
        ub = _clef_backend()
        result, backend, *_ = await _call(_BODY, cfg=cfg, upstream_backend=ub, outcome=DecisionBackendError("backend 502", 502))
        assert result["model"] == "clef" and result["metadata"]["fallback"]["requested"] == "qwen3.8-27b"

    async def test_a_model_with_no_healthy_replica_falls_back_before_any_work(self):
        cfg = {**_FB_CFG, "default_model": "qwen3.8-27b", "fallbacks": {"qwen3.8-27b": "clef"}}
        ub = _clef_backend()
        result, backend, crud, *_ = await _call(_BODY, cfg=cfg, upstream_backend=ub, availability="unavailable")
        assert result["model"] == "clef" and result["metadata"]["fallback"]["reason"] == "no healthy replica"
        backend.decide.assert_not_awaited()
        assert crud.create_request.await_count == 1                     # nothing was tried on the dead model


class TestMonitoredDecisionServer:
    async def test_a_server_known_to_be_down_is_skipped_without_dialing(self):
        ub = _clef_backend()
        result, backend, crud, *_ = await _call(_BODY, cfg=_FB_CFG, upstream_backend=ub,
                                                registry=_monitored((42, "unhealthy")))
        ub.answer.assert_not_awaited()                                  # no timeout spent on a dead server
        assert result["model"] == "qwen3.8-27b"
        assert result["metadata"]["fallback"] == {"requested": "clef", "reason": "unhealthy"}
        assert crud.create_request.await_count == 1

    async def test_a_failing_alternative_is_not_tried_twice(self):
        # Clef is known down, so the alternative is already answering; when it fails there is nothing left.
        with pytest.raises(HTTPException) as e:
            await _call(_BODY, cfg=_FB_CFG, upstream_backend=_clef_backend(), registry=_monitored((42, "unhealthy")),
                        outcome=DecisionBackendError("backend 502", 502))
        assert e.value.status_code == 502
        backend, crud = e.value.mocks[0], e.value.mocks[1]
        assert backend.decide.await_count == 1 and crud.create_request.await_count == 1

    async def test_down_with_no_alternative_is_503_with_retry_after(self):
        ub = _clef_backend()
        with pytest.raises(HTTPException) as e:
            await _call(_BODY, cfg={**_FB_CFG, "fallbacks": {}}, upstream_backend=ub, registry=_monitored((42, "disabled")))
        assert e.value.status_code == 503 and "disabled" in e.value.detail and e.value.headers["Retry-After"]
        ub.answer.assert_not_awaited()
        e.value.mocks[1].create_request.assert_not_awaited()

    async def test_requests_are_recorded_against_the_registered_backend(self):
        reg = _monitored((42, None))
        result, _, crud, *_ = await _call(_BODY, cfg=_FB_CFG, upstream_backend=_clef_backend(), registry=reg)
        assert result["model"] == "clef"
        assert crud.update_request_started.call_args.kwargs["backend_id"] == 42
        assert crud.update_request_completed.call_args.kwargs["backend_id"] == 42
        reg.report_live_success.assert_awaited_once_with(42)

    async def test_live_failures_count_against_its_circuit_but_busy_does_not(self):
        for status_code, counted in ((502, True), (504, True), (503, False)):
            reg = _monitored((42, None))
            ub = _clef_backend(DecisionBackendError("x", status_code))
            await _call(_BODY, cfg=_FB_CFG, upstream_backend=ub, registry=reg)
            assert reg.report_live_failure.await_count == (1 if counted else 0), status_code

    async def test_an_unregistered_upstream_is_dialed_as_before(self):
        reg = _monitored((None, None))
        result, _, crud, *_ = await _call(_BODY, cfg=_FB_CFG, upstream_backend=_clef_backend(), registry=reg)
        assert result["model"] == "clef" and crud.update_request_started.call_args.kwargs["backend_id"] is None
        reg.report_live_success.assert_not_awaited()

    async def test_a_failed_lookup_dials_the_server_instead_of_failing_the_request(self):
        reg = _fake_registry([])
        reg.decision_server_state = AsyncMock(side_effect=TimeoutError("pool exhausted"))
        ub = _clef_backend()
        result, _, crud, *_ = await _call(_BODY, cfg=_FB_CFG, upstream_backend=ub, registry=reg)
        assert result["model"] == "clef" and "fallback" not in result["metadata"]
        ub.answer.assert_awaited_once()

    async def test_the_lookup_uses_the_requests_own_session(self):
        reg = _monitored((42, None))
        db = MagicMock(commit=AsyncMock(), rollback=AsyncMock())
        await _call(_BODY, cfg=_FB_CFG, upstream_backend=_clef_backend(), registry=reg, db=db)
        assert reg.decision_server_state.call_args.args == (_CLEF.url, db)

    async def test_bookkeeping_failure_never_fails_the_request(self):
        reg = _monitored((42, None))
        reg.report_live_success = AsyncMock(side_effect=RuntimeError("db down"))
        result, *_ = await _call(_BODY, cfg=_FB_CFG, upstream_backend=_clef_backend(), registry=reg)
        assert result["model"] == "clef"


class TestDecisionServerState:
    """The registry lookup itself, against stub backend rows."""

    async def _state(self, rows, url, open_circuits=(), db="request-session"):
        from backend.app.core.telemetry import registry as registry_mod
        from backend.app.core.telemetry.registry import BackendRegistry
        from backend.app.db.models import BackendStatus

        servers = [(i, u, BackendStatus(s)) for i, u, s in rows]
        reg = MagicMock()
        reg.is_backend_available = AsyncMock(side_effect=lambda bid: bid not in open_circuits)
        lookup = AsyncMock(return_value=servers)
        with patch.object(registry_mod.crud, "get_decision_servers", lookup):
            state = await BackendRegistry.decision_server_state(reg, url, db)
        if db is not None:
            lookup.assert_awaited_once_with(db)             # the caller's session, no second connection
        return state

    async def test_states(self):
        url = "https://aspen4.example.edu:8001"
        assert await self._state([], url) == (None, None)
        assert await self._state([(9, "https://other:8001", "healthy")], url) == (None, None)
        assert await self._state([(9, url + "/", "healthy")], url) == (9, None)          # trailing slash ignored
        assert await self._state([(9, url, "unknown")], url + "/") == (9, None)          # just registered: usable
        for status_ in ("unhealthy", "disabled", "draining"):
            assert await self._state([(9, url, status_)], url) == (9, status_)
        assert await self._state([(9, url, "healthy")], url, open_circuits={9}) == (9, "circuit open")

    async def test_the_same_server_spelled_differently_still_matches(self):
        # A mismatch here would mean a disabled server is still dialed, silently.
        registered = "https://aspen4.example.edu:8001"
        for spelling in ("https://Aspen4.Example.EDU:8001/", "HTTPS://aspen4.example.edu:8001"):
            assert await self._state([(9, registered, "disabled")], spelling) == (9, "disabled")
        assert await self._state([(9, "https://h", "disabled")], "https://h:443/") == (9, "disabled")
        assert await self._state([(9, "http://h:80/", "disabled")], "http://h") == (9, "disabled")
        assert await self._state([(9, "https://h:8001", "disabled")], "https://h:8002") == (None, None)
        assert await self._state([(9, "https://h:8001", "disabled")], "http://h:8001") == (None, None)

    async def test_without_a_session_it_opens_its_own(self):
        from backend.app.core.telemetry import registry as registry_mod

        class _Ctx:
            async def __aenter__(self):
                return "own-session"

            async def __aexit__(self, *a):
                return False

        with patch.object(registry_mod, "get_async_db_context", lambda: _Ctx()):
            assert await self._state([(9, "https://h:8001", "healthy")], "https://h:8001", db=None) == (9, None)
