############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_hotpath_trims.py: Unit tests for hot-path per-request trims
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Unit tests for the hot-path per-request trims.

Covers:
- /tokenize gate in cap_max_tokens: far-from-boundary requests skip the
  backend /tokenize HTTP call and cap against a conservative tiktoken
  bound; near-boundary and auto_truncate requests still count exactly
- token-count memoization on the request object (retries don't
  re-tokenize) and force_exact invalidation
- context-length-400 safety net in _proxy_with_retry (recount exactly,
  retry once on the same backend)
- scheduler estimate_tokens chars/4 estimator (no tiktoken encode)
- Job.request_data field deleted
- ollama.enforce_num_ctx TTL cache

Loads inference.py / policy.py with backend.app.db* and friends
pre-mocked in sys.modules (see MEMORY.md "Import Chain Gotcha"),
spec-loading the pure modules they need (canonical_schemas, translators,
queue, stream_coalesce) directly from files so collection survives the
stub pollution earlier test modules leave in sys.modules.  Everything is
restored afterward so later test modules get their original imports.
"""

import asyncio
import importlib.util
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import HTTPException

_APP_DIR = Path(__file__).resolve().parents[2]

# Sentinels for engine comparison (identity-based ==, per MEMORY.md)
_VLLM_SENTINEL = object()
_OLLAMA_SENTINEL = object()

_mock_db_models = MagicMock()
_mock_db_models.BackendEngine.VLLM = _VLLM_SENTINEL
_mock_db_models.BackendEngine.OLLAMA = _OLLAMA_SENTINEL

_MOCKED_MODULES = [
    "backend.app.db",
    "backend.app.db.session",
    "backend.app.db.crud",
    "backend.app.db.models",
    "backend.app.settings",
    "backend.app.logging_config",
    "backend.app.core.redis_client",
    "backend.app.core.telemetry",
    "backend.app.core.telemetry.registry",
    "backend.app.core.scheduler.policy",
    "backend.app.core.scheduler.queue",
    "backend.app.core.scheduler.routing",
]

# Pure modules spec-loaded from file and seeded under their canonical
# names so inference.py / policy.py import the real thing regardless of
# what earlier test modules left in sys.modules.
_SEEDED_MODULES = [
    "backend.app.core.canonical_schemas",
    "backend.app.core.stream_coalesce",
    "backend.app.core.translators",
]

_saved_modules = {
    k: sys.modules[k]
    for k in sys.modules
    if k in _MOCKED_MODULES or k in _SEEDED_MODULES
    or k.startswith("backend.app.core.translators.")
}

for _mod_name in _MOCKED_MODULES:
    sys.modules[_mod_name] = MagicMock()
sys.modules["backend.app.db.models"] = _mock_db_models


def _load_module(name: str, path: Path, register: bool = False, pkg_dir: Path = None):
    spec = importlib.util.spec_from_file_location(
        name, path,
        submodule_search_locations=[str(pkg_dir)] if pkg_dir else None,
    )
    mod = importlib.util.module_from_spec(spec)
    if register:
        # Must be in sys.modules before exec (packages import their own
        # submodules; inference/policy import these canonical names).
        sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


canonical = _load_module(
    "backend.app.core.canonical_schemas",
    _APP_DIR / "core" / "canonical_schemas.py", register=True,
)
_load_module(
    "backend.app.core.stream_coalesce",
    _APP_DIR / "core" / "stream_coalesce.py", register=True,
)
_load_module(
    "backend.app.core.translators",
    _APP_DIR / "core" / "translators" / "__init__.py", register=True,
    pkg_dir=_APP_DIR / "core" / "translators",
)
# Real queue.py (pure module — no db chain) so policy gets the real Job
queue_mod = _load_module(
    "backend.app.core.scheduler.queue",
    _APP_DIR / "core" / "scheduler" / "queue.py", register=True,
)

inf = _load_module("mr2_inference_under_test", _APP_DIR / "services" / "inference.py")
policy_mod = _load_module("mr2_policy_under_test", _APP_DIR / "core" / "scheduler" / "policy.py")

CanonicalChatRequest = canonical.CanonicalChatRequest
CanonicalMessage = canonical.CanonicalMessage
MessageRole = canonical.MessageRole

# Restore sys.modules so subsequent test modules get their originals.
for _mod_name in list(sys.modules):
    if (
        _mod_name in _MOCKED_MODULES or _mod_name in _SEEDED_MODULES
        or _mod_name.startswith("backend.app.core.translators.")
    ):
        if _mod_name in _saved_modules:
            sys.modules[_mod_name] = _saved_modules[_mod_name]
        else:
            sys.modules.pop(_mod_name, None)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

class _FakeTokenizeResponse:
    def __init__(self, count: int):
        self._count = count

    def raise_for_status(self):
        pass

    def json(self):
        return {"count": self._count}


class _FakeTokenizeClient:
    """Stands in for the shared httpx client; counts /tokenize calls."""

    def __init__(self, count: int):
        self.count = count
        self.calls = 0

    async def post(self, url, json=None):
        self.calls += 1
        return _FakeTokenizeResponse(self.count)


def _make_service(tokenize_count: int = 100):
    svc = inf.InferenceService.__new__(inf.InferenceService)
    svc.db = MagicMock()
    svc._settings = SimpleNamespace(
        backend_retry_max_attempts=3,
        backend_request_timeout_per_attempt=5,
        thinking_off_by_default=False,
    )
    svc._scheduler = AsyncMock()
    svc._registry = AsyncMock()
    svc._latency_tracker = AsyncMock()
    svc._http_client = None
    fake_client = _FakeTokenizeClient(tokenize_count)
    svc._get_http_client = AsyncMock(return_value=fake_client)
    return svc, fake_client


def _make_request(content: str = "hello world", max_tokens=None, auto_truncate=False):
    return CanonicalChatRequest(
        model="test-model",
        messages=[CanonicalMessage(role=MessageRole.USER, content=content)],
        max_tokens=max_tokens,
        auto_truncate=auto_truncate,
        request_id="req-1",
    )


def _vllm_backend():
    return SimpleNamespace(engine=_VLLM_SENTINEL, url="http://backend", id=1)


async def _conservative_bound(request) -> int:
    est = await inf._tiktoken_estimate(request)
    return int(est * 1.3) + 16 * len(request.messages) + inf._TOKEN_BUFFER


# ----------------------------------------------------------------------
# Tokenize gate
# ----------------------------------------------------------------------

class TestTokenizeGate:
    @pytest.mark.asyncio
    async def test_far_from_boundary_skips_tokenize(self):
        svc, client = _make_service()
        request = _make_request(max_tokens=1000)
        await svc.cap_max_tokens(request, _vllm_backend(), 32768)
        assert client.calls == 0
        assert request.max_tokens == 1000  # generous room — cap is a no-op

    @pytest.mark.asyncio
    async def test_far_from_boundary_no_max_tokens_hits_hard_cap(self):
        # Unset budget with room >= the hard cap: the exact path would also
        # cap at 65536, so the shortcut is provably equivalent to main.
        svc, client = _make_service()
        request = _make_request()
        assert 131072 - await _conservative_bound(_make_request()) >= inf._MAX_OUTPUT_TOKENS
        await svc.cap_max_tokens(request, _vllm_backend(), 131072)
        assert client.calls == 0
        assert request.max_tokens == inf._MAX_OUTPUT_TOKENS

    @pytest.mark.asyncio
    async def test_no_max_tokens_below_hard_cap_room_counts_exactly(self):
        # Unset budget without full-hard-cap room must NOT be shrunk by the
        # conservative bound — fall through to exact counting so the default
        # completion budget stays context - exact_input - buffer, as on main.
        svc, client = _make_service(tokenize_count=100)
        request = _make_request()
        await svc.cap_max_tokens(request, _vllm_backend(), 32768)
        assert client.calls == 1
        assert request.max_tokens == 32768 - 100 - inf._TOKEN_BUFFER

    @pytest.mark.asyncio
    async def test_near_boundary_calls_tokenize(self):
        svc, client = _make_service(tokenize_count=100)
        request = _make_request()
        # room = 2048 - bound (~1050) « hard cap → exact counting required
        await svc.cap_max_tokens(request, _vllm_backend(), 2048)
        assert client.calls == 1
        assert request.max_tokens == 2048 - 100 - inf._TOKEN_BUFFER

    @pytest.mark.asyncio
    async def test_requested_exceeding_room_calls_tokenize(self):
        svc, client = _make_service(tokenize_count=3200)
        request = _make_request(max_tokens=5000)
        request._est_input_tokens = 3194  # seed memo: bound=5192, room=3000
        await svc.cap_max_tokens(request, _vllm_backend(), 8192)
        assert client.calls == 1  # room >= 2048 but < requested 5000
        assert request.max_tokens == 8192 - 3200 - inf._TOKEN_BUFFER

    @pytest.mark.asyncio
    async def test_requested_within_room_skips_tokenize(self):
        svc, client = _make_service()
        request = _make_request(max_tokens=2500)
        request._est_input_tokens = 3194  # bound=5192, room=3000 >= 2500
        await svc.cap_max_tokens(request, _vllm_backend(), 8192)
        assert client.calls == 0
        assert request.max_tokens == 2500

    @pytest.mark.asyncio
    async def test_auto_truncate_forces_exact(self):
        svc, client = _make_service(tokenize_count=50)
        request = _make_request(max_tokens=1000, auto_truncate=True)
        await svc.cap_max_tokens(request, _vllm_backend(), 32768)
        assert client.calls == 1  # gate bypassed despite generous room

    @pytest.mark.asyncio
    async def test_exact_count_memoized_across_attempts(self):
        svc, client = _make_service(tokenize_count=100)
        request = _make_request()
        backend = _vllm_backend()
        await svc.cap_max_tokens(request, backend, 2048)
        await svc.cap_max_tokens(request, backend, 2048)  # retry attempt
        assert client.calls == 1
        assert request.max_tokens == 2048 - 100 - inf._TOKEN_BUFFER

    @pytest.mark.asyncio
    async def test_force_exact_invalidates_memo(self):
        svc, client = _make_service(tokenize_count=100)
        request = _make_request()
        backend = _vllm_backend()
        await svc.cap_max_tokens(request, backend, 2048)
        assert client.calls == 1
        await svc.cap_max_tokens(request, backend, 2048, force_exact=True)
        assert client.calls == 2  # memo dropped, /tokenize re-called

    @pytest.mark.asyncio
    async def test_ollama_never_calls_tokenize(self):
        svc, client = _make_service()
        request = _make_request()
        backend = SimpleNamespace(engine=_OLLAMA_SENTINEL, url="http://backend", id=2)
        await svc.cap_max_tokens(request, backend, 2048)  # near boundary
        assert client.calls == 0  # tiktoken estimate only


class TestContextLengthErrorDetection:
    def test_matches_context_length_text(self):
        assert inf._is_context_length_error(
            "This model's maximum context length is 131072 tokens."
        )

    def test_matches_max_model_len_in_dict(self):
        detail = {"error": {"message": "prompt exceeds max_model_len"}}
        assert inf._is_context_length_error(detail)

    def test_ignores_other_400s(self):
        assert not inf._is_context_length_error({"error": {"message": "invalid role"}})


class TestContextRecapSafetyNet:
    def _make_retry_service(self):
        svc, _client = _make_service()
        backend = _vllm_backend()
        models = [SimpleNamespace(name="test-model", context_length=2048,
                                  supports_thinking=False)]
        svc._route_request = AsyncMock(return_value=(backend, models))
        svc.cap_max_tokens = AsyncMock()
        job = SimpleNamespace(model="test-model", request_id="req-1")
        user = MagicMock()
        return svc, backend, job, user

    def _status_error(self, message: str):
        req = httpx.Request("POST", "http://backend/v1/chat/completions")
        resp = httpx.Response(400, json={"error": {"message": message}}, request=req)
        return httpx.HTTPStatusError("400", request=req, response=resp)

    @pytest.mark.asyncio
    async def test_context_400_recaps_exact_and_retries_same_backend(self):
        svc, backend, job, user = self._make_retry_service()
        svc._proxy_chat_request = AsyncMock(side_effect=[
            self._status_error("maximum context length is 2048 tokens"),
            {"ok": True},
        ])
        request = _make_request(max_tokens=1000)
        response, used_backend = await svc._proxy_with_retry(request, job, user)
        assert response == {"ok": True}
        assert used_backend is backend
        # normal cap, then forced-exact recap, then normal cap on retry
        force_flags = [c.kwargs.get("force_exact", False)
                       for c in svc.cap_max_tokens.await_args_list]
        assert force_flags == [False, True, False]

    @pytest.mark.asyncio
    async def test_non_context_400_raises_without_recap(self):
        svc, backend, job, user = self._make_retry_service()
        svc._proxy_chat_request = AsyncMock(
            side_effect=self._status_error("invalid role: banana"),
        )
        request = _make_request(max_tokens=1000)
        with pytest.raises(HTTPException) as exc_info:
            await svc._proxy_with_retry(request, job, user)
        assert exc_info.value.status_code == 400
        assert svc.cap_max_tokens.await_count == 1  # no forced recap

    @pytest.mark.asyncio
    async def test_context_400_recap_happens_at_most_once(self):
        svc, backend, job, user = self._make_retry_service()
        err = self._status_error("maximum context length is 2048 tokens")
        svc._proxy_chat_request = AsyncMock(side_effect=[err, err])
        request = _make_request(max_tokens=1000)
        with pytest.raises(HTTPException) as exc_info:
            await svc._proxy_with_retry(request, job, user)
        assert exc_info.value.status_code == 400  # second 400 surfaces
        force_flags = [c.kwargs.get("force_exact", False)
                       for c in svc.cap_max_tokens.await_args_list]
        assert force_flags.count(True) == 1


# ----------------------------------------------------------------------
# Coalescer drain on mid-stream backend failure
# ----------------------------------------------------------------------

def _make_stream_service(monkeypatch, chunks, error, ollama=False):
    """Service wired for stream_chat_completion / stream_ollama_chat with a
    proxy that yields `chunks` then raises `error`. Coalescing is enabled
    with a huge delay so intermediate chunks stay buffered until failure."""
    svc, _client = _make_service()
    svc._settings = SimpleNamespace(
        stream_coalesce_events=8,
        stream_coalesce_ms=50_000,
    )
    svc._check_quota = AsyncMock()
    svc._create_request_record = AsyncMock(
        return_value=SimpleNamespace(
            request_uuid="req-uuid-1", id=1, user_id=1, api_key_id=1,
        )
    )
    svc._scheduler = MagicMock()
    job = SimpleNamespace(request_id=None)
    svc._scheduler.create_job_from_chat_request = MagicMock(return_value=job)
    calls = []
    svc._fail_request = AsyncMock(side_effect=lambda *a, **k: calls.append("fail"))
    svc._complete_streaming_request = AsyncMock()
    backend = _vllm_backend()

    async def _proxy(*args, **kwargs):
        for chunk in chunks:
            yield chunk, backend
        raise error

    svc._proxy_stream_with_retry = _proxy
    monkeypatch.setattr(inf, "incr_inflight_tokens", AsyncMock())
    monkeypatch.setattr(inf, "decr_inflight_tokens", AsyncMock())
    return svc, calls


def _sse_chunk(content):
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "m",
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }


class TestCoalescerDrainOnFailure:
    @pytest.mark.asyncio
    async def test_sse_buffered_chunks_drained_before_raise(self, monkeypatch):
        # First chunk flushes (TTFT); chunks 2-3 sit in the coalescer when
        # the backend dies — they were already delivered and must reach the
        # client before the exception propagates (main wrote them per-event).
        svc, calls = _make_stream_service(
            monkeypatch,
            [_sse_chunk("Hello"), _sse_chunk("world"), _sse_chunk("!!")],
            httpx.RemoteProtocolError("peer closed connection"),
        )
        outputs = []
        with pytest.raises(httpx.RemoteProtocolError):
            async for out in svc.stream_chat_completion(
                _make_request(), MagicMock(), MagicMock(), MagicMock()
            ):
                outputs.append(out)
        assert len(outputs) == 2
        assert b"Hello" in outputs[0]
        assert b"world" in outputs[1] and b"!!" in outputs[1]
        # Scheduler slot released before the drain yield (handler ordering)
        assert calls == ["fail"]

    @pytest.mark.asyncio
    async def test_sse_cancelled_yields_nothing_after_buffer(self, monkeypatch):
        # CancelledError means the client is gone — never yield the buffer
        svc, calls = _make_stream_service(
            monkeypatch,
            [_sse_chunk("Hello"), _sse_chunk("world")],
            asyncio.CancelledError(),
        )
        outputs = []
        with pytest.raises(asyncio.CancelledError):
            async for out in svc.stream_chat_completion(
                _make_request(), MagicMock(), MagicMock(), MagicMock()
            ):
                outputs.append(out)
        assert len(outputs) == 1  # only the TTFT flush
        assert calls == ["fail"]

    @pytest.mark.asyncio
    async def test_ollama_buffered_chunks_drained_before_raise(self, monkeypatch):
        svc, calls = _make_stream_service(
            monkeypatch,
            [
                {"message": {"content": "Hello"}, "done": False},
                {"message": {"content": "world"}, "done": False},
                {"message": {"content": "!!"}, "done": False},
            ],
            httpx.ReadTimeout("stalled"),
        )
        outputs = []
        with pytest.raises(httpx.ReadTimeout):
            async for out in svc.stream_ollama_chat(
                _make_request(), MagicMock(), MagicMock(), MagicMock()
            ):
                outputs.append(out)
        assert len(outputs) == 2
        assert b"Hello" in outputs[0]
        assert b"world" in outputs[1] and b"!!" in outputs[1]
        assert calls == ["fail"]

    @pytest.mark.asyncio
    async def test_ollama_cancelled_yields_nothing_after_buffer(self, monkeypatch):
        svc, calls = _make_stream_service(
            monkeypatch,
            [
                {"message": {"content": "Hello"}, "done": False},
                {"message": {"content": "world"}, "done": False},
            ],
            asyncio.CancelledError(),
        )
        outputs = []
        with pytest.raises(asyncio.CancelledError):
            async for out in svc.stream_ollama_chat(
                _make_request(), MagicMock(), MagicMock(), MagicMock()
            ):
                outputs.append(out)
        assert len(outputs) == 1
        assert calls == ["fail"]


# ----------------------------------------------------------------------
# Scheduler estimator + Job slimming
# ----------------------------------------------------------------------

class TestCharsEstimator:
    def test_four_chars_per_token(self):
        assert policy_mod.SchedulerPolicy.estimate_tokens(None, "a" * 100) == 25

    def test_empty_string(self):
        assert policy_mod.SchedulerPolicy.estimate_tokens(None, "") == 0

    def test_short_string_floors_to_zero(self):
        assert policy_mod.SchedulerPolicy.estimate_tokens(None, "abc") == 0

    def test_no_tiktoken_import_in_policy(self):
        src = (_APP_DIR / "core" / "scheduler" / "policy.py").read_text()
        assert "import tiktoken" not in src
        assert "tiktoken.get_encoding" not in src


class TestJobSlimming:
    def test_job_has_no_request_data_field(self):
        assert "request_data" not in queue_mod.Job.__dataclass_fields__

    def test_policy_no_longer_dumps_request(self):
        src = (_APP_DIR / "core" / "scheduler" / "policy.py").read_text()
        assert "request_data" not in src

    def test_chat_job_builds_without_request_data(self):
        policy = policy_mod.SchedulerPolicy.__new__(policy_mod.SchedulerPolicy)
        request = _make_request(max_tokens=500)
        job = policy.create_job_from_chat_request(request, user_id=1, api_key_id=2)
        assert job.model == "test-model"
        assert job.estimated_completion_tokens == 500
        assert not hasattr(job, "request_data")


# ----------------------------------------------------------------------
# enforce_num_ctx TTL cache
# ----------------------------------------------------------------------

class _FakeSessionCM:
    async def __aenter__(self):
        return MagicMock()

    async def __aexit__(self, *args):
        return False


@pytest.fixture
def fake_db_session(monkeypatch):
    """Intercept the lazy `from backend.app.db.session import ...`."""
    mod = types.ModuleType("backend.app.db.session")
    mod.get_async_db_context = lambda: _FakeSessionCM()
    monkeypatch.setitem(sys.modules, "backend.app.db.session", mod)
    inf._enforce_num_ctx_cache = None
    yield
    inf._enforce_num_ctx_cache = None


class TestEnforceNumCtxCache:
    @pytest.mark.asyncio
    async def test_first_call_reads_db_and_caches(self, fake_db_session):
        inf.crud.get_config_json = AsyncMock(return_value=False)
        assert await inf._get_enforce_num_ctx() is False
        assert await inf._get_enforce_num_ctx() is False
        assert inf.crud.get_config_json.await_count == 1  # second call cached

    @pytest.mark.asyncio
    async def test_fresh_cache_skips_db(self, fake_db_session):
        inf.crud.get_config_json = AsyncMock(return_value=True)
        inf._enforce_num_ctx_cache = (False, time.monotonic())
        assert await inf._get_enforce_num_ctx() is False
        assert inf.crud.get_config_json.await_count == 0

    @pytest.mark.asyncio
    async def test_expired_cache_rereads_db(self, fake_db_session):
        inf.crud.get_config_json = AsyncMock(return_value=False)
        inf._enforce_num_ctx_cache = (True, time.monotonic() - inf._ENFORCE_NUM_CTX_TTL_S - 1)
        assert await inf._get_enforce_num_ctx() is False
        assert inf.crud.get_config_json.await_count == 1

    @pytest.mark.asyncio
    async def test_value_coerced_to_bool(self, fake_db_session):
        inf.crud.get_config_json = AsyncMock(return_value=1)
        assert await inf._get_enforce_num_ctx() is True
        assert inf._enforce_num_ctx_cache[0] is True
