############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_streaming_session_isolation.py: Detached streaming DB writes run on
# isolated sessions, never on the request-scoped session
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Streaming session race ("readexactly() called while another coroutine is
already waiting for incoming data").

Streaming routes build InferenceService on the request-scoped get_async_db
session.  After ``data: [DONE]`` the generator awaits a SHIELDED completion
write.  When the client closes right then, Starlette cancels the generator;
the shielded task keeps running detached, anyio swallows the cancel,
StreamingResponse returns and FastAPI tears down get_async_db — commit /
rollback / close on the SAME session while the detached write is awaiting a
MySQL reply.  Two coroutines on one aiomysql connection: readexactly(), MySQL
2013/2014, pool resets, lost accounting.  The mid-stream failure write had the
same shape.

The fix runs every detached write on its own isolated session from scalar ids
captured when the row is created.  Covered here:

- the post-[DONE] disconnect for stream_chat_completion and stream_ollama_chat
  (+ stream_ollama_generate), with the request-session teardown running while
  the detached write is in flight, against a fake session that raises the
  production RuntimeError on concurrent use (this FAILS on the pre-fix code);
- mid-stream disconnect and backend HTTP errors → the failure write;
- _run_completion_db retry classification on isolated sessions (1213 retries on
  a fresh session, 1205 once, 2013 logs once + DLP, CancelledError contract),
  and the non-streaming path keeping its self.db rollback semantics;
- db.session.isolated_async_session (a failing rollback never masks the
  original exception; shielded close; invalidate when close fails);
- the LOCK INVARIANT, structurally: after _create_request_record commits, the
  streaming path runs only allow-listed non-locking reads on self.db;
- pool hold-and-wait: the routing read's transaction on the request session is
  ended after every routing attempt, so no pooled connection is held while
  chunks stream or while a detached write opens its own session;
- the generator-level asyncio.shield (disconnect before the write starts) and
  the failure write's error path (logged, never rolls back self.db).

inference.py and db/session.py are spec-loaded with their backend.app.*
dependencies stubbed only for the duration of the load (module-scoped
fixtures; every touched sys.modules key restored in a finally).  Call-time
imports (db.session, dlp_worker, core.redis_client) are pinned per test with
monkeypatch.setitem, using fresh stub module objects.
"""

import ast
import asyncio
import importlib.util
import re
import sys
import types
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

_APP_DIR = Path(__file__).resolve().parents[2]
_INFERENCE_PATH = _APP_DIR / "services" / "inference.py"
_SESSION_PATH = _APP_DIR / "db" / "session.py"
_CRUD_PATH = _APP_DIR / "db" / "crud.py"

READEXACTLY = (
    "readexactly() called while another coroutine is already waiting for "
    "incoming data"
)
_TIMEOUT = 5.0

REQUEST_ID, USER_ID, API_KEY_ID = 101, 7, 9
BACKEND_ID = 3
USER = SimpleNamespace(id=USER_ID)
API_KEY = SimpleNamespace(id=API_KEY_ID)


# ----------------------------------------------------------------------
# Loading (no module-level sys.modules writes)
# ----------------------------------------------------------------------

def _exec_file(name: str, path: Path, register: bool):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    if register:
        sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_with_stubs(name: str, path: Path, stubs: dict, seeded: dict = None):
    """Exec ``path`` with ``stubs`` force-installed and ``seeded`` real pure
    modules registered, restoring every touched key afterwards (even if the
    load fails)."""
    seeded = seeded or {}
    touched = set(stubs) | set(seeded)
    saved = {key: sys.modules[key] for key in touched if key in sys.modules}
    try:
        for key, stub in stubs.items():
            sys.modules[key] = stub
        for key, seed_path in seeded.items():
            _exec_file(key, seed_path, register=True)
        return _exec_file(name, path, register=False)
    finally:
        for key in touched:
            if key in saved:
                sys.modules[key] = saved[key]
            else:
                sys.modules.pop(key, None)


@pytest.fixture(scope="module")
def inf():
    stubbed = [
        "backend.app.db",
        "backend.app.db.session",
        "backend.app.db.crud",
        "backend.app.db.models",
        "backend.app.settings",
        "backend.app.logging_config",
        "backend.app.core.redis_client",
        "backend.app.core.quota_budget",
        "backend.app.core.telemetry",
        "backend.app.core.telemetry.registry",
        "backend.app.core.scheduler.policy",
        "backend.app.core.scheduler.queue",
        "backend.app.core.translators",
        "backend.app.core.translators.vllm_out",
    ]
    return _load_with_stubs(
        "mr2_inference_session_isolation",
        _INFERENCE_PATH,
        stubs={key: MagicMock() for key in stubbed},
        seeded={
            "backend.app.core.canonical_schemas": _APP_DIR / "core" / "canonical_schemas.py",
            "backend.app.core.stream_coalesce": _APP_DIR / "core" / "stream_coalesce.py",
        },
    )


@pytest.fixture(scope="module")
def session_mod():
    settings = types.ModuleType("backend.app.settings")
    settings.get_settings = lambda: SimpleNamespace(
        database_url="mysql+pymysql://test:test@127.0.0.1:9/test",
        database_pool_size=1,
        database_max_overflow=0,
        database_echo=False,
        archive_database_url=None,
    )
    return _load_with_stubs(
        "mr2_db_session_isolation", _SESSION_PATH,
        stubs={"backend.app.settings": settings},
    )


# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------

class OperationalError(Exception):
    """Shape of sqlalchemy.exc.OperationalError: MySQL code at .orig.args[0]."""

    def __init__(self, code: int):
        super().__init__(f"({code}) simulated")
        self.orig = SimpleNamespace(args=(code, "simulated"))


class _Hold:
    """Parks one DB statement mid-flight, as if awaiting the MySQL reply."""

    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self):
        self.started.set()
        await self.release.wait()


_TXN_END_OPS = frozenset({"commit", "rollback", "close"})


class FakeSession:
    """AsyncSession stand-in bound to ONE simulated aiomysql connection.

    Every awaited operation holds the connection for a few loop turns; a
    second task using it meanwhile raises the production RuntimeError — and is
    recorded, so a swallowed error still fails the test.  ``in_txn`` mirrors
    an autobegun transaction (a checked-out pooled connection): any statement
    sets it, a successful commit / rollback / close clears it.
    """

    def __init__(self, name, holds=None, fail=None, on_rollback=None):
        self.name = name
        self.ops = []
        self.violations = []
        self.commits = 0
        self.rollbacks = 0
        self.closed = False
        self.invalidated = False
        self._holds = holds if holds is not None else {}
        self._fail = dict(fail or {})
        self._on_rollback = on_rollback
        self._holder = None
        self.in_txn = False

    async def io(self, op):
        me = asyncio.current_task()
        if self._holder is not None and self._holder is not me:
            self.violations.append(op)
            raise RuntimeError(READEXACTLY)
        self._holder = me
        try:
            self.ops.append(op)
            if op not in _TXN_END_OPS:
                self.in_txn = True
            hold = self._holds.get(op)
            if hold is not None:
                await hold()
            for _ in range(3):
                await asyncio.sleep(0)
            exc = self._fail.pop(op, None)
            if exc is not None:
                raise exc
            if op in _TXN_END_OPS:
                self.in_txn = False
        finally:
            self._holder = None

    async def commit(self):
        await self.io("commit")
        self.commits += 1

    async def rollback(self):
        await self.io("rollback")
        self.rollbacks += 1
        if self._on_rollback is not None:
            self._on_rollback()

    async def close(self):
        await self.io("close")
        self.closed = True

    async def invalidate(self):
        self.ops.append("invalidate")
        self.invalidated = True


class FakeDbRequest:
    """ORM row stand-in.  Once ``frozen`` (routing began) every attribute read
    is recorded as a late read; once the request session rolls back it is
    expired and reads raise, as they do on an async SQLAlchemy session."""

    _FIELDS = {
        "id": REQUEST_ID,
        "user_id": USER_ID,
        "api_key_id": API_KEY_ID,
        "request_uuid": "req-uuid-101",
        "is_streaming": True,
    }

    def __init__(self):
        self.frozen = False
        self.expired = False
        self.late_reads = []

    def __getattr__(self, name):
        fields = type(self)._FIELDS
        if name not in fields:
            raise AttributeError(name)
        if self.frozen:
            self.late_reads.append(name)
        if self.expired:
            raise RuntimeError(
                f"MissingGreenlet: expired attribute {name!r} read after the "
                "request session rolled back"
            )
        return fields[name]


class FakeCrud:
    """The crud functions the completion/failure writes call.  Each runs one
    statement on the session it was given; ``failures[fn]`` queues exceptions
    raised after the statement."""

    def __init__(self):
        self.calls = []
        self.failures = {}
        self.incr_quota_redis = AsyncMock()

    async def _run(self, fn, db, *args):
        self.calls.append((fn, db.name, args))
        await db.io(fn)
        queue = self.failures.get(fn)
        if queue:
            raise queue.pop(0)

    async def update_request_completed(self, db, request_id, **_):
        await self._run("update_request_completed", db, request_id)

    async def create_response(self, db, request_id, **_):
        await self._run("create_response", db, request_id)

    async def update_quota_usage(self, db, user_id, tokens):
        await self._run("update_quota_usage", db, user_id, tokens)

    async def update_api_key_usage(self, db, api_key_id):
        await self._run("update_api_key_usage", db, api_key_id)

    async def update_request_failed(self, db, request_id, error_message=None, **_):
        await self._run("update_request_failed", db, request_id)

    async def mark_model_loaded(self, db, backend_id, model):
        await self._run("mark_model_loaded", db, backend_id, model)


class _Env:
    """Per-test world: the request session, the ORM row, isolated sessions."""

    def __init__(self):
        self.holds = {}
        self.db_request = FakeDbRequest()
        self.request_db = FakeSession(
            "request", holds=self.holds, on_rollback=self._expire_row,
        )
        self.isolated = []
        # failures raised by every isolated session, e.g. {"commit": exc}
        self.isolated_fail = {}
        # the request session's in_txn each time a detached write opened a session
        self.request_session_held_at_isolated_open = []
        self.crud = FakeCrud()
        self.logger = MagicMock()
        self.dlp_enqueued = asyncio.Event()
        self.enqueue_for_dlp = AsyncMock(side_effect=lambda rid: self.dlp_enqueued.set())

    def _expire_row(self):
        self.db_request.expired = True

    def hold(self, op) -> _Hold:
        self.holds[op] = _Hold()
        return self.holds[op]

    def logged(self, event):
        return [c for c in self.logger.error.call_args_list if c.args and c.args[0] == event]

    def request_session_crud(self):
        return [c for c in self.crud.calls if c[1] == "request"]

    # patched db.session.isolated_async_session: faithful minimal mimic
    @asynccontextmanager
    async def isolated_async_session(self):
        self.request_session_held_at_isolated_open.append(self.request_db.in_txn)
        session = FakeSession(
            f"isolated-{len(self.isolated) + 1}", holds=self.holds, fail=self.isolated_fail,
        )
        self.isolated.append(session)
        try:
            yield session
        except BaseException:
            try:
                await asyncio.shield(session.rollback())
            except BaseException:
                pass
            raise
        finally:
            try:
                await asyncio.shield(session.close())
            except BaseException:
                pass

    # db.session.get_async_db_context, used by mark_model_loaded
    @asynccontextmanager
    async def get_async_db_context(self):
        yield FakeSession("mark", holds=self.holds)


@pytest.fixture
def env(inf, monkeypatch):
    e = _Env()
    session_stub = types.ModuleType("backend.app.db.session")
    session_stub.isolated_async_session = e.isolated_async_session
    session_stub.get_async_db_context = e.get_async_db_context
    dlp_stub = types.ModuleType("backend.app.services.dlp_worker")
    dlp_stub.enqueue_for_dlp = e.enqueue_for_dlp
    redis_stub = types.ModuleType("backend.app.core.redis_client")
    redis_stub.incr_cluster_tokens = AsyncMock()
    core_stub = types.ModuleType("backend.app.core")
    core_stub.__path__ = []
    core_stub.redis_client = redis_stub
    monkeypatch.setitem(sys.modules, "backend.app.db.session", session_stub)
    monkeypatch.setitem(sys.modules, "backend.app.services.dlp_worker", dlp_stub)
    monkeypatch.setitem(sys.modules, "backend.app.core", core_stub)
    monkeypatch.setitem(sys.modules, "backend.app.core.redis_client", redis_stub)
    monkeypatch.setattr(inf, "crud", e.crud)
    monkeypatch.setattr(inf, "logger", e.logger)
    monkeypatch.setattr(inf, "incr_inflight_tokens", AsyncMock())
    monkeypatch.setattr(inf, "decr_inflight_tokens", AsyncMock())
    return e


# ----------------------------------------------------------------------
# Harness
# ----------------------------------------------------------------------

def _make_service(inf, env, proxy):
    svc = inf.InferenceService.__new__(inf.InferenceService)
    svc.db = env.request_db
    svc._settings = SimpleNamespace(
        stream_coalesce_events=0,
        stream_coalesce_ms=0,
        audit_log_enabled=True,
        audit_log_responses=True,
        backend_retry_max_attempts=3,
        thinking_off_by_default=False,
    )
    svc._http_client = None
    svc._pending_prompt_redactions = []
    svc._registry = AsyncMock()
    svc._latency_tracker = AsyncMock()
    svc._check_quota = AsyncMock()
    svc._create_request_record = AsyncMock(return_value=env.db_request)
    job = SimpleNamespace(
        request_id=None, model="test-model",
        estimated_prompt_tokens=11, assigned_backend_id=None,
    )
    scheduler = MagicMock()
    scheduler.create_job_from_chat_request = MagicMock(return_value=job)
    scheduler.on_job_completed = AsyncMock()
    scheduler.on_job_failed = AsyncMock()
    scheduler.cancel_job = AsyncMock()
    scheduler.estimate_tokens = MagicMock(return_value=4)
    svc._scheduler = scheduler
    if proxy is not None:
        svc._proxy_stream_with_retry = proxy
    return svc, job


def _proxy(env, chunks, then=None):
    """Stands in for _proxy_stream_with_retry: yields chunks, then ``then()``."""
    backend = SimpleNamespace(id=BACKEND_ID)

    async def proxy(request, job, user, proxy_fn="_proxy_stream_request", modality=None):
        # Routing began: the row is committed and the scalars captured.  From
        # here on nothing may read the ORM row.
        env.db_request.frozen = True
        for chunk in chunks:
            yield chunk, backend
        if then is not None:
            await then()

    return proxy


def _request():
    return SimpleNamespace(request_id="client-id", include_usage=False, model="test-model")


def _sse(content=None, finish=None):
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "test-model",
        "choices": [{
            "index": 0,
            "delta": {"content": content} if content else {},
            "finish_reason": finish,
        }],
    }


_OPENAI_CHUNKS = [
    _sse("Hello"),
    _sse(" world"),
    _sse(finish="stop"),
    {  # vLLM include_usage chunk: real counts, empty choices, not forwarded
        "id": "chatcmpl-1", "object": "chat.completion.chunk",
        "created": 1700000000, "model": "test-model", "choices": [],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
    },
]

# usage None → prompt = job.estimated_prompt_tokens (11) + estimate_tokens (4)
_OLLAMA_CHUNKS = [
    {"message": {"content": "Hello"}, "done": False},
    {"message": {"content": " world"}, "done": True, "done_reason": "stop"},
]

_TEARDOWN_OPS = {"commit": ["commit", "close"], "rollback": ["rollback", "close"]}


async def _drain(stream, received):
    """What Starlette's StreamingResponse does with the body iterator."""
    async for chunk in stream:
        received.append(chunk)


async def _get_async_db_teardown(session, outcome):
    """get_async_db's teardown on the request session once StreamingResponse
    returned: commit on normal exit or rollback on the exception branch, then
    a shielded close (invalidate if that fails)."""
    try:
        if outcome == "commit":
            await session.commit()
        else:
            await asyncio.shield(session.rollback())
    except Exception:
        try:
            await asyncio.shield(session.rollback())
        except Exception:
            pass
    finally:
        try:
            await asyncio.shield(session.close())
        except Exception:
            try:
                await session.invalidate()
            except Exception:
                pass


async def _wait(event, what):
    try:
        await asyncio.wait_for(event.wait(), timeout=_TIMEOUT)
    except asyncio.TimeoutError:
        pytest.fail(f"timed out waiting for {what}")


async def _eventually(predicate, what):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _TIMEOUT
    while not predicate():
        if loop.time() > deadline:
            pytest.fail(f"timed out waiting for {what}")
        await asyncio.sleep(0.001)


async def _spin(turns=30):
    for _ in range(turns):
        await asyncio.sleep(0)


async def _disconnect_after_terminal_event(inf, env, generator, chunks, teardown):
    """Client reads the terminal event and closes while the post-[DONE]
    completion write is awaiting MySQL; FastAPI then tears down get_async_db."""
    svc, job = _make_service(inf, env, _proxy(env, chunks))
    write = env.hold("update_request_completed")
    received = []
    stream = getattr(svc, generator)(_request(), USER, API_KEY, MagicMock())
    consumer = asyncio.ensure_future(_drain(stream, received))

    # The generator is parked in the shielded completion; its first statement
    # is in flight.
    await _wait(write.started, "the detached completion write")
    # Client disconnect: Starlette cancels the task iterating the body.
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer
    # anyio swallows that cancellation, StreamingResponse returns and FastAPI
    # runs the get_async_db teardown while the detached write is still running.
    await _get_async_db_teardown(env.request_db, teardown)
    write.release.set()
    await _wait(env.dlp_enqueued, "the post-commit DLP enqueue")
    await _spin()
    return svc, job, received, consumer


def _assert_completion_committed_on_isolated_session(env, svc, job, consumer, teardown):
    assert env.request_db.violations == [], (
        "request-scoped session used by two tasks at once (readexactly race)"
    )
    assert env.request_session_crud() == [], "detached write ran on self.db"
    assert env.request_db.ops == _TEARDOWN_OPS[teardown]
    assert [s.name for s in env.isolated] == ["isolated-1"]
    iso = env.isolated[0]
    assert iso.violations == []
    assert iso.ops == [
        "update_request_completed", "create_response",
        "update_quota_usage", "update_api_key_usage", "commit", "close",
    ]
    assert (iso.commits, iso.rollbacks, iso.closed) == (1, 0, True)
    assert ("update_request_completed", "isolated-1", (REQUEST_ID,)) in env.crud.calls
    assert ("update_quota_usage", "isolated-1", (USER_ID, 15)) in env.crud.calls
    assert ("update_api_key_usage", "isolated-1", (API_KEY_ID,)) in env.crud.calls
    env.crud.incr_quota_redis.assert_awaited_once_with(USER_ID, 15)
    env.enqueue_for_dlp.assert_awaited_once_with(REQUEST_ID)
    assert env.db_request.late_reads == [], "detached path read the ORM row"
    assert consumer.cancelled(), "generator cancellation must still propagate"
    svc._scheduler.on_job_completed.assert_awaited_once_with(job, BACKEND_ID, 15)
    svc._scheduler.on_job_failed.assert_not_awaited()
    assert env.logged("request_completion_db_failed") == []


# ----------------------------------------------------------------------
# (a)/(b) disconnect right after the terminal event
# ----------------------------------------------------------------------

class TestDisconnectAfterTerminalEvent:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("teardown", ["commit", "rollback"])
    async def test_stream_chat_completion(self, inf, env, teardown):
        svc, job, received, consumer = await _disconnect_after_terminal_event(
            inf, env, "stream_chat_completion", _OPENAI_CHUNKS, teardown,
        )
        assert received[-1].endswith(b"data: [DONE]\n\n")
        _assert_completion_committed_on_isolated_session(env, svc, job, consumer, teardown)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("teardown", ["commit", "rollback"])
    @pytest.mark.parametrize("generator", ["stream_ollama_chat", "stream_ollama_generate"])
    async def test_stream_ollama(self, inf, env, generator, teardown):
        svc, job, received, consumer = await _disconnect_after_terminal_event(
            inf, env, generator, _OLLAMA_CHUNKS, teardown,
        )
        assert b'"done": true' in received[-1]
        _assert_completion_committed_on_isolated_session(env, svc, job, consumer, teardown)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("held", ["on_job_completed", "mark_model_loaded"])
    @pytest.mark.parametrize("generator,chunks", [
        ("stream_chat_completion", _OPENAI_CHUNKS),
        ("stream_ollama_chat", _OLLAMA_CHUNKS),
    ])
    async def test_disconnect_before_the_write_starts(self, inf, env, generator, chunks, held):
        """The generator-level asyncio.shield keeps _complete_streaming_request
        running when the client leaves while it is still releasing the slot or
        marking the model loaded, before the inner-shielded write exists."""
        svc, job = _make_service(inf, env, _proxy(env, chunks))
        if held == "on_job_completed":
            gate = _Hold()

            async def on_job_completed(*_args):
                await gate()

            svc._scheduler.on_job_completed = AsyncMock(side_effect=on_job_completed)
        else:
            gate = env.hold("mark_model_loaded")
        received = []
        consumer = asyncio.ensure_future(
            _drain(getattr(svc, generator)(_request(), USER, API_KEY, MagicMock()), received)
        )
        await _wait(gate.started, held)
        assert env.isolated == [], "the completion write must not have started yet"

        consumer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consumer
        await _get_async_db_teardown(env.request_db, "commit")
        gate.release.set()
        await _wait(env.dlp_enqueued, "the post-commit DLP enqueue")
        await _spin()
        _assert_completion_committed_on_isolated_session(env, svc, job, consumer, "commit")


# ----------------------------------------------------------------------
# (c) failure writes from the streaming generators
# ----------------------------------------------------------------------

_GENERATOR_FIRST_CHUNK = [
    ("stream_chat_completion", _sse("Hello")),
    ("stream_ollama_chat", {"message": {"content": "Hello"}, "done": False}),
]


async def _mid_stream_disconnect(inf, env, generator, first_chunk):
    """Client disconnects mid-stream (backend stalled); FastAPI tears down
    get_async_db while the detached failure write is in flight."""
    stalled = asyncio.Event()  # the backend never sends another chunk
    svc, job = _make_service(inf, env, _proxy(env, [first_chunk], then=stalled.wait))
    fail = env.hold("update_request_failed")
    received = []
    consumer = asyncio.ensure_future(
        _drain(getattr(svc, generator)(_request(), USER, API_KEY, MagicMock()), received)
    )
    await _eventually(lambda: received, "the first chunk")

    consumer.cancel()  # client disconnects mid-stream
    await _wait(fail.started, "the detached failure write")
    # anyio keeps re-delivering cancellation while its scope is cancelled:
    # the generator's await on the shielded failure write is interrupted
    # too, so that write outlives the generator.
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer
    await _get_async_db_teardown(env.request_db, "rollback")
    fail.release.set()
    await _eventually(
        lambda: any(s.closed for s in env.isolated) or env.request_db.commits,
        "the failure write to finish",
    )
    await _spin()
    return svc, job, consumer


async def _backend_http_error_before_first_chunk(inf, env, generator):
    async def reject():
        raise inf.HTTPException(status_code=503, detail="No suitable backend: busy")

    svc, job = _make_service(inf, env, _proxy(env, [], then=reject))
    received = []
    # Draining to the end without raising also proves no exception escaped
    # the generator's failure branch.
    await asyncio.wait_for(
        _drain(getattr(svc, generator)(_request(), USER, API_KEY, MagicMock()), received),
        timeout=_TIMEOUT,
    )
    await _spin()
    return svc, job, received


class TestStreamingFailureWrite:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("generator,first_chunk", _GENERATOR_FIRST_CHUNK)
    async def test_mid_stream_disconnect(self, inf, env, generator, first_chunk):
        svc, job, consumer = await _mid_stream_disconnect(inf, env, generator, first_chunk)

        assert env.request_db.violations == [], (
            "request-scoped session used by two tasks at once (readexactly race)"
        )
        assert env.request_session_crud() == [], "failure write ran on self.db"
        assert env.request_db.ops == ["rollback", "close"]
        assert [s.name for s in env.isolated] == ["isolated-1"]
        assert env.isolated[0].ops == [
            "update_request_failed", "update_api_key_usage", "commit", "close",
        ]
        assert ("update_request_failed", "isolated-1", (REQUEST_ID,)) in env.crud.calls
        assert ("update_api_key_usage", "isolated-1", (API_KEY_ID,)) in env.crud.calls
        assert env.db_request.late_reads == [], "failure path read the ORM row"
        assert consumer.cancelled()
        svc._scheduler.on_job_failed.assert_awaited_once_with(job, BACKEND_ID)
        svc._scheduler.on_job_completed.assert_not_awaited()
        assert env.logged("request_fail_db_failed") == []
        assert env.logged("request_fail_db_skipped") == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("generator", ["stream_chat_completion", "stream_ollama_chat"])
    async def test_backend_http_error_before_first_chunk(self, inf, env, generator):
        svc, job, received = await _backend_http_error_before_first_chunk(inf, env, generator)

        assert b"No suitable backend: busy" in received[-1]
        assert env.request_session_crud() == [], "failure write ran on self.db"
        assert env.request_db.ops == []
        assert [s.name for s in env.isolated] == ["isolated-1"]
        assert env.isolated[0].ops == [
            "update_request_failed", "update_api_key_usage", "commit", "close",
        ]
        assert ("update_request_failed", "isolated-1", (REQUEST_ID,)) in env.crud.calls
        assert ("update_api_key_usage", "isolated-1", (API_KEY_ID,)) in env.crud.calls
        assert env.db_request.late_reads == []
        svc._scheduler.cancel_job.assert_awaited_once_with("req-uuid-101")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("failing", ["update_request_failed", "commit"])
    @pytest.mark.parametrize("scenario", ["http_error", "mid_stream_disconnect"])
    @pytest.mark.parametrize("generator,first_chunk", _GENERATOR_FIRST_CHUNK)
    async def test_failing_failure_write_logs_and_leaves_request_session_alone(
        self, inf, env, generator, first_chunk, scenario, failing,
    ):
        lost = OperationalError(2013)
        if failing == "commit":
            env.isolated_fail["commit"] = lost
            written = ["update_request_failed", "update_api_key_usage", "commit"]
        else:
            env.crud.failures["update_request_failed"] = [lost]
            written = ["update_request_failed"]

        if scenario == "http_error":
            _svc, _job, received = await _backend_http_error_before_first_chunk(
                inf, env, generator,
            )
            assert b"No suitable backend: busy" in received[-1]
            teardown_ops = []
        else:
            _svc, _job, consumer = await _mid_stream_disconnect(inf, env, generator, first_chunk)
            assert consumer.cancelled()
            teardown_ops = ["rollback", "close"]

        # Only get_async_db's teardown touched the request session: the
        # detached failure path never rolls it back.
        assert env.request_db.ops == teardown_ops
        assert env.request_db.violations == []
        assert [s.name for s in env.isolated] == ["isolated-1"]
        iso = env.isolated[0]
        assert iso.ops == written + ["rollback", "close"]
        assert (iso.commits, iso.closed) == (0, True)
        failures = env.logged("request_fail_db_failed")
        assert len(failures) == 1, "a failed failure write must be logged loudly, once"
        assert failures[0].kwargs == {"request_id": REQUEST_ID, "error": "OperationalError"}


# ----------------------------------------------------------------------
# (c2) request session released before streaming (pool hold-and-wait)
# ----------------------------------------------------------------------

_STREAM_BACKEND = SimpleNamespace(id=BACKEND_ID, engine="vllm")

_STREAM_VARIANTS = [
    ("stream_chat_completion", _OPENAI_CHUNKS, "_proxy_stream_request"),
    ("stream_ollama_chat", _OLLAMA_CHUNKS, "_proxy_ollama_stream"),
]


def _routing(env, outcomes=()):
    """Stands in for _route_request: the quota read autobegins a transaction on
    the request session (checking out its pooled connection, as
    crud.get_user_quota does), then routes, or raises the next queued outcome."""
    queued = list(outcomes)
    calls = []

    async def route(job, user, modality=None, exclude_backend_ids=None, max_wait=None):
        env.db_request.frozen = True
        calls.append(set(exclude_backend_ids) if exclude_backend_ids else None)
        await env.request_db.io("get_user_quota")
        if queued:
            raise queued.pop(0)
        return _STREAM_BACKEND, []

    route.calls = calls
    return route


def _backend_stream(env, chunks, fail_first_attempt=None):
    """Stands in for the proxy_fn; records whether the request session still
    held its connection as each chunk went out."""
    attempts = []
    held = []

    async def proxy_fn(request, backend):
        attempts.append(backend.id)
        if fail_first_attempt is not None and len(attempts) == 1:
            raise fail_first_attempt
        for chunk in chunks:
            held.append(env.request_db.in_txn)
            yield chunk

    proxy_fn.attempts = attempts
    proxy_fn.held = held
    return proxy_fn


_ISOLATED_COMPLETION_OPS = [
    "update_request_completed", "create_response",
    "update_quota_usage", "update_api_key_usage", "commit", "close",
]
_ISOLATED_FAILURE_OPS = ["update_request_failed", "update_api_key_usage", "commit", "close"]


class TestRequestSessionReleasedBeforeStreaming:
    """Real _proxy_stream_with_retry.  If routing's quota read kept the request
    session's pooled connection for the whole stream, every detached write would
    need a SECOND connection while the first is held: a worker pool full of
    finishing streams then times every write out (non-retryable TimeoutError,
    accounting lost)."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("generator,chunks,proxy_fn", _STREAM_VARIANTS)
    async def test_no_connection_held_while_streaming_or_writing(
        self, inf, env, generator, chunks, proxy_fn,
    ):
        svc, job = _make_service(inf, env, proxy=None)
        svc._route_request = _routing(env)
        backend_stream = _backend_stream(env, chunks)
        setattr(svc, proxy_fn, backend_stream)
        received = []
        await asyncio.wait_for(
            _drain(getattr(svc, generator)(_request(), USER, API_KEY, MagicMock()), received),
            timeout=_TIMEOUT,
        )
        await _spin()

        assert env.request_db.ops == ["get_user_quota", "commit"]
        assert backend_stream.held == [False] * len(chunks)
        assert env.request_session_held_at_isolated_open == [False]
        assert env.isolated[0].ops == _ISOLATED_COMPLETION_OPS
        svc._scheduler.on_job_completed.assert_awaited_once_with(job, BACKEND_ID, 15)
        env.enqueue_for_dlp.assert_awaited_once_with(REQUEST_ID)

    @pytest.mark.asyncio
    async def test_released_again_after_a_retry_reroutes(self, inf, env):
        svc, job = _make_service(inf, env, proxy=None)
        svc._route_request = _routing(env)
        backend_stream = _backend_stream(
            env, _OPENAI_CHUNKS, fail_first_attempt=httpx.ConnectError("connection refused"),
        )
        svc._proxy_stream_request = backend_stream
        received = []
        await asyncio.wait_for(
            _drain(svc.stream_chat_completion(_request(), USER, API_KEY, MagicMock()), received),
            timeout=_TIMEOUT,
        )
        await _spin()

        assert svc._route_request.calls == [None, {BACKEND_ID}]
        assert backend_stream.attempts == [BACKEND_ID, BACKEND_ID]
        assert env.request_db.ops == ["get_user_quota", "commit"] * 2
        assert backend_stream.held == [False] * len(_OPENAI_CHUNKS)
        assert env.request_session_held_at_isolated_open == [False]
        assert received[-1].endswith(b"data: [DONE]\n\n")
        svc._scheduler.on_job_failed.assert_awaited_once_with(job, BACKEND_ID)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("generator", ["stream_chat_completion", "stream_ollama_chat"])
    async def test_routing_failure_released_before_the_failure_write(self, inf, env, generator):
        svc, _job = _make_service(inf, env, proxy=None)
        svc._route_request = _routing(env, outcomes=[
            inf.HTTPException(status_code=503, detail="No suitable backend: busy"),
        ])
        received = []
        await asyncio.wait_for(
            _drain(getattr(svc, generator)(_request(), USER, API_KEY, MagicMock()), received),
            timeout=_TIMEOUT,
        )
        await _spin()

        assert b"No suitable backend: busy" in received[-1]
        assert env.request_db.ops == ["get_user_quota", "commit"]
        assert env.request_session_held_at_isolated_open == [False]
        assert env.isolated[0].ops == _ISOLATED_FAILURE_OPS
        svc._scheduler.cancel_job.assert_awaited_once_with("req-uuid-101")

    @pytest.mark.asyncio
    async def test_cancelled_while_routing_starts_no_request_session_io(self, inf, env):
        waiting = _Hold()  # queued for backend capacity that never frees up

        async def route(job, user, modality=None, exclude_backend_ids=None, max_wait=None):
            await env.request_db.io("get_user_quota")
            await waiting()
            return _STREAM_BACKEND, []

        svc, _job = _make_service(inf, env, proxy=None)
        svc._route_request = route
        received = []
        consumer = asyncio.ensure_future(
            _drain(svc.stream_chat_completion(_request(), USER, API_KEY, MagicMock()), received)
        )
        await _wait(waiting.started, "routing to wait for capacity")
        consumer.cancel()  # the client gives up while queued
        with pytest.raises(asyncio.CancelledError):
            await consumer
        await _eventually(lambda: env.isolated and env.isolated[0].closed, "the failure write")
        await _get_async_db_teardown(env.request_db, "rollback")
        await _spin()

        # No commit is started in the cancelled task (anyio would interrupt it
        # and the connection would be discarded); the teardown returns it.
        assert env.request_db.ops == ["get_user_quota", "rollback", "close"]
        assert env.request_db.violations == []
        assert env.isolated[0].ops == _ISOLATED_FAILURE_OPS


# ----------------------------------------------------------------------
# (d) _run_completion_db on isolated sessions
# ----------------------------------------------------------------------

_WRITE_OPS = [
    "update_request_completed", "create_response",
    "update_quota_usage", "update_api_key_usage",
]


async def _complete_streaming_db(inf, env):
    svc, _job = _make_service(inf, env, proxy=None)
    await svc._do_complete_streaming_db(
        inf._RequestIds(REQUEST_ID, USER_ID, API_KEY_ID), BACKEND_ID,
        "hello", 2, 12, 3, 15, finish_reason="stop",
    )
    return svc


def _completion_failures(env):
    return env.logged("request_completion_db_failed")


class TestCompletionRetriesOnIsolatedSessions:
    @pytest.mark.asyncio
    async def test_deadlock_retries_on_a_fresh_isolated_session(self, inf, env):
        env.crud.failures["update_quota_usage"] = [OperationalError(1213)]
        await _complete_streaming_db(inf, env)

        assert [s.name for s in env.isolated] == ["isolated-1", "isolated-2"]
        first, second = env.isolated
        assert first.ops == _WRITE_OPS[:3] + ["rollback", "close"]
        assert first.commits == 0
        assert second.ops == _WRITE_OPS + ["commit", "close"]
        assert second.commits == 1
        assert env.request_db.ops == [], "isolated path must never roll back self.db"
        env.crud.incr_quota_redis.assert_awaited_once_with(USER_ID, 15)
        env.enqueue_for_dlp.assert_awaited_once_with(REQUEST_ID)
        assert _completion_failures(env) == []

    @pytest.mark.asyncio
    async def test_connection_lost_logs_once_and_enqueues_dlp(self, inf, env):
        env.crud.failures["update_request_completed"] = [OperationalError(2013)]
        await _complete_streaming_db(inf, env)

        failures = _completion_failures(env)
        assert len(failures) == 1
        assert failures[0].kwargs == {
            "request_id": REQUEST_ID,
            "path": "complete_streaming",
            "error": "OperationalError",
            "db_error_code": 2013,
            "attempts": 1,
        }
        env.enqueue_for_dlp.assert_awaited_once_with(REQUEST_ID)
        assert [s.name for s in env.isolated] == ["isolated-1"]
        assert env.isolated[0].ops == ["update_request_completed", "rollback", "close"]
        env.crud.incr_quota_redis.assert_not_awaited()
        assert env.request_db.ops == []

    @pytest.mark.asyncio
    async def test_lock_wait_timeout_retried_once(self, inf, env):
        env.crud.failures["update_api_key_usage"] = [OperationalError(1205), OperationalError(1205)]
        await _complete_streaming_db(inf, env)

        assert len(env.isolated) == 2
        assert all(s.commits == 0 and s.rollbacks == 1 and s.closed for s in env.isolated)
        failures = _completion_failures(env)
        assert len(failures) == 1
        assert failures[0].kwargs["db_error_code"] == 1205
        assert failures[0].kwargs["attempts"] == 2
        env.enqueue_for_dlp.assert_awaited_once_with(REQUEST_ID)
        assert env.request_db.ops == []

    @pytest.mark.asyncio
    async def test_persistent_deadlock_stops_after_five_attempts(self, inf, env):
        env.crud.failures["update_quota_usage"] = [OperationalError(1213) for _ in range(5)]
        await _complete_streaming_db(inf, env)

        assert len(env.isolated) == inf.InferenceService._COMPLETION_DB_ATTEMPTS == 5
        failures = _completion_failures(env)
        assert len(failures) == 1
        assert failures[0].kwargs["attempts"] == 5
        assert failures[0].kwargs["db_error_code"] == 1213
        env.enqueue_for_dlp.assert_awaited_once_with(REQUEST_ID)
        assert env.request_db.ops == []

    @pytest.mark.asyncio
    async def test_cancelled_error_contract(self, inf, env):
        env.crud.failures["create_response"] = [asyncio.CancelledError()]
        with pytest.raises(asyncio.CancelledError):
            await _complete_streaming_db(inf, env)

        failures = _completion_failures(env)
        assert len(failures) == 1
        assert failures[0].kwargs == {
            "request_id": REQUEST_ID,
            "path": "complete_streaming",
            "error": "CancelledError",
            "db_error_code": None,
            "attempts": 1,
        }
        env.enqueue_for_dlp.assert_awaited_once_with(REQUEST_ID)
        assert env.isolated[0].ops == [
            "update_request_completed", "create_response", "rollback", "close",
        ]
        env.crud.incr_quota_redis.assert_not_awaited()
        assert env.request_db.ops == []

    @pytest.mark.asyncio
    async def test_non_streaming_path_keeps_request_session_semantics(self, inf, env):
        env.crud.failures["update_quota_usage"] = [OperationalError(1213)]
        svc, _job = _make_service(inf, env, proxy=None)
        response = {
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3},
        }
        await svc._do_complete_db(env.db_request, BACKEND_ID, response, 12, 3, False, 15)

        assert env.isolated == []
        assert env.request_db.ops == (
            _WRITE_OPS[:3] + ["rollback"] + _WRITE_OPS + ["commit"]
        )
        env.crud.incr_quota_redis.assert_awaited_once_with(USER_ID, 15)
        env.enqueue_for_dlp.assert_awaited_once_with(REQUEST_ID)
        assert _completion_failures(env) == []


# ----------------------------------------------------------------------
# (e) db.session.isolated_async_session
# ----------------------------------------------------------------------

def _use_fake(session_mod, monkeypatch, fake):
    monkeypatch.setattr(session_mod, "AsyncSessionLocal", lambda: fake)


class TestIsolatedAsyncSession:
    @pytest.mark.asyncio
    async def test_success_does_not_commit_and_closes(self, session_mod, monkeypatch):
        fake = FakeSession("iso")
        _use_fake(session_mod, monkeypatch, fake)
        async with session_mod.isolated_async_session() as session:
            assert session is fake
        assert fake.ops == ["close"]
        assert (fake.commits, fake.rollbacks, fake.closed) == (0, 0, True)

    @pytest.mark.asyncio
    async def test_failing_rollback_does_not_mask_original_exception(self, session_mod, monkeypatch):
        fake = FakeSession("iso", fail={"rollback": RuntimeError("rollback failed: gone away")})
        _use_fake(session_mod, monkeypatch, fake)
        with pytest.raises(OperationalError) as info:
            async with session_mod.isolated_async_session():
                raise OperationalError(1213)
        assert info.value.orig.args[0] == 1213
        assert fake.ops == ["rollback", "close"]
        assert fake.closed and not fake.invalidated

    @pytest.mark.asyncio
    async def test_cancelled_body_rolls_back_closes_and_propagates(self, session_mod, monkeypatch):
        fake = FakeSession("iso")
        _use_fake(session_mod, monkeypatch, fake)
        with pytest.raises(asyncio.CancelledError):
            async with session_mod.isolated_async_session():
                raise asyncio.CancelledError()
        assert fake.ops == ["rollback", "close"]
        assert fake.rollbacks == 1 and fake.closed

    @pytest.mark.asyncio
    async def test_close_is_shielded_from_cancellation(self, session_mod, monkeypatch):
        hold = _Hold()
        fake = FakeSession("iso", holds={"close": hold})
        _use_fake(session_mod, monkeypatch, fake)

        async def use():
            async with session_mod.isolated_async_session():
                pass

        task = asyncio.ensure_future(use())
        await _wait(hold.started, "close")
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not fake.closed  # still in flight, detached from the cancelled task
        hold.release.set()
        await _eventually(lambda: fake.closed, "the shielded close")
        assert fake.ops == ["close"]
        assert fake.violations == [] and not fake.invalidated

    @pytest.mark.asyncio
    async def test_cancel_during_cleanup_neither_masks_nor_overlaps(self, session_mod, monkeypatch):
        hold = _Hold()
        fake = FakeSession("iso", holds={"rollback": hold})
        _use_fake(session_mod, monkeypatch, fake)

        async def use():
            async with session_mod.isolated_async_session():
                raise OperationalError(2013)

        task = asyncio.ensure_future(use())
        await _wait(hold.started, "rollback")
        task.cancel()
        with pytest.raises(OperationalError) as info:
            await task
        assert info.value.orig.args[0] == 2013
        hold.release.set()
        await _eventually(lambda: fake.closed, "cleanup to finish")
        assert fake.ops == ["rollback", "close"]
        assert fake.violations == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("body_fails", [False, True])
    async def test_invalidate_attempted_when_close_fails(self, session_mod, monkeypatch, body_fails):
        fake = FakeSession("iso", fail={"close": RuntimeError("close failed")})
        _use_fake(session_mod, monkeypatch, fake)
        if body_fails:
            with pytest.raises(OperationalError):
                async with session_mod.isolated_async_session():
                    raise OperationalError(2014)
            assert fake.ops == ["rollback", "close", "invalidate"]
        else:
            async with session_mod.isolated_async_session():
                pass
            assert fake.ops == ["close", "invalidate"]
        assert fake.invalidated


# ----------------------------------------------------------------------
# (f) LOCK INVARIANT (structural)
# ----------------------------------------------------------------------

_STREAM_ROOTS = ("stream_chat_completion", "stream_ollama_chat", "stream_ollama_generate")
# Run before / as the request row is committed; _create_request_record's commit
# ends their transaction before any streaming starts.
_PRE_COMMIT = frozenset({"_check_quota", "_create_request_record"})
# The detached writers themselves (behaviourally tested above to run on
# isolated sessions when handed request_ids).
_DETACHED_WRITERS = frozenset({"_complete_streaming_request", "_fail_request"})
# crud readers the streaming path may run on self.db after that commit.
_ALLOWED_REQUEST_SESSION_READS = frozenset({"get_user_quota"})
# ...and the one place it ENDS that read transaction, handing the pooled
# connection back before streaming (no locks involved).
_REQUEST_SESSION_RELEASE = "_release_request_session"
_RELEASE_USES = frozenset({"method:commit", "method:rollback"})
_LOCKING_CALLS = frozenset({
    "with_for_update", "update", "insert", "delete", "add", "add_all",
    "merge", "flush", "commit", "text",
})
_LOCKING_SQL = re.compile(r"for\s+update|lock\s+in\s+share\s+mode|for\s+share\b", re.I)


def _service_methods(inference_src):
    for node in ast.parse(inference_src).body:
        if isinstance(node, ast.ClassDef) and node.name == "InferenceService":
            return {
                n.name: n for n in node.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    raise AssertionError("InferenceService not found")


def _is_self_db(node):
    return (
        isinstance(node, ast.Attribute) and node.attr == "db"
        and isinstance(node.value, ast.Name) and node.value.id == "self"
    )


def streaming_reachable_methods(methods):
    """Service methods the streaming generators reach after the row commit.

    Follows ``self.<method>`` references and string constants naming a method
    (``getattr(self, proxy_fn)`` targets are passed as strings)."""
    seen, stack = set(), list(_STREAM_ROOTS)
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        for node in ast.walk(methods[name]):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name) and node.value.id == "self"
            ):
                ref = node.attr
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                ref = node.value
            else:
                continue
            if ref in methods and ref not in _PRE_COMMIT | _DETACHED_WRITERS:
                stack.append(ref)
    return seen


def request_session_uses(methods, names):
    """Every ``self.db`` use in ``names`` as (method, how it is used)."""
    uses = []
    for name in sorted(names):
        fn = methods[name]
        parent_of = {
            child: parent for parent in ast.walk(fn)
            for child in ast.iter_child_nodes(parent)
        }
        for node in ast.walk(fn):
            if not _is_self_db(node):
                continue
            parent = parent_of[node]
            if isinstance(parent, ast.keyword):
                parent = parent_of[parent]
            if isinstance(parent, ast.Call) and parent.func is not node:
                uses.append((name, f"arg:{ast.unparse(parent.func)}"))
            elif isinstance(parent, ast.Attribute):
                uses.append((name, f"method:{parent.attr}"))
            else:
                uses.append((name, f"other:{ast.unparse(parent)}"))
    return uses


def lock_invariant_violations(inference_src, crud_src):
    """Problems with the LOCK INVARIANT, plus what the audit walked."""
    methods = _service_methods(inference_src)
    reachable = streaming_reachable_methods(methods)
    uses = request_session_uses(methods, reachable)
    allowed = {f"arg:crud.{fn}" for fn in _ALLOWED_REQUEST_SESSION_READS}
    problems = [
        f"{name}: self.db {use} on the streaming path after the request row commit"
        for name, use in uses
        if use not in allowed
        and not (name == _REQUEST_SESSION_RELEASE and use in _RELEASE_USES)
    ]
    crud_fns = {
        n.name: n for n in ast.parse(crud_src).body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for fname in sorted(_ALLOWED_REQUEST_SESSION_READS):
        for node in ast.walk(crud_fns[fname]):
            if isinstance(node, ast.Call):
                callee = (
                    node.func.attr if isinstance(node.func, ast.Attribute)
                    else getattr(node.func, "id", None)
                )
                if callee in _LOCKING_CALLS:
                    problems.append(f"crud.{fname} calls {callee}() — no longer a plain read")
            elif (
                isinstance(node, ast.Constant) and isinstance(node.value, str)
                and _LOCKING_SQL.search(node.value)
            ):
                problems.append(f"crud.{fname} contains locking SQL {node.value!r}")
    return problems, reachable, uses


def _calls_named(fn, dotted):
    return [
        n for n in ast.walk(fn)
        if isinstance(n, ast.Call) and ast.unparse(n.func) == dotted
    ]


class TestLockInvariant:
    def test_streaming_path_takes_no_locks_on_request_session(self):
        problems, _reachable, _uses = lock_invariant_violations(
            _INFERENCE_PATH.read_text(), _CRUD_PATH.read_text(),
        )
        assert problems == [], (
            "The detached streaming writes UPDATE requests/quotas/api_keys on an "
            "isolated session; a write or locking read left open on self.db would "
            "deadlock every stream against itself (see _RequestIds):\n"
            + "\n".join(problems)
        )

    def test_audit_is_not_vacuous(self):
        _problems, reachable, uses = lock_invariant_violations(
            _INFERENCE_PATH.read_text(), _CRUD_PATH.read_text(),
        )
        assert {
            "_proxy_stream_with_retry", "_route_request", "_route_request_inner",
            "cap_max_tokens", "_count_input_tokens",
            "_proxy_stream_request", "_proxy_ollama_stream", "_openai_chunk_to_ollama",
            _REQUEST_SESSION_RELEASE,
        } <= reachable
        assert not (_PRE_COMMIT | _DETACHED_WRITERS) & reachable
        assert ("_route_request_inner", "arg:crud.get_user_quota") in uses
        assert (_REQUEST_SESSION_RELEASE, "method:commit") in uses

    def test_request_row_committed_before_the_stream_path(self):
        methods = _service_methods(_INFERENCE_PATH.read_text())
        record = methods["_create_request_record"]
        created = _calls_named(record, "crud.create_request")[0].lineno
        committed = [c.lineno for c in _calls_named(record, "self.db.commit")]
        assert committed and min(committed) > created
        for root in ("stream_chat_completion", "stream_ollama_chat"):
            fn = methods[root]
            assert (
                _calls_named(fn, "self._create_request_record")[0].lineno
                < _calls_named(fn, "self._proxy_stream_with_retry")[0].lineno
            )

    def test_request_session_released_after_every_routing_attempt(self):
        methods = _service_methods(_INFERENCE_PATH.read_text())
        fn = methods["_proxy_stream_with_retry"]
        routes = _calls_named(fn, "self._route_request")
        assert len(routes) == 2  # first attempt + the cleared-exclusions retry
        guarded = set()
        for node in ast.walk(fn):
            if isinstance(node, ast.Try) and any(
                _calls_named(stmt, f"self.{_REQUEST_SESSION_RELEASE}")
                for stmt in node.finalbody
            ):
                for part in node.body + node.handlers:
                    guarded.update(id(call) for call in _calls_named(part, "self._route_request"))
        assert {id(call) for call in routes} <= guarded, (
            "every _route_request call must sit in a try whose finally releases the "
            "request session (success, break and raise paths alike)"
        )
        (backend_loop,) = [n for n in ast.walk(fn) if isinstance(n, ast.AsyncFor)]
        releases = _calls_named(fn, f"self.{_REQUEST_SESSION_RELEASE}")
        assert releases and max(c.lineno for c in releases) < backend_loop.lineno
        release = methods[_REQUEST_SESSION_RELEASE]
        assert _calls_named(release, "self.db.commit")
        # A shielded commit could outlive a cancelled generator and collide
        # with get_async_db's teardown: the release must stay unshielded.
        assert not _calls_named(release, "asyncio.shield")

    def test_generators_capture_scalars_and_hand_them_to_detached_writers(self):
        methods = _service_methods(_INFERENCE_PATH.read_text())
        for root in ("stream_chat_completion", "stream_ollama_chat"):
            fn = methods[root]
            body = fn.body
            created_at = next(
                i for i, stmt in enumerate(body)
                if "self._create_request_record" in ast.unparse(stmt)
            )
            # Captured by the very next statement: no await in between.
            assert ast.unparse(body[created_at + 1]) == "request_ids = _RequestIds.of(db_request)"

            (complete,) = _calls_named(fn, "self._complete_streaming_request")
            assert ast.unparse(complete.args[0]) == "request_ids"
            fails = _calls_named(fn, "self._fail_request")
            assert len(fails) == 2
            for call in fails:
                assert ast.unparse(call.args[0]) == "None"
                kwargs = {k.arg: ast.unparse(k.value) for k in call.keywords}
                assert kwargs.get("request_ids") == "request_ids"

    def test_detached_streaming_writers_never_touch_request_session_or_row(self):
        methods = _service_methods(_INFERENCE_PATH.read_text())
        for name in ("_complete_streaming_request", "_do_complete_streaming_db"):
            nodes = list(ast.walk(methods[name]))
            assert not any(_is_self_db(n) for n in nodes), name
            assert not any(isinstance(n, ast.Name) and n.id == "db_request" for n in nodes), name
        assert _calls_named(methods["_do_complete_streaming_db"], "_isolated_db_session")
