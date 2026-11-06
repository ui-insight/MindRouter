"""OCR load guards and batched key usage (2026-11-05 outage).

What happened: one account's client sent PDFs to /v1/ocr faster than they
could be processed and retried the ones that timed out. Nothing bounded how
many documents the gateway accepted; each converts its pages to images on
the gateway's own CPUs (pdftoppm), so 39 conversions ran at once and load
reached ~90 on 32 cores. Failed chunks retried instantly. Every chunk's
accounting transaction began by X-locking the key's api_keys row, so ~110
transactions queued on that one row, each holding a pooled connection, until
MariaDB refused connections (1040) and every user got 500s.

The guards pinned here:

- ocr_admission.document_slot: per-user and server-wide limits on documents
  in flight (Redis, with a per-worker fallback); over the limit = OcrBusy,
  surfaced as 429 + Retry-After before any page is converted.
- ocr._ConversionLimiter: page conversions per worker are capped.
- ocr_chunk: a 4xx (429 above all) is not retried; other errors back off.
- ocr._run_chunks: one failed chunk cancels its siblings; a client that
  disconnected stops the document (OcrCancelled).
- validate_client_params: caller-supplied dpi/chunk_size/overlap bounded.
- /v1/ocr and /v1/ocrmd: 429 mapping, 400 for bad params, and the request's
  DB session is released before the (minutes-long) OCR work starts.
- key_usage: usage counted in memory and written in batches, one short
  statement per key with a bounded lock wait.

ocr.py is spec-loaded with the backend.app.* chain stubbed, as in
test_ocr_pipeline_json_and_runaway.py, so its own logic runs for real.
"""

import asyncio
import importlib.util
import os
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]

# kwargs each stubbed InferenceService.chat_completion call received
calls_kw: list = []


# --------------------------------------------------------------------------
# ocr.py under test, with its import chain stubbed
# --------------------------------------------------------------------------


class _Session:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def rollback(self):
        return None


class _HTTPError(Exception):
    def __init__(self, status_code, detail="x"):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _load_ocr(chat_completion=None, admission=None):
    saved = {k: v for k, v in sys.modules.items() if k.startswith("backend")}
    for pkg in ("backend", "backend.app", "backend.app.core", "backend.app.core.translators",
                "backend.app.core.telemetry", "backend.app.db", "backend.app.services",
                "backend.app.dashboard", "backend.app.storage"):
        mod = ModuleType(pkg); mod.__path__ = []; sys.modules[pkg] = mod
    canon = ModuleType("backend.app.core.canonical_schemas")
    canon.CanonicalChatRequest = lambda **kw: SimpleNamespace(**kw)
    canon.CanonicalMessage = lambda **kw: SimpleNamespace(**kw)
    canon.ImageUrlContent = lambda **kw: SimpleNamespace(**kw)
    canon.TextContent = lambda **kw: SimpleNamespace(**kw)
    canon.CanonicalModelInfo = MagicMock()
    sys.modules["backend.app.core.canonical_schemas"] = canon
    sys.modules["backend.app.db.crud"] = ModuleType("backend.app.db.crud")
    lc = ModuleType("backend.app.logging_config"); lc.get_logger = lambda *a, **k: MagicMock()
    sys.modules["backend.app.logging_config"] = lc
    st = ModuleType("backend.app.settings")
    st.get_settings = lambda: SimpleNamespace(ocr_max_frames=50, chat_upload_max_uncompressed_mb=100, app_version="t")
    sys.modules["backend.app.settings"] = st
    sess = ModuleType("backend.app.db.session"); sess.AsyncSessionLocal = _Session
    sys.modules["backend.app.db.session"] = sess
    inf = ModuleType("backend.app.services.inference")

    class InferenceService:
        def __init__(self, db):
            self.db = db

        async def chat_completion(self, canonical, user, api_key, http_request, **kw):
            calls_kw.append(kw)
            return await chat_completion(canonical)

    inf.InferenceService = InferenceService
    sys.modules["backend.app.services.inference"] = inf
    if admission is not None:
        sys.modules["backend.app.services.ocr_admission"] = admission
    spec = importlib.util.spec_from_file_location("ocr_under_test", REPO_ROOT / "backend/app/services/ocr.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod, saved


def _restore(saved):
    for k in [k for k in sys.modules if k.startswith("backend")]:
        del sys.modules[k]
    sys.modules.update(saved)


@pytest.fixture
def ocr():
    mod, saved = _load_ocr(lambda c: None)
    try:
        yield mod
    finally:
        _restore(saved)


CFG = {"temperature": 0.1, "max_tokens": 16384, "max_tokens_per_page": 4096,
       "max_retries": 2, "min_chars_per_page": 0, "max_pages": 200,
       "max_concurrent_chunks": 4, "prompt_ocrmd": "Transcribe {num_pages} pages."}

OK_RESULT = {"choices": [{"message": {"content": "text"}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


# --------------------------------------------------------------------------
# ocr_chunk retry policy
# --------------------------------------------------------------------------


async def _run_chunk(ocr_mod, monkeypatch, http_request=None):
    sleeps = []

    async def fake_sleep(s):
        sleeps.append(s)

    monkeypatch.setattr(ocr_mod.asyncio, "sleep", fake_sleep)
    result = await ocr_mod.ocr_chunk(
        [b"img"], 0, 0, 1, 1, 1, CFG["prompt_ocrmd"], "m", dict(CFG), None, None, http_request,
    )
    return result, sleeps


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [429, 403, 400])
async def test_a_refused_chunk_is_not_retried(monkeypatch, code):
    calls = []

    async def refuse(canonical):
        calls.append(1)
        raise _HTTPError(code, "Rate limit exceeded: 200 requests per minute")

    mod, saved = _load_ocr(refuse)
    try:
        with pytest.raises(_HTTPError):
            await _run_chunk(mod, monkeypatch)
    finally:
        _restore(saved)
    assert len(calls) == 1, "a 4xx is the same answer next time; retrying it only adds load"


@pytest.mark.asyncio
async def test_a_server_error_is_retried_after_a_backoff(monkeypatch):
    calls = []

    async def flaky(canonical):
        calls.append(1)
        if len(calls) < 3:
            raise _HTTPError(503, "busy")
        return OK_RESULT

    mod, saved = _load_ocr(flaky)
    try:
        (_, text, _, _), sleeps = await _run_chunk(mod, monkeypatch)
    finally:
        _restore(saved)
    assert text == "text" and len(calls) == 3
    assert sleeps == [mod.CHUNK_RETRY_BACKOFF_SECONDS, mod.CHUNK_RETRY_BACKOFF_SECONDS * 2]


@pytest.mark.asyncio
async def test_chunks_of_an_admitted_document_skip_the_per_minute_check(monkeypatch):
    calls_kw.clear()

    async def ok(canonical):
        return OK_RESULT

    mod, saved = _load_ocr(ok)
    try:
        await _run_chunk(mod, monkeypatch)
    finally:
        _restore(saved)
    assert calls_kw == [{"skip_rpm_check": True}]


@pytest.mark.asyncio
async def test_skip_rpm_check_keeps_the_token_quota_but_skips_the_rpm_counter(monkeypatch):
    from backend.app.services import inference as inf

    rpm = AsyncMock(return_value=(True, 1))
    monkeypatch.setattr("backend.app.core.redis_client.check_rpm", rpm)
    monkeypatch.setattr(inf.crud, "reset_quota_if_needed", AsyncMock())
    quota = SimpleNamespace(tokens_used=10, rpm_limit=200)
    monkeypatch.setattr(inf.crud, "get_user_quota", AsyncMock(return_value=quota))
    monkeypatch.setattr(inf, "effective_token_budget", lambda u, q: 100)
    svc = inf.InferenceService.__new__(inf.InferenceService)
    svc.db = None
    user, key = SimpleNamespace(id=5), SimpleNamespace(rpm_limit=None)

    await svc._check_quota(user, key, check_rpm=False)
    rpm.assert_not_awaited()
    await svc._check_quota(user, key)
    rpm.assert_awaited_once()

    quota.tokens_used = 100  # over budget: refused even without the RPM check
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        await svc._check_quota(user, key, check_rpm=False)
    assert exc.value.detail == "Token quota exceeded"


def test_chat_completion_passes_skip_rpm_check_through():
    src = (REPO_ROOT / "backend/app/services/inference.py").read_text()
    assert "await self._check_quota(user, api_key, check_rpm=not skip_rpm_check)" in src


@pytest.mark.asyncio
async def test_no_retry_once_the_client_has_gone(monkeypatch):
    calls = []

    async def failing(canonical):
        calls.append(1)
        raise RuntimeError("backend timeout")

    gone = SimpleNamespace(is_disconnected=AsyncMock(return_value=True))
    mod, saved = _load_ocr(failing)
    try:
        with pytest.raises(mod.OcrCancelled):
            await _run_chunk(mod, monkeypatch, http_request=gone)
    finally:
        _restore(saved)
    assert len(calls) == 1


def test_retryable_classification(ocr):
    assert ocr._is_retryable(RuntimeError("timeout")) is True
    assert ocr._is_retryable(_HTTPError(500)) is True
    assert ocr._is_retryable(_HTTPError(429)) is False
    assert ocr._is_retryable(_HTTPError(404)) is False


# --------------------------------------------------------------------------
# One failure stops the document; a gone client stops the document
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_failed_chunk_stops_the_rest_without_cancelling_in_flight_work(ocr):
    """Stopping is cooperative: chunks not yet started never start, and a
    chunk already in flight (possibly inside its accounting write on its own
    DB session) is allowed to finish rather than cancelled mid-write."""
    stop = asyncio.Event()
    finished, cancelled, skipped = [], [], []
    gate = asyncio.Event()

    async def in_flight():
        try:
            await gate.wait()
            finished.append("in_flight")
            return 1
        except asyncio.CancelledError:
            cancelled.append("in_flight")
            raise

    async def boom():
        await asyncio.sleep(0)
        raise _HTTPError(429)

    async def later():
        await asyncio.sleep(0.01)
        if await ocr._should_stop(stop, None):
            skipped.append("later")
            raise ocr.OcrCancelled("document stopped")
        return 2

    run = asyncio.create_task(ocr._run_chunks([in_flight(), boom(), later()], stop))
    await asyncio.sleep(0.05)
    assert stop.is_set() and skipped == ["later"]
    assert not run.done(), "returned before the in-flight chunk finished"
    gate.set()
    with pytest.raises(_HTTPError):
        await run
    assert finished == ["in_flight"] and cancelled == []


@pytest.mark.asyncio
async def test_the_real_error_wins_over_the_cancellations_it_caused(ocr):
    stop = asyncio.Event()

    async def cancelled_first():
        raise ocr.OcrCancelled("document stopped")

    async def real():
        await asyncio.sleep(0.01)
        raise _HTTPError(429)

    with pytest.raises(_HTTPError):
        await ocr._run_chunks([cancelled_first(), real()], stop)


@pytest.mark.asyncio
async def test_a_cancelled_request_waits_for_its_chunks_to_wind_down(ocr):
    stop = asyncio.Event()
    finished = []

    async def chunk():
        await asyncio.sleep(0.02)
        finished.append(1)

    run = asyncio.create_task(ocr._run_chunks([chunk(), chunk()], stop))
    await asyncio.sleep(0)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run
    assert stop.is_set() and finished == [1, 1]


@pytest.mark.asyncio
async def test_a_disconnected_client_stops_the_document_before_converting(ocr):
    converted = []

    async def fake_range(pdf_bytes, first_page, last_page, dpi=200):
        converted.append(first_page)
        return [b"img"] * (last_page - first_page + 1)

    ocr._pdf_page_count = AsyncMock(return_value=12)
    ocr._pdf_to_images_range = fake_range
    gone = SimpleNamespace(is_disconnected=AsyncMock(return_value=True))
    with pytest.raises(ocr.OcrCancelled):
        await ocr.perform_ocr(
            file_bytes=b"%PDF", content_type="application/pdf", filename="a.pdf",
            model="m", output_format="markdown", chunk_size=6, overlap=0, dpi=200,
            ocr_config=dict(CFG), user=None, api_key=None, http_request=gone,
        )
    assert converted == []


# --------------------------------------------------------------------------
# Conversions per worker are capped
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_conversion_limiter_caps_concurrency(ocr):
    limiter = ocr._ConversionLimiter(2)
    peak, running = 0, 0

    async def work():
        nonlocal peak, running
        async with limiter:
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0.01)
            running -= 1

    await asyncio.gather(*(work() for _ in range(8)))
    assert peak == 2 and limiter.active == 0


@pytest.mark.asyncio
async def test_raising_the_conversion_limit_wakes_waiters(ocr):
    limiter = ocr._ConversionLimiter(1)
    release = asyncio.Event()
    entered = []

    async def hold():
        async with limiter:
            entered.append("hold")
            await release.wait()

    async def second():
        async with limiter:
            entered.append("second")

    t1 = asyncio.create_task(hold())
    await asyncio.sleep(0)
    t2 = asyncio.create_task(second())
    await asyncio.sleep(0.01)
    assert entered == ["hold"]
    limiter.set_limit(2)
    await asyncio.sleep(0.01)
    assert entered == ["hold", "second"]
    release.set()
    await asyncio.gather(t1, t2)


@pytest.mark.asyncio
async def test_every_conversion_goes_through_the_limited_thread_helper(ocr, monkeypatch):
    calls = []

    async def spy(fn, *args):
        calls.append(fn.__name__)
        return [b"p"]

    monkeypatch.setattr(ocr, "_convert_in_thread", spy)
    await ocr._pdf_to_images_range(b"%PDF", 1, 2)
    await ocr._pdf_to_images(b"%PDF")
    await ocr.document_to_images(b"img", "image/png", "a.png")
    assert calls == ["_convert", "_convert", "_image_bytes_to_pages"]


@pytest.mark.asyncio
async def test_the_limit_is_held_until_the_thread_finishes_even_if_cancelled(ocr, monkeypatch):
    """Cancelling the task cannot stop pdftoppm; the slot must stay taken
    until the thread is really done, or the limit counts tasks not processes."""
    import threading

    limiter = ocr._ConversionLimiter(1)
    monkeypatch.setattr(ocr, "_conversions", limiter)
    release = threading.Event()

    def blocking():
        release.wait(2)
        return "done"

    task = asyncio.create_task(ocr._convert_in_thread(blocking))
    await asyncio.sleep(0.02)
    task.cancel()
    await asyncio.sleep(0.02)
    assert limiter.active == 1, "limit released while the thread still runs"
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert limiter.active == 0


@pytest.mark.asyncio
async def test_a_cancelled_waiter_does_not_strand_the_others(ocr):
    """notify() woke one waiter; if that one was cancelled the wake-up was
    lost and the next conversion hung with nothing running (review repro)."""
    limiter = ocr._ConversionLimiter(1)
    release = asyncio.Event()
    got = []

    async def holder():
        async with limiter:
            await release.wait()

    async def waiter(name):
        async with limiter:
            got.append(name)

    h = asyncio.create_task(holder())
    await asyncio.sleep(0)
    w1 = asyncio.create_task(waiter("w1"))
    w2 = asyncio.create_task(waiter("w2"))
    await asyncio.sleep(0.01)
    release.set()
    await asyncio.sleep(0)  # the holder exits and wakes w1 ...
    w1.cancel()             # ... which is cancelled before it can run
    await asyncio.wait_for(asyncio.gather(h, w2), timeout=1)
    assert got == ["w2"] and limiter.active == 0


@pytest.mark.asyncio
async def test_perform_ocr_applies_the_configured_conversion_limit(ocr, monkeypatch):
    limits = []
    monkeypatch.setattr(ocr._conversions, "set_limit", limits.append)
    ocr._perform_ocr_admitted = AsyncMock(return_value={})
    await ocr.perform_ocr(
        file_bytes=b"x", content_type="image/png", filename="a.png", model="m",
        output_format="markdown", chunk_size=6, overlap=0, dpi=200,
        ocr_config=dict(CFG, max_concurrent_conversions=6), user=None, api_key=None,
        http_request=None,
    )
    assert limits == [6]


# --------------------------------------------------------------------------
# Admission: perform_ocr holds a document slot with the configured limits
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_perform_ocr_runs_inside_a_document_slot():
    seen = {}

    class _Slot:
        def __init__(self, user_id, per_user, global_limit, **kw):
            seen.update(user_id=user_id, per_user=per_user, global_limit=global_limit)
            seen["wait"] = (kw["max_wait_seconds"], kw["max_waiting_per_user"], kw["max_waiting"])
            seen["has_abort"] = callable(kw["should_abort"])

        async def __aenter__(self):
            seen["entered"] = True

        async def __aexit__(self, *exc):
            seen["exited"] = True

    admission = ModuleType("backend.app.services.ocr_admission")
    admission.document_slot = _Slot
    admission.WaitAbandoned = type("WaitAbandoned", (Exception,), {})
    mod, saved = _load_ocr(lambda c: None, admission=admission)
    try:
        mod._perform_ocr_admitted = AsyncMock(return_value={"content": "ok"})
        cfg = dict(CFG, max_documents_per_user=3, max_documents=11,
                   queue_wait_seconds=45, max_waiting_per_user=5, max_waiting=20)
        out = await mod.perform_ocr(
            file_bytes=b"x", content_type="image/png", filename="a.png", model="m",
            output_format="markdown", chunk_size=6, overlap=0, dpi=200, ocr_config=cfg,
            user=SimpleNamespace(id=42), api_key=None, http_request=None,
        )
    finally:
        _restore(saved)
    assert out == {"content": "ok"}
    assert seen == {"user_id": 42, "per_user": 3, "global_limit": 11,
                    "wait": (45, 5, 20), "has_abort": True,
                    "entered": True, "exited": True}


@pytest.mark.asyncio
async def test_a_client_leaving_while_queued_is_ocr_cancelled():
    admission = ModuleType("backend.app.services.ocr_admission")
    admission.WaitAbandoned = type("WaitAbandoned", (Exception,), {})

    class _Slot:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            raise admission.WaitAbandoned()

        async def __aexit__(self, *exc):
            return False

    admission.document_slot = _Slot
    mod, saved = _load_ocr(lambda c: None, admission=admission)
    try:
        with pytest.raises(mod.OcrCancelled):
            await mod.perform_ocr(
                file_bytes=b"x", content_type="image/png", filename="a.png", model="m",
                output_format="markdown", chunk_size=6, overlap=0, dpi=200,
                ocr_config=dict(CFG), user=SimpleNamespace(id=1), api_key=None,
                http_request=None,
            )
    finally:
        _restore(saved)


# --------------------------------------------------------------------------
# Caller-supplied parameters
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kw", [
    {"dpi": 1200}, {"dpi": 10}, {"chunk_size": 200}, {"chunk_size": 0}, {"overlap": 50},
])
def test_out_of_range_client_params_are_rejected(ocr, kw):
    with pytest.raises(ValueError):
        ocr.validate_client_params(**kw)


def test_in_range_and_unsent_client_params_pass(ocr):
    ocr.validate_client_params(dpi=200, chunk_size=6, overlap=2)
    ocr.validate_client_params()


# --------------------------------------------------------------------------
# ocr_admission.document_slot
# --------------------------------------------------------------------------


class _FakeSlots:
    """In-memory stand-in for the redis_client slot functions, with the Lua
    scripts' semantics (the scripts themselves run in the real-Redis test)."""

    def __init__(self):
        self.sets = {}
        self.down = False
        self.renewed = 0

    def _set(self, key):
        return self.sets.setdefault(key, {})

    def _prune(self, key, before):
        self.sets[key] = {t: sc for t, sc in self._set(key).items() if sc > before}

    def size(self, key):
        return len(self._set(key))

    async def acquire_slot(self, uk, gk, token, ul, gl, ttl):
        if self.down:
            return None
        now = time.time()
        self._prune(uk, now); self._prune(gk, now)
        if ul > 0 and self.size(uk) >= ul:
            return 1
        if gl > 0 and self.size(gk) >= gl:
            return 2
        self._set(uk)[token] = now + ttl; self._set(gk)[token] = now + ttl
        return 0

    async def enqueue_waiter(self, uw, gw, token, ul, gl, stale):
        if self.down:
            return None
        now = time.time()
        self._prune(uw, now - stale); self._prune(gw, now - stale)
        if ul > 0 and self.size(uw) >= ul:
            return 1
        if gl > 0 and self.size(gw) >= gl:
            return 2
        self._set(uw)[token] = now; self._set(gw)[token] = now
        return 0

    async def acquire_queued_slot(self, uk, gk, uw, gw, token, ul, gl, ttl, stale):
        if self.down:
            return None
        now = time.time()
        self._prune(uk, now); self._prune(gk, now)
        self._prune(uw, now - stale); self._prune(gw, now - stale)
        line = self._set(uw)
        if token not in line:
            return 3
        if min(line.items(), key=lambda kv: (kv[1], kv[0]))[0] != token:
            return 4
        if ul > 0 and self.size(uk) >= ul:
            return 1
        if gl > 0 and self.size(gk) >= gl:
            return 2
        self._set(uk)[token] = now + ttl; self._set(gk)[token] = now + ttl
        self._set(uw).pop(token, None); self._set(gw).pop(token, None)
        return 0

    async def release_slot(self, keys, token):
        if self.down:
            return
        for k in keys:
            self._set(k).pop(token, None)

    async def renew_slot(self, keys, token, ttl):
        for k in keys:
            if token in self._set(k):
                self._set(k)[token] = time.time() + ttl
                self.renewed += 1


@pytest.fixture
def admission(monkeypatch):
    import backend.app.services.ocr_admission as adm

    fake = _FakeSlots()
    for name in ("acquire_slot", "enqueue_waiter", "acquire_queued_slot",
                 "release_slot", "renew_slot"):
        monkeypatch.setattr(adm.redis_client, name, getattr(fake, name))
    monkeypatch.setattr(adm, "POLL_SECONDS", 0.01)
    monkeypatch.setattr(adm, "_local_users", adm.Counter())
    monkeypatch.setattr(adm, "_local_total", 0)
    monkeypatch.setattr(adm, "_local_waiting_users", adm.Counter())
    monkeypatch.setattr(adm, "_local_waiting_total", 0)
    return adm, fake


U7, ALL, W7, WALL = "ocr:docs:user:7", "ocr:docs:all", "ocr:wait:user:7", "ocr:wait:all"


async def _hold(adm, user_id, release: asyncio.Event, entered: list, name, **kw):
    async with adm.document_slot(user_id, **kw):
        entered.append(name)
        await release.wait()


@pytest.mark.asyncio
async def test_slot_is_taken_and_released(admission):
    adm, fake = admission
    async with adm.document_slot(7, per_user=2, global_limit=8):
        assert fake.size(U7) == 1 and fake.size(ALL) == 1
        assert fake.size(W7) == 0, "a document that got its slot is no longer waiting"
    assert fake.size(U7) == 0 and fake.size(ALL) == 0


@pytest.mark.asyncio
async def test_slot_is_released_when_the_work_fails(admission):
    adm, fake = admission
    with pytest.raises(RuntimeError):
        async with adm.document_slot(7):
            raise RuntimeError("boom")
    assert fake.size(U7) == 0 and fake.size(ALL) == 0


@pytest.mark.asyncio
async def test_over_the_limit_waits_then_proceeds(admission):
    adm, fake = admission
    release, entered = asyncio.Event(), []
    first = asyncio.create_task(_hold(adm, 7, release, entered, "first", per_user=1))
    await asyncio.sleep(0.02)
    second = asyncio.create_task(_hold(adm, 7, asyncio.Event(), entered, "second", per_user=1))
    await asyncio.sleep(0.05)
    assert entered == ["first"] and fake.size(W7) == 1
    release.set()
    await asyncio.sleep(0.05)
    assert entered == ["first", "second"] and fake.size(W7) == 0
    second.cancel()
    await asyncio.gather(first, second, return_exceptions=True)
    assert fake.size(U7) == 0


@pytest.mark.asyncio
async def test_a_waiter_gives_up_after_max_wait_with_a_429(admission):
    adm, fake = admission
    release, entered = asyncio.Event(), []
    holder = asyncio.create_task(_hold(adm, 7, release, entered, "h", per_user=1))
    await asyncio.sleep(0.02)
    ran = False
    with pytest.raises(adm.OcrBusy) as exc:
        async with adm.document_slot(7, per_user=1, max_wait_seconds=0.05):
            ran = True
    assert not ran and exc.value.scope == "user" and exc.value.waited
    assert fake.size(W7) == 0, "a waiter that gave up left its place in line"
    release.set(); await holder


@pytest.mark.asyncio
async def test_an_accounts_documents_start_in_arrival_order(admission):
    adm, _ = admission
    release, entered = asyncio.Event(), []
    holder = asyncio.create_task(_hold(adm, 7, release, entered, "h", per_user=1))
    await asyncio.sleep(0.02)
    gates = {n: asyncio.Event() for n in ("a", "b", "c")}
    waiters = []
    for n in ("a", "b", "c"):
        waiters.append(asyncio.create_task(_hold(adm, 7, gates[n], entered, n, per_user=1)))
        await asyncio.sleep(0.02)
    release.set()
    for n in ("a", "b", "c"):
        await asyncio.sleep(0.05)
        gates[n].set()
    await asyncio.gather(holder, *waiters)
    assert entered == ["h", "a", "b", "c"]


@pytest.mark.asyncio
async def test_a_full_wait_line_refuses_at_once(admission):
    adm, fake = admission
    release, entered = asyncio.Event(), []
    holder = asyncio.create_task(_hold(adm, 7, release, entered, "h", per_user=1))
    await asyncio.sleep(0.02)
    waiter = asyncio.create_task(_hold(adm, 7, asyncio.Event(), entered, "w",
                                       per_user=1, max_waiting_per_user=1))
    await asyncio.sleep(0.02)
    with pytest.raises(adm.OcrBusy) as exc:
        async with adm.document_slot(7, per_user=1, max_waiting_per_user=1):
            pass
    assert exc.value.scope == "user_waiting" and not exc.value.waited
    waiter.cancel(); release.set()
    await asyncio.gather(holder, waiter, return_exceptions=True)
    assert fake.size(W7) == 0 and fake.size(U7) == 0


@pytest.mark.asyncio
async def test_the_server_wide_wait_line_is_bounded_too(admission):
    adm, _ = admission
    release, entered = asyncio.Event(), []
    holder = asyncio.create_task(_hold(adm, 1, release, entered, "h", global_limit=1))
    await asyncio.sleep(0.02)
    waiter = asyncio.create_task(_hold(adm, 2, asyncio.Event(), entered, "w",
                                       global_limit=1, max_waiting=1))
    await asyncio.sleep(0.02)
    with pytest.raises(adm.OcrBusy) as exc:
        async with adm.document_slot(3, global_limit=1, max_waiting=1):
            pass
    assert exc.value.scope == "global_waiting"
    waiter.cancel(); release.set()
    await asyncio.gather(holder, waiter, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("verdict_owner,scope,limit", [(7, "user", 1), (8, "global", 1)])
async def test_zero_wait_refuses_at_once(admission, verdict_owner, scope, limit):
    adm, _ = admission
    release, entered = asyncio.Event(), []
    holder = asyncio.create_task(_hold(adm, verdict_owner, release, entered, "h",
                                       per_user=1, global_limit=1))
    await asyncio.sleep(0.02)
    with pytest.raises(adm.OcrBusy) as exc:
        async with adm.document_slot(7, per_user=1, global_limit=1, max_wait_seconds=0):
            pass
    assert exc.value.scope == scope and exc.value.limit == limit and not exc.value.waited
    release.set(); await holder


@pytest.mark.asyncio
async def test_a_client_that_leaves_while_waiting_leaves_the_line(admission):
    adm, fake = admission
    release, entered = asyncio.Event(), []
    holder = asyncio.create_task(_hold(adm, 7, release, entered, "h", per_user=1))
    await asyncio.sleep(0.02)
    gone = AsyncMock(side_effect=[False, False, True])
    with pytest.raises(adm.WaitAbandoned):
        async with adm.document_slot(7, per_user=1, should_abort=gone):
            pass
    assert fake.size(W7) == 0 and fake.size(WALL) == 0
    release.set(); await holder


@pytest.mark.asyncio
async def test_cancelled_while_waiting_leaves_the_line_and_holds_nothing(admission):
    adm, fake = admission
    release, entered = asyncio.Event(), []
    holder = asyncio.create_task(_hold(adm, 7, release, entered, "h", per_user=1))
    await asyncio.sleep(0.02)
    waiter = asyncio.create_task(_hold(adm, 7, asyncio.Event(), entered, "w", per_user=1))
    await asyncio.sleep(0.03)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert fake.size(W7) == 0 and fake.size(U7) == 1  # only the holder
    release.set(); await holder
    assert fake.size(U7) == 0


@pytest.mark.asyncio
async def test_a_long_document_keeps_renewing_its_slot(admission, monkeypatch):
    adm, fake = admission
    monkeypatch.setattr(adm, "HEARTBEAT_SECONDS", 0.01)
    async with adm.document_slot(7):
        await asyncio.sleep(0.05)
    assert fake.renewed >= 2
    after = fake.renewed
    await asyncio.sleep(0.03)
    assert fake.renewed == after, "heartbeat outlived the slot"


@pytest.mark.asyncio
async def test_without_redis_the_same_limits_and_wait_apply_per_worker(admission):
    adm, fake = admission
    fake.down = True
    release, entered = asyncio.Event(), []
    holder = asyncio.create_task(_hold(adm, 7, release, entered, "h", per_user=1))
    await asyncio.sleep(0.02)
    # waits, then gives up
    with pytest.raises(adm.OcrBusy) as exc:
        async with adm.document_slot(7, per_user=1, max_wait_seconds=0.05):
            pass
    assert exc.value.scope == "user" and exc.value.waited
    # the local wait line is bounded
    w = asyncio.create_task(_hold(adm, 7, asyncio.Event(), entered, "w",
                                  per_user=1, max_waiting_per_user=1))
    await asyncio.sleep(0.02)
    with pytest.raises(adm.OcrBusy) as exc:
        async with adm.document_slot(7, per_user=1, max_waiting_per_user=1):
            pass
    assert exc.value.scope == "user_waiting"
    # and a waiter gets the slot when it frees
    release.set()
    await asyncio.sleep(0.05)
    assert entered == ["h", "w"]
    w.cancel()
    await asyncio.gather(holder, w, return_exceptions=True)
    assert adm._local_total == 0 and not adm._local_users
    assert adm._local_waiting_total == 0 and not adm._local_waiting_users


@pytest.mark.asyncio
async def test_cancelled_while_acquiring_drops_the_slot(admission, monkeypatch):
    adm, fake = admission
    seen = []

    async def slow(*a):
        await asyncio.sleep(1)

    async def release(keys, token):
        seen.append(tuple(keys))

    monkeypatch.setattr(adm.redis_client, "acquire_slot", slow)
    monkeypatch.setattr(adm.redis_client, "release_slot", release)

    async def use():
        async with adm.document_slot(7, max_wait_seconds=0):
            pass

    t = asyncio.create_task(use())
    await asyncio.sleep(0.01)
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t
    assert (U7, ALL) in seen


def test_busy_messages_say_what_to_do(admission):
    adm, _ = admission
    assert "limit 2 at a time" in str(adm.OcrBusy("user", 2))
    assert "Retry in" in str(adm.OcrBusy("global", 8))
    assert "already waiting" in str(adm.OcrBusy("user_waiting", 4))
    assert "Waited" in str(adm.OcrBusy("user", 2, waited=True))


@pytest.mark.asyncio
async def test_acquire_slot_without_redis_returns_none(monkeypatch):
    from backend.app.core import redis_client

    monkeypatch.setattr(redis_client, "_available", False)
    assert await redis_client.acquire_slot("u", "g", "t", 1, 1, 60) is None


@pytest.mark.asyncio
@pytest.mark.skipif(not os.environ.get("MINDROUTER_TEST_REDIS_URL"),
                    reason="set MINDROUTER_TEST_REDIS_URL to run the Lua script against a real Redis")
async def test_slot_script_against_real_redis(monkeypatch):
    """The Lua script itself: per-user and global limits, expiry, release."""
    import redis.asyncio as aioredis

    from backend.app.core import redis_client as rc

    client = aioredis.from_url(os.environ["MINDROUTER_TEST_REDIS_URL"], decode_responses=True)
    monkeypatch.setattr(rc, "_redis", client)
    monkeypatch.setattr(rc, "_available", True)
    u1, u2, g = "test:ocr:u1", "test:ocr:u2", "test:ocr:all"
    await client.delete(u1, u2, g)
    try:
        assert await rc.acquire_slot(u1, g, "a", 2, 3, 60) == rc.SLOT_OK
        assert await rc.acquire_slot(u1, g, "b", 2, 3, 60) == rc.SLOT_OK
        assert await rc.acquire_slot(u1, g, "c", 2, 3, 60) == rc.SLOT_USER_FULL
        assert await rc.acquire_slot(u2, g, "d", 2, 3, 60) == rc.SLOT_OK
        assert await rc.acquire_slot(u2, g, "e", 2, 3, 60) == rc.SLOT_GLOBAL_FULL
        assert await client.zcard(g) == 3, "a refused slot must not be recorded"
        await rc.release_slot((u1, g), "a")
        assert await rc.acquire_slot(u2, g, "e", 2, 3, 60) == rc.SLOT_OK
        # An expired holder (a dead worker) frees its slot.
        await client.zadd(u1, {"b": 1}); await client.zadd(g, {"b": 1})
        assert await rc.acquire_slot(u1, g, "f", 2, 3, 60) == rc.SLOT_OK
        # Renewal never resurrects a released slot.
        await rc.release_slot((u1, g), "f")
        await rc.renew_slot((u1, g), "f", 60)
        assert await client.zscore(u1, "f") is None
        # 0 = unlimited
        assert await rc.acquire_slot(u1, g, "g", 0, 0, 60) == rc.SLOT_OK

        # Wait line: bounded, oldest-first within a user, leaves on success.
        await client.delete(u1, u2, g)
        uw, gw = "test:ocr:wait:u1", "test:ocr:wait:all"
        await client.delete(uw, gw)
        assert await rc.acquire_slot(u1, g, "h", 1, 0, 60) == rc.SLOT_OK
        assert await rc.enqueue_waiter(uw, gw, "w1", 2, 0, 120) == rc.SLOT_OK
        await asyncio.sleep(0.01)
        assert await rc.enqueue_waiter(uw, gw, "w2", 2, 0, 120) == rc.SLOT_OK
        assert await rc.enqueue_waiter(uw, gw, "w3", 2, 0, 120) == rc.SLOT_USER_FULL
        assert await rc.acquire_queued_slot(u1, g, uw, gw, "w2", 1, 0, 60, 120) == rc.SLOT_NOT_YOUR_TURN
        assert await rc.acquire_queued_slot(u1, g, uw, gw, "w1", 1, 0, 60, 120) == rc.SLOT_USER_FULL
        await rc.release_slot((u1, g), "h")
        assert await rc.acquire_queued_slot(u1, g, uw, gw, "w2", 1, 0, 60, 120) == rc.SLOT_NOT_YOUR_TURN
        assert await rc.acquire_queued_slot(u1, g, uw, gw, "w1", 1, 0, 60, 120) == rc.SLOT_OK
        assert await client.zscore(uw, "w1") is None and await client.zscore(gw, "w1") is None
        assert await client.zscore(u1, "w1") is not None
        # A stale waiter (dead worker) is dropped and its line moves on.
        await client.zadd(uw, {"dead": 1}); await client.zadd(gw, {"dead": 1})
        await rc.release_slot((u1, g), "w1")
        assert await rc.acquire_queued_slot(u1, g, uw, gw, "w2", 1, 0, 60, 120) == rc.SLOT_OK
        assert await rc.acquire_queued_slot(u1, g, uw, gw, "gone", 1, 0, 60, 120) == rc.SLOT_NOT_QUEUED
        # The server-wide line is bounded too.
        await client.delete(uw, gw)
        assert await rc.enqueue_waiter(uw, gw, "x", 0, 1, 120) == rc.SLOT_OK
        assert await rc.enqueue_waiter("test:ocr:wait:u2", gw, "y", 0, 1, 120) == rc.SLOT_GLOBAL_FULL
    finally:
        await client.delete(u1, u2, g, "test:ocr:wait:u1", "test:ocr:wait:u2", "test:ocr:wait:all")
        await client.aclose()


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------


@pytest.fixture
def api():
    import backend.app.api.v1_openai as api_mod
    return api_mod


class _Upload:
    def __init__(self, filename="scan.png", content_type="image/png", data=b"\x89PNG\r\n"):
        self.filename, self.content_type, self._data = filename, content_type, data

    async def read(self):
        return self._data


_CONFIG = {"enabled": True, "model": "dots.MOCR", "chunk_size": 4, "overlap": 1,
           "dpi": 150, "max_file_size_mb": 10, "prompt_ocr": "p", "prompt_ocrmd": "q"}


async def _call(api, handler, perform, **form):
    registry = SimpleNamespace(resolve_alias=lambda m: (m, None))
    form = {"model": None, "chunk_size": None, "overlap": None, "dpi": None, **form}
    db = SimpleNamespace(commit=AsyncMock())
    with (
        patch("backend.app.services.ocr.get_ocr_config", AsyncMock(return_value=dict(_CONFIG))),
        patch("backend.app.services.ocr.perform_ocr", perform),
        patch.object(api, "get_registry", lambda: registry),
        patch.object(api, "model_availability", AsyncMock(return_value=api.AVAILABLE)),
        patch.object(api, "bind_request_context"),
    ):
        return await handler(request=SimpleNamespace(), file=_Upload(), db=db,
                             auth=(SimpleNamespace(id=1), SimpleNamespace(id=2)), **form), db


@pytest.mark.asyncio
async def test_ocr_busy_is_a_429_with_retry_after(api):
    from fastapi import HTTPException
    from backend.app.services.ocr_admission import OcrBusy

    perform = AsyncMock(side_effect=OcrBusy("user", 2))
    with pytest.raises(HTTPException) as exc:
        await _call(api, api.ocr, perform, output_format="markdown")
    assert exc.value.status_code == 429
    assert exc.value.headers == {"Retry-After": "30"}
    assert "limit 2 at a time" in exc.value.detail


@pytest.mark.asyncio
async def test_ocrmd_busy_is_a_429_with_retry_after(api):
    from backend.app.services.ocr_admission import OcrBusy

    perform = AsyncMock(side_effect=OcrBusy("global", 8))
    resp, _ = await _call(api, api.ocrmd, perform)
    assert resp.status_code == 429 and resp.headers["retry-after"] == "30"


@pytest.mark.asyncio
async def test_ocr_rejects_out_of_range_dpi_before_any_work(api):
    from fastapi import HTTPException

    perform = AsyncMock()
    with pytest.raises(HTTPException) as exc:
        await _call(api, api.ocr, perform, output_format="markdown", dpi=1200)
    assert exc.value.status_code == 400 and "dpi" in exc.value.detail
    perform.assert_not_awaited()


@pytest.mark.asyncio
async def test_ocrmd_rejects_out_of_range_chunk_size(api):
    perform = AsyncMock()
    resp, _ = await _call(api, api.ocrmd, perform, chunk_size=500)
    assert resp.status_code == 400
    perform.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("handler_name", ["ocr", "ocrmd"])
async def test_request_session_is_released_before_ocr_starts(api, handler_name):
    order = []
    perform = AsyncMock(side_effect=lambda **kw: order.append("ocr") or {
        "content": "x", "format": "markdown", "pages": 1, "chunks_processed": 1,
        "usage": {}})
    registry = SimpleNamespace(resolve_alias=lambda m: (m, None))
    db = SimpleNamespace(commit=AsyncMock(side_effect=lambda: order.append("commit")))
    form = {"model": None, "chunk_size": None, "overlap": None, "dpi": None}
    if handler_name == "ocr":
        form["output_format"] = "markdown"
    with (
        patch("backend.app.services.ocr.get_ocr_config", AsyncMock(return_value=dict(_CONFIG))),
        patch("backend.app.services.ocr.perform_ocr", perform),
        patch.object(api, "get_registry", lambda: registry),
        patch.object(api, "model_availability", AsyncMock(return_value=api.AVAILABLE)),
        patch.object(api, "bind_request_context"),
    ):
        await getattr(api, handler_name)(request=SimpleNamespace(), file=_Upload(), db=db,
                                         auth=(SimpleNamespace(id=1), SimpleNamespace(id=2)), **form)
    assert order == ["commit", "ocr"]


def test_admin_form_saves_the_load_limits():
    src = (REPO_ROOT / "backend/app/dashboard/routes.py").read_text()
    for key in ("ocr.max_documents_per_user", "ocr.max_documents", "ocr.max_concurrent_conversions"):
        assert f'"{key}"' in src
    tpl = (REPO_ROOT / "backend/app/dashboard/templates/admin/ocr_config.html").read_text()
    for field in ("max_documents_per_user", "max_documents", "max_concurrent_conversions"):
        assert f'name="{field}"' in tpl


# --------------------------------------------------------------------------
# key_usage: counted in memory, written in batches
# --------------------------------------------------------------------------


@pytest.fixture
def key_usage(monkeypatch):
    import backend.app.services.key_usage as ku

    monkeypatch.setattr(ku, "_pending", {})
    writes, fail_ids = [], set()

    class _Sess:
        async def commit(self):
            writes.append("commit")

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def fake_session():
        yield _Sess()

    async def add(db, key_id, count, lock_wait_seconds=None):
        if key_id in fail_ids:
            raise RuntimeError("lock wait timeout")
        writes.append((key_id, count, lock_wait_seconds))

    crud_stub = SimpleNamespace(add_api_key_usage=add)
    session_stub = SimpleNamespace(isolated_async_session=fake_session)
    monkeypatch.setitem(sys.modules, "backend.app.db.session", session_stub)
    import backend.app.db as db_pkg
    monkeypatch.setattr(db_pkg, "crud", crud_stub, raising=False)
    monkeypatch.setitem(sys.modules, "backend.app.db.crud", crud_stub)
    return ku, writes, fail_ids


@pytest.mark.asyncio
async def test_uses_are_summed_and_written_once_per_key(key_usage):
    ku, writes, _ = key_usage
    for k in (5, 5, 9, 5, None):
        ku.record(k)
    assert ku.pending() == {5: 3, 9: 1}
    assert await ku.flush() == 2
    assert writes == [(5, 3, ku.LOCK_WAIT_SECONDS), "commit", (9, 1, ku.LOCK_WAIT_SECONDS), "commit"]
    assert ku.pending() == {}
    assert await ku.flush() == 0


@pytest.mark.asyncio
async def test_a_failed_write_keeps_its_count_for_next_time(key_usage):
    ku, writes, fail_ids = key_usage
    ku.record(5); ku.record(9); ku.record(9)
    fail_ids.add(9)
    assert await ku.flush() == 1
    assert ku.pending() == {9: 2}
    ku.record(9)
    fail_ids.clear()
    assert await ku.flush() == 1
    assert writes[-2:] == [(9, 3, ku.LOCK_WAIT_SECONDS), "commit"]


@pytest.mark.asyncio
async def test_a_flush_cancelled_mid_way_keeps_the_unwritten_counts(key_usage, monkeypatch):
    ku, writes, _ = key_usage
    import backend.app.db.crud as crud_mod  # the stub installed by the fixture

    async def slow_add(db, key_id, count, lock_wait_seconds=None):
        if key_id == 9:
            await asyncio.sleep(1)
        writes.append((key_id, count, lock_wait_seconds))

    monkeypatch.setattr(crud_mod, "add_api_key_usage", slow_add)
    ku.record(5); ku.record(9); ku.record(9); ku.record(12)
    t = asyncio.create_task(ku.flush())
    await asyncio.sleep(0.01)
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t
    assert ku.pending() == {9: 2, 12: 1}, "counts dropped on cancellation"


@pytest.mark.asyncio
async def test_bounded_lock_wait_is_per_statement():
    """SET STATEMENT ... FOR keeps the timeout off the pooled connection."""
    from backend.app.db import crud

    db = SimpleNamespace(execute=AsyncMock())
    await crud.add_api_key_usage(db, 5, 3, lock_wait_seconds=1)
    stmt, params = db.execute.await_args.args
    sql = str(stmt)
    assert sql.startswith("SET STATEMENT innodb_lock_wait_timeout = :wait FOR UPDATE api_keys")
    assert "SET SESSION" not in sql
    assert params == {"wait": 1, "count": 3, "id": 5}


def test_flush_loop_is_started_and_flushed_at_shutdown():
    src = (REPO_ROOT / "backend/app/main.py").read_text()
    assert "key_usage.flush_loop()" in src
    assert "await key_usage.flush()" in src
