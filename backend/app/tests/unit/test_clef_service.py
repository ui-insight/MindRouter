############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_clef_service.py: Unit tests for the standalone System
# One server for Cloudflare's Clef decision model
# (clef_service/). Must pass with NO gpu and NO torch — the
# model is replaced through the MODEL_FACTORY hook.
#
# Covers: refusing to start without a key, bearer auth (401),
# open liveness vs keyed health detail, request validation
# (422) and body size (413), the System One response shape and
# truncation reporting, dynamic batching (concurrent requests
# answered by one forward pass), the padded-token budget, a
# bad request failing alone inside a batch, the bounded queue
# (503), out-of-memory batches retried one request at a time,
# and the invariant that request text never reaches the logs.
#
############################################################

"""Unit tests for the Clef System One service."""

import asyncio
import logging
import sys
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Same pattern as test_dlp_service.py: make the repo root importable so
# `clef_service` resolves regardless of pytest rootdir.
_REPO_ROOT = str(Path(__file__).resolve().parents[4])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from clef_service import server  # noqa: E402
from clef_service.server import Batcher, Encoded, ServiceConfig  # noqa: E402

KEY = "test-clef-key"
AUTH = {"Authorization": f"Bearer {KEY}"}
SECRET = "SECRET-STATE-TEXT"


class OutOfMemoryError(RuntimeError):
    """Named like torch's so the service's out-of-memory check recognises it."""


class FakeEngine:
    """Stands in for the loaded Clef model. One "token" per whitespace word."""

    device = "cpu"
    short_answers = False

    def __init__(self, config, blocker=None, fail=None, oom_above=None):
        self.config = config
        self.batches = []          # size of every forward pass
        self.batch_tokens = []     # padded tokens of every forward pass
        self.seen_images = []      # image bytes handed to the model, per request
        self.encoded = []          # states tokenized
        self.inferred = []         # states that got a forward pass
        self._blocker, self._fail, self._oom_above = blocker, fail, oom_above

    def encode(self, request):
        if request["state"] == "SCHEMA-TOO-LONG":
            raise ValueError("schema requires 99999 tokens before state; maximum is 16384")
        if request["state"] == "ENCODER-BUG":
            raise RuntimeError("unexpected failure quoting " + SECRET)
        self.encoded.append(request["state"])
        self.seen_images.append(list(request.get("_images") or []))
        full = len(str(request["state"]).split()) + 10
        dropped = max(0, full - self.config.max_length)
        return Encoded(payload=request, tokens=full - dropped, state_tokens_dropped=dropped,
                       images=len(request.get("_images") or []))

    def infer(self, batch, requests):
        if self._blocker is not None:
            self._blocker.wait(timeout=5)
        if self._oom_above is not None and len(batch) > self._oom_above:
            raise OutOfMemoryError("CUDA out of memory")
        if self._fail is not None:
            raise self._fail
        self.batches.append(len(batch))
        self.inferred.extend(r["state"] for r in requests)
        if self.short_answers:
            return [{}] * (len(batch) - 1)
        self.batch_tokens.append(len(batch) * max(item.tokens for item in batch))
        out = []
        for request in requests:
            answers = {}
            for qid, q in request["questions"].items():
                if q["type"] == "noul":
                    answers[qid] = {"type": "noul", "noul": 0.75}
                elif q["type"] == "choice":
                    keys = [str(k) for k in q["criteria"]]
                    answers[qid] = {"type": "choice", "choice": keys[0], "confidence": 0.9,
                                    "probabilities": {k: (0.9 if i == 0 else 0.1 / max(1, len(keys) - 1))
                                                      for i, k in enumerate(keys)}}
                else:
                    levels = [str(i) for i in range(len(q["criteria"]))]
                    answers[qid] = {"type": "score", "score": 0.0, "confidence": 1.0,
                                    "legend": dict(zip(levels, q["criteria"])),
                                    "probabilities": {lvl: (1.0 if lvl == "0" else 0.0) for lvl in levels}}
            out.append(answers)
        return out


def _config(**over):
    return ServiceConfig(api_key=KEY, batch_wait_ms=0, **over)


def _body(state="A customer reports checkout failing.", **over):
    body = {"model": "clef", "state": state,
            "questions": {"urgent": {"type": "noul", "instructions": "Is this urgent?"}}}
    body.update(over)
    return body


@pytest.fixture
def engine_box(monkeypatch):
    """Installs a factory and exposes the engine the service built."""
    box = {"kwargs": {}}

    def factory(config):
        box["engine"] = FakeEngine(config, **box["kwargs"])
        return box["engine"]

    monkeypatch.setattr(server, "MODEL_FACTORY", factory)
    return box


@pytest.fixture
def client(engine_box):
    with TestClient(server.create_app(_config())) as c:
        yield c


# ---------------------------------------------------------------------------
# start-up and auth
# ---------------------------------------------------------------------------

class TestStartAndAuth:
    def test_refuses_to_start_without_a_key(self):
        with pytest.raises(SystemExit):
            server.create_app(ServiceConfig(api_key=None))

    def test_no_auth_must_be_asked_for_explicitly(self, engine_box):
        with TestClient(server.create_app(ServiceConfig(api_key=None, allow_no_auth=True, batch_wait_ms=0))) as c:
            assert c.post("/v1/systemone", json=_body()).status_code == 200

    @pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}, {"Authorization": KEY},
                                         {"Authorization": "Basic " + KEY}])
    def test_inference_needs_the_bearer_key(self, client, headers):
        assert client.post("/v1/systemone", json=_body(), headers=headers).status_code == 401
        assert client.get("/v1/models", headers=headers).status_code == 401

    def test_health_is_liveness_for_anyone_and_detail_with_the_key(self, client):
        assert client.get("/health").json() == {"status": "ok"}
        detail = client.get("/health", headers=AUTH).json()
        assert detail["status"] == "ok" and detail["model"] == "clef" and detail["device"] == "cpu"
        assert {"queue_depth", "max_queue", "max_batch", "max_length", "stats"} <= set(detail)

    def test_model_list_is_typesafes_shape(self, client):
        models = client.get("/v1/models", headers=AUTH).json()["models"]
        assert [m["name"] for m in models] == ["clef"]
        assert set(models[0]) == {"name", "description", "release_date"}


# ---------------------------------------------------------------------------
# request validation
# ---------------------------------------------------------------------------

class TestValidation:
    @pytest.mark.parametrize("body", [
        {"questions": {"q": {"type": "noul"}}},                                      # no state
        {"state": 5, "questions": {"q": {"type": "noul"}}},                          # state not text/object/array
        {"state": "s"},                                                              # no questions
        {"state": "s", "questions": {}},
        {"state": "s", "questions": {"q": {"type": "essay"}}},
        {"state": "s", "questions": {"q": {"type": "choice"}}},                      # choice without options
        {"state": "s", "questions": {"q": {"type": "choice", "criteria": ["a", "b"]}}},
        {"state": "s", "questions": {"q": {"type": "score", "criteria": {"a": "b"}}}},
        {"state": "s", "questions": {"q": {"type": "score", "criteria": ["x"] * 11}}},
        {"state": "s", "questions": {"q": {"type": "noul", "criteria": {"maybe": "x"}}}},
        {"state": "s", "model": 7, "questions": {"q": {"type": "noul"}}},
        {"state": "s", "images": ["https://example.com/a.png"], "questions": {"q": {"type": "noul"}}},  # no remote URLs
        {"state": "s", "videos": [[1, 2]], "questions": {"q": {"type": "noul"}}},      # video is not accepted
        {"state": "s", "questions": {f"q{i}": {"type": "noul"} for i in range(server.MAX_QUESTIONS + 1)}},
        ["not", "an", "object"],
    ])
    def test_bad_requests_are_422_and_never_reach_the_model(self, client, engine_box, body):
        r = client.post("/v1/systemone", json=body, headers=AUTH)
        assert r.status_code == 422 and isinstance(r.json()["detail"], str)
        assert engine_box["engine"].batches == []

    def test_invalid_json_is_422(self, client):
        r = client.post("/v1/systemone", content=b"{not json", headers={**AUTH, "Content-Type": "application/json"})
        assert r.status_code == 422

    def test_oversized_body_is_413(self, engine_box):
        with TestClient(server.create_app(_config(max_body_bytes=2048))) as c:
            r = c.post("/v1/systemone", json=_body(state="x" * 5000), headers=AUTH)
        assert r.status_code == 413 and engine_box["engine"].batches == []

    def test_a_request_the_model_cannot_take_is_422(self, client):
        r = client.post("/v1/systemone", json=_body(state="SCHEMA-TOO-LONG"), headers=AUTH)
        assert r.status_code == 422 and "schema requires" in r.json()["detail"]


# ---------------------------------------------------------------------------
# responses
# ---------------------------------------------------------------------------

class TestResponse:
    def test_system_one_shape(self, client):
        body = _body(questions={
            "urgent": {"type": "noul", "instructions": "Urgent?"},
            "team": {"type": "choice", "criteria": {"billing": "Payments", "technical": None}},
            "severity": {"type": "score", "criteria": ["Minor", "Major", "Critical"]},
        })
        d = client.post("/v1/systemone", json=body, headers=AUTH).json()
        assert d["model"] == "clef" and set(d["answers"]) == {"urgent", "team", "severity"}
        assert d["answers"]["urgent"] == {"type": "noul", "noul": 0.75}
        assert d["answers"]["team"]["choice"] == "billing" and set(d["answers"]["team"]["probabilities"]) == {"billing", "technical"}
        assert d["answers"]["severity"]["legend"] == {"0": "Minor", "1": "Major", "2": "Critical"}
        assert d["usage"]["output_tokens"] == 0 and d["usage"]["input_tokens"] > 0
        assert d["usage"]["truncated"] is False and d["usage"]["state_tokens_dropped"] == 0

    def test_a_cut_state_is_reported(self, engine_box):
        with TestClient(server.create_app(_config(max_length=256))) as c:
            d = c.post("/v1/systemone", json=_body(state="word " * 400), headers=AUTH).json()
        assert d["usage"]["truncated"] is True and d["usage"]["state_tokens_dropped"] == 400 + 10 - 256
        assert d["usage"]["input_tokens"] == 256

    def test_model_field_is_optional_and_the_reply_names_the_served_model(self, client):
        body = _body()
        del body["model"]
        assert client.post("/v1/systemone", json=body, headers=AUTH).json()["model"] == "clef"


# ---------------------------------------------------------------------------
# batching (the Batcher on its own, so concurrency is under the test's control)
# ---------------------------------------------------------------------------

async def _started(monkeypatch, config, **engine_kwargs):
    engines = []

    def factory(cfg):
        engines.append(FakeEngine(cfg, **engine_kwargs))
        return engines[-1]

    monkeypatch.setattr(server, "MODEL_FACTORY", factory)
    batcher = Batcher(config)
    await batcher.start()
    return batcher, engines[0]


class TestBatching:
    async def test_concurrent_requests_share_one_forward_pass(self, monkeypatch):
        batcher, engine = await _started(monkeypatch, ServiceConfig(api_key=KEY, batch_wait_ms=60, max_batch=8))
        try:
            results = await asyncio.gather(*(batcher.submit(_body(state=f"ticket {i}")) for i in range(5)))
        finally:
            await batcher.stop()
        assert engine.batches == [5] and batcher.stats.largest_batch == 5
        assert all(answers["urgent"]["noul"] == 0.75 for answers, _ in results)

    async def test_batch_size_is_capped(self, monkeypatch):
        batcher, engine = await _started(monkeypatch, ServiceConfig(api_key=KEY, batch_wait_ms=60, max_batch=3))
        try:
            await asyncio.gather(*(batcher.submit(_body(state=f"t {i}")) for i in range(7)))
        finally:
            await batcher.stop()
        assert max(engine.batches) <= 3 and sum(engine.batches) == 7

    async def test_padded_token_budget_splits_a_batch(self, monkeypatch):
        # Each request is ~110 tokens; a 250-token budget allows two per pass.
        cfg = ServiceConfig(api_key=KEY, batch_wait_ms=60, max_batch=8, max_batch_tokens=250)
        batcher, engine = await _started(monkeypatch, cfg)
        try:
            await asyncio.gather(*(batcher.submit(_body(state="w " * 100)) for _ in range(5)))
        finally:
            await batcher.stop()
        assert sum(engine.batches) == 5 and max(engine.batch_tokens) <= 250 and max(engine.batches) == 2

    async def test_one_long_request_still_runs_alone(self, monkeypatch):
        cfg = ServiceConfig(api_key=KEY, batch_wait_ms=0, max_batch_tokens=256, max_length=16384)
        batcher, engine = await _started(monkeypatch, cfg)
        try:
            answers, item = await batcher.submit(_body(state="w " * 5000))
        finally:
            await batcher.stop()
        assert engine.batches == [1] and item.tokens == 5010

    async def test_a_bad_request_fails_alone_inside_a_batch(self, monkeypatch):
        batcher, engine = await _started(monkeypatch, ServiceConfig(api_key=KEY, batch_wait_ms=60))
        try:
            results = await asyncio.gather(
                batcher.submit(_body(state="fine one")),
                batcher.submit(_body(state="SCHEMA-TOO-LONG")),
                batcher.submit(_body(state="fine two")),
                return_exceptions=True,
            )
        finally:
            await batcher.stop()
        assert isinstance(results[1], server.BadRequest)
        assert not isinstance(results[0], Exception) and not isinstance(results[2], Exception)
        assert engine.batches == [2]

    async def test_an_unexpected_encode_failure_also_fails_alone(self, monkeypatch):
        batcher, engine = await _started(monkeypatch, ServiceConfig(api_key=KEY, batch_wait_ms=60))
        try:
            results = await asyncio.gather(
                batcher.submit(_body(state="fine one")),
                batcher.submit(_body(state="ENCODER-BUG")),
                batcher.submit(_body(state="fine two")),
                return_exceptions=True,
            )
        finally:
            await batcher.stop()
        assert isinstance(results[1], RuntimeError) and SECRET not in str(results[1])
        assert not isinstance(results[0], Exception) and not isinstance(results[2], Exception)
        assert engine.batches == [2] and batcher.stats.failed == 1

    async def test_a_caller_who_left_gets_no_forward_pass(self, monkeypatch):
        gate = threading.Event()
        batcher, engine = await _started(monkeypatch, ServiceConfig(api_key=KEY, batch_wait_ms=0, max_batch=1),
                                         blocker=gate)
        try:
            first = asyncio.ensure_future(batcher.submit(_body(state="first")))
            await asyncio.sleep(0.05)                       # "first" is on the GPU, held by the gate
            gone = asyncio.ensure_future(batcher.submit(_body(state="gone")))
            stays = asyncio.ensure_future(batcher.submit(_body(state="stays")))
            await asyncio.sleep(0.02)
            gone.cancel()                                   # its client disconnected while queued
            gate.set()
            await asyncio.gather(first, stays)
        finally:
            await batcher.stop()
        assert engine.inferred == ["first", "stays"] and "gone" not in engine.encoded

    async def test_a_caller_leaving_mid_batch_does_not_hurt_the_others(self, monkeypatch):
        batcher, engine = await _started(monkeypatch, ServiceConfig(api_key=KEY, batch_wait_ms=60))
        try:
            tasks = [asyncio.ensure_future(batcher.submit(_body(state=s)))
                     for s in ("one", "SCHEMA-TOO-LONG", "two")]
            await asyncio.sleep(0.01)
            tasks[1].cancel()                               # leaves before its encode fails
            results = await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            await batcher.stop()
        assert isinstance(results[1], asyncio.CancelledError)
        assert not isinstance(results[0], Exception) and not isinstance(results[2], Exception)

    async def test_a_disconnect_cancels_the_waiting_work(self):
        class Gone:
            async def is_disconnected(self):
                return True

        cancelled = []

        async def work():
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelled.append(True)
                raise

        with pytest.raises(server.ClientGone):
            await server._unless_disconnected(Gone(), work(), poll_seconds=0.01)
        await asyncio.sleep(0)
        assert cancelled == [True]

    async def test_a_connected_caller_gets_the_result(self):
        class Here:
            async def is_disconnected(self):
                return False

        async def work():
            await asyncio.sleep(0.03)
            return "answer"

        assert await server._unless_disconnected(Here(), work(), poll_seconds=0.01) == "answer"

    async def test_stop_releases_everyone_still_waiting(self, monkeypatch):
        gate = threading.Event()
        batcher, engine = await _started(monkeypatch, ServiceConfig(api_key=KEY, batch_wait_ms=0, max_batch=1),
                                         blocker=gate)
        tasks = [asyncio.ensure_future(batcher.submit(_body(state=f"t{i}"))) for i in range(4)]
        await asyncio.sleep(0.05)
        await batcher.stop()
        gate.set()
        done, waiting = await asyncio.wait(tasks, timeout=2)
        assert not waiting                                  # nobody is left hanging

    async def test_a_short_reply_from_the_engine_fails_the_batch_instead_of_hanging(self, monkeypatch):
        batcher, engine = await _started(monkeypatch, ServiceConfig(api_key=KEY, batch_wait_ms=60))
        engine.short_answers = True
        try:
            results = await asyncio.wait_for(asyncio.gather(
                *(batcher.submit(_body(state=f"t{i}")) for i in range(3)), return_exceptions=True), timeout=2)
        finally:
            await batcher.stop()
        assert all(isinstance(r, RuntimeError) for r in results)

    async def test_out_of_memory_batch_is_retried_one_at_a_time(self, monkeypatch):
        batcher, engine = await _started(monkeypatch, ServiceConfig(api_key=KEY, batch_wait_ms=60), oom_above=1)
        try:
            results = await asyncio.gather(*(batcher.submit(_body(state=f"t {i}")) for i in range(3)))
        finally:
            await batcher.stop()
        assert len(results) == 3 and engine.batches == [1, 1, 1]

    async def test_a_full_queue_is_refused_not_grown(self, monkeypatch):
        blocker = threading.Event()
        cfg = ServiceConfig(api_key=KEY, batch_wait_ms=0, max_batch=1, max_queue=2)
        batcher, engine = await _started(monkeypatch, cfg, blocker=blocker)
        try:
            first = asyncio.create_task(batcher.submit(_body(state="in flight")))
            await asyncio.sleep(0.05)                      # the worker has taken it and is blocked in infer
            waiting = [asyncio.create_task(batcher.submit(_body(state=f"waiting {i}"))) for i in range(2)]
            await asyncio.sleep(0.01)
            with pytest.raises(server.Oversubscribed):
                await batcher.submit(_body(state="one too many"))
            assert batcher.stats.rejected_busy == 1
            blocker.set()
            await asyncio.gather(first, *waiting)
        finally:
            blocker.set()
            await batcher.stop()


# ---------------------------------------------------------------------------
# failures and logging
# ---------------------------------------------------------------------------

class TestFailuresAndLogs:
    def test_busy_is_503_with_retry_after(self, engine_box, monkeypatch):
        async def busy(self, request):
            raise server.Oversubscribed()

        monkeypatch.setattr(Batcher, "submit", busy)
        with TestClient(server.create_app(_config())) as c:
            r = c.post("/v1/systemone", json=_body(), headers=AUTH)
        assert r.status_code == 503 and r.headers["Retry-After"] == "1"

    def test_inference_failure_is_500_without_the_errors_text(self, engine_box, caplog):
        engine_box["kwargs"] = {"fail": RuntimeError(f"tensor mismatch near {SECRET}")}
        with caplog.at_level(logging.DEBUG):
            with TestClient(server.create_app(_config())) as c:
                r = c.post("/v1/systemone", json=_body(state=SECRET), headers=AUTH)
        assert r.status_code == 500 and r.json() == {"detail": "inference failed"}
        assert SECRET not in caplog.text and "RuntimeError" in caplog.text

    def test_request_text_never_reaches_the_logs(self, client, caplog):
        with caplog.at_level(logging.DEBUG):
            client.post("/v1/systemone", headers=AUTH, json=_body(
                state=SECRET, questions={"q": {"type": "noul", "instructions": f"Does it mention {SECRET}?"}}))
            client.post("/v1/systemone", headers=AUTH, json={"state": SECRET, "questions": {"q": {"type": "essay"}}})
        assert SECRET not in caplog.text


class TestEnvConfig:
    def test_defaults(self, monkeypatch):
        for name in list(__import__("os").environ):
            if name.startswith("CLEF_"):
                monkeypatch.delenv(name)
        cfg = ServiceConfig.from_env()
        assert (cfg.model, cfg.served_name, cfg.host, cfg.port) == ("Cloudflare/clef", "clef", "127.0.0.1", 18004)
        assert cfg.api_key is None and cfg.max_length == 16384 and cfg.max_batch == 8

    def test_environment_overrides_and_bad_values_stop_the_service(self, monkeypatch):
        monkeypatch.setenv("CLEF_API_KEY", "  k  ")
        monkeypatch.setenv("CLEF_MAX_BATCH", "4")
        cfg = ServiceConfig.from_env()
        assert cfg.api_key == "k" and cfg.max_batch == 4
        monkeypatch.setenv("CLEF_MAX_BATCH", "many")
        with pytest.raises(SystemExit):
            ServiceConfig.from_env()
        monkeypatch.setenv("CLEF_MAX_BATCH", "0")
        with pytest.raises(SystemExit):
            ServiceConfig.from_env()


# ---------------------------------------------------------------------------
# images
# ---------------------------------------------------------------------------

import base64  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 40          # signature is all this layer checks; pixels are the engine's job
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 40
WEBP = b"RIFF\x24\x00\x00\x00WEBPVP8 " + b"\x00" * 30


def _url(blob, kind):
    return f"data:{kind};base64,{base64.b64encode(blob).decode()}"


class TestImages:
    def test_data_urls_and_objects_reach_the_model_as_bytes(self, client, engine_box):
        body = _body(images=[_url(PNG, "image/png"),
                             {"content_type": "image/jpeg", "base64": base64.b64encode(JPEG).decode()},
                             _url(WEBP, "image/webp")])
        r = client.post("/v1/systemone", json=body, headers=AUTH)
        assert r.status_code == 200 and r.json()["metadata"]["images"] == 3
        assert engine_box["engine"].seen_images == [[PNG, JPEG, WEBP]]

    def test_text_only_requests_carry_no_images(self, client, engine_box):
        r = client.post("/v1/systemone", json=_body(), headers=AUTH)
        assert r.json()["metadata"]["images"] == 0 and engine_box["engine"].seen_images == [[]]

    @pytest.mark.parametrize("images", [
        "one.png",                                                   # not an array
        [_url(PNG, "image/png")] * (server.MAX_IMAGES + 1),
        ["data:image/gif;base64,R0lGODlhAQABAAAAACw="],
        ["data:image/png;base64,@@@"],
        [_url(JPEG, "image/png")],                                   # bytes are not the declared type
        [_url(b"plain text", "image/png")],
        [{"content_type": "image/png"}],
        [7],
    ])
    def test_bad_images_are_422_and_never_reach_the_model(self, client, engine_box, images):
        r = client.post("/v1/systemone", json=_body(images=images), headers=AUTH)
        assert r.status_code == 422 and engine_box["engine"].batches == []

    def test_size_limits(self, client, monkeypatch):
        monkeypatch.setattr(server, "MAX_IMAGE_BYTES", 20)
        assert client.post("/v1/systemone", json=_body(images=[_url(PNG, "image/png")]), headers=AUTH).status_code == 422
        monkeypatch.undo()
        monkeypatch.setattr(server, "MAX_TOTAL_IMAGE_BYTES", len(PNG) + 5)
        r = client.post("/v1/systemone", json=_body(images=[_url(PNG, "image/png")] * 2), headers=AUTH)
        assert r.status_code == 422 and "in total" in r.json()["detail"]

    def test_an_image_the_model_cannot_decode_is_422(self, engine_box, monkeypatch):
        def refuse(self, request):
            raise ValueError("image 0 could not be decoded")

        monkeypatch.setattr(FakeEngine, "encode", refuse)
        with TestClient(server.create_app(_config())) as c:
            r = c.post("/v1/systemone", json=_body(images=[_url(PNG, "image/png")]), headers=AUTH)
        assert r.status_code == 422 and "could not be decoded" in r.json()["detail"]

    def test_oversized_base64_is_refused_before_decoding(self, client, monkeypatch):
        calls = []
        monkeypatch.setattr(server.base64, "b64decode", lambda *a, **k: calls.append(1) or b"")
        monkeypatch.setattr(server, "MAX_IMAGE_CHARS", 200)
        padded = "AA " * 100
        r = client.post("/v1/systemone", json=_body(images=[f"data:image/png;base64,{padded}"]), headers=AUTH)
        assert r.status_code == 422 and "MiB" in r.json()["detail"] and calls == []

    def test_a_chunked_body_is_cut_off_at_the_limit(self, engine_box):
        def chunks():
            for _ in range(50):
                yield b"x" * 100
        with TestClient(server.create_app(_config(max_body_bytes=1000))) as c:
            r = c.post("/v1/systemone", content=chunks(), headers=AUTH)     # no Content-Length
        assert r.status_code == 413 and engine_box["engine"].batches == []

    def test_real_images_are_decoded_for_the_model(self):
        """The real engine's image step (no model needed): real files in, RGB pictures out."""
        import io
        from PIL import Image

        def picture(fmt, size=(12, 8), mode="RGB"):
            buf = io.BytesIO()
            Image.new(mode, size, "red").save(buf, format=fmt)
            return buf.getvalue()

        out = server.ClefEngine._open_images([picture("PNG", mode="RGBA"), picture("JPEG"), picture("WEBP")])
        assert [(p.mode, p.size) for p in out] == [("RGB", (12, 8))] * 3
        assert server.ClefEngine._open_images([]) == []

    def test_real_image_limits_are_the_callers_422(self, monkeypatch):
        import io
        from PIL import Image
        buf = io.BytesIO(); Image.new("RGB", (64, 64), "red").save(buf, format="PNG")
        whole = buf.getvalue()
        with pytest.raises(ValueError, match="could not be decoded"):
            server.ClefEngine._open_images([whole[: len(whole) // 2]])           # pixel data cut off
        gif = io.BytesIO(); Image.new("RGB", (4, 4)).save(gif, format="GIF")
        with pytest.raises(ValueError, match="could not be decoded"):
            server.ClefEngine._open_images([gif.getvalue()])                      # not one of the three types
        monkeypatch.setattr(server, "MAX_IMAGE_PIXELS", 100)
        with pytest.raises(ValueError, match="megapixels"):
            server.ClefEngine._open_images([whole])                               # checked before decoding

    def test_health_says_the_server_takes_images(self, client):
        assert client.get("/health", headers=AUTH).json()["images"] is True
