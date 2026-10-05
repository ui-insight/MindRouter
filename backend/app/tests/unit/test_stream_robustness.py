############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_stream_robustness.py: Two ways a healthy streamed
# response used to be cut off by the gateway (found on Kimi K3
# traffic, 2026-10-05).
#
# 1. A character split across two network chunks raised
#    UnicodeDecodeError and failed the stream. Covers the
#    incremental decoder on its own and inside the vLLM and
#    Ollama stream translators and the Ollama-format proxy path.
# 2. One 180 s read timeout cut off any stream that went silent
#    that long, including a model writing a long tool call that
#    its server only sends once complete. Covers the two
#    separate limits (before the first byte / after the stream
#    has started), that only the wait for the backend is timed,
#    that outside cancellation passes through, the HTTP-level
#    context manager against a fake backend, and the setting.
#
############################################################

"""Streams survive split characters and long mid-stream silences."""

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest

from backend.app.core.text_stream import Utf8StreamDecoder

_REPO = Path(__file__).resolve().parents[4]
TEXT = "Idaho — spuds 🥔, snow ❄️, 山 and naïve café"


# ---------------------------------------------------------------------------
# 1. characters split across chunks
# ---------------------------------------------------------------------------

class TestDecoder:
    def test_any_split_point_gives_the_same_text(self):
        raw = TEXT.encode("utf-8")
        for cut in range(len(raw) + 1):
            d = Utf8StreamDecoder()
            assert d.feed(raw[:cut]) + d.feed(raw[cut:]) + d.flush() == TEXT, cut

    def test_one_byte_at_a_time(self):
        d = Utf8StreamDecoder()
        assert "".join(d.feed(bytes([b])) for b in TEXT.encode("utf-8")) + d.flush() == TEXT

    def test_the_first_half_of_a_character_is_held_not_raised(self):
        potato = "🥔".encode("utf-8")                     # four bytes
        d = Utf8StreamDecoder()
        assert d.feed(b"a" + potato[:2]) == "a"           # plain .decode() raised here
        assert d.feed(potato[2:] + b"b") == "🥔b"

    def test_invalid_bytes_become_a_replacement_character(self):
        d = Utf8StreamDecoder()
        assert d.feed(b"ok \xff\xfe done") == "ok �� done"
        d = Utf8StreamDecoder()
        assert d.feed("é".encode("utf-8")[:1]) == "" and d.flush() == "�"   # stream ended mid-character

    def test_text_passes_through_and_other_byte_types_work(self):
        d = Utf8StreamDecoder()
        assert d.feed("already text 🥔") == "already text 🥔"
        assert d.feed(bytearray("é".encode())) == "é" and d.feed(memoryview("山".encode())) == "山"


def _sse(deltas):
    frames = []
    for delta in deltas:
        chunk = {"id": "c", "object": "chat.completion.chunk", "model": "m",
                 "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
        frames.append(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8"))
    frames.append(b"data: [DONE]\n\n")
    return b"".join(frames)


async def _chunks(raw, size):
    for i in range(0, len(raw), size):
        yield raw[i:i + size]


class TestTranslatorsSurviveSplitCharacters:
    @pytest.mark.parametrize("size", [1, 3, 7, 4096])
    async def test_vllm_stream(self, size):
        from backend.app.core.translators.vllm_out import VLLMOutTranslator

        raw = _sse([{"role": "assistant"}, {"content": TEXT[:9]}, {"content": TEXT[9:]}])
        out = [c async for c in VLLMOutTranslator.translate_chat_stream(_chunks(raw, size), "rid", "m")]
        text = "".join((c.get("choices") or [{}])[0].get("delta", {}).get("content") or "" for c in out)
        assert text == TEXT

    async def test_vllm_stream_split_exactly_inside_a_character(self):
        # The production failure: a 4-byte character straddling a chunk boundary.
        from backend.app.core.translators.vllm_out import VLLMOutTranslator

        raw = _sse([{"content": "x 🥔 y"}])
        cut = raw.index("🥔".encode("utf-8")) + 2

        async def two():
            yield raw[:cut]
            yield raw[cut:]

        out = [c async for c in VLLMOutTranslator.translate_chat_stream(two(), "rid", "m")]
        assert "".join((c.get("choices") or [{}])[0].get("delta", {}).get("content") or "" for c in out) == "x 🥔 y"

    @pytest.mark.parametrize("size", [1, 5, 4096])
    async def test_ollama_stream(self, size):
        from backend.app.core.translators.ollama_out import OllamaOutTranslator

        lines = [{"model": "m", "message": {"role": "assistant", "content": TEXT[:9]}, "done": False},
                 {"model": "m", "message": {"role": "assistant", "content": TEXT[9:]}, "done": False},
                 {"model": "m", "message": {"role": "assistant", "content": ""}, "done": True}]
        raw = "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines).encode("utf-8")
        out = [c async for c in OllamaOutTranslator.translate_chat_stream(_chunks(raw, size), "rid", "m")]
        assert "".join((choice.delta.content or "") for c in out for choice in c.choices) == TEXT

    def test_no_raw_decode_is_left_on_a_stream_path(self):
        for rel in ("backend/app/services/inference.py", "backend/app/core/translators/vllm_out.py",
                    "backend/app/core/translators/ollama_out.py", "backend/app/core/translators/responses_stream.py"):
            src = (_REPO / rel).read_text()
            assert "chunk_bytes.decode(" not in src, rel
            assert "Utf8StreamDecoder" in src, rel


# ---------------------------------------------------------------------------
# 2. silence limits
# ---------------------------------------------------------------------------

async def _timed(chunks):
    """Yield (delay_before, chunk) pairs as a byte stream."""
    for delay, chunk in chunks:
        if delay:
            await asyncio.sleep(delay)
        yield chunk


class TestSilenceLimits:
    async def test_chunks_pass_through_and_the_stream_ends(self):
        from backend.app.services.inference import limit_stream_silence

        got = [c async for c in limit_stream_silence(_timed([(0, b"a"), (0, b"b"), (0, b"c")]), 0.2, 0.2)]
        assert got == [b"a", b"b", b"c"]
        assert [c async for c in limit_stream_silence(_timed([]), 0.2, 0.2)] == []

    async def test_no_first_byte_is_a_timeout_that_says_so(self):
        from backend.app.services.inference import limit_stream_silence

        with pytest.raises(httpx.ReadTimeout) as e:
            async for _ in limit_stream_silence(_timed([(0.3, b"late")]), 0.05, 5.0):
                pass
        assert "before its first byte" in str(e.value) and str(e.value)          # never a blank error again

    async def test_a_long_pause_after_the_stream_started_is_allowed(self):
        # The Kimi case: a first chunk, then silence longer than the first-byte limit
        # while the model writes a tool call its server sends in one piece.
        from backend.app.services.inference import limit_stream_silence

        stream = _timed([(0, b"role"), (0.25, b"the whole tool call"), (0, b"done")])
        got = [c async for c in limit_stream_silence(stream, 0.05, 5.0)]
        assert got == [b"role", b"the whole tool call", b"done"]

    async def test_silence_beyond_the_idle_limit_is_a_timeout_that_says_so(self):
        from backend.app.services.inference import limit_stream_silence

        got = []
        with pytest.raises(httpx.ReadTimeout) as e:
            async for c in limit_stream_silence(_timed([(0, b"role"), (0.4, b"never")]), 0.05, 0.1):
                got.append(c)
        assert got == [b"role"] and "mid-stream" in str(e.value)

    async def test_a_slow_consumer_is_not_counted_as_backend_silence(self):
        from backend.app.services.inference import limit_stream_silence

        got = []
        async for c in limit_stream_silence(_timed([(0, b"a"), (0, b"b"), (0, b"c")]), 0.05, 0.05):
            got.append(c)
            await asyncio.sleep(0.12)                     # the client is slow; the backend is not
        assert got == [b"a", b"b", b"c"]

    async def test_cancellation_from_outside_is_not_turned_into_a_timeout(self):
        from backend.app.services.inference import limit_stream_silence

        async def consume():
            async for _ in limit_stream_silence(_timed([(0, b"a"), (5, b"b")]), 1.0, 10.0):
                pass

        task = asyncio.ensure_future(consume())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


def _service(first=0.1, idle=1.0, handler=None):
    """An InferenceService with only what the streaming path touches, talking to a fake backend."""
    from backend.app.services.inference import InferenceService

    svc = InferenceService.__new__(InferenceService)
    svc._settings = MagicMock(backend_request_timeout_per_attempt=first, backend_stream_idle_timeout=idle)
    seen = {}

    def make(read_timeout=None):
        seen["read_timeout"] = read_timeout
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    svc._make_inference_client = make
    return svc, seen


class TestBackendStream:
    async def test_a_stream_with_a_long_tool_call_pause_arrives_whole(self):
        async def handler(request):
            return httpx.Response(200, content=_timed([(0, b"data: a\n\n"), (0.3, b"data: b\n\n")]))

        svc, seen = _service(first=0.1, idle=2.0, handler=handler)
        async with svc._backend_stream("http://backend/v1/chat/completions", {"x": 1}) as body:
            got = [c async for c in body]
        assert b"".join(got) == b"data: a\n\ndata: b\n\n"
        assert seen["read_timeout"] > 2.0                 # httpx's own limit sits above ours

    async def test_a_backend_that_never_answers_times_out_at_the_first_byte_limit(self):
        async def handler(request):
            await asyncio.sleep(5)
            return httpx.Response(200, content=b"late")

        svc, _ = _service(first=0.1, idle=30.0, handler=handler)
        started = asyncio.get_running_loop().time()
        with pytest.raises(httpx.ReadTimeout) as e:
            async with svc._backend_stream("http://backend/x", {}) as body:
                async for _ in body:
                    pass
        assert asyncio.get_running_loop().time() - started < 1.0          # not the 30 s idle limit
        assert "no response within" in str(e.value)

    async def test_an_error_status_is_raised_with_its_body_readable(self):
        async def handler(request):
            return httpx.Response(503, json={"error": "loading"})

        svc, _ = _service(handler=handler)
        with pytest.raises(httpx.HTTPStatusError) as e:
            async with svc._backend_stream("http://backend/x", {}):
                pass
        assert e.value.response.status_code == 503 and e.value.response.json() == {"error": "loading"}

    async def test_an_error_status_whose_body_stalls_fails_at_the_first_byte_limit(self):
        # Without its own bound this waited for the long mid-stream limit, holding the backend slot.
        async def handler(request):
            return httpx.Response(502, content=_timed([(5, b"never")]))

        svc, _ = _service(first=0.1, idle=30.0, handler=handler)
        started = asyncio.get_running_loop().time()
        with pytest.raises(httpx.ReadTimeout) as e:
            async with svc._backend_stream("http://backend/x", {}):
                pass
        assert asyncio.get_running_loop().time() - started < 2.0
        assert "HTTP 502" in str(e.value)

    @pytest.mark.parametrize("how", ["finished", "left early", "failed"])
    async def test_the_response_is_always_closed(self, how):
        # Closing is what tells the backend to stop generating for a caller who has gone.
        closed = []

        class Body(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b"a"
                yield b"b"

            async def aclose(self):
                closed.append(True)

        async def handler(request):
            return httpx.Response(200, stream=Body())

        svc, _ = _service(handler=handler)
        try:
            async with svc._backend_stream("http://backend/x", {}) as body:
                async for _ in body:
                    if how == "left early":
                        break
                    if how == "failed":
                        raise RuntimeError("consumer error")
        except RuntimeError:
            pass
        assert closed == [True]

    def test_the_real_client_factory_honours_the_read_timeout(self):
        # _backend_stream asks for idle + 30 s; if the factory ignored it, streams would
        # again be cut at the 180 s per-attempt value in production.
        from backend.app.services.inference import InferenceService

        svc = InferenceService.__new__(InferenceService)
        svc._settings = MagicMock(backend_request_timeout_per_attempt=180, backend_stream_idle_timeout=600)
        assert InferenceService._make_inference_client(svc, read_timeout=630.0).timeout.read == 630.0
        assert InferenceService._make_inference_client(svc).timeout.read == 180.0        # everything else unchanged

    async def test_the_stream_asks_the_factory_for_the_idle_limit_plus_a_margin(self):
        async def handler(request):
            return httpx.Response(200, content=b"a")

        svc, seen = _service(first=180, idle=600, handler=handler)
        async with svc._backend_stream("http://backend/x", {}) as body:
            assert [c async for c in body] == [b"a"]
        assert seen["read_timeout"] == 630.0

    async def test_the_idle_limit_is_never_shorter_than_the_first_byte_limit(self):
        async def handler(request):
            return httpx.Response(200, content=_timed([(0, b"a"), (0.1, b"b")]))

        svc, _ = _service(first=1.5, idle=0.01, handler=handler)          # misconfigured: idle < first
        async with svc._backend_stream("http://backend/x", {}) as body:
            assert [c async for c in body] == [b"a", b"b"]

    async def test_the_openai_stream_path_end_to_end(self):
        from backend.app.core.canonical_schemas import CanonicalChatRequest, CanonicalMessage, MessageRole
        from backend.app.db.models import BackendEngine

        raw = _sse([{"role": "assistant"}, {"content": "x 🥔 y"}])
        cut = raw.index("🥔".encode("utf-8")) + 2

        async def handler(request):
            assert json.loads(request.content)["stream"] is True
            return httpx.Response(200, content=_timed([(0, raw[:cut]), (0.25, raw[cut:])]))   # split character AND a pause

        svc, _ = _service(first=0.1, idle=2.0, handler=handler)
        request = CanonicalChatRequest(model="m", messages=[CanonicalMessage(role=MessageRole.USER, content="hi")], stream=True)
        backend = MagicMock(url="http://backend", engine=BackendEngine.VLLM)
        out = [c async for c in svc._proxy_stream_request(request, backend)]
        assert "".join((c.get("choices") or [{}])[0].get("delta", {}).get("content") or "" for c in out) == "x 🥔 y"

    async def test_the_ollama_format_path_end_to_end(self):
        from backend.app.core.canonical_schemas import CanonicalChatRequest, CanonicalMessage, MessageRole
        from backend.app.db.models import BackendEngine

        raw = _sse([{"role": "assistant"}, {"content": "x 🥔 y"}])
        cut = raw.index("🥔".encode("utf-8")) + 2

        async def handler(request):
            return httpx.Response(200, content=_timed([(0, raw[:cut]), (0.25, raw[cut:])]))

        svc, _ = _service(first=0.1, idle=2.0, handler=handler)
        request = CanonicalChatRequest(model="m", messages=[CanonicalMessage(role=MessageRole.USER, content="hi")], stream=True)
        backend = MagicMock(url="http://backend", engine=BackendEngine.VLLM)
        out = [c async for c in svc._proxy_ollama_stream(request, backend)]
        assert "".join((c.get("message") or {}).get("content") or "" for c in out) == "x 🥔 y"


class TestSetting:
    def test_the_orphan_sweep_allows_for_one_long_silence(self):
        src = (_REPO / "backend/app/core/scheduler/routing.py").read_text()
        block = src[src.index("max_lifetime_s = ("):]
        block = block[: block.index(")")]
        assert "settings.backend_stream_idle_timeout" in block

    def test_default_and_how_it_is_configured(self):
        import re

        from backend.app.settings import Settings

        default = Settings.model_fields["backend_stream_idle_timeout"].default
        first = Settings.model_fields["backend_request_timeout_per_attempt"].default
        assert default == 600 and default > first
        # An env var the app reads must be passed through by compose, or it is silently inert.
        compose = (_REPO / "docker-compose.yml").read_text()
        assert "BACKEND_STREAM_IDLE_TIMEOUT=${BACKEND_STREAM_IDLE_TIMEOUT:-600}" in compose
        # A silent stream must not outlast the front proxy, or the client is cut off there instead.
        nginx = (_REPO / "nginx" / "nginx.conf").read_text()
        front = min(int(x) for x in re.findall(r"proxy_read_timeout\s+(\d+)s", nginx))
        assert default < front
