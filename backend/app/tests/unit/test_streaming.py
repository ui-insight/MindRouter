"""Streaming format translation tests.

Tests ndjson (Ollama) <-> SSE (vLLM/OpenAI) stream parsing, chunk
conversion, and edge cases.
"""

import asyncio
import json
import pytest

from backend.app.core.translators import OllamaOutTranslator, VLLMOutTranslator
from backend.app.core.canonical_schemas import (
    CanonicalStreamChunk,
    MessageRole,
)


# --- Helpers ---


async def _async_iter(chunks):
    """Convert a list of bytes into an async iterator."""
    for chunk in chunks:
        yield chunk


async def _collect_stream(async_gen):
    """Collect all items from an async generator."""
    items = []
    async for item in async_gen:
        items.append(item)
    return items


class TestOllamaStreamParsing:
    """Parse Ollama ndjson stream -> CanonicalStreamChunks."""

    @pytest.mark.asyncio
    async def test_basic_stream(self, ollama_stream_chunks):
        """Parse ndjson Ollama stream into CanonicalStreamChunks."""
        chunks = await _collect_stream(
            OllamaOutTranslator.translate_chat_stream(
                _async_iter(ollama_stream_chunks), "req-1", "llama3.2"
            )
        )
        assert len(chunks) == 3
        assert all(isinstance(c, CanonicalStreamChunk) for c in chunks)

    @pytest.mark.asyncio
    async def test_stream_with_content(self, ollama_stream_chunks):
        """Content accumulation across chunks."""
        chunks = await _collect_stream(
            OllamaOutTranslator.translate_chat_stream(
                _async_iter(ollama_stream_chunks), "req-1", "llama3.2"
            )
        )
        contents = [c.choices[0].delta.content for c in chunks if c.choices[0].delta.content]
        assert contents == ["Hello", " there", "!"]

    @pytest.mark.asyncio
    async def test_stream_final_chunk_usage(self, ollama_stream_chunks):
        """Final chunk (done:true) extracts usage info."""
        chunks = await _collect_stream(
            OllamaOutTranslator.translate_chat_stream(
                _async_iter(ollama_stream_chunks), "req-1", "llama3.2"
            )
        )
        final = chunks[-1]
        assert final.choices[0].finish_reason == "stop"
        assert final.usage is not None
        assert final.usage.prompt_tokens == 10
        assert final.usage.completion_tokens == 3
        assert final.usage.total_tokens == 13

    @pytest.mark.asyncio
    async def test_stream_empty_content(self):
        """Chunks with empty content handled."""
        raw_chunks = [
            json.dumps({
                "model": "llama3.2",
                "message": {"role": "assistant", "content": ""},
                "done": False,
            }).encode() + b"\n",
            json.dumps({
                "model": "llama3.2",
                "message": {"role": "assistant", "content": "Hi"},
                "done": True,
                "prompt_eval_count": 5,
                "eval_count": 1,
            }).encode() + b"\n",
        ]

        chunks = await _collect_stream(
            OllamaOutTranslator.translate_chat_stream(
                _async_iter(raw_chunks), "req-1", "llama3.2"
            )
        )
        assert len(chunks) == 2

    @pytest.mark.asyncio
    async def test_stream_partial_lines(self):
        """Buffer handles partial byte chunks that split mid-JSON."""
        full_line = json.dumps({
            "model": "llama3.2",
            "message": {"role": "assistant", "content": "Hi"},
            "done": True,
            "prompt_eval_count": 5,
            "eval_count": 1,
        }) + "\n"

        # Split in the middle
        part1 = full_line[:15].encode()
        part2 = full_line[15:].encode()

        chunks = await _collect_stream(
            OllamaOutTranslator.translate_chat_stream(
                _async_iter([part1, part2]), "req-1", "llama3.2"
            )
        )
        assert len(chunks) == 1
        assert chunks[0].choices[0].delta.content == "Hi"

    @pytest.mark.asyncio
    async def test_stream_multiple_newlines(self):
        """Extra blank lines between chunks are ignored."""
        raw = (
            json.dumps({
                "model": "llama3.2",
                "message": {"role": "assistant", "content": "A"},
                "done": False,
            })
            + "\n\n\n"
            + json.dumps({
                "model": "llama3.2",
                "message": {"role": "assistant", "content": "B"},
                "done": True,
                "prompt_eval_count": 1,
                "eval_count": 2,
            })
            + "\n"
        ).encode()

        chunks = await _collect_stream(
            OllamaOutTranslator.translate_chat_stream(
                _async_iter([raw]), "req-1", "llama3.2"
            )
        )
        assert len(chunks) == 2


class TestVLLMSSEParsing:
    """Parse vLLM/OpenAI SSE stream -> CanonicalStreamChunks."""

    @pytest.mark.asyncio
    async def test_basic_sse_stream(self, vllm_sse_chunks):
        """Parse SSE data: {json}\\n\\n into CanonicalStreamChunks."""
        chunks = await _collect_stream(
            VLLMOutTranslator.translate_chat_stream(
                _async_iter(vllm_sse_chunks), "req-1", "llama3.2"
            )
        )
        # 4 data chunks before [DONE]
        assert len(chunks) == 4

    @pytest.mark.asyncio
    async def test_sse_done_signal(self, vllm_sse_chunks):
        """data: [DONE]\\n\\n terminates stream."""
        chunks = await _collect_stream(
            VLLMOutTranslator.translate_chat_stream(
                _async_iter(vllm_sse_chunks), "req-1", "llama3.2"
            )
        )
        # Stream should end before [DONE] — [DONE] is not a chunk
        for c in chunks:
            assert c.id  # all chunks have an id

    @pytest.mark.asyncio
    async def test_sse_with_role_delta(self, vllm_sse_chunks):
        """First chunk includes delta.role."""
        chunks = await _collect_stream(
            VLLMOutTranslator.translate_chat_stream(
                _async_iter(vllm_sse_chunks), "req-1", "llama3.2"
            )
        )
        first = chunks[0]
        assert first.choices[0].delta.role == MessageRole.ASSISTANT

    @pytest.mark.asyncio
    async def test_sse_usage_in_final_chunk(self, vllm_sse_chunks):
        """Usage info in final SSE chunk."""
        chunks = await _collect_stream(
            VLLMOutTranslator.translate_chat_stream(
                _async_iter(vllm_sse_chunks), "req-1", "llama3.2"
            )
        )
        final = chunks[-1]
        assert final.usage is not None
        assert final.usage.prompt_tokens == 10
        assert final.usage.completion_tokens == 2

    @pytest.mark.asyncio
    async def test_sse_comment_lines(self):
        """Lines starting with : (SSE comments) are ignored."""
        raw = (
            ": this is a comment\n\n"
            "data: " + json.dumps({
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "llama3.2",
                "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": None}],
            }) + "\n\n"
            "data: [DONE]\n\n"
        ).encode()

        chunks = await _collect_stream(
            VLLMOutTranslator.translate_chat_stream(
                _async_iter([raw]), "req-1", "llama3.2"
            )
        )
        assert len(chunks) == 1
        assert chunks[0].choices[0].delta.content == "Hi"

    @pytest.mark.asyncio
    async def test_sse_crlf_endings(self):
        """\\r\\n\\r\\n endings handled."""
        raw = (
            "data: " + json.dumps({
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "llama3.2",
                "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": None}],
            }) + "\r\n\r\n"
            "data: [DONE]\r\n\r\n"
        ).encode()

        chunks = await _collect_stream(
            VLLMOutTranslator.translate_chat_stream(
                _async_iter([raw]), "req-1", "llama3.2"
            )
        )
        assert len(chunks) == 1
        assert chunks[0].choices[0].delta.content == "Hi"

    @pytest.mark.asyncio
    async def test_sse_partial_chunks(self):
        """Partial byte delivery across buffer boundaries."""
        full = (
            "data: " + json.dumps({
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "llama3.2",
                "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}],
            }) + "\n\n"
            "data: [DONE]\n\n"
        )

        # Split at arbitrary points
        part1 = full[:20].encode()
        part2 = full[20:60].encode()
        part3 = full[60:].encode()

        chunks = await _collect_stream(
            VLLMOutTranslator.translate_chat_stream(
                _async_iter([part1, part2, part3]), "req-1", "llama3.2"
            )
        )
        assert len(chunks) == 1
        assert chunks[0].choices[0].delta.content == "Hello"

    @pytest.mark.asyncio
    async def test_sse_finish_reason(self):
        """finish_reason:"stop" in final choice."""
        raw = (
            "data: " + json.dumps({
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "llama3.2",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }) + "\n\n"
            "data: [DONE]\n\n"
        ).encode()

        chunks = await _collect_stream(
            VLLMOutTranslator.translate_chat_stream(
                _async_iter([raw]), "req-1", "llama3.2"
            )
        )
        assert chunks[0].choices[0].finish_reason == "stop"


def _openai_chunk_to_ollama(openai_chunk: dict) -> dict:
    """Convert OpenAI streaming chunk to Ollama format.

    Mirrors InferenceService._openai_chunk_to_ollama exactly.
    """
    choices = openai_chunk.get("choices", [])
    if not choices:
        return {"done": True}

    delta = choices[0].get("delta", {})
    finish = choices[0].get("finish_reason")

    return {
        "model": openai_chunk.get("model", ""),
        "message": {
            "role": delta.get("role", "assistant"),
            "content": delta.get("content", ""),
        },
        "done": finish is not None,
    }


class TestOpenAIChunkToOllamaConversion:
    """Test OpenAI-to-Ollama chunk conversion logic."""

    def test_basic_chunk_conversion(self):
        """OpenAI delta chunk -> Ollama ndjson format."""
        openai_chunk = {
            "id": "chatcmpl-1",
            "model": "llama3.2",
            "choices": [
                {"index": 0, "delta": {"content": "Hi"}, "finish_reason": None}
            ],
        }

        result = _openai_chunk_to_ollama(openai_chunk)

        assert result["model"] == "llama3.2"
        assert result["message"]["content"] == "Hi"
        assert result["done"] is False

    def test_chunk_with_content(self):
        """Content preserved in conversion."""
        openai_chunk = {
            "model": "llama3.2",
            "choices": [
                {"index": 0, "delta": {"content": "Hello world"}, "finish_reason": None}
            ],
        }

        result = _openai_chunk_to_ollama(openai_chunk)
        assert result["message"]["content"] == "Hello world"

    def test_chunk_done(self):
        """finish_reason -> done:true."""
        openai_chunk = {
            "model": "llama3.2",
            "choices": [
                {"index": 0, "delta": {}, "finish_reason": "stop"}
            ],
        }

        result = _openai_chunk_to_ollama(openai_chunk)
        assert result["done"] is True

    def test_chunk_no_choices(self):
        """Empty choices -> done:true."""
        openai_chunk = {"model": "llama3.2", "choices": []}

        result = _openai_chunk_to_ollama(openai_chunk)
        assert result["done"] is True

    def test_chunk_role_preserved(self):
        """delta.role -> message.role."""
        openai_chunk = {
            "model": "llama3.2",
            "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}
            ],
        }

        result = _openai_chunk_to_ollama(openai_chunk)
        assert result["message"]["role"] == "assistant"


class TestStreamingWithStructuredOutput:
    """Streaming with structured output formats."""

    @pytest.mark.asyncio
    async def test_ollama_json_stream(self):
        """Streaming with format:"json" produces valid chunks."""
        raw_chunks = [
            json.dumps({
                "model": "llama3.2",
                "message": {"role": "assistant", "content": '{"name":'},
                "done": False,
            }).encode() + b"\n",
            json.dumps({
                "model": "llama3.2",
                "message": {"role": "assistant", "content": '"Alice"}'},
                "done": True,
                "prompt_eval_count": 10,
                "eval_count": 5,
            }).encode() + b"\n",
        ]

        chunks = await _collect_stream(
            OllamaOutTranslator.translate_chat_stream(
                _async_iter(raw_chunks), "req-1", "llama3.2"
            )
        )
        assert len(chunks) == 2
        # Accumulate content
        full = "".join(
            c.choices[0].delta.content
            for c in chunks
            if c.choices[0].delta.content
        )
        assert json.loads(full) == {"name": "Alice"}

    @pytest.mark.asyncio
    async def test_vllm_json_schema_stream(self):
        """Streaming with response_format produces valid chunks."""
        raw = (
            "data: " + json.dumps({
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "llama3.2",
                "choices": [{"index": 0, "delta": {"content": '{"age": 30}'}, "finish_reason": None}],
            }) + "\n\n"
            "data: " + json.dumps({
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "llama3.2",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }) + "\n\n"
            "data: [DONE]\n\n"
        ).encode()

        chunks = await _collect_stream(
            VLLMOutTranslator.translate_chat_stream(
                _async_iter([raw]), "req-1", "llama3.2"
            )
        )
        assert len(chunks) == 2
        content_chunks = [
            c.choices[0].delta.content
            for c in chunks
            if c.choices[0].delta.content
        ]
        assert json.loads("".join(content_chunks)) == {"age": 30}
