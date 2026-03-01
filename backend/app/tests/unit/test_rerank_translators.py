"""Unit tests for rerank/score translator methods and canonical schemas."""

import pytest

from backend.app.core.canonical_schemas import (
    CanonicalRerankRequest,
    CanonicalRerankResponse,
    CanonicalRerankResult,
    CanonicalScoreData,
    CanonicalScoreRequest,
    CanonicalScoreResponse,
    UsageInfo,
)
from backend.app.core.translators.openai_in import OpenAIInTranslator
from backend.app.core.translators.vllm_out import VLLMOutTranslator


# ── OpenAIInTranslator: rerank ──────────────────────────────


class TestOpenAIInRerank:
    def test_translate_rerank_request_basic(self):
        data = {
            "model": "Qwen/Qwen3-Reranker-8B",
            "query": "What is ML?",
            "documents": ["ML is AI.", "Weather is nice."],
        }
        canonical = OpenAIInTranslator.translate_rerank_request(data)
        assert canonical.model == "Qwen/Qwen3-Reranker-8B"
        assert canonical.query == "What is ML?"
        assert canonical.documents == ["ML is AI.", "Weather is nice."]
        assert canonical.top_n is None
        assert canonical.return_documents is True

    def test_translate_rerank_request_with_top_n(self):
        data = {
            "model": "reranker",
            "query": "query",
            "documents": ["a", "b", "c"],
            "top_n": 2,
            "return_documents": False,
        }
        canonical = OpenAIInTranslator.translate_rerank_request(data)
        assert canonical.top_n == 2
        assert canonical.return_documents is False

    def test_translate_rerank_request_missing_model_raises(self):
        with pytest.raises(KeyError):
            OpenAIInTranslator.translate_rerank_request({"query": "q", "documents": []})

    def test_translate_rerank_request_missing_query_raises(self):
        with pytest.raises(KeyError):
            OpenAIInTranslator.translate_rerank_request({"model": "m", "documents": []})


# ── OpenAIInTranslator: score ───────────────────────────────


class TestOpenAIInScore:
    def test_translate_score_request_single(self):
        data = {
            "model": "Qwen/Qwen3-Reranker-8B",
            "text_1": "hello",
            "text_2": "world",
        }
        canonical = OpenAIInTranslator.translate_score_request(data)
        assert canonical.model == "Qwen/Qwen3-Reranker-8B"
        assert canonical.text_1 == "hello"
        assert canonical.text_2 == "world"

    def test_translate_score_request_batch(self):
        data = {
            "model": "reranker",
            "text_1": "query",
            "text_2": ["doc1", "doc2", "doc3"],
        }
        canonical = OpenAIInTranslator.translate_score_request(data)
        assert isinstance(canonical.text_2, list)
        assert len(canonical.text_2) == 3

    def test_translate_score_request_missing_text_raises(self):
        with pytest.raises(KeyError):
            OpenAIInTranslator.translate_score_request({"model": "m", "text_1": "a"})


# ── VLLMOutTranslator: rerank ───────────────────────────────


class TestVLLMOutRerank:
    def test_translate_rerank_request(self):
        canonical = CanonicalRerankRequest(
            model="reranker",
            query="What is ML?",
            documents=["ML is AI.", "Weather."],
            top_n=1,
        )
        payload = VLLMOutTranslator.translate_rerank_request(canonical)
        assert payload["model"] == "reranker"
        assert payload["query"] == "What is ML?"
        assert payload["documents"] == ["ML is AI.", "Weather."]
        assert payload["top_n"] == 1

    def test_translate_rerank_request_no_top_n(self):
        canonical = CanonicalRerankRequest(
            model="reranker",
            query="q",
            documents=["a"],
        )
        payload = VLLMOutTranslator.translate_rerank_request(canonical)
        assert "top_n" not in payload

    def test_translate_rerank_response(self):
        raw = {
            "id": "rnk-123",
            "model": "reranker",
            "results": [
                {"index": 1, "relevance_score": 0.95, "document": {"text": "ML is AI."}},
                {"index": 0, "relevance_score": 0.30, "document": {"text": "Weather."}},
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 0, "total_tokens": 20},
        }
        canonical = VLLMOutTranslator.translate_rerank_response(raw)
        assert isinstance(canonical, CanonicalRerankResponse)
        assert canonical.id == "rnk-123"
        assert len(canonical.results) == 2
        assert canonical.results[0].relevance_score == 0.95
        assert canonical.results[0].index == 1
        assert canonical.usage.prompt_tokens == 20

    def test_translate_rerank_response_empty(self):
        raw = {"id": "", "model": "", "results": [], "usage": {}}
        canonical = VLLMOutTranslator.translate_rerank_response(raw)
        assert canonical.results == []
        assert canonical.usage.prompt_tokens == 0


# ── VLLMOutTranslator: score ────────────────────────────────


class TestVLLMOutScore:
    def test_translate_score_request(self):
        canonical = CanonicalScoreRequest(
            model="reranker",
            text_1="hello",
            text_2=["a", "b"],
        )
        payload = VLLMOutTranslator.translate_score_request(canonical)
        assert payload["model"] == "reranker"
        assert payload["text_1"] == "hello"
        assert payload["text_2"] == ["a", "b"]

    def test_translate_score_response(self):
        raw = {
            "id": "scr-456",
            "object": "list",
            "model": "reranker",
            "data": [
                {"index": 0, "score": 0.85},
                {"index": 1, "score": 0.12},
            ],
            "usage": {"prompt_tokens": 15, "completion_tokens": 0, "total_tokens": 15},
        }
        canonical = VLLMOutTranslator.translate_score_response(raw)
        assert isinstance(canonical, CanonicalScoreResponse)
        assert canonical.id == "scr-456"
        assert len(canonical.data) == 2
        assert canonical.data[0].score == 0.85
        assert canonical.usage.prompt_tokens == 15

    def test_translate_score_response_empty(self):
        raw = {"id": "", "model": "", "data": [], "usage": {}}
        canonical = VLLMOutTranslator.translate_score_response(raw)
        assert canonical.data == []


# ── Canonical Schema Validation ──────────────────────────────


class TestCanonicalSchemas:
    def test_rerank_request_validation(self):
        req = CanonicalRerankRequest(
            model="m", query="q", documents=["a", "b"]
        )
        assert req.model == "m"
        d = req.model_dump()
        assert "request_id" in d

    def test_rerank_response_roundtrip(self):
        resp = CanonicalRerankResponse(
            id="test",
            model="m",
            results=[CanonicalRerankResult(index=0, relevance_score=0.9)],
            usage=UsageInfo(prompt_tokens=10, total_tokens=10),
        )
        d = resp.model_dump()
        assert d["results"][0]["relevance_score"] == 0.9

    def test_score_request_text_2_string(self):
        req = CanonicalScoreRequest(model="m", text_1="a", text_2="b")
        assert req.text_2 == "b"

    def test_score_request_text_2_list(self):
        req = CanonicalScoreRequest(model="m", text_1="a", text_2=["b", "c"])
        assert isinstance(req.text_2, list)

    def test_score_response_roundtrip(self):
        resp = CanonicalScoreResponse(
            id="test",
            model="m",
            data=[CanonicalScoreData(index=0, score=0.5)],
            usage=UsageInfo(prompt_tokens=5, total_tokens=5),
        )
        d = resp.model_dump()
        assert d["data"][0]["score"] == 0.5
