"""
RAG pipeline integration tests.

Tests embedding, reranking, and scoring endpoints through the MindRouter2 proxy,
culminating in an end-to-end RAG pipeline test.

Requirements:
  - Live MindRouter2 deployment with Qwen3-Embedding-8B and Qwen3-Reranker-8B models
  - Valid admin API key
"""

import math
import os

import httpx
import pytest

BASE_URL = os.environ.get("MINDROUTER_BASE_URL", "https://localhost:8000")
API_KEY = os.environ.get("MINDROUTER_API_KEY", "")
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
RERANKER_MODEL = "Qwen/Qwen3-Reranker-8B"

if not API_KEY:
    pytest.skip("MINDROUTER_API_KEY not set", allow_module_level=True)

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

TIMEOUT = 60.0


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


@pytest.fixture(scope="module")
def client():
    with httpx.Client(base_url=BASE_URL, headers=HEADERS, timeout=TIMEOUT) as c:
        yield c


class TestEmbedding:
    def test_embedding_single(self, client):
        """Embed a single query and verify vector dimensions."""
        resp = client.post("/v1/embeddings", json={
            "model": EMBEDDING_MODEL,
            "input": "What is machine learning?",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "data" in data
        assert len(data["data"]) == 1
        embedding = data["data"][0]["embedding"]
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        # All values should be floats
        assert all(isinstance(v, (int, float)) for v in embedding)

    def test_embedding_batch(self, client):
        """Embed multiple documents in a single request."""
        docs = [
            "Machine learning is a subset of artificial intelligence.",
            "The weather forecast predicts rain tomorrow.",
            "Deep learning uses neural networks with many layers.",
        ]
        resp = client.post("/v1/embeddings", json={
            "model": EMBEDDING_MODEL,
            "input": docs,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["data"]) == 3
        # All embeddings should have the same dimensions
        dims = set(len(d["embedding"]) for d in data["data"])
        assert len(dims) == 1


class TestRerank:
    def test_rerank(self, client):
        """Rerank documents against a query, verify scores and ordering."""
        documents = [
            "The weather is sunny today.",
            "Machine learning is a branch of artificial intelligence.",
            "I enjoy cooking pasta for dinner.",
            "Neural networks are inspired by the human brain.",
        ]
        resp = client.post("/v1/rerank", json={
            "model": RERANKER_MODEL,
            "query": "What is machine learning?",
            "documents": documents,
            "top_n": 2,
        })
        assert resp.status_code == 200
        data = resp.json()
        results = data["results"]
        assert len(results) == 2
        # Results should be sorted by relevance_score descending
        assert results[0]["relevance_score"] >= results[1]["relevance_score"]
        # The ML-related document should rank highest
        assert results[0]["index"] in (1, 3)  # indices of ML-related docs


class TestScore:
    def test_score_single(self, client):
        """Score a single query-document pair."""
        resp = client.post("/v1/score", json={
            "model": RERANKER_MODEL,
            "text_1": "What is machine learning?",
            "text_2": "Machine learning is a subset of AI.",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "data" in data
        assert len(data["data"]) == 1
        assert "score" in data["data"][0]
        assert isinstance(data["data"][0]["score"], (int, float))

    def test_score_batch(self, client):
        """Score a query against multiple documents."""
        resp = client.post("/v1/score", json={
            "model": RERANKER_MODEL,
            "text_1": "What is deep learning?",
            "text_2": [
                "Deep learning uses neural networks.",
                "I like to go hiking.",
                "Transformers revolutionized NLP.",
            ],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["data"]) == 3


class TestRAGEndToEnd:
    def test_rag_end_to_end(self, client):
        """Full RAG pipeline: embed → cosine top-k → rerank → verify best doc."""
        corpus = [
            "Python is a popular programming language for data science.",
            "The Eiffel Tower is located in Paris, France.",
            "Machine learning models learn patterns from data.",
            "Photosynthesis converts sunlight into chemical energy in plants.",
            "Neural networks are computational models inspired by the brain.",
            "The Great Wall of China is visible from space.",
        ]
        query = "How do neural networks learn from data?"

        # Step 1: Embed the corpus and query
        embed_resp = client.post("/v1/embeddings", json={
            "model": EMBEDDING_MODEL,
            "input": [query] + corpus,
        })
        assert embed_resp.status_code == 200
        embed_data = embed_resp.json()["data"]

        query_vec = embed_data[0]["embedding"]
        doc_vecs = [d["embedding"] for d in embed_data[1:]]

        # Step 2: Cosine similarity top-k retrieval
        similarities = [
            (i, cosine_similarity(query_vec, dv))
            for i, dv in enumerate(doc_vecs)
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = 4
        candidate_indices = [idx for idx, _ in similarities[:top_k]]
        candidate_docs = [corpus[i] for i in candidate_indices]

        # Step 3: Rerank the candidates
        rerank_resp = client.post("/v1/rerank", json={
            "model": RERANKER_MODEL,
            "query": query,
            "documents": candidate_docs,
        })
        assert rerank_resp.status_code == 200
        rerank_results = rerank_resp.json()["results"]

        # The top reranked result should be one of the ML/neural-network docs
        top_result = max(rerank_results, key=lambda r: r["relevance_score"])
        # Map back to original corpus index
        original_idx = candidate_indices[top_result["index"]]
        top_doc = corpus[original_idx]

        # The best document should be about neural networks or ML learning from data
        assert any(
            keyword in top_doc.lower()
            for keyword in ["neural network", "machine learning", "learn"]
        ), f"Unexpected top doc: {top_doc}"
