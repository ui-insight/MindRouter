"""Hyperparameter fidelity tests.

Ensures that extended sampling parameters (top_k, repeat_penalty, min_p),
thinking mode (think), and opaque backend_options survive translation
through all four translator directions without silent data loss.
"""

import pytest

from backend.app.core.translators import (
    OpenAIInTranslator,
    OllamaInTranslator,
    OllamaOutTranslator,
    VLLMOutTranslator,
)
from backend.app.core.canonical_schemas import (
    CanonicalChatRequest,
    CanonicalMessage,
    MessageRole,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_canonical(**overrides) -> CanonicalChatRequest:
    """Build a minimal canonical chat request with overrides."""
    defaults = dict(
        model="test-model",
        messages=[CanonicalMessage(role=MessageRole.USER, content="Hi")],
    )
    defaults.update(overrides)
    return CanonicalChatRequest(**defaults)


def _ollama_req(**extra_options):
    """Build an Ollama chat request dict with extra options."""
    base = {
        "model": "llama3.2",
        "messages": [{"role": "user", "content": "Hi"}],
        "options": {},
    }
    base["options"].update(extra_options)
    return base


def _openai_req(**extra):
    """Build an OpenAI chat request dict with extra params."""
    base = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hi"}],
    }
    base.update(extra)
    return base


# ===========================================================================
# top_k
# ===========================================================================

class TestTopKTranslation:
    """top_k through all 4 translator directions + round-trips."""

    def test_ollama_in_reads_top_k(self):
        canonical = OllamaInTranslator.translate_chat_request(_ollama_req(top_k=40))
        assert canonical.top_k == 40

    def test_openai_in_reads_top_k(self):
        canonical = OpenAIInTranslator.translate_chat_request(_openai_req(top_k=50))
        assert canonical.top_k == 50

    def test_ollama_out_emits_top_k(self):
        payload = OllamaOutTranslator.translate_chat_request(_minimal_canonical(top_k=40))
        assert payload["options"]["top_k"] == 40

    def test_vllm_out_emits_top_k(self):
        payload = VLLMOutTranslator.translate_chat_request(_minimal_canonical(top_k=50))
        assert payload["top_k"] == 50

    def test_ollama_to_vllm_round_trip(self):
        canonical = OllamaInTranslator.translate_chat_request(_ollama_req(top_k=40))
        payload = VLLMOutTranslator.translate_chat_request(canonical)
        assert payload["top_k"] == 40

    def test_openai_to_ollama_round_trip(self):
        canonical = OpenAIInTranslator.translate_chat_request(_openai_req(top_k=50))
        payload = OllamaOutTranslator.translate_chat_request(canonical)
        assert payload["options"]["top_k"] == 50

    def test_top_k_none_omitted_ollama(self):
        payload = OllamaOutTranslator.translate_chat_request(_minimal_canonical())
        assert "top_k" not in payload.get("options", {})

    def test_top_k_none_omitted_vllm(self):
        payload = VLLMOutTranslator.translate_chat_request(_minimal_canonical())
        assert "top_k" not in payload


# ===========================================================================
# repeat_penalty  (Ollama) ↔ repetition_penalty (vLLM/OpenAI)
# ===========================================================================

class TestRepeatPenaltyTranslation:
    """repeat_penalty through all directions, including the vLLM rename."""

    def test_ollama_in_reads_repeat_penalty(self):
        canonical = OllamaInTranslator.translate_chat_request(
            _ollama_req(repeat_penalty=1.2)
        )
        assert canonical.repeat_penalty == 1.2

    def test_openai_in_reads_repetition_penalty(self):
        """OpenAI/vLLM sends 'repetition_penalty', mapped to canonical repeat_penalty."""
        canonical = OpenAIInTranslator.translate_chat_request(
            _openai_req(repetition_penalty=1.3)
        )
        assert canonical.repeat_penalty == 1.3

    def test_ollama_out_emits_repeat_penalty(self):
        payload = OllamaOutTranslator.translate_chat_request(
            _minimal_canonical(repeat_penalty=1.2)
        )
        assert payload["options"]["repeat_penalty"] == 1.2

    def test_vllm_out_emits_repetition_penalty(self):
        """Canonical repeat_penalty → vLLM 'repetition_penalty'."""
        payload = VLLMOutTranslator.translate_chat_request(
            _minimal_canonical(repeat_penalty=1.2)
        )
        assert payload["repetition_penalty"] == 1.2
        assert "repeat_penalty" not in payload

    def test_ollama_to_vllm_rename(self):
        canonical = OllamaInTranslator.translate_chat_request(
            _ollama_req(repeat_penalty=1.1)
        )
        payload = VLLMOutTranslator.translate_chat_request(canonical)
        assert payload["repetition_penalty"] == 1.1

    def test_openai_to_ollama_rename(self):
        canonical = OpenAIInTranslator.translate_chat_request(
            _openai_req(repetition_penalty=1.5)
        )
        payload = OllamaOutTranslator.translate_chat_request(canonical)
        assert payload["options"]["repeat_penalty"] == 1.5

    def test_repeat_penalty_none_omitted(self):
        payload = VLLMOutTranslator.translate_chat_request(_minimal_canonical())
        assert "repetition_penalty" not in payload


# ===========================================================================
# min_p
# ===========================================================================

class TestMinPTranslation:
    """min_p through all 4 translator directions + round-trips."""

    def test_ollama_in_reads_min_p(self):
        canonical = OllamaInTranslator.translate_chat_request(_ollama_req(min_p=0.05))
        assert canonical.min_p == 0.05

    def test_openai_in_reads_min_p(self):
        canonical = OpenAIInTranslator.translate_chat_request(_openai_req(min_p=0.1))
        assert canonical.min_p == 0.1

    def test_ollama_out_emits_min_p(self):
        payload = OllamaOutTranslator.translate_chat_request(
            _minimal_canonical(min_p=0.05)
        )
        assert payload["options"]["min_p"] == 0.05

    def test_vllm_out_emits_min_p(self):
        payload = VLLMOutTranslator.translate_chat_request(
            _minimal_canonical(min_p=0.1)
        )
        assert payload["min_p"] == 0.1

    def test_ollama_to_vllm_round_trip(self):
        canonical = OllamaInTranslator.translate_chat_request(_ollama_req(min_p=0.05))
        payload = VLLMOutTranslator.translate_chat_request(canonical)
        assert payload["min_p"] == 0.05

    def test_openai_to_ollama_round_trip(self):
        canonical = OpenAIInTranslator.translate_chat_request(_openai_req(min_p=0.1))
        payload = OllamaOutTranslator.translate_chat_request(canonical)
        assert payload["options"]["min_p"] == 0.1

    def test_min_p_none_omitted(self):
        payload = OllamaOutTranslator.translate_chat_request(_minimal_canonical())
        assert "min_p" not in payload.get("options", {})


# ===========================================================================
# backend_options  (opaque pass-through for Ollama-only params)
# ===========================================================================

class TestBackendOptionsPassthrough:
    """Ollama-only options (mirostat, tfs_z, num_ctx, etc.) via backend_options."""

    def test_ollama_in_collects_unknown_options(self):
        """Unknown options keys collected into backend_options."""
        req = _ollama_req(
            temperature=0.7,
            mirostat=2,
            mirostat_tau=5.0,
            tfs_z=0.9,
            num_ctx=4096,
        )
        canonical = OllamaInTranslator.translate_chat_request(req)
        assert canonical.temperature == 0.7
        assert canonical.backend_options == {
            "mirostat": 2,
            "mirostat_tau": 5.0,
            "tfs_z": 0.9,
            "num_ctx": 4096,
        }

    def test_ollama_out_merges_backend_options(self):
        """backend_options merged into Ollama options dict."""
        canonical = _minimal_canonical(
            temperature=0.7,
            backend_options={"mirostat": 2, "num_ctx": 4096},
        )
        payload = OllamaOutTranslator.translate_chat_request(canonical)
        opts = payload["options"]
        assert opts["temperature"] == 0.7
        assert opts["mirostat"] == 2
        assert opts["num_ctx"] == 4096

    def test_vllm_out_ignores_backend_options(self):
        """vLLM output should not contain backend_options."""
        canonical = _minimal_canonical(
            backend_options={"mirostat": 2, "num_ctx": 4096},
        )
        payload = VLLMOutTranslator.translate_chat_request(canonical)
        assert "mirostat" not in payload
        assert "num_ctx" not in payload
        assert "backend_options" not in payload

    def test_no_unknown_options_yields_none(self):
        """When all options keys are known, backend_options is None."""
        req = _ollama_req(temperature=0.7, top_k=40)
        canonical = OllamaInTranslator.translate_chat_request(req)
        assert canonical.backend_options is None

    def test_backend_options_round_trip_ollama(self):
        """Ollama in → canonical → Ollama out preserves opaque options."""
        req = _ollama_req(mirostat=2, tfs_z=0.9)
        canonical = OllamaInTranslator.translate_chat_request(req)
        payload = OllamaOutTranslator.translate_chat_request(canonical)
        assert payload["options"]["mirostat"] == 2
        assert payload["options"]["tfs_z"] == 0.9


# ===========================================================================
# think  (reasoning mode)
# ===========================================================================

class TestThinkingMode:
    """think flag: Ollama in/out, ignored by vLLM."""

    def test_ollama_in_reads_think(self):
        req = {
            "model": "deepseek-r1",
            "messages": [{"role": "user", "content": "Think step by step"}],
            "think": True,
        }
        canonical = OllamaInTranslator.translate_chat_request(req)
        assert canonical.think is True

    def test_ollama_in_think_false(self):
        req = {
            "model": "deepseek-r1",
            "messages": [{"role": "user", "content": "Quick answer"}],
            "think": False,
        }
        canonical = OllamaInTranslator.translate_chat_request(req)
        assert canonical.think is False

    def test_ollama_out_emits_think_true(self):
        payload = OllamaOutTranslator.translate_chat_request(
            _minimal_canonical(think=True)
        )
        assert payload["think"] is True
        # think should NOT be in options
        assert "think" not in payload.get("options", {})

    def test_ollama_out_omits_think_none(self):
        payload = OllamaOutTranslator.translate_chat_request(_minimal_canonical())
        assert "think" not in payload

    def test_vllm_out_ignores_think(self):
        payload = VLLMOutTranslator.translate_chat_request(
            _minimal_canonical(think=True)
        )
        assert "think" not in payload

    def test_ollama_round_trip_think(self):
        req = {
            "model": "deepseek-r1",
            "messages": [{"role": "user", "content": "Reason"}],
            "think": True,
        }
        canonical = OllamaInTranslator.translate_chat_request(req)
        payload = OllamaOutTranslator.translate_chat_request(canonical)
        assert payload["think"] is True
