"""Source-check tests for install-friction bug fixes (from field feedback).

Source inspection (no imports) to avoid the telemetry/DB import chains while
still guarding against regressions of two confirmed bugs.
"""

import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def _read(rel):
    with open(os.path.join(ROOT, rel)) as f:
        return f.read()


def test_multimodal_heuristic_covers_qwen36_and_dots():
    # #6: the vLLM name heuristic previously missed qwen3.6-* and dots.* (vision/OCR).
    src = _read("backend/app/core/telemetry/adapters/vllm.py")
    block = src[src.index("supports_multimodal = any("):]
    block = block[: block.index(")")]
    assert '"qwen3.5"' in block  # existing coverage preserved
    assert '"qwen3.6"' in block  # newly covered
    assert '"dots"' in block     # dots.OCR / dots.mocr


def test_sync_script_drops_nonexistent_supports_embeddings():
    # #10a: Backend has no supports_embeddings column -> reading it crashed the export.
    src = _read("scripts/sync_nodes_to_prod.py")
    assert "supports_embeddings" not in src           # removed
    assert "b.supports_multimodal" in src             # real field, kept
    assert "b.supports_thinking" in src               # real fields, now synced
    assert "b.supports_tools" in src
