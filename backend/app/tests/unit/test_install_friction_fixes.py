"""Source-check tests for install-friction bug fixes (from field feedback).

Source inspection (no imports) to avoid the telemetry/DB import chains while
still guarding against regressions of two confirmed bugs.
"""

import importlib.util
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


def test_vllm_injects_include_usage_for_streaming():
    # #10b: streaming requests must ask vLLM for usage so quota accounting is real.
    from backend.app.core.translators.vllm_out import VLLMOutTranslator
    from backend.app.core.canonical_schemas import CanonicalChatRequest, CanonicalMessage

    msgs = [CanonicalMessage(role="user", content="hi")]
    streamed = VLLMOutTranslator.translate_chat_request(
        CanonicalChatRequest(model="m", messages=msgs, stream=True)
    )
    assert streamed.get("stream_options") == {"include_usage": True}
    nonstreamed = VLLMOutTranslator.translate_chat_request(
        CanonicalChatRequest(model="m", messages=msgs, stream=False)
    )
    assert nonstreamed.get("stream_options") is None


def test_streaming_loop_harvests_usage_and_completion_uses_it():
    # #10b wiring: capture the empty-choices usage chunk, suppress it from the
    # client, pass it to completion, and use it instead of the estimate.
    src = _read("backend/app/services/inference.py")
    assert "if chunk.usage is not None and not chunk.choices:" in src
    assert "real_usage = chunk.usage" in src
    assert "usage=real_usage," in src
    comp = src[src.index("async def _complete_streaming_request"):]
    comp = comp[: comp.index("\n    async def ", 1)]
    assert "if usage is not None:" in comp
    assert "usage.completion_tokens" in comp


def test_sync_script_drops_nonexistent_supports_embeddings():
    # #10a: Backend has no supports_embeddings column -> reading it crashed the export.
    src = _read("scripts/sync_nodes_to_prod.py")
    assert "supports_embeddings" not in src           # removed
    assert "b.supports_multimodal" in src             # real field, kept
    assert "b.supports_thinking" in src               # real fields, now synced
    assert "b.supports_tools" in src


# --- P0: install-hardening -------------------------------------------------
def test_startup_migrations_are_opt_in():
    # #1: RUN_MIGRATIONS gates an alembic upgrade before init_registry.
    settings = _read("backend/app/settings.py")
    assert "run_migrations: bool = False" in settings
    main = _read("backend/app/main.py")
    assert "async def _run_migrations()" in main
    assert "if get_settings().run_migrations:" in main
    assert 'command.upgrade(Config("alembic.ini"), "head")' in main
    # runs before the registry reads the backends table
    assert main.index("if get_settings().run_migrations:") < main.index("await init_registry()")


def test_seed_admin_is_automation_friendly():
    # #2a: env overrides + single parseable key line + supplied-key support.
    src = _read("scripts/seed_dev_data.py")
    assert 'os.environ.get("ADMIN_PASSWORD"' in src
    assert 'os.environ.get("ADMIN_API_KEY")' in src
    assert 'os.environ.get("MINT_ADMIN_KEY"' in src
    assert 'print(f"ADMIN_API_KEY={full_key}")' in src   # one parseable line
    assert "hash_api_key(admin_api_key)" in src          # honor a supplied key


def test_ocr_image_limit_error_detection():
    # #3: only image-related 400s trigger the single-page degrade.
    spec = importlib.util.spec_from_file_location(
        "ocr_mod", os.path.join(ROOT, "backend/app/services/ocr.py"))
    ocr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ocr)
    f = ocr._is_image_limit_error
    assert f(Exception("400: does not support multimodal/image input")) is True
    assert f(Exception("400: At most 1 image(s) may be provided")) is True
    assert f(Exception("500: internal error")) is False
    assert f(Exception("400: bad temperature value")) is False


def test_ocr_degrades_to_single_page():
    # #3 wiring: on an image-limit 400 with >1 page, fall back to per-page.
    src = _read("backend/app/services/ocr.py")
    assert "async def _ocr_pages_individually(" in src
    assert "if num_pages > 1 and _is_image_limit_error(e):" in src
    assert "await _ocr_pages_individually(" in src
