"""OCR pipeline: gateway-built JSON, per-page token budget, runaway collapse.

Background (2.9.74). `/v1/ocr` in JSON mode used to ask the MODEL to author
`{"pages": [{"page_number", "content", "tables"}]}` for a chunk of page
images. A general instruction model copes; an OCR specialist trained on a
few fixed prompts does not — dots.MOCR, the production default via the
`default-ocr` alias, re-emitted the page block 349 times until the flat
16,384-token ceiling cut it off, on a three-line test page, charging the
caller 16k tokens. The `/v1/ocr` handler also passed that JSON prompt for
`output_format=markdown`, so markdown mode returned JSON too.

Now the model only transcribes (the markdown prompt, both formats); JSON
mode runs one page per request and the pipeline assembles the envelope; the
completion budget scales with the pages in a request; and a completion that
hit its budget with a periodic tail is collapsed to one copy and flagged
`degraded` rather than returned in full — and never retried, since the
"you stopped early" prompt would just make a looping model loop again.

ocr.py is spec-loaded with the backend.app.* chain stubbed, exactly as
test_media_dos_guards.py does, so the module's own logic runs for real.
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]


class _Session:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def rollback(self):
        return None


def _stub_backend_chain(chat_completion):
    saved = {k: v for k, v in sys.modules.items() if k.startswith("backend")}
    for pkg in ("backend", "backend.app", "backend.app.core", "backend.app.core.translators",
                "backend.app.core.telemetry", "backend.app.db", "backend.app.services",
                "backend.app.dashboard", "backend.app.storage"):
        mod = ModuleType(pkg); mod.__path__ = []; sys.modules[pkg] = mod
    canon = ModuleType("backend.app.core.canonical_schemas")
    # Keep constructor kwargs inspectable: the budget the pipeline chose is
    # read back off the request object.
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

        async def chat_completion(self, canonical, user, api_key, http_request):
            return await chat_completion(canonical)

    inf.InferenceService = InferenceService
    sys.modules["backend.app.services.inference"] = inf
    return saved


def _load(chat_completion=None):
    saved = _stub_backend_chain(chat_completion or (lambda c: None))
    spec = importlib.util.spec_from_file_location("ocr_under_test", REPO_ROOT / "backend/app/services/ocr.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod, saved


def _restore(saved):
    for k in [k for k in sys.modules if k.startswith("backend")]:
        del sys.modules[k]
    sys.modules.update(saved)


@pytest.fixture
def ocr():
    mod, saved = _load()
    try:
        yield mod
    finally:
        _restore(saved)


CFG = {"temperature": 0.1, "max_tokens": 16384, "max_tokens_per_page": 4096,
       "max_retries": 2, "min_chars_per_page": 400, "max_pages": 200,
       "max_concurrent_chunks": 4, "prompt_ocrmd": "Transcribe {num_pages} pages as markdown."}

# The production failure shape: '[' then the same page object over and over,
# cut off mid-object by the token cap.
UNIT = '{"page_number": 1, "content": "MindRouter OCR test page\\nInvoice #2026-0923"}, '
PROD_LOOP = "[" + UNIT * 349 + UNIT[:17]


# --------------------------------------------------------------------------
# collapse_runaway_repetition
# --------------------------------------------------------------------------


def test_collapses_the_production_loop_to_one_copy(ocr):
    kept, collapsed = ocr.collapse_runaway_repetition(PROD_LOOP)
    assert collapsed is True
    assert kept == "[" + UNIT


def test_collapses_a_looping_line_after_real_content(ocr):
    """Real content survives; the loop is reduced to one copy.

    The seam is ambiguous by nature: the prefix and the looping line both end
    in a newline, so "prefix-minus-newline + newline-led line" is an equally
    valid parse and the cut may land one such character earlier. What must
    hold is one copy of the line, all of the real content, nothing invented.
    """
    prefix = "# Invoice\n\nTotal: $1,234.56\n\n"
    line = "| 2026-09-23 | Widget | 3 | $12.00 |\n"
    kept, collapsed = ocr.collapse_runaway_repetition(prefix + line * 80)
    assert collapsed is True
    assert kept.count("Widget") == 1
    assert kept.startswith(prefix.rstrip("\n"))
    assert kept.rstrip("\n") == (prefix + line).rstrip("\n")


def test_repeated_table_rows_in_a_real_document_are_untouched(ocr):
    """Three identical rows are ordinary; they neither dominate the text nor
    (in production) arrive with the token cap hit."""
    row = "| Site A | 2026-01-01 | 0.00 | none |\n"
    doc = "# Report\n\n" + ("Narrative paragraph about the site survey. " * 30) + row * 3 + "\n\nEnd."
    kept, collapsed = ocr.collapse_runaway_repetition(doc)
    assert collapsed is False and kept == doc


def test_non_periodic_text_is_untouched(ocr):
    text = "".join(f"line {i} value {i * 7919 % 1013}\n" for i in range(400))
    assert ocr.collapse_runaway_repetition(text) == (text, False)


def test_loop_unit_shorter_than_the_tail_window_is_found_at_one_period(ocr):
    """Regression: the period search must allow a match overlapping the tail.
    Bounding it at the tail's start skipped the copy one period back whenever
    the unit was shorter than the 40-char window, doubled the period, and
    kept two copies."""
    unit = "| a | b |\n"  # 10 chars, well under the window
    kept, collapsed = ocr.collapse_runaway_repetition("Head\n" + unit * 200)
    assert collapsed is True
    assert kept.count("| a | b |") == 1


def test_short_text_is_untouched(ocr):
    assert ocr.collapse_runaway_repetition("ab" * 30) == ("ab" * 30, False)


# --------------------------------------------------------------------------
# Per-page budget and JSON assembly
# --------------------------------------------------------------------------


def test_budget_scales_per_page_under_the_ceiling(ocr):
    assert ocr._effective_max_tokens(CFG, 1) == 4096
    assert ocr._effective_max_tokens(CFG, 3) == 12288
    assert ocr._effective_max_tokens(CFG, 5) == 16384  # ceiling wins
    assert ocr._effective_max_tokens({**CFG, "max_tokens_per_page": 0}, 1) == 16384


def test_pages_json_is_built_by_the_pipeline(ocr):
    out = json.loads(ocr._pages_json(["first ✓", "second"]))
    assert out == {"pages": [{"page_number": 1, "content": "first ✓"},
                             {"page_number": 2, "content": "second"}]}
    assert "✓" in ocr._pages_json(["✓"])  # not escaped to \\u2713


# --------------------------------------------------------------------------
# perform_ocr: JSON mode = transcription prompt, one page per request
# --------------------------------------------------------------------------


async def _run_perform(ocr, fmt, chunk_size=6, overlap=2, pages=3):
    calls = []

    async def fake_chunk(page_images, idx, start, end, total, nchunks, prompt_template, *a, **k):
        calls.append({"pages": end - start, "prompt": prompt_template, "start": start})
        return idx, f"page {start + 1} text", {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}, False

    async def fake_images(file_bytes, content_type, filename, dpi):
        return [b"img"] * pages

    ocr.ocr_chunk = fake_chunk
    ocr.document_to_images = fake_images
    result = await ocr.perform_ocr(
        file_bytes=b"x", content_type="image/png", filename="scan.png", model="dots.MOCR",
        output_format=fmt, chunk_size=chunk_size, overlap=overlap, dpi=200, ocr_config=dict(CFG),
        user=None, api_key=None, http_request=None,
    )
    return result, calls


@pytest.mark.asyncio
async def test_json_mode_runs_one_page_per_request_with_the_transcription_prompt(ocr):
    result, calls = await _run_perform(ocr, "json")
    assert [c["pages"] for c in calls] == [1, 1, 1]
    assert {c["prompt"] for c in calls} == {CFG["prompt_ocrmd"]}
    assert result["format"] == "json" and result["pages"] == 3 and result["degraded"] is False
    assert json.loads(result["content"]) == {"pages": [
        {"page_number": 1, "content": "page 1 text"},
        {"page_number": 2, "content": "page 2 text"},
        {"page_number": 3, "content": "page 3 text"},
    ]}


@pytest.mark.asyncio
async def test_markdown_mode_keeps_chunking_and_the_transcription_prompt(ocr):
    result, calls = await _run_perform(ocr, "markdown")
    assert [c["pages"] for c in calls] == [3]  # one chunk of three pages
    assert calls[0]["prompt"] == CFG["prompt_ocrmd"]
    assert result["content"] == "page 1 text" and result["format"] == "markdown"


# --------------------------------------------------------------------------
# ocr_chunk: the guard in situ — capped, looping completion is collapsed,
# flagged, budgeted per page, and not retried.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chunk_collapses_a_capped_loop_and_does_not_retry():
    seen = []

    async def looping(canonical):
        seen.append(canonical)
        return {"choices": [{"message": {"content": PROD_LOOP}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": 382, "completion_tokens": canonical.max_tokens,
                          "total_tokens": 382 + canonical.max_tokens}}

    mod, saved = _load(looping)
    try:
        idx, text, usage, degraded = await mod.ocr_chunk(
            [b"img"], 0, 0, 1, 1, 1, CFG["prompt_ocrmd"], "dots.MOCR", dict(CFG), None, None, None,
        )
    finally:
        _restore(saved)
    assert degraded is True
    assert text == "[" + UNIT
    assert len(seen) == 1, "a collapsed runaway must not be retried with the stronger prompt"
    assert seen[0].max_tokens == 4096  # one page -> per-page budget, not the 16k ceiling
    assert usage["completion_tokens"] == 4096


@pytest.mark.asyncio
async def test_chunk_leaves_an_uncapped_completion_alone():
    async def fine(canonical):
        return {"choices": [{"message": {"content": "x" * 900}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 300, "total_tokens": 310}}

    mod, saved = _load(fine)
    try:
        _, text, _, degraded = await mod.ocr_chunk(
            [b"img"], 0, 0, 1, 1, 1, CFG["prompt_ocrmd"], "m", dict(CFG), None, None, None,
        )
    finally:
        _restore(saved)
    assert degraded is False and text == "x" * 900


# --------------------------------------------------------------------------
# Wiring: handlers defer the prompt to the service; the admin page lists
# aliases so `default-ocr` no longer reads "not currently available".
# --------------------------------------------------------------------------


def test_handlers_no_longer_pick_a_json_prompt():
    src = (REPO_ROOT / "backend/app/api/v1_openai.py").read_text()
    assert 'prompt_template=ocr_config["prompt_ocr"]' not in src
    assert '"degraded": result.get("degraded", False)' in src
    assert "X-OCR-Degraded" in src


def test_admin_dropdown_offers_aliases_whose_target_is_multimodal():
    routes = (REPO_ROOT / "backend/app/dashboard/routes.py").read_text()
    start = routes.index("def _ocr_model_choices(")
    ns = {}
    exec(routes[start:routes.index("\n\n\nasync def admin_ocr_config(")], ns)  # the helper alone
    models = [SimpleNamespace(name="dots.MOCR", supports_multimodal=True),
              SimpleNamespace(name="qwen/qwen3.5-122b", supports_multimodal=True),
              SimpleNamespace(name="text-only", supports_multimodal=False)]
    aliases = [SimpleNamespace(alias_name="default-ocr", target_model="dots.MOCR"),
               SimpleNamespace(alias_name="default-chat", target_model="text-only")]
    names, alias_map = ns["_ocr_model_choices"](models, aliases)
    assert names == ["default-ocr", "dots.MOCR", "qwen/qwen3.5-122b"]
    assert alias_map == {"default-ocr": "dots.MOCR"}
    tpl = (REPO_ROOT / "backend/app/dashboard/templates/admin/ocr_config.html").read_text()
    assert "multimodal_aliases" in tpl
    assert 'name="prompt_ocr"' not in tpl, "the retired JSON prompt must not linger as a dead control"
    assert 'name="max_tokens_per_page"' in tpl
