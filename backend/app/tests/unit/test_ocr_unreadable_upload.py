############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_ocr_unreadable_upload.py: a file OCR cannot read is a 400
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""An upload that is not a readable PDF or image is the client's error.

pdf2image raises PDFPageCountError for anything poppler's pdfinfo cannot read
(not a PDF, truncated, empty, password-protected) and PIL raises
UnidentifiedImageError for a file that is no image.  Neither is a ValueError,
which is what the OCR endpoints map to 400, so both came back as 500s (in
June 2026 a large share of one user's OCR traffic).  They are now
ValueErrors with a plain message, raised before any page is converted.

Uses the real pdf2image/poppler and PIL on deliberately broken input; ocr.py
is spec-loaded with the backend chain stubbed, restored afterwards.
"""

import importlib.util
import io
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]

BROKEN_PDFS = {
    "not a pdf": b"hello, this is plain text",
    "empty": b"",
    "truncated": b"%PDF-1.4\n1 0 obj\n<<>>\n",
}


@pytest.fixture
def ocr():
    saved = {k: v for k, v in sys.modules.items() if k.startswith("backend")}
    try:
        for pkg in ("backend", "backend.app", "backend.app.core", "backend.app.core.translators",
                    "backend.app.core.telemetry", "backend.app.db", "backend.app.services",
                    "backend.app.dashboard", "backend.app.storage"):
            mod = ModuleType(pkg); mod.__path__ = []; sys.modules[pkg] = mod
        canon = ModuleType("backend.app.core.canonical_schemas")
        for name in ("CanonicalChatRequest", "CanonicalMessage", "ImageUrlContent", "TextContent",
                     "CanonicalModelInfo"):
            setattr(canon, name, MagicMock())
        sys.modules["backend.app.core.canonical_schemas"] = canon
        sys.modules["backend.app.db.crud"] = ModuleType("backend.app.db.crud")
        log = MagicMock()
        lc = ModuleType("backend.app.logging_config"); lc.get_logger = lambda *a, **k: log
        sys.modules["backend.app.logging_config"] = lc
        st = ModuleType("backend.app.settings")
        st.get_settings = lambda: SimpleNamespace(ocr_max_frames=50, chat_upload_max_uncompressed_mb=100)
        sys.modules["backend.app.settings"] = st
        spec = importlib.util.spec_from_file_location(
            "ocr_unreadable_under_test", REPO_ROOT / "backend/app/services/ocr.py")
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        mod._test_log = log
        yield mod
    finally:
        for k in [k for k in sys.modules if k.startswith("backend")]:
            del sys.modules[k]
        sys.modules.update(saved)


@pytest.mark.parametrize("name", BROKEN_PDFS)
async def test_page_count_of_a_broken_pdf_is_a_value_error(ocr, name):
    with pytest.raises(ValueError) as info:
        await ocr._pdf_page_count(BROKEN_PDFS[name])
    assert str(info.value) == ocr.UNREADABLE_PDF_MESSAGE
    # Poppler's own message goes to the log, not to the client.
    assert "Unable to get page count" in str(info.value.__cause__)
    ocr._test_log.info.assert_any_call(
        "ocr_unreadable_upload", reason=ocr.UNREADABLE_PDF_MESSAGE,
        error=str(info.value.__cause__)[:300],
    )


async def test_converting_a_broken_pdf_is_a_value_error_and_frees_the_slot(ocr):
    with pytest.raises(ValueError, match="could not be read as a PDF"):
        await ocr._pdf_to_images(BROKEN_PDFS["truncated"], dpi=72)
    # The per-worker conversion limit is released on the error.
    assert ocr._conversions._active == 0


async def test_a_pdfinfo_that_died_is_our_error_not_the_files(ocr, monkeypatch):
    # pdfinfo killed (say by the OOM killer) on a valid PDF prints nothing
    # after pdf2image's "Unable to get page count."  Telling the user their
    # file is corrupt would be wrong; it stays a server error.
    from pdf2image import pdf2image as p2i
    from pdf2image.exceptions import PDFPageCountError

    def killed(*a, **k):
        raise PDFPageCountError("Unable to get page count.\n")

    monkeypatch.setattr(p2i, "pdfinfo_from_bytes", killed)
    with pytest.raises(PDFPageCountError):
        await ocr._pdf_page_count(b"%PDF-1.4 a perfectly good file")


async def test_a_range_conversion_failure_is_never_called_unreadable(ocr):
    # The pipeline counted these bytes' pages first, so a range failure is
    # poppler's own; it is passed on, not turned into a 400.
    from pdf2image.exceptions import PDFPageCountError

    with pytest.raises(PDFPageCountError):
        await ocr._pdf_to_images_range(BROKEN_PDFS["truncated"], 1, 2, dpi=72)
    assert ocr._conversions._active == 0


async def test_a_conversion_that_returns_no_pages_fails_the_chunk(ocr, monkeypatch):
    # A pdftoppm that crashed after pdfinfo succeeded returns nothing; the
    # chunk must not go to the model with no images.
    import pdf2image

    monkeypatch.setattr(pdf2image, "convert_from_bytes", lambda *a, **k: [])
    with pytest.raises(ocr.OcrConversionError):
        await ocr._pdf_to_images_range(b"%PDF", 3, 4, dpi=72)
    assert not issubclass(ocr.OcrConversionError, (ValueError, RuntimeError))


def _png(size=(64, 64)):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", size, (200, 10, 10)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.parametrize("damage", ["no image at all", "truncated", "corrupted body"])
async def test_an_image_that_cannot_be_decoded_is_a_value_error(ocr, damage):
    good = _png()
    data = {
        "no image at all": b"\x00not an image at all",
        # Image.open reads only the header; these fail while decoding.
        "truncated": good[: len(good) // 2],
        "corrupted body": good[:60] + bytes(b ^ 0xFF for b in good[60:120]) + good[120:],
    }[damage]
    with pytest.raises(ValueError) as info:
        await ocr.document_to_images(data, "image/png", "scan.png", 72)
    assert str(info.value) == ocr.UNREADABLE_IMAGE_MESSAGE


async def test_an_image_bomb_is_a_value_error_saying_too_large(ocr, monkeypatch):
    from PIL import Image

    # Any image over twice the limit raises DecompressionBombError.
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)
    with pytest.raises(ValueError) as info:
        await ocr.document_to_images(_png((64, 64)), "image/png", "scan.png", 72)
    assert str(info.value) == ocr.OVERSIZED_IMAGE_MESSAGE


async def test_a_good_image_still_decodes(ocr):
    pages = await ocr.document_to_images(_png(), "image/png", "scan.png", 72)
    assert len(pages) == 1 and pages[0][:8] == b"\x89PNG\r\n\x1a\n"


async def test_a_broken_pdf_fails_before_any_chunk_is_sent(ocr):
    sent = []

    async def fake_chunk(*a, **k):
        sent.append(a)

    ocr.ocr_chunk = fake_chunk
    with pytest.raises(ValueError, match="could not be read as a PDF"):
        await ocr.perform_ocr(
            file_bytes=BROKEN_PDFS["not a pdf"], content_type="application/pdf",
            filename="scan.pdf", model="dots.MOCR", output_format="markdown",
            chunk_size=6, overlap=2, dpi=72,
            ocr_config={"max_pages": 200, "max_concurrent_chunks": 4, "prompt_ocrmd": "x"},
            user=None, api_key=None, http_request=None,
        )
    assert sent == []


async def test_a_readable_pdf_still_converts(ocr):
    fixture = REPO_ROOT / "backend/app/tests/unit/fixtures/test_ocr.pdf"
    pdf = fixture.read_bytes()
    assert await ocr._pdf_page_count(pdf) >= 1
    pages = await ocr._pdf_to_images_range(pdf, 1, 1, dpi=72)
    assert len(pages) == 1 and pages[0][:8] == b"\x89PNG\r\n\x1a\n"
