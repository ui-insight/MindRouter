############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# ocr.py: OCR service using multimodal LLM
#
# Converts images, PDFs, and Office documents to markdown
# or JSON by sending page images to a multimodal model via
# the existing chat completion infrastructure.
#
# For multi-page documents, pages are processed in overlapping
# chunks (default: 6 pages, 2-page overlap) and merged
# deterministically using difflib sequence matching.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""OCR service: document → markdown/JSON via multimodal LLM."""

import asyncio
import base64
import collections
import contextlib
import contextvars
import difflib
import functools
import io
import json
import re
import shutil
import struct
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from backend.app.core.canonical_schemas import (
    CanonicalChatRequest,
    CanonicalMessage,
    ImageUrlContent,
    TextContent,
)
from backend.app.db import crud
from backend.app.logging_config import get_logger
from backend.app.settings import get_settings

logger = get_logger(__name__)

# Guard against decompression-bomb images (a small file that declares an
# enormous pixel grid). PIL raises DecompressionBombError when an image's
# pixel count exceeds this bound, before allocating the full buffer. The
# limit is generous relative to real documents (a US-Letter page at 300 DPI
# is ~8 MP) so no legitimate upload is rejected.
Image.MAX_IMAGE_PIXELS = 256_000_000


_DEFAULT_PROMPT_OCRMD = (
    "Convert ALL of the following page images to well-structured markdown. "
    "Render tables as proper markdown tables with correct columns and rows. "
    "Do not add any preamble like 'Here is the markdown' - just output "
    "the markdown directly. "
    "Preserve all text exactly as it appears. "
    "Do not summarize or omit anything. Do not add any commentary. "
    "There are {num_pages} page images - make sure you process EVERY page."
)

_DEFAULT_PROMPT_OCR = (
    "Output the result as a JSON object with a top-level 'pages' array. "
    "Each page should have 'page_number' (int), 'content' (string with "
    "the full text), and 'tables' (array of objects with 'headers' and 'rows'). "
    "Output valid JSON only — no markdown fences or commentary. "
    "Preserve all text exactly as it appears. "
    "Do not summarize or omit anything. Do not add any commentary. "
    "There are {num_pages} page images - make sure you process EVERY page."
)


async def get_ocr_config(db) -> dict:
    """Load OCR configuration from admin config (app_config table)."""
    return {
        "model": await crud.get_config_json(db, "ocr.default_model", "qwen/qwen3.5-122b"),
        "chunk_size": await crud.get_config_json(db, "ocr.chunk_size", 6),
        "overlap": await crud.get_config_json(db, "ocr.overlap", 2),
        "dpi": await crud.get_config_json(db, "ocr.dpi", 200),
        "max_pages": await crud.get_config_json(db, "ocr.max_pages", 200),
        "max_file_size_mb": await crud.get_config_json(db, "ocr.max_file_size_mb", 100),
        "max_concurrent_chunks": await crud.get_config_json(db, "ocr.max_concurrent_chunks", 4),
        "min_chars_per_page": await crud.get_config_json(db, "ocr.min_chars_per_page", 400),
        "max_retries": await crud.get_config_json(db, "ocr.max_retries", 2),
        "enabled": await crud.get_config_json(db, "ocr.enabled", True),
        "max_tokens": await crud.get_config_json(db, "ocr.max_tokens", 16384),
        # Per-request budget scales with the pages in the request; max_tokens
        # stays the absolute ceiling. Bounds a runaway generation by page count.
        "max_tokens_per_page": await crud.get_config_json(db, "ocr.max_tokens_per_page", 4096),
        "temperature": await crud.get_config_json(db, "ocr.temperature", 0.1),
        "prompt_ocr": await crud.get_config_json(db, "ocr.prompt_ocr", _DEFAULT_PROMPT_OCR),
        "prompt_ocrmd": await crud.get_config_json(db, "ocr.prompt_ocrmd", _DEFAULT_PROMPT_OCRMD),
        # Load limits (see services/ocr_admission.py). Documents are counted
        # fleet-wide; conversions are per uvicorn worker. 0 = unlimited for
        # the document limits.
        "max_documents_per_user": await crud.get_config_json(
            db, "ocr.max_documents_per_user", DEFAULT_MAX_DOCUMENTS_PER_USER),
        "max_documents": await crud.get_config_json(
            db, "ocr.max_documents", DEFAULT_MAX_DOCUMENTS),
        "max_concurrent_conversions": await crud.get_config_json(
            db, "ocr.max_concurrent_conversions", DEFAULT_MAX_CONVERSIONS),
        # A document over a limit waits this long for a slot (0 = refuse at
        # once); the wait line is bounded per user and server-wide.
        "queue_wait_seconds": await crud.get_config_json(
            db, "ocr.queue_wait_seconds", DEFAULT_QUEUE_WAIT_SECONDS),
        "max_waiting_per_user": await crud.get_config_json(
            db, "ocr.max_waiting_per_user", DEFAULT_MAX_WAITING_PER_USER),
        "max_waiting": await crud.get_config_json(
            db, "ocr.max_waiting", DEFAULT_MAX_WAITING),
    }


# ---------------------------------------------------------------------------
# Load limits
# ---------------------------------------------------------------------------

DEFAULT_MAX_DOCUMENTS_PER_USER = 2
DEFAULT_MAX_DOCUMENTS = 8
# Per uvicorn worker: 8 prod workers x 2 = 16 conversions at most, half the
# gateway's 32 cores.
DEFAULT_MAX_CONVERSIONS = 2
DEFAULT_QUEUE_WAIT_SECONDS = 60
DEFAULT_MAX_WAITING_PER_USER = 4
DEFAULT_MAX_WAITING = 16

# Bounds on what a caller may ask for. A naive client asking for 1200 dpi or
# 200-page chunks would otherwise make every conversion enormous.
CLIENT_DPI_RANGE = (72, 300)
CLIENT_CHUNK_SIZE_RANGE = (1, 20)
CLIENT_OVERLAP_RANGE = (0, 10)

# Seconds to wait before retrying a chunk whose request failed (doubles per
# attempt). Immediate retries multiplied the load in the 2026-11-05 outage.
CHUNK_RETRY_BACKOFF_SECONDS = 2.0


class OcrCancelled(Exception):
    """The client went away; the rest of the document is not worth doing."""


class _ConversionLimiter:
    """Caps page-image conversions running at once in this worker.

    Turning PDF pages into images (pdftoppm, LibreOffice, PIL) runs on the
    gateway's own CPUs. Without a cap, every chunk of every document converted
    at once: 39 pdftoppm processes and load ~90 on 32 cores in the 2026-11-05
    outage, which starved the event loop until database writes timed out. The
    limit is re-read from admin config on each document, so it can change
    without a restart.

    A FIFO of futures rather than an asyncio.Condition: releasing is fully
    synchronous (no lock to await, so a cancellation during release can never
    lose a slot), and a freed slot is handed straight to the next waiter (no
    wake-up can be lost if that waiter is cancelled — it hands it on).
    """

    def __init__(self, limit: int):
        self._limit = max(1, int(limit))
        self._active = 0
        self._waiters: "collections.deque[asyncio.Future]" = collections.deque()

    def set_limit(self, limit) -> None:
        try:
            new = max(1, int(limit))
        except (TypeError, ValueError):
            return
        if new != self._limit:
            self._limit = new
            self._wake()

    @property
    def active(self) -> int:
        return self._active

    def _wake(self) -> None:
        while self._waiters and self._active < self._limit:
            fut = self._waiters.popleft()
            if not fut.done():
                self._active += 1  # the slot passes to this waiter
                fut.set_result(None)

    async def __aenter__(self):
        if self._active < self._limit and not self._waiters:
            self._active += 1
            return self
        fut = asyncio.get_running_loop().create_future()
        self._waiters.append(fut)
        try:
            await fut
        except asyncio.CancelledError:
            if fut.done() and not fut.cancelled():
                # We were handed a slot just as we were cancelled: pass it on.
                self._active -= 1
                self._wake()
            else:
                with contextlib.suppress(ValueError):
                    self._waiters.remove(fut)
            raise
        return self

    async def __aexit__(self, *exc):
        self._active -= 1
        self._wake()
        return False


_conversions = _ConversionLimiter(DEFAULT_MAX_CONVERSIONS)


async def _convert_in_thread(fn, *args):
    """Run a blocking conversion in a thread, under the conversion limit.

    The limit is held until the THREAD finishes, not just until the awaiting
    task stops waiting: cancelling a task cannot stop pdftoppm, so releasing
    on cancellation would let new conversions start while the old ones still
    run — the limit would count tasks, not processes.
    """
    async with _conversions:
        # copy_context: keep request_id/user_id on log lines from the thread,
        # as asyncio.to_thread did.
        ctx = contextvars.copy_context()
        fut = asyncio.get_running_loop().run_in_executor(
            None, functools.partial(ctx.run, fn, *args),
        )
        cancelled = False
        while not fut.done():
            try:
                await asyncio.shield(fut)
            except asyncio.CancelledError:
                cancelled = True
            except BaseException:
                break
        if cancelled:
            raise asyncio.CancelledError()
        return fut.result()


def validate_client_params(dpi=None, chunk_size=None, overlap=None) -> None:
    """Reject caller-supplied conversion settings outside sane bounds.

    Only values the caller actually sent are checked (None = not sent); the
    admin's own defaults are bounded by the settings form instead.
    """
    for name, value, (lo, hi) in (
        ("dpi", dpi, CLIENT_DPI_RANGE),
        ("chunk_size", chunk_size, CLIENT_CHUNK_SIZE_RANGE),
        ("overlap", overlap, CLIENT_OVERLAP_RANGE),
    ):
        if value is None:
            continue
        if not isinstance(value, int) or not lo <= value <= hi:
            raise ValueError(f"{name} must be between {lo} and {hi}")


# ---------------------------------------------------------------------------
# Document → page images
# ---------------------------------------------------------------------------

def _image_to_png_bytes(img: Image.Image, max_dim: int = 2048) -> bytes:
    """Convert a PIL Image to PNG bytes, optionally downscaling."""
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# An upload that is not a readable document is the client's problem: answer
# 400 with this (ValueError, which the OCR endpoints map to 400), not a 500.
# pdf2image raises PDFPageCountError for anything pdfinfo cannot read (not a
# PDF, truncated, empty, password-protected); PIL raises OSError subclasses
# (UnidentifiedImageError, "image file is truncated", "broken data stream")
# for an image it cannot decode, and DecompressionBombError for one whose
# declared size is absurd.
UNREADABLE_PDF_MESSAGE = (
    "The uploaded file could not be read as a PDF. It may be corrupt, "
    "password-protected, or not a PDF."
)
UNREADABLE_IMAGE_MESSAGE = (
    "The uploaded file could not be read as an image. It may be corrupt or "
    "not an image."
)
OVERSIZED_IMAGE_MESSAGE = (
    "The uploaded image is too large to process (too many pixels)."
)


class OcrConversionError(Exception):
    """Page conversion failed on our side (poppler killed or crashed): a
    500, unlike an unreadable upload (ValueError, 400)."""


# pdfinfo's exit status says whose fault a failure is.  Poppler exits 1 when
# it cannot open the document: not a PDF, empty, truncated, wrong password.
# A process killed by a signal (the OOM killer; poppler's own "Out of memory"
# abort) exits negative, a missing shared library exits 127, and poppler
# prints warnings even for files it then reads fine -- so stderr alone cannot
# tell a bad file from a fault of ours, but the exit status can.
_PDFINFO_TIMEOUT_SECONDS = 60


def _pdfinfo_page_count(pdf_bytes: bytes) -> int:
    """Pages in a PDF, via pdfinfo.  Synchronous: run in a thread.

    Raises ValueError (a 400) only when poppler rejected the file itself;
    any failure of our own raises OcrConversionError (a 500).
    """
    with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
        f.write(pdf_bytes)
        f.flush()
        proc = subprocess.run(
            ["pdfinfo", f.name], capture_output=True,
            timeout=_PDFINFO_TIMEOUT_SECONDS,
        )
    stderr = proc.stderr.decode("utf-8", "replace").strip()
    if proc.returncode == 0:
        for line in proc.stdout.decode("utf-8", "replace").splitlines():
            key, _, value = line.partition(":")
            if key.strip() == "Pages":
                return int(value.strip())
        raise OcrConversionError("pdfinfo reported no page count")
    if proc.returncode == 1 and stderr and "out of memory" not in stderr.lower():
        raise _unreadable(UNREADABLE_PDF_MESSAGE, RuntimeError(stderr))
    raise OcrConversionError(
        f"pdfinfo failed (exit {proc.returncode}): {stderr[:300]}"
    )


def _unreadable(message: str, err: Exception) -> ValueError:
    # Poppler's own words stay in the log; the client gets a plain sentence.
    logger.info("ocr_unreadable_upload", reason=message, error=str(err)[:300])
    return ValueError(message)


def _image_bytes_to_pages(file_bytes: bytes, max_frames: int) -> List[bytes]:
    """Decode an image (possibly multi-frame) to a list of PNG byte buffers.

    Synchronous — call via ``asyncio.to_thread``. The frame walk stops after
    ``max_frames`` so a crafted animated GIF / multi-page TIFF cannot force
    unbounded decoding.
    """
    # Image.open reads only the header; a damaged file fails later, while
    # seek/copy/convert decode it, so the whole walk is guarded.  All I/O is
    # on an in-memory buffer, so an OSError here is the file's -- except
    # PIL's "out of memory" decoder error, which is ours and stays a 500.
    # Corrupt TIFF tags surface as KeyError/TypeError/IndexError/struct.error
    # from the plugin's frame setup (as Image.open itself treats them).
    try:
        img = Image.open(io.BytesIO(file_bytes))
        pages: List[bytes] = []
        try:
            frame = 0
            while True:
                if frame >= max_frames:
                    logger.warning(
                        "ocr_image_frame_cap",
                        max_frames=max_frames,
                    )
                    break
                img.seek(frame)
                pages.append(_image_to_png_bytes(img.copy()))
                frame += 1
        except EOFError:
            pass
        if not pages:
            pages.append(_image_to_png_bytes(img))
        return pages
    except Image.DecompressionBombError as e:
        raise _unreadable(OVERSIZED_IMAGE_MESSAGE, e) from e
    except OSError as e:
        if "out of memory" in str(e).lower():
            raise
        raise _unreadable(UNREADABLE_IMAGE_MESSAGE, e) from e
    except (SyntaxError, KeyError, TypeError, IndexError, ZeroDivisionError,
            struct.error) as e:
        raise _unreadable(UNREADABLE_IMAGE_MESSAGE, e) from e


async def document_to_images(
    file_bytes: bytes,
    content_type: str,
    filename: str,
    dpi: int = 200,
) -> List[bytes]:
    """
    Convert a document to a list of PNG byte buffers (one per page).

    Supports: images, PDFs, DOCX, PPTX, XLSX.
    Office formats are converted via LibreOffice headless → PDF → images.
    """
    # Images: single page (or multi-frame GIF/TIFF). Decoding runs off the
    # event loop like the PDF branch, and the frame count is bounded so a
    # hostile many-frame image can't pin a CPU decoding unbounded frames.
    if content_type.startswith("image/"):
        max_frames = get_settings().ocr_max_frames
        return await _convert_in_thread(_image_bytes_to_pages, file_bytes, max_frames)

    # PDF
    if content_type == "application/pdf":
        return await _pdf_to_images(file_bytes, dpi)

    # Office formats → PDF → images
    office_types = {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
        "application/msword": ".doc",
        "application/vnd.ms-powerpoint": ".ppt",
        "application/vnd.ms-excel": ".xls",
    }
    suffix = office_types.get(content_type)
    if suffix is None:
        # Try to infer from filename extension
        ext = Path(filename).suffix.lower()
        if ext in (".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls"):
            suffix = ext
    if suffix:
        pdf_bytes = await _office_to_pdf(file_bytes, suffix)
        return await _pdf_to_images(pdf_bytes, dpi)

    raise ValueError(f"Unsupported content type: {content_type}")


async def _pdf_to_images(pdf_bytes: bytes, dpi: int = 200) -> List[bytes]:
    """Convert PDF bytes to a list of PNG byte buffers using pdf2image.

    Not reached by perform_ocr, which sends PDFs and Office files down the
    pipelined path (page count first, then ranges); kept for
    document_to_images' other callers.
    """
    from pdf2image import convert_from_bytes

    def _convert():
        pil_images = convert_from_bytes(pdf_bytes, dpi=dpi, fmt="png")
        return [_image_to_png_bytes(img) for img in pil_images]

    return await _convert_in_thread(_convert)


async def _pdf_page_count(pdf_bytes: bytes) -> int:
    """Get the number of pages in a PDF without converting to images.

    Runs before any pdftoppm process starts for the document, and is where
    an unreadable PDF is turned away (see _pdfinfo_page_count).
    """
    return await asyncio.to_thread(_pdfinfo_page_count, pdf_bytes)


async def _pdf_to_images_range(
    pdf_bytes: bytes, first_page: int, last_page: int, dpi: int = 200
) -> List[bytes]:
    """Convert a range of PDF pages to PNG byte buffers (1-indexed)."""
    from pdf2image import convert_from_bytes

    def _convert():
        # Not translated: the pipeline counted the pages of these same bytes
        # first, so a failure here is poppler's (killed, crashed), not the
        # file's.  A pdftoppm that died returns no pages at all: fail the
        # chunk rather than send the model an empty page set.
        pil_images = convert_from_bytes(
            pdf_bytes, dpi=dpi, fmt="png",
            first_page=first_page, last_page=last_page,
        )
        # A pdftoppm killed part way returns the pages it finished: fail the
        # chunk rather than send the model fewer pages than it is told.
        # Callers clamp the range to the page count, so every page exists.
        expected = last_page - first_page + 1
        if len(pil_images) != expected:
            raise OcrConversionError(
                f"PDF page conversion produced {len(pil_images)} of {expected} "
                f"images for pages {first_page}-{last_page}"
            )
        return [_image_to_png_bytes(img) for img in pil_images]

    return await _convert_in_thread(_convert)


async def _office_to_pdf(file_bytes: bytes, suffix: str) -> bytes:
    """Convert Office document to PDF via LibreOffice headless."""
    async with _conversions:
        return await _office_to_pdf_unlimited(file_bytes, suffix)


async def _office_to_pdf_unlimited(file_bytes: bytes, suffix: str) -> bytes:
    """The LibreOffice conversion itself; _office_to_pdf holds the limit."""
    lo_path = shutil.which("libreoffice") or shutil.which("soffice")
    if not lo_path:
        raise RuntimeError(
            "LibreOffice is not installed. Office document OCR requires "
            "'libreoffice' or 'soffice' on PATH."
        )

    with tempfile.TemporaryDirectory(prefix="ocr_lo_") as tmpdir:
        input_path = Path(tmpdir) / f"input{suffix}"
        input_path.write_bytes(file_bytes)

        # Isolate the user profile per invocation so concurrent conversions
        # don't share (and corrupt) a single LibreOffice profile, and so a
        # hostile document can't poison a persistent profile. --norestore
        # stops LibreOffice from trying to recover a "crashed" prior session.
        user_install = f"file://{tmpdir}/lo_profile_{uuid.uuid4().hex}"

        proc = await asyncio.create_subprocess_exec(
            lo_path,
            "--headless",
            "--norestore",
            f"-env:UserInstallation={user_install}",
            "--convert-to", "pdf",
            "--outdir", tmpdir,
            str(input_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Bound the conversion: a malformed/hostile document can otherwise
        # hang LibreOffice indefinitely. Kill the process group on timeout.
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()
            raise RuntimeError("LibreOffice conversion timed out")
        if proc.returncode != 0:
            raise RuntimeError(
                f"LibreOffice conversion failed (exit {proc.returncode}): "
                f"{stderr.decode(errors='replace')}"
            )

        pdf_path = input_path.with_suffix(".pdf")
        if not pdf_path.exists():
            # Sometimes LibreOffice names it differently
            pdfs = list(Path(tmpdir).glob("*.pdf"))
            if not pdfs:
                raise RuntimeError("LibreOffice produced no PDF output")
            pdf_path = pdfs[0]

        return pdf_path.read_bytes()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def make_chunks(
    num_pages: int, chunk_size: int = 6, overlap: int = 2
) -> List[Tuple[int, int]]:
    """
    Generate overlapping chunk ranges for page indices.

    For num_pages=12, chunk_size=6, overlap=2:
      chunk 0: pages 0-5
      chunk 1: pages 4-9   (overlaps 4-5 with chunk 0)
      chunk 2: pages 8-11  (overlaps 8-9 with chunk 1)
    """
    stride = chunk_size - overlap
    chunks = []
    i = 0
    while i < num_pages:
        end = min(i + chunk_size, num_pages)
        chunks.append((i, end))
        if end >= num_pages:
            break
        i += stride
    return chunks


# ---------------------------------------------------------------------------
# Single-chunk OCR via chat completion
# ---------------------------------------------------------------------------

def _strip_fences(text: str) -> str:
    """Remove markdown code fences the model sometimes wraps output in."""
    text = text.strip()
    if text.startswith("```markdown"):
        text = text[len("```markdown"):].strip()
    if text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


def _effective_max_tokens(ocr_config: dict, num_pages: int) -> int:
    """Completion budget for one request: the per-page allowance times the
    pages it carries, never above the absolute ceiling. A model that loops is
    then bounded by the page count rather than by a document-wide ceiling."""
    ceiling = int(ocr_config.get("max_tokens", 16384) or 16384)
    per_page = int(ocr_config.get("max_tokens_per_page", 0) or 0)
    if per_page <= 0:
        return ceiling
    return max(1, min(ceiling, per_page * max(1, num_pages)))


# A looping generation is recognised by a periodic tail: the last chars recur
# exactly one period earlier, that period tiles the tail at least this many
# times, and the run covers at least this fraction of the text.
_RUNAWAY_MIN_UNIT = 40
_RUNAWAY_MIN_REPEATS = 3
_RUNAWAY_MIN_FRACTION = 0.5


def collapse_runaway_repetition(text: str) -> Tuple[str, bool]:
    """Collapse a generation that looped on itself to one copy of the loop.

    Meant for a completion that hit its token cap: an OCR specialist handed
    a prompt shape it was not trained on can re-emit the same block until the
    budget runs out (dots.MOCR did this 349 times on a three-line page). The
    period is found from the tail, then extended backwards as far as the text
    stays periodic; everything before the run is kept, plus one full period.

    Ordinary content is untouched: a real document with repeated table rows
    neither tiles half its length with exact copies nor, being called only
    on capped completions, reaches here in the first place.

    The seam between the real content and the loop is ambiguous when both end
    in the same character (a newline, say): the cut may land one such
    character earlier. The kept text is always a prefix of the original.
    """
    n = len(text)
    if n < _RUNAWAY_MIN_UNIT * _RUNAWAY_MIN_REPEATS:
        return text, False
    tail = text[-_RUNAWAY_MIN_UNIT:]
    # Search up to the last character so an occurrence that OVERLAPS the tail
    # counts: for a loop unit shorter than the tail, the nearest earlier copy
    # ends inside the tail, and bounding the search at the tail's start would
    # skip it and report a doubled period (two copies kept instead of one).
    prev = text.rfind(tail, 0, n - 1)
    if prev < 0:
        return text, False
    period = (n - _RUNAWAY_MIN_UNIT) - prev
    if period <= 0:
        return text, False
    # Walk back while the text stays periodic with that period.
    start = n - period
    while start > 0 and text[start - 1] == text[start - 1 + period]:
        start -= 1
    run = n - start
    if run < _RUNAWAY_MIN_REPEATS * period or run < _RUNAWAY_MIN_FRACTION * n:
        return text, False
    return text[:start + period], True


def _pages_json(page_texts: List[str]) -> str:
    """The /v1/ocr JSON body, built by the pipeline rather than by the model.

    Asking the model to author this structure is what broke dots.MOCR; the
    pipeline already knows which page each transcription came from.
    """
    return json.dumps(
        {"pages": [{"page_number": i + 1, "content": t} for i, t in enumerate(page_texts)]},
        ensure_ascii=False,
    )


def _guard_runaway(text: str, result: dict, budget: int, *, model: str,
                   chunk: Optional[int] = None) -> Tuple[str, bool]:
    """Collapse a looping generation, but only one that hit its budget."""
    choice = (result.get("choices") or [{}])[0]
    completion = (result.get("usage") or {}).get("completion_tokens", 0) or 0
    hit_cap = choice.get("finish_reason") == "length" or completion >= budget
    if not hit_cap:
        return text, False
    kept, collapsed = collapse_runaway_repetition(text)
    if collapsed:
        logger.warning(
            "ocr_runaway_collapsed",
            model=model,
            chunk=chunk,
            budget=budget,
            chars_before=len(text),
            chars_after=len(kept),
        )
    return kept, collapsed


def _build_ocr_prompt(
    num_pages: int,
    prompt_template: str,
    is_retry: bool = False,
    prev_length: int = 0,
) -> str:
    """Build the OCR prompt from an admin-configurable template.

    The template may contain ``{num_pages}`` which is substituted at
    call time.  On retry, a prefix is prepended urging completeness.
    """
    prompt = prompt_template.format(num_pages=num_pages)

    if is_retry:
        retry_prefix = (
            f"IMPORTANT: You MUST convert ALL {num_pages} page images below. "
            f"Your previous attempt only produced {prev_length} characters which "
            "is too short. Every single page must be fully transcribed. "
            "Convert ALL text, tables, headers, and content from EVERY page image. "
            "Do not stop early. Do not skip any pages. "
        )
        return retry_prefix + prompt

    return prompt


def _is_image_limit_error(e: Exception) -> bool:
    """True if a backend rejected a multi-image request — a single-page OCR
    specialist, or vLLM ``--limit-mm-per-prompt`` below the chunk size."""
    msg = str(getattr(e, "detail", "") or e).lower()
    is_400 = "400" in msg or getattr(e, "status_code", None) == 400
    return is_400 and any(
        k in msg for k in ("multimodal", "image", "at most", "limit-mm", "mm_per_prompt")
    )


def _is_retryable(e: Exception) -> bool:
    """Whether a failed chunk request is worth retrying.

    A 4xx says the request itself is refused — over the caller's rate limit
    (429), over quota, invalid — and the same request will be refused again.
    Retrying those at once is what multiplied a rate-limited client's load in
    the 2026-11-05 outage. Server errors and timeouts may be transient.
    """
    code = getattr(e, "status_code", None)
    return not (isinstance(code, int) and 400 <= code < 500)


async def _should_stop(stop: Optional[asyncio.Event], http_request) -> bool:
    """Whether a chunk should not start (or retry): the document already
    failed, or the client went away."""
    return bool(stop is not None and stop.is_set()) or await _client_gone(http_request)


async def _client_gone(http_request) -> bool:
    """True once the HTTP client that asked for this document has gone away."""
    is_disconnected = getattr(http_request, "is_disconnected", None)
    if is_disconnected is None:
        return False
    try:
        return bool(await is_disconnected())
    except Exception:
        return False


async def _run_chunks(coros, stop: asyncio.Event) -> list:
    """Run a document's chunks concurrently; the first failure stops the rest.

    asyncio.gather alone raises the first error but leaves the other chunks
    running, so a failed or abandoned document kept converting pages and
    calling the model for nobody. Stopping is cooperative: ``stop`` is set and
    each chunk checks it before converting, before inference and before a
    retry. Chunks are NOT cancelled mid-flight — a chunk inside its
    completion's accounting write must finish it before its DB session
    closes, and a cancelled conversion would leave pdftoppm running unseen.
    So this returns (or raises) only once every chunk has actually finished.
    """
    tasks = [asyncio.ensure_future(c) for c in coros]
    first_error: Optional[BaseException] = None
    pending = set(tasks)
    try:
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_EXCEPTION)
            for t in done:
                exc = t.exception()
                if exc is None:
                    continue
                stop.set()
                # Prefer the real cause over the OcrCancelled it triggered.
                if first_error is None or isinstance(first_error, OcrCancelled):
                    first_error = exc
    except asyncio.CancelledError:
        # The request itself was cancelled: stop the chunks and let them
        # wind down before the document's slot is released.
        stop.set()
        await asyncio.shield(asyncio.gather(*tasks, return_exceptions=True))
        raise
    if first_error is not None:
        raise first_error
    return [t.result() for t in tasks]


async def _ocr_pages_individually(
    service, page_images: List[bytes], prompt_template: str, model: str,
    ocr_config: dict, user, api_key, http_request,
    stop: Optional[asyncio.Event] = None,
) -> Tuple[str, Dict[str, int], bool]:
    """Fallback OCR: one page per request, concatenated. Used when a backend
    cannot accept multiple images in a single request. The bool says whether
    any page's generation had to be collapsed as a runaway."""
    parts: List[str] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    degraded = False
    budget = _effective_max_tokens(ocr_config, 1)
    for img_bytes in page_images:
        if await _should_stop(stop, http_request):
            raise OcrCancelled("document stopped")
        b64 = base64.b64encode(img_bytes).decode()
        canonical = CanonicalChatRequest(
            model=model,
            messages=[CanonicalMessage(role="user", content=[
                TextContent(text=_build_ocr_prompt(1, prompt_template)),
                ImageUrlContent(image_url={"url": f"data:image/png;base64,{b64}"}),
            ])],
            temperature=ocr_config["temperature"],
            max_tokens=budget,
            stream=False,
            think=False,
        )
        result = await service.chat_completion(
            canonical, user, api_key, http_request, skip_rpm_check=True,
        )
        u = result.get("usage", {})
        for k in usage:
            usage[k] += u.get(k, 0)
        text = _strip_fences(
            result.get("choices", [{}])[0].get("message", {}).get("content", "")
        )
        text, collapsed = _guard_runaway(text, result, budget, model=model)
        degraded = degraded or collapsed
        parts.append(text)
    return "\n\n".join(parts), usage, degraded


async def ocr_chunk(
    page_images: List[bytes],
    chunk_idx: int,
    start_page: int,
    end_page: int,
    total_pages: int,
    total_chunks: int,
    prompt_template: str,
    model: str,
    ocr_config: dict,
    user: "User",
    api_key: "ApiKey",
    http_request: "Request",
    stop: Optional[asyncio.Event] = None,
) -> Tuple[int, str, Dict[str, int], bool]:
    """
    Send a chunk of page images to the LLM and return OCR text.

    Each chunk gets its own DB session and InferenceService to avoid
    session state conflicts when multiple chunks run concurrently.

    Returns (chunk_idx, text, usage_dict, degraded) — degraded is True when
    the generation hit its token budget looping and was collapsed.
    """
    from backend.app.db.session import AsyncSessionLocal
    from backend.app.services.inference import InferenceService

    num_pages = end_page - start_page
    max_retries = ocr_config["max_retries"]

    budget = _effective_max_tokens(ocr_config, num_pages)

    # Build content blocks: prompt + images
    prompt = _build_ocr_prompt(num_pages, prompt_template)
    content_blocks: List[Any] = [TextContent(text=prompt)]
    for img_bytes in page_images:
        b64 = base64.b64encode(img_bytes).decode()
        content_blocks.append(
            ImageUrlContent(image_url={"url": f"data:image/png;base64,{b64}"})
        )

    canonical = CanonicalChatRequest(
        model=model,
        messages=[CanonicalMessage(role="user", content=content_blocks)],
        temperature=ocr_config["temperature"],
        max_tokens=budget,
        stream=False,
        think=False,
    )

    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    expected_min_chars = ocr_config["min_chars_per_page"] * num_pages

    # Each chunk gets its own DB session to avoid "prepared state" errors
    # when multiple chunks are processed concurrently.
    async with AsyncSessionLocal() as chunk_db:
        service = InferenceService(chunk_db)

        for attempt in range(max_retries + 1):
            try:
                # The document was admitted as a whole; its chunks are not
                # separate requests against the per-minute limit.
                result = await service.chat_completion(
                    canonical, user, api_key, http_request, skip_rpm_check=True,
                )
            except Exception as e:
                logger.warning(
                    "ocr_chunk_error",
                    chunk=chunk_idx,
                    attempt=attempt,
                    error=str(e),
                )
                # If the backend can't take a multi-image request, degrade to one
                # page per request rather than retrying the same failing payload.
                if num_pages > 1 and _is_image_limit_error(e):
                    logger.info(
                        "ocr_chunk_degrade_single_page",
                        chunk=chunk_idx, pages=num_pages,
                    )
                    await chunk_db.rollback()
                    text, deg_usage, degraded = await _ocr_pages_individually(
                        service, page_images, prompt_template, model,
                        ocr_config, user, api_key, http_request, stop=stop,
                    )
                    for k in total_usage:
                        total_usage[k] += deg_usage.get(k, 0)
                    return chunk_idx, text, total_usage, degraded
                if attempt == max_retries or not _is_retryable(e):
                    raise
                # Reset session state for retry, and give the server a moment:
                # retrying at once only adds load to whatever just failed.
                await chunk_db.rollback()
                await asyncio.sleep(CHUNK_RETRY_BACKOFF_SECONDS * (2 ** attempt))
                if await _should_stop(stop, http_request):
                    raise OcrCancelled("document stopped")
                continue

            # Accumulate usage
            usage = result.get("usage", {})
            for k in total_usage:
                total_usage[k] += usage.get(k, 0)

            text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            text = _strip_fences(text)
            text, collapsed = _guard_runaway(
                text, result, budget, model=model, chunk=chunk_idx + 1,
            )

            logger.info(
                "ocr_chunk_result",
                chunk=chunk_idx + 1,
                total_chunks=total_chunks,
                pages=f"{start_page + 1}-{end_page}",
                chars=len(text),
                attempt=attempt + 1,
            )

            # A collapsed runaway is never retried: the "you stopped early"
            # prompt would only make a looping model loop again at full cost.
            if collapsed or len(text) >= expected_min_chars or attempt == max_retries:
                return chunk_idx, text, total_usage, collapsed

            # Retry with stronger prompt
            logger.info(
                "ocr_chunk_retry",
                chunk=chunk_idx + 1,
                chars=len(text),
                expected=expected_min_chars,
            )
            retry_prompt = _build_ocr_prompt(
                num_pages, prompt_template, is_retry=True, prev_length=len(text)
            )
            content_blocks[0] = TextContent(text=retry_prompt)
            canonical.messages[0].content = content_blocks

    # Unreachable, but satisfy type checker
    return chunk_idx, "", total_usage, False


# ---------------------------------------------------------------------------
# Deterministic chunk merging (adapted from chunked_ocr.py)
# ---------------------------------------------------------------------------

def _normalize_line(line: str) -> str:
    """Normalize a line for fuzzy comparison."""
    line = line.strip().lower()
    line = re.sub(r"\s+", " ", line)
    return line


def _merge_nearby_blocks(blocks, gap: int = 5):
    """Merge matching blocks within `gap` lines of each other."""
    if not blocks:
        return []
    sorted_blocks = sorted(blocks, key=lambda b: b.a)
    merged = []
    cur_a, cur_b, cur_size = sorted_blocks[0].a, sorted_blocks[0].b, sorted_blocks[0].size

    for i in range(1, len(sorted_blocks)):
        blk = sorted_blocks[i]
        a_end = cur_a + cur_size
        b_end = cur_b + cur_size
        if blk.a <= a_end + gap and blk.b <= b_end + gap:
            new_a_end = blk.a + blk.size
            new_b_end = blk.b + blk.size
            cur_size = max(new_a_end - cur_a, new_b_end - cur_b)
        else:
            merged.append((cur_a, cur_b, cur_size))
            cur_a, cur_b, cur_size = blk.a, blk.b, blk.size

    merged.append((cur_a, cur_b, cur_size))
    return merged


def find_overlap_boundary(
    chunk_a: str, chunk_b: str, overlap_fraction: float = 0.4
) -> Tuple[int, int]:
    """
    Find where chunk_a and chunk_b overlap using sequence matching.

    Returns (a_cut, b_cut): line indices for splicing.
    """
    lines_a = chunk_a.split("\n")
    lines_b = chunk_b.split("\n")

    search_a_start = max(0, int(len(lines_a) * (1.0 - overlap_fraction)))
    search_b_end = min(len(lines_b), int(len(lines_b) * overlap_fraction))

    tail_a = [_normalize_line(l) for l in lines_a[search_a_start:]]
    head_b = [_normalize_line(l) for l in lines_b[:search_b_end]]

    sm = difflib.SequenceMatcher(
        isjunk=lambda x: x.strip() == "",
        a=tail_a,
        b=head_b,
        autojunk=False,
    )

    blocks = [b for b in sm.get_matching_blocks() if b.size > 0]

    if not blocks:
        return len(lines_a), 0

    blocks.sort(key=lambda b: b.size, reverse=True)
    best = blocks[0]

    min_reliable = 8
    if best.size < min_reliable:
        merged_blocks = _merge_nearby_blocks(blocks, gap=5)
        if merged_blocks:
            merged_blocks.sort(key=lambda b: b[2], reverse=True)
            best_merged = merged_blocks[0]
            if best_merged[2] >= min_reliable:
                a_cut = search_a_start + best_merged[0] + best_merged[2]
                b_cut = best_merged[1] + best_merged[2]
                return a_cut, b_cut

    a_cut = search_a_start + best.a + best.size
    b_cut = best.b + best.size
    return a_cut, b_cut


def merge_chunks(chunks_text: List[str]) -> str:
    """
    Merge overlapping chunks deterministically via sequence matching.
    No LLM calls needed.
    """
    if len(chunks_text) == 1:
        return chunks_text[0]

    merged_lines = chunks_text[0].split("\n")

    for i in range(1, len(chunks_text)):
        current = "\n".join(merged_lines)
        next_chunk = chunks_text[i]

        a_cut, b_cut = find_overlap_boundary(current, next_chunk)

        current_lines = current.split("\n")
        next_lines = next_chunk.split("\n")

        a_part = current_lines[:a_cut]
        while a_part and a_part[-1].strip() == "":
            a_part.pop()

        b_part = next_lines[b_cut:]
        while b_part and b_part[0].strip() == "":
            b_part.pop(0)

        merged_lines = a_part + [""] + b_part

    return "\n".join(merged_lines)


def failure_note(err: Exception) -> str:
    """What to tell a chat user whose scanned PDF could not be OCR'd.

    perform_ocr raises ValueError for the document itself (no pages, more
    than ``ocr.max_pages``): the same upload fails the same way every time,
    so say what is wrong.  Anything else (no healthy backend, a timeout, the
    service busy) may pass, so suggest trying again.
    """
    if isinstance(err, ValueError):
        return (
            f"This PDF appears to be scanned, but OCR could not process it ({err}), "
            "so little or no text could be extracted."
        )
    return (
        "This PDF appears to be scanned, but OCR is unavailable right now, "
        "so little or no text could be extracted. Try again later."
    )


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

async def perform_ocr(
    file_bytes: bytes,
    content_type: str,
    filename: str,
    model: str,
    output_format: str,
    chunk_size: int,
    overlap: int,
    dpi: int,
    ocr_config: dict,
    user: "User",
    api_key: "ApiKey",
    http_request: "Request",
    prompt_template: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one document through OCR, within the load limits.

    A document over the in-progress limits waits for a slot (bounded); if it
    cannot get one, ocr_admission.OcrBusy is raised before any work. Raises
    OcrCancelled if the client disconnects while waiting or part way through.
    """
    _conversions.set_limit(
        ocr_config.get("max_concurrent_conversions", DEFAULT_MAX_CONVERSIONS)
    )
    args = dict(
        file_bytes=file_bytes, content_type=content_type, filename=filename,
        model=model, output_format=output_format, chunk_size=chunk_size,
        overlap=overlap, dpi=dpi, ocr_config=ocr_config, user=user,
        api_key=api_key, http_request=http_request,
        prompt_template=prompt_template,
    )
    if user is None:
        return await _perform_ocr_admitted(**args)

    from backend.app.services.ocr_admission import WaitAbandoned, document_slot

    async def _gone() -> bool:
        return await _client_gone(http_request)

    try:
        async with document_slot(
            user.id,
            per_user=ocr_config.get("max_documents_per_user", DEFAULT_MAX_DOCUMENTS_PER_USER),
            global_limit=ocr_config.get("max_documents", DEFAULT_MAX_DOCUMENTS),
            max_wait_seconds=ocr_config.get("queue_wait_seconds", DEFAULT_QUEUE_WAIT_SECONDS),
            max_waiting_per_user=ocr_config.get("max_waiting_per_user", DEFAULT_MAX_WAITING_PER_USER),
            max_waiting=ocr_config.get("max_waiting", DEFAULT_MAX_WAITING),
            should_abort=_gone,
        ):
            return await _perform_ocr_admitted(**args)
    except WaitAbandoned:
        raise OcrCancelled("client disconnected while waiting for a slot")


async def _perform_ocr_admitted(
    file_bytes: bytes,
    content_type: str,
    filename: str,
    model: str,
    output_format: str,
    chunk_size: int,
    overlap: int,
    dpi: int,
    ocr_config: dict,
    user: "User",
    api_key: "ApiKey",
    http_request: "Request",
    prompt_template: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Full OCR pipeline: document → images → chunked LLM OCR → merge.

    For PDFs, uses a pipelined approach: converts page ranges in parallel
    and fires OCR chunks as soon as their pages are ready, overlapping
    conversion with inference. For images and small documents, falls back
    to the simpler convert-all-then-OCR path.

    Returns dict with content, pages, chunks_processed, usage.
    """
    import time as _time

    from backend.app.core.telemetry.registry import get_registry

    t_total_start = _time.monotonic()
    is_pdf = (content_type == "application/pdf")

    # Routing looks backends up by the real model name, so an alias that
    # reaches the chunks unresolved 404s every one of them.  Prod's OCR model
    # is the alias `default-ocr`: /v1/ocr resolved it before calling here, but
    # the dashboard chat's scanned-PDF path passed it through, and every
    # scanned PDF uploaded in chat failed OCR ("Model 'default-ocr' not
    # found").  Resolving here covers every caller; a real name is unchanged.
    model, _ = get_registry().resolve_alias(model)

    # For Office docs, convert to PDF first (then treat as PDF)
    office_types = {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
        "application/msword": ".doc",
        "application/vnd.ms-powerpoint": ".ppt",
        "application/vnd.ms-excel": ".xls",
    }
    suffix = office_types.get(content_type)
    if suffix is None and not is_pdf and not content_type.startswith("image/"):
        ext = Path(filename).suffix.lower()
        if ext in (".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls"):
            suffix = ext
    if suffix:
        t0 = _time.monotonic()
        file_bytes = await _office_to_pdf(file_bytes, suffix)
        is_pdf = True
        logger.info("ocr_office_convert", ms=round((_time.monotonic() - t0) * 1000))

    # The model only ever transcribes; the pipeline owns the structure. JSON
    # output used to be requested FROM the model with a page/table schema,
    # which an OCR specialist trained on a few fixed prompts cannot follow —
    # dots.MOCR re-emitted the page block until the token budget ran out. Now
    # both formats use the transcription prompt, and JSON mode runs one page
    # per request so each transcription is attributed to its page exactly,
    # with no overlap merge to reconcile.
    if output_format == "json":
        chunk_size, overlap = 1, 0
    if prompt_template is None:
        prompt_template = ocr_config.get("prompt_ocrmd", _DEFAULT_PROMPT_OCRMD)

    # For PDFs with multiple pages, use pipelined conversion + inference
    if is_pdf:
        return await _perform_ocr_pipelined(
            file_bytes, model, output_format, chunk_size, overlap, dpi,
            ocr_config, user, api_key, http_request, t_total_start,
            prompt_template,
        )

    # For images: simple path (single page, no chunking needed usually)
    t0 = _time.monotonic()
    page_images = await document_to_images(file_bytes, content_type, filename, dpi)
    t_convert = _time.monotonic() - t0

    return await _perform_ocr_simple(
        page_images, model, output_format, chunk_size, overlap,
        ocr_config, user, api_key, http_request, t_total_start, t_convert,
        prompt_template,
    )


async def _perform_ocr_pipelined(
    pdf_bytes: bytes,
    model: str,
    output_format: str,
    chunk_size: int,
    overlap: int,
    dpi: int,
    ocr_config: dict,
    user: "User",
    api_key: "ApiKey",
    http_request: "Request",
    t_total_start: float,
    prompt_template: str,
) -> Dict[str, Any]:
    """
    Pipelined OCR for PDFs: convert page ranges and run inference concurrently.

    Each chunk's pages are converted independently and in parallel. As soon
    as a chunk's pages are ready, its OCR inference fires — no waiting for
    the entire document to be converted first.
    """
    import time as _time

    # Get page count without converting (fast)
    t0 = _time.monotonic()
    total_pages = await _pdf_page_count(pdf_bytes)
    t_count = _time.monotonic() - t0

    if total_pages == 0:
        raise ValueError("Document produced no pages")
    if total_pages > ocr_config["max_pages"]:
        raise ValueError(
            f"Document has {total_pages} pages, exceeding the maximum "
            f"of {ocr_config['max_pages']}"
        )

    chunk_ranges = make_chunks(total_pages, chunk_size, overlap)
    total_chunks = len(chunk_ranges)

    logger.info(
        "ocr_start",
        pages=total_pages,
        chunks=total_chunks,
        model=model,
        format=output_format,
        pipeline="true",
        page_count_ms=round(t_count * 1000),
    )

    semaphore = asyncio.Semaphore(ocr_config["max_concurrent_chunks"])
    stop = asyncio.Event()
    t_inference_start = _time.monotonic()

    async def _convert_and_ocr(idx, start, end):
        """Convert this chunk's pages then immediately OCR them."""
        # Semaphore gates both conversion and inference together to bound
        # total CPU (pdftoppm processes) and memory (page images in RAM).
        async with semaphore:
            if await _should_stop(stop, http_request):
                raise OcrCancelled("document stopped")
            chunk_images = await _pdf_to_images_range(
                pdf_bytes, first_page=start + 1, last_page=end, dpi=dpi,
            )
            if await _should_stop(stop, http_request):
                raise OcrCancelled("document stopped")
            return await ocr_chunk(
                chunk_images, idx, start, end,
                total_pages, total_chunks,
                prompt_template, model, ocr_config,
                user, api_key, http_request, stop=stop,
            )

    tasks = [
        _convert_and_ocr(i, start, end)
        for i, (start, end) in enumerate(chunk_ranges)
    ]
    results = await _run_chunks(tasks, stop)
    t_inference = _time.monotonic() - t_inference_start

    # Sort by chunk index and extract text + usage
    results = sorted(results, key=lambda r: r[0])
    chunk_texts = [r[1] for r in results]
    degraded = any(r[3] for r in results)
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for _, _, usage, _ in results:
        for k in total_usage:
            total_usage[k] += usage.get(k, 0)

    # Merge chunks
    t0 = _time.monotonic()
    if output_format == "json":
        content = _pages_json(chunk_texts)  # one chunk per page in this mode
    elif total_chunks == 1:
        content = chunk_texts[0]
    else:
        content = await asyncio.to_thread(merge_chunks, chunk_texts)
    t_merge = _time.monotonic() - t0

    t_total = _time.monotonic() - t_total_start

    logger.info(
        "ocr_complete",
        pages=total_pages,
        chunks=total_chunks,
        output_chars=len(content),
        pipeline="true",
        inference_ms=round(t_inference * 1000),
        merge_ms=round(t_merge * 1000),
        total_ms=round(t_total * 1000),
    )

    return {
        "content": content,
        "format": output_format,
        "pages": total_pages,
        "chunks_processed": total_chunks,
        "usage": total_usage,
        "degraded": degraded,
    }


async def _perform_ocr_simple(
    page_images: List[bytes],
    model: str,
    output_format: str,
    chunk_size: int,
    overlap: int,
    ocr_config: dict,
    user: "User",
    api_key: "ApiKey",
    http_request: "Request",
    t_total_start: float,
    t_convert: float,
    prompt_template: str,
) -> Dict[str, Any]:
    """Simple OCR path for images (no pipelining needed)."""
    import time as _time

    total_pages = len(page_images)
    if total_pages == 0:
        raise ValueError("Document produced no pages")
    if total_pages > ocr_config["max_pages"]:
        raise ValueError(
            f"Document has {total_pages} pages, exceeding the maximum "
            f"of {ocr_config['max_pages']}"
        )

    total_image_bytes = sum(len(img) for img in page_images)
    logger.info(
        "ocr_start",
        pages=total_pages,
        model=model,
        format=output_format,
        pipeline="false",
        convert_ms=round(t_convert * 1000),
        total_image_kb=round(total_image_bytes / 1024),
    )

    chunk_ranges = make_chunks(total_pages, chunk_size, overlap)
    total_chunks = len(chunk_ranges)

    semaphore = asyncio.Semaphore(ocr_config["max_concurrent_chunks"])
    stop = asyncio.Event()
    t_inference_start = _time.monotonic()

    async def _bounded_ocr(idx, start, end):
        async with semaphore:
            if await _should_stop(stop, http_request):
                raise OcrCancelled("document stopped")
            return await ocr_chunk(
                page_images[start:end],
                idx, start, end,
                total_pages, total_chunks,
                prompt_template, model, ocr_config,
                user, api_key, http_request, stop=stop,
            )

    tasks = [
        _bounded_ocr(i, start, end)
        for i, (start, end) in enumerate(chunk_ranges)
    ]
    results = await _run_chunks(tasks, stop)
    t_inference = _time.monotonic() - t_inference_start

    results = sorted(results, key=lambda r: r[0])
    chunk_texts = [r[1] for r in results]
    degraded = any(r[3] for r in results)
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for _, _, usage, _ in results:
        for k in total_usage:
            total_usage[k] += usage.get(k, 0)

    t0 = _time.monotonic()
    if output_format == "json":
        content = _pages_json(chunk_texts)  # one chunk per page in this mode
    elif total_chunks == 1:
        content = chunk_texts[0]
    else:
        content = await asyncio.to_thread(merge_chunks, chunk_texts)
    t_merge = _time.monotonic() - t0

    t_total = _time.monotonic() - t_total_start

    logger.info(
        "ocr_complete",
        pages=total_pages,
        chunks=total_chunks,
        output_chars=len(content),
        pipeline="false",
        convert_ms=round(t_convert * 1000),
        inference_ms=round(t_inference * 1000),
        merge_ms=round(t_merge * 1000),
        total_ms=round(t_total * 1000),
    )

    return {
        "content": content,
        "format": output_format,
        "pages": total_pages,
        "chunks_processed": total_chunks,
        "usage": total_usage,
        "degraded": degraded,
    }
