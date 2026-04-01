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
import difflib
import io
import re
import shutil
import subprocess
import tempfile
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
        "temperature": await crud.get_config_json(db, "ocr.temperature", 0.1),
    }


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
    # Images: single page
    if content_type.startswith("image/"):
        img = Image.open(io.BytesIO(file_bytes))
        # Handle multi-frame images (GIF, TIFF)
        pages = []
        try:
            frame = 0
            while True:
                img.seek(frame)
                pages.append(_image_to_png_bytes(img.copy()))
                frame += 1
        except EOFError:
            pass
        if not pages:
            pages.append(_image_to_png_bytes(img))
        return pages

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
    """Convert PDF bytes to a list of PNG byte buffers using pdf2image."""
    from pdf2image import convert_from_bytes

    def _convert():
        pil_images = convert_from_bytes(pdf_bytes, dpi=dpi, fmt="png")
        return [_image_to_png_bytes(img) for img in pil_images]

    return await asyncio.to_thread(_convert)


async def _pdf_page_count(pdf_bytes: bytes) -> int:
    """Get the number of pages in a PDF without converting to images."""
    from pdf2image.pdf2image import pdfinfo_from_bytes
    def _count():
        info = pdfinfo_from_bytes(pdf_bytes)
        return info.get("Pages", 0)
    return await asyncio.to_thread(_count)


async def _pdf_to_images_range(
    pdf_bytes: bytes, first_page: int, last_page: int, dpi: int = 200
) -> List[bytes]:
    """Convert a range of PDF pages to PNG byte buffers (1-indexed)."""
    from pdf2image import convert_from_bytes

    def _convert():
        pil_images = convert_from_bytes(
            pdf_bytes, dpi=dpi, fmt="png",
            first_page=first_page, last_page=last_page,
        )
        return [_image_to_png_bytes(img) for img in pil_images]

    return await asyncio.to_thread(_convert)


async def _office_to_pdf(file_bytes: bytes, suffix: str) -> bytes:
    """Convert Office document to PDF via LibreOffice headless."""
    lo_path = shutil.which("libreoffice") or shutil.which("soffice")
    if not lo_path:
        raise RuntimeError(
            "LibreOffice is not installed. Office document OCR requires "
            "'libreoffice' or 'soffice' on PATH."
        )

    with tempfile.TemporaryDirectory(prefix="ocr_lo_") as tmpdir:
        input_path = Path(tmpdir) / f"input{suffix}"
        input_path.write_bytes(file_bytes)

        proc = await asyncio.create_subprocess_exec(
            lo_path,
            "--headless",
            "--convert-to", "pdf",
            "--outdir", tmpdir,
            str(input_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
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


def _build_ocr_prompt(
    num_pages: int,
    output_format: str = "markdown",
    is_retry: bool = False,
    prev_length: int = 0,
) -> str:
    """Build the OCR system prompt."""
    if output_format == "json":
        format_instruction = (
            "Output the result as a JSON object with a top-level 'pages' array. "
            "Each page should have 'page_number' (int), 'content' (string with "
            "the full text), and 'tables' (array of objects with 'headers' and 'rows'). "
            "Output valid JSON only — no markdown fences or commentary."
        )
    else:
        format_instruction = (
            "Convert ALL of the following page images to well-structured markdown. "
            "Render tables as proper markdown tables with correct columns and rows. "
            "Do not add any preamble like 'Here is the markdown' - just output "
            "the markdown directly."
        )

    if is_retry:
        return (
            f"IMPORTANT: You MUST convert ALL {num_pages} page images below. "
            f"Your previous attempt only produced {prev_length} characters which "
            "is too short. Every single page must be fully transcribed. "
            "Convert ALL text, tables, headers, and content from EVERY page image. "
            f"Do not stop early. Do not skip any pages. {format_instruction}"
        )

    return (
        f"{format_instruction} "
        "Preserve all text exactly as it appears. "
        "Do not summarize or omit anything. Do not add any commentary. "
        f"There are {num_pages} page images - make sure you process EVERY page."
    )


async def ocr_chunk(
    page_images: List[bytes],
    chunk_idx: int,
    start_page: int,
    end_page: int,
    total_pages: int,
    total_chunks: int,
    output_format: str,
    model: str,
    ocr_config: dict,
    user: "User",
    api_key: "ApiKey",
    http_request: "Request",
) -> Tuple[int, str, Dict[str, int]]:
    """
    Send a chunk of page images to the LLM and return OCR text.

    Each chunk gets its own DB session and InferenceService to avoid
    session state conflicts when multiple chunks run concurrently.

    Returns (chunk_idx, text, usage_dict).
    """
    from backend.app.db.session import AsyncSessionLocal
    from backend.app.services.inference import InferenceService

    num_pages = end_page - start_page
    max_retries = ocr_config["max_retries"]

    # Build content blocks: prompt + images
    prompt = _build_ocr_prompt(num_pages, output_format)
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
        max_tokens=ocr_config["max_tokens"],
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
                result = await service.chat_completion(
                    canonical, user, api_key, http_request
                )
            except Exception as e:
                logger.warning(
                    "ocr_chunk_error",
                    chunk=chunk_idx,
                    attempt=attempt,
                    error=str(e),
                )
                if attempt == max_retries:
                    raise
                # Reset session state for retry
                await chunk_db.rollback()
                continue

            # Accumulate usage
            usage = result.get("usage", {})
            for k in total_usage:
                total_usage[k] += usage.get(k, 0)

            text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            text = _strip_fences(text)

            logger.info(
                "ocr_chunk_result",
                chunk=chunk_idx + 1,
                total_chunks=total_chunks,
                pages=f"{start_page + 1}-{end_page}",
                chars=len(text),
                attempt=attempt + 1,
            )

            if len(text) >= expected_min_chars or attempt == max_retries:
                return chunk_idx, text, total_usage

            # Retry with stronger prompt
            logger.info(
                "ocr_chunk_retry",
                chunk=chunk_idx + 1,
                chars=len(text),
                expected=expected_min_chars,
            )
            retry_prompt = _build_ocr_prompt(
                num_pages, output_format, is_retry=True, prev_length=len(text)
            )
            content_blocks[0] = TextContent(text=retry_prompt)
            canonical.messages[0].content = content_blocks

    # Unreachable, but satisfy type checker
    return chunk_idx, "", total_usage


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

    t_total_start = _time.monotonic()
    is_pdf = (content_type == "application/pdf")

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

    # For PDFs with multiple pages, use pipelined conversion + inference
    if is_pdf:
        return await _perform_ocr_pipelined(
            file_bytes, model, output_format, chunk_size, overlap, dpi,
            ocr_config, user, api_key, http_request, t_total_start,
        )

    # For images: simple path (single page, no chunking needed usually)
    t0 = _time.monotonic()
    page_images = await document_to_images(file_bytes, content_type, filename, dpi)
    t_convert = _time.monotonic() - t0

    return await _perform_ocr_simple(
        page_images, model, output_format, chunk_size, overlap,
        ocr_config, user, api_key, http_request, t_total_start, t_convert,
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
    t_inference_start = _time.monotonic()

    async def _convert_and_ocr(idx, start, end):
        """Convert this chunk's pages then immediately OCR them."""
        # Convert just this chunk's page range (1-indexed for pdf2image)
        chunk_images = await _pdf_to_images_range(
            pdf_bytes, first_page=start + 1, last_page=end, dpi=dpi,
        )
        # Send to LLM as soon as pages are ready
        async with semaphore:
            return await ocr_chunk(
                chunk_images, idx, start, end,
                total_pages, total_chunks,
                output_format, model, ocr_config,
                user, api_key, http_request,
            )

    tasks = [
        _convert_and_ocr(i, start, end)
        for i, (start, end) in enumerate(chunk_ranges)
    ]
    results = await asyncio.gather(*tasks)
    t_inference = _time.monotonic() - t_inference_start

    # Sort by chunk index and extract text + usage
    results = sorted(results, key=lambda r: r[0])
    chunk_texts = [r[1] for r in results]
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for _, _, usage in results:
        for k in total_usage:
            total_usage[k] += usage.get(k, 0)

    # Merge chunks
    t0 = _time.monotonic()
    if total_chunks == 1:
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
    t_inference_start = _time.monotonic()

    async def _bounded_ocr(idx, start, end):
        async with semaphore:
            return await ocr_chunk(
                page_images[start:end],
                idx, start, end,
                total_pages, total_chunks,
                output_format, model, ocr_config,
                user, api_key, http_request,
            )

    tasks = [
        _bounded_ocr(i, start, end)
        for i, (start, end) in enumerate(chunk_ranges)
    ]
    results = await asyncio.gather(*tasks)
    t_inference = _time.monotonic() - t_inference_start

    results = sorted(results, key=lambda r: r[0])
    chunk_texts = [r[1] for r in results]
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for _, _, usage in results:
        for k in total_usage:
            total_usage[k] += usage.get(k, 0)

    t0 = _time.monotonic()
    if total_chunks == 1:
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
    }
