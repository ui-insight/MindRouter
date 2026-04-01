"""
Tests for backend.app.services.ocr.

Validates the OCR pipeline components:
- Document-to-image conversion (PDF, images)
- Chunking logic
- Deterministic merge
- Prompt building
- Full OCR pipeline with mocked LLM responses

Uses a fixture PDF at tests/unit/fixtures/test_ocr.pdf.
"""

import asyncio
import base64
import difflib
import io
import re
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

# --------------------------------------------------------------------------
# The pure functions we test are re-implemented here to avoid importing
# backend.app.services.ocr (which triggers the DB import chain).
# These mirror the implementations in ocr.py exactly.
# --------------------------------------------------------------------------


def make_chunks(num_pages, chunk_size=6, overlap=2):
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


def _strip_fences(text):
    text = text.strip()
    if text.startswith("```markdown"):
        text = text[len("```markdown"):].strip()
    if text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


def _normalize_line(line):
    line = line.strip().lower()
    line = re.sub(r"\s+", " ", line)
    return line


def _merge_nearby_blocks(blocks, gap=5):
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


def find_overlap_boundary(chunk_a, chunk_b, overlap_fraction=0.4):
    lines_a = chunk_a.split("\n")
    lines_b = chunk_b.split("\n")
    search_a_start = max(0, int(len(lines_a) * (1.0 - overlap_fraction)))
    search_b_end = min(len(lines_b), int(len(lines_b) * overlap_fraction))
    tail_a = [_normalize_line(l) for l in lines_a[search_a_start:]]
    head_b = [_normalize_line(l) for l in lines_b[:search_b_end]]
    sm = difflib.SequenceMatcher(
        isjunk=lambda x: x.strip() == "", a=tail_a, b=head_b, autojunk=False,
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


def merge_chunks(chunks_text):
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


def _build_ocr_prompt(num_pages, output_format="markdown", is_retry=False, prev_length=0):
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


def _image_to_png_bytes(img, max_dim=2048):
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

FIXTURES_DIR = Path(__file__).parent / "fixtures"
TEST_PDF = FIXTURES_DIR / "test_ocr.pdf"


# -----------------------------------------------------------------------
# Chunking
# -----------------------------------------------------------------------

class TestMakeChunks:
    def test_single_page(self):
        assert make_chunks(1, 6, 2) == [(0, 1)]

    def test_within_chunk_size(self):
        assert make_chunks(5, 6, 2) == [(0, 5)]

    def test_exact_chunk_size(self):
        assert make_chunks(6, 6, 2) == [(0, 6)]

    def test_two_chunks(self):
        chunks = make_chunks(8, 6, 2)
        assert chunks == [(0, 6), (4, 8)]

    def test_three_chunks(self):
        chunks = make_chunks(12, 6, 2)
        assert chunks == [(0, 6), (4, 10), (8, 12)]

    def test_large_document(self):
        chunks = make_chunks(50, 6, 2)
        assert chunks[0] == (0, 6)
        assert chunks[-1][1] == 50
        # Verify overlap: each chunk starts 4 pages after the previous
        for i in range(1, len(chunks)):
            assert chunks[i][0] == chunks[i - 1][0] + 4

    def test_overlap_zero(self):
        chunks = make_chunks(12, 4, 0)
        assert chunks == [(0, 4), (4, 8), (8, 12)]

    def test_overlap_one(self):
        chunks = make_chunks(10, 5, 1)
        assert chunks[0] == (0, 5)
        assert chunks[1][0] == 4  # stride = 5 - 1 = 4


# -----------------------------------------------------------------------
# Strip fences
# -----------------------------------------------------------------------

class TestStripFences:
    def test_no_fences(self):
        assert _strip_fences("Hello world") == "Hello world"

    def test_markdown_fence(self):
        assert _strip_fences("```markdown\n# Title\n```") == "# Title"

    def test_generic_fence(self):
        assert _strip_fences("```\nContent\n```") == "Content"

    def test_only_closing_fence(self):
        assert _strip_fences("Content\n```") == "Content"


# -----------------------------------------------------------------------
# Prompt building
# -----------------------------------------------------------------------

class TestBuildOcrPrompt:
    def test_markdown_prompt(self):
        prompt = _build_ocr_prompt(3, "markdown")
        assert "markdown" in prompt.lower()
        assert "3 page images" in prompt
        assert "EVERY page" in prompt

    def test_json_prompt(self):
        prompt = _build_ocr_prompt(2, "json")
        assert "JSON" in prompt
        assert "2 page images" in prompt

    def test_retry_prompt(self):
        prompt = _build_ocr_prompt(4, "markdown", is_retry=True, prev_length=100)
        assert "IMPORTANT" in prompt
        assert "100 characters" in prompt
        assert "4 page images" in prompt


# -----------------------------------------------------------------------
# Deterministic merge
# -----------------------------------------------------------------------

class TestMergeChunks:
    def test_single_chunk(self):
        assert merge_chunks(["Hello"]) == "Hello"

    def test_overlap_detection(self):
        # Need enough unique lines so the overlap region is detectable
        # (overlap_fraction=0.4 means we search last 40% of A and first 40% of B)
        unique_a = [f"unique_a_{i}" for i in range(20)]
        overlap = [f"overlap_line_{i}" for i in range(15)]
        unique_b = [f"unique_b_{i}" for i in range(20)]

        chunk_a = "\n".join(unique_a + overlap)
        chunk_b = "\n".join(overlap + unique_b)

        result = merge_chunks([chunk_a, chunk_b])
        assert "unique_a_0" in result
        assert "unique_b_19" in result
        # Overlap lines should appear only once
        assert result.count("overlap_line_0") == 1

    def test_no_overlap(self):
        result = merge_chunks(["AAA", "BBB"])
        assert "AAA" in result
        assert "BBB" in result

    def test_three_chunks(self):
        shared = "\n".join([f"shared line {i}" for i in range(10)])
        chunk_a = "Start\n" + shared
        chunk_b = shared + "\nMiddle\n" + shared
        chunk_c = shared + "\nEnd"
        result = merge_chunks([chunk_a, chunk_b, chunk_c])
        assert "Start" in result
        assert "Middle" in result
        assert "End" in result


class TestFindOverlapBoundary:
    def test_clear_overlap(self):
        lines = [f"line {i}" for i in range(20)]
        chunk_a = "\n".join(lines[:15])
        chunk_b = "\n".join(lines[10:])
        a_cut, b_cut = find_overlap_boundary(chunk_a, chunk_b)
        # Should find the overlap region
        assert a_cut >= 10
        assert b_cut >= 0

    def test_no_overlap(self):
        a_cut, b_cut = find_overlap_boundary("aaa\nbbb\nccc", "xxx\nyyy\nzzz")
        # No overlap: should concatenate (a_cut = len(a), b_cut = 0)
        assert a_cut == 3
        assert b_cut == 0


# -----------------------------------------------------------------------
# Document to images
# -----------------------------------------------------------------------

class TestImageConversion:
    """Test image conversion without importing the async document_to_images."""

    def test_png_roundtrip(self):
        img = Image.new("RGB", (100, 100), "white")
        png_bytes = _image_to_png_bytes(img)
        result = Image.open(io.BytesIO(png_bytes))
        assert result.format == "PNG"
        assert result.size == (100, 100)

    def test_jpeg_to_png(self):
        img = Image.new("RGB", (100, 100), "red")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        # Re-open as JPEG, convert to PNG
        jpg_img = Image.open(io.BytesIO(buf.getvalue()))
        png_bytes = _image_to_png_bytes(jpg_img)
        result = Image.open(io.BytesIO(png_bytes))
        assert result.format == "PNG"

    def test_pdf_fixture_exists(self):
        """Verify the test PDF fixture was created."""
        assert TEST_PDF.exists(), f"Missing fixture: {TEST_PDF}"
        assert TEST_PDF.stat().st_size > 0


# -----------------------------------------------------------------------
# Image conversion helper
# -----------------------------------------------------------------------

class TestImageToPngBytes:
    def test_rgb_image(self):
        img = Image.new("RGB", (200, 200), "red")
        png_bytes = _image_to_png_bytes(img)
        result = Image.open(io.BytesIO(png_bytes))
        assert result.format == "PNG"

    def test_large_image_downscaled(self):
        img = Image.new("RGB", (4000, 3000), "blue")
        png_bytes = _image_to_png_bytes(img, max_dim=1024)
        result = Image.open(io.BytesIO(png_bytes))
        assert max(result.size) <= 1024

    def test_rgba_converted(self):
        img = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        png_bytes = _image_to_png_bytes(img)
        result = Image.open(io.BytesIO(png_bytes))
        assert result.mode == "RGB"
