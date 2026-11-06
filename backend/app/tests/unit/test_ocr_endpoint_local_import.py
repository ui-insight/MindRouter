"""Regression tests for /v1/ocr and /v1/ocrmd (backend/app/api/v1_openai.py).

Both handlers 500'd on every upload that carried a recognised Content-Type —
curl's `image/png`, a browser's `application/pdf` — with
`UnboundLocalError: cannot access local variable 'os'`. The cause was a
function-local `import os` inside the octet-stream fallback branch
(`if content_type == "application/octet-stream" and file.filename:`). A
local import binds the name for the WHOLE function, so when that branch was
skipped the later `os.path.basename(file.filename)` — added by the 2026-08-17
filename hardening, months after the inner import — read an unbound local
instead of the module-level `os`. The endpoints were broken from that
release until this fix; nobody noticed because OCR traffic was nil.

The first four tests call the handlers directly with the OCR service,
registry and availability check stubbed, so they exercise exactly the
control flow that failed. The last test is the anti-recurrence guard: no
function in `backend/app/api/` may re-import a name that the module already
imports at top level — that is the whole recipe for this class of bug.

Module-level import of v1_openai mirrors test_moderations_endpoint.py.
"""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

import backend.app.api.v1_openai as api

_API_DIR = Path(api.__file__).resolve().parent

_CONFIG = {
    "enabled": True, "model": "dots.MOCR", "chunk_size": 4, "overlap": 1,
    "dpi": 150, "max_file_size_mb": 10, "prompt_ocr": "p", "prompt_ocrmd": "q",
}
_RESULT = {"content": "# extracted", "format": "markdown", "pages": 1,
           "chunks_processed": 1, "usage": {"total_tokens": 3}}


class _Upload:
    """The parts of starlette's UploadFile the handlers read."""

    def __init__(self, filename, content_type, data=b"\x89PNG\r\n"):
        self.filename, self.content_type, self._data = filename, content_type, data

    async def read(self):
        return self._data


class _Db:
    """Request session stand-in: records whether it was released (committed)
    before OCR started, since a document can run for minutes."""

    def __init__(self, perform):
        self._perform = perform
        self.committed_before_ocr = None

    async def commit(self):
        if self.committed_before_ocr is None:
            self.committed_before_ocr = not self._perform.await_count


def _auth():
    return SimpleNamespace(id=1), SimpleNamespace(id=2)


async def _call(handler, upload, perform=None, **form):
    """Invoke a handler with everything past content-type resolution stubbed.

    perform_ocr is imported INSIDE the handler from backend.app.services.ocr,
    so it is patched on that module, not on the api module.
    """
    perform = perform or AsyncMock(return_value=dict(_RESULT))
    registry = SimpleNamespace(resolve_alias=lambda m: (m, None))
    form = {"model": None, "chunk_size": None, "overlap": None, "dpi": None, **form}
    with (
        patch("backend.app.services.ocr.get_ocr_config", AsyncMock(return_value=dict(_CONFIG))),
        patch("backend.app.services.ocr.perform_ocr", perform),
        patch.object(api, "get_registry", lambda: registry),
        patch.object(api, "model_availability", AsyncMock(return_value=api.AVAILABLE)),
        patch.object(api, "bind_request_context"),
        patch.object(api, "InferenceService", lambda db: SimpleNamespace(_check_quota=AsyncMock())),
    ):
        result = await handler(request=SimpleNamespace(), file=upload, db=_Db(perform), auth=_auth(), **form)
    return result, perform


# --------------------------------------------------------------------------
# The regression: a typed upload skips the fallback branch.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ocr_typed_upload_reaches_perform_ocr():
    result, perform = await _call(api.ocr, _Upload("scan.png", "image/png"), output_format="markdown")
    assert result["object"] == "ocr.result"
    assert result["content"] == "# extracted"
    kw = perform.call_args.kwargs
    assert kw["content_type"] == "image/png"
    assert kw["filename"] == "scan.png"


@pytest.mark.asyncio
async def test_ocrmd_typed_upload_returns_raw_markdown():
    result, perform = await _call(api.ocrmd, _Upload("scan.pdf", "application/pdf"))
    assert result.status_code == 200
    assert result.media_type == "text/markdown"
    assert result.body == b"# extracted"
    assert perform.call_args.kwargs["content_type"] == "application/pdf"


# --------------------------------------------------------------------------
# The branch the inner import lived in must keep working too.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_octet_stream_upload_is_resolved_by_extension():
    _, perform = await _call(api.ocr, _Upload("report.pdf", "application/octet-stream"), output_format="json")
    assert perform.call_args.kwargs["content_type"] == api._OCR_EXT_MAP[".pdf"]


@pytest.mark.asyncio
async def test_filename_is_reduced_to_its_basename():
    """The hardening that exposed the bug must itself keep working: a
    client-supplied path never reaches the service as a path."""
    _, perform = await _call(api.ocr, _Upload("../../etc/passwd.png", "image/png"), output_format="markdown")
    assert perform.call_args.kwargs["filename"] == "passwd.png"


# --------------------------------------------------------------------------
# Anti-recurrence guard.
# --------------------------------------------------------------------------


_BACKEND = _API_DIR.parent
_GUARDED = ("api", "dashboard", "services", "core")


def _module_level_names(tree):
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.update(a.asname or a.name for a in node.names)
    return names


def _bound_names(node):
    if isinstance(node, ast.Import):
        return [(a.asname or a.name).split(".")[0] for a in node.names]
    if isinstance(node, ast.ImportFrom):
        return [a.asname or a.name for a in node.names]
    return []


def _shadowing_hazards(tree, module_names):
    """(function, name, line) for each function-local import that rebinds a
    module-level name AND can actually fail at runtime: it sits inside a
    nested block (a path may skip it) or the name is read on an earlier line.

    A duplicate import that is the first thing a function does is only
    redundant — the local is always bound before any read — so it is not
    reported; the point is to catch the recipe that broke /v1/ocr, not to
    police style.
    """
    hits = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        top_level = {id(stmt) for stmt in fn.body}
        loads = {}
        for node in ast.walk(fn):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                loads.setdefault(node.id, []).append(node.lineno)
        for node in ast.walk(fn):
            for name in _bound_names(node):
                if name not in module_names:
                    continue
                nested = id(node) not in top_level
                used_before = any(line < node.lineno for line in loads.get(name, []))
                if nested or used_before:
                    hits.append((fn.name, name, node.lineno))
    return hits


def _guarded_files():
    for sub in _GUARDED:
        for path in sorted((_BACKEND / sub).rglob("*.py")):
            yield str(path.relative_to(_BACKEND))


@pytest.mark.parametrize("rel", list(_guarded_files()))
def test_no_function_shadows_a_module_import_where_it_can_fail(rel):
    """A function-local import of a name the module already imports makes
    that name local to the ENTIRE function. If the import sits inside a
    block that a code path can skip, or the name is read before the import
    line, that path raises UnboundLocalError at runtime — precisely how
    /v1/ocr and /v1/ocrmd broke. Import once at module level instead."""
    tree = ast.parse((_BACKEND / rel).read_text())
    hits = _shadowing_hazards(tree, _module_level_names(tree))
    assert hits == [], f"{rel}: local import shadows a module import on a skippable path: {hits}"


# --------------------------------------------------------------------------
# A file OCR cannot read is the client's error: 400, not 500.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("handler", ["ocr", "ocrmd"])
async def test_a_broken_pdf_is_a_400_with_a_plain_message(handler):
    from fastapi import HTTPException

    import backend.app.services.ocr as ocr_service

    async def page_count_first(**kw):
        # The real page count on garbage, as perform_ocr runs it first.
        await ocr_service._pdf_page_count(kw["file_bytes"])

    perform = AsyncMock(side_effect=page_count_first)
    upload = _Upload("scan.pdf", "application/pdf", data=b"%PDF-1.4 truncated")
    if handler == "ocr":
        with pytest.raises(HTTPException) as info:
            await _call(api.ocr, upload, perform=perform, output_format="markdown")
        status, detail = info.value.status_code, info.value.detail
    else:
        # /v1/ocrmd answers in plain text instead of raising.
        response, _ = await _call(api.ocrmd, upload, perform=perform)
        status, detail = response.status_code, response.body.decode()
    assert status == 400
    assert detail == ocr_service.UNREADABLE_PDF_MESSAGE
