"""Unit tests for blog HTML sanitization (WS10 stored-XSS hardening, F47).

Loads ``backend/app/dashboard/blog.py`` with its heavy backend.app.* imports
pre-stubbed in ``sys.modules`` so the DB/telemetry import chain never runs
(per the repo's import-hygiene rules). Only ``markdown`` is exercised for real.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

BLOG_PATH = Path(__file__).resolve().parents[2] / "dashboard" / "blog.py"


def _stub(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


@pytest.fixture(scope="module")
def blog():
    """Import blog.py in isolation with all backend.app.* deps stubbed."""
    saved = {k: sys.modules.get(k) for k in list(sys.modules) if k.startswith("backend")}

    backend = _stub("backend")
    app = _stub("backend.app")
    backend.app = app
    db = _stub("backend.app.db")
    app.db = db
    db.crud = _stub("backend.app.db.crud")
    session = _stub("backend.app.db.session", get_async_db=lambda: None)
    db.session = session
    dashboard = _stub("backend.app.dashboard")
    app.dashboard = dashboard
    dashboard.routes = _stub(
        "backend.app.dashboard.routes",
        get_session_user_id=lambda *a, **k: None,
        _admin_masquerade_context=lambda *a, **k: {},
        templates=object(),
    )
    dashboard.blog_export = _stub("backend.app.dashboard.blog_export")
    services = _stub("backend.app.services")
    app.services = services
    services.email_service = _stub("backend.app.services.email_service")
    services.branding = _stub(
        "backend.app.services.branding", get_branding=lambda: {"app_name": "X"}
    )
    storage = _stub("backend.app.storage")
    app.storage = storage
    storage.artifacts = _stub(
        "backend.app.storage.artifacts", get_artifact_storage=lambda: None
    )
    app.settings = _stub(
        "backend.app.settings",
        get_settings=lambda: types.SimpleNamespace(app_base_url=""),
    )

    spec = importlib.util.spec_from_file_location("blog_under_test", BLOG_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    yield mod

    # Restore prior backend.* modules so we don't pollute other test files.
    for k in [m for m in sys.modules if m.startswith("backend")]:
        del sys.modules[k]
    for k, v in saved.items():
        if v is not None:
            sys.modules[k] = v


def test_strips_script_tag(blog):
    out = blog._sanitize_html('<div>ok</div><script>alert(1)</script>')
    assert "<script" not in out.lower()
    assert "<div>ok</div>" in out


def test_strips_iframe_and_object(blog):
    out = blog._sanitize_html(
        '<iframe src="//evil"></iframe><object data="x"></object>safe'
    )
    assert "<iframe" not in out.lower()
    assert "<object" not in out.lower()
    assert "safe" in out


def test_strips_inline_event_handlers(blog):
    out = blog._sanitize_html('<a href="#" onclick="steal()">x</a>')
    assert "onclick" not in out.lower()
    assert "<a" in out  # the anchor itself is preserved


def test_neutralizes_javascript_uri(blog):
    out = blog._sanitize_html('<a href="javascript:alert(1)">x</a>')
    assert "javascript:" not in out.lower()
    assert "unsafe:" in out


def test_render_markdown_strips_raw_script(blog):
    out = blog._render_markdown("Hello\n\n<script>alert(1)</script>\n\nBye")
    assert "<script" not in out.lower()
    assert "Hello" in out


def test_render_markdown_preserves_code_fence_script_sample(blog):
    # A <script> inside a code fence is entity-encoded by markdown, so it must
    # survive sanitization verbatim (as display text), not be stripped.
    out = blog._render_markdown("```\n<script>alert(1)</script>\n```")
    assert "&lt;script&gt;" in out
    # No executable raw tag leaked through.
    assert "<script>alert" not in out


def test_render_markdown_normal_content_unchanged(blog):
    out = blog._render_markdown("# Title\n\nSome **bold** and a [link](/x).")
    assert "<h1" in out
    assert "<strong>bold</strong>" in out
    assert 'href="/x"' in out


# ---------------------------------------------------------------------------
# Inline code spans containing "</" must not be mistaken for raw HTML by the
# html.parser-based block scanner (which, on current CPython builds, swallows
# everything after the span).
# ---------------------------------------------------------------------------


def test_code_span_with_closing_tag_shape_renders_as_code(blog):
    md = (
        "- **Scripts need `</dev/null`.** When stdin is not a TTY.\n"
        "- Next item.\n\n## Another heading\n\nMore text after.\n"
    )
    out = blog._render_markdown(md)
    assert "<code>&lt;/dev/null</code>" in out
    assert 'id="another-heading"' in out and "More text after." in out
    assert "**Scripts" not in out  # the emphasis around the span still renders
    assert "<strong>Scripts need <code>&lt;/dev/null</code>.</strong>" in out


def test_protect_code_spans_leaves_fenced_blocks_and_plain_spans_alone(blog):
    md = "Use `ls` then:\n\n```bash\ncodex exec \"hi\" </dev/null\n```\n\nand `a</b` here."
    protected = blog._protect_code_spans(md)
    assert "`ls`" in protected                       # no "</" -> untouched
    assert 'codex exec "hi" </dev/null' in protected  # inside a fence -> untouched
    assert "<code>a&lt;/b</code>" in protected
    out = blog._render_markdown(md)
    assert "&lt;/dev/null" in out and "and <code>a&lt;/b</code> here." in out
