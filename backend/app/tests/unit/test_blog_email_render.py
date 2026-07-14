############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_blog_email_render.py: Blog-email HTML rendering.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Covers the email-specific rendering of blog posts (email clients
strip <style> blocks, so borders/formatting must be inline):
- [TOC] marker stripped (no 'toc' extension in the email renderer)
- tables get visible black inline borders (incl. alignment merge, and
  the <thead>-not-matched-by-<th> guard)
- fenced code blocks get a bordered container
"""

import importlib.util
import re
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

# Direct-load email_service, stubbing its heavy imports.
sys.modules.setdefault("aiosmtplib", MagicMock())
for _name in [
    "backend", "backend.app", "backend.app.db", "backend.app.db.crud",
    "backend.app.db.session", "backend.app.core",
    "backend.app.core.redis_client", "backend.app.settings",
    "backend.app.logging_config",
]:
    if _name not in sys.modules:
        _m = types.ModuleType(_name)
        _m.__path__ = []
        sys.modules[_name] = _m
sys.modules["backend.app.db.crud"].get_config_json = lambda *a, **k: None
sys.modules["backend.app.db.session"].get_async_db_context = lambda *a, **k: None
sys.modules["backend.app.logging_config"].get_logger = lambda *a, **k: MagicMock()
sys.modules["backend.app.settings"].get_settings = lambda: MagicMock()

_spec = importlib.util.spec_from_file_location(
    "email_service",
    Path(__file__).resolve().parents[2] / "services" / "email_service.py",
)
_es = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_es)


def _render(md: str) -> str:
    return _es._render_blog_email(md_title := "T", md, "slug", "Admin",
                                  "https://mindrouter.uidaho.edu")


class TestTocStripping:
    def test_toc_marker_removed(self):
        html = _render("[TOC]\n\n# Heading\n\nBody text.")
        assert "[TOC]" not in html
        assert "Heading" in html

    def test_toc_case_insensitive(self):
        assert "[toc]" not in _render("[toc]\n\ntext").lower()


class TestTableBorders:
    _MD = (
        "| Feature | A | B |\n"
        "|---|---|---|\n"
        "| memory | no | yes |\n"
    )

    def test_table_has_black_border(self):
        html = _render(self._MD)
        assert "border:2px solid #000000" in html
        assert 'cellspacing="0"' in html

    def test_cells_have_black_borders(self):
        html = _render(self._MD)
        # 3 header + 3 body cells, all bordered black
        assert html.count("border:1px solid #000000") >= 6

    def test_thead_not_styled_as_cell(self):
        # <th> must not also match <thead>
        html = _render(self._MD)
        thead = re.search(r"<thead[^>]*>", html)
        assert thead is not None
        assert "padding:8px 10px" not in thead.group(0)

    def test_alignment_style_merged_not_duplicated(self):
        # Colon alignment makes python-markdown emit style="text-align:right"
        aligned = (
            "| L | R |\n"
            "|:---|---:|\n"
            "| a | b |\n"
        )
        html = _render(aligned)
        # existing alignment preserved AND border appended, single style attr
        assert "text-align" in html
        assert "border:1px solid #000000" in html
        assert not re.search(r'<t[dh][^>]*style="[^"]*"[^>]*style=', html)


class TestCodeContainer:
    def test_code_block_has_border(self):
        html = _render("```python\nx = 1\n```")
        pre = re.search(r"<pre[^>]*>", html)
        assert pre is not None
        assert "border:1px solid #333333" in pre.group(0)

    def test_inline_code_still_styled(self):
        html = _render("Use `client.responses.create()` here.")
        assert "responses.create" in html
