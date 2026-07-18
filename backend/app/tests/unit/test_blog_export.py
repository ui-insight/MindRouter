"""Unit tests for blog_export (static-HTML export for mindrouter.ai).

Loaded via importlib to bypass the backend.app package __init__ import chain
(see project memory: dashboard/db package inits pull in the DB stack). The
module itself has no app-level imports at import time — ArtifactStorage is
imported lazily inside fetch_images — so it loads clean standalone.
"""

import importlib.util
import os
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

_MODPATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "dashboard", "blog_export.py")
)
_spec = importlib.util.spec_from_file_location("blog_export", _MODPATH)
be = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(be)


def make_post(**kw):
    """A BlogPost stand-in with just the attributes the exporter touches."""
    author = SimpleNamespace(
        full_name=kw.pop("author_full_name", "Luke Sheneman"),
        username=kw.pop("author_username", "admin"),
    )
    defaults = dict(
        id=1,
        slug="hello-world",
        title="Hello World",
        content="# Hi\n\nThis is **bold** and a [link](https://example.com).",
        excerpt=None,
        published_at=datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc),
        author=author,
    )
    defaults.update(kw)
    return SimpleNamespace(**defaults)


# --- markdown ---------------------------------------------------------------
def test_render_markdown_codehilite_and_tables():
    html = be.render_markdown("```python\nx = 1\n```\n\n| a | b |\n|---|---|\n| 1 | 2 |")
    assert 'class="codehilite"' in html
    assert "<table>" in html


def test_render_markdown_handles_none():
    assert be.render_markdown(None) == ""


# --- image collection -------------------------------------------------------
def test_collect_image_paths_from_markdown_and_html_dedup_and_order():
    content = (
        "![one](/blog/images/2026/07/18/ab/one_x.png)\n"
        '<img src="/blog/images/2026/07/18/cd/two_y.jpg">\n'
        "![dup](/blog/images/2026/07/18/ab/one_x.png)\n"
        "![external](https://example.com/z.png)"
    )
    paths = be.collect_image_paths(content)
    assert paths == [
        "2026/07/18/ab/one_x.png",
        "2026/07/18/cd/two_y.jpg",
    ]  # deduped, ordered, external ignored


def test_collect_image_paths_empty():
    assert be.collect_image_paths("no images here") == []
    assert be.collect_image_paths(None) == []


# --- description derivation -------------------------------------------------
def test_derive_description_prefers_excerpt():
    assert be.derive_description("A short excerpt.", "# ignored body") == "A short excerpt."


def test_derive_description_strips_markdown_and_truncates():
    body = "# Title\n\nSome **strong** text with a [link](http://x) and ![img](/blog/images/a.png). " + ("word " * 60)
    desc = be.derive_description(None, body, limit=40)
    assert "#" not in desc and "![" not in desc and "](" not in desc
    assert "strong" in desc  # markdown emphasis markers removed, text kept
    assert len(desc) <= 40


# --- post page --------------------------------------------------------------
def test_render_post_html_structure():
    html = be.render_post_html(make_post())
    assert html.startswith("<!DOCTYPE html>")
    assert "<title>Hello World — MindRouter Blog</title>" in html
    assert '<link rel="canonical" href="https://mindrouter.ai/blog/hello-world/">' in html
    assert 'property="og:type" content="article"' in html
    assert 'application/ld+json' in html and '"BlogPosting"' in html
    # site shell present
    assert 'class="navbar-brand" href="/"' in html
    assert 'href="/blog/"' in html  # Blog nav link
    assert "site-footer" in html
    assert '/css/style.css' in html and '/css/blog.css' in html
    assert 'themeToggleBtn' in html
    # content rendered + byline
    assert 'class="blog-content"' in html
    assert "<strong>bold</strong>" in html
    assert "By Luke Sheneman" in html
    assert "July 18, 2026" in html


def test_render_post_html_escapes_title():
    html = be.render_post_html(make_post(title='<script>alert(1)</script>'))
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_render_post_html_falls_back_to_username_then_org():
    html = be.render_post_html(make_post(author_full_name=None))
    assert "By admin" in html
    p = make_post()
    p.author = None
    assert "By MindRouter" in be.render_post_html(p)


def test_post_path_helpers():
    assert be.post_repo_path("my-slug") == "blog/my-slug/index.html"
    assert be.post_url("my-slug") == "/blog/my-slug/"
    assert be.post_canonical("my-slug") == "https://mindrouter.ai/blog/my-slug/"


# --- index page -------------------------------------------------------------
def test_render_index_lists_selected_posts_in_order():
    posts = [make_post(slug="newest", title="Newest"), make_post(slug="older", title="Older")]
    html = be.render_index_html(posts)
    assert html.index('href="/blog/newest/"') < html.index('href="/blog/older/"')
    assert ">Newest</a>" in html and ">Older</a>" in html


def test_render_index_empty_state():
    html = be.render_index_html([])
    assert "No posts yet" in html


# --- RSS feed ---------------------------------------------------------------
def test_render_feed_xml_valid_and_has_items():
    import xml.dom.minidom as minidom

    xml = be.render_feed_xml([make_post(slug="a", title="Post A")])
    minidom.parseString(xml)  # raises if malformed
    assert "<item>" in xml
    assert "https://mindrouter.ai/blog/a/" in xml
    assert "<title>Post A</title>" in xml


# --- image fetching (async) -------------------------------------------------
class _FakeStorage:
    def __init__(self, present):
        self._present = present  # dict path -> bytes

    async def retrieve(self, path):
        return self._present.get(path)


@pytest.mark.asyncio
async def test_fetch_images_present_and_missing():
    content = "![a](/blog/images/x/a.png) ![b](/blog/images/y/b.png)"
    storage = _FakeStorage({"x/a.png": b"\x89PNG-data"})
    images = await be.fetch_images(content, storage=storage)
    assert len(images) == 2
    a, b = images
    assert a.storage_path == "x/a.png" and a.repo_path == "blog/images/x/a.png"
    assert a.url == "/blog/images/x/a.png" and a.data == b"\x89PNG-data"
    assert a.size == len(b"\x89PNG-data") and a.missing is False
    assert b.missing is True and b.data is None and b.size == 0


@pytest.mark.asyncio
async def test_export_post_combines_html_and_images():
    post = make_post(content="Body ![a](/blog/images/x/a.png)")
    storage = _FakeStorage({"x/a.png": b"IMG"})
    exported = await be.export_post(post, storage=storage)
    assert exported.slug == "hello-world"
    assert exported.repo_path == "blog/hello-world/index.html"
    assert exported.canonical == "https://mindrouter.ai/blog/hello-world/"
    assert "<!DOCTYPE html>" in exported.html
    assert len(exported.images) == 1 and exported.images[0].data == b"IMG"
    assert exported.missing_images == []


@pytest.mark.asyncio
async def test_export_post_reports_missing_images():
    post = make_post(content="![gone](/blog/images/x/gone.png)")
    exported = await be.export_post(post, storage=_FakeStorage({}))
    assert exported.missing_images == ["x/gone.png"]
