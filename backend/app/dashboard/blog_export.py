############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# blog_export.py: Render blog posts as self-contained static
#                 HTML for publishing to the mindrouter.ai
#                 website (github.com/sheneman/mindrouter-website).
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Static-HTML export of blog posts for the public mindrouter.ai site.

This module is intentionally decoupled from the app's Jinja/dashboard stack
and the database: the render functions take plain post data and return
strings/bytes, so they can be unit-tested in isolation and reused by the
GitHub publisher (see services/website_publisher.py, PR3).

Design notes:
* Pages mirror the mindrouter.ai shell (Bootstrap 5.3.2 via CDN, the site's
  navbar/footer, dark/light theme via ``data-bs-theme`` + ``localStorage``),
  so generated posts look native next to index.html / documentation.html.
* Markdown rendering matches dashboard.blog._render_markdown exactly
  (fenced_code, codehilite, tables, toc), so what an author previews in the
  app is what ships to the public site.
* Blog images stay at their root-relative ``/blog/images/<path>`` URLs; the
  publisher copies the referenced files into the repo at ``blog/images/<path>``
  so the URLs resolve as-is on mindrouter.ai. No content rewriting needed.
"""

import html
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, List, Optional

import markdown

# Canonical public site (used for canonical URLs, OpenGraph, RSS, JSON-LD).
SITE_BASE_URL = "https://mindrouter.ai"

# Matches blog image references in raw Markdown or rendered HTML:
#   ![alt](/blog/images/2026/07/18/ab/hash_uuid.png)
#   <img src="/blog/images/2026/07/18/ab/hash_uuid.png">
# Capture group 1 is the ArtifactStorage-relative path (after /blog/images/).
_IMG_REF_RE = re.compile(r"/blog/images/([^\s\"')]+)")

# Strip HTML tags / markdown image+link syntax for deriving a plain-text
# description from post content when no excerpt is set.
_TAG_RE = re.compile(r"<[^>]+>")
_MD_IMG_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_WS_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def render_markdown(text: str) -> str:
    """Render post Markdown to HTML (mirrors dashboard.blog._render_markdown)."""
    return markdown.markdown(
        text or "",
        extensions=["fenced_code", "codehilite", "tables", "toc"],
        extension_configs={
            "codehilite": {"css_class": "codehilite", "guess_lang": False},
        },
    )


def _esc(value: Any) -> str:
    """HTML-escape a text value (attributes and body text)."""
    return html.escape("" if value is None else str(value), quote=True)


def derive_description(excerpt: Optional[str], content_md: str, limit: int = 160) -> str:
    """Plain-text description for meta/OG tags: excerpt, else start of content."""
    text = excerpt.strip() if excerpt and excerpt.strip() else (content_md or "")
    if not (excerpt and excerpt.strip()):
        text = _MD_IMG_RE.sub("", text)                    # ![alt](url)
        text = _MD_LINK_RE.sub(r"\1", text)                # [text](url) -> text
        text = _TAG_RE.sub("", text)                       # <tags>
        text = re.sub(r"`+", "", text)                     # `code`
        text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)  # # ATX headings
        text = re.sub(r"(?m)^\s{0,3}>\s?", "", text)       # > blockquotes
        text = re.sub(r"\*{1,3}", "", text)                # *em* **strong**
    text = _WS_RE.sub(" ", text).strip()
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text


# ---------------------------------------------------------------------------
# Data holders
# ---------------------------------------------------------------------------
@dataclass
class ExportedImage:
    """One image referenced by a post, resolved from ArtifactStorage."""

    storage_path: str          # e.g. 2026/07/18/ab/hash_uuid.png
    repo_path: str             # blog/images/2026/07/18/ab/hash_uuid.png
    url: str                   # /blog/images/2026/07/18/ab/hash_uuid.png
    data: Optional[bytes] = None
    size: int = 0
    missing: bool = False


@dataclass
class ExportedPost:
    """A rendered post plus everything the publisher needs to commit it."""

    slug: str
    title: str
    html: str
    repo_path: str             # blog/<slug>/index.html
    url: str                   # /blog/<slug>/
    canonical: str             # https://mindrouter.ai/blog/<slug>/
    images: List[ExportedImage] = field(default_factory=list)

    @property
    def missing_images(self) -> List[str]:
        return [img.storage_path for img in self.images if img.missing]


def post_repo_path(slug: str) -> str:
    """Repo path for a post page (clean URLs: /blog/<slug>/)."""
    return f"blog/{slug}/index.html"


def post_url(slug: str) -> str:
    return f"/blog/{slug}/"


def post_canonical(slug: str) -> str:
    return f"{SITE_BASE_URL}/blog/{slug}/"


# ---------------------------------------------------------------------------
# Image collection
# ---------------------------------------------------------------------------
def collect_image_paths(content: str) -> List[str]:
    """Return the unique ArtifactStorage paths referenced by a post, in order."""
    seen: List[str] = []
    for match in _IMG_REF_RE.finditer(content or ""):
        path = match.group(1)
        if path not in seen:
            seen.append(path)
    return seen


async def fetch_images(content: str, storage=None) -> List[ExportedImage]:
    """Resolve every referenced image to bytes via ArtifactStorage.

    ``storage`` is any object exposing ``async retrieve(path) -> bytes|None``;
    defaults to the app's global ArtifactStorage. Missing files are returned
    with ``missing=True`` (not fatal) so the caller can warn/skip.
    """
    if storage is None:
        from backend.app.storage.artifacts import get_artifact_storage

        storage = get_artifact_storage()

    images: List[ExportedImage] = []
    for path in collect_image_paths(content):
        data = await storage.retrieve(path)
        images.append(
            ExportedImage(
                storage_path=path,
                repo_path=f"blog/images/{path}",
                url=f"/blog/images/{path}",
                data=data,
                size=len(data) if data else 0,
                missing=data is None,
            )
        )
    return images


# ---------------------------------------------------------------------------
# Page shell (mirrors mindrouter.ai)
# ---------------------------------------------------------------------------
_THEME_INIT = """<script>
    (function() {
        var t = localStorage.getItem('mr-theme') ||
                (matchMedia('(prefers-color-scheme:dark)').matches ? 'dark' : 'light');
        document.documentElement.setAttribute('data-bs-theme', t);
    })();
    </script>"""

_HEAD_LINKS = (
    '<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">\n'
    '<link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css" rel="stylesheet">\n'
    '<link href="/css/style.css" rel="stylesheet">\n'
    '<link href="/css/blog.css" rel="stylesheet">\n'
    '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/pygments-css@1.0.0/github.min.css">'
)


def _navbar(active_blog: bool = True) -> str:
    active = " active" if active_blog else ""
    return f"""<nav class="navbar navbar-expand-lg navbar-dark bg-dark sticky-top" aria-label="Main navigation">
    <div class="container">
        <a class="navbar-brand" href="/"><i class="bi bi-router"></i> MindRouter</a>
        <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
            <span class="navbar-toggler-icon"></span>
        </button>
        <div class="collapse navbar-collapse" id="navbarNav">
            <ul class="navbar-nav me-auto">
                <li class="nav-item"><a class="nav-link" href="/#features">Features</a></li>
                <li class="nav-item"><a class="nav-link" href="/#telemetry">Telemetry</a></li>
                <li class="nav-item"><a class="nav-link" href="/configurator.html"><i class="bi bi-sliders"></i> Configurator</a></li>
                <li class="nav-item"><a class="nav-link" href="/#docs">Docs</a></li>
                <li class="nav-item"><a class="nav-link{active}" href="/blog/">Blog</a></li>
                <li class="nav-item"><a class="nav-link" href="https://github.com/ui-insight/MindRouter" target="_blank"><i class="bi bi-github"></i> GitHub</a></li>
                <li class="nav-item"><a class="nav-link" href="/#contact">Contact</a></li>
            </ul>
            <ul class="navbar-nav">
                <li class="nav-item">
                    <button class="theme-toggle nav-link" id="themeToggleBtn" title="Toggle dark mode" aria-label="Toggle dark mode"><i class="bi bi-sun-fill"></i></button>
                </li>
            </ul>
        </div>
    </div>
</nav>"""


_FOOTER = """<footer class="site-footer py-4 mt-5">
    <div class="container text-center">
        <p class="mb-2" style="font-size:1.1rem; font-weight:600;">
            <i class="bi bi-router" style="color:var(--mindrouter-primary);"></i> MindRouter
        </p>
        <p class="text-muted mb-2" style="font-size:0.9rem;">Open-Source LLM Inference Load Balancer</p>
        <p class="text-muted mb-2" style="font-size:0.82rem;">
            Developed by <a href="https://hpc.uidaho.edu" class="text-muted">Research Computing &amp; Data Services (RCDS)</a>
            at the <a href="https://www.uidaho.edu" class="text-muted">University of Idaho</a>
            &middot;
            <a href="https://www.iids.uidaho.edu" class="text-muted">Institute for Interdisciplinary Data Sciences (IIDS)</a>
        </p>
        <p class="text-muted mb-2" style="font-size:0.82rem;">
            <a href="https://github.com/ui-insight/MindRouter" class="text-muted"><i class="bi bi-github"></i> GitHub</a>
            &nbsp;&middot;&nbsp; Apache 2.0 License
        </p>
    </div>
</footer>"""

_SCRIPTS = """<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
<script>
    (function() {
        var btn = document.getElementById('themeToggleBtn');
        if (!btn) return;
        var icon = btn.querySelector('i');
        function updateIcon() {
            var isDark = document.documentElement.getAttribute('data-bs-theme') === 'dark';
            if (icon) icon.className = isDark ? 'bi bi-moon-fill' : 'bi bi-sun-fill';
        }
        updateIcon();
        btn.addEventListener('click', function() {
            var next = document.documentElement.getAttribute('data-bs-theme') === 'dark' ? 'light' : 'dark';
            document.documentElement.setAttribute('data-bs-theme', next);
            localStorage.setItem('mr-theme', next);
            updateIcon();
        });
    })();
</script>"""


def _page(*, title: str, description: str, canonical: str, body: str,
          head_extra: str = "", og_type: str = "website",
          og_title: Optional[str] = None) -> str:
    """Assemble a full standalone HTML page in the mindrouter.ai shell."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
    <title>{_esc(title)}</title>
    <meta name="description" content="{_esc(description)}">
    <meta property="og:title" content="{_esc(og_title or title)}">
    <meta property="og:description" content="{_esc(description)}">
    <meta property="og:type" content="{og_type}">
    <meta property="og:url" content="{_esc(canonical)}">
    <link rel="canonical" href="{_esc(canonical)}">
    {_THEME_INIT}
    {_HEAD_LINKS}
    {head_extra}
</head>
<body>
    <a class="skip-link" href="#main-content">Skip to main content</a>
    {_navbar()}
    <main id="main-content">
{body}
    </main>
    {_FOOTER}
    {_SCRIPTS}
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Public render functions
# ---------------------------------------------------------------------------
def _author_name(post) -> str:
    author = getattr(post, "author", None)
    if author is None:
        return "MindRouter"
    return getattr(author, "full_name", None) or getattr(author, "username", None) or "MindRouter"


def _fmt_date(dt: Optional[datetime]) -> str:
    # Portable "July 8, 2026" (avoid the non-portable %-d strftime extension).
    return f"{dt.strftime('%B')} {dt.day}, {dt.year}" if dt else ""


def render_post_html(post) -> str:
    """Render one post to a full standalone HTML page.

    ``post`` is any object with: slug, title, content, excerpt, published_at,
    updated_at, and an ``author`` with full_name/username (a BlogPost or a
    test double). Images are NOT fetched here (see fetch_images); their URLs
    are left root-relative so they resolve on the target site.
    """
    slug = post.slug
    title = post.title
    content_html = render_markdown(post.content)
    description = derive_description(getattr(post, "excerpt", None), post.content)
    canonical = post_canonical(slug)
    author = _author_name(post)
    published_at = getattr(post, "published_at", None)
    date_display = _fmt_date(published_at)

    json_ld = {
        "@context": "https://schema.org",
        "@type": "BlogPosting",
        "headline": title,
        "description": description,
        "url": canonical,
        "author": {"@type": "Organization", "name": author},
        "publisher": {"@type": "Organization", "name": "MindRouter"},
    }
    if published_at:
        json_ld["datePublished"] = published_at.isoformat()
    updated_at = getattr(post, "updated_at", None)
    if updated_at:
        json_ld["dateModified"] = updated_at.isoformat()
    # Escape < > & so a title containing "</script>" cannot break out of the
    # JSON-LD <script> block (still valid JSON via \uXXXX escapes).
    ld_json = (
        json.dumps(json_ld)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    head_extra = f'<script type="application/ld+json">{ld_json}</script>'

    byline = f"By {_esc(author)}"
    if date_display:
        byline += f" &middot; {_esc(date_display)}"

    body = f"""        <div class="container py-5">
            <div class="row">
                <div class="col-lg-8 mx-auto">
                    <a href="/blog/" class="text-decoration-none small d-inline-block mb-3">&larr; Back to Blog</a>
                    <h1 class="mb-2">{_esc(title)}</h1>
                    <p class="text-muted mb-4">{byline}</p>
                    <article class="blog-content">
{content_html}
                    </article>
                </div>
            </div>
        </div>"""

    return _page(
        title=f"{title} — MindRouter Blog",
        og_title=title,
        description=description,
        canonical=canonical,
        body=body,
        head_extra=head_extra,
        og_type="article",
    )


def render_index_html(posts) -> str:
    """Render the blog listing page from an ordered list of posts.

    Caller passes ONLY the website-selected posts, newest first.
    """
    items = []
    for post in posts:
        slug = post.slug
        title = post.title
        date_display = _fmt_date(getattr(post, "published_at", None))
        author = _author_name(post)
        excerpt = derive_description(getattr(post, "excerpt", None), getattr(post, "content", ""), limit=240)
        meta = _esc(date_display)
        if date_display and author:
            meta += " &middot; "
        meta += _esc(author)
        items.append(
            f"""                    <article class="border-bottom py-4">
                        <h2 class="h4 mb-1"><a class="text-decoration-none" href="{_esc(post_url(slug))}">{_esc(title)}</a></h2>
                        <p class="text-muted small mb-2">{meta}</p>
                        <p class="mb-2">{_esc(excerpt)}</p>
                        <a class="text-decoration-none" href="{_esc(post_url(slug))}">Read more &rarr;</a>
                    </article>"""
        )
    if not items:
        items.append('                    <p class="text-muted">No posts yet. Check back soon.</p>')

    body = f"""        <div class="container py-5">
            <div class="row">
                <div class="col-lg-8 mx-auto">
                    <h1 class="mb-1">Blog</h1>
                    <p class="text-muted mb-4">Updates, tutorials, and best practices from the MindRouter team.</p>
{chr(10).join(items)}
                </div>
            </div>
        </div>"""

    return _page(
        title="Blog — MindRouter",
        description="Updates, tutorials, and best practices from the MindRouter team.",
        canonical=f"{SITE_BASE_URL}/blog/",
        body=body,
    )


def render_feed_xml(posts) -> str:
    """Render an RSS 2.0 feed from the website-selected posts (newest first)."""
    def rfc822(dt: Optional[datetime]) -> str:
        return dt.strftime("%a, %d %b %Y %H:%M:%S %z") if dt else ""

    entries = []
    for post in posts:
        canonical = post_canonical(post.slug)
        desc = derive_description(getattr(post, "excerpt", None), getattr(post, "content", ""), limit=300)
        pub = rfc822(getattr(post, "published_at", None))
        entries.append(
            "    <item>\n"
            f"      <title>{_esc(post.title)}</title>\n"
            f"      <link>{_esc(canonical)}</link>\n"
            f"      <guid isPermaLink=\"true\">{_esc(canonical)}</guid>\n"
            + (f"      <pubDate>{_esc(pub)}</pubDate>\n" if pub else "")
            + f"      <description>{_esc(desc)}</description>\n"
            "    </item>"
        )
    body = "\n".join(entries)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<rss version="2.0">\n'
        "  <channel>\n"
        "    <title>MindRouter Blog</title>\n"
        f"    <link>{SITE_BASE_URL}/blog/</link>\n"
        "    <description>Updates, tutorials, and best practices from the MindRouter team.</description>\n"
        f"{body}\n"
        "  </channel>\n"
        "</rss>\n"
    )


async def export_post(post, storage=None) -> ExportedPost:
    """Full export of one post: standalone HTML + resolved images."""
    page_html = render_post_html(post)
    images = await fetch_images(post.content, storage=storage)
    return ExportedPost(
        slug=post.slug,
        title=post.title,
        html=page_html,
        repo_path=post_repo_path(post.slug),
        url=post_url(post.slug),
        canonical=post_canonical(post.slug),
        images=images,
    )


# CSS for the generated pages (written to the repo as css/blog.css by the
# publisher, PR3). Mirrors dashboard/templates/blog/post.html so highlighting
# and dark-mode token colors match the app exactly.
BLOG_CSS = """/* MindRouter blog article styles (generated; see blog_export.BLOG_CSS) */
.blog-content { line-height: 1.8; }
.blog-content h1, .blog-content h2, .blog-content h3 { margin-top: 1.5em; margin-bottom: 0.5em; }
.blog-content pre { background: var(--bs-tertiary-bg); color: var(--bs-body-color); padding: 1rem; border-radius: 6px; overflow-x: auto; }
.blog-content code { font-size: 0.9em; }
.blog-content p code { background: var(--bs-tertiary-bg); color: var(--bs-body-color); padding: 0.2em 0.4em; border-radius: 3px; }
.blog-content img { max-width: 100%; height: auto; }
.blog-content table { width: 100%; margin-bottom: 1rem; }
.blog-content table th, .blog-content table td { padding: 0.5rem; border: 1px solid var(--bs-border-color); }
.blog-content table th { background: var(--bs-tertiary-bg); }
.codehilite { background: var(--bs-tertiary-bg); color: var(--bs-body-color); padding: 1rem; border-radius: 6px; overflow-x: auto; }
[data-bs-theme="dark"] .codehilite,
[data-bs-theme="dark"] .blog-content pre { background: #1e1e2e; color: #cdd6f4; }
[data-bs-theme="dark"] .codehilite span { color: #cdd6f4; }
[data-bs-theme="dark"] .codehilite .c,
[data-bs-theme="dark"] .codehilite .cm,
[data-bs-theme="dark"] .codehilite .c1,
[data-bs-theme="dark"] .codehilite .cs { color: #6c7086; }
[data-bs-theme="dark"] .codehilite .k,
[data-bs-theme="dark"] .codehilite .kc,
[data-bs-theme="dark"] .codehilite .kd,
[data-bs-theme="dark"] .codehilite .kn,
[data-bs-theme="dark"] .codehilite .kp,
[data-bs-theme="dark"] .codehilite .kr,
[data-bs-theme="dark"] .codehilite .kt { color: #cba6f7; }
[data-bs-theme="dark"] .codehilite .s,
[data-bs-theme="dark"] .codehilite .s1,
[data-bs-theme="dark"] .codehilite .s2,
[data-bs-theme="dark"] .codehilite .se { color: #a6e3a1; }
[data-bs-theme="dark"] .codehilite .n,
[data-bs-theme="dark"] .codehilite .na,
[data-bs-theme="dark"] .codehilite .nb,
[data-bs-theme="dark"] .codehilite .nc,
[data-bs-theme="dark"] .codehilite .nf { color: #89b4fa; }
[data-bs-theme="dark"] .codehilite .mi,
[data-bs-theme="dark"] .codehilite .mf { color: #fab387; }
[data-bs-theme="dark"] .codehilite .o,
[data-bs-theme="dark"] .codehilite .p { color: #89dceb; }
"""
