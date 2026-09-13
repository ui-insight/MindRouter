############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_email_branding.py: Branding applied to outgoing emails
#
# Research Computing and Data Services (RCDS)
# University of Idaho
#
############################################################

"""Emails must follow the shared branding config: no hardcoded blue, the brand
accent (contrast-adjusted for legibility), the org name, and the raster email
logo embedded via CID. ``aiosmtplib`` is stubbed so the module imports without
the optional SMTP dependency installed."""

import sys
import types

import pytest

# Stub aiosmtplib (optional dep) before importing the email service.
if "aiosmtplib" not in sys.modules:
    _stub = types.ModuleType("aiosmtplib")
    _stub.SMTP = object
    _stub.SMTPException = Exception
    sys.modules["aiosmtplib"] = _stub

from backend.app.services import branding  # noqa: E402
from backend.app.services import email_service as es  # noqa: E402

GOLD = "#F1B300"


@pytest.fixture
def gold_brand_with_logo(tmp_path, monkeypatch):
    """U of I-style gold branding with a raster email logo configured."""
    monkeypatch.setattr(branding, "storage_dir", lambda: str(tmp_path))
    png = b"\x89PNG\r\n\x1a\n" + b"0" * 64
    stored = branding.save_asset("email_logo", "logo.png", png)
    branding._CACHE = branding._build_view(
        {"app_name": "MindRouter", "primary_light": GOLD, "primary_dark": GOLD, "email_logo": stored}
    )
    return branding.get_branding()


def test_wrapper_has_no_bsu_blue(gold_brand_with_logo):
    html = es._wrap_html("<p>hi</p>", base_url="https://x")
    assert "003da5" not in html.lower(), "the old blue #003DA5 must be gone"


def test_wrapper_uses_gold_accent_and_cid_logo(gold_brand_with_logo):
    html = es._wrap_html("<p>hi</p>", base_url="https://x")
    assert "border-bottom:3px solid #f1b300" in html      # gold accent rule
    assert "cid:brandlogo" in html                        # logo referenced for CID embed
    assert "color:#906900" in html                        # footer link uses accessible gold ink


def test_wrapper_falls_back_to_app_name_without_logo(monkeypatch):
    branding._CACHE = branding._build_view({"app_name": "Acme University", "primary_light": GOLD, "primary_dark": GOLD})
    html = es._wrap_html("<p>hi</p>", base_url="https://x")
    assert "cid:brandlogo" not in html
    assert ">Acme University<" in html


def test_content_with_braces_is_not_reformatted(gold_brand_with_logo):
    # Regression: the old _EMAIL_WRAPPER.format() would crash on code containing braces.
    body = '<pre>{"json": true}</pre>'
    html = es._wrap_html(body, base_url="https://x")
    assert '{"json": true}' in html


def test_blog_email_title_uses_headline_color_default_neutral(gold_brand_with_logo):
    # Headline uses the configurable headline_color, which defaults to neutral
    # near-black (not the gold accent shade). Button + links keep the gold accent.
    html = es._render_blog_email("Post", "# Hi\n\ntext", "post", "Luke", "https://x")
    assert "003da5" not in html.lower()
    assert "margin:0 0 16px 0;color:#231f20;" in html      # headline = neutral default
    assert "background:#f1b300;color:#000000" in html      # gold button, black (legible) text
    assert "color:#906900" in html                         # footer link keeps the gold accent


def test_blog_email_headline_color_is_configurable(monkeypatch):
    # An admin-set headline color flows through to the email title.
    branding._CACHE = branding._build_view(
        {"app_name": "MindRouter", "primary_light": GOLD, "primary_dark": GOLD, "headline_color": "#008080"}
    )
    html = es._render_blog_email("Post", "# Hi\n\ntext", "post", "Luke", "https://x")
    h2 = html[html.find("<h2"):html.find("</h2>")]
    assert "color:#008080" in h2                            # Clearwater teal headline
    assert "#231f20" not in h2                              # neutral default no longer used in the headline


def test_default_branding_has_no_blue():
    branding._CACHE = branding._build_view({})
    html = es._wrap_html("<p>hi</p>", base_url="https://x")
    assert "003da5" not in html.lower()


@pytest.mark.asyncio
async def test_send_one_embeds_cid_logo(gold_brand_with_logo):
    captured = {}

    class FakeSMTP:
        async def send_message(self, msg):
            captured["msg"] = msg

    html = es._wrap_html("<p>hi</p>", base_url="https://x")  # references cid:brandlogo
    await es._send_one(FakeSMTP(), "from@x", "to@y", "Subj", html)
    msg = captured["msg"]
    assert msg.get_content_type() == "multipart/related"
    ctypes = [p.get_content_type() for p in msg.walk()]
    assert "multipart/alternative" in ctypes and "image/png" in ctypes
    img = next(p for p in msg.walk() if p.get_content_type() == "image/png")
    assert img.get("Content-ID") == "<brandlogo>"
    assert "inline" in img.get("Content-Disposition", "")


@pytest.mark.asyncio
async def test_send_one_without_logo_is_plain_alternative(monkeypatch):
    branding._CACHE = branding._build_view({"app_name": "MindRouter"})  # no email logo
    captured = {}

    class FakeSMTP:
        async def send_message(self, msg):
            captured["msg"] = msg

    html = es._wrap_html("<p>hi</p>", base_url="https://x")
    await es._send_one(FakeSMTP(), "a@x", "b@y", "S", html)
    msg = captured["msg"]
    assert msg.get_content_type() == "multipart/alternative"
    assert not any(p.get_content_type().startswith("image/") for p in msg.walk())


# ---------------------------------------------------------------------------
# Self-contained blog emails: images shrunk + embedded as CID parts, <details>
# (web-only expandable notes) dropped from the email copy.
# ---------------------------------------------------------------------------


def _png(width, height, mode="RGB"):
    from io import BytesIO
    from PIL import Image
    buf = BytesIO()
    Image.new(mode, (width, height), (200, 30, 30) if mode == "RGB" else (200, 30, 30, 128)).save(buf, "PNG")
    return buf.getvalue()


def test_shrink_image_downscales_to_email_width_as_jpeg():
    from io import BytesIO
    from PIL import Image
    data, subtype = es.shrink_image_for_email(_png(2400, 1200))
    assert subtype == "jpeg"
    im = Image.open(BytesIO(data))
    assert im.size == (es._EMAIL_IMAGE_MAX_WIDTH, es._EMAIL_IMAGE_MAX_WIDTH // 2)


def test_shrink_image_keeps_alpha_as_png_and_small_images_unscaled():
    from io import BytesIO
    from PIL import Image
    data, subtype = es.shrink_image_for_email(_png(300, 100, "RGBA"))
    assert subtype == "png"
    assert Image.open(BytesIO(data)).size == (300, 100)  # already narrower than the cap


def test_shrink_image_passes_unreadable_bytes_through():
    assert es.shrink_image_for_email(b"not an image") == (b"not an image", "png")


def test_blog_email_uses_cid_for_embedded_images_and_absolute_urls_otherwise(gold_brand_with_logo):
    md = (
        'Hero: <img src="/blog/images/2026/09/13/aa/hero.png" alt="hero" style="width:100%">\n\n'
        "![shot](/blog/images/2026/09/13/bb/shot.png)\n\n"
        '<img src="/blog/images/2026/09/13/cc/not-embedded.png">'
    )
    inline = {
        "2026/09/13/aa/hero.png": (b"\x89PNG-hero", "png"),
        "2026/09/13/bb/shot.png": (b"\xff\xd8-shot", "jpeg"),
    }
    html = es._render_blog_email("T", md, "slug", "Admin", "https://x", inline_images=inline)
    assert 'src="cid:blogimg1"' in html and 'src="cid:blogimg2"' in html
    # the image past the budget keeps a remote link; imgs without an author
    # style get an inline size, an existing style attribute is left alone
    assert 'src="https://x/blog/images/2026/09/13/cc/not-embedded.png"' in html
    assert html.count("max-width:100%;height:auto;") == 2
    assert 'style="width:100%"' in html
    atts = es.blog_inline_attachments(inline)
    assert atts == {"blogimg1": inline["2026/09/13/aa/hero.png"], "blogimg2": inline["2026/09/13/bb/shot.png"]}


def test_blog_email_without_inline_map_keeps_absolute_urls(gold_brand_with_logo):
    html = es._render_blog_email("T", "![a](/blog/images/x/y.png)", "s", "A", "https://x")
    assert 'src="https://x/blog/images/x/y.png"' in html and "cid:blogimg" not in html
    assert es.blog_inline_attachments(None) == {}


def test_blog_email_drops_details_blocks(gold_brand_with_logo):
    md = (
        "Intro paragraph.\n\n"
        "<details>\n<summary><strong>Gotchas</strong></summary>\n<ul><li>secret gotcha text</li></ul>\n</details>\n\n"
        "Closing paragraph."
    )
    html = es._render_blog_email("T", md, "s", "A", "https://x")
    assert "secret gotcha text" not in html and "<details" not in html
    assert "omitted from the email version" in html
    assert "Intro paragraph." in html and "Closing paragraph." in html


@pytest.mark.asyncio
async def test_send_one_embeds_blog_images_alongside_logo(gold_brand_with_logo):
    captured = {}

    class FakeSMTP:
        async def send_message(self, msg):
            captured["msg"] = msg

    html = es._wrap_html('<img src="cid:blogimg1"><p>hi</p>', base_url="https://x")
    await es._send_one(FakeSMTP(), "from@x", "to@y", "Subj", html, {"blogimg1": (b"\xff\xd8jpg", "jpeg")})
    msg = captured["msg"]
    assert msg.get_content_type() == "multipart/related"
    ids = {p.get("Content-ID") for p in msg.walk() if p.get_content_type().startswith("image/")}
    assert ids == {"<blogimg1>", "<brandlogo>"}
    jpg = next(p for p in msg.walk() if p.get_content_type() == "image/jpeg")
    assert "inline" in jpg.get("Content-Disposition", "") and "blogimg1.jpeg" in jpg.get("Content-Disposition", "")


@pytest.mark.asyncio
async def test_load_blog_inline_images_fetches_shrinks_and_respects_budget(monkeypatch):
    import types
    big = _png(1600, 800)

    class FakeStorage:
        async def retrieve(self, path):
            return None if path.endswith("missing.png") else big

    fake_mod = types.ModuleType("backend.app.storage.artifacts")
    fake_mod.get_artifact_storage = lambda: FakeStorage()
    pkg = types.ModuleType("backend.app.storage"); pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "backend.app.storage", pkg)
    monkeypatch.setitem(sys.modules, "backend.app.storage.artifacts", fake_mod)

    md = ('<img src="/blog/images/a/one.png"> ![b](/blog/images/b/two.png) '
          '<img src="/blog/images/a/one.png"> <img src="/blog/images/m/missing.png">')
    images = await es.load_blog_inline_images(md)
    assert list(images) == ["a/one.png", "b/two.png"]  # de-duplicated, document order, missing skipped
    from io import BytesIO
    from PIL import Image
    assert Image.open(BytesIO(images["a/one.png"][0])).width == es._EMAIL_IMAGE_MAX_WIDTH

    # a tiny budget keeps the first image and drops the rest (they stay remote links)
    monkeypatch.setattr(es, "_EMAIL_IMAGE_BUDGET_BYTES", len(images["a/one.png"][0]) + 1)
    assert list(await es.load_blog_inline_images(md)) == ["a/one.png"]
