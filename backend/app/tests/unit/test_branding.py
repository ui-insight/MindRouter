############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_branding.py: Unit tests for the UI branding service
#
# Research Computing and Data Services (RCDS)
# University of Idaho
#
############################################################

"""Unit tests for backend.app.services.branding.

Covers the pure color/normalization logic, the template-ready view builder,
and on-disk asset save/resolve/delete (traversal-safe). No live DB is used;
the DB-backed loaders are exercised via the render tests elsewhere.
"""

import pytest

from backend.app.services import branding


# ── Color helpers ────────────────────────────────────────────────

def test_is_valid_hex():
    assert branding.is_valid_hex("#0d6efd")
    assert branding.is_valid_hex("#abc")
    assert not branding.is_valid_hex("0d6efd")       # missing #
    assert not branding.is_valid_hex("#12345")       # wrong length
    assert not branding.is_valid_hex("#gggggg")      # non-hex
    assert not branding.is_valid_hex(None)
    assert not branding.is_valid_hex(123)


def test_normalize_hex_expands_and_falls_back():
    assert branding._normalize_hex("#ABC", "#000000") == "#aabbcc"
    assert branding._normalize_hex("#0D6EFD", "#000000") == "#0d6efd"
    assert branding._normalize_hex("nonsense", "#0d6efd") == "#0d6efd"
    assert branding._normalize_hex(None, "#123456") == "#123456"


def test_hex_to_rgb():
    assert branding._hex_to_rgb("#000000") == (0, 0, 0)
    assert branding._hex_to_rgb("#ffffff") == (255, 255, 255)
    assert branding._hex_to_rgb("#0d6efd") == (13, 110, 253)


def test_shade_darken_and_lighten():
    # Darkening black stays black; lightening white stays white (clamped).
    assert branding._shade("#000000", -0.5) == "#000000"
    assert branding._shade("#ffffff", 0.5) == "#ffffff"
    # Darkening reduces channels.
    darker = branding._shade("#808080", -0.5)
    assert darker == "#404040"


# ── View builder ─────────────────────────────────────────────────

def test_build_view_defaults():
    v = branding._build_view({})
    assert v["app_name"] == branding.DEFAULTS["app_name"]
    assert v["tagline"] == branding.DEFAULTS["tagline"]
    assert v["primary_light"] == "#0d6efd"
    assert v["primary_light_rgb"] == "13, 110, 253"
    assert v["logo_light_url"] is None
    assert v["logo_dark_url"] is None
    assert v["favicon_url"] is None
    assert v["has_custom_logo"] is False
    assert v["is_customized"] is False
    # Derived shades present for both themes.
    for k in ("primary_light_hover", "primary_dark_hover",
              "primary_light_active", "primary_dark_active"):
        assert branding.is_valid_hex(v[k]), k


def test_build_view_custom_and_asset_urls():
    v = branding._build_view({
        "app_name": "Acme University",
        "tagline": "Powered by Acme",
        "primary_light": "#8B1E3F",
        "primary_dark": "#e0729a",
        "logo_light": "logo_light-abc123.png",
        "logo_dark": "logo_dark-def456.svg",
        "favicon": "favicon-999.ico",
    })
    assert v["app_name"] == "Acme University"
    assert v["primary_light"] == "#8b1e3f"           # normalized to lowercase
    assert v["logo_light_url"] == "/branding/asset/logo_light-abc123.png"
    assert v["logo_dark_url"] == "/branding/asset/logo_dark-def456.svg"
    assert v["favicon_url"] == "/branding/asset/favicon-999.ico"
    assert v["has_custom_logo"] is True
    assert v["is_customized"] is True


def test_build_view_invalid_color_falls_back_to_default():
    v = branding._build_view({"primary_light": "red", "primary_dark": "#zzz"})
    assert v["primary_light"] == branding.DEFAULTS["primary_light"]
    assert v["primary_dark"] == branding.DEFAULTS["primary_dark"]


def test_get_branding_never_empty():
    branding._CACHE = {}
    v = branding.get_branding()
    assert v["app_name"] == branding.DEFAULTS["app_name"]


# ── Asset storage (filesystem) ───────────────────────────────────

@pytest.fixture
def tmp_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(branding, "storage_dir", lambda: str(tmp_path))
    return tmp_path


def test_save_asset_writes_file_with_slot_prefix(tmp_storage):
    stored = branding.save_asset("logo_light", "mylogo.PNG", b"\x89PNG\r\n\x1a\n")
    assert stored.startswith("logo_light-")
    assert stored.endswith(".png")            # extension lower-cased
    assert (tmp_storage / stored).read_bytes() == b"\x89PNG\r\n\x1a\n"


def test_save_asset_rejects_bad_extension(tmp_storage):
    with pytest.raises(ValueError):
        branding.save_asset("logo_light", "evil.exe", b"data")


def test_save_asset_favicon_allows_ico(tmp_storage):
    stored = branding.save_asset("favicon", "fav.ico", b"icodata")
    assert stored.endswith(".ico")


def test_save_asset_favicon_rejects_webp(tmp_storage):
    # webp is allowed for logos but not favicons.
    with pytest.raises(ValueError):
        branding.save_asset("favicon", "fav.webp", b"data")


def test_asset_path_resolves_and_blocks_traversal(tmp_storage):
    stored = branding.save_asset("logo_dark", "l.svg", b"<svg/>")
    assert branding.asset_path(stored) is not None
    # Traversal / absolute / hidden names are rejected.
    assert branding.asset_path("../../etc/passwd") is None
    assert branding.asset_path("/etc/passwd") is None
    assert branding.asset_path(".hidden") is None
    assert branding.asset_path("does-not-exist.png") is None


def test_delete_asset_removes_file(tmp_storage):
    stored = branding.save_asset("logo_light", "l.png", b"x")
    assert branding.asset_path(stored) is not None
    branding.delete_asset(stored)
    assert branding.asset_path(stored) is None
    # Deleting a missing / None asset is a no-op (no exception).
    branding.delete_asset(None)
    branding.delete_asset("nope.png")


def test_content_type_for():
    assert branding.content_type_for("x.png") == "image/png"
    assert branding.content_type_for("x.svg") == "image/svg+xml"
    assert branding.content_type_for("x.ico") == "image/x-icon"
    assert branding.content_type_for("x.unknown") == "application/octet-stream"


# ── Accessible foreground / ink derivation ───────────────────────

def test_contrast_bounds():
    assert round(branding._contrast("#000000", "#ffffff"), 1) == 21.0
    assert branding._contrast("#123456", "#123456") == 1.0


def test_best_fg_prefers_white_but_rescues_light_accents():
    # Conventional mid-tone accents keep white text (Bootstrap convention).
    assert branding._best_fg("#0d6efd") == "#ffffff"   # blue
    assert branding._best_fg("#dc3545") == "#ffffff"   # red
    # Light accents flip to black so text stays legible.
    assert branding._best_fg("#f1b300") == "#000000"   # U of I Pride Gold
    assert branding._best_fg("#ffc107") == "#000000"   # warning yellow
    # The chosen fg always clears the 3.0 UI-text threshold on its fill.
    for accent in ("#0d6efd", "#f1b300", "#ffc107", "#dc3545", "#008080"):
        assert branding._contrast(branding._best_fg(accent), accent) >= 3.0, accent


def test_accessible_ink_meets_target_on_body_bg():
    # A light accent on a white page is darkened until it reads as text.
    ink_light = branding._accessible_ink("#f1b300", "#ffffff")
    assert branding._contrast(ink_light, "#ffffff") >= 4.5
    assert ink_light != "#f1b300"                       # was too light, got darkened
    # The same gold on a dark page already passes, so it's left as-is.
    ink_dark = branding._accessible_ink("#f1b300", "#1a1d21")
    assert ink_dark == "#f1b300"
    assert branding._contrast(ink_dark, "#1a1d21") >= 4.5


def test_build_view_exposes_on_and_ink_for_both_themes():
    v = branding._build_view({"primary_light": "#F1B300", "primary_dark": "#F1B300"})
    for k in ("primary_light_on", "primary_dark_on",
              "primary_light_ink", "primary_dark_ink"):
        assert branding.is_valid_hex(v[k]), k
    # Gold button text is black and high-contrast; light-mode links are darkened.
    assert v["primary_light_on"] == "#000000"
    assert branding._contrast(v["primary_light_ink"], "#ffffff") >= 4.5


def test_default_look_unchanged():
    """Unbranded install keeps stock blue with white button text."""
    d = branding._build_view({})
    assert d["primary_light_on"] == "#ffffff"
    assert d["primary_light_ink"] == "#0d6efd"
