"""UI branding / theming service.

Administrators can rebrand the MindRouter dashboard (organization name, tagline,
navbar logos, favicon, and light/dark accent colors) so a single deployment can
match one institution's visual identity. All values are stored as ``branding.*``
rows in the ``app_config`` key-value table; uploaded logo/favicon *files* live on
disk under ``settings.branding_storage_path`` (the DB TEXT column is too small
for base64 images).

Branding must be readable synchronously from Jinja templates on *every* page
without threading a value through ~50 route handlers. We keep a module-level
cache (``_CACHE``) that:

* is loaded once at startup (``refresh_branding_cache`` in the app lifespan),
* is refreshed on a short interval by a background loop (so a save in one uvicorn
  worker propagates to the others within ``_REFRESH_INTERVAL`` seconds), and
* is invalidated immediately in the worker that handles an admin save.

``get_branding()`` is the synchronous accessor registered as the Jinja global
``branding`` and read by ``base.html``.
"""

from __future__ import annotations

import asyncio
import os
import re
import secrets
from typing import Any, Optional

from backend.app.db import crud
from backend.app.db.session import get_async_db_context
from backend.app.logging_config import get_logger
from backend.app.settings import get_settings

logger = get_logger(__name__)

# --- Config keys -----------------------------------------------------------
KEY_APP_NAME = "branding.app_name"
KEY_TAGLINE = "branding.tagline"
KEY_PRIMARY_LIGHT = "branding.primary_light"
KEY_PRIMARY_DARK = "branding.primary_dark"
KEY_LOGO_LIGHT = "branding.logo_light"      # stored filename (in branding_storage_path)
KEY_LOGO_DARK = "branding.logo_dark"        # stored filename
KEY_FAVICON = "branding.favicon"            # stored filename

# Product defaults (used when a value is unset or invalid). These mirror the
# stock MindRouter look so an un-branded install is unchanged.
DEFAULTS: dict[str, Any] = {
    "app_name": "MindRouter",
    "tagline": "LLM Inference Load Balancer",
    "primary_light": "#0d6efd",
    "primary_dark": "#3d8bfd",
    "logo_light": None,
    "logo_dark": None,
    "favicon": None,
}

_HEX_RE = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")

# Allowed image types for uploads: extension -> content-type.
ALLOWED_IMAGE_EXT: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
    ".gif": "image/gif",
}
ALLOWED_FAVICON_EXT: dict[str, str] = {
    ".ico": "image/x-icon",
    ".png": "image/png",
    ".svg": "image/svg+xml",
}

_REFRESH_INTERVAL = 15  # seconds; how quickly a save propagates across workers

# Module-level cache, seeded with defaults so templates work before first load.
_CACHE: dict[str, Any] = {}


# --- Color helpers ---------------------------------------------------------
def is_valid_hex(value: Any) -> bool:
    return isinstance(value, str) and bool(_HEX_RE.match(value.strip()))


def _normalize_hex(value: Any, fallback: str) -> str:
    """Return a canonical ``#rrggbb`` string, or ``fallback`` if invalid."""
    if not is_valid_hex(value):
        return fallback
    v = value.strip().lower()
    if len(v) == 4:  # #abc -> #aabbcc
        v = "#" + "".join(c * 2 for c in v[1:])
    return v


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    v = hex_color.lstrip("#")
    return (int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16))


def _shade(hex_color: str, factor: float) -> str:
    """Lighten (factor>0) or darken (factor<0) a hex color by ``|factor|``.

    Used to derive button hover/active shades from the accent color.
    """
    r, g, b = _hex_to_rgb(hex_color)
    if factor >= 0:
        r = int(r + (255 - r) * factor)
        g = int(g + (255 - g) * factor)
        b = int(b + (255 - b) * factor)
    else:
        f = 1 + factor
        r, g, b = int(r * f), int(g * f), int(b * f)
    clamp = lambda x: max(0, min(255, x))  # noqa: E731
    return f"#{clamp(r):02x}{clamp(g):02x}{clamp(b):02x}"


def _derive_colors(primary_light: str, primary_dark: str) -> dict[str, str]:
    """Compute the extra color values base.html needs for Bootstrap overrides."""
    return {
        "primary_light": primary_light,
        "primary_dark": primary_dark,
        "primary_light_rgb": ", ".join(str(c) for c in _hex_to_rgb(primary_light)),
        "primary_dark_rgb": ", ".join(str(c) for c in _hex_to_rgb(primary_dark)),
        "primary_light_hover": _shade(primary_light, -0.12),
        "primary_dark_hover": _shade(primary_dark, -0.12),
        "primary_light_active": _shade(primary_light, -0.20),
        "primary_dark_active": _shade(primary_dark, -0.20),
    }


# --- Load / cache ----------------------------------------------------------
def _asset_url(filename: Optional[str]) -> Optional[str]:
    """Public URL for an uploaded asset, cache-busted by its filename token."""
    if not filename:
        return None
    return f"/branding/asset/{filename}"


def _build_view(raw: dict[str, Any]) -> dict[str, Any]:
    """Merge raw config with defaults and add derived/template-ready fields."""
    app_name = (raw.get("app_name") or "").strip() or DEFAULTS["app_name"]
    tagline = raw.get("tagline")
    if tagline is None:
        tagline = DEFAULTS["tagline"]
    primary_light = _normalize_hex(raw.get("primary_light"), DEFAULTS["primary_light"])
    primary_dark = _normalize_hex(raw.get("primary_dark"), DEFAULTS["primary_dark"])
    logo_light = raw.get("logo_light") or None
    logo_dark = raw.get("logo_dark") or None
    favicon = raw.get("favicon") or None

    view: dict[str, Any] = {
        "app_name": app_name,
        "tagline": tagline,
        # Raw stored filenames (used by the admin form / removal).
        "logo_light_file": logo_light,
        "logo_dark_file": logo_dark,
        "favicon_file": favicon,
        # Public URLs (None when unset).
        "logo_light_url": _asset_url(logo_light),
        "logo_dark_url": _asset_url(logo_dark),
        "favicon_url": _asset_url(favicon),
        "has_custom_logo": bool(logo_light or logo_dark),
        # True when anything differs from stock defaults.
        "is_customized": bool(
            (raw.get("app_name") and raw.get("app_name") != DEFAULTS["app_name"])
            or raw.get("tagline") not in (None, DEFAULTS["tagline"])
            or primary_light != DEFAULTS["primary_light"]
            or primary_dark != DEFAULTS["primary_dark"]
            or logo_light or logo_dark or favicon
        ),
    }
    view.update(_derive_colors(primary_light, primary_dark))
    return view


async def load_branding(db) -> dict[str, Any]:
    """Read branding config from the DB and return a template-ready view dict."""
    raw = {
        "app_name": await crud.get_config_json(db, KEY_APP_NAME, None),
        "tagline": await crud.get_config_json(db, KEY_TAGLINE, None),
        "primary_light": await crud.get_config_json(db, KEY_PRIMARY_LIGHT, None),
        "primary_dark": await crud.get_config_json(db, KEY_PRIMARY_DARK, None),
        "logo_light": await crud.get_config_json(db, KEY_LOGO_LIGHT, None),
        "logo_dark": await crud.get_config_json(db, KEY_LOGO_DARK, None),
        "favicon": await crud.get_config_json(db, KEY_FAVICON, None),
    }
    return _build_view(raw)


async def refresh_branding_cache(db=None) -> dict[str, Any]:
    """Reload branding into the module cache. Never raises to callers."""
    global _CACHE
    try:
        if db is not None:
            view = await load_branding(db)
        else:
            async with get_async_db_context() as own_db:
                view = await load_branding(own_db)
        _CACHE = view
        return view
    except Exception:  # pragma: no cover - defensive; keep serving last-known
        logger.warning("branding_cache_refresh_failed", exc_info=True)
        if not _CACHE:
            _CACHE = _build_view({})
        return _CACHE


def get_branding() -> dict[str, Any]:
    """Synchronous accessor for Jinja templates. Always returns a valid dict."""
    if not _CACHE:
        return _build_view({})
    return _CACHE


async def branding_refresh_loop() -> None:
    """Background task: periodically refresh the cache so multi-worker installs
    converge after an admin save."""
    while True:
        try:
            await asyncio.sleep(_REFRESH_INTERVAL)
            await refresh_branding_cache()
        except asyncio.CancelledError:  # pragma: no cover
            break
        except Exception:  # pragma: no cover
            logger.warning("branding_refresh_loop_error", exc_info=True)


# --- Asset storage ---------------------------------------------------------
def storage_dir() -> str:
    return get_settings().branding_storage_path


def _ensure_storage_dir() -> str:
    d = storage_dir()
    os.makedirs(d, exist_ok=True)
    return d


def asset_path(filename: str) -> Optional[str]:
    """Resolve a stored asset filename to an absolute path, guarding traversal.

    Returns None if the name is unsafe or the file does not exist.
    """
    if not filename or "/" in filename or "\\" in filename or filename.startswith("."):
        return None
    d = storage_dir()
    full = os.path.abspath(os.path.join(d, filename))
    if os.path.dirname(full) != os.path.abspath(d):
        return None
    if not os.path.isfile(full):
        return None
    return full


def content_type_for(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    return ALLOWED_IMAGE_EXT.get(ext) or ALLOWED_FAVICON_EXT.get(ext) or "application/octet-stream"


def save_asset(slot: str, original_filename: str, data: bytes) -> str:
    """Write an uploaded asset to disk and return its stored (unique) filename.

    ``slot`` is one of ``logo_light``/``logo_dark``/``favicon`` and becomes part
    of the filename; a random token cache-busts the public URL on replacement.
    """
    ext = os.path.splitext(original_filename or "")[1].lower()
    allowed = ALLOWED_FAVICON_EXT if slot == "favicon" else ALLOWED_IMAGE_EXT
    if ext not in allowed:
        raise ValueError(
            f"Unsupported file type '{ext or '(none)'}'. Allowed: {', '.join(sorted(allowed))}"
        )
    d = _ensure_storage_dir()
    token = secrets.token_hex(6)
    stored = f"{slot}-{token}{ext}"
    full = os.path.join(d, stored)
    with open(full, "wb") as f:
        f.write(data)
    return stored


def delete_asset(filename: Optional[str]) -> None:
    """Best-effort removal of a stored asset file."""
    if not filename:
        return
    full = asset_path(filename)
    if full:
        try:
            os.remove(full)
        except OSError:  # pragma: no cover
            logger.warning("branding_asset_delete_failed", filename=filename)
