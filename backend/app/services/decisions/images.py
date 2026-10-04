############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# services/decisions/images.py: images on a System One request
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""The ``images`` extension to the System One request.

TypeSafe's contract is text only. Cloudflare's Clef adds ``images``, and this
module follows Cloudflare's published schema for it
(developers.cloudflare.com/workers-ai/models/clef/schema-input.json) so a
request written for Workers AI is accepted unchanged:

* ``images`` is an array of at most 4 items, placed before the state;
* each item is a base64 data URL (``data:image/png;base64,...``) or an object
  ``{"content_type": "image/png", "base64": "..."}``;
* PNG, JPEG or WebP only; 4 MiB and 16 megapixels each; 8 MiB in total;
* remote URLs are not accepted (the gateway does not fetch on a caller's behalf).

Every image is checked here, before any model sees it: the bytes must decode,
must really be the type they claim, and must be within the size limits. Only
the header is read; nothing is decoded to pixels in the gateway.
"""
from __future__ import annotations

import base64
import binascii
import io
import re
from typing import Any

MAX_IMAGES = 4
MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_TOTAL_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000

# content type <-> Pillow format name
_FORMATS = {"image/png": "PNG", "image/jpeg": "JPEG", "image/webp": "WEBP"}
_TYPE_OF = {v: k for k, v in _FORMATS.items()}
_DATA_URL = re.compile(r"^data:(?P<type>[\w.+-]+/[\w.+-]+);base64,(?P<data>.*)$", re.IGNORECASE | re.DOTALL)


class ImageError(ValueError):
    """An image the request cannot carry. ``index`` is its position in ``images``
    (None when the problem is with the list itself)."""

    def __init__(self, message: str, index: int | None = None):
        super().__init__(message)
        self.index = index


def normalize_images(raw: Any) -> list[str]:
    """Validate ``images`` and return each as a canonical base64 data URL.
    Raises ImageError. An absent or empty list is no images."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ImageError("images must be an array")
    if len(raw) > MAX_IMAGES:
        raise ImageError(f"at most {MAX_IMAGES} images per request")
    out: list[str] = []
    total = 0
    for index, item in enumerate(raw):
        declared, encoded = _split(item, index)
        try:
            data = base64.b64decode("".join(encoded.split()), validate=True)
        except (binascii.Error, ValueError):
            raise ImageError("image is not valid base64", index) from None
        if not data:
            raise ImageError("image is empty", index)
        if len(data) > MAX_IMAGE_BYTES:
            raise ImageError(f"image exceeds {MAX_IMAGE_BYTES // (1024 * 1024)} MiB", index)
        total += len(data)
        if total > MAX_TOTAL_IMAGE_BYTES:
            raise ImageError(f"images exceed {MAX_TOTAL_IMAGE_BYTES // (1024 * 1024)} MiB in total")
        actual = _inspect(data, index)
        if declared != actual:
            raise ImageError(f"image is {actual}, not the declared {declared}", index)
        out.append(f"data:{actual};base64,{base64.b64encode(data).decode('ascii')}")
    return out


def _split(item: Any, index: int) -> tuple[str, str]:
    """(declared content type, base64 text) of one ``images`` item."""
    if isinstance(item, str):
        match = _DATA_URL.match(item.strip())
        if not match:
            raise ImageError("image must be a base64 data URL (data:image/png;base64,...); "
                             "remote URLs are not accepted", index)
        declared, encoded = match.group("type").lower(), match.group("data")
    elif isinstance(item, dict):
        declared, encoded = item.get("content_type"), item.get("base64")
        if not isinstance(declared, str) or not isinstance(encoded, str):
            raise ImageError("image object needs string fields content_type and base64", index)
        declared = declared.strip().lower()
    else:
        raise ImageError("image must be a data URL string or an object with content_type and base64", index)
    if declared == "image/jpg":
        declared = "image/jpeg"
    if declared not in _FORMATS:
        raise ImageError("image type must be image/png, image/jpeg or image/webp", index)
    return declared, encoded


def _inspect(data: bytes, index: int) -> str:
    """The image's real content type, from its header. Rejects anything that
    is not a PNG, JPEG or WebP within the pixel limit."""
    from PIL import Image

    try:
        with Image.open(io.BytesIO(data)) as image:
            fmt, (width, height) = image.format, image.size
    except Exception:
        # Not an image, a truncated one, or one Pillow refuses as a decompression bomb.
        raise ImageError("image could not be read as PNG, JPEG or WebP", index) from None
    if fmt not in _TYPE_OF:
        raise ImageError("image type must be image/png, image/jpeg or image/webp", index)
    if width * height > MAX_IMAGE_PIXELS:
        raise ImageError(f"image exceeds {MAX_IMAGE_PIXELS // 1_000_000} megapixels", index)
    return _TYPE_OF[fmt]
