############################################################
#
# mindrouter mesh-worker - async image-to-3D mesh service
#
# inputs.py: Request validation for POST /v1/meshes. Everything here raises
#     RequestError (-> HTTP 400) on bad input, so no malformed request ever
#     reaches the GPU queue.
#
# Luke Sheneman — University of Idaho RCDS — sheneman@uidaho.edu
#
############################################################

"""Validation of mesh requests and their input image."""

import base64
import binascii
import io
import secrets
import warnings
from dataclasses import dataclass
from typing import Any, Dict

from PIL import Image

from config import MAX_SEED, WorkerConfig

ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP"}

# The TRELLIS family crops to pixels with alpha > 0.8*255 and crashes on an
# empty set; refuse such inputs up front instead of failing on the GPU.
FOREGROUND_ALPHA = int(0.8 * 255)


class RequestError(ValueError):
    """A client error: the message is safe to return in a 400 body."""


@dataclass
class MeshSpec:
    model: str
    image_bytes: bytes
    image_format: str
    width: int
    height: int
    has_alpha: bool
    seed: int
    texture: bool
    texture_size: int
    target_faces: int


def _int_field(body: Dict[str, Any], name: str, default: int) -> int:
    value = body.get(name)
    if value is None:
        return default
    # bool is an int subclass in Python; true/false is never a valid count.
    if isinstance(value, bool) or not isinstance(value, int):
        if isinstance(value, float) and value.is_integer():
            return int(value)
        raise RequestError(f"'{name}' must be an integer")
    return value


def decode_image(b64: Any, config: WorkerConfig) -> tuple:
    """Decode + verify the base64 image. Returns (bytes, format, w, h, has_alpha)."""
    if not isinstance(b64, str) or not b64.strip():
        raise RequestError("'image' is required (base64 PNG, JPEG or WEBP)")
    text = b64.strip()
    # The contract says no data: prefix, but tolerate one rather than 400.
    if text.startswith("data:"):
        comma = text.find(",")
        if comma < 0:
            raise RequestError("'image' is not valid base64")
        text = text[comma + 1:]
    if len(text) > config.max_image_bytes * 4 // 3 + 8:
        raise RequestError(f"'image' exceeds {config.max_image_bytes} bytes")
    try:
        raw = base64.b64decode("".join(text.split()), validate=True)
    except (binascii.Error, ValueError):
        raise RequestError("'image' is not valid base64")
    if not raw:
        raise RequestError("'image' is empty")
    if len(raw) > config.max_image_bytes:
        raise RequestError(f"'image' exceeds {config.max_image_bytes} bytes")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as img:
                fmt = (img.format or "").upper()
                if fmt not in ALLOWED_FORMATS:
                    raise RequestError(f"image format {fmt or 'unknown'} not supported (PNG, JPEG or WEBP)")
                width, height = img.size
                if width * height > config.max_image_pixels:
                    raise RequestError(f"image is {width}x{height}, over the {config.max_image_pixels}-pixel limit")
                if min(width, height) < config.min_image_side:
                    raise RequestError(f"image is {width}x{height}; each side must be >= {config.min_image_side}")
                img.load()  # full decode: catches truncated/corrupt files
                has_alpha = img.mode in ("RGBA", "LA", "PA") or (
                    img.mode == "P" and "transparency" in img.info
                )
                if has_alpha:
                    alpha = img.convert("RGBA").getchannel("A")
                    lo, hi = alpha.getextrema()
                    if lo == 255:
                        has_alpha = False  # fully opaque: treat like RGB
                    elif hi <= FOREGROUND_ALPHA:
                        raise RequestError("image alpha channel has no opaque foreground")
    except RequestError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise RequestError("image dimensions too large")
    except Exception:
        raise RequestError("'image' could not be decoded as PNG, JPEG or WEBP")
    return raw, fmt, width, height, has_alpha


def validate_request(body: Any, config: WorkerConfig) -> MeshSpec:
    if not isinstance(body, dict):
        raise RequestError("request body must be a JSON object")

    model = body.get("model")
    if model is not None and not isinstance(model, str):
        raise RequestError("'model' must be a string")

    seed = body.get("seed")
    if seed is None:
        seed = secrets.randbelow(MAX_SEED + 1)
    else:
        seed = _int_field(body, "seed", 0)
        if not (0 <= seed <= MAX_SEED):
            raise RequestError(f"'seed' must be in [0, {MAX_SEED}]")

    texture = body.get("texture", True)
    if not isinstance(texture, bool):
        raise RequestError("'texture' must be a boolean")

    texture_size = _int_field(body, "texture_size", config.default_texture_size)
    if texture_size not in config.texture_sizes:
        raise RequestError(f"'texture_size' must be one of {config.texture_sizes}")

    target_faces = _int_field(body, "target_faces", config.default_faces)
    if not (config.min_faces <= target_faces <= config.max_faces):
        raise RequestError(f"'target_faces' must be in [{config.min_faces}, {config.max_faces}]")

    raw, fmt, width, height, has_alpha = decode_image(body.get("image"), config)
    return MeshSpec(
        model=model or config.model_id,
        image_bytes=raw,
        image_format=fmt,
        width=width,
        height=height,
        has_alpha=has_alpha,
        seed=seed,
        texture=texture,
        texture_size=texture_size,
        target_faces=target_faces,
    )


def open_image(spec: MeshSpec) -> Image.Image:
    """Decode the validated bytes for the engine: RGBA when the input carries a
    real alpha mask (the gateway's matting output), RGB otherwise."""
    with Image.open(io.BytesIO(spec.image_bytes)) as img:
        img.load()
        return img.convert("RGBA" if spec.has_alpha else "RGB")
