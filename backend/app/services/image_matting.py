############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# image_matting.py: Transparent backgrounds for generated
#     images (`background: "transparent"` on the images API).
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Transparent backgrounds for generated images.

The diffusion backends return opaque RGB (FLUX.2 Klein's decoder has no alpha
channel), and asking for transparency in the prompt only changes what is
drawn: Klein paints a grey-and-white checkerboard. So a transparent picture is
made in a second step. A matting server (``matting_service/``, a
background-removal model on a GPU node) returns an alpha matte for the
finished picture, and this module attaches it.

Order matters: the picture is watermarked FIRST (image_watermark.py works on
RGB and would drop an alpha channel), then cut out here. The colour of every
fully opaque and every fully transparent pixel is left exactly as the
watermark step wrote it, so the delivered file still carries the whole
watermark for anyone who reads it ignoring alpha. Only the thin band of
partly transparent edge pixels is recoloured, to take the old background's
colour out of them (otherwise a subject drawn on white keeps a white fringe
on a dark slide).

Everything here FAILS OPEN, like the watermark: the picture has already cost
its GPU seconds, so when the matting server is off, unreachable, slow or
wrong, the caller gets the opaque picture and ``has_alpha: false`` rather
than an error.

Config (app_config, editable on /admin/images-config, read per request):
  img.transparent_enabled   bool, default False
  img.matting_url           base URL of the matting server
  img.matting_api_key       its bearer key
  img.matting_timeout       seconds per picture, default 30
"""

import asyncio
import base64
import io
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import httpx
from prometheus_client import Counter

from backend.app.logging_config import get_logger

logger = get_logger(__name__)

# Failing open hides failures from the caller's status code, so count them.
IMAGE_BACKGROUNDS = Counter(
    "mindrouter_image_transparent_background_total",
    "Images returned for requests that asked for a transparent background, by what happened",
    ["outcome"],
)

# The values OpenAI's images API takes for ``background``. ``auto`` lets the
# server choose; here that is opaque, which is what the model draws.
BACKGROUND_TRANSPARENT = "transparent"
BACKGROUND_OPAQUE = "opaque"
BACKGROUNDS = (BACKGROUND_TRANSPARENT, BACKGROUND_OPAQUE, "auto")

DEFAULT_TIMEOUT = 30.0
MAX_TIMEOUT = 300.0

# A matte is never exactly 0 or 255 over a flat area. Values this close to
# either end are snapped to it, so the background is really gone and the
# subject is really solid.
ALPHA_FLOOR = 4
ALPHA_CEILING = 251
# A cut-out needs a subject and a background. Below this share of solid
# (fully opaque) pixels the model found no subject: a faint haze over the
# whole picture is not one. Below this share of removed (fully transparent)
# pixels nothing worth calling a background was taken away.
MIN_SOLID_FRACTION = 0.002
MIN_REMOVED_FRACTION = 0.002
# The largest reply read from the matting server. A greyscale PNG matte of a
# 2048 x 2048 picture is a few megabytes at most.
MAX_MATTE_BYTES = 16 * 1024 * 1024
# Pixels of context kept round the edge band: the larger blur window.
_EDGE_MARGIN = 96

# What happened to one picture; also the label of the outcome metric.
OUTCOME_TRANSPARENT = "transparent"
OUTCOME_DISABLED = "disabled"              # the feature is off or has no server
OUTCOME_UNAVAILABLE = "unavailable"        # the server is registered and known to be down
OUTCOME_FAILED = "failed"                  # the server is sick: unreachable, 5xx, or a wrong answer
OUTCOME_BUSY = "busy"                      # the server is loaded, not sick: queue full (503) or too slow
OUTCOME_REJECTED = "rejected"              # the server refused THIS request (4xx): wrong key, picture too large
OUTCOME_ERROR = "error"                    # something went wrong on the gateway's side
OUTCOME_NO_SUBJECT = "no_subject"          # the matte kept (almost) nothing
OUTCOME_NOTHING_REMOVED = "nothing_removed"  # the matte kept everything


class MattingError(Exception):
    """The matting server gave no usable matte. ``outcome`` says whose fault
    that was (OUTCOME_FAILED, OUTCOME_BUSY or OUTCOME_REJECTED); only
    OUTCOME_FAILED counts against the server's circuit breaker."""

    def __init__(self, message: str, outcome: str = OUTCOME_FAILED):
        super().__init__(message)
        self.outcome = outcome


def parse_background(value: Any) -> Optional[str]:
    """The caller's ``background`` value, lower-cased, or None when absent.

    Raises ValueError for anything that is not one of BACKGROUNDS, so an
    unknown value is refused instead of silently ignored.
    """
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if not cleaned:
            return None
        if cleaned in BACKGROUNDS:
            return cleaned
    raise ValueError("'background' must be one of: " + ", ".join(BACKGROUNDS))


@dataclass(frozen=True)
class MattingConfig:
    enabled: bool = False
    url: str = ""
    # Kept out of repr(), so the key cannot reach a log or a traceback that
    # prints local variables.
    api_key: Optional[str] = field(default=None, repr=False)
    timeout: float = DEFAULT_TIMEOUT       # seconds allowed per picture, start to finish

    @property
    def usable(self) -> bool:
        return bool(self.enabled and self.url)


def validate_server_url(url: str) -> Optional[str]:
    """Return an error message for an unusable matting server URL, or None.

    The URL is the server's base address: scheme, host and optional port,
    nothing else. Refusing a path catches the commonest mistake, pasting the
    endpoint (``.../v1/matte``) instead of the server.
    """
    from urllib.parse import urlsplit

    if not isinstance(url, str) or any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in url):
        return "Matting server URL must not contain spaces or control characters."
    try:
        parts = urlsplit(url)
        host, _ = parts.hostname, parts.port       # .port raises for a port that is not a number in range
    except ValueError:
        return "Matting server URL is not a valid URL (check the port)."
    if parts.scheme not in ("http", "https") or not host:
        return "Matting server URL must start with http:// or https:// and name a host."
    if parts.username or parts.password or parts.query or parts.fragment or parts.path not in ("", "/"):
        return ("Matting server URL must be the server's base address only, like https://host:port "
                "(no path, credentials, query or fragment).")
    return None


def clean_timeout(value: Any) -> float:
    """A timeout in seconds within (0, MAX_TIMEOUT], or the default."""
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT
    if isinstance(value, bool) or not 0 < seconds <= MAX_TIMEOUT:
        return DEFAULT_TIMEOUT
    return seconds


async def load_config(db: Any) -> MattingConfig:
    """The matting settings from app_config."""
    from backend.app.db import crud

    enabled = bool(await crud.get_config_json(db, "img.transparent_enabled", False))
    url = await crud.get_config_json(db, "img.matting_url", "")
    key = await crud.get_config_json(db, "img.matting_api_key", "")
    timeout = await crud.get_config_json(db, "img.matting_timeout", DEFAULT_TIMEOUT)
    return MattingConfig(
        enabled=enabled,
        url=url.strip().rstrip("/") if isinstance(url, str) else "",
        api_key=(key.strip() or None) if isinstance(key, str) else None,
        timeout=clean_timeout(timeout),
    )


# ---------------------------------------------------------------------------
# The matting server
# ---------------------------------------------------------------------------

def new_client(config: MattingConfig) -> httpx.AsyncClient:
    """The HTTP client for the matting server: this request's timeouts and the
    cluster's internal TLS setting (the same one the health check uses)."""
    from backend.app.settings import get_settings

    verify = bool(getattr(get_settings(), "internal_tls_verify", True))
    timeout = httpx.Timeout(connect=10.0, read=config.timeout, write=config.timeout, pool=10.0)
    return httpx.AsyncClient(timeout=timeout, verify=verify)


async def _request_matte(client: httpx.AsyncClient, image_bytes: bytes, config: MattingConfig) -> bytes:
    headers = {"Content-Type": "image/png"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    try:
        async with client.stream("POST", f"{config.url}/v1/matte", content=image_bytes, headers=headers) as response:
            status = response.status_code
            if status != 200:
                # 503 is "queue full"; other 4xx are about this request (wrong
                # key, picture too large for the server); the rest is sickness.
                outcome = OUTCOME_BUSY if status == 503 else OUTCOME_REJECTED if 400 <= status < 500 else OUTCOME_FAILED
                raise MattingError(f"matting server returned HTTP {status}", outcome)
            declared = response.headers.get("content-length", "")
            if declared.isdigit() and int(declared) > MAX_MATTE_BYTES:
                raise MattingError("matting server reply is too large")
            chunks, size = [], 0
            async for chunk in response.aiter_bytes():
                size += len(chunk)
                if size > MAX_MATTE_BYTES:
                    raise MattingError("matting server reply is too large")
                chunks.append(chunk)
    except httpx.ConnectTimeout as error:
        raise MattingError(f"matting server unreachable: {type(error).__name__}") from error
    except httpx.TimeoutException as error:
        # Connected, then too slow: the picture outran the wait (a queue
        # behind other pictures, usually). That is load, not sickness.
        raise MattingError(f"matting server too slow: {type(error).__name__}", OUTCOME_BUSY) from error
    except httpx.HTTPError as error:
        raise MattingError(f"matting server unreachable: {type(error).__name__}") from error
    body = b"".join(chunks)
    if not body.startswith(b"\x89PNG\r\n\x1a\n"):
        raise MattingError("matting server did not return a PNG")
    return body


async def fetch_matte(
    image_bytes: bytes,
    config: MattingConfig,
    client: Optional[httpx.AsyncClient] = None,
) -> bytes:
    """Ask the matting server for the matte of one picture (PNG bytes back).

    The whole exchange, waiting in the server's queue included, gets
    ``config.timeout`` seconds; httpx's own read timeout only bounds the gap
    between two bytes, which a slow drip would never trip. Raises
    MattingError for any failure; its text never includes the reply body.
    """
    own = client is None
    if own:
        client = new_client(config)
    try:
        return await asyncio.wait_for(_request_matte(client, image_bytes, config), timeout=config.timeout)
    except asyncio.TimeoutError:
        raise MattingError(f"no matte within {config.timeout:g} s", OUTCOME_BUSY) from None
    finally:
        if own:
            await client.aclose()


# ---------------------------------------------------------------------------
# Attaching the matte
# ---------------------------------------------------------------------------

def _box_blur(values: Any, size: int) -> Any:
    """Mean of each ``size`` x ``size`` neighbourhood (edges repeated).

    Running sums in float32: over a picture's width the rounding error is
    about 5e-5, far below one 8-bit step, and it is twice as fast as float64.
    """
    import numpy as np

    before, after = size // 2, size - size // 2 - 1
    for axis in (0, 1):
        pad = [(0, 0)] * values.ndim
        pad[axis] = (before + 1, after)
        total = np.cumsum(np.pad(values, pad, mode="edge"), axis=axis, dtype=np.float32)
        upper = [slice(None)] * values.ndim
        lower = [slice(None)] * values.ndim
        upper[axis], lower[axis] = slice(size, None), slice(0, -size)
        values = (total[tuple(upper)] - total[tuple(lower)]) / np.float32(size)
    return values


def _foreground_step(image: Any, foreground: Any, background: Any, alpha: Any, size: int) -> Tuple[Any, Any]:
    import numpy as np

    blurred_alpha = _box_blur(alpha, size)
    blurred_foreground = _box_blur(foreground * alpha, size) / (blurred_alpha + 1e-5)
    blurred_background = _box_blur(background * (1.0 - alpha), size) / ((1.0 - blurred_alpha) + 1e-5)
    estimate = blurred_foreground + alpha * (
        image - alpha * blurred_foreground - (1.0 - alpha) * blurred_background
    )
    return np.clip(estimate, 0.0, 1.0), blurred_background


def estimate_foreground(image: Any, alpha: Any) -> Any:
    """The subject's own colour at every pixel, with the old background's
    share taken out of the partly transparent ones.

    ``image`` is H x W x 3 and ``alpha`` H x W x 1, both float32 in 0..1.
    This is the two-pass blur-fusion estimate (Forte, "Approximate Fast
    Foreground Colour Estimation", ICIP 2021), the method BiRefNet's own
    reference code uses, with the same two window sizes.
    """
    foreground, background = _foreground_step(image, image, image, alpha, 90)
    return _foreground_step(image, foreground, background, alpha, 6)[0]


def attach_matte(image_bytes: bytes, matte_bytes: bytes) -> Tuple[Optional[bytes], str]:
    """Cut a picture out with its matte.

    Returns ``(RGBA PNG bytes, OUTCOME_TRANSPARENT)``, or ``(None, outcome)``
    when the matte is not a cut-out (no solid subject, or no background
    removed). Raises MattingError when the matte does not belong to the
    picture. Runs in a worker thread.
    """
    import numpy as np
    from PIL import Image

    try:
        with Image.open(io.BytesIO(image_bytes)) as opened:
            picture = opened.convert("RGB")
        with Image.open(io.BytesIO(matte_bytes), formats=["PNG"]) as opened:
            # From the header, BEFORE any pixel is decoded: a small file can
            # declare an enormous picture, and the server's reply is not trusted.
            if opened.size != picture.size:
                raise MattingError("matte and picture are not the same size")
            matte = opened.convert("L")
    except MattingError:
        raise
    except Exception as error:
        raise MattingError(f"could not decode the picture or its matte: {type(error).__name__}") from None

    alpha = np.asarray(matte, dtype=np.uint8).copy()
    alpha[alpha <= ALPHA_FLOOR] = 0
    alpha[alpha >= ALPHA_CEILING] = 255
    if np.count_nonzero(alpha == 255) < MIN_SOLID_FRACTION * alpha.size:
        return None, OUTCOME_NO_SUBJECT
    if np.count_nonzero(alpha == 0) < MIN_REMOVED_FRACTION * alpha.size:
        return None, OUTCOME_NOTHING_REMOVED

    pixels = np.asarray(picture, dtype=np.uint8).copy()
    edge = (alpha > 0) & (alpha < 255)
    if edge.any():
        # Work on the part of the picture that holds the edge band, with
        # enough margin for the larger blur window.
        rows, cols = np.flatnonzero(edge.any(axis=1)), np.flatnonzero(edge.any(axis=0))
        top, bottom = max(0, rows[0] - _EDGE_MARGIN), min(alpha.shape[0], rows[-1] + 1 + _EDGE_MARGIN)
        left, right = max(0, cols[0] - _EDGE_MARGIN), min(alpha.shape[1], cols[-1] + 1 + _EDGE_MARGIN)
        box = (slice(top, bottom), slice(left, right))
        estimate = estimate_foreground(
            pixels[box].astype(np.float32) / 255.0,
            (alpha[box].astype(np.float32) / 255.0)[:, :, None],
        )
        # Only the edge band: every other pixel keeps the colour (and the
        # watermark) it arrived with.
        band = edge[box]
        pixels[box][band] = np.rint(estimate[band] * 255.0).astype(np.uint8)

    out = io.BytesIO()
    Image.fromarray(np.dstack([pixels, alpha]), "RGBA").save(out, format="PNG")
    return out.getvalue(), OUTCOME_TRANSPARENT


async def make_transparent(
    b64_image: str,
    config: MattingConfig,
    client: Optional[httpx.AsyncClient] = None,
) -> Tuple[str, str]:
    """Cut out one base64 picture: ``(base64 PNG, outcome)``.

    Fails OPEN: for any outcome other than OUTCOME_TRANSPARENT the ORIGINAL
    picture is returned unchanged. Never raises (cancellation passes through).
    """
    try:
        raw = base64.b64decode(b64_image)
    except Exception:
        logger.warning("image_matting_picture_not_base64")
        return b64_image, OUTCOME_ERROR
    try:
        matte = await fetch_matte(raw, config, client=client)
        cut, outcome = await asyncio.to_thread(attach_matte, raw, matte)
        if cut is None:
            logger.info("image_matting_not_applied", outcome=outcome)
            return b64_image, outcome
        return base64.b64encode(cut).decode("ascii"), outcome
    except MattingError as error:
        logger.warning("image_matting_failed_returning_opaque_image", reason=str(error), outcome=error.outcome)
        return b64_image, error.outcome
    except Exception as error:
        # Type only: a traceback could print local variables.
        logger.error("image_matting_failed_returning_opaque_image", error_type=type(error).__name__)
    return b64_image, OUTCOME_ERROR


# ---------------------------------------------------------------------------
# Answering `background` on a finished response
# ---------------------------------------------------------------------------

async def _report(report: Any, backend_id: int) -> None:
    """Tell the registry how a request to a monitored matting server went.
    Never lets bookkeeping fail the image."""
    try:
        await report(backend_id)
    except Exception:
        logger.warning("image_matting_circuit_report_failed", backend_id=backend_id)


# After one of these the server is not asked again within the same response.
_STOP_DIALING = (OUTCOME_FAILED, OUTCOME_BUSY, OUTCOME_REJECTED)
# The server returned a matte that fitted the picture: it is working.
_SERVER_ANSWERED = (OUTCOME_TRANSPARENT, OUTCOME_NO_SUBJECT, OUTCOME_NOTHING_REMOVED)


async def _cut_out_all(
    pictures: List[Any], config: MattingConfig, registry: Any
) -> List[Tuple[str, Optional[str]]]:
    """Cut out every image of one response. One ``(outcome, cut-out)`` per
    image; the cut-out is base64 PNG for OUTCOME_TRANSPARENT and None otherwise."""
    if not config.usable:
        return [(OUTCOME_DISABLED, None)] * len(pictures)

    # A matting server registered as a backend (engine "matting") is
    # health-polled; when it is known to be down, do not dial it.
    monitor_id: Optional[int] = None
    if registry is not None:
        try:
            monitor_id, problem = await registry.matting_server_state(config.url)
        except Exception as error:
            # Monitoring is an optimisation. If the lookup itself fails, dial
            # the server as if it were not registered.
            logger.warning("image_matting_server_lookup_failed", error_type=type(error).__name__)
            monitor_id, problem = None, None
        if problem:
            logger.info("image_matting_server_unavailable", backend_id=monitor_id, reason=problem)
            return [(OUTCOME_UNAVAILABLE, None)] * len(pictures)

    results: List[Tuple[str, Optional[str]]] = []
    dialed: List[str] = []          # outcomes of the pictures actually sent to the server
    async with new_client(config) as client:
        for picture in pictures:
            if dialed and dialed[-1] in _STOP_DIALING:
                # The server just failed, said it is full, or refused the
                # request: do not make the caller wait out the same answer
                # for every remaining image.
                results.append((dialed[-1], None))
                continue
            # An entry with no bytes (url only) comes back as OUTCOME_ERROR
            # without the server being asked.
            cut, outcome = await make_transparent(picture, config, client=client)
            dialed.append(outcome)
            results.append((outcome, cut if outcome == OUTCOME_TRANSPARENT else None))

    # Only sickness counts against the server's circuit breaker. Busy, slow,
    # and a refusal of one request (wrong key, picture too large) do not:
    # three of those must not switch transparency off for everybody.
    if monitor_id is not None and registry is not None:
        if OUTCOME_FAILED in dialed:
            await _report(registry.report_live_failure, monitor_id)
        elif any(outcome in _SERVER_ANSWERED for outcome in dialed):
            await _report(registry.report_live_success, monitor_id)
    return results


async def apply_background(
    background: Optional[str],
    response: Dict[str, Any],
    config: Optional[MattingConfig],
    registry: Any = None,
) -> None:
    """Answer the caller's ``background`` on a finished image response, in place.

    Does nothing when the caller did not send the field. Otherwise every
    image gets ``has_alpha`` and the response gets ``background`` (what was
    produced: ``"transparent"`` only when every image is). Never raises.
    """
    if background is None:
        return
    items = [item for item in (response.get("data") or []) if isinstance(item, dict)]
    results: List[Tuple[str, Optional[str]]] = []
    if background == BACKGROUND_TRANSPARENT:
        try:
            results = await _cut_out_all(
                [item.get("b64_json") for item in items], config or MattingConfig(), registry)
        except Exception as error:
            logger.error("image_background_failed_returning_opaque_images", error_type=type(error).__name__)
            results = [(OUTCOME_ERROR, None)] * len(items)
        for outcome, _ in results:
            IMAGE_BACKGROUNDS.labels(outcome=outcome).inc()
    # Pictures are replaced only here, after everything that can fail, so
    # `has_alpha` always describes the bytes the caller receives.
    for index, item in enumerate(items):
        cut = results[index][1] if index < len(results) else None
        if cut:
            item["b64_json"] = cut
        item["has_alpha"] = bool(cut)
    transparent = bool(items) and all(item["has_alpha"] for item in items)
    response["background"] = BACKGROUND_TRANSPARENT if transparent else BACKGROUND_OPAQUE
