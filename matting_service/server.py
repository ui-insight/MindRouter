############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# matting_service/server.py: HTTP server for a background-
# removal (matting) model.
#
# The diffusion backends return opaque RGB: FLUX.2 Klein's
# decoder has three colour channels and no alpha. A picture
# with a transparent background is therefore made in a second
# step, by a model that marks which pixels are the subject.
# This serves that model so MindRouter can answer
# `background: "transparent"` on /v1/images/generations and
# /v1/images/edits (services/image_matting.py).
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Matting server: a picture in, its alpha matte out.

One process holds one copy of the model on one GPU and answers one picture at
a time on a single inference thread; requests that arrive meanwhile wait in a
bounded queue. A cut-out takes a fraction of a second on a GPU, so there is
no batching.

Endpoints
---------
``POST /v1/matte``  body = the image bytes (PNG, JPEG or WebP) -> ``image/png``,
                    an 8-bit greyscale matte of the same size: 255 = subject,
                    0 = background (bearer key)
``GET  /health``    liveness for anyone; details with the bearer key. The port
                    opens only once the model is loaded, so a server that is
                    still starting refuses connections rather than answering

The reply is only the matte, never a recoloured picture: the gateway attaches
it to its own (watermarked) pixels, so this server cannot change what the
caller's image looks like, only which parts of it show.

Response headers: ``X-Matting-Model`` (the served name), ``X-Matting-Seconds``
(model time) and ``X-Matting-Coverage`` (mean matte value, 0..1).

Nothing about a picture is logged, and no picture is written to disk.
Failures log their type only.

The model is reached through ``MODEL_FACTORY`` so the unit tests can run the
whole server with no GPU and no torch.
"""

# No `from __future__ import annotations` here: the route handlers are defined
# inside create_app() with fastapi imported locally, and FastAPI must see the
# real `Request` type, not a string it cannot resolve from module globals
# (it would treat the parameter as a query field and answer 422).
import asyncio
import hmac
import io
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("matting_service")

# What a caller may send. The type is read from the file's own first bytes,
# never from the Content-Type header.
IMAGE_FORMATS = ("PNG", "JPEG", "WEBP")
DEFAULT_MAX_BODY_BYTES = 12 * 1024 * 1024
# 2048 x 2048. MindRouter's largest generated image is 1024 x 1024.
DEFAULT_MAX_PIXELS = 4_194_304


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        raise SystemExit(f"invalid {name}={raw!r}: must be an integer") from None
    if value < minimum:
        raise SystemExit(f"invalid {name}={raw!r}: must be at least {minimum}")
    return value


@dataclass
class ServiceConfig:
    # HF repo id or a local snapshot directory. BiRefNet_dynamic is the variant
    # that handled every sample in the 2026-10-06 comparison on FLUX.2 Klein
    # output (README.md); the general and the lite model each broke an emblem.
    model: str = "ZhengPeng7/BiRefNet_dynamic"
    # The model's code ships with its weights and is executed at load
    # (trust_remote_code). Pin the commit that was reviewed.
    revision: Optional[str] = None
    served_name: str = "birefnet-dynamic"  # reported in X-Matting-Model and /health
    device: str = "cuda"
    half: bool = True                    # half precision on a GPU; ignored on cpu
    side: int = 1024                     # the model's square input size
    host: str = "127.0.0.1"
    port: int = 18006
    api_key: Optional[str] = None
    allow_no_auth: bool = False
    max_queue: int = 16                  # pictures waiting; beyond this the answer is 503
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES
    max_pixels: int = DEFAULT_MAX_PIXELS

    @classmethod
    def from_env(cls) -> "ServiceConfig":
        return cls(
            model=os.environ.get("MATTING_MODEL", cls.model),
            revision=(os.environ.get("MATTING_REVISION") or "").strip() or None,
            served_name=os.environ.get("MATTING_SERVED_NAME", cls.served_name),
            device=os.environ.get("MATTING_DEVICE", cls.device),
            half=os.environ.get("MATTING_HALF", "1") != "0",
            side=_env_int("MATTING_SIDE", cls.side, 256),
            host=os.environ.get("MATTING_HOST", cls.host),
            port=_env_int("MATTING_PORT", cls.port),
            api_key=(os.environ.get("MATTING_API_KEY") or "").strip() or None,
            allow_no_auth=os.environ.get("MATTING_ALLOW_NO_AUTH", "") == "1",
            max_queue=_env_int("MATTING_MAX_QUEUE", cls.max_queue),
            max_body_bytes=_env_int("MATTING_MAX_BODY_BYTES", cls.max_body_bytes, 1024),
            max_pixels=_env_int("MATTING_MAX_PIXELS", cls.max_pixels, 4096),
        )


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

# MODEL_FACTORY(config) -> engine. An engine has:
#   matte(image: PIL.Image "RGB") -> PIL.Image "L" of the same size
#   device: str
# Unset, the real BiRefNet engine is built.
MODEL_FACTORY: Optional[Callable[[ServiceConfig], Any]] = None

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class BiRefNetEngine:
    """BiRefNet (github.com/ZhengPeng7/BiRefNet, MIT) and its variants, loaded
    the way the model card documents: transformers with the repo's own code."""

    def __init__(self, config: ServiceConfig):
        import torch
        from transformers import AutoModelForImageSegmentation

        self._torch = torch
        self._side = config.side
        kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if config.revision:
            kwargs["revision"] = config.revision
        elif not os.path.isdir(config.model):
            logger.warning("matting_model_revision_not_pinned model=%s: the repository's code runs in this "
                           "process; set MATTING_REVISION to the commit that was reviewed", config.model)
        model = AutoModelForImageSegmentation.from_pretrained(config.model, **kwargs)
        self.half = bool(config.half and config.device.startswith("cuda"))
        # The published weights are stored in half precision; say which one
        # this process runs in rather than inherit it.
        model = model.half() if self.half else model.float()
        self.model = model.to(config.device).eval()
        self.device = str(next(self.model.parameters()).device)
        dtype = torch.float16 if self.half else torch.float32
        self._mean = torch.tensor(_IMAGENET_MEAN, device=self.device, dtype=dtype).view(1, 3, 1, 1)
        self._std = torch.tensor(_IMAGENET_STD, device=self.device, dtype=dtype).view(1, 3, 1, 1)

    def matte(self, image: Any) -> Any:
        import numpy as np
        from PIL import Image

        torch = self._torch
        width, height = image.size
        small = image.resize((self._side, self._side), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(small, dtype=np.uint8).copy()).to(self.device)
        x = x.permute(2, 0, 1).unsqueeze(0).to(self._mean.dtype).div(255.0)
        x = (x - self._mean) / self._std
        with torch.inference_mode():
            prediction = self.model(x)[-1].sigmoid().float()
            prediction = torch.nn.functional.interpolate(
                prediction, size=(height, width), mode="bilinear", align_corners=False)
            matte = (prediction[0, 0].clamp(0, 1) * 255.0).round().to(torch.uint8).cpu().numpy()
        return Image.fromarray(matte, "L")


# ---------------------------------------------------------------------------
# Request validation (what the server refuses before it queues anything)
# ---------------------------------------------------------------------------

class BadRequest(ValueError):
    """A request the caller has to fix; becomes a 422."""


def _image_format(data: bytes) -> Optional[str]:
    """Pillow format name from the file's own signature, whatever the request claims."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    if data.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "WEBP"
    return None


def open_image(data: bytes, max_pixels: int) -> Any:
    """The picture as RGB, or BadRequest. The size is checked from the header,
    before any pixel is decoded."""
    from PIL import Image

    if not data:
        raise BadRequest("request body is empty; send the image bytes")
    kind = _image_format(data)
    if kind is None:
        raise BadRequest("image must be PNG, JPEG or WebP")
    try:
        with Image.open(io.BytesIO(data), formats=[kind]) as image:
            width, height = image.size
            if width < 1 or height < 1 or width * height > max_pixels:
                raise BadRequest(f"image must be at most {max_pixels} pixels (got {width} x {height})")
            return image.convert("RGB")
    except BadRequest:
        raise
    except Exception:
        raise BadRequest("image could not be decoded") from None


# ---------------------------------------------------------------------------
# One picture at a time
# ---------------------------------------------------------------------------

class Oversubscribed(Exception):
    pass


@dataclass
class Stats:
    requests: int = 0
    answered: int = 0
    rejected_busy: int = 0
    failed: int = 0
    seconds: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        avg = self.seconds / self.answered if self.answered else 0.0
        return {"requests": self.requests, "answered": self.answered, "rejected_busy": self.rejected_busy,
                "failed": self.failed, "avg_seconds": round(avg, 4)}


class Worker:
    """Runs the model on one thread and bounds how many pictures may wait."""

    def __init__(self, config: ServiceConfig):
        self.config = config
        self.engine: Any = None
        self.stats = Stats()
        self._waiting = 0
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="matting-infer")

    @property
    def ready(self) -> bool:
        return self.engine is not None

    def queue_depth(self) -> int:
        return self._waiting

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        factory = MODEL_FACTORY or BiRefNetEngine
        self.engine = await loop.run_in_executor(self._executor, factory, self.config)

    async def stop(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    async def matte(self, image: Any) -> Tuple[Any, float]:
        """(matte, model seconds). Raises Oversubscribed when the queue is full."""
        if self._waiting >= self.config.max_queue:
            self.stats.rejected_busy += 1
            raise Oversubscribed()
        self._waiting += 1
        self.stats.requests += 1
        try:
            return await asyncio.get_running_loop().run_in_executor(self._executor, self._timed, image)
        finally:
            self._waiting -= 1

    def _timed(self, image: Any) -> Tuple[Any, float]:
        started = time.monotonic()
        matte = self.engine.matte(image)
        if matte.size != image.size or matte.mode != "L":
            raise RuntimeError("engine returned a matte of the wrong shape")
        return matte, time.monotonic() - started


async def _read_body(request: Any, limit: int) -> Optional[bytes]:
    """The request body, or None once it passes ``limit`` (nothing beyond the
    limit is kept, whether or not the client declared a length)."""
    chunks: List[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


async def _unless_disconnected(request: Any, work: Any, poll_seconds: float = 0.25) -> Any:
    """Await ``work``; cancel it if the client disconnects first, so a picture
    nobody is waiting for leaves the queue instead of costing a forward pass."""
    task = asyncio.ensure_future(work)
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=poll_seconds)
            if done:
                return task.result()
            if await request.is_disconnected():
                raise ClientGone()
    finally:
        if not task.done():
            task.cancel()


class ClientGone(Exception):
    """The caller disconnected before its answer was ready."""


def _png(matte: Any) -> bytes:
    out = io.BytesIO()
    # A matte is mostly flat; level 3 is a few times faster than the default
    # for a file only a few percent larger.
    matte.save(out, format="PNG", compress_level=3)
    return out.getvalue()


def _coverage(matte: Any) -> float:
    """Mean matte value, 0 (nothing kept) to 1 (everything kept)."""
    histogram = matte.histogram()
    pixels = sum(histogram) or 1
    return sum(value * count for value, count in enumerate(histogram)) / (255.0 * pixels)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def create_app(config: ServiceConfig):
    from fastapi import FastAPI, Header, HTTPException, Request
    from fastapi.responses import Response

    if not config.api_key and not config.allow_no_auth:
        raise SystemExit("MATTING_API_KEY is not set. Set it, or MATTING_ALLOW_NO_AUTH=1 to serve without a key.")

    worker = Worker(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await worker.start()
        logger.info("matting_ready model=%s device=%s", config.served_name, getattr(worker.engine, "device", "?"))
        try:
            yield
        finally:
            await worker.stop()

    app = FastAPI(title="MindRouter matting service", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.worker = worker

    def authorized(authorization: Optional[str]) -> bool:
        if not config.api_key:
            return True
        if not authorization or not authorization.startswith("Bearer "):
            return False
        return hmac.compare_digest(authorization[len("Bearer "):].strip().encode(), config.api_key.encode())

    @app.get("/health")
    async def health(authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
        # Reachable only once the model is loaded: uvicorn finishes start-up
        # (the load) before it opens the port.
        if not authorized(authorization):
            return {"status": "ok"}        # liveness only without the key
        return {"status": "ok", "model": config.served_name, "source": config.model,
                "revision": config.revision, "device": getattr(worker.engine, "device", None),
                "half": getattr(worker.engine, "half", None), "side": config.side,
                "max_pixels": config.max_pixels, "queue_depth": worker.queue_depth(),
                "max_queue": config.max_queue, "stats": worker.stats.as_dict()}

    @app.post("/v1/matte")
    async def matte(request: Request, authorization: Optional[str] = Header(default=None)):
        if not authorized(authorization):
            raise HTTPException(status_code=401, detail="missing or invalid bearer key")
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > config.max_body_bytes:
            raise HTTPException(status_code=413, detail="image is too large")
        raw = await _read_body(request, config.max_body_bytes)
        if raw is None:
            raise HTTPException(status_code=413, detail="image is too large")
        try:
            image = open_image(raw, config.max_pixels)
        except BadRequest as error:
            raise HTTPException(status_code=422, detail=str(error)) from None

        try:
            result, seconds = await _unless_disconnected(request, worker.matte(image))
        except ClientGone:
            raise HTTPException(status_code=499, detail="client closed request") from None
        except Oversubscribed:
            raise HTTPException(status_code=503, detail="server busy, try again shortly",
                                headers={"Retry-After": "1"}) from None
        except Exception as error:
            # Never a made-up matte, and never the error's text.
            worker.stats.failed += 1
            logger.error("matting_failed error_type=%s", type(error).__name__)
            raise HTTPException(status_code=500, detail="matting failed") from None
        worker.stats.answered += 1
        worker.stats.seconds += seconds
        return Response(content=_png(result), media_type="image/png", headers={
            "X-Matting-Model": config.served_name,
            "X-Matting-Seconds": f"{seconds:.4f}",
            "X-Matting-Coverage": f"{_coverage(result):.4f}",
        })

    return app
