############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# clef_service/server.py: HTTP server for Cloudflare's Clef
# decision model, speaking TypeSafe's System One wire format.
#
# Clef (huggingface.co/Cloudflare/clef, Apache-2.0) ships
# weights and a Python function, but no server. This wraps
# that function so MindRouter can use Clef as a decisions
# upstream (decisions.upstreams), with what a shared GPU
# service needs: a bearer key, dynamic batching, a bounded
# queue, and a health probe.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""System One server for Clef.

One process holds one copy of the model on one GPU. Concurrency comes from
batching, not from extra copies: requests that arrive within a few
milliseconds of each other are padded into one tensor and answered by a single
forward pass (Clef's own ``collate_records``). Two processes on one GPU would
take turns on the same compute and hold the 55 GB of weights twice.

Endpoints
---------
``POST /v1/systemone``  TypeSafe System One request -> response (bearer key)
``GET  /health``        liveness for anyone; details with the bearer key
``GET  /v1/models``     the served name, in TypeSafe's list shape (bearer key)

Requests may carry ``images`` (Cloudflare's extension for Clef: up to four
base64 PNG/JPEG/WebP images, shown to the model before the state).

Nothing about a request is logged: not the state, not the questions, not an
exception's text (which can quote them). Failures log their type only.

The model is reached through ``MODEL_FACTORY`` so the unit tests can run the
whole server with no GPU and no torch.
"""

# No `from __future__ import annotations` here: the route handlers are defined
# inside create_app() with fastapi imported locally, and FastAPI must see the
# real `Request` type, not a string it cannot resolve from module globals
# (it would treat the parameter as a query field and answer 422).
import asyncio
import base64
import binascii
import hmac
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("clef_service")

QUESTION_TYPES = ("noul", "choice", "score")

# Guardrails on remote input. The whole request is tokenized into one sequence,
# so these bound what a single caller can make the GPU hold.
MAX_QUESTIONS = 64
MAX_CHOICE_OPTIONS = 255     # TypeSafe's own limit
MAX_SCORE_LEVELS = 10
# Images: Cloudflare's published limits for Clef's ``images`` extension
# (developers.cloudflare.com/workers-ai/models/clef/schema-input.json).
MAX_IMAGES = 4
MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_TOTAL_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
IMAGE_TYPES = ("image/png", "image/jpeg", "image/webp")
# Longest base64 text for one image (5% allowed for line breaks); checked before decoding.
MAX_IMAGE_CHARS = int((MAX_IMAGE_BYTES + 2) // 3 * 4 * 1.05) + 16
_WHITESPACE = str.maketrans("", "", " \t\r\n")
DEFAULT_MAX_BODY_BYTES = 13 * 1024 * 1024


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
    model: str = "Cloudflare/clef"       # HF repo id or a local snapshot directory
    served_name: str = "clef"            # what callers send in ``model`` and what replies report
    device: str = "cuda"
    host: str = "127.0.0.1"
    port: int = 18004
    api_key: Optional[str] = None
    allow_no_auth: bool = False
    max_length: int = 16384              # tokens per request (state is cut to fit; reported in usage)
    # Requests per forward pass. 1 = one at a time, the measured best for real
    # traffic: requests of different lengths are padded to the longest, which
    # costs as much as running them separately, and padded batches sometimes
    # stall for seconds. Raise it only for many short requests of equal length.
    max_batch: int = 1
    batch_wait_ms: int = 5               # how long the first request waits for company
    max_batch_tokens: int = 65536        # padded tokens per forward pass (batch size x longest request)
    max_queue: int = 64                  # requests waiting; beyond this the answer is 503
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES
    # PyTorch's cuDNN attention backend re-plans for every new input length
    # (measured: +1.2 s on the first request at each length, i.e. on nearly
    # every real request). Off by default.
    cudnn_attention: bool = False

    @classmethod
    def from_env(cls) -> "ServiceConfig":
        return cls(
            model=os.environ.get("CLEF_MODEL", cls.model),
            served_name=os.environ.get("CLEF_SERVED_NAME", cls.served_name),
            device=os.environ.get("CLEF_DEVICE", cls.device),
            host=os.environ.get("CLEF_HOST", cls.host),
            port=_env_int("CLEF_PORT", cls.port),
            api_key=(os.environ.get("CLEF_API_KEY") or "").strip() or None,
            allow_no_auth=os.environ.get("CLEF_ALLOW_NO_AUTH", "") == "1",
            max_length=_env_int("CLEF_MAX_LENGTH", cls.max_length, 256),
            max_batch=_env_int("CLEF_MAX_BATCH", cls.max_batch),
            batch_wait_ms=_env_int("CLEF_BATCH_WAIT_MS", cls.batch_wait_ms, 0),
            max_batch_tokens=_env_int("CLEF_MAX_BATCH_TOKENS", cls.max_batch_tokens, 256),
            max_queue=_env_int("CLEF_MAX_QUEUE", cls.max_queue),
            max_body_bytes=_env_int("CLEF_MAX_BODY_BYTES", cls.max_body_bytes, 1024),
            cudnn_attention=os.environ.get("CLEF_CUDNN_ATTENTION", "") == "1",
        )


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

@dataclass
class Encoded:
    """One request, tokenized. ``payload`` is whatever the engine needs back."""
    payload: Any
    tokens: int                 # sequence length, i.e. what this request costs in a batch
    state_tokens_dropped: int   # state tokens cut to fit max_length (0 = nothing cut)
    images: int = 0             # images the model was shown


# MODEL_FACTORY(config) -> engine. An engine has:
#   encode(request: dict) -> Encoded          raises ValueError for a request the model cannot take
#   infer(batch: list[Encoded], requests: list[dict]) -> list[dict]   one {question id: answer} per request
#   device: str
# Unset, the real Clef engine is built.
MODEL_FACTORY: Optional[Callable[[ServiceConfig], Any]] = None


def _fast_path_available() -> Optional[bool]:
    """Whether the backbone runs its fused kernels (flash-linear-attention +
    causal-conv1d) rather than the plain-torch fallback. None if unknown."""
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5
        return bool(modeling_qwen3_5.is_fast_path_available)
    except Exception:
        return None


class ClefEngine:
    """Cloudflare's release code (``joint_schema_model.py`` from the model repo)."""

    def __init__(self, config: ServiceConfig):
        import sys
        from pathlib import Path

        import torch
        from huggingface_hub import snapshot_download

        path = Path(config.model)
        if not path.is_dir():
            path = Path(snapshot_download(config.model))
        # The model's code ships with its weights; import it from the snapshot.
        sys.path.insert(0, str(path))
        import joint_schema_model as release

        self._torch, self._release = torch, release
        self._max_length = config.max_length
        if not config.cudnn_attention:
            torch.backends.cuda.enable_cudnn_sdp(False)
        self.model, self.processor = release.load_release_model(path, device=config.device)
        self.device = str(next(self.model.parameters()).device)
        self.fast_path = _fast_path_available()
        if not self.fast_path:
            logger.warning("clef_slow_path: flash-linear-attention and causal-conv1d are not both installed; "
                           "requests take about 2.5x longer (see README)")

    def encode(self, request: Dict[str, Any]) -> Encoded:
        release, tokenizer = self._release, self.processor.tokenizer
        record = {"state": request["state"], "questions": request["questions"]}
        pictures = self._open_images(request.get("_images") or [])
        if pictures:
            record["images"] = pictures
        # Encode without a limit first to learn the full length; Clef cuts the
        # state silently, and a caller should be told when that happened.
        full = release.encode_record(tokenizer, record, max_length=10**9, processor=self.processor)
        dropped = max(0, len(full.input_ids) - self._max_length)
        encoded = full if not dropped else release.encode_record(
            tokenizer, record, max_length=self._max_length, processor=self.processor)
        return Encoded(payload=encoded, tokens=len(encoded.input_ids), state_tokens_dropped=dropped,
                       images=len(pictures))

    @staticmethod
    def _open_images(blobs: List[bytes]) -> List[Any]:
        """Decode validated image bytes for the processor. A ValueError here is
        the caller's 422 (an image that is not what its header claims)."""
        if not blobs:
            return []
        import io

        from PIL import Image

        pictures = []
        for index, blob in enumerate(blobs):
            try:
                with Image.open(io.BytesIO(blob), formats=["PNG", "JPEG", "WEBP"]) as image:
                    if image.width * image.height > MAX_IMAGE_PIXELS:
                        raise ValueError(f"image {index} exceeds {MAX_IMAGE_PIXELS // 1_000_000} megapixels")
                    pictures.append(image.convert("RGB"))
            except ValueError:
                raise
            except Exception:
                raise ValueError(f"image {index} could not be decoded") from None
        return pictures

    def infer(self, batch: List[Encoded], requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        release, torch = self._release, self._torch
        records = [item.payload for item in batch]
        collated = release.collate_records(records, self.processor.tokenizer.pad_token_id,
                                           torch.device(self.device))
        with torch.inference_mode():
            logits = self.model(collated)
        out = []
        for record, record_logits, request in zip(records, logits, requests):
            questions = request["questions"]
            out.append({
                q.question_id: release.systemone_answer(
                    questions[q.question_id],
                    dict(zip(q.option_ids, q_logits.float().softmax(-1).tolist())),
                )
                for q, q_logits in zip(record.questions, record_logits)
            })
        return out


def _is_out_of_memory(error: BaseException) -> bool:
    return "OutOfMemory" in type(error).__name__ or "out of memory" in str(error).lower()


# ---------------------------------------------------------------------------
# Request validation (what the server refuses before it queues anything)
# ---------------------------------------------------------------------------

class BadRequest(ValueError):
    """A request the caller has to fix; becomes a 422."""


def validate_request(body: Any) -> Dict[str, Any]:
    if not isinstance(body, dict):
        raise BadRequest("request body must be a JSON object")
    if "state" not in body or not isinstance(body["state"], (str, dict, list)):
        raise BadRequest("state is required and must be text, an object or an array")
    if "model" in body and not isinstance(body["model"], str):
        raise BadRequest("model must be a string")
    if body.get("videos"):
        raise BadRequest("videos are not supported by this server")
    # Decoded once here; the raw base64 is dropped so a queued request does not hold both.
    body["_images"] = decode_images(body.pop("images", None))
    questions = body.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise BadRequest("questions must be a non-empty object")
    if len(questions) > MAX_QUESTIONS:
        raise BadRequest(f"at most {MAX_QUESTIONS} questions per request")
    for qid, q in questions.items():
        if not isinstance(q, dict) or q.get("type") not in QUESTION_TYPES:
            raise BadRequest(f"question {qid!r}: type must be noul, choice or score")
        criteria = q.get("criteria")
        if q["type"] == "choice":
            if not isinstance(criteria, dict) or not 1 <= len(criteria) <= MAX_CHOICE_OPTIONS:
                raise BadRequest(f"question {qid!r}: criteria must map 1 to {MAX_CHOICE_OPTIONS} options")
        elif q["type"] == "score":
            if not isinstance(criteria, list) or not 1 <= len(criteria) <= MAX_SCORE_LEVELS:
                raise BadRequest(f"question {qid!r}: criteria must list 1 to {MAX_SCORE_LEVELS} levels")
        elif criteria is not None and (not isinstance(criteria, dict) or not set(criteria) <= {"true", "false"}):
            raise BadRequest(f"question {qid!r}: noul criteria may only describe true and false")
    return body


def _image_type(data: bytes) -> Optional[str]:
    """Content type from the file's own signature, whatever the request claims."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def decode_images(raw: Any) -> List[bytes]:
    """Validate the ``images`` extension and return each image's bytes.

    Each item is a base64 data URL (``data:image/png;base64,...``) or an
    object ``{"content_type": ..., "base64": ...}``. PNG, JPEG or WebP; at most
    MAX_IMAGES; MAX_IMAGE_BYTES each and MAX_TOTAL_IMAGE_BYTES together.
    Remote URLs are not fetched.
    """
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise BadRequest("images must be an array")
    if len(raw) > MAX_IMAGES:
        raise BadRequest(f"at most {MAX_IMAGES} images per request")
    out: List[bytes] = []
    total = 0
    for index, item in enumerate(raw):
        if isinstance(item, str):
            head, sep, encoded = item.strip().partition(",")
            head = head.lower()
            if not sep or not head.startswith("data:") or not head.endswith(";base64"):
                raise BadRequest(f"image {index} must be a base64 data URL; remote URLs are not accepted")
            declared = head[len("data:"):-len(";base64")]
        elif isinstance(item, dict) and isinstance(item.get("content_type"), str) and isinstance(item.get("base64"), str):
            declared, encoded = item["content_type"].strip().lower(), item["base64"]
        else:
            raise BadRequest(f"image {index} must be a data URL string or an object with content_type and base64")
        if declared == "image/jpg":
            declared = "image/jpeg"
        if declared not in IMAGE_TYPES:
            raise BadRequest(f"image {index}: type must be image/png, image/jpeg or image/webp")
        if len(encoded) > MAX_IMAGE_CHARS:
            raise BadRequest(f"image {index} exceeds {MAX_IMAGE_BYTES // (1024 * 1024)} MiB")
        try:
            data = base64.b64decode(encoded.translate(_WHITESPACE), validate=True)
        except (binascii.Error, ValueError):
            raise BadRequest(f"image {index} is not valid base64") from None
        if not data or len(data) > MAX_IMAGE_BYTES:
            raise BadRequest(f"image {index} must be between 1 byte and {MAX_IMAGE_BYTES // (1024 * 1024)} MiB")
        total += len(data)
        if total > MAX_TOTAL_IMAGE_BYTES:
            raise BadRequest(f"images exceed {MAX_TOTAL_IMAGE_BYTES // (1024 * 1024)} MiB in total")
        if _image_type(data) != declared:
            raise BadRequest(f"image {index} is not the declared {declared}")
        out.append(data)
    return out


# ---------------------------------------------------------------------------
# Dynamic batching
# ---------------------------------------------------------------------------

class Oversubscribed(Exception):
    pass


@dataclass
class _Pending:
    request: Dict[str, Any]
    future: "asyncio.Future"
    enqueued: float = field(default_factory=time.monotonic)


@dataclass
class Stats:
    requests: int = 0
    batches: int = 0
    batched_requests: int = 0
    rejected_busy: int = 0
    failed: int = 0
    largest_batch: int = 0

    def as_dict(self) -> Dict[str, Any]:
        avg = self.batched_requests / self.batches if self.batches else 0.0
        return {"requests": self.requests, "batches": self.batches, "avg_batch_size": round(avg, 2),
                "largest_batch": self.largest_batch, "rejected_busy": self.rejected_busy, "failed": self.failed}


class Batcher:
    """Queues requests and answers them in batches on one inference thread."""

    def __init__(self, config: ServiceConfig):
        self.config = config
        self.engine: Any = None
        self.stats = Stats()
        self._queue: "asyncio.Queue[_Pending]" = asyncio.Queue()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="clef-infer")
        self._task: Optional[asyncio.Task] = None

    @property
    def ready(self) -> bool:
        return self.engine is not None

    def queue_depth(self) -> int:
        return self._queue.qsize()

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        factory = MODEL_FACTORY or ClefEngine
        self.engine = await loop.run_in_executor(self._executor, factory, self.config)
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        # Nobody will answer what is still queued; do not leave its callers waiting.
        while not self._queue.empty():
            pending = self._queue.get_nowait()
            if not pending.future.done():
                pending.future.cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)

    async def submit(self, request: Dict[str, Any]) -> Tuple[Dict[str, Any], Encoded]:
        if self._queue.qsize() >= self.config.max_queue:
            self.stats.rejected_busy += 1
            raise Oversubscribed()
        pending = _Pending(request=request, future=asyncio.get_running_loop().create_future())
        self._queue.put_nowait(pending)
        self.stats.requests += 1
        return await pending.future

    async def _run(self) -> None:
        while True:
            first = await self._queue.get()
            batch = [first]
            deadline = time.monotonic() + self.config.batch_wait_ms / 1000.0
            try:
                while len(batch) < self.config.max_batch:
                    timeout = deadline - time.monotonic()
                    if timeout <= 0:
                        # Past the wait: take only what is already queued.
                        if self._queue.empty():
                            break
                        batch.append(self._queue.get_nowait())
                        continue
                    try:
                        batch.append(await asyncio.wait_for(self._queue.get(), timeout))
                    except asyncio.TimeoutError:
                        break
                await self._process(batch)
            except asyncio.CancelledError:
                for p in batch:
                    if not p.future.done():
                        p.future.cancel()
                raise
            except Exception as error:  # the loop must survive anything
                logger.error("clef_batch_failed error_type=%s", type(error).__name__)
                for p in batch:
                    if not p.future.done():
                        p.future.set_exception(RuntimeError("inference failed"))

    async def _process(self, batch: List[_Pending]) -> None:
        loop = asyncio.get_running_loop()
        # Tokenize each request on the inference thread. A request the model
        # cannot take fails alone; the rest of the batch goes ahead.
        encoded: List[Tuple[_Pending, Encoded]] = []
        for pending in batch:
            if pending.future.done():       # caller went away
                continue
            try:
                item = await loop.run_in_executor(self._executor, self.engine.encode, pending.request)
            except ValueError as error:
                _fail(pending, BadRequest(str(error)))
                continue
            except Exception as error:      # anything else is this request's failure, not the batch's
                self.stats.failed += 1
                logger.error("clef_encode_failed error_type=%s", type(error).__name__)
                _fail(pending, RuntimeError("inference failed"))
                continue
            encoded.append((pending, item))

        # Split so no forward pass exceeds the padded-token budget.
        for group in self._groups(encoded):
            await self._infer(group)

    def _groups(self, encoded: List[Tuple[_Pending, Encoded]]) -> List[List[Tuple[_Pending, Encoded]]]:
        groups: List[List[Tuple[_Pending, Encoded]]] = []
        current: List[Tuple[_Pending, Encoded]] = []
        longest = 0
        for pair in encoded:
            new_longest = max(longest, pair[1].tokens)
            if current and new_longest * (len(current) + 1) > self.config.max_batch_tokens:
                groups.append(current)
                current, new_longest = [], pair[1].tokens
            current.append(pair)
            longest = new_longest
        if current:
            groups.append(current)
        return groups

    async def _infer(self, group: List[Tuple[_Pending, Encoded]]) -> None:
        loop = asyncio.get_running_loop()
        # A caller that has gone (disconnected, timed out) gets no forward pass.
        group = [pair for pair in group if not pair[0].future.done()]
        if not group:
            return
        items = [item for _, item in group]
        requests = [pending.request for pending, _ in group]
        retry_singly = False
        try:
            answers = await loop.run_in_executor(self._executor, self.engine.infer, items, requests)
            if len(answers) != len(group):
                raise RuntimeError("engine returned the wrong number of answers")
        except Exception as error:
            if len(group) > 1 and _is_out_of_memory(error):
                retry_singly = True
            else:
                self.stats.failed += len(group)
                logger.error("clef_inference_failed error_type=%s batch=%d", type(error).__name__, len(group))
                for pending, _ in group:
                    _fail(pending, RuntimeError("inference failed"))
                return
        if retry_singly:
            # The batch did not fit; answer its requests one at a time. Done
            # outside the except block so the failed batch's traceback (and
            # the tensors it holds) is released first.
            logger.warning("clef_batch_oom size=%d; retrying singly", len(group))
            for pair in group:
                await self._infer([pair])
            return
        self.stats.batches += 1
        self.stats.batched_requests += len(group)
        self.stats.largest_batch = max(self.stats.largest_batch, len(group))
        for (pending, item), answer in zip(group, answers):
            if not pending.future.done():
                pending.future.set_result((answer, item))


def _fail(pending: "_Pending", error: BaseException) -> None:
    """Fail one request, unless its caller has already gone."""
    if not pending.future.done():
        pending.future.set_exception(error)


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
    """Await ``work``; cancel it if the client disconnects first, so a request
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


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def create_app(config: ServiceConfig):
    from fastapi import FastAPI, Header, HTTPException, Request
    from fastapi.responses import JSONResponse

    if not config.api_key and not config.allow_no_auth:
        raise SystemExit("CLEF_API_KEY is not set. Set it, or CLEF_ALLOW_NO_AUTH=1 to serve without a key.")

    batcher = Batcher(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await batcher.start()
        logger.info("clef_ready model=%s device=%s", config.served_name, getattr(batcher.engine, "device", "?"))
        try:
            yield
        finally:
            await batcher.stop()

    app = FastAPI(title="MindRouter Clef service", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.batcher = batcher

    def authorized(authorization: Optional[str]) -> bool:
        if not config.api_key:
            return True
        if not authorization or not authorization.startswith("Bearer "):
            return False
        return hmac.compare_digest(authorization[len("Bearer "):].strip().encode(), config.api_key.encode())

    def require_key(authorization: Optional[str]) -> None:
        if not authorized(authorization):
            raise HTTPException(status_code=401, detail="missing or invalid bearer key")

    @app.get("/health")
    async def health(authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
        status = "ok" if batcher.ready else "loading"
        if not authorized(authorization):
            return {"status": status}      # liveness only without the key
        return {"status": status, "model": config.served_name, "source": config.model,
                "device": getattr(batcher.engine, "device", None), "max_length": config.max_length,
                "max_batch": config.max_batch, "images": True,
                "fast_path": getattr(batcher.engine, "fast_path", None), "queue_depth": batcher.queue_depth(),
                "max_queue": config.max_queue, "stats": batcher.stats.as_dict()}

    @app.get("/v1/models")
    async def models(authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
        require_key(authorization)
        return {"models": [{"name": config.served_name, "release_date": "2026-09-30",
                            "description": "Cloudflare Clef decision model (System One)"}]}

    @app.post("/v1/systemone")
    async def systemone(request: Request, authorization: Optional[str] = Header(default=None)):
        require_key(authorization)
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > config.max_body_bytes:
            raise HTTPException(status_code=413, detail="request body is too large")
        raw = await _read_body(request, config.max_body_bytes)
        if raw is None:
            raise HTTPException(status_code=413, detail="request body is too large")
        try:
            body = validate_request(json.loads(raw))
        except BadRequest as error:
            raise HTTPException(status_code=422, detail=str(error)) from None
        except (ValueError, UnicodeDecodeError):
            raise HTTPException(status_code=422, detail="request body is not valid JSON") from None
        if not batcher.ready:
            raise HTTPException(status_code=503, detail="model is loading", headers={"Retry-After": "10"})

        started = time.monotonic()
        try:
            answers, item = await _unless_disconnected(request, batcher.submit(body))
        except ClientGone:
            raise HTTPException(status_code=499, detail="client closed request") from None
        except Oversubscribed:
            raise HTTPException(status_code=503, detail="server busy, try again shortly",
                                headers={"Retry-After": "1"}) from None
        except BadRequest as error:
            raise HTTPException(status_code=422, detail=str(error)) from None
        except Exception:
            # Never a made-up answer, and never the error's text (it can quote the request).
            raise HTTPException(status_code=500, detail="inference failed") from None
        return JSONResponse({
            "model": config.served_name,
            "answers": answers,
            "usage": {
                "input_tokens": item.tokens,
                "output_tokens": 0,
                # Not in TypeSafe's contract: Clef cuts the state to fit; say so.
                "truncated": item.state_tokens_dropped > 0,
                "state_tokens_dropped": item.state_tokens_dropped,
            },
            "metadata": {"seconds": round(time.monotonic() - started, 4), "images": item.images},
        })

    return app
