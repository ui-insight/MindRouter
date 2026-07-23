############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# v1_openai.py: OpenAI-compatible API endpoints (/v1/*)
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""OpenAI-compatible API endpoints."""

import base64
import json
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.auth import authenticate_request, get_current_api_key
from backend.app.core.canonical_schemas import (
    CanonicalChatRequest,
    CanonicalEmbeddingRequest,
    CanonicalImageRequest,
    CanonicalRerankRequest,
    CanonicalScoreRequest,
)
from backend.app.core.scheduler.policy import get_scheduler
from backend.app.core.telemetry.registry import get_registry
from backend.app.core.translators import OpenAIInTranslator, OllamaOutTranslator, VLLMOutTranslator
from backend.app.db import crud
from backend.app.db.models import ApiKey, BackendEngine, Modality, RequestStatus, User
from backend.app.db.session import get_async_db
from backend.app.logging_config import bind_request_context, get_logger
from backend.app.services.inference import InferenceService
from backend.app.settings import get_settings

logger = get_logger(__name__)
router = APIRouter(prefix="/v1", tags=["openai"])


@router.post("/chat/completions")
async def chat_completions(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
    auth: Tuple[User, ApiKey] = Depends(authenticate_request),
):
    """
    OpenAI-compatible chat completions endpoint.

    Supports:
    - Streaming and non-streaming responses
    - Multi-turn conversations
    - Vision/multimodal inputs
    - Structured outputs (JSON mode, JSON schema)
    """
    user, api_key = auth

    # Parse request body
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON body",
        )

    # Generate request ID
    request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    bind_request_context(request_id=request_id, user_id=user.id)

    # Translate to canonical format
    try:
        canonical = OpenAIInTranslator.translate_chat_request(body)
        canonical.request_id = request_id
        canonical.user_id = user.id
        canonical.api_key_id = api_key.id
    except Exception as e:
        logger.warning("translation_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request format: {str(e)}",
        )

    # Early model validation — reject unknown models before queuing
    registry = get_registry()
    canonical.model, _ = registry.resolve_alias(canonical.model)
    if not await registry.model_exists(canonical.model):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "message": f"The model '{canonical.model}' does not exist",
                    "type": "invalid_request_error",
                    "code": "model_not_found",
                }
            },
        )

    # Create inference service
    service = InferenceService(db)

    # Handle streaming vs non-streaming
    if canonical.stream:
        return StreamingResponse(
            service.stream_chat_completion(canonical, user, api_key, request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Request-ID": request_id,
            },
        )
    else:
        response = await service.chat_completion(canonical, user, api_key, request)
        return response


@router.post("/completions")
async def completions(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
    auth: Tuple[User, ApiKey] = Depends(authenticate_request),
):
    """
    OpenAI-compatible text completions endpoint (legacy).

    Internally converts to chat format for processing.
    """
    user, api_key = auth

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON body",
        )

    request_id = f"cmpl-{uuid.uuid4().hex[:24]}"
    bind_request_context(request_id=request_id, user_id=user.id)

    # Translate to canonical completion request
    try:
        from backend.app.core.translators.openai_in import OpenAIInTranslator
        completion_req = OpenAIInTranslator.translate_completion_request(body)
        completion_req.request_id = request_id
        completion_req.user_id = user.id
        completion_req.api_key_id = api_key.id

        # Convert to chat for unified processing
        canonical = completion_req.to_chat_request()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request format: {str(e)}",
        )

    # Early model validation
    registry = get_registry()
    canonical.model, _ = registry.resolve_alias(canonical.model)
    if not await registry.model_exists(canonical.model):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "message": f"The model '{canonical.model}' does not exist",
                    "type": "invalid_request_error",
                    "code": "model_not_found",
                }
            },
        )

    service = InferenceService(db)

    if canonical.stream:
        return StreamingResponse(
            service.stream_chat_completion(canonical, user, api_key, request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Request-ID": request_id,
            },
        )
    else:
        response = await service.chat_completion(canonical, user, api_key, request)
        return response


@router.post("/embeddings")
async def embeddings(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
    auth: Tuple[User, ApiKey] = Depends(authenticate_request),
):
    """
    OpenAI-compatible embeddings endpoint.
    """
    user, api_key = auth

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON body",
        )

    request_id = f"emb-{uuid.uuid4().hex[:24]}"
    bind_request_context(request_id=request_id, user_id=user.id)

    try:
        canonical = OpenAIInTranslator.translate_embedding_request(body)
        canonical.request_id = request_id
        canonical.user_id = user.id
        canonical.api_key_id = api_key.id
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request format: {str(e)}",
        )

    # Early model validation
    registry = get_registry()
    canonical.model, _ = registry.resolve_alias(canonical.model)
    if not await registry.model_exists(canonical.model):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "message": f"The model '{canonical.model}' does not exist",
                    "type": "invalid_request_error",
                    "code": "model_not_found",
                }
            },
        )

    service = InferenceService(db)
    response = await service.embedding(canonical, user, api_key, request)
    return response


@router.post("/rerank")
async def rerank(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
    auth: Tuple[User, ApiKey] = Depends(authenticate_request),
):
    """
    Rerank documents against a query.
    """
    user, api_key = auth

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON body",
        )

    request_id = f"rnk-{uuid.uuid4().hex[:24]}"
    bind_request_context(request_id=request_id, user_id=user.id)

    try:
        canonical = OpenAIInTranslator.translate_rerank_request(body)
        canonical.request_id = request_id
        canonical.user_id = user.id
        canonical.api_key_id = api_key.id
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request format: {str(e)}",
        )

    # Early model validation
    registry = get_registry()
    canonical.model, _ = registry.resolve_alias(canonical.model)
    if not await registry.model_exists(canonical.model):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "message": f"The model '{canonical.model}' does not exist",
                    "type": "invalid_request_error",
                    "code": "model_not_found",
                }
            },
        )

    service = InferenceService(db)
    response = await service.rerank(canonical, user, api_key, request)
    return response


@router.post("/score")
async def score(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
    auth: Tuple[User, ApiKey] = Depends(authenticate_request),
):
    """
    Score similarity between text pairs.
    """
    user, api_key = auth

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON body",
        )

    request_id = f"scr-{uuid.uuid4().hex[:24]}"
    bind_request_context(request_id=request_id, user_id=user.id)

    try:
        canonical = OpenAIInTranslator.translate_score_request(body)
        canonical.request_id = request_id
        canonical.user_id = user.id
        canonical.api_key_id = api_key.id
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request format: {str(e)}",
        )

    # Early model validation
    registry = get_registry()
    canonical.model, _ = registry.resolve_alias(canonical.model)
    if not await registry.model_exists(canonical.model):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "message": f"The model '{canonical.model}' does not exist",
                    "type": "invalid_request_error",
                    "code": "model_not_found",
                }
            },
        )

    service = InferenceService(db)
    response = await service.score(canonical, user, api_key, request)
    return response


@router.post("/tokenize")
async def tokenize(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
    auth: Tuple[User, ApiKey] = Depends(authenticate_request),
):
    """
    Count input tokens for a chat request.

    Uses the backend's native tokenizer for vLLM models (exact count including
    chat template and tool definitions). Falls back to tiktoken estimation for
    Ollama models.

    Returns:
        count: number of input tokens
        max_model_len: model context window size
        is_estimate: true if count is a tiktoken estimate (Ollama or fallback)
    """
    user, api_key = auth

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON body",
        )

    model = body.get("model")
    if not model:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="'model' is required",
        )

    registry = get_registry()
    model, _ = registry.resolve_alias(model)
    if not await registry.model_exists(model):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "message": f"The model '{model}' does not exist",
                    "type": "invalid_request_error",
                    "code": "model_not_found",
                }
            },
        )

    # Translate to canonical format for token counting
    try:
        canonical = OpenAIInTranslator.translate_chat_request(body)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request format: {str(e)}",
        )

    # Find a backend that has this model
    backends = await registry.get_backends_with_model(model)
    if not backends:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No healthy backends available for this model",
        )

    backend = backends[0]

    # Get model context length
    models = await crud.get_models_for_backend(db, backend.id)
    target = next((m for m in models if m.name == model), None)
    max_model_len = target.context_length if target else None

    service = InferenceService(db)
    count, is_estimate = await service._count_input_tokens(canonical, backend)

    return {
        "count": count,
        "max_model_len": max_model_len,
        "is_estimate": is_estimate,
    }


# ---------------------------------------------------------------------------
# /v1/ocr – Document OCR via multimodal LLM
# ---------------------------------------------------------------------------

_OCR_ALLOWED_TYPES = {
    "image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif",
    "image/tiff", "image/bmp",
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/msword",
    "application/vnd.ms-powerpoint",
    "application/vnd.ms-excel",
}

# Extension → MIME for cases where content_type is generic
_OCR_EXT_MAP = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".webp": "image/webp", ".gif": "image/gif", ".tiff": "image/tiff",
    ".tif": "image/tiff", ".bmp": "image/bmp", ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".doc": "application/msword",
    ".ppt": "application/vnd.ms-powerpoint",
    ".xls": "application/vnd.ms-excel",
}


@router.post("/ocr")
async def ocr(
    request: Request,
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
    output_format: Optional[str] = Form("markdown"),
    chunk_size: Optional[int] = Form(None),
    overlap: Optional[int] = Form(None),
    dpi: Optional[int] = Form(None),
    db: AsyncSession = Depends(get_async_db),
    auth: Tuple[User, ApiKey] = Depends(authenticate_request),
):
    """
    OCR endpoint: convert images, PDFs, and Office documents to markdown or JSON.

    Accepts multipart file upload. For multi-page documents, pages are processed
    in overlapping chunks and merged deterministically.
    """
    from backend.app.services.ocr import get_ocr_config, perform_ocr

    user, api_key = auth
    request_id = f"ocr-{uuid.uuid4().hex[:24]}"
    bind_request_context(request_id=request_id, user_id=user.id)

    # Load OCR config from admin settings
    ocr_config = await get_ocr_config(db)

    if not ocr_config["enabled"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OCR is currently disabled by the administrator",
        )

    # Apply defaults from config
    if model is None:
        model = ocr_config["model"]
    if chunk_size is None:
        chunk_size = ocr_config["chunk_size"]
    if overlap is None:
        overlap = ocr_config["overlap"]
    if dpi is None:
        dpi = ocr_config["dpi"]

    # Validate output format
    if output_format not in ("markdown", "json"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="output_format must be 'markdown' or 'json'",
        )

    # Validate chunk params
    if overlap >= chunk_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="overlap must be less than chunk_size",
        )

    # Resolve content type from file extension if generic
    content_type = file.content_type or "application/octet-stream"
    if content_type == "application/octet-stream" and file.filename:
        import os
        ext = os.path.splitext(file.filename)[1].lower()
        content_type = _OCR_EXT_MAP.get(ext, content_type)

    if content_type not in _OCR_ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {content_type}. Supported: images, PDF, DOCX, PPTX, XLSX",
        )

    # Validate model exists
    registry = get_registry()
    model, _ = registry.resolve_alias(model)
    if not await registry.model_exists(model):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model}' not found",
        )

    # Read file
    file_bytes = await file.read()
    max_size = ocr_config["max_file_size_mb"] * 1024 * 1024
    if len(file_bytes) > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum size of {ocr_config['max_file_size_mb']}MB",
        )

    try:
        result = await perform_ocr(
            file_bytes=file_bytes,
            content_type=content_type,
            filename=file.filename or "document",
            model=model,
            output_format=output_format,
            chunk_size=chunk_size,
            overlap=overlap,
            dpi=dpi,
            ocr_config=ocr_config,
            user=user,
            api_key=api_key,
            http_request=request,
            prompt_template=ocr_config["prompt_ocr"],
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(e),
        )

    return {
        "id": request_id,
        "object": "ocr.result",
        "created": int(time.time()),
        "model": model,
        "content": result["content"],
        "format": result["format"],
        "pages": result["pages"],
        "chunks_processed": result["chunks_processed"],
        "usage": result["usage"],
    }


@router.post("/ocrmd")
async def ocrmd(
    request: Request,
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
    chunk_size: Optional[int] = Form(None),
    overlap: Optional[int] = Form(None),
    dpi: Optional[int] = Form(None),
    db: AsyncSession = Depends(get_async_db),
    auth: Tuple[User, ApiKey] = Depends(authenticate_request),
):
    """
    Simplified OCR endpoint that returns raw markdown (not JSON-wrapped).

    Same pipeline as /v1/ocr but the response body is plain text/markdown.
    """
    from fastapi.responses import PlainTextResponse
    from backend.app.services.ocr import get_ocr_config, perform_ocr

    user, api_key = auth
    request_id = f"ocr-{uuid.uuid4().hex[:24]}"
    bind_request_context(request_id=request_id, user_id=user.id)

    ocr_config = await get_ocr_config(db)

    if not ocr_config["enabled"]:
        return PlainTextResponse("OCR is currently disabled", status_code=503)

    if model is None:
        model = ocr_config["model"]
    if chunk_size is None:
        chunk_size = ocr_config["chunk_size"]
    if overlap is None:
        overlap = ocr_config["overlap"]
    if dpi is None:
        dpi = ocr_config["dpi"]

    if overlap >= chunk_size:
        return PlainTextResponse("overlap must be less than chunk_size", status_code=400)

    content_type = file.content_type or "application/octet-stream"
    if content_type == "application/octet-stream" and file.filename:
        import os
        ext = os.path.splitext(file.filename)[1].lower()
        content_type = _OCR_EXT_MAP.get(ext, content_type)

    if content_type not in _OCR_ALLOWED_TYPES:
        return PlainTextResponse(f"Unsupported file type: {content_type}", status_code=400)

    registry = get_registry()
    model, _ = registry.resolve_alias(model)
    if not await registry.model_exists(model):
        return PlainTextResponse(f"Model '{model}' not found", status_code=404)

    file_bytes = await file.read()
    max_size = ocr_config["max_file_size_mb"] * 1024 * 1024
    if len(file_bytes) > max_size:
        return PlainTextResponse(
            f"File exceeds maximum size of {ocr_config['max_file_size_mb']}MB",
            status_code=413,
        )

    try:
        result = await perform_ocr(
            file_bytes=file_bytes,
            content_type=content_type,
            filename=file.filename or "document",
            model=model,
            output_format="markdown",
            chunk_size=chunk_size,
            overlap=overlap,
            dpi=dpi,
            ocr_config=ocr_config,
            user=user,
            api_key=api_key,
            http_request=request,
            prompt_template=ocr_config["prompt_ocrmd"],
        )
    except ValueError as e:
        return PlainTextResponse(str(e), status_code=400)
    except RuntimeError as e:
        return PlainTextResponse(str(e), status_code=501)

    return PlainTextResponse(result["content"], media_type="text/markdown")


async def _prepare_image_canonical(
    *,
    db: AsyncSession,
    request: Request,
    user: User,
    api_key: ApiKey,
    request_id: str,
    endpoint: str,
    params: Dict[str, Any],
    images_b64: Optional[List[str]] = None,
    strength: Optional[float] = None,
) -> CanonicalImageRequest:
    """Shared access-control → policy → guardrails → canonical build for both
    ``/images/generations`` (txt2img) and ``/images/edits`` (img2img).

    ``params`` holds the generation knobs (model/prompt/n/size/quality/style/
    response_format/num_inference_steps/guidance_scale/seed/user). When
    ``images_b64`` is set the returned canonical carries reference image(s) and
    is routed to the backend edits route downstream.
    """
    # ── Access control ───────────────────────────────────────────
    img_enabled = await crud.get_config_json(db, "img.enabled", True)
    if not img_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Image generation is currently disabled",
        )
    if not user.image_generation_enabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Image generation is not enabled for your account. Contact an administrator.",
        )

    prompt = params.get("prompt")
    if not prompt:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="'prompt' is required",
        )

    # ── Load config defaults and guardrails ──────────────────────
    default_model = await crud.get_config_json(db, "img.default_model", "black-forest-labs/FLUX.2-dev")
    default_size = await crud.get_config_json(db, "img.default_size", "1024x1024")
    default_steps = await crud.get_config_json(db, "img.default_steps", 20)
    default_guidance = await crud.get_config_json(db, "img.default_guidance_scale", 3.5)
    max_n = await crud.get_config_json(db, "img.max_n", 4)
    max_steps = await crud.get_config_json(db, "img.max_steps", 50)
    allowed_sizes_str = await crud.get_config_json(db, "img.allowed_sizes", "512x512,768x768,1024x1024,1024x768,768x1024")
    allowed_sizes = [s.strip() for s in allowed_sizes_str.split(",") if s.strip()]

    model = params.get("model") or default_model

    # ── LLM-as-judge policy check ────────────────────────────────
    policy_verdict = None
    policy_text = await crud.get_config_json(db, "img.policy", "")
    if policy_text and policy_text.strip():
        from backend.app.services.image_policy import evaluate_prompt

        primary_judge = await crud.get_config_json(db, "img.judge_model", "")
        secondary_judge = await crud.get_config_json(db, "img.judge_model_secondary", "")

        policy_verdict = await evaluate_prompt(
            prompt=prompt,
            policy=policy_text,
            primary_model=primary_judge,
            secondary_model=secondary_judge,
            is_edit=bool(images_b64),
        )

        if not policy_verdict.passed:
            # Record the denied request for auditing
            denied_req = await crud.create_request(
                db=db,
                user_id=user.id,
                api_key_id=api_key.id,
                endpoint=endpoint,
                model=model,
                modality=Modality.IMAGE_GENERATION,
                prompt=prompt,
                parameters={
                    "policy_verdict": policy_verdict.to_dict(),
                    "size": params.get("size", default_size),
                    "n": params.get("n", 1),
                },
                client_ip=(
                    request.headers.get("x-forwarded-for", "").split(",")[0].strip()
                    or request.headers.get("x-real-ip")
                    or (request.client.host if request.client else None)
                ),
                user_agent=request.headers.get("user-agent"),
            )
            denied_req.status = RequestStatus.FAILED
            denied_req.error_message = f"Policy violation: {policy_verdict.reason}"
            denied_req.error_code = "policy_violation"
            await db.commit()

            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": {
                        "message": f"Your image request was denied by content policy: {policy_verdict.reason}",
                        "type": "content_policy_violation",
                        "code": "policy_violation",
                    }
                },
            )

    # Enforce guardrails
    req_n = min(params.get("n", 1), max_n)
    req_size = params.get("size", default_size)
    if allowed_sizes and req_size not in allowed_sizes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Size '{req_size}' is not allowed. Allowed sizes: {', '.join(allowed_sizes)}",
        )

    # Enforce max dimensions
    max_width = await crud.get_config_json(db, "img.max_width", 1024)
    max_height = await crud.get_config_json(db, "img.max_height", 1024)
    try:
        w, h = req_size.split("x")
        w, h = int(w), int(h)
        if w > max_width or h > max_height:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Image dimensions {req_size} exceed maximum allowed ({max_width}x{max_height})",
            )
    except ValueError:
        pass  # Non-standard size format — let the backend handle it

    req_steps = params.get("num_inference_steps") or default_steps
    if req_steps > max_steps:
        req_steps = max_steps

    req_guidance = params.get("guidance_scale") if params.get("guidance_scale") is not None else default_guidance

    # Build canonical request
    try:
        canonical = CanonicalImageRequest(
            model=model,
            prompt=prompt,
            n=req_n,
            size=req_size,
            quality=params.get("quality", "standard"),
            style=params.get("style"),
            response_format=params.get("response_format", "url"),
            num_inference_steps=req_steps,
            guidance_scale=req_guidance,
            seed=params.get("seed"),
            user=params.get("user"),
            image=images_b64 or None,
            strength=strength,
            request_id=request_id,
            user_id=user.id,
            api_key_id=api_key.id,
            policy_verdict=policy_verdict.to_dict() if policy_verdict else None,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request format: {str(e)}",
        )

    # Early model validation
    registry = get_registry()
    canonical.model, _ = registry.resolve_alias(canonical.model)
    if not await registry.model_exists(canonical.model):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "message": f"The model '{canonical.model}' does not exist",
                    "type": "invalid_request_error",
                    "code": "model_not_found",
                }
            },
        )

    return canonical


@router.post("/images/generations")
async def image_generations(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
    auth: Tuple[User, ApiKey] = Depends(authenticate_request),
):
    """
    OpenAI-compatible image generation endpoint.

    Routes to diffusion backends (e.g. FLUX via openedai-images-flux).
    """
    user, api_key = auth

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON body",
        )

    request_id = f"img-{uuid.uuid4().hex[:24]}"
    bind_request_context(request_id=request_id, user_id=user.id)

    canonical = await _prepare_image_canonical(
        db=db, request=request, user=user, api_key=api_key,
        request_id=request_id, endpoint="/v1/images/generations",
        params=body,
    )

    service = InferenceService(db)
    response = await service.image_generation(canonical, user, api_key, request)
    return response


# How many base64 reference images an edit request may carry — mirrors the
# backend server's MAX_REF_IMAGES cap (VRAM protection).
_MAX_EDIT_IMAGES = 4


@router.post("/images/edits")
async def image_edits(
    request: Request,
    image: List[UploadFile] = File(...),
    prompt: str = Form(...),
    model: Optional[str] = Form(None),
    n: int = Form(1),
    size: Optional[str] = Form(None),
    response_format: str = Form("url"),
    strength: Optional[float] = Form(None),
    num_inference_steps: Optional[int] = Form(None),
    guidance_scale: Optional[float] = Form(None),
    seed: Optional[int] = Form(None),
    user_field: Optional[str] = Form(None, alias="user"),
    db: AsyncSession = Depends(get_async_db),
    auth: Tuple[User, ApiKey] = Depends(authenticate_request),
):
    """OpenAI-compatible image **edit** (img2img / reference-edit) endpoint.

    Accepts multipart/form-data with one or more ``image`` files plus a
    ``prompt``. The reference image(s) condition generation on the diffusion
    backend's ``/v1/images/edits`` route. FLUX.2 Klein edits are structure-
    preserving; ``strength`` is accepted for forward-compat but ignored.
    """
    user, api_key = auth

    if not image:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="at least one 'image' file is required")
    if len(image) > _MAX_EDIT_IMAGES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"at most {_MAX_EDIT_IMAGES} reference image(s) allowed (got {len(image)})",
        )

    # Read + base64-encode each reference image (bounded by img.max_image_upload_mb).
    max_bytes = int(await crud.get_config_json(db, "img.max_image_upload_mb", 10)) * 1024 * 1024
    images_b64: List[str] = []
    for up in image:
        ct = (up.content_type or "").lower()
        if not ct.startswith("image/"):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="each 'image' must be an image file")
        data = await up.read()
        if len(data) > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"reference image exceeds {max_bytes // 1024 // 1024}MB",
            )
        images_b64.append(base64.b64encode(data).decode("utf-8"))

    request_id = f"img-{uuid.uuid4().hex[:24]}"
    bind_request_context(request_id=request_id, user_id=user.id)

    params: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "n": n,
        "size": size,
        "response_format": response_format,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "user": user_field,
    }
    canonical = await _prepare_image_canonical(
        db=db, request=request, user=user, api_key=api_key,
        request_id=request_id, endpoint="/v1/images/edits",
        params=params, images_b64=images_b64, strength=strength,
    )

    service = InferenceService(db)
    response = await service.image_generation(canonical, user, api_key, request)
    return response
