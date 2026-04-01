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

import json
import time
import uuid
from typing import Any, Dict, Optional, Tuple

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.auth import authenticate_request, get_current_api_key
from backend.app.core.canonical_schemas import (
    CanonicalChatRequest,
    CanonicalEmbeddingRequest,
    CanonicalRerankRequest,
    CanonicalScoreRequest,
)
from backend.app.core.scheduler.policy import get_scheduler
from backend.app.core.telemetry.registry import get_registry
from backend.app.core.translators import OpenAIInTranslator, OllamaOutTranslator, VLLMOutTranslator
from backend.app.db import crud
from backend.app.db.models import ApiKey, BackendEngine, Modality, User
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

    service = InferenceService(db)

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
            service=service,
            user=user,
            api_key=api_key,
            http_request=request,
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
