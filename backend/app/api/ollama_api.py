############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# ollama_api.py: Ollama-compatible API endpoints (/api/*)
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Ollama-compatible API endpoints."""

import uuid
from typing import Any, Dict, Tuple

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.auth import authenticate_request
from backend.app.core.translators import OllamaInTranslator
from backend.app.db.models import ApiKey, User
from backend.app.db.session import get_async_db
from backend.app.logging_config import bind_request_context, get_logger
from backend.app.services.inference import InferenceService
from backend.app.core.telemetry.registry import get_registry

logger = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["ollama"])


@router.post("/chat")
async def ollama_chat(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
    auth: Tuple[User, ApiKey] = Depends(authenticate_request),
):
    """
    Ollama-compatible /api/chat endpoint.

    Supports:
    - Streaming (default) and non-streaming
    - Multi-turn conversations
    - Images via base64
    - Format parameter for JSON mode
    """
    user, api_key = auth

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON body",
        )

    request_id = f"ollama-chat-{uuid.uuid4().hex[:16]}"
    bind_request_context(request_id=request_id, user_id=user.id)

    # Translate to canonical format
    try:
        canonical = OllamaInTranslator.translate_chat_request(body)
        canonical.request_id = request_id
        canonical.user_id = user.id
        canonical.api_key_id = api_key.id
    except Exception as e:
        logger.warning("ollama_translation_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request format: {str(e)}",
        )

    # Early model validation — reject unknown models before queuing
    registry = get_registry()
    if not await registry.model_exists(canonical.model):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"model '{canonical.model}' not found",
        )

    service = InferenceService(db)

    # Ollama defaults to streaming
    if canonical.stream:
        return StreamingResponse(
            service.stream_ollama_chat(canonical, user, api_key, request),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
    else:
        response = await service.ollama_chat(canonical, user, api_key, request)
        return response


@router.post("/generate")
async def ollama_generate(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
    auth: Tuple[User, ApiKey] = Depends(authenticate_request),
):
    """
    Ollama-compatible /api/generate endpoint.

    For text completion (non-chat) style requests.
    """
    user, api_key = auth

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON body",
        )

    request_id = f"ollama-gen-{uuid.uuid4().hex[:16]}"
    bind_request_context(request_id=request_id, user_id=user.id)

    try:
        canonical = OllamaInTranslator.translate_generate_request(body)
        canonical.request_id = request_id
        canonical.user_id = user.id
        canonical.api_key_id = api_key.id

        # Convert to chat for unified processing
        chat_canonical = canonical.to_chat_request()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request format: {str(e)}",
        )

    # Early model validation
    registry = get_registry()
    if not await registry.model_exists(chat_canonical.model):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"model '{chat_canonical.model}' not found",
        )

    service = InferenceService(db)

    if canonical.stream:
        return StreamingResponse(
            service.stream_ollama_generate(chat_canonical, user, api_key, request),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
    else:
        response = await service.ollama_generate(chat_canonical, user, api_key, request)
        return response


@router.get("/tags")
async def ollama_tags(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
    auth: Tuple[User, ApiKey] = Depends(authenticate_request),
):
    """
    Ollama-compatible /api/tags endpoint.

    Lists available models in Ollama format.
    """
    user, api_key = auth

    registry = get_registry()
    backends = await registry.get_healthy_backends()

    # Collect all models
    models = []
    seen_models = set()

    for backend in backends:
        backend_models = await registry.get_backend_models(backend.id)
        for model in backend_models:
            if model.name not in seen_models:
                seen_models.add(model.name)
                models.append({
                    "name": model.name,
                    "model": model.name,
                    "modified_at": model.updated_at.isoformat() if model.updated_at else None,
                    "size": 0,  # We don't track size
                    "digest": "",
                    "details": {
                        "parent_model": model.parent_model or "",
                        "format": model.model_format or "gguf",
                        "family": model.family or "",
                        "parameter_size": model.parameter_count or "",
                        "quantization_level": model.quantization or "",
                    },
                    "context_length": model.context_length,
                    "model_max_context": model.model_max_context,
                })

    return {"models": models}


@router.post("/embeddings")
async def ollama_embeddings(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
    auth: Tuple[User, ApiKey] = Depends(authenticate_request),
):
    """
    Ollama-compatible /api/embeddings endpoint.
    """
    user, api_key = auth

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON body",
        )

    request_id = f"ollama-emb-{uuid.uuid4().hex[:16]}"
    bind_request_context(request_id=request_id, user_id=user.id)

    try:
        canonical = OllamaInTranslator.translate_embedding_request(body)
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
            detail=f"model '{canonical.model}' not found",
        )

    service = InferenceService(db)
    response = await service.ollama_embedding(canonical, user, api_key, request)
    return response
