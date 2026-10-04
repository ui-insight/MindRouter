############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# models_api.py: Models listing API endpoints
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Models listing API endpoint."""

import time
from typing import List, Tuple

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.auth import authenticate_request
from backend.app.core.canonical_schemas import CanonicalModelInfo, CanonicalModelList
from backend.app.core.telemetry.registry import get_registry
from backend.app.db.models import MODELLESS_ENGINES, ApiKey, BackendEngine, Modality, User
from backend.app.db.session import get_async_db

# Modalities published in the general model catalogs (/v1/models, /api/tags,
# /anthropic/v1/models). These are the text-in/text-out families an OpenAI
# client can actually send to a chat, completion, embedding or rerank endpoint.
#
# Image, video and speech models are deliberately excluded: advertising them
# here presents them as LLMs, and a client that picks one for
# /v1/chat/completions gets a confusing failure. Each has its own discovery
# endpoint — /v1/images/models, /videos/models, /v1/audio/voices.
CATALOG_MODALITIES = frozenset({
    Modality.CHAT,
    Modality.COMPLETION,
    Modality.MULTIMODAL,
    Modality.EMBEDDING,
    Modality.RERANKING,
})


def is_catalog_model(model) -> bool:
    """True when a model belongs in the general LLM catalog.

    Unknown/NULL modality is treated as catalogable so a discovery gap can
    never silently hide a working chat model.
    """
    modality = getattr(model, "modality", None)
    if modality is None:
        return True
    return modality in CATALOG_MODALITIES


# Modalities a user can hold a conversation with. Narrower than
# CATALOG_MODALITIES: embedding and reranking models belong in the API
# catalogs but cannot chat, so pickers that exist to start a conversation
# (the chat UI, the admin core-model config) use this set instead.
CHAT_MODALITIES = frozenset({
    Modality.CHAT,
    Modality.COMPLETION,
    Modality.MULTIMODAL,
})


def is_chat_model(model) -> bool:
    """True when a model can serve a chat conversation.

    Fails open on unknown/NULL modality, same as is_catalog_model: a
    discovery gap must never hide a working chat model from the picker.
    """
    modality = getattr(model, "modality", None)
    if modality is None:
        return True
    return modality in CHAT_MODALITIES

router = APIRouter(tags=["models"])


@router.get("/v1/models")
async def list_models(
    request: Request,
    db: AsyncSession = Depends(get_async_db),
    auth: Tuple[User, ApiKey] = Depends(authenticate_request),
) -> CanonicalModelList:
    """
    List available models (OpenAI-compatible).

    Returns all models available across healthy backends.
    """
    user, api_key = auth

    # TypeSafe's SDK (System One clients pointed at MindRouter) lists models in
    # its own shape and identifies itself with this header; everyone else gets
    # the OpenAI list below. See services/decisions/systemone.py.
    if request.headers.get("x-typesafe-sdk"):
        from backend.app.services.decisions import get_decisions_config
        from backend.app.services.decisions.systemone import typesafe_model_list

        return JSONResponse(typesafe_model_list(await get_decisions_config(db)))

    registry = get_registry()
    backends = await registry.get_healthy_backends()

    # Collect models with their capabilities and backends
    model_data: dict = {}

    for backend in backends:
        # Model-less engines (DLP) serve no inference and never belong in the
        # catalog, even if a stale model row lingered from an engine change.
        if backend.engine in MODELLESS_ENGINES:
            continue
        backend_models = await registry.get_backend_models(backend.id)

        for model in backend_models:
            if not is_catalog_model(model):
                continue
            if model.name not in model_data:
                model_data[model.name] = {
                    "backends": [],
                    "capabilities": {
                        "multimodal": False,
                        "embeddings": False,
                        "structured_output": True,
                        "thinking": False,
                        "tools": False,
                    },
                    "created": int(model.created_at.timestamp()) if model.created_at else int(time.time()),
                    "context_length": None,
                    "model_max_context": None,
                    "parameter_count": None,
                    "quantization": None,
                    "family": None,
                }

            model_data[model.name]["backends"].append(backend.name)

            # Update capabilities
            if model.supports_multimodal:
                model_data[model.name]["capabilities"]["multimodal"] = True
            if model.supports_thinking:
                model_data[model.name]["capabilities"]["thinking"] = True
            if model.supports_tools:
                model_data[model.name]["capabilities"]["tools"] = True
            if "embed" in model.name.lower():
                model_data[model.name]["capabilities"]["embeddings"] = True

            # Use max context_length across backends
            if model.context_length is not None:
                cur = model_data[model.name]["context_length"]
                if cur is None or model.context_length > cur:
                    model_data[model.name]["context_length"] = model.context_length

            if model.model_max_context is not None:
                cur = model_data[model.name]["model_max_context"]
                if cur is None or model.model_max_context > cur:
                    model_data[model.name]["model_max_context"] = model.model_max_context

            # Take first non-None value for these fields
            if model.parameter_count and not model_data[model.name]["parameter_count"]:
                model_data[model.name]["parameter_count"] = model.parameter_count
            if model.quantization and not model_data[model.name]["quantization"]:
                model_data[model.name]["quantization"] = model.quantization
            if model.family and not model_data[model.name]["family"]:
                model_data[model.name]["family"] = model.family

    # Reasoning controls per model (switch + levels), from the family table
    # in core/reasoning.py, so clients can pick a level the model accepts.
    from backend.app.core.reasoning import profile_for

    for name, data in model_data.items():
        data["reasoning"] = profile_for(
            name, family=data["family"], supports_thinking=data["capabilities"]["thinking"]
        ).describe()

    # Append model aliases (inherit target model's metadata)
    alias_map = registry.get_alias_cache()
    for alias_name, target_model in alias_map.items():
        if target_model in model_data:
            target = model_data[target_model]
            model_data[alias_name] = {
                **target,
                "is_alias": True,
                "alias_target": target_model,
            }

    # Build response
    models: List[CanonicalModelInfo] = []
    for name, data in sorted(model_data.items()):
        models.append(
            CanonicalModelInfo(
                id=name,
                created=data["created"],
                owned_by="mindrouter",
                capabilities=data["capabilities"],
                backends=data["backends"],
                context_length=data["context_length"],
                model_max_context=data["model_max_context"],
                parameter_count=data["parameter_count"],
                quantization=data["quantization"],
                family=data["family"],
                reasoning=data.get("reasoning"),
                is_alias=data.get("is_alias"),
                alias_target=data.get("alias_target"),
            )
        )

    return CanonicalModelList(data=models)
