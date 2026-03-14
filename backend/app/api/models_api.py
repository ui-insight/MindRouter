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
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.auth import authenticate_request
from backend.app.core.canonical_schemas import CanonicalModelInfo, CanonicalModelList
from backend.app.core.telemetry.registry import get_registry
from backend.app.db.models import ApiKey, User
from backend.app.db.session import get_async_db

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

    registry = get_registry()
    backends = await registry.get_healthy_backends()

    # Collect models with their capabilities and backends
    model_data: dict = {}

    for backend in backends:
        backend_models = await registry.get_backend_models(backend.id)

        for model in backend_models:
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
            )
        )

    return CanonicalModelList(data=models)
