############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# __init__.py: API endpoints package and router configuration
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""API endpoints for MindRouter2."""

from fastapi import APIRouter

from backend.app.api.v1_openai import router as openai_router
from backend.app.api.ollama_api import router as ollama_router
from backend.app.api.anthropic_api import router as anthropic_router
from backend.app.api.admin_api import router as admin_router
from backend.app.api.models_api import router as models_router
from backend.app.api.health import router as health_router
from backend.app.api.telemetry_api import router as telemetry_router

# Create main API router
api_router = APIRouter()

# Include all sub-routers
api_router.include_router(health_router)
api_router.include_router(openai_router)
api_router.include_router(ollama_router)
api_router.include_router(anthropic_router)
api_router.include_router(models_router)
api_router.include_router(admin_router, prefix="/api/admin", tags=["admin"])
api_router.include_router(telemetry_router, prefix="/api/admin/telemetry", tags=["telemetry"])

__all__ = ["api_router"]
