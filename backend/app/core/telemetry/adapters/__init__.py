############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# __init__.py: Backend telemetry adapters package
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Backend adapters for telemetry collection."""

from backend.app.core.telemetry.adapters.ollama import OllamaAdapter
from backend.app.core.telemetry.adapters.vllm import VLLMAdapter

__all__ = ["OllamaAdapter", "VLLMAdapter"]
