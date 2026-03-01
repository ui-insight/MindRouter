############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# __init__.py: API translation layer package exports
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""API translation layer for MindRouter2.

Provides bidirectional translation between:
- OpenAI API format
- Ollama API format
- Anthropic Messages API format
- vLLM API format
- Internal canonical format
"""

from backend.app.core.translators.openai_in import OpenAIInTranslator
from backend.app.core.translators.ollama_in import OllamaInTranslator
from backend.app.core.translators.ollama_out import OllamaOutTranslator
from backend.app.core.translators.vllm_out import VLLMOutTranslator
from backend.app.core.translators.anthropic_in import AnthropicInTranslator

__all__ = [
    "OpenAIInTranslator",
    "OllamaInTranslator",
    "OllamaOutTranslator",
    "VLLMOutTranslator",
    "AnthropicInTranslator",
]
