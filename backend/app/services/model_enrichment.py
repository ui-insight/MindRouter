############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# model_enrichment.py: Automatic model description generation
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Automatic model description enrichment using web search + LLM summarization."""

from typing import Optional

import httpx

from backend.app.logging_config import get_logger
from backend.app.services.web_search import brave_web_search, format_search_results

logger = get_logger(__name__)

_ENRICH_SYSTEM_PROMPT = """\
You are a technical writer creating concise model descriptions for an LLM catalog.
Given information about a model (name, metadata, web search results), produce a
description with a brief narrative paragraph followed by a structured bullet list.

Format rules:
- Start with a 2-3 sentence narrative paragraph summarizing what the model is,
  its architecture, and what it is designed for.
- Do NOT include any URLs or hyperlinks in the description text.
- Follow the paragraph with a blank line, then bullet points (- ) for key specs.
- Bold key labels with **bold** (e.g. - **Architecture:** ...)
- Cover: architecture/family, parameter count, context window, quantization,
  capabilities, and recommended use cases
- Keep it to 4-8 bullet points, concise and factual
- Do NOT include the model name as a heading — it is shown separately
- If information is unavailable, omit that bullet rather than guessing
- Do NOT include any citation references, footnote markers, or source annotations
  such as 【1†URL】, [1], (source), etc. Write clean prose only.
- Do NOT include any URLs, links, or web addresses anywhere in the output.
- Do NOT include any preamble like "Here is..." or closing remarks
"""


async def _call_mindrouter_llm(
    prompt: str,
    system_prompt: str,
    model: str,
    api_key: str,
    port: int = 8000,
) -> Optional[str]:
    """Call MindRouter's own /v1/chat/completions endpoint for enrichment.

    Returns the response content string, or None on failure.
    """
    url = f"http://localhost:{port}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 1500,
        "think": False,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
    except Exception:
        logger.warning("model enrichment LLM call failed", exc_info=True)
        return None


async def enrich_model_description(
    model_name: str,
    model_metadata: dict,
    enrich_model: str,
    api_key: str,
    brave_api_key: Optional[str] = None,
    port: int = 8000,
) -> Optional[str]:
    """Generate a markdown description for a model via web search + LLM.

    Args:
        model_name: Full model name (e.g. "qwen3:32b-fp16")
        model_metadata: Dict with keys like family, parameter_count, quantization, etc.
        enrich_model: Model name to use for the LLM call
        api_key: API key for MindRouter inference
        brave_api_key: Optional Brave Search API key override
        port: Port for localhost MindRouter API

    Returns:
        Markdown description string, or None on failure.
    """
    # Strip tag suffixes for a cleaner search query
    base_name = model_name.split(":")[0] if ":" in model_name else model_name

    # Step 1: Web search for model card info
    search_query = f"{base_name} LLM model card"
    search_results = await brave_web_search(
        search_query, num_results=5, api_key=brave_api_key
    )
    search_context = format_search_results(search_results) if search_results else ""

    # Step 2: Build prompt with metadata + search context
    meta_lines = []
    if model_metadata.get("family"):
        meta_lines.append(f"- Family: {model_metadata['family']}")
    if model_metadata.get("parameter_count"):
        meta_lines.append(f"- Parameters: {model_metadata['parameter_count']}")
    if model_metadata.get("quantization"):
        meta_lines.append(f"- Quantization: {model_metadata['quantization']}")
    if model_metadata.get("context_length"):
        meta_lines.append(f"- Context length: {model_metadata['context_length']}")
    if model_metadata.get("capabilities"):
        meta_lines.append(f"- Capabilities: {model_metadata['capabilities']}")
    if model_metadata.get("supports_thinking"):
        meta_lines.append("- Supports thinking/reasoning mode")
    if model_metadata.get("supports_multimodal"):
        meta_lines.append("- Supports multimodal (vision) input")

    metadata_block = "\n".join(meta_lines) if meta_lines else "No metadata available."

    user_prompt = f"""Write a concise description for this LLM model:

Model name: {model_name}

Known metadata:
{metadata_block}

{search_context}"""

    # Step 3: Call LLM
    result = await _call_mindrouter_llm(
        prompt=user_prompt,
        system_prompt=_ENRICH_SYSTEM_PROMPT,
        model=enrich_model,
        api_key=api_key,
        port=port,
    )

    if result and result.strip():
        import re
        # Strip citation references like 【1†URL】, 【metadata】, [1], etc.
        cleaned = re.sub(r'【[^】]*】', '', result)
        cleaned = re.sub(r'\[\d+†?[^\]]*\]', '', cleaned)
        # Strip any URLs that slipped through
        cleaned = re.sub(r'https?://\S+', '', cleaned)
        # Clean up artifacts: "Learn more at " with nothing after, trailing punctuation
        cleaned = re.sub(r'[Ll]earn more at\s*\.?\s*', '', cleaned)
        cleaned = re.sub(r'[Ss]ee\s*\.?\s*$', '', cleaned, flags=re.MULTILINE)
        # Collapse any resulting double spaces
        cleaned = re.sub(r'  +', ' ', cleaned)
        return cleaned.strip()
    return None
