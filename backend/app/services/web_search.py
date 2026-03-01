############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# web_search.py: Brave Search API integration for web search
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Brave Web Search integration for injecting live web results into chat context."""

import httpx

from backend.app.logging_config import get_logger
from backend.app.settings import get_settings

logger = get_logger(__name__)

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"


async def brave_web_search(query: str, num_results: int = 5) -> list[dict]:
    """Search the web via Brave Search API.

    Returns a list of dicts with keys: title, url, description.
    Returns empty list on any failure (missing key, timeout, API error).
    """
    settings = get_settings()
    api_key = settings.brave_search_api_key
    if not api_key:
        return []

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                BRAVE_SEARCH_URL,
                params={"q": query, "count": num_results},
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": api_key,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        results = []
        for item in data.get("web", {}).get("results", [])[:num_results]:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "description": item.get("description", ""),
            })
        return results

    except Exception:
        logger.warning("brave_web_search failed", exc_info=True)
        return []


def format_search_results(results: list[dict]) -> str:
    """Format search results into a context block for system prompt injection."""
    if not results:
        return ""

    lines = ["[Web Search Results]"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}")
        lines.append(f"   URL: {r['url']}")
        if r["description"]:
            lines.append(f"   {r['description']}")
    lines.append(
        "\nUse the above web search results to inform your answer. "
        "Cite sources with URLs when relevant. If the search results are not "
        "relevant to the user's question, you may ignore them."
    )
    return "\n".join(lines)
