############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# search/brave.py: Brave Search API provider
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Brave Search API provider implementation."""

from __future__ import annotations

import httpx

from backend.app.logging_config import get_logger
from backend.app.services.search.base import SearchProvider, SearchResult

logger = get_logger(__name__)

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"


class BraveSearchProvider(SearchProvider):
    """Search via the Brave Search API."""

    display_name = "Brave Search"
    provider_key = "brave"
    config_keys = [
        "search.brave.api_key",
        "search.brave.endpoint",
    ]

    async def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        config: dict | None = None,
    ) -> list[SearchResult]:
        config = config or {}
        api_key = config.get("search.brave.api_key", "")
        endpoint = config.get("search.brave.endpoint", BRAVE_SEARCH_URL)

        if not api_key:
            raise ValueError("Brave Search API key is not configured")

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                endpoint,
                params={"q": query, "count": max_results},
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": api_key,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        results: list[SearchResult] = []
        for item in data.get("web", {}).get("results", [])[:max_results]:
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("description", ""),
                    published=item.get("page_age", None),
                )
            )
        return results

    async def health_check(self, config: dict | None = None) -> tuple[bool, str]:
        config = config or {}
        api_key = config.get("search.brave.api_key", "")
        if not api_key:
            return False, "API key not configured"

        endpoint = config.get("search.brave.endpoint", BRAVE_SEARCH_URL)
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    endpoint,
                    params={"q": "test", "count": 1},
                    headers={
                        "Accept": "application/json",
                        "X-Subscription-Token": api_key,
                    },
                )
                resp.raise_for_status()
            return True, "OK"
        except httpx.HTTPStatusError as e:
            return False, f"HTTP {e.response.status_code}"
        except Exception as e:
            return False, str(e)
