############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# search/searxng.py: SearXNG search provider
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""SearXNG search provider implementation.

SearXNG is a self-hosted meta-search engine that aggregates results from
multiple upstream engines.  It requires no API key — just a running instance.
"""

from __future__ import annotations

import httpx

from backend.app.logging_config import get_logger
from backend.app.services.search.base import SearchProvider, SearchResult

logger = get_logger(__name__)


class SearXNGSearchProvider(SearchProvider):
    """Search via a SearXNG instance."""

    display_name = "SearXNG"
    provider_key = "searxng"
    config_keys = [
        "search.searxng.endpoint",
    ]

    async def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        config: dict | None = None,
    ) -> list[SearchResult]:
        config = config or {}
        endpoint = config.get("search.searxng.endpoint", "")

        if not endpoint:
            raise ValueError("SearXNG endpoint URL is not configured")

        # SearXNG JSON API: GET /search?q=...&format=json
        url = endpoint.rstrip("/") + "/search"

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                url,
                params={
                    "q": query,
                    "format": "json",
                    "pageno": 1,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        results: list[SearchResult] = []
        for item in data.get("results", [])[:max_results]:
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", ""),
                    published=item.get("publishedDate", None),
                    extra={"engine": item.get("engine", "")},
                )
            )
        return results

    async def health_check(self, config: dict | None = None) -> tuple[bool, str]:
        config = config or {}
        endpoint = config.get("search.searxng.endpoint", "")
        if not endpoint:
            return False, "Endpoint URL not configured"

        try:
            url = endpoint.rstrip("/") + "/search"
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    url,
                    params={"q": "test", "format": "json", "pageno": 1},
                )
                resp.raise_for_status()
            return True, "OK"
        except httpx.HTTPStatusError as e:
            return False, f"HTTP {e.response.status_code}"
        except Exception as e:
            return False, str(e)
