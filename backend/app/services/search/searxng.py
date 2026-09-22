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
from backend.app.services.search.base import (
    SearchExchange,
    SearchProvider,
    SearchResult,
    attach_exchange,
    response_meta,
)

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
        extra_snippets: bool = False,
    ) -> list[SearchResult]:
        """Parsed results only — see search_exchange for the full round-trip."""
        exchange = await self.search_exchange(
            query, max_results=max_results, config=config
        )
        return exchange.results

    async def search_exchange(
        self,
        query: str,
        *,
        max_results: int = 5,
        config: dict | None = None,
        extra_snippets: bool = False,  # SearXNG has no such feature; accepted for the shared signature
    ) -> SearchExchange:
        config = config or {}
        endpoint = config.get("search.searxng.endpoint", "")

        # SearXNG JSON API: GET /search?q=...&format=json
        url = endpoint.rstrip("/") + "/search" if endpoint else ""
        params = {"q": query, "format": "json", "pageno": 1}
        exchange = SearchExchange(request_url=url or None, request_params=dict(params))

        if not endpoint:
            err = ValueError("SearXNG endpoint URL is not configured")
            attach_exchange(err, exchange)
            raise err

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url, params=params)
                # Body first: a SearXNG 403/429 explains itself in the body.
                exchange.http_status = resp.status_code
                exchange.response_body = resp.text
                exchange.response_meta = response_meta(resp)
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            attach_exchange(e, exchange)
            raise

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
        exchange.results = results
        meta = exchange.response_meta or {}
        # Which engines actually answered explains a thin or odd result set.
        meta["engines"] = sorted({
            str(i.get("engine", "")) for i in (data.get("results") or []) if i.get("engine")
        })
        meta["total_results_reported"] = len(data.get("results") or [])
        exchange.response_meta = meta
        return exchange

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
