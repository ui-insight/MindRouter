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
from backend.app.services.search.base import (
    SearchExchange,
    SearchProvider,
    SearchResult,
    attach_exchange,
    response_meta,
)

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
        extra_snippets: bool = False,
    ) -> list[SearchResult]:
        """Parsed results only — see search_exchange for the full round-trip."""
        exchange = await self.search_exchange(
            query, max_results=max_results, config=config, extra_snippets=extra_snippets
        )
        return exchange.results

    async def search_exchange(
        self,
        query: str,
        *,
        max_results: int = 5,
        config: dict | None = None,
        extra_snippets: bool = False,
    ) -> SearchExchange:
        config = config or {}
        api_key = config.get("search.brave.api_key", "")
        endpoint = config.get("search.brave.endpoint", BRAVE_SEARCH_URL)

        params: dict = {"q": query, "count": max_results}
        if extra_snippets:
            # Up to five further excerpts per result; Brave serves them only
            # on plans that include the feature and otherwise leaves the
            # field out, so the request is harmless elsewhere.
            params["extra_snippets"] = "true"
        # The subscription token is deliberately included: the audit layer
        # redacts it centrally, so no provider has to remember which of its
        # own headers are secret.
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": api_key,
        }
        exchange = SearchExchange(
            request_url=endpoint, request_params=dict(params), request_headers=dict(headers)
        )

        if not api_key:
            err = ValueError("Brave Search API key is not configured")
            attach_exchange(err, exchange)
            raise err

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(endpoint, params=params, headers=headers)
                # Read the body BEFORE raise_for_status: on a 4xx/5xx the body
                # is the only thing that says WHY (quota exhausted, bad key),
                # and that is exactly what an auditor opens the row to see.
                exchange.http_status = resp.status_code
                exchange.response_body = resp.text
                exchange.response_meta = response_meta(resp)
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            attach_exchange(e, exchange)
            raise

        results: list[SearchResult] = []
        for item in data.get("web", {}).get("results", [])[:max_results]:
            extra: dict = {}
            more = item.get("extra_snippets")
            if isinstance(more, list) and more:
                extra["extra_snippets"] = [str(x) for x in more if isinstance(x, str) and x.strip()][:5]
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("description", ""),
                    published=item.get("page_age", None),
                    extra=extra,
                )
            )
        exchange.results = results
        # Brave echoes the interpreted query and flags altered/spellcheck —
        # worth keeping, since it explains results that do not match the input.
        meta = exchange.response_meta or {}
        if isinstance(data.get("query"), dict):
            meta["provider_query"] = data["query"]
        meta["total_results_reported"] = len(data.get("web", {}).get("results", []) or [])
        exchange.response_meta = meta
        return exchange

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
