############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# search/registry.py: Provider registry and convenience helpers
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Provider registry and convenience helpers for web search."""

from __future__ import annotations

import json as _json
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.logging_config import get_logger
from backend.app.services.search.base import SearchProvider, SearchResult
from backend.app.services.search.brave import BraveSearchProvider
from backend.app.services.search.searxng import SearXNGSearchProvider

logger = get_logger(__name__)

# All known providers — register new ones here
PROVIDERS: dict[str, SearchProvider] = {
    "brave": BraveSearchProvider(),
    "searxng": SearXNGSearchProvider(),
}

# Config keys and their default values
_CONFIG_DEFAULTS: dict[str, Any] = {
    "search.enabled": True,
    "search.provider": "brave",
    "search.max_results": 10,
    "search.quota_tokens_per_request": 50,
    "search.brave.api_key": "",
    "search.brave.endpoint": "https://api.search.brave.com/res/v1/web/search",
    "search.searxng.endpoint": "",
}


async def get_search_config(db: AsyncSession) -> dict[str, Any]:
    """Load all search.* config keys from AppConfig."""
    from backend.app.db.models import AppConfig

    result = await db.execute(
        select(AppConfig.key, AppConfig.value).where(
            AppConfig.key.like("search.%")
        )
    )
    rows = {r.key: r.value for r in result.all()}

    config: dict[str, Any] = {}
    for key, default in _CONFIG_DEFAULTS.items():
        raw = rows.get(key)
        if raw is not None:
            try:
                config[key] = _json.loads(raw)
            except (ValueError, TypeError):
                config[key] = raw
        else:
            config[key] = default

    # Also pull the legacy settings.brave_search_api_key as fallback
    if not config.get("search.brave.api_key"):
        from backend.app.settings import get_settings
        settings = get_settings()
        if settings.brave_search_api_key:
            config["search.brave.api_key"] = settings.brave_search_api_key

    return config


async def save_search_config(
    db: AsyncSession, updates: dict[str, Any]
) -> None:
    """Persist search config values to AppConfig."""
    from backend.app.db import crud

    for key, value in updates.items():
        if key in _CONFIG_DEFAULTS:
            await crud.set_config(
                db, key, value,
                description=f"Web search: {key}",
            )
    await db.flush()


def get_provider(name: str) -> SearchProvider | None:
    """Get a provider instance by key name."""
    return PROVIDERS.get(name)


def list_providers() -> list[dict]:
    """List all registered providers with their metadata."""
    return [
        {
            "key": p.provider_key,
            "name": p.display_name,
            "config_keys": p.config_keys,
        }
        for p in PROVIDERS.values()
    ]


async def get_search_provider(db: AsyncSession) -> SearchProvider:
    """Get the currently configured search provider.

    Raises ValueError if search is disabled or provider is unknown.
    """
    config = await get_search_config(db)

    if not config.get("search.enabled", True):
        raise ValueError("Web search is disabled")

    provider_key = config.get("search.provider", "brave")
    provider = PROVIDERS.get(provider_key)
    if not provider:
        raise ValueError(f"Unknown search provider: {provider_key}")

    return provider


async def search(
    db: AsyncSession,
    query: str,
    *,
    max_results: int | None = None,
) -> list[SearchResult]:
    """Convenience wrapper: load config, get provider, execute search."""
    config = await get_search_config(db)

    if not config.get("search.enabled", True):
        raise ValueError("Web search is disabled")

    provider_key = config.get("search.provider", "brave")
    provider = PROVIDERS.get(provider_key)
    if not provider:
        raise ValueError(f"Unknown search provider: {provider_key}")

    if max_results is None:
        max_results = int(config.get("search.max_results", 10))

    return await provider.search(query, max_results=max_results, config=config)
