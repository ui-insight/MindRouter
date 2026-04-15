############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# search/base.py: Abstract base class for search providers
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Abstract base class and data types for web search providers."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SearchResult:
    """A single search result."""

    title: str
    url: str
    snippet: str
    published: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
        }
        if self.published:
            d["published"] = self.published
        if self.extra:
            d["extra"] = self.extra
        return d


class SearchProvider(abc.ABC):
    """Interface that every search provider must implement."""

    # Human-readable name shown in the admin UI
    display_name: str = "Unknown"

    # Machine key used in config (e.g. "brave", "searxng")
    provider_key: str = "unknown"

    # List of config keys this provider needs (shown in admin UI)
    config_keys: list[str] = []

    @abc.abstractmethod
    async def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        config: dict | None = None,
    ) -> list[SearchResult]:
        """Execute a search and return results.

        Args:
            query: The search query string.
            max_results: Maximum number of results to return.
            config: Provider-specific config values from AppConfig.

        Returns:
            List of SearchResult objects.
        """
        ...

    @abc.abstractmethod
    async def health_check(self, config: dict | None = None) -> tuple[bool, str]:
        """Check if the provider is configured and reachable.

        Returns:
            (healthy, message) — healthy is True if the provider can serve
            requests, message is a human-readable status string.
        """
        ...
