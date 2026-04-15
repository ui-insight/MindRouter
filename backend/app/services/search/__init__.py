############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# search/__init__.py: Pluggable web search provider registry
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Pluggable web search provider registry.

Usage::

    from backend.app.services.search import get_search_provider, search

    # Direct use
    provider = await get_search_provider(db)
    results = await provider.search("query", max_results=5)

    # Or the convenience wrapper
    results = await search(db, "query", max_results=5)
"""

from backend.app.services.search.base import SearchProvider, SearchResult
from backend.app.services.search.registry import get_search_provider, search

__all__ = [
    "SearchProvider",
    "SearchResult",
    "get_search_provider",
    "search",
]
