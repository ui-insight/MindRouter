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
import inspect
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


@dataclass
class SearchExchange:
    """Everything observed about ONE provider round-trip.

    ``search()`` returns only the parsed results, which is all a caller needs
    but nothing an auditor can use: no provider URL, no HTTP status, no
    verbatim body. Providers therefore implement ``search_exchange()`` — the
    single real implementation — and ``search()`` is the thin wrapper over it.

    Every field except ``results`` is best-effort: a provider that cannot
    report it leaves it None, and the audit row simply records less.
    """

    results: list = field(default_factory=list)
    request_url: Optional[str] = None
    # What was sent. Credentials are redacted by the audit layer, not here —
    # a provider must not have to remember which of its own params are secret.
    request_params: Optional[dict] = None
    request_headers: Optional[dict] = None
    http_status: Optional[int] = None
    # The provider's verbatim response text, uncapped at this layer.
    response_body: Optional[str] = None
    # Response headers worth keeping (rate limits, request ids) and any
    # provider-level metadata parsed out of the body.
    response_meta: Optional[dict] = None


# Response headers worth keeping on an audit row: rate-limit accounting and
# the provider's own request id, which is what a support ticket will ask for.
# Everything else is noise or a credential echo.
KEEP_RESPONSE_HEADERS = (
    "x-ratelimit-limit",
    "x-ratelimit-remaining",
    "x-ratelimit-reset",
    "x-request-id",
    "retry-after",
    "content-type",
)


def response_meta(resp) -> dict:
    """Subset of an httpx response's headers, preserved verbatim for the log."""
    try:
        headers = {
            k: v for k, v in resp.headers.items()
            if k.lower() in KEEP_RESPONSE_HEADERS
        }
    except Exception:
        headers = {}
    return {"response_headers": headers}


_EXCHANGE_ATTR = "_mindrouter_search_exchange"


def attach_exchange(exc: BaseException, exchange: "SearchExchange") -> None:
    """Carry a partial round-trip out on the exception that aborted it.

    The audit log needs the HTTP status and body of a FAILED call, but giving
    providers a new exception type would ripple through every caller's except
    clauses (httpx.HTTPStatusError, httpx.TimeoutException, ValueError are all
    handled today). Riding along on the original exception keeps the provider
    error contract byte-identical while still giving the log everything.
    """
    try:
        setattr(exc, _EXCHANGE_ATTR, exchange)
    except Exception:  # pragma: no cover — exception types using __slots__
        pass


def exchange_from_exception(exc: BaseException) -> Optional["SearchExchange"]:
    """The partial exchange attached by attach_exchange, if any."""
    return getattr(exc, _EXCHANGE_ATTR, None)


def accepts_kwarg(fn, name: str) -> bool:
    """Whether ``fn(..., name=...)`` can be called at all.

    A provider written before a keyword existed — the pre-audit-log case the
    default ``search_exchange`` exists for, or a third-party one — must keep
    working when a caller asks for a feature it never heard of: the feature
    is simply not honoured. Forwarding the keyword regardless turns "ignored"
    into TypeError at every call site.
    """
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    if name in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


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
        extra_snippets: bool = False,
    ) -> list[SearchResult]:
        """Execute a search and return results.

        Args:
            query: The search query string.
            max_results: Maximum number of results to return.
            config: Provider-specific config values from AppConfig.
            extra_snippets: Ask for additional excerpts per result where the
                provider offers them (Brave); others ignore it. They land in
                ``SearchResult.extra["extra_snippets"]``.

        Returns:
            List of SearchResult objects.
        """
        ...

    async def search_exchange(
        self,
        query: str,
        *,
        max_results: int = 5,
        config: dict | None = None,
        extra_snippets: bool = False,
    ) -> "SearchExchange":
        """Execute a search and report the full round-trip.

        Deliberately NOT abstract: a provider written before the audit log
        existed (or a third-party one) keeps working and simply contributes an
        audit row with no HTTP detail. The two first-party providers override
        this with the real implementation and delegate ``search()`` to it.
        """
        kwargs: dict = {"max_results": max_results, "config": config}
        # Only a provider whose search() knows the keyword receives it; an
        # older one is called exactly as before and the request is ignored.
        if extra_snippets and accepts_kwarg(self.search, "extra_snippets"):
            kwargs["extra_snippets"] = True
        results = await self.search(query, **kwargs)
        return SearchExchange(results=results)

    @abc.abstractmethod
    async def health_check(self, config: dict | None = None) -> tuple[bool, str]:
        """Check if the provider is configured and reachable.

        Returns:
            (healthy, message) — healthy is True if the provider can serve
            requests, message is a human-readable status string.
        """
        ...
