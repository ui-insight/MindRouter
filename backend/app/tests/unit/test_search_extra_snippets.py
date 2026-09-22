"""/v1/search `extra_snippets`: Brave asks for them only when told to, keeps
up to five per result, the API item carries them — and a provider that never
heard of the flag keeps working.

VandalChat Deep Research needs more than one description for sources whose
pages refuse to be read (paywalls, bot walls); Brave offers up to five extra
excerpts per result on plans that include them.

The compatibility half matters as much as the feature: the default
``search_exchange`` exists precisely so a provider written before the audit
log (or a third-party one) keeps working, and ``run_logged_search`` is the
one door every surface uses. Forwarding a keyword such a provider's signature
lacks turns "ignored" into TypeError — which is what the first cut of this
change did, failing three existing tests. The keyword is therefore forwarded
only when it is requested AND the callee accepts it.

Module-level imports mirror test_web_search_audit.py (same package, same
env expectations); search_api is imported lazily because it pulls the auth
and db chains.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.app.services.search import audit as A
from backend.app.services.search import brave as brave_mod
from backend.app.services.search import searxng as searxng_mod
from backend.app.services.search.base import (
    SearchExchange,
    SearchProvider,
    SearchResult,
    accepts_kwarg,
)


# ------------------------------------------------------------------
# httpx stand-in: the providers build their own AsyncClient, so the
# class is swapped rather than a transport injected.
# ------------------------------------------------------------------


class _Resp:
    def __init__(self, payload):
        self.status_code = 200
        self.text = json.dumps(payload)
        self.headers = {"content-type": "application/json"}
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _fake_client(payload, seen):
    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, params=None, headers=None):
            seen.append(dict(params or {}))
            # Like Brave: extra_snippets come back only when the request
            # asked for them. The parser keeps whatever the response carries,
            # so a fake that always includes them would test the fake.
            if (params or {}).get("extra_snippets") == "true":
                return _Resp(payload)
            stripped = json.loads(json.dumps(payload))
            for item in stripped.get("web", {}).get("results", []):
                item.pop("extra_snippets", None)
            return _Resp(stripped)

    return _Client


BRAVE_PAYLOAD = {"web": {"results": [
    {"title": "T", "url": "https://example.com/a", "description": "d", "page_age": "2025-01-01",
     # six strings, a non-string and an empty one: keep five, drop the junk
     "extra_snippets": ["one", "two", "three", "four", "five", "six", 7, ""]},
    {"title": "U", "url": "https://example.com/b", "description": "e"},
]}}

SEARXNG_PAYLOAD = {"results": [{"title": "T", "url": "https://x", "content": "c"}]}


@pytest.fixture
def brave(monkeypatch):
    seen = []
    monkeypatch.setattr(brave_mod.httpx, "AsyncClient", _fake_client(BRAVE_PAYLOAD, seen))
    return brave_mod.BraveSearchProvider(), {"search.brave.api_key": "k"}, seen


@pytest.fixture
def searxng(monkeypatch):
    seen = []
    monkeypatch.setattr(searxng_mod.httpx, "AsyncClient", _fake_client(SEARXNG_PAYLOAD, seen))
    return searxng_mod.SearXNGSearchProvider(), {"search.searxng.endpoint": "https://sx/"}, seen


# ------------------------------------------------------------------
# Brave: the feature itself.
# ------------------------------------------------------------------


class TestBrave:
    def test_not_requested_means_the_call_is_unchanged(self, brave):
        provider, cfg, seen = brave
        out = asyncio.run(provider.search("q", max_results=5, config=cfg))
        assert "extra_snippets" not in seen[0]
        assert out[0].extra == {} and out[1].extra == {}

    def test_requested_means_asked_for_and_parsed_capped_at_five(self, brave):
        provider, cfg, seen = brave
        out = asyncio.run(provider.search("q", max_results=5, config=cfg, extra_snippets=True))
        assert seen[0]["extra_snippets"] == "true"
        assert out[0].extra["extra_snippets"] == ["one", "two", "three", "four", "five"]
        # a result Brave gave no extras for carries nothing, not an empty key
        assert out[1].extra == {}

    def test_the_flag_is_on_the_audit_exchange(self, brave):
        """What was sent is what the audit row records, flag included."""
        provider, cfg, _ = brave
        ex = asyncio.run(provider.search_exchange("q", config=cfg, extra_snippets=True))
        assert ex.request_params["extra_snippets"] == "true"
        ex = asyncio.run(provider.search_exchange("q", config=cfg))
        assert "extra_snippets" not in ex.request_params


class TestSearXNG:
    def test_accepts_and_ignores_the_flag(self, searxng):
        provider, cfg, seen = searxng
        out = asyncio.run(provider.search("q", config=cfg, extra_snippets=True))
        assert "extra_snippets" not in seen[0]
        assert "extra_snippets" not in out[0].extra  # (`engine` has always been there)


# ------------------------------------------------------------------
# Compatibility: providers that predate the keyword.
# ------------------------------------------------------------------


class _Legacy(SearchProvider):
    """A provider written against the interface before extra_snippets existed."""

    provider_key = "legacy"

    def __init__(self):
        self.calls = []

    async def search(self, query, *, max_results=5, config=None):
        self.calls.append({"max_results": max_results, "config": config})
        return [SearchResult(title="t", url="u", snippet="s")]

    async def health_check(self, config=None):
        return True, "OK"


class _Modern(SearchProvider):
    provider_key = "modern"

    def __init__(self):
        self.calls = []

    async def search(self, query, *, max_results=5, config=None, extra_snippets=False):
        self.calls.append({"extra_snippets": extra_snippets})
        return []

    async def health_check(self, config=None):
        return True, "OK"


class TestCompatibility:
    def test_accepts_kwarg(self):
        async def old(q, *, max_results=5, config=None):
            ...

        async def new(q, *, max_results=5, config=None, extra_snippets=False):
            ...

        async def splat(q, **kw):
            ...

        assert accepts_kwarg(old, "extra_snippets") is False
        assert accepts_kwarg(new, "extra_snippets") is True
        assert accepts_kwarg(splat, "extra_snippets") is True
        assert accepts_kwarg(object(), "extra_snippets") is False

    def test_default_exchange_never_forwards_to_a_provider_that_cannot_take_it(self):
        p = _Legacy()
        ex = asyncio.run(p.search_exchange("q", extra_snippets=True))
        assert len(ex.results) == 1
        assert p.calls == [{"max_results": 5, "config": None}]

    def test_default_exchange_forwards_only_when_requested(self):
        p = _Modern()
        asyncio.run(p.search_exchange("q"))
        asyncio.run(p.search_exchange("q", extra_snippets=True))
        assert p.calls == [{"extra_snippets": False}, {"extra_snippets": True}]

    def _patch_audit(self, monkeypatch):
        monkeypatch.setattr(A, "record_search", AsyncMock(return_value="uuid"))
        monkeypatch.setattr(A, "load_audit_config", AsyncMock(return_value={
            "enabled": True, "store_body": True, "max_body_chars": 1000}))

    def test_run_logged_search_never_forwards_to_an_older_exchange(self, monkeypatch):
        """The stub is shaped like every pre-existing provider and test double."""
        self._patch_audit(monkeypatch)
        seen = []

        async def _ex(query, *, max_results=5, config=None):
            seen.append({"max_results": max_results})
            return SearchExchange(results=[SearchResult(title="t", url="u", snippet="s")])

        provider = MagicMock()
        provider.provider_key = "brave"
        provider.search_exchange = _ex
        out = asyncio.run(A.run_logged_search(
            MagicMock(), "q", source="search_api", max_results=3,
            config={"search.provider": "brave"}, provider=provider, extra_snippets=True,
        ))
        assert [r.title for r in out] == ["t"]
        assert seen == [{"max_results": 3}]

    def test_run_logged_search_forwards_to_an_exchange_that_takes_it(self, monkeypatch):
        self._patch_audit(monkeypatch)
        seen = []

        async def _ex(query, *, max_results=5, config=None, extra_snippets=False):
            seen.append(extra_snippets)
            return SearchExchange(results=[])

        provider = MagicMock()
        provider.provider_key = "brave"
        provider.search_exchange = _ex
        common = dict(source="search_api", config={"search.provider": "brave"}, provider=provider)
        asyncio.run(A.run_logged_search(MagicMock(), "q", **common))
        asyncio.run(A.run_logged_search(MagicMock(), "q", extra_snippets=True, **common))
        assert seen == [False, True]


# ------------------------------------------------------------------
# API surface.
# ------------------------------------------------------------------


class TestApi:
    def test_request_defaults_off_and_item_carries_the_excerpts(self):
        from backend.app.api.search_api import SearchRequest, SearchResultItem

        assert SearchRequest(query="q").extra_snippets is False
        assert SearchRequest(query="q", extra_snippets=True).extra_snippets is True
        assert SearchResultItem(title="T", url="https://e.com", snippet="s").extra_snippets == []
        r = SearchResult(title="T", url="https://e.com", snippet="s", extra={"extra_snippets": ["x"]})
        item = SearchResultItem(
            title=r.title, url=r.url, snippet=r.snippet, published=r.published,
            extra_snippets=list((r.extra or {}).get("extra_snippets") or []),
        )
        assert item.extra_snippets == ["x"]

    def test_the_endpoint_passes_the_flag_through_and_maps_it_back(self):
        """Structural: the request field reaches run_logged_search and each
        result's extra lands on the item — the two ends of the plumbing."""
        from pathlib import Path

        src = (Path(A.__file__).resolve().parents[2] / "api" / "search_api.py").read_text()
        assert "extra_snippets=body.extra_snippets" in src
        assert 'extra_snippets=list((r.extra or {}).get("extra_snippets") or [])' in src
