"""/v1/search extra_snippets: the Brave provider asks for them only when
told to, keeps up to five per result, and the API item carries them."""
import asyncio
import json

import pytest

from backend.app.services.search import brave as brave_mod


class _Resp:
    def __init__(self, payload):
        self.status_code = 200
        self.text = json.dumps(payload)
        self.headers = {}
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _Client:
    seen: list = []

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, params=None, headers=None):
        _Client.seen.append(dict(params or {}))
        return _Resp({"web": {"results": [
            {"title": "T", "url": "https://example.com/a", "description": "d", "page_age": "2025-01-01",
             "extra_snippets": ["one", "two", "three", "four", "five", "six", 7, ""]},
            {"title": "U", "url": "https://example.com/b", "description": "e"},
        ]}})


@pytest.fixture
def fake_httpx(monkeypatch):
    _Client.seen = []
    monkeypatch.setattr(brave_mod.httpx, "AsyncClient", _Client)
    return _Client


def test_brave_extra_snippets_are_requested_and_parsed(fake_httpx):
    provider = brave_mod.BraveSearchProvider()
    cfg = {"search.brave.api_key": "k"}
    plain = asyncio.run(provider.search("q", max_results=5, config=cfg))
    assert "extra_snippets" not in fake_httpx.seen[0]
    assert plain[0].extra == {}
    rich = asyncio.run(provider.search("q", max_results=5, config=cfg, extra_snippets=True))
    assert fake_httpx.seen[1]["extra_snippets"] == "true"
    assert rich[0].extra["extra_snippets"] == ["one", "two", "three", "four", "five"]
    assert rich[1].extra == {}


def test_api_item_carries_extra_snippets():
    from backend.app.api.search_api import SearchRequest, SearchResultItem
    from backend.app.services.search.base import SearchResult

    assert SearchRequest(query="q").extra_snippets is False
    r = SearchResult(title="T", url="https://e.com", snippet="s", extra={"extra_snippets": ["x"]})
    item = SearchResultItem(title=r.title, url=r.url, snippet=r.snippet, published=r.published, extra_snippets=list((r.extra or {}).get("extra_snippets") or []))
    assert item.extra_snippets == ["x"]
