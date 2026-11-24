"""Unit tests for HttpMeshWorkerClient (services/mesh_worker_client.py).

A fake httpx.AsyncClient records each call, so the tests check the worker
contract (paths, X-Worker-Key, the ?variant= parameter), the submit error
classification (retryable vs not), and the streamed-download cap, with no
live worker. The module is spec-loaded with the backend package chain stubbed
(its services/__init__ pulls the DB); sys.modules is restored afterwards.
"""

import asyncio
import importlib.util
import secrets
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

_SVC = Path(__file__).resolve().parents[2] / "services"
_MISSING = object()
_settings = SimpleNamespace(internal_tls_verify=True, mesh_worker_fetch_max_mb=512, video_worker_fetch_max_mb=2048)
_STUBS = {
    "backend": MagicMock(),
    "backend.app": MagicMock(),
    "backend.app.services": MagicMock(),
    "backend.app.settings": MagicMock(get_settings=lambda: _settings),
}
_prev = {n: sys.modules.get(n, _MISSING) for n in list(_STUBS) + ["backend.app.services.video_worker_client"]}
for _n, _s in _STUBS.items():
    sys.modules[_n] = _s


def _load(name, path, register=None):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    if register:
        sys.modules[register] = mod
    spec.loader.exec_module(mod)
    return mod


_vwc = _load("vwc_for_mesh_tests", _SVC / "video_worker_client.py", "backend.app.services.video_worker_client")
mwc = _load("mesh_worker_client_under_test", _SVC / "mesh_worker_client.py")
for _n, _old in _prev.items():
    if _old is _MISSING:
        sys.modules.pop(_n, None)
    else:
        sys.modules[_n] = _old

KEY = secrets.token_hex(16)
BASE = "https://worker.example:8006"


class _Resp:
    def __init__(self, status_code=200, json_data=None, chunks=(), text=""):
        self.status_code = status_code
        self._json = json_data
        self._chunks = list(chunks)
        self.text = text

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("GET", BASE)
            raise httpx.HTTPStatusError("err", request=request, response=httpx.Response(self.status_code, request=request))

    async def aiter_bytes(self):
        for c in self._chunks:
            yield c


class _Stream:
    def __init__(self, resp):
        self.resp = resp

    async def __aenter__(self):
        return self.resp

    async def __aexit__(self, *exc):
        return False


def _fake_client(calls, responses):
    """An AsyncClient stand-in that answers from `responses` (a list, in order)."""

    class Fake:
        def __init__(self, timeout=None, verify=None):
            calls.append(("init", {"timeout": timeout, "verify": verify}))

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def _next(self):
            r = responses.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        async def post(self, url, json=None, headers=None):
            calls.append(("post", url, json, headers))
            return self._next()

        async def get(self, url, headers=None):
            calls.append(("get", url, headers))
            return self._next()

        async def delete(self, url, headers=None):
            calls.append(("delete", url, headers))
            return self._next()

        def stream(self, method, url, params=None, headers=None):
            calls.append(("stream", method, url, params, headers))
            return _Stream(self._next())

    return Fake


def _client(monkeypatch, responses, key=KEY):
    calls = []
    monkeypatch.setattr(mwc.httpx, "AsyncClient", _fake_client(calls, responses))
    return mwc.HttpMeshWorkerClient(api_key=key), calls


def test_submit_posts_to_v1_meshes_with_the_worker_key(monkeypatch):
    client, calls = _client(monkeypatch, [_Resp(202, {"id": "w-1", "status": "queued"})])
    job_id = asyncio.run(client.submit(BASE + "/", {"model": "m", "image": "AA=="}))
    assert job_id == "w-1"
    post = [c for c in calls if c[0] == "post"][0]
    assert post[1] == BASE + "/v1/meshes"
    assert post[3] == {"X-Worker-Key": KEY}
    assert calls[0][1]["verify"] is True


def test_no_key_means_no_header(monkeypatch):
    client, calls = _client(monkeypatch, [_Resp(202, {"id": "w-1"})], key="")
    asyncio.run(client.submit(BASE, {}))
    assert [c for c in calls if c[0] == "post"][0][3] == {}


@pytest.mark.parametrize("resp,retryable", [
    (_Resp(503, text="queue full"), True),
    (_Resp(500), True),
    (_Resp(400, text="undecodable image"), False),
    (_Resp(202, {}), False),          # no id in the answer
    (httpx.ConnectError("refused"), True),
])
def test_submit_failures_are_classified(monkeypatch, resp, retryable):
    client, _ = _client(monkeypatch, [resp])
    with pytest.raises(mwc.WorkerSubmitError) as info:
        asyncio.run(client.submit(BASE, {}))
    assert info.value.retryable is retryable


def test_poll_and_cancel_paths(monkeypatch):
    client, calls = _client(monkeypatch, [_Resp(200, {"status": "in_progress"}), _Resp(200, {})])
    assert asyncio.run(client.poll(BASE, "w-9"))["status"] == "in_progress"
    asyncio.run(client.cancel(BASE, "w-9"))
    assert ("get", BASE + "/v1/meshes/w-9", {"X-Worker-Key": KEY}) in calls
    assert ("delete", BASE + "/v1/meshes/w-9", {"X-Worker-Key": KEY}) in calls


def test_poll_404_raises_http_status_error(monkeypatch):
    client, _ = _client(monkeypatch, [_Resp(404)])
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(client.poll(BASE, "gone"))


def test_cancel_swallows_network_errors(monkeypatch):
    client, _ = _client(monkeypatch, [httpx.ConnectError("down")])
    asyncio.run(client.cancel(BASE, "w-1"))  # no exception


def test_fetch_streams_the_variant_to_disk_with_hash(monkeypatch, tmp_path):
    client, calls = _client(monkeypatch, [_Resp(200, chunks=[b"glTF", b"rest"])])
    dest = tmp_path / "out.glb"
    result = asyncio.run(client.fetch(BASE, "w-1", mwc.VARIANT_GLB, str(dest)))
    assert dest.read_bytes() == b"glTFrest"
    assert result.size_bytes == 8
    import hashlib
    assert result.sha256 == hashlib.sha256(b"glTFrest").hexdigest()
    stream = [c for c in calls if c[0] == "stream"][0]
    assert stream[2] == BASE + "/v1/meshes/w-1/content"
    assert stream[3] == {"variant": "glb"}
    assert list(tmp_path.glob("*.part")) == []


def test_fetch_over_the_cap_leaves_no_file(monkeypatch, tmp_path):
    monkeypatch.setattr(_settings, "mesh_worker_fetch_max_mb", 1)
    client, _ = _client(monkeypatch, [_Resp(200, chunks=[b"x" * 600_000, b"x" * 600_000])])
    dest = tmp_path / "big.glb"
    with pytest.raises(mwc.WorkerFetchError):
        asyncio.run(client.fetch(BASE, "w-1", mwc.VARIANT_GLB, str(dest)))
    assert not dest.exists() and list(tmp_path.glob("*.part")) == []


def test_failed_fetch_keeps_an_earlier_file_intact(monkeypatch, tmp_path):
    dest = tmp_path / "keep.png"
    dest.write_bytes(b"old")
    client, _ = _client(monkeypatch, [_Resp(500)])
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(client.fetch(BASE, "w-1", mwc.VARIANT_PREVIEW, str(dest)))
    assert dest.read_bytes() == b"old"
