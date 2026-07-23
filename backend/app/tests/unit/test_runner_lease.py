############################################################
#
# mindrouter - unit tests for the video-runner leader lease (redis CAS)
#
# The app runs multiple uvicorn workers; only ONE may run the video runner.
# These verify the compare-and-act lease semantics so a stalled leader can never
# clobber the new one.
#
############################################################

"""Unit tests for the Redis leader-lease primitives."""

import pytest

import backend.app.core.redis_client as rc


class FakeRedis:
    """Minimal async Redis emulating SET NX EX + our two CAS Lua scripts."""

    def __init__(self):
        self.store = {}

    async def set(self, key, val, nx=False, ex=None):
        if nx and key in self.store:
            return None
        self.store[key] = val
        return True

    async def eval(self, script, numkeys, key, *args):
        token = args[0]
        if self.store.get(key) != token:
            return 0
        if "expire" in script:      # renew
            return 1
        if "del" in script:         # release
            self.store.pop(key, None)
            return 1
        return 0


@pytest.fixture
def fake(monkeypatch):
    r = FakeRedis()
    monkeypatch.setattr(rc, "_redis", r)
    monkeypatch.setattr(rc, "_available", True)
    return r


@pytest.mark.asyncio
async def test_acquire_is_exclusive(fake):
    assert await rc.acquire_lease("k", "tok-A", 30) is True
    assert await rc.acquire_lease("k", "tok-B", 30) is False  # already held


@pytest.mark.asyncio
async def test_only_owner_renews(fake):
    await rc.acquire_lease("k", "tok-A", 30)
    assert await rc.renew_lease("k", "tok-A", 30) is True
    assert await rc.renew_lease("k", "tok-B", 30) is False  # not the owner


@pytest.mark.asyncio
async def test_only_owner_releases_then_reacquirable(fake):
    await rc.acquire_lease("k", "tok-A", 30)
    await rc.release_lease("k", "tok-B")               # wrong owner → no-op
    assert await rc.acquire_lease("k", "tok-B", 30) is False  # still held by A
    await rc.release_lease("k", "tok-A")               # owner releases
    assert await rc.acquire_lease("k", "tok-B", 30) is True   # now free


@pytest.mark.asyncio
async def test_unavailable_redis_is_safe(monkeypatch):
    monkeypatch.setattr(rc, "_redis", None)
    monkeypatch.setattr(rc, "_available", False)
    assert await rc.acquire_lease("k", "t", 30) is False
    assert await rc.renew_lease("k", "t", 30) is False
    await rc.release_lease("k", "t")  # no raise
