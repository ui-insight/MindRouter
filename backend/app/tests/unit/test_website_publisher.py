"""Unit tests for the mindrouter.ai website publisher (PR3).

Uses a fake GitHub client (no network) and a fake ArtifactStorage. The module
is spec-loaded to bypass the backend.app package import chain; its own light
deps (settings, blog_export, httpx, structlog) resolve normally.
"""

import importlib.util
import os
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

_MODPATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "services", "website_publisher.py")
)
_spec = importlib.util.spec_from_file_location("website_publisher", _MODPATH)
wp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wp)


def settings(enabled=True, token="ghp_test", repo="sheneman/mindrouter-website", branch="main"):
    return SimpleNamespace(
        website_publish_enabled=enabled,
        website_publish_github_token=token,
        website_publish_repo=repo,
        website_publish_branch=branch,
    )


def make_post(**kw):
    author = SimpleNamespace(full_name="Luke Sheneman", username="admin")
    d = dict(
        id=7, slug="hello-world", title="Hello World",
        content="Body with an image ![a](/blog/images/2026/07/18/ab/pic_x.png)",
        excerpt="An excerpt.",
        published_at=datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc),
        author=author,
    )
    d.update(kw)
    return SimpleNamespace(**d)


class FakeResponse:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


class FakeGitHub:
    """Minimal GitHub Git-Data API stub; records calls and captures the tree."""

    def __init__(self, fail_on=None):
        self.calls = []            # list of (method, url, json)
        self.trees = []            # captured tree entry lists
        self.fail_on = fail_on     # (method, url_substr) -> return 422
        self._blob_n = 0

    async def request(self, method, url, headers=None, json=None):
        self.calls.append((method, url, json))
        if self.fail_on and self.fail_on[0] == method and self.fail_on[1] in url:
            return FakeResponse(422, {}, "boom")
        if method == "GET" and "/git/ref/heads/" in url:
            return FakeResponse(200, {"object": {"sha": "BASE"}})
        if method == "GET" and "/git/commits/" in url:
            return FakeResponse(200, {"tree": {"sha": "BASETREE"}})
        if method == "POST" and url.endswith("/git/blobs"):
            self._blob_n += 1
            return FakeResponse(201, {"sha": f"blob{self._blob_n}"})
        if method == "POST" and url.endswith("/git/trees"):
            self.trees.append(json["tree"])
            return FakeResponse(201, {"sha": "NEWTREE"})
        if method == "POST" and url.endswith("/git/commits"):
            return FakeResponse(201, {"sha": "NEWCOMMIT"})
        if method == "PATCH" and "/git/refs/heads/" in url:
            return FakeResponse(200, {})
        return FakeResponse(500, {}, f"unexpected {method} {url}")


class FakeStorage:
    def __init__(self, present):
        self._present = present

    async def retrieve(self, path):
        return self._present.get(path)


def _pub(fail_on=None, **skw):
    gh = FakeGitHub(fail_on=fail_on)
    return wp.WebsitePublisher(settings=settings(**skw), client=gh), gh


# --- config / safety --------------------------------------------------------
def test_enabled_requires_config_and_allowlist():
    assert wp.WebsitePublisher(settings=settings()).enabled is True
    assert wp.WebsitePublisher(settings=settings(enabled=False)).enabled is False
    assert wp.WebsitePublisher(settings=settings(token="")).enabled is False
    assert wp.WebsitePublisher(settings=settings(repo="evil/repo")).enabled is False


def test_ensure_allowed_rejects_other_repos():
    pub = wp.WebsitePublisher(settings=settings(repo="ui-insight/MindRouter"))
    with pytest.raises(wp.WebsitePublishError):
        pub._ensure_allowed()
    # allowlisted repo is fine
    wp.WebsitePublisher(settings=settings())._ensure_allowed()


@pytest.mark.asyncio
async def test_publish_to_disallowed_repo_raises():
    pub, gh = _pub(repo="attacker/repo")
    with pytest.raises(wp.WebsitePublishError):
        await pub.publish(make_post(content="no images"), [make_post()], storage=FakeStorage({}))
    assert gh.calls == []  # never touched the API


# --- file sets --------------------------------------------------------------
@pytest.mark.asyncio
async def test_build_publish_fileset():
    pub, _ = _pub()
    post = make_post()
    storage = FakeStorage({"2026/07/18/ab/pic_x.png": b"PNGDATA"})
    writes, deletes = await pub.build_publish_fileset(post, [post], storage=storage)
    assert deletes == []
    assert "blog/hello-world/index.html" in writes
    assert "blog/index.html" in writes
    assert "blog/feed.xml" in writes
    assert "css/blog.css" in writes
    assert writes["blog/images/2026/07/18/ab/pic_x.png"] == b"PNGDATA"
    assert b"<!DOCTYPE html>" in writes["blog/hello-world/index.html"]


@pytest.mark.asyncio
async def test_build_publish_fileset_skips_missing_image():
    pub, _ = _pub()
    post = make_post()
    writes, _ = await pub.build_publish_fileset(post, [post], storage=FakeStorage({}))
    assert "blog/images/2026/07/18/ab/pic_x.png" not in writes  # missing -> skipped
    assert "blog/hello-world/index.html" in writes               # page still written


@pytest.mark.asyncio
async def test_build_unpublish_fileset_deletes_page_and_images():
    pub, _ = _pub()
    post = make_post()
    writes, deletes = await pub.build_unpublish_fileset(post, [])  # no posts left selected
    assert set(writes) == {"blog/index.html", "blog/feed.xml"}
    assert "blog/hello-world/index.html" in deletes
    assert "blog/images/2026/07/18/ab/pic_x.png" in deletes


# --- commit mechanics -------------------------------------------------------
@pytest.mark.asyncio
async def test_publish_commit_sequence_and_tree():
    pub, gh = _pub()
    post = make_post()
    sha = await pub.publish(post, [post], storage=FakeStorage({"2026/07/18/ab/pic_x.png": b"IMG"}))
    assert sha == "NEWCOMMIT"

    methods = [(m, u.rsplit("/git/", 1)[-1] if "/git/" in u else u) for m, u, _ in gh.calls]
    # base ref -> base commit -> blobs -> tree -> commit -> move ref
    assert methods[0][0] == "GET" and "ref/heads/main" in methods[0][1]
    assert methods[1][0] == "GET" and methods[1][1].startswith("commits/")
    assert any(m == "POST" and u == "blobs" for m, u in methods)
    assert any(m == "POST" and u == "trees" for m, u in methods)
    assert any(m == "POST" and u == "commits" for m, u in methods)
    assert methods[-1][0] == "PATCH" and "refs/heads/main" in methods[-1][1]

    tree = gh.trees[-1]
    paths = {e["path"]: e for e in tree}
    assert "blog/hello-world/index.html" in paths and paths["blog/hello-world/index.html"]["sha"]
    assert "blog/index.html" in paths and "css/blog.css" in paths
    # all writes carry a (non-null) blob sha
    assert all(e["sha"] for e in tree)


@pytest.mark.asyncio
async def test_unpublish_uses_null_sha_deletions():
    pub, gh = _pub()
    post = make_post()
    sha = await pub.unpublish(post, [])
    assert sha == "NEWCOMMIT"
    tree = gh.trees[-1]
    by_path = {e["path"]: e for e in tree}
    assert by_path["blog/hello-world/index.html"]["sha"] is None            # deletion
    assert by_path["blog/images/2026/07/18/ab/pic_x.png"]["sha"] is None    # deletion
    assert by_path["blog/index.html"]["sha"] is not None                     # rewrite


@pytest.mark.asyncio
async def test_api_error_is_raised():
    pub, gh = _pub(fail_on=("POST", "/git/trees"))
    with pytest.raises(wp.WebsitePublishError) as ei:
        await pub.publish(make_post(content="no images"), [make_post()], storage=FakeStorage({}))
    assert "422" in str(ei.value)
