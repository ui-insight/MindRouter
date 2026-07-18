############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# website_publisher.py: Publish selected blog posts to the
#   public mindrouter.ai static site by committing generated
#   HTML/images to github.com/sheneman/mindrouter-website.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""GitHub publisher for the public mindrouter.ai blog.

Commits are atomic (one commit per publish/unpublish) via the GitHub Git Data
API: create blobs -> build a tree on top of the current tree -> create a commit
-> move the branch ref. The target repository is HARD-ALLOWLISTED here so this
can never write blog content to any repo other than mindrouter-website — the
app repo never receives post content.

Deploy model: the commit lands on the mindrouter-website repo; the site goes
live when the operator manually pulls on the mindrouter.ai host.
"""

import base64
from typing import Dict, List, Optional, Tuple

import httpx
import structlog

from backend.app.settings import get_settings
from backend.app.dashboard import blog_export

logger = structlog.get_logger(__name__)

_GITHUB_API = "https://api.github.com"

# Hard allowlist. The publisher may target ONLY these repos, regardless of the
# configured setting — enforces "blog content lives only in the website repo".
_ALLOWED_REPOS = frozenset({"sheneman/mindrouter-website"})


class WebsitePublishError(RuntimeError):
    """Raised when publishing to mindrouter-website fails or is disallowed."""


class WebsitePublisher:
    """Publishes/removes blog posts on the mindrouter-website repo via GitHub."""

    def __init__(self, *, settings=None, client: Optional[httpx.AsyncClient] = None):
        self._settings = settings or get_settings()
        self._client = client  # injectable for tests; created lazily otherwise

    # -- configuration ------------------------------------------------------
    @property
    def repo(self) -> str:
        return self._settings.website_publish_repo

    @property
    def branch(self) -> str:
        return self._settings.website_publish_branch

    @property
    def enabled(self) -> bool:
        """True only when configured AND the target repo is allowlisted."""
        return bool(
            getattr(self._settings, "website_publish_enabled", False)
            and getattr(self._settings, "website_publish_github_token", "")
            and self.repo in _ALLOWED_REPOS
        )

    def _ensure_allowed(self) -> None:
        if self.repo not in _ALLOWED_REPOS:
            raise WebsitePublishError(
                f"Refusing to publish: repo '{self.repo}' is not in the "
                f"allowlist {sorted(_ALLOWED_REPOS)}."
            )

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    # -- GitHub API ---------------------------------------------------------
    async def _api(self, method: str, path: str, *, json=None, ok=(200, 201)) -> dict:
        self._ensure_allowed()
        token = self._settings.website_publish_github_token
        if not token:
            raise WebsitePublishError("No GitHub token configured (website_publish_github_token).")
        client = await self._get_client()
        resp = await client.request(
            method,
            f"{_GITHUB_API}{path}",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            json=json,
        )
        if resp.status_code not in ok:
            raise WebsitePublishError(
                f"GitHub API {method} {path} -> {resp.status_code}: {resp.text[:300]}"
            )
        return resp.json()

    async def _commit(self, message: str, writes: Dict[str, bytes], deletes: List[str]) -> str:
        """Create one atomic commit that applies ``writes`` and ``deletes``."""
        ref = await self._api("GET", f"/repos/{self.repo}/git/ref/heads/{self.branch}")
        base_sha = ref["object"]["sha"]
        base_commit = await self._api("GET", f"/repos/{self.repo}/git/commits/{base_sha}")
        base_tree = base_commit["tree"]["sha"]

        tree: List[dict] = []
        for repo_path, data in writes.items():
            blob = await self._api(
                "POST", f"/repos/{self.repo}/git/blobs",
                json={"content": base64.b64encode(data).decode("ascii"), "encoding": "base64"},
            )
            tree.append({"path": repo_path, "mode": "100644", "type": "blob", "sha": blob["sha"]})
        for repo_path in deletes:
            # A null sha removes the path from the new tree.
            tree.append({"path": repo_path, "mode": "100644", "type": "blob", "sha": None})

        new_tree = await self._api(
            "POST", f"/repos/{self.repo}/git/trees",
            json={"base_tree": base_tree, "tree": tree},
        )
        commit = await self._api(
            "POST", f"/repos/{self.repo}/git/commits",
            json={"message": message, "tree": new_tree["sha"], "parents": [base_sha]},
        )
        await self._api(
            "PATCH", f"/repos/{self.repo}/git/refs/heads/{self.branch}",
            json={"sha": commit["sha"], "force": False},
        )
        return commit["sha"]

    # -- file sets ----------------------------------------------------------
    async def build_publish_fileset(self, post, selected_posts, storage=None) -> Tuple[Dict[str, bytes], List[str]]:
        """Files to write for publishing ``post``: its page, images, index, feed, css."""
        exported = await blog_export.export_post(post, storage=storage)
        writes: Dict[str, bytes] = {
            exported.repo_path: exported.html.encode("utf-8"),
            "blog/index.html": blog_export.render_index_html(selected_posts).encode("utf-8"),
            "blog/feed.xml": blog_export.render_feed_xml(selected_posts).encode("utf-8"),
            "css/blog.css": blog_export.BLOG_CSS.encode("utf-8"),
        }
        for img in exported.images:
            if img.missing or img.data is None:
                logger.warning("website_publish_missing_image", path=img.storage_path, slug=post.slug)
                continue
            writes[img.repo_path] = img.data
        return writes, []

    async def build_unpublish_fileset(self, post, selected_posts) -> Tuple[Dict[str, bytes], List[str]]:
        """Files to write/delete for removing ``post``: delete its page+images, refresh index/feed."""
        writes: Dict[str, bytes] = {
            "blog/index.html": blog_export.render_index_html(selected_posts).encode("utf-8"),
            "blog/feed.xml": blog_export.render_feed_xml(selected_posts).encode("utf-8"),
        }
        deletes = [blog_export.post_repo_path(post.slug)]
        deletes += [f"blog/images/{p}" for p in blog_export.collect_image_paths(post.content)]
        return writes, deletes

    # -- public operations --------------------------------------------------
    async def publish(self, post, selected_posts, storage=None) -> str:
        """Commit ``post`` (and a refreshed index/feed) to mindrouter-website."""
        writes, deletes = await self.build_publish_fileset(post, selected_posts, storage=storage)
        sha = await self._commit(
            f"blog: publish {post.slug} (post {getattr(post, 'id', '?')})", writes, deletes
        )
        logger.info("website_published", slug=post.slug, commit=sha, files=len(writes))
        return sha

    async def unpublish(self, post, selected_posts) -> str:
        """Remove ``post`` from mindrouter-website (and refresh index/feed)."""
        writes, deletes = await self.build_unpublish_fileset(post, selected_posts)
        sha = await self._commit(
            f"blog: remove {post.slug} (post {getattr(post, 'id', '?')})", writes, deletes
        )
        logger.info("website_unpublished", slug=post.slug, commit=sha)
        return sha


_publisher: Optional[WebsitePublisher] = None


def get_website_publisher() -> WebsitePublisher:
    """Global publisher instance."""
    global _publisher
    if _publisher is None:
        _publisher = WebsitePublisher()
    return _publisher
