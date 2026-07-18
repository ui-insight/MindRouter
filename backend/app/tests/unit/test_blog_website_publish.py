"""PR2 contract tests for selective mindrouter.ai publishing.

Source-inspection tests (no DB): verify the migration, model, routes, and CRUD
that manage the ``website_published`` state stay consistent. This mirrors the
migration/source-check style used by the responses-store tests and avoids the
backend.app package import chain that pulls in the DB stack.
"""

import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def _read(rel):
    with open(os.path.join(ROOT, rel), "r") as f:
        return f.read()


MIGRATION = "backend/app/db/migrations/versions/20260718_000000_064_add_blog_website_publish.py"
MODELS = "backend/app/db/models.py"
CRUD = "backend/app/db/crud.py"
BLOG = "backend/app/dashboard/blog.py"

_COLS = ["website_published", "website_published_at", "website_commit_sha"]


def test_migration_adds_and_drops_the_three_columns():
    src = _read(MIGRATION)
    assert 'revision = "064"' in src and 'down_revision = "063"' in src
    for col in _COLS:
        assert f'add_column(\n        "blog_posts",' in src  # sanity: targets blog_posts
        assert f'"{col}"' in src, f"migration missing add for {col}"
        assert f'op.drop_column("blog_posts", "{col}")' in src, f"downgrade missing drop for {col}"
    # not-null boolean with a server default so existing rows backfill cleanly
    assert "website_published" in src and 'server_default=sa.text("0")' in src
    assert "nullable=False" in src  # website_published


def test_model_has_website_fields():
    src = _read(MODELS)
    block = src[src.index("class BlogPost("):]
    block = block[: block.index("class ", 10)] if "class " in block[10:] else block
    for col in _COLS:
        assert re.search(rf"\b{col}\b\s*:\s*Mapped", block), f"BlogPost missing {col}"
    assert "website_commit_sha" in block and "String(64)" in block


def test_crud_query_filters_selected_published_undeleted():
    src = _read(CRUD)
    fn = src[src.index("async def get_website_published_blog_posts"):]
    fn = fn[: fn.index("\nasync def ")]
    assert "BlogPost.website_published.is_(True)" in fn
    assert "BlogPost.is_published.is_(True)" in fn      # never leak drafts
    assert "BlogPost.deleted_at.is_(None)" in fn         # never leak deleted
    assert "order_by(BlogPost.published_at.desc())" in fn


def test_routes_exist_with_guard_and_state_transitions():
    src = _read(BLOG)
    assert '"/admin/blog/{post_id}/website-publish"' in src
    assert '"/admin/blog/{post_id}/website-unpublish"' in src

    pub = src[src.index("async def admin_blog_website_publish"):]
    pub = pub[: pub.index("\nasync def ")]
    assert "_require_admin(" in pub                       # full admin, mutating
    assert "if not post.is_published:" in pub             # gate: no draft leaks
    assert "website_published=True" in pub
    assert "website_published_at=datetime.now(timezone.utc)" in pub

    unpub = src[src.index("async def admin_blog_website_unpublish"):]
    unpub = unpub[: unpub.index("\nasync def ")]
    assert "website_published=False" in unpub
    assert "website_published_at=None" in unpub
    assert "website_commit_sha=None" in unpub
