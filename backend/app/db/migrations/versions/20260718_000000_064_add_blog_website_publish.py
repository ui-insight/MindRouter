"""Add selective mindrouter.ai publish state to blog_posts.

Adds three columns supporting opt-in publishing of individual blog posts to
the public mindrouter.ai static site (github.com/sheneman/mindrouter-website),
independent of the app's own is_published flag:

  * website_published      - selected for the public site
  * website_published_at    - when it was last pushed
  * website_commit_sha      - the mindrouter-website commit that carries it

NOTE: MariaDB DDL is non-transactional — if this migration fails midway,
manual cleanup may be required.

Revision ID: 064
Revises: 063
Create Date: 2026-07-18
"""

import sqlalchemy as sa
from alembic import op

revision = "064"
down_revision = "063"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "blog_posts",
        sa.Column(
            "website_published",
            sa.Boolean(),
            server_default=sa.text("0"),
            nullable=False,
        ),
    )
    op.add_column(
        "blog_posts",
        sa.Column("website_published_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "blog_posts",
        sa.Column("website_commit_sha", sa.String(64), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("blog_posts", "website_commit_sha")
    op.drop_column("blog_posts", "website_published_at")
    op.drop_column("blog_posts", "website_published")
