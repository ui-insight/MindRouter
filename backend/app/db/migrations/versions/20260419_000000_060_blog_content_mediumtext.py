"""Widen blog_posts.content from TEXT to MEDIUMTEXT.

TEXT is limited to 65,535 bytes in MariaDB, which is too small for
long blog posts with embedded images. MEDIUMTEXT supports up to 16 MB.

Revision ID: 060
Revises: 059
Create Date: 2026-04-19
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.mysql import MEDIUMTEXT

revision = "060"
down_revision = "059"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "blog_posts",
        "content",
        type_=MEDIUMTEXT,
        existing_nullable=False,
    )


def downgrade() -> None:
    op.alter_column(
        "blog_posts",
        "content",
        type_=sa.Text(),
        existing_nullable=False,
    )
