"""Add user_images table for image gallery.

Revision ID: 059
Revises: 058
Create Date: 2026-04-17
"""

from alembic import op
import sqlalchemy as sa

revision = "059"
down_revision = "058"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "user_images",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.Integer, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("prompt", sa.Text, nullable=False),
        sa.Column("model", sa.String(200), nullable=False),
        sa.Column("size", sa.String(20), nullable=False),
        sa.Column("steps", sa.Integer, nullable=True),
        sa.Column("guidance_scale", sa.Float, nullable=True),
        sa.Column("seed", sa.BigInteger, nullable=True),
        sa.Column("storage_path", sa.String(500), nullable=False),
        sa.Column("content_type", sa.String(100), nullable=False, server_default="image/png"),
        sa.Column("size_bytes", sa.BigInteger, nullable=False, server_default="0"),
        sa.Column("request_id", sa.BigInteger, sa.ForeignKey("requests.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
    )
    op.create_index("ix_user_images_user_id", "user_images", ["user_id"])
    op.create_index("ix_user_images_created_at", "user_images", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_user_images_created_at", table_name="user_images")
    op.drop_index("ix_user_images_user_id", table_name="user_images")
    op.drop_table("user_images")
