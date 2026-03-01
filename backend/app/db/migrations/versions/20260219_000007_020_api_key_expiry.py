"""Add api_key_expiry_days and max_api_keys to groups.

Revision ID: 020
Revises: 019
Create Date: 2026-02-19
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = "020"
down_revision = "019"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "groups",
        sa.Column("api_key_expiry_days", sa.Integer(), nullable=False, server_default="45"),
    )
    op.add_column(
        "groups",
        sa.Column("max_api_keys", sa.Integer(), nullable=False, server_default="8"),
    )


def downgrade() -> None:
    op.drop_column("groups", "max_api_keys")
    op.drop_column("groups", "api_key_expiry_days")
