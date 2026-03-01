"""Add Azure AD SSO support: azure_oid column, nullable password_hash

Revision ID: 019
Revises: 018
Create Date: 2026-02-19
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "019"
down_revision: Union[str, None] = "018"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add azure_oid column
    op.add_column(
        "users",
        sa.Column("azure_oid", sa.String(36), nullable=True),
    )
    op.create_index("ix_users_azure_oid", "users", ["azure_oid"], unique=True)

    # Make password_hash nullable (SSO-only users won't have a password)
    op.alter_column(
        "users",
        "password_hash",
        existing_type=sa.String(255),
        nullable=True,
    )


def downgrade() -> None:
    # Make password_hash non-nullable again
    op.alter_column(
        "users",
        "password_hash",
        existing_type=sa.String(255),
        nullable=False,
    )

    # Drop azure_oid column
    op.drop_index("ix_users_azure_oid", table_name="users")
    op.drop_column("users", "azure_oid")
