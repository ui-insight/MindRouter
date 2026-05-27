"""Add model_aliases table for default model alias mappings.

Stores named aliases (e.g. default-llm) that resolve to real model
names at request time, allowing stable API contracts while admins
swap underlying models.

Revision ID: 061
Revises: 060
Create Date: 2026-05-27
"""

import sqlalchemy as sa
from alembic import op

revision = "061"
down_revision = "060"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "model_aliases",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("alias_name", sa.String(200), nullable=False),
        sa.Column("target_model", sa.String(200), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("alias_name", name="uq_model_aliases_alias_name"),
    )
    op.create_index("ix_model_aliases_alias_name", "model_aliases", ["alias_name"])


def downgrade() -> None:
    op.drop_index("ix_model_aliases_alias_name", table_name="model_aliases")
    op.drop_table("model_aliases")
