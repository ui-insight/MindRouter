"""Add stored_responses table for the OpenAI Responses API dialect.

Server-side state for store=true / previous_response_id (Tier 2 of the
Responses API plan, docs/responses-api-plan.md).

NOTE: MariaDB DDL is non-transactional — if this migration fails
midway, manual cleanup may be required.  Any future ALTER touching the
FK-backed indexes must drop the FK constraints first (MariaDB error
1553).

Revision ID: 062
Revises: 061
Create Date: 2026-07-15
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.mysql import MEDIUMTEXT

revision = "062"
down_revision = "061"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "stored_responses",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("response_id", sa.String(80), nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("api_key_id", sa.Integer(), sa.ForeignKey("api_keys.id"), nullable=False),
        sa.Column("request_id", sa.BigInteger(), sa.ForeignKey("requests.id"), nullable=True),
        sa.Column("model", sa.String(200), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "in_progress", "completed", "failed", "incomplete", "cancelled",
                name="storedresponsestatus",
            ),
            nullable=False,
        ),
        sa.Column("previous_response_id", sa.String(80), nullable=True),
        sa.Column("input_items", sa.JSON(), nullable=True),
        sa.Column("output_items", sa.JSON(), nullable=True),
        sa.Column("instructions", MEDIUMTEXT, nullable=True),
        sa.Column("parameters", sa.JSON(), nullable=True),
        sa.Column("usage", sa.JSON(), nullable=True),
        sa.Column("error", sa.JSON(), nullable=True),
        sa.Column("offloaded_images", sa.JSON(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True),
            server_default=sa.func.now(), nullable=False,
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True),
            server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("response_id", name="uq_stored_responses_response_id"),
    )
    op.create_index("ix_stored_responses_response_id", "stored_responses", ["response_id"])
    op.create_index("ix_stored_responses_user_created", "stored_responses", ["user_id", "created_at"])
    op.create_index("ix_stored_responses_created_at", "stored_responses", ["created_at"])
    op.create_index("ix_stored_responses_prev", "stored_responses", ["previous_response_id"])


def downgrade() -> None:
    op.drop_index("ix_stored_responses_prev", table_name="stored_responses")
    op.drop_index("ix_stored_responses_created_at", table_name="stored_responses")
    op.drop_index("ix_stored_responses_user_created", table_name="stored_responses")
    op.drop_index("ix_stored_responses_response_id", table_name="stored_responses")
    op.drop_table("stored_responses")
