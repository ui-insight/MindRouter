"""Add conversations + conversation_items tables (Conversations API).

Server-side conversation state for the OpenAI Conversations API
(conv_* objects) consumed by POST /v1/responses via the
``conversation`` parameter.

NOTE: MariaDB DDL is non-transactional — if this migration fails
midway, manual cleanup may be required.  Future ALTERs touching the
FK-backed indexes must drop the FK constraints first (MariaDB 1553).

Revision ID: 063
Revises: 062
Create Date: 2026-07-16
"""

import sqlalchemy as sa
from alembic import op

revision = "063"
down_revision = "062"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "conversations",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("conversation_id", sa.String(80), nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("api_key_id", sa.Integer(), sa.ForeignKey("api_keys.id"), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True),
            server_default=sa.func.now(), nullable=False,
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True),
            server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("conversation_id", name="uq_conversations_conversation_id"),
    )
    op.create_index("ix_conversations_conversation_id", "conversations", ["conversation_id"])
    op.create_index("ix_conversations_user_created", "conversations", ["user_id", "created_at"])
    op.create_index("ix_conversations_updated_at", "conversations", ["updated_at"])

    op.create_table(
        "conversation_items",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column(
            "conversation_pk", sa.BigInteger(),
            sa.ForeignKey("conversations.id"), nullable=False,
        ),
        sa.Column("item_id", sa.String(80), nullable=False),
        sa.Column("item", sa.JSON(), nullable=False),
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
        sa.UniqueConstraint(
            "conversation_pk", "item_id", name="uq_conversation_items_conv_item"
        ),
    )
    op.create_index(
        "ix_conversation_items_conv", "conversation_items",
        ["conversation_pk", "id"],
    )


def downgrade() -> None:
    op.drop_index("ix_conversation_items_conv", table_name="conversation_items")
    op.drop_table("conversation_items")
    op.drop_index("ix_conversations_updated_at", table_name="conversations")
    op.drop_index("ix_conversations_user_created", table_name="conversations")
    op.drop_index("ix_conversations_conversation_id", table_name="conversations")
    op.drop_table("conversations")
