############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# 010_scale_and_retention.py: Scale columns to MEDIUMTEXT,
#     add thumbnail_path, add indexes for retention and audit
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Scale columns to MEDIUMTEXT, add thumbnail_path and indexes

Revision ID: 010
Revises: 009
Create Date: 2026-02-14 00:00:00.000000

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.mysql import MEDIUMTEXT

# revision identifiers, used by Alembic
revision = "010"
down_revision = "009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Widen chat_messages.content from TEXT to MEDIUMTEXT (supports >64KB responses)
    op.alter_column(
        "chat_messages", "content",
        existing_type=sa.Text(),
        type_=MEDIUMTEXT,
        existing_nullable=True,
    )

    # 2. Widen responses.content from TEXT to MEDIUMTEXT
    op.alter_column(
        "responses", "content",
        existing_type=sa.Text(),
        type_=MEDIUMTEXT,
        existing_nullable=True,
    )

    # 3. Add thumbnail_path column to chat_attachments
    op.add_column(
        "chat_attachments",
        sa.Column("thumbnail_path", sa.String(500), nullable=True),
    )

    # 4. Add index on chat_conversations.updated_at for retention cleanup queries
    op.create_index(
        "ix_chat_conversations_updated_at",
        "chat_conversations",
        ["updated_at"],
    )

    # 5. Add index on requests.created_at for admin audit queries
    op.create_index(
        "ix_requests_created_at",
        "requests",
        ["created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_requests_created_at", table_name="requests")
    op.drop_index("ix_chat_conversations_updated_at", table_name="chat_conversations")
    op.drop_column("chat_attachments", "thumbnail_path")
    op.alter_column(
        "responses", "content",
        existing_type=MEDIUMTEXT,
        type_=sa.Text(),
        existing_nullable=True,
    )
    op.alter_column(
        "chat_messages", "content",
        existing_type=MEDIUMTEXT,
        type_=sa.Text(),
        existing_nullable=True,
    )
