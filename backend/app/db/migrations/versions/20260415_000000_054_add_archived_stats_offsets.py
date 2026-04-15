############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# 054_add_archived_stats_offsets.py: Add per-user and
#     per-model archived token/request offsets so that
#     metrics stay correct after retention deletes old
#     requests from the main DB.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Add archived stats offset columns and model_archived_stats table.

Revision ID: 054
Revises: 053
Create Date: 2026-04-15
"""

import sqlalchemy as sa
from alembic import op

revision = "054"
down_revision = "053"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Per-user archived offsets on the quotas table
    op.add_column(
        "quotas",
        sa.Column("archived_total_tokens", sa.BigInteger(), nullable=False, server_default="0"),
    )
    op.add_column(
        "quotas",
        sa.Column("archived_prompt_tokens", sa.BigInteger(), nullable=False, server_default="0"),
    )
    op.add_column(
        "quotas",
        sa.Column("archived_completion_tokens", sa.BigInteger(), nullable=False, server_default="0"),
    )
    op.add_column(
        "quotas",
        sa.Column("archived_request_count", sa.BigInteger(), nullable=False, server_default="0"),
    )

    # Per-model archived offsets
    op.create_table(
        "model_archived_stats",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("model", sa.String(200), nullable=False, unique=True),
        sa.Column("archived_total_tokens", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("archived_prompt_tokens", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("archived_completion_tokens", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("archived_request_count", sa.BigInteger(), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_table("model_archived_stats")
    op.drop_column("quotas", "archived_request_count")
    op.drop_column("quotas", "archived_completion_tokens")
    op.drop_column("quotas", "archived_prompt_tokens")
    op.drop_column("quotas", "archived_total_tokens")
