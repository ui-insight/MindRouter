############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# 055_add_api_key_archived_offsets.py: Add per-API-key
#     archived token/request offsets so that per-key lifetime
#     counters stay correct after retention deletes old
#     requests from the main DB.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Add archived token offset columns to api_keys table.

Revision ID: 055
Revises: 054
Create Date: 2026-04-16
"""

import sqlalchemy as sa
from alembic import op

revision = "055"
down_revision = "054"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "api_keys",
        sa.Column("archived_total_tokens", sa.BigInteger(), nullable=False, server_default="0"),
    )
    op.add_column(
        "api_keys",
        sa.Column("archived_prompt_tokens", sa.BigInteger(), nullable=False, server_default="0"),
    )
    op.add_column(
        "api_keys",
        sa.Column("archived_completion_tokens", sa.BigInteger(), nullable=False, server_default="0"),
    )
    op.add_column(
        "api_keys",
        sa.Column("archived_request_count", sa.BigInteger(), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_column("api_keys", "archived_request_count")
    op.drop_column("api_keys", "archived_completion_tokens")
    op.drop_column("api_keys", "archived_prompt_tokens")
    op.drop_column("api_keys", "archived_total_tokens")
