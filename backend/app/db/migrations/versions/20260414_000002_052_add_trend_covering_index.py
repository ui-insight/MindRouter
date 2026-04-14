############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# 052_add_trend_covering_index.py: Add a covering index for
#     the status page trend queries (token totals and active
#     users bucketed by time).
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Add covering index for status-page trend queries

The token-trend and active-users-trend queries on the public status page
scan ``requests`` with ``WHERE created_at >= :cutoff`` and aggregate
token columns / COUNT(DISTINCT user_id).  For week/month/year ranges
this touches millions of rows.

A covering index on ``(created_at, total_tokens, prompt_tokens,
completion_tokens, user_id)`` lets MariaDB satisfy both queries with
index-only scans, avoiding heap reads entirely.

Revision ID: 052
Revises: 051
Create Date: 2026-04-14 00:00:02.000000

"""
from alembic import op

# revision identifiers, used by Alembic.
revision = '052'
down_revision = '051'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        'ix_requests_created_tokens_user_covering',
        'requests',
        ['created_at', 'total_tokens', 'prompt_tokens', 'completion_tokens', 'user_id'],
    )


def downgrade() -> None:
    op.drop_index('ix_requests_created_tokens_user_covering', table_name='requests')
