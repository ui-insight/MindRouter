############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# 050_add_user_detail_covering_indices.py: Add covering
#     indices for the admin user detail page queries that
#     were doing full table scans on 1M+ user rows.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Add covering indices for admin user detail page

Two queries on the admin user detail page caused 100-120 second
loads for high-volume users (bbaum: 775K+ requests):

1. Favorite models query (GROUP BY model, 102s): The existing
   (user_id, created_at) index requires a heap read for every
   row to fetch the ``model`` column.  A covering index on
   ``(user_id, model)`` lets MariaDB answer the query from
   the index alone.

2. Monthly usage query (GROUP BY month with SUM of tokens, 118s):
   MariaDB chose a full table scan (4.7M rows) because no index
   covers user_id + created_at + the three token columns.  A
   covering index on ``(user_id, created_at, total_tokens,
   prompt_tokens, completion_tokens)`` provides an ordered scan
   on (user_id, created_at) with all SUM columns in the index.

Revision ID: 050
Revises: 049
Create Date: 2026-04-14 00:00:00.000000

"""
from alembic import op

# revision identifiers, used by Alembic.
revision = '050'
down_revision = '049'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        'ix_requests_user_model',
        'requests',
        ['user_id', 'model'],
    )
    op.create_index(
        'ix_requests_user_created_tokens',
        'requests',
        ['user_id', 'created_at', 'total_tokens', 'prompt_tokens', 'completion_tokens'],
    )


def downgrade() -> None:
    op.drop_index('ix_requests_user_created_tokens', table_name='requests')
    op.drop_index('ix_requests_user_model', table_name='requests')
