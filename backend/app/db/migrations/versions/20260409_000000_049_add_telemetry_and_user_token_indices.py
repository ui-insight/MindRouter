############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# 049_add_telemetry_and_user_token_indices.py: Add covering
#     indices for the cluster energy history query and the
#     per-user token totals query that were doing full table
#     scans on the multi-million-row node_telemetry and
#     requests tables.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Add covering indices for energy and user token aggregation queries

Two slow queries identified in production:

1. ``get_cluster_power_history`` (Admin -> Energy page) filters
   ``node_telemetry`` by ``timestamp`` only.  The existing
   ``(node_id, timestamp)`` index is unusable for this access
   pattern, so MariaDB scans all 7.6M rows.  A covering index on
   ``(timestamp, node_id, server_power_watts, gpu_power_watts)``
   lets the query be answered entirely from the index.

2. ``get_user_token_totals`` (user dashboard) sums prompt /
   completion / total tokens for a user.  The existing
   ``(user_id, created_at)`` index narrows the scan but still
   requires a heap read for every row to fetch the token columns
   and check ``total_tokens IS NOT NULL``.  For high-volume users
   (>500K requests) this took ~67s.  A covering index on
   ``(user_id, total_tokens, prompt_tokens, completion_tokens)``
   answers the query from the index alone.

Revision ID: 049
Revises: 048
Create Date: 2026-04-09 00:00:00.000000

"""
from alembic import op

# revision identifiers, used by Alembic.
revision = '049'
down_revision = '048'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        'ix_node_telemetry_time_covering',
        'node_telemetry',
        ['timestamp', 'node_id', 'server_power_watts', 'gpu_power_watts'],
    )
    op.create_index(
        'ix_requests_user_tokens_covering',
        'requests',
        ['user_id', 'total_tokens', 'prompt_tokens', 'completion_tokens'],
    )


def downgrade() -> None:
    op.drop_index('ix_requests_user_tokens_covering', table_name='requests')
    op.drop_index('ix_node_telemetry_time_covering', table_name='node_telemetry')
