############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# 048_add_queue_monitor_index.py: Add covering index for
#     queue monitor queries that scan requests by
#     created_at, model, and status
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Add index on requests(created_at, model, status) for queue monitor

The queue monitor endpoint queries requests within a time window
grouped by model and status. This index avoids full table scans
on the 24-hour window option.

Revision ID: 048
Revises: 047
Create Date: 2026-04-08 00:00:00.000000

"""
from alembic import op

# revision identifiers, used by Alembic.
revision = '048'
down_revision = '047'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        'ix_requests_created_model_status',
        'requests',
        ['created_at', 'model', 'status'],
    )


def downgrade() -> None:
    op.drop_index('ix_requests_created_model_status', table_name='requests')
