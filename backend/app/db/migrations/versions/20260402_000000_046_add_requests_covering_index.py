############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# 046_add_requests_covering_index.py: Add covering index
#     for the token throughput query that was causing
#     MariaDB to do full table scans on 800K+ rows
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Add covering index on requests(status, completed_at, total_tokens)

The token throughput query (SUM(total_tokens) WHERE status='completed'
AND completed_at >= ...) was scanning the full requests table on every
dashboard load and API call. This covering index lets MariaDB answer
the query entirely from the index.

Revision ID: 046
Revises: 045
Create Date: 2026-04-02 00:00:00.000000

"""
from alembic import op

# revision identifiers, used by Alembic.
revision = '046'
down_revision = '045'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        'ix_requests_status_completed',
        'requests',
        ['status', 'completed_at', 'total_tokens'],
    )


def downgrade() -> None:
    op.drop_index('ix_requests_status_completed', table_name='requests')
