############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# 051_add_user_detail_ip_and_apikey_indices.py: Add
#     covering indices for the remaining slow queries on the
#     admin user detail page (recent IPs and per-API-key
#     token totals).
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Add covering indices for recent-IPs and API-key token queries

Two more queries on the admin user detail page caused 90-100 second
loads for high-volume users:

1. Recent IPs query (GROUP BY client_ip, 100s): The existing
   (user_id, created_at) index filters rows but requires a heap
   read for ``client_ip`` on every row.  A covering index on
   ``(user_id, created_at, client_ip)`` resolves this.

2. API key token totals query (GROUP BY api_key_id, 89s): The
   existing single-column ``api_key_id`` index requires a heap
   read for the three token columns.  A covering index on
   ``(api_key_id, total_tokens, prompt_tokens, completion_tokens)``
   makes this an index-only scan.

Revision ID: 051
Revises: 050
Create Date: 2026-04-14 00:00:01.000000

"""
from alembic import op

# revision identifiers, used by Alembic.
revision = '051'
down_revision = '050'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        'ix_requests_user_created_ip',
        'requests',
        ['user_id', 'created_at', 'client_ip'],
    )
    op.create_index(
        'ix_requests_apikey_tokens_covering',
        'requests',
        ['api_key_id', 'total_tokens', 'prompt_tokens', 'completion_tokens'],
    )


def downgrade() -> None:
    op.drop_index('ix_requests_apikey_tokens_covering', table_name='requests')
    op.drop_index('ix_requests_user_created_ip', table_name='requests')
