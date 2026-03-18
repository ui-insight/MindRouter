############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# 042_add_lifetime_tokens.py: Add lifetime_tokens_used to quotas
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Add lifetime_tokens_used column to quotas table

Revision ID: 042
Revises: 041
Create Date: 2026-03-18 00:00:00.000000

"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic
revision = "042"
down_revision = "041"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add lifetime_tokens_used column (defaults to 0 for all existing rows)
    op.add_column(
        "quotas",
        sa.Column("lifetime_tokens_used", sa.BigInteger(), nullable=False, server_default="0"),
    )

    # Seed from existing request data so current users don't start at 0.
    # This uses the same SUM(total_tokens) that was previously computed
    # dynamically. After this migration, the column is the source of truth.
    op.execute("""
        UPDATE quotas q
        INNER JOIN (
            SELECT user_id, COALESCE(SUM(total_tokens), 0) AS total
            FROM requests
            WHERE total_tokens IS NOT NULL
            GROUP BY user_id
        ) r ON q.user_id = r.user_id
        SET q.lifetime_tokens_used = r.total
    """)


def downgrade() -> None:
    op.drop_column("quotas", "lifetime_tokens_used")
