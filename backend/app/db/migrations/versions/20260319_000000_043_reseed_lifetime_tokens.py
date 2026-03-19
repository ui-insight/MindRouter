############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# 043_reseed_lifetime_tokens.py: Fix lifetime_tokens_used
#     that was seeded too low in migration 042
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Re-seed lifetime_tokens_used to max of current value, SUM(requests), and tokens_used

Revision ID: 043
Revises: 042
Create Date: 2026-03-19 00:00:00.000000

"""

from alembic import op

# revision identifiers, used by Alembic
revision = "043"
down_revision = "042"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Set lifetime_tokens_used to the maximum of:
    #   1. Its current value (may be correct for users active since migration 042)
    #   2. SUM(requests.total_tokens) (authoritative if retention hasn't purged yet)
    #   3. tokens_used (current period usage, which should never exceed lifetime)
    #
    # This fixes users whose lifetime counter was seeded too low in 042
    # because retention had already purged some of their request history.

    # First: update from SUM(requests) where it's higher
    op.execute("""
        UPDATE quotas q
        INNER JOIN (
            SELECT user_id, COALESCE(SUM(total_tokens), 0) AS total
            FROM requests
            WHERE total_tokens IS NOT NULL
            GROUP BY user_id
        ) r ON q.user_id = r.user_id
        SET q.lifetime_tokens_used = r.total
        WHERE r.total > q.lifetime_tokens_used
    """)

    # Second: ensure lifetime is never less than current period tokens_used
    op.execute("""
        UPDATE quotas
        SET lifetime_tokens_used = tokens_used
        WHERE tokens_used > lifetime_tokens_used
    """)


def downgrade() -> None:
    # No downgrade needed — the values are only more correct now
    pass
