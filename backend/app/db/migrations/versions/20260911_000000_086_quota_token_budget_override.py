############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# 086_quota_token_budget_override.py: per-user budget override
#     so an approved quota increase can actually be applied.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Per-user token budget override.

Revision ID: 086
Revises: 085

Migration 023 moved the token budget from `quotas.token_budget` to
`groups.token_budget`, but left the quota-increase request workflow in place.
That workflow is inherently per-user: a student asks for more tokens, an admin
approves a specific amount. With only a group-scoped budget there was nowhere
to put the granted number, so approval recorded `granted_tokens` in the audit
log and changed nothing a user could spend (GitHub issue #11). The only
"workaround" was raising the whole group's budget, which grants it to all 461
members of `other` at once.

This restores the per-user dimension WITHOUT undoing 023: the group budget
stays the default for everyone, and the override is the documented exception.

Semantics (kept identical to the group budget so there is one rule, not two):
  * NULL  -> inherit the group budget (every existing row; today's behaviour)
  * 0     -> unlimited, exactly as `groups.token_budget = 0` means today
  * n > 0 -> that many tokens per budget period

Additive and nullable, so the pre-migration container keeps working against
the new schema during a migrate-then-recreate deploy.
"""

import sqlalchemy as sa
from alembic import op

revision = "086"
down_revision = "085"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "quotas",
        sa.Column("token_budget_override", sa.BigInteger(), nullable=True),
    )


def downgrade() -> None:
    # Dropping the column restores group-only budgets. Any per-user grants are
    # lost, which is the same state as before this revision.
    op.drop_column("quotas", "token_budget_override")
