"""Set unlimited token budgets for admin and nerds groups.

Revision ID: 022
Revises: 021
Create Date: 2026-02-20
"""

from alembic import op


revision = "022"
down_revision = "021"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Set admin and nerds groups to unlimited (token_budget=0)
    op.execute(
        "UPDATE groups SET token_budget = 0 WHERE name IN ('admin', 'nerds')"
    )
    # Update existing quotas for users in those groups
    op.execute(
        "UPDATE quotas SET token_budget = 0 "
        "WHERE user_id IN (SELECT id FROM users WHERE group_id IN "
        "(SELECT id FROM groups WHERE name IN ('admin', 'nerds')))"
    )


def downgrade() -> None:
    # Restore default budget for admin and nerds groups
    op.execute(
        "UPDATE groups SET token_budget = 100000 WHERE name IN ('admin', 'nerds')"
    )
    op.execute(
        "UPDATE quotas SET token_budget = 100000 "
        "WHERE user_id IN (SELECT id FROM users WHERE group_id IN "
        "(SELECT id FROM groups WHERE name IN ('admin', 'nerds')))"
    )
