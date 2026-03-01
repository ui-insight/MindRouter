"""Remove token_budget from quotas table — budget now comes from group.

Revision ID: 023
Revises: 022
Create Date: 2026-02-20
"""

from alembic import op
import sqlalchemy as sa


revision = "023"
down_revision = "022"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_column("quotas", "token_budget")


def downgrade() -> None:
    op.add_column(
        "quotas",
        sa.Column("token_budget", sa.BigInteger(), nullable=False, server_default="100000"),
    )
    # Backfill from user's group
    op.execute(
        "UPDATE quotas q JOIN users u ON q.user_id = u.id "
        "JOIN groups g ON u.group_id = g.id "
        "SET q.token_budget = g.token_budget"
    )
