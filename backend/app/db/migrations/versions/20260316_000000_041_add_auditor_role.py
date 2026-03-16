############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# 041_add_auditor_role.py: Add is_auditor flag to groups
#                          and seed an auditor group
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Add is_auditor column to groups table and seed auditor group

Revision ID: 041
Revises: 040
Create Date: 2026-03-16 00:00:00.000000

"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic
revision = "041"
down_revision = "040"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add is_auditor column (defaults to False for all existing groups)
    op.add_column(
        "groups",
        sa.Column("is_auditor", sa.Boolean(), nullable=False, server_default="0"),
    )

    # Seed an auditor group
    op.execute("""
        INSERT INTO groups (name, display_name, description, token_budget, rpm_limit,
                            max_concurrent, scheduler_weight, is_admin, is_auditor,
                            api_key_expiry_days, max_api_keys)
        VALUES ('auditor', 'Auditor', 'Read-only admin access for oversight and compliance',
                100000, 30, 2, 1, 0, 1, 45, 8)
    """)


def downgrade() -> None:
    # Remove the seeded auditor group (only if it still has the auditor flag)
    op.execute("DELETE FROM groups WHERE name = 'auditor' AND is_auditor = 1")
    op.drop_column("groups", "is_auditor")
