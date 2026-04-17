############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# 056_drop_max_concurrent_from_quotas.py: Remove the
#     max_concurrent column from groups, quotas, and
#     api_keys.  RPM rate limiting is now enforced via
#     Redis; per-user concurrency limits are removed.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Drop max_concurrent columns from groups, quotas, and api_keys.

Revision ID: 056
Revises: 055
Create Date: 2026-04-17
"""

from alembic import op

revision = "056"
down_revision = "055"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_column("groups", "max_concurrent")
    op.drop_column("quotas", "max_concurrent")
    op.drop_column("api_keys", "max_concurrent")


def downgrade() -> None:
    import sqlalchemy as sa

    op.add_column(
        "groups",
        sa.Column("max_concurrent", sa.Integer(), nullable=False, server_default="2"),
    )
    op.add_column(
        "quotas",
        sa.Column("max_concurrent", sa.Integer(), nullable=False, server_default="2"),
    )
    op.add_column(
        "api_keys",
        sa.Column("max_concurrent", sa.Integer(), nullable=True),
    )
