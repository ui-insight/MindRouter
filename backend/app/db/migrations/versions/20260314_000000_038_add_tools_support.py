############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# 038_add_tools_support.py: Add supports_tools and
#     tools_override columns to models table.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Add supports_tools and tools_override to models.

Revision ID: 038
Revises: 037
"""

from alembic import op
import sqlalchemy as sa

revision = "038"
down_revision = "037"


def upgrade() -> None:
    op.add_column(
        "models",
        sa.Column("supports_tools", sa.Boolean(), nullable=False, server_default=sa.text("0")),
    )
    op.add_column(
        "models",
        sa.Column("tools_override", sa.Boolean(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("models", "tools_override")
    op.drop_column("models", "supports_tools")
