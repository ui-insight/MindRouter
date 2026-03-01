############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# 013_add_sidecar_version_to_node.py: Add sidecar_version
#     column to nodes table for version alignment tracking
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Add sidecar_version to nodes table.

Revision ID: 013
Revises: 012
Create Date: 2026-02-19
"""

from alembic import op
import sqlalchemy as sa

revision = "013"
down_revision = "012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("nodes", sa.Column("sidecar_version", sa.String(50), nullable=True))


def downgrade() -> None:
    op.drop_column("nodes", "sidecar_version")
