############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# 014_add_model_quantization.py: Add quantization column
#     to models table for auto-detected quantization level
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Add quantization to models table.

Revision ID: 014
Revises: 013
Create Date: 2026-02-19
"""

from alembic import op
import sqlalchemy as sa

revision = "014"
down_revision = "013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("models", sa.Column("quantization", sa.String(50), nullable=True))


def downgrade() -> None:
    op.drop_column("models", "quantization")
