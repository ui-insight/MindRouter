############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# 017_add_model_max_context.py: Add model_max_context column
#     to track architectural maximum context length separately
#     from the effective (num_ctx) context length.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Add model_max_context column to models table.

Revision ID: 017
Revises: 016
Create Date: 2026-02-19
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic
revision = "017"
down_revision = "016"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("models", sa.Column("model_max_context", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("models", "model_max_context")
