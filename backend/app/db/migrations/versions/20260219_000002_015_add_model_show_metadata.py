############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# 015_add_model_show_metadata.py: Add rich metadata columns
#     to models table from Ollama /api/show endpoint
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Add /api/show metadata columns to models table.

Revision ID: 015
Revises: 014
Create Date: 2026-02-19
"""

from alembic import op
import sqlalchemy as sa

revision = "015"
down_revision = "014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("models", sa.Column("model_format", sa.String(50), nullable=True))
    op.add_column("models", sa.Column("capabilities", sa.Text(), nullable=True))
    op.add_column("models", sa.Column("embedding_length", sa.Integer(), nullable=True))
    op.add_column("models", sa.Column("head_count", sa.Integer(), nullable=True))
    op.add_column("models", sa.Column("layer_count", sa.Integer(), nullable=True))
    op.add_column("models", sa.Column("feed_forward_length", sa.Integer(), nullable=True))
    op.add_column("models", sa.Column("parent_model", sa.String(200), nullable=True))


def downgrade() -> None:
    op.drop_column("models", "parent_model")
    op.drop_column("models", "feed_forward_length")
    op.drop_column("models", "layer_count")
    op.drop_column("models", "head_count")
    op.drop_column("models", "embedding_length")
    op.drop_column("models", "capabilities")
    op.drop_column("models", "model_format")
