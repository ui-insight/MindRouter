############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# 016_add_model_metadata_overrides.py: Add admin-editable
#     override columns for model metadata
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Add admin override columns to models table.

Revision ID: 016
Revises: 015
Create Date: 2026-02-19
"""

from alembic import op
import sqlalchemy as sa

revision = "016"
down_revision = "015"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("models", sa.Column("context_length_override", sa.Integer(), nullable=True))
    op.add_column("models", sa.Column("embedding_length_override", sa.Integer(), nullable=True))
    op.add_column("models", sa.Column("head_count_override", sa.Integer(), nullable=True))
    op.add_column("models", sa.Column("layer_count_override", sa.Integer(), nullable=True))
    op.add_column("models", sa.Column("feed_forward_length_override", sa.Integer(), nullable=True))
    op.add_column("models", sa.Column("capabilities_override", sa.Text(), nullable=True))
    op.add_column("models", sa.Column("family_override", sa.String(100), nullable=True))
    op.add_column("models", sa.Column("parameter_count_override", sa.String(50), nullable=True))
    op.add_column("models", sa.Column("quantization_override", sa.String(50), nullable=True))
    op.add_column("models", sa.Column("model_format_override", sa.String(50), nullable=True))
    op.add_column("models", sa.Column("parent_model_override", sa.String(200), nullable=True))


def downgrade() -> None:
    op.drop_column("models", "parent_model_override")
    op.drop_column("models", "model_format_override")
    op.drop_column("models", "quantization_override")
    op.drop_column("models", "parameter_count_override")
    op.drop_column("models", "family_override")
    op.drop_column("models", "capabilities_override")
    op.drop_column("models", "feed_forward_length_override")
    op.drop_column("models", "layer_count_override")
    op.drop_column("models", "head_count_override")
    op.drop_column("models", "embedding_length_override")
    op.drop_column("models", "context_length_override")
