############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# 018_add_thinking_description_url.py: Add supports_thinking,
#     thinking_override, description, and model_url columns
#     to the models table.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Add thinking, description, model_url columns to models table.

Revision ID: 018
Revises: 017
Create Date: 2026-02-19
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic
revision = "018"
down_revision = "017"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("models", sa.Column("supports_thinking", sa.Boolean(), nullable=False, server_default=sa.text("0")))
    op.add_column("models", sa.Column("thinking_override", sa.Boolean(), nullable=True))
    op.add_column("models", sa.Column("description", sa.Text(), nullable=True))
    op.add_column("models", sa.Column("model_url", sa.String(500), nullable=True))


def downgrade() -> None:
    op.drop_column("models", "model_url")
    op.drop_column("models", "description")
    op.drop_column("models", "thinking_override")
    op.drop_column("models", "supports_thinking")
