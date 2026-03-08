############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# 030_add_model_description_cache.py: Create the
#     model_description_cache table for persisting enriched
#     descriptions across model row deletions.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Add model_description_cache table.

Revision ID: 030
Revises: 029
Create Date: 2026-03-08
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic
revision = "030"
down_revision = "029"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "model_description_cache",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(200), unique=True, nullable=False),
        sa.Column("description", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("model_description_cache")
