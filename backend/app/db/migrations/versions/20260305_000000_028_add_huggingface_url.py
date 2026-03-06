############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# 028_add_huggingface_url.py: Add huggingface_url column
#     to the models table.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Add huggingface_url column to models table.

Revision ID: 028
Revises: 027
Create Date: 2026-03-05
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic
revision = "028"
down_revision = "027"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("models", sa.Column("huggingface_url", sa.String(500), nullable=True))


def downgrade() -> None:
    op.drop_column("models", "huggingface_url")
