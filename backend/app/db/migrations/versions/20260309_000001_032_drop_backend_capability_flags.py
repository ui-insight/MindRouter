############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# 032_drop_backend_capability_flags.py: Remove redundant
#     backend-level capability flags (supports_multimodal,
#     supports_embeddings, supports_structured_output).
#     These are now derived from model-level data.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Drop redundant backend-level capability flags.

Revision ID: 032
Revises: 031
"""

from alembic import op
import sqlalchemy as sa

revision = "032"
down_revision = "031"
branch_labels = None
depends_on = None


def upgrade():
    op.drop_column("backends", "supports_multimodal")
    op.drop_column("backends", "supports_embeddings")
    op.drop_column("backends", "supports_structured_output")


def downgrade():
    op.add_column(
        "backends",
        sa.Column("supports_structured_output", sa.Boolean(), nullable=False, server_default=sa.text("1")),
    )
    op.add_column(
        "backends",
        sa.Column("supports_embeddings", sa.Boolean(), nullable=False, server_default=sa.text("0")),
    )
    op.add_column(
        "backends",
        sa.Column("supports_multimodal", sa.Boolean(), nullable=False, server_default=sa.text("0")),
    )
