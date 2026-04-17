############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# 058_add_image_generation_enabled.py: Per-user image
#     generation access flag (default off).
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Add image_generation_enabled to users table.

Revision ID: 058
Revises: 057
"""

import sqlalchemy as sa
from alembic import op

revision = "058"
down_revision = "057"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "users",
        sa.Column(
            "image_generation_enabled",
            sa.Boolean(),
            nullable=False,
            server_default="0",
        ),
    )


def downgrade():
    op.drop_column("users", "image_generation_enabled")
