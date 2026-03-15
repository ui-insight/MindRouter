############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# 040_add_registry_meta.py: Single-row table tracking
#     registry mutation version for cross-worker sync.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Add registry_meta table for cross-worker synchronization.

Revision ID: 040
Revises: 039
"""

from alembic import op
import sqlalchemy as sa

revision = "040"
down_revision = "039"


def upgrade() -> None:
    op.create_table(
        "registry_meta",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("version", sa.Integer, nullable=False, server_default="1"),
    )
    op.execute("INSERT INTO registry_meta (id, version) VALUES (1, 1)")


def downgrade() -> None:
    op.drop_table("registry_meta")
