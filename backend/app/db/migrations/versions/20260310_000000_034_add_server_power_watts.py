############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# 034_add_server_power_watts.py: Add server_power_watts
#     column to the nodes table for IPMI DCMI power readings.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Add server_power_watts to nodes table.

Revision ID: 034
Revises: 033
"""

from alembic import op
import sqlalchemy as sa

revision = "034"
down_revision = "033"


def upgrade() -> None:
    op.add_column("nodes", sa.Column("server_power_watts", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("nodes", "server_power_watts")
