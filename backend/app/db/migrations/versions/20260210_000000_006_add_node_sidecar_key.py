############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# 006_add_node_sidecar_key.py: Add sidecar_key column to nodes table
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Add sidecar_key column to nodes table for sidecar authentication

Revision ID: 006
Revises: 005
Create Date: 2026-02-10 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '006'
down_revision: Union[str, None] = '005'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('nodes', sa.Column('sidecar_key', sa.String(500), nullable=True))


def downgrade() -> None:
    op.drop_column('nodes', 'sidecar_key')
