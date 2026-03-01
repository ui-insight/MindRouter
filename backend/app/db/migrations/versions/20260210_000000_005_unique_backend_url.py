############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# 005_unique_backend_url.py: Add unique constraint on backend URL
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Add unique constraint on backends.url to prevent duplicate endpoint registration

Revision ID: 005
Revises: 004
Create Date: 2026-02-10 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '005'
down_revision: Union[str, None] = '004'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_unique_constraint("uq_backends_url", "backends", ["url"])


def downgrade() -> None:
    op.drop_constraint("uq_backends_url", "backends", type_="unique")
