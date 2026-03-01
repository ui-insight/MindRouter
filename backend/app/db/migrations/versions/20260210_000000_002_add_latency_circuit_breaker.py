############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# 002_add_latency_circuit_breaker.py: Add latency EMA and circuit breaker columns
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Add latency EMA and circuit breaker columns to backends

Revision ID: 002
Revises: 001
Create Date: 2026-02-10 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '002'
down_revision: Union[str, None] = '001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('backends', sa.Column('latency_ema_ms', sa.Float(), nullable=True))
    op.add_column('backends', sa.Column('ttft_ema_ms', sa.Float(), nullable=True))
    op.add_column('backends', sa.Column('live_failure_count', sa.Integer(), nullable=False, server_default='0'))
    op.add_column('backends', sa.Column('circuit_open_until', sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column('backends', 'circuit_open_until')
    op.drop_column('backends', 'live_failure_count')
    op.drop_column('backends', 'ttft_ema_ms')
    op.drop_column('backends', 'latency_ema_ms')
