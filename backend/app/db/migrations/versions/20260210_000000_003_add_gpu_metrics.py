############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# 003_add_gpu_metrics.py: Add GPU device tables and sidecar columns
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Add GPU metrics tables and backend sidecar columns

Revision ID: 003
Revises: 002
Create Date: 2026-02-10 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '003'
down_revision: Union[str, None] = '002'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add columns to backends
    op.add_column('backends', sa.Column('sidecar_url', sa.String(500), nullable=True))
    op.add_column('backends', sa.Column('gpu_count', sa.Integer(), nullable=True))
    op.add_column('backends', sa.Column('driver_version', sa.String(50), nullable=True))
    op.add_column('backends', sa.Column('cuda_version', sa.String(20), nullable=True))

    # Add columns to backend_telemetry
    op.add_column('backend_telemetry', sa.Column('gpu_power_draw_watts', sa.Float(), nullable=True))
    op.add_column('backend_telemetry', sa.Column('gpu_fan_speed_percent', sa.Float(), nullable=True))

    # Create gpu_devices table
    op.create_table(
        'gpu_devices',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('backend_id', sa.Integer(), nullable=False),
        sa.Column('gpu_index', sa.Integer(), nullable=False),
        sa.Column('uuid', sa.String(100), nullable=True),
        sa.Column('name', sa.String(200), nullable=True),
        sa.Column('pci_bus_id', sa.String(50), nullable=True),
        sa.Column('compute_capability', sa.String(10), nullable=True),
        sa.Column('memory_total_gb', sa.Float(), nullable=True),
        sa.Column('power_limit_watts', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['backend_id'], ['backends.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_gpu_devices_backend_index', 'gpu_devices', ['backend_id', 'gpu_index'], unique=True)

    # Create gpu_device_telemetry table
    op.create_table(
        'gpu_device_telemetry',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('gpu_device_id', sa.Integer(), nullable=False),
        sa.Column('backend_id', sa.Integer(), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('utilization_gpu', sa.Float(), nullable=True),
        sa.Column('utilization_memory', sa.Float(), nullable=True),
        sa.Column('memory_used_gb', sa.Float(), nullable=True),
        sa.Column('memory_free_gb', sa.Float(), nullable=True),
        sa.Column('temperature_gpu', sa.Float(), nullable=True),
        sa.Column('temperature_memory', sa.Float(), nullable=True),
        sa.Column('power_draw_watts', sa.Float(), nullable=True),
        sa.Column('fan_speed_percent', sa.Float(), nullable=True),
        sa.Column('clock_sm_mhz', sa.Integer(), nullable=True),
        sa.Column('clock_memory_mhz', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['gpu_device_id'], ['gpu_devices.id']),
        sa.ForeignKeyConstraint(['backend_id'], ['backends.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        'ix_gpu_device_telemetry_device_time', 'gpu_device_telemetry',
        ['gpu_device_id', 'timestamp'],
    )
    op.create_index(
        'ix_gpu_device_telemetry_backend_time', 'gpu_device_telemetry',
        ['backend_id', 'timestamp'],
    )


def downgrade() -> None:
    op.drop_table('gpu_device_telemetry')
    op.drop_table('gpu_devices')
    op.drop_column('backend_telemetry', 'gpu_fan_speed_percent')
    op.drop_column('backend_telemetry', 'gpu_power_draw_watts')
    op.drop_column('backends', 'cuda_version')
    op.drop_column('backends', 'driver_version')
    op.drop_column('backends', 'gpu_count')
    op.drop_column('backends', 'sidecar_url')
