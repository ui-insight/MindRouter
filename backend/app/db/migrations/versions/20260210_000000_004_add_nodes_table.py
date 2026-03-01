############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# 004_add_nodes_table.py: Separate Node from Backend
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Separate Node (physical server) from Backend (inference endpoint)

Revision ID: 004
Revises: 003
Create Date: 2026-02-10 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '004'
down_revision: Union[str, None] = '003'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Create nodes table
    op.create_table(
        'nodes',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('hostname', sa.String(255), nullable=True),
        sa.Column('sidecar_url', sa.String(500), nullable=True),
        sa.Column('status', sa.Enum('online', 'offline', 'unknown', name='nodestatus'), nullable=False, server_default='unknown'),
        sa.Column('gpu_count', sa.Integer(), nullable=True),
        sa.Column('driver_version', sa.String(50), nullable=True),
        sa.Column('cuda_version', sa.String(20), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name'),
    )

    # 2. Auto-create Node rows from backends that have sidecar_url
    # Use raw SQL for the data migration
    conn = op.get_bind()
    backends_with_sidecar = conn.execute(
        sa.text("SELECT id, name, sidecar_url, gpu_count, driver_version, cuda_version FROM backends WHERE sidecar_url IS NOT NULL")
    ).fetchall()

    for b in backends_with_sidecar:
        node_name = f"node-{b[1]}"  # node-<backend_name>
        conn.execute(
            sa.text(
                "INSERT INTO nodes (name, sidecar_url, gpu_count, driver_version, cuda_version) "
                "VALUES (:name, :sidecar_url, :gpu_count, :driver_version, :cuda_version)"
            ),
            {
                "name": node_name,
                "sidecar_url": b[2],
                "gpu_count": b[3],
                "driver_version": b[4],
                "cuda_version": b[5],
            },
        )

    # 3. Add node_id and gpu_indices columns to backends
    op.add_column('backends', sa.Column('node_id', sa.Integer(), nullable=True))
    op.add_column('backends', sa.Column('gpu_indices', sa.JSON(), nullable=True))
    op.create_foreign_key('fk_backends_node_id', 'backends', 'nodes', ['node_id'], ['id'])

    # 4. Populate backends.node_id from auto-created nodes
    for b in backends_with_sidecar:
        node_name = f"node-{b[1]}"
        conn.execute(
            sa.text(
                "UPDATE backends SET node_id = (SELECT id FROM nodes WHERE name = :node_name) "
                "WHERE id = :backend_id"
            ),
            {"node_name": node_name, "backend_id": b[0]},
        )

    # 5. Add node_id column to gpu_devices, populate from backends
    op.add_column('gpu_devices', sa.Column('node_id', sa.Integer(), nullable=True))

    # Populate gpu_devices.node_id from gpu_devices.backend_id -> backends.node_id
    conn.execute(
        sa.text(
            "UPDATE gpu_devices SET node_id = "
            "(SELECT node_id FROM backends WHERE backends.id = gpu_devices.backend_id)"
        )
    )

    # 6. Drop old gpu_devices.backend_id FK and column, make node_id NOT NULL
    # Drop the FK constraint first (MariaDB won't drop an index backing an FK)
    op.drop_constraint('gpu_devices_ibfk_1', 'gpu_devices', type_='foreignkey')
    op.drop_index('ix_gpu_devices_backend_index', table_name='gpu_devices')
    op.drop_column('gpu_devices', 'backend_id')

    # Make node_id NOT NULL and add FK
    op.alter_column('gpu_devices', 'node_id', existing_type=sa.Integer(), nullable=False)
    op.create_foreign_key('fk_gpu_devices_node_id', 'gpu_devices', 'nodes', ['node_id'], ['id'])
    op.create_index('ix_gpu_devices_node_index', 'gpu_devices', ['node_id', 'gpu_index'], unique=True)

    # 7. Drop gpu_device_telemetry.backend_id column and index
    op.drop_constraint('gpu_device_telemetry_ibfk_2', 'gpu_device_telemetry', type_='foreignkey')
    op.drop_index('ix_gpu_device_telemetry_backend_time', table_name='gpu_device_telemetry')
    op.drop_column('gpu_device_telemetry', 'backend_id')

    # 8. Drop sidecar/hardware columns from backends
    op.drop_column('backends', 'sidecar_url')
    op.drop_column('backends', 'gpu_count')
    op.drop_column('backends', 'driver_version')
    op.drop_column('backends', 'cuda_version')


def downgrade() -> None:
    # Add back sidecar/hardware columns to backends
    op.add_column('backends', sa.Column('sidecar_url', sa.String(500), nullable=True))
    op.add_column('backends', sa.Column('gpu_count', sa.Integer(), nullable=True))
    op.add_column('backends', sa.Column('driver_version', sa.String(50), nullable=True))
    op.add_column('backends', sa.Column('cuda_version', sa.String(20), nullable=True))

    # Restore backend_id to gpu_device_telemetry
    op.add_column('gpu_device_telemetry', sa.Column('backend_id', sa.Integer(), nullable=True))
    op.create_foreign_key('gpu_device_telemetry_ibfk_2', 'gpu_device_telemetry', 'backends', ['backend_id'], ['id'])
    op.create_index('ix_gpu_device_telemetry_backend_time', 'gpu_device_telemetry', ['backend_id', 'timestamp'])

    # Restore backend_id to gpu_devices
    op.drop_index('ix_gpu_devices_node_index', table_name='gpu_devices')
    op.drop_constraint('fk_gpu_devices_node_id', 'gpu_devices', type_='foreignkey')
    op.add_column('gpu_devices', sa.Column('backend_id', sa.Integer(), nullable=True))
    op.create_foreign_key('gpu_devices_ibfk_1', 'gpu_devices', 'backends', ['backend_id'], ['id'])
    op.create_index('ix_gpu_devices_backend_index', 'gpu_devices', ['backend_id', 'gpu_index'], unique=True)
    op.drop_column('gpu_devices', 'node_id')

    # Remove node_id and gpu_indices from backends
    op.drop_constraint('fk_backends_node_id', 'backends', type_='foreignkey')
    op.drop_column('backends', 'gpu_indices')
    op.drop_column('backends', 'node_id')

    # Drop nodes table
    op.drop_table('nodes')
