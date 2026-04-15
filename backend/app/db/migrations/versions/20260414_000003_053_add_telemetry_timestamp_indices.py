############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# 053_add_telemetry_timestamp_indices.py: Add standalone
#     timestamp indices on backend_telemetry and
#     gpu_device_telemetry for efficient retention DELETEs.
#     node_telemetry already has timestamp as the leading
#     column in ix_node_telemetry_time_covering (migration 049).
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Add standalone timestamp indices on telemetry tables for retention.

Revision ID: 053
Revises: 052
Create Date: 2026-04-14
"""

from alembic import op

revision = "053"
down_revision = "052"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        "ix_backend_telemetry_timestamp",
        "backend_telemetry",
        ["timestamp"],
    )
    op.create_index(
        "ix_gpu_device_telemetry_timestamp",
        "gpu_device_telemetry",
        ["timestamp"],
    )


def downgrade() -> None:
    op.drop_index("ix_gpu_device_telemetry_timestamp", table_name="gpu_device_telemetry")
    op.drop_index("ix_backend_telemetry_timestamp", table_name="backend_telemetry")
