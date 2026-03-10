############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# 035_add_node_telemetry.py: Add node_telemetry table for
#     time-series server power (IPMI DCMI) tracking.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Add node_telemetry table for time-series server power tracking.

Revision ID: 035
Revises: 034
"""

from alembic import op
import sqlalchemy as sa

revision = "035"
down_revision = "034"


def upgrade() -> None:
    op.create_table(
        "node_telemetry",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("node_id", sa.Integer(), sa.ForeignKey("nodes.id"), nullable=False),
        sa.Column(
            "timestamp",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("server_power_watts", sa.Integer(), nullable=True),
        sa.Column("gpu_power_watts", sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_node_telemetry_node_time",
        "node_telemetry",
        ["node_id", "timestamp"],
    )


def downgrade() -> None:
    op.drop_index("ix_node_telemetry_node_time", table_name="node_telemetry")
    op.drop_table("node_telemetry")
