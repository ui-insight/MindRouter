############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# 039_add_admin_audit_log.py: Persistent audit log for all
#     admin actions (API and dashboard).
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Add admin_audit_log table.

Revision ID: 039
Revises: 038
"""

from alembic import op
import sqlalchemy as sa

revision = "039"
down_revision = "038"


def upgrade() -> None:
    op.create_table(
        "admin_audit_log",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("user_id", sa.Integer, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("action", sa.String(100), nullable=False),
        sa.Column("entity_type", sa.String(50), nullable=False),
        sa.Column("entity_id", sa.String(100), nullable=True),
        sa.Column("before_value", sa.JSON, nullable=True),
        sa.Column("after_value", sa.JSON, nullable=True),
        sa.Column("ip_address", sa.String(45), nullable=True),
        sa.Column("detail", sa.Text, nullable=True),
    )
    op.create_index("ix_admin_audit_timestamp", "admin_audit_log", ["timestamp"])
    op.create_index("ix_admin_audit_user_time", "admin_audit_log", ["user_id", "timestamp"])
    op.create_index("ix_admin_audit_entity", "admin_audit_log", ["entity_type", "entity_id"])
    op.create_index("ix_admin_audit_action", "admin_audit_log", ["action"])


def downgrade() -> None:
    op.drop_table("admin_audit_log")
