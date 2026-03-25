############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# 045_add_service_keys.py: Add service API key support
#     with promotion workflow and request table
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Add service key fields to api_keys and service_key_requests table

Revision ID: 045
Revises: 044
Create Date: 2026-03-25 00:00:00.000000

"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic
revision = "045"
down_revision = "044"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add service key columns to api_keys
    op.add_column("api_keys", sa.Column("is_service", sa.Boolean(), nullable=False, server_default=sa.text("0")))
    op.add_column("api_keys", sa.Column("service_name", sa.String(200), nullable=True))
    op.add_column("api_keys", sa.Column("service_contacts", sa.Text(), nullable=True))
    op.add_column("api_keys", sa.Column("data_risk_level", sa.String(20), nullable=True))
    op.add_column("api_keys", sa.Column("compliance_tags", sa.Text(), nullable=True))
    op.add_column("api_keys", sa.Column("promoted_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("api_keys", sa.Column("promoted_by", sa.Integer(), sa.ForeignKey("users.id"), nullable=True))
    op.add_column("api_keys", sa.Column("revocation_requested_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("api_keys", sa.Column("revocation_reason", sa.Text(), nullable=True))
    op.create_index("ix_api_keys_is_service", "api_keys", ["is_service"])

    # Create service_key_requests table
    op.create_table(
        "service_key_requests",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("api_key_id", sa.Integer(), sa.ForeignKey("api_keys.id"), nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("service_name", sa.String(200), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("alternative_contacts", sa.Text(), nullable=True),
        sa.Column("data_risk_level", sa.String(20), nullable=False, server_default="low"),
        sa.Column("compliance_tags", sa.Text(), nullable=True),
        sa.Column("compliance_other", sa.Text(), nullable=True),
        sa.Column(
            "status",
            sa.Enum("pending", "approved", "denied", name="servicekeyrequeststatusenum"),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("reviewed_by", sa.Integer(), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("review_notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
    )
    op.create_index("ix_service_key_requests_status", "service_key_requests", ["status"])


def downgrade() -> None:
    # Drop service_key_requests table
    op.drop_index("ix_service_key_requests_status", table_name="service_key_requests")
    op.drop_table("service_key_requests")
    # MariaDB may keep the ENUM type; that's fine

    # Drop api_keys service columns — find and drop FK constraint on promoted_by
    # before dropping the column (MariaDB error 1553 if FK backs an index)
    conn = op.get_bind()
    fk_rows = conn.execute(sa.text(
        "SELECT CONSTRAINT_NAME FROM information_schema.KEY_COLUMN_USAGE "
        "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'api_keys' "
        "AND COLUMN_NAME = 'promoted_by' AND REFERENCED_TABLE_NAME IS NOT NULL"
    )).fetchall()
    for row in fk_rows:
        op.drop_constraint(row[0], "api_keys", type_="foreignkey")

    op.drop_index("ix_api_keys_is_service", table_name="api_keys")
    op.drop_column("api_keys", "revocation_reason")
    op.drop_column("api_keys", "revocation_requested_at")
    op.drop_column("api_keys", "promoted_by")
    op.drop_column("api_keys", "promoted_at")
    op.drop_column("api_keys", "compliance_tags")
    op.drop_column("api_keys", "data_risk_level")
    op.drop_column("api_keys", "service_contacts")
    op.drop_column("api_keys", "service_name")
    op.drop_column("api_keys", "is_service")
