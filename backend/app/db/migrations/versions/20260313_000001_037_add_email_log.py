############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# 037_add_email_log.py: Add email_log table for tracking
#     sent emails and seed SMTP configuration defaults.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Add email_log table and SMTP config defaults.

Revision ID: 037
Revises: 036
"""

import json

from alembic import op
import sqlalchemy as sa

revision = "037"
down_revision = "036"


def upgrade() -> None:
    op.create_table(
        "email_log",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("subject", sa.String(500), nullable=False),
        sa.Column("body_preview", sa.Text, nullable=True),
        sa.Column("recipient_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("success_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("fail_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("sent_by", sa.Integer, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("blog_post_id", sa.Integer, sa.ForeignKey("blog_posts.id"), nullable=True),
        sa.Column(
            "status",
            sa.Enum("pending", "sending", "completed", "failed", name="email_status"),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_email_log_created", "email_log", ["created_at"])

    # Seed SMTP config defaults
    app_config = sa.table(
        "app_config",
        sa.column("key", sa.String),
        sa.column("value", sa.Text),
        sa.column("description", sa.String),
    )
    defaults = [
        ("email.smtp_host", "", "SMTP server hostname"),
        ("email.smtp_port", 587, "SMTP server port"),
        ("email.smtp_username", "", "SMTP authentication username"),
        ("email.smtp_password", "", "SMTP authentication password"),
        ("email.use_tls", True, "Use STARTTLS for SMTP connection"),
        ("email.default_sender", "", "Default From address for emails"),
        ("email.test_address", "", "Test email recipient address"),
        ("email.blog_sender", "", "From address for blog notification emails"),
    ]
    for key, value, desc in defaults:
        op.execute(
            app_config.insert().prefix_with("IGNORE").values(
                key=key, value=json.dumps(value), description=desc,
            )
        )


def downgrade() -> None:
    op.drop_table("email_log")
    op.execute(
        sa.text("DELETE FROM app_config WHERE `key` LIKE 'email.%'")
    )
