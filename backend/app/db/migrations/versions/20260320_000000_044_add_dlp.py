############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# 044_add_dlp.py: Add DLP (Data Loss Prevention) alerts
#     table and seed default configuration
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Add dlp_alerts table and seed DLP configuration defaults

Revision ID: 044
Revises: 043
Create Date: 2026-03-20 00:00:00.000000

"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic
revision = "044"
down_revision = "043"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "dlp_alerts",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("request_id", sa.BigInteger, sa.ForeignKey("requests.id"), nullable=True),
        sa.Column("user_id", sa.Integer, sa.ForeignKey("users.id"), nullable=True),
        sa.Column("severity", sa.String(20), nullable=False),
        sa.Column("scanner", sa.String(20), nullable=False),
        sa.Column("categories", sa.JSON, nullable=True),
        sa.Column("entities", sa.JSON, nullable=True),
        sa.Column("confidence", sa.Float, nullable=True),
        sa.Column("scan_latency_ms", sa.Integer, nullable=True),
        sa.Column("scanned_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("acknowledged", sa.Boolean, nullable=False, server_default=sa.text("0")),
        sa.Column("acknowledged_by", sa.Integer, sa.ForeignKey("users.id"), nullable=True),
        sa.Column("acknowledged_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("detail", sa.Text, nullable=True),
    )

    op.create_index("ix_dlp_alerts_severity", "dlp_alerts", ["severity"])
    op.create_index("ix_dlp_alerts_user_time", "dlp_alerts", ["user_id", "scanned_at"])
    op.create_index("ix_dlp_alerts_request", "dlp_alerts", ["request_id"])
    op.create_index("ix_dlp_alerts_scanner", "dlp_alerts", ["scanner"])

    # Seed default DLP configuration
    app_config = sa.table(
        "app_config",
        sa.column("key", sa.String),
        sa.column("value", sa.Text),
        sa.column("description", sa.String),
    )
    op.bulk_insert(app_config, [
        {"key": "dlp.enabled", "value": "false", "description": "Master DLP scanning toggle"},
        {"key": "dlp.regex.enabled", "value": "true", "description": "Enable regex/keyword scanner"},
        {"key": "dlp.regex.patterns", "value": '[]', "description": "Custom regex patterns [{name, pattern, severity}]"},
        {"key": "dlp.regex.keywords", "value": '[]', "description": "Keyword list for simple matching"},
        {"key": "dlp.gliner.enabled", "value": "false", "description": "Enable GLiNER NER scanner"},
        {"key": "dlp.gliner.threshold", "value": "0.5", "description": "GLiNER confidence threshold (0.0-1.0)"},
        {"key": "dlp.gliner.categories", "value": '["person", "phone number", "email", "credit card number", "social security number", "date of birth", "driver license number", "passport number", "bank account number", "medical record number", "student id"]', "description": "GLiNER entity categories to detect"},
        {"key": "dlp.llm.enabled", "value": "false", "description": "Enable LLM contextual scanner"},
        {"key": "dlp.llm.model", "value": '"qwen/qwen3.5-4b"', "description": "Model name for LLM scanner (self-routed)"},
        {"key": "dlp.llm.system_prompt", "value": '"You are a data loss prevention analyst. Analyze the following text for sensitive data including PII (names, SSNs, emails, phone numbers), HIPAA/PHI (medical records, diagnoses), FERPA (student records, grades), PCI (credit card numbers), and CUI (controlled unclassified information). Respond with a JSON array of findings, each with: category, text, confidence (0-1). If no sensitive data found, respond with an empty array []."', "description": "System prompt for LLM scanner"},
        {"key": "dlp.severity_rules", "value": '{"social security number": "major", "credit card number": "major", "bank account number": "major", "medical record number": "major", "passport number": "major", "driver license number": "moderate", "person": "minor", "phone number": "minor", "email": "minor", "date of birth": "moderate", "student id": "moderate"}', "description": "Category to severity mapping"},
        {"key": "dlp.email.minor_recipients", "value": '""', "description": "Email recipients for minor alerts (comma-separated)"},
        {"key": "dlp.email.moderate_recipients", "value": '""', "description": "Email recipients for moderate alerts (comma-separated)"},
        {"key": "dlp.email.major_recipients", "value": '""', "description": "Email recipients for major alerts (comma-separated)"},
        {"key": "dlp.internal_api_key_id", "value": "null", "description": "API key ID used by DLP LLM scanner (auto-created)"},
    ])


def downgrade() -> None:
    op.drop_index("ix_dlp_alerts_scanner", table_name="dlp_alerts")
    op.drop_index("ix_dlp_alerts_request", table_name="dlp_alerts")
    op.drop_index("ix_dlp_alerts_user_time", table_name="dlp_alerts")
    op.drop_index("ix_dlp_alerts_severity", table_name="dlp_alerts")
    op.drop_table("dlp_alerts")

    # Remove DLP config keys
    op.execute("DELETE FROM app_config WHERE `key` LIKE 'dlp.%'")
