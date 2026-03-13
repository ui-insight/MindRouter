############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# 036_add_user_agreement.py: Add agreement tracking columns
#     to users table and seed AppConfig with default agreement.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Add user agreement tracking.

Revision ID: 036
Revises: 035
"""

import json

from alembic import op
import sqlalchemy as sa

revision = "036"
down_revision = "035"

_DEFAULT_AGREEMENT = (
    "<h4>University of Idaho &mdash; MindRouter Acceptable Use Agreement</h4>"
    "<p>By using MindRouter, you agree to the following terms:</p>"
    "<ul>"
    "<li>You will use this service for legitimate academic, research, "
    "and institutional purposes only.</li>"
    "<li>You will not submit private, confidential, or export-controlled "
    "data to language models without appropriate authorization.</li>"
    "<li>You will not attempt to circumvent rate limits, quotas, "
    "or access controls.</li>"
    "<li>You understand that prompts and responses may be logged for "
    "auditing, security, and quality purposes.</li>"
    "<li>You will comply with all applicable University of Idaho policies "
    "and applicable laws.</li>"
    "</ul>"
    "<p>This service is provided as-is. The University of Idaho reserves "
    "the right to modify or revoke access at any time.</p>"
)


def upgrade() -> None:
    op.add_column("users", sa.Column("agreement_version_accepted", sa.Integer(), nullable=True))
    op.add_column("users", sa.Column("agreement_accepted_at", sa.DateTime(timezone=True), nullable=True))

    # Seed default agreement config (values are JSON-encoded per AppConfig convention)
    app_config = sa.table(
        "app_config",
        sa.column("key", sa.String),
        sa.column("value", sa.Text),
        sa.column("description", sa.String),
    )
    op.execute(
        app_config.insert().prefix_with("IGNORE").values(
            key="agreement.version",
            value=json.dumps(1),
            description="Current use agreement version number",
        )
    )
    op.execute(
        app_config.insert().prefix_with("IGNORE").values(
            key="agreement.text",
            value=json.dumps(_DEFAULT_AGREEMENT),
            description="Use agreement HTML content",
        )
    )


def downgrade() -> None:
    op.drop_column("users", "agreement_accepted_at")
    op.drop_column("users", "agreement_version_accepted")
    op.execute(
        sa.text("DELETE FROM app_config WHERE `key` IN ('agreement.version', 'agreement.text')")
    )
