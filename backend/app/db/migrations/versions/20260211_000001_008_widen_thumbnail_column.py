############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# 008_widen_thumbnail_column.py: Widen chat_attachments
#                                thumbnail_base64 to MEDIUMTEXT
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Widen chat_attachments thumbnail_base64 to MEDIUMTEXT

Revision ID: 008
Revises: 007
Create Date: 2026-02-11 00:00:01.000000

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.mysql import MEDIUMTEXT

# revision identifiers, used by Alembic
revision = "008"
down_revision = "007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "chat_attachments",
        "thumbnail_base64",
        existing_type=sa.Text(),
        type_=MEDIUMTEXT(),
        existing_nullable=True,
    )


def downgrade() -> None:
    op.alter_column(
        "chat_attachments",
        "thumbnail_base64",
        existing_type=MEDIUMTEXT(),
        type_=sa.Text(),
        existing_nullable=True,
    )
