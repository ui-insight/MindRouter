############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# 033_add_tts_stt_modality.py: Add TTS and STT values to
#     the modality enum column on the requests table.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Add TTS and STT to modality enum.

Revision ID: 033
Revises: 032
"""

from alembic import op

revision = "033"
down_revision = "032"
branch_labels = None
depends_on = None


def upgrade():
    op.execute(
        "ALTER TABLE requests MODIFY COLUMN modality "
        "ENUM('chat','completion','embedding','multimodal','reranking','tts','stt') "
        "NULL"
    )


def downgrade():
    op.execute(
        "ALTER TABLE requests MODIFY COLUMN modality "
        "ENUM('chat','completion','embedding','multimodal','reranking') "
        "NULL"
    )
