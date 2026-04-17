############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# 057_add_image_generation_and_diffusion.py: Add image_generation
#     modality and diffusion backend engine type.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Add image_generation modality and diffusion backend engine.

Revision ID: 057
Revises: 056
"""

from alembic import op

revision = "057"
down_revision = "056"
branch_labels = None
depends_on = None

NEW_MODALITY = "'chat','completion','embedding','multimodal','reranking','tts','stt','image_generation'"
OLD_MODALITY = "'chat','completion','embedding','multimodal','reranking','tts','stt'"


def upgrade():
    # Add image_generation to modality enum on requests table
    op.execute(
        f"ALTER TABLE requests MODIFY COLUMN modality "
        f"ENUM({NEW_MODALITY}) NOT NULL"
    )

    # Add image_generation to modality enum on models table
    op.execute(
        f"ALTER TABLE models MODIFY COLUMN modality "
        f"ENUM({NEW_MODALITY}) NOT NULL DEFAULT 'chat'"
    )

    # Add diffusion to backend engine enum on backends table
    op.execute(
        "ALTER TABLE backends MODIFY COLUMN engine "
        "ENUM('ollama','vllm','diffusion') NOT NULL"
    )


def downgrade():
    # Remove diffusion from backend engine enum
    op.execute(
        "ALTER TABLE backends MODIFY COLUMN engine "
        "ENUM('ollama','vllm') NOT NULL"
    )

    # Remove image_generation from models modality
    op.execute(
        f"ALTER TABLE models MODIFY COLUMN modality "
        f"ENUM({OLD_MODALITY}) NOT NULL DEFAULT 'chat'"
    )

    # Remove image_generation from requests modality
    op.execute(
        f"ALTER TABLE requests MODIFY COLUMN modality "
        f"ENUM({OLD_MODALITY}) NOT NULL"
    )
