############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# 087_add_decision_backend_engine.py: Add the 'decision'
#     backend engine type so a System One decision server
#     (Clef behind clef_service, Laya) can be registered as a
#     backend: health-polled, shown on the backends page and
#     charted through the node's GPU sidecar, while serving NO
#     chat models.
#
# Mirrors 077 (dlp engine).
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Add decision backend engine type.

Revision ID: 087
Revises: 086

MariaDB / OPS NOTES (DDL here is NON-TRANSACTIONAL):

  1. `backends` has tens of rows; this APPENDS one value to the end of a
     7-value ENUM, so the column stays 1 byte and no row rewrite is needed.
     ALGORITHM=INSTANT, LOCK=NONE is requested explicitly. FALLBACK: if the
     server rejects INSTANT, re-run the statement without the ALGORITHM/LOCK
     clause.
  2. The downgrade NARROWS the enum and would corrupt surviving rows, so it
     REFUSES to run while any backend is still engine='decision'.
"""

from alembic import op

revision = "087"
down_revision = "086"
branch_labels = None
depends_on = None

# Spelled out in full: never rely on the ORM enum at migration time.
OLD_ENGINE = "'ollama','vllm','diffusion','video','tts','stt','dlp'"
NEW_ENGINE = "'ollama','vllm','diffusion','video','tts','stt','dlp','decision'"


def upgrade() -> None:
    op.execute(
        f"ALTER TABLE backends MODIFY COLUMN engine "
        f"ENUM({NEW_ENGINE}) NOT NULL, ALGORITHM=INSTANT, LOCK=NONE"
    )


def downgrade() -> None:
    bind = op.get_bind()
    remaining = bind.exec_driver_sql(
        "SELECT COUNT(*) FROM backends WHERE engine = 'decision'"
    ).scalar()
    if remaining:
        raise RuntimeError(
            f"Cannot downgrade: {remaining} backend row(s) still use "
            f"engine='decision'. Delete or re-engine them first "
            f"(SELECT id, name FROM backends WHERE engine='decision';)."
        )

    op.execute(
        f"ALTER TABLE backends MODIFY COLUMN engine "
        f"ENUM({OLD_ENGINE}) NOT NULL"
    )
