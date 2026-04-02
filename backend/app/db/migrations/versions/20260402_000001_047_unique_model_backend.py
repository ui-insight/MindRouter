############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# 047_unique_model_backend.py: Add unique constraint on
#     models(backend_id, name) to prevent duplicate model
#     entries from concurrent discovery cycles
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Add unique constraint on models(backend_id, name)

Prevents duplicate model rows from concurrent discovery cycles.
Also cleans up any existing duplicates before adding the constraint.

Revision ID: 047
Revises: 046
Create Date: 2026-04-02 00:00:01.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '047'
down_revision = '046'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Clean up duplicates first (keep lowest ID per backend_id+name)
    conn = op.get_bind()
    conn.execute(sa.text("""
        DELETE m FROM models m
        INNER JOIN (
            SELECT name, backend_id, MIN(id) AS keep_id
            FROM models
            GROUP BY name, backend_id
        ) keeper ON m.name = keeper.name AND m.backend_id = keeper.backend_id
        WHERE m.id > keeper.keep_id
    """))

    op.create_unique_constraint(
        'uq_models_backend_name',
        'models',
        ['backend_id', 'name'],
    )


def downgrade() -> None:
    op.drop_constraint('uq_models_backend_name', 'models', type_='unique')
