############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# 031_drop_usage_ledger.py: Remove the redundant usage_ledger
#     table.  All token usage queries now read from the
#     requests table directly.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Drop usage_ledger table (redundant with requests).

Revision ID: 031
Revises: 030
"""

from alembic import op

revision = "031"
down_revision = "030"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # MariaDB requires FK constraints to be dropped before their backing
    # indexes can be removed.  DDL is non-transactional on MariaDB, so if
    # this migration fails midway, manual cleanup may be required.

    # 1. Drop foreign key constraints on usage_ledger
    op.drop_constraint("usage_ledger_ibfk_1", "usage_ledger", type_="foreignkey")
    op.drop_constraint("usage_ledger_ibfk_2", "usage_ledger", type_="foreignkey")
    op.drop_constraint("usage_ledger_ibfk_3", "usage_ledger", type_="foreignkey")
    op.drop_constraint("usage_ledger_ibfk_4", "usage_ledger", type_="foreignkey")

    # 2. Drop the table (indexes are dropped automatically with the table)
    op.drop_table("usage_ledger")

    # 3. Drop archived_usage_ledger if it exists (archive DB may share
    #    the same database or be separate; ignore errors if it doesn't exist)
    op.execute("DROP TABLE IF EXISTS archived_usage_ledger")


def downgrade() -> None:
    # Recreate usage_ledger table
    op.execute("""
        CREATE TABLE usage_ledger (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            api_key_id INT NOT NULL,
            request_id BIGINT NOT NULL,
            prompt_tokens INT NOT NULL DEFAULT 0,
            completion_tokens INT NOT NULL DEFAULT 0,
            total_tokens INT NOT NULL DEFAULT 0,
            is_estimated TINYINT(1) NOT NULL DEFAULT 0,
            model VARCHAR(100) NOT NULL,
            backend_id INT NULL,
            created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
            updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
            INDEX ix_usage_ledger_user_id (user_id),
            INDEX ix_usage_ledger_user_created (user_id, created_at),
            CONSTRAINT usage_ledger_ibfk_1 FOREIGN KEY (user_id) REFERENCES users(id),
            CONSTRAINT usage_ledger_ibfk_2 FOREIGN KEY (api_key_id) REFERENCES api_keys(id),
            CONSTRAINT usage_ledger_ibfk_3 FOREIGN KEY (request_id) REFERENCES requests(id),
            CONSTRAINT usage_ledger_ibfk_4 FOREIGN KEY (backend_id) REFERENCES backends(id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)
