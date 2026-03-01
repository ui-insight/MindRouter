############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# 009_add_groups.py: Add groups table, user group membership,
#                    and profile fields
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Add groups table and user profile fields

Revision ID: 009
Revises: 008
Create Date: 2026-02-13 00:00:00.000000

"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic
revision = "009"
down_revision = "008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Create groups table
    op.create_table(
        "groups",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(50), unique=True, nullable=False),
        sa.Column("display_name", sa.String(100), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("token_budget", sa.BigInteger(), nullable=False, server_default="100000"),
        sa.Column("rpm_limit", sa.Integer(), nullable=False, server_default="30"),
        sa.Column("max_concurrent", sa.Integer(), nullable=False, server_default="2"),
        sa.Column("scheduler_weight", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("is_admin", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # 2. Insert 7 default groups
    groups_table = sa.table(
        "groups",
        sa.column("name", sa.String),
        sa.column("display_name", sa.String),
        sa.column("description", sa.Text),
        sa.column("token_budget", sa.BigInteger),
        sa.column("rpm_limit", sa.Integer),
        sa.column("max_concurrent", sa.Integer),
        sa.column("scheduler_weight", sa.Integer),
        sa.column("is_admin", sa.Boolean),
    )

    op.bulk_insert(groups_table, [
        {
            "name": "students",
            "display_name": "Students",
            "description": "University students",
            "token_budget": 100000,
            "rpm_limit": 30,
            "max_concurrent": 2,
            "scheduler_weight": 1,
            "is_admin": False,
        },
        {
            "name": "staff",
            "display_name": "Staff",
            "description": "University staff",
            "token_budget": 500000,
            "rpm_limit": 60,
            "max_concurrent": 4,
            "scheduler_weight": 2,
            "is_admin": False,
        },
        {
            "name": "faculty",
            "display_name": "Faculty",
            "description": "University faculty members",
            "token_budget": 1000000,
            "rpm_limit": 120,
            "max_concurrent": 8,
            "scheduler_weight": 3,
            "is_admin": False,
        },
        {
            "name": "researchers",
            "display_name": "Researchers",
            "description": "Research group members",
            "token_budget": 1000000,
            "rpm_limit": 120,
            "max_concurrent": 8,
            "scheduler_weight": 3,
            "is_admin": False,
        },
        {
            "name": "admin",
            "display_name": "Admin",
            "description": "System administrators",
            "token_budget": 10000000,
            "rpm_limit": 1000,
            "max_concurrent": 50,
            "scheduler_weight": 10,
            "is_admin": True,
        },
        {
            "name": "nerds",
            "display_name": "Nerds",
            "description": "Power users and enthusiasts",
            "token_budget": 500000,
            "rpm_limit": 60,
            "max_concurrent": 4,
            "scheduler_weight": 2,
            "is_admin": False,
        },
        {
            "name": "other",
            "display_name": "Other",
            "description": "General users",
            "token_budget": 100000,
            "rpm_limit": 30,
            "max_concurrent": 2,
            "scheduler_weight": 1,
            "is_admin": False,
        },
    ])

    # 3. Add new columns to users table (group_id nullable initially)
    op.add_column("users", sa.Column("group_id", sa.Integer(), nullable=True))
    op.add_column("users", sa.Column("college", sa.String(255), nullable=True))
    op.add_column("users", sa.Column("department", sa.String(255), nullable=True))
    op.add_column("users", sa.Column("intended_use", sa.Text(), nullable=True))

    # 4. Map existing users from role to group_id
    # Use raw SQL for the data migration since we need subqueries
    op.execute("""
        UPDATE users SET group_id = (
            SELECT id FROM groups WHERE groups.name = CASE users.role
                WHEN 'student' THEN 'students'
                WHEN 'staff' THEN 'staff'
                WHEN 'faculty' THEN 'faculty'
                WHEN 'admin' THEN 'admin'
                ELSE 'other'
            END
        )
    """)

    # 5. Make group_id NOT NULL now that all rows are populated
    op.alter_column("users", "group_id", existing_type=sa.Integer(), nullable=False)

    # 6. Add FK constraint and index
    op.create_foreign_key(
        "fk_users_group_id", "users", "groups", ["group_id"], ["id"]
    )
    op.create_index("ix_users_group_active", "users", ["group_id", "is_active"])


def downgrade() -> None:
    # Drop index first, then FK, then columns, then table
    op.drop_index("ix_users_group_active", table_name="users")
    op.drop_constraint("fk_users_group_id", "users", type_="foreignkey")
    op.drop_column("users", "intended_use")
    op.drop_column("users", "department")
    op.drop_column("users", "college")
    op.drop_column("users", "group_id")
    op.drop_table("groups")
