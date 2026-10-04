############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# 088_direct_login_and_email_audience.py: Record when an
#     account last signed in to MindRouter ITSELF (as opposed
#     to being created or refreshed by a registered app), so
#     bulk email can be sent to "everyone" or only to people
#     who use MindRouter directly; and record which audience
#     each bulk email went to.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Direct sign-in timestamp on users; audience on email_log.

Revision ID: 088
Revises: 087

users.last_direct_login_at is set by the web sign-in paths (local password,
Azure, generic SSO) and never by a registered app provisioning an account.
NULL means "this account exists only because an app created it".

`last_login_at` cannot serve: app provisioning shares the Azure find-or-create
and bumps it too. `group_classified` cannot either: it stays 0 after a direct
sign-in when the directory has no job title for the person.

BACKFILL. Nothing recorded direct sign-ins before this revision, so existing
accounts are classified once from the evidence that exists. An account is
direct when ANY of these holds:

  * it accepted the use agreement (only ever offered on the MindRouter site);
  * it has an API key it created itself (api_keys.app_id IS NULL);
  * it has a local password;
  * it was created before the first registered app existed, when a direct
    sign-in was the only way to get an account.

With no registered app at all, every account is direct. The stored time is the
best available approximation (agreement time, else last login, else creation);
only NULL vs NOT NULL carries meaning for backfilled rows.

Both columns are nullable additions: instant on MariaDB, no table rewrite.
"""

from alembic import op
import sqlalchemy as sa

revision = "088"
down_revision = "087"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("last_direct_login_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("email_log", sa.Column("audience", sa.String(16), nullable=True))

    bind = op.get_bind()
    first_app = bind.exec_driver_sql("SELECT MIN(created_at) FROM apps").scalar()
    if first_app is None:
        bind.exec_driver_sql(
            "UPDATE users SET last_direct_login_at = COALESCE(last_login_at, created_at)"
        )
        return
    bind.execute(
        sa.text(
            "UPDATE users u "
            "SET u.last_direct_login_at = COALESCE(u.agreement_accepted_at, u.last_login_at, u.created_at) "
            "WHERE u.agreement_accepted_at IS NOT NULL "
            "   OR u.password_hash IS NOT NULL "
            "   OR u.created_at < :first_app "
            "   OR EXISTS (SELECT 1 FROM api_keys k WHERE k.user_id = u.id AND k.app_id IS NULL)"
        ),
        {"first_app": first_app},
    )


def downgrade() -> None:
    # Loses the direct/app-only distinction; recipients fall back to "everyone".
    op.drop_column("email_log", "audience")
    op.drop_column("users", "last_direct_login_at")
