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
    sign-in was the only way to get an account;
  * its group is settled (group_classified = 1). An account an app creates
    starts at 0 and is settled only by a direct sign-in or by an administrator
    placing it in a group, so 1 catches people who signed in directly after
    the first app existed without accepting the agreement or making a key.

With no registered app at all, every account is direct. The stored time is the
best available approximation (agreement time, else last login, else creation);
only NULL vs NOT NULL carries meaning for backfilled rows.

Both columns are nullable additions: instant on MariaDB, no table rewrite.

RE-RUNNABLE. MariaDB DDL is not transactional: if the backfill failed (a lock
wait under live traffic), the columns would already exist while the revision
stayed at 087, and a plain re-run would die on "Duplicate column". Each column
is therefore added only when missing, and the backfill only fills rows that are
still NULL, so running this again after a partial failure finishes the job.
"""

from alembic import op
import sqlalchemy as sa

revision = "088"
down_revision = "087"
branch_labels = None
depends_on = None


def _has_column(bind, table: str, column: str) -> bool:
    return column in {c["name"] for c in sa.inspect(bind).get_columns(table)}


def backfill(bind) -> None:
    """Mark existing accounts that show evidence of direct use. Only touches
    rows that are still NULL, so it can be run again. Plain SQL that MariaDB
    and SQLite both accept (the unit tests run it for real)."""
    first_app = bind.execute(sa.text("SELECT MIN(created_at) FROM apps")).scalar()
    if first_app is None:
        # No registered app has ever existed: nobody can be app-only.
        bind.execute(sa.text(
            "UPDATE users SET last_direct_login_at = COALESCE(last_login_at, created_at) "
            "WHERE last_direct_login_at IS NULL"
        ))
        return
    bind.execute(
        sa.text(
            "UPDATE users "
            "SET last_direct_login_at = COALESCE(agreement_accepted_at, last_login_at, created_at) "
            "WHERE last_direct_login_at IS NULL AND ("
            "      agreement_accepted_at IS NOT NULL "
            "   OR password_hash IS NOT NULL "
            "   OR created_at < :first_app "
            "   OR group_classified = 1 "
            "   OR EXISTS (SELECT 1 FROM api_keys k WHERE k.user_id = users.id AND k.app_id IS NULL))"
        ),
        {"first_app": first_app},
    )


def upgrade() -> None:
    bind = op.get_bind()
    if not _has_column(bind, "users", "last_direct_login_at"):
        op.add_column("users", sa.Column("last_direct_login_at", sa.DateTime(timezone=True), nullable=True))
    if not _has_column(bind, "email_log", "audience"):
        op.add_column("email_log", sa.Column("audience", sa.String(16), nullable=True))
    backfill(bind)


def downgrade() -> None:
    # Loses the direct/app-only distinction; recipients fall back to "everyone".
    op.drop_column("email_log", "audience")
    op.drop_column("users", "last_direct_login_at")
