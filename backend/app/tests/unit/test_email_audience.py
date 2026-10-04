############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_email_audience.py: Bulk-email audiences (migration 088).
#
# A bulk email goes to "everyone" or only to "direct" MindRouter
# users: people who have signed in to MindRouter itself, as
# opposed to accounts that exist only because a registered app
# (e.g. VandalChat) provisioned them.
#
# Covers, against a real in-memory schema: the audience filter
# and its combinations (groups, picked users, blog opt-out,
# inactive accounts), an unknown audience raising instead of
# widening the send, the email log recording the audience, and
# which sign-in paths stamp last_direct_login_at (web sign-ins
# do, an app provisioning the account does not). Covers with
# mocks: both send routes refusing to send without an explicit
# audience, passing the chosen one through, and the count
# endpoint reporting both audiences. Plus the migration.
#
############################################################

"""Bulk-email audiences: everyone, or direct MindRouter users only."""

import importlib.util
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("aiosqlite")

_APP = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
async def db():
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from backend.app.db.models import Base

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    names = ("groups", "users", "quotas", "app_config")
    tables = [Base.metadata.tables[t] for t in names]
    async with engine.begin() as conn:
        await conn.run_sync(lambda s: Base.metadata.create_all(s, tables=tables))
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as session:
        yield session
    await engine.dispose()


@pytest.fixture
async def people(db):
    """Two groups; direct users, app-only users, and accounts no audience should reach."""
    from backend.app.db.models import AppConfig, Group, User

    staff = Group(name="staff", display_name="Staff", token_budget=1, rpm_limit=30)
    students = Group(name="students", display_name="Students", token_budget=1, rpm_limit=30)
    db.add_all([staff, students])
    await db.flush()

    def user(name, group, direct, **kw):
        return User(username=name, email=kw.pop("email", f"{name}@example.edu"), group_id=group.id,
                    last_login_at=NOW, last_direct_login_at=NOW if direct else None, **kw)

    rows = {
        "dana_direct": user("dana_direct", staff, True),
        "dev_direct": user("dev_direct", students, True),
        "optout_direct": user("optout_direct", staff, True),
        "vera_app_only": user("vera_app_only", students, False),      # VandalChat only; last_login_at IS set
        "vic_app_only": user("vic_app_only", staff, False),
        "inactive_direct": user("inactive_direct", staff, True, is_active=False),
        "blank_email": user("blank_email", staff, True, email=""),
    }
    db.add_all(rows.values())
    await db.flush()
    db.add(AppConfig(key=f"user.{rows['optout_direct'].id}.email_optout", value='"true"'))
    await db.commit()
    return {"staff": staff, "students": students, **rows}


def _names(users):
    return sorted(u.username for u in users)


class TestAudienceFilter:
    async def test_everyone_and_direct(self, db, people):
        from backend.app.db import crud

        everyone = _names(await crud.get_emailable_users(db))
        assert everyone == ["dana_direct", "dev_direct", "optout_direct", "vera_app_only", "vic_app_only"]
        assert _names(await crud.get_emailable_users(db, audience="all")) == everyone       # the default is "all"
        direct = _names(await crud.get_emailable_users(db, audience="direct"))
        assert direct == ["dana_direct", "dev_direct", "optout_direct"]

    async def test_an_app_login_does_not_make_someone_direct(self, db, people):
        # last_login_at is bumped by app provisioning too; only last_direct_login_at decides.
        from backend.app.db import crud

        assert people["vera_app_only"].last_login_at is not None
        assert "vera_app_only" not in _names(await crud.get_emailable_users(db, audience="direct"))

    async def test_combines_with_groups_picked_users_and_the_blog_opt_out(self, db, people):
        from backend.app.db import crud

        staff = [people["staff"].id]
        assert _names(await crud.get_emailable_users(db, group_ids=staff, audience="all")) == ["dana_direct", "optout_direct", "vic_app_only"]
        assert _names(await crud.get_emailable_users(db, group_ids=staff, audience="direct")) == ["dana_direct", "optout_direct"]
        picked = [people["dana_direct"].id, people["vera_app_only"].id]
        assert _names(await crud.get_emailable_users(db, user_ids=picked, audience="direct")) == ["dana_direct"]
        assert _names(await crud.get_emailable_users(db, exclude_blog_optout=True, audience="direct")) == ["dana_direct", "dev_direct"]
        assert _names(await crud.get_emailable_users(db, exclude_blog_optout=True, audience="all")) == ["dana_direct", "dev_direct", "vera_app_only", "vic_app_only"]

    @pytest.mark.parametrize("bad", ["", "everyone", "Direct", None, "all "])
    async def test_an_unknown_audience_raises_rather_than_sending_to_everyone(self, db, people, bad):
        from backend.app.db import crud

        with pytest.raises(ValueError):
            await crud.get_emailable_users(db, audience=bad)

    async def test_the_email_log_records_the_audience(self):
        from backend.app.db import crud

        db = MagicMock(add=MagicMock(), flush=AsyncMock())     # email_log's BIGINT key does not autoincrement on SQLite
        log = await crud.create_email_log(db, subject="s", sent_by=1, recipient_count=3, audience="direct")
        assert log.audience == "direct" and db.add.call_args.args[0] is log
        older = await crud.create_email_log(db, subject="s", sent_by=1, recipient_count=3)
        assert older.audience is None
        assert set(crud.EMAIL_AUDIENCE_LABELS) == set(crud.EMAIL_AUDIENCES) == {"all", "direct"}


class TestWhoCountsAsDirect:
    _PROFILE = {"id": "11111111-2222-3333-4444-555555555555", "mail": "new.person@example.edu",
                "userPrincipalName": "new.person@example.edu", "displayName": "New Person", "jobTitle": "Student"}

    async def _groups(self, db):
        from backend.app.db.models import Group

        for name in ("students", "staff", "faculty", "other", "admin"):
            db.add(Group(name=name, display_name=name.title(), token_budget=1, rpm_limit=30))
        await db.flush()

    async def test_an_app_creating_the_account_is_not_a_direct_sign_in(self, db):
        from backend.app.dashboard.azure_auth import find_or_create_azure_user

        await self._groups(db)
        user = await find_or_create_azure_user(db, dict(self._PROFILE), direct=False)
        assert user is not None and user.last_login_at is not None
        assert user.last_direct_login_at is None

        # The app refreshing the same account later still does not make it direct...
        again = await find_or_create_azure_user(db, dict(self._PROFILE), direct=False)
        assert again.id == user.id and again.last_direct_login_at is None
        # ...and the person signing in to MindRouter itself does, for good.
        signed_in = await find_or_create_azure_user(db, dict(self._PROFILE))
        assert signed_in.id == user.id and signed_in.last_direct_login_at is not None
        stamp = signed_in.last_direct_login_at
        after_app = await find_or_create_azure_user(db, dict(self._PROFILE), direct=False)
        assert after_app.last_direct_login_at == stamp

    async def test_a_web_sign_in_creating_the_account_is_direct(self, db):
        from backend.app.dashboard.azure_auth import find_or_create_azure_user

        await self._groups(db)
        user = await find_or_create_azure_user(db, dict(self._PROFILE))
        assert user.last_direct_login_at is not None and user.last_direct_login_at == user.last_login_at

    def test_the_app_route_is_the_only_caller_that_says_not_direct(self):
        apps = (_APP / "api" / "apps_api.py").read_text()
        assert "find_or_create_azure_user(db, profile, direct=False)" in apps
        azure = (_APP / "dashboard" / "azure_auth.py").read_text()
        assert "user = await find_or_create_azure_user(db, profile)\n" in azure            # the web callback: direct

    def test_password_and_generic_sso_sign_ins_stamp_it(self):
        routes = (_APP / "dashboard" / "routes.py").read_text()
        assert "user.last_direct_login_at = user.last_login_at" in routes
        sso = (_APP / "dashboard" / "sso" / "base.py").read_text()
        assert sso.count("user.last_direct_login_at = user.last_login_at") == 2             # existing and new accounts


def _admin():
    user = MagicMock()
    user.id, user.username, user.full_name = 1, "admin", "Admin"
    return user


class TestAdminEmailRoute:
    async def _send(self, audience, **form):
        from backend.app.dashboard import email_routes as er

        recipients = [MagicMock(email="a@example.edu", username="a", full_name="A")]
        crud = MagicMock()
        crud.EMAIL_AUDIENCES = ("all", "direct")
        crud.get_emailable_users = AsyncMock(return_value=recipients)
        crud.create_email_log = AsyncMock(return_value=MagicMock(id=7))
        svc = MagicMock()
        svc.get_smtp_config = AsyncMock(return_value={"default_sender": "noreply@example.edu"})
        svc.is_smtp_configured = MagicMock(return_value=True)
        svc.get_base_url = AsyncMock(return_value="https://x")
        svc.send_bulk_email = MagicMock(return_value="coro")
        with patch.object(er, "_require_admin", AsyncMock(return_value=(_admin(), None))), \
             patch.object(er, "crud", crud), patch.object(er, "email_service", svc), \
             patch.object(er.asyncio, "create_task") as task:
            args = {"subject": "Maintenance", "body": "<p>hi</p>", "recipient_mode": "all", "group_ids": None, "user_ids": None, **form}
            resp = await er.send_email(MagicMock(), audience=audience, db=MagicMock(commit=AsyncMock()), **args)
        return resp, crud, task

    @pytest.mark.parametrize("audience", [None, "", "everyone", "ALL"])
    async def test_nothing_is_sent_without_an_explicit_audience(self, audience):
        resp, crud, task = await self._send(audience)
        assert resp.status_code == 302 and "error=Choose+an+audience" in resp.headers["location"]
        crud.get_emailable_users.assert_not_awaited(); crud.create_email_log.assert_not_awaited(); task.assert_not_called()

    @pytest.mark.parametrize("audience", ["all", "direct"])
    async def test_the_chosen_audience_selects_recipients_and_is_logged(self, audience):
        resp, crud, task = await self._send(audience)
        assert "success=Sending+to+1+recipients" in resp.headers["location"]
        assert crud.get_emailable_users.call_args.kwargs == {"audience": audience}
        assert crud.create_email_log.call_args.kwargs["audience"] == audience
        task.assert_called_once()

    async def test_audience_also_applies_to_groups_and_picked_users(self):
        _, crud, _ = await self._send("direct", recipient_mode="groups", group_ids="3,4")
        assert crud.get_emailable_users.call_args.kwargs == {"group_ids": [3, 4], "audience": "direct"}
        _, crud, _ = await self._send("direct", recipient_mode="users", user_ids="9")
        assert crud.get_emailable_users.call_args.kwargs == {"user_ids": [9], "audience": "direct"}

    async def test_the_count_endpoint_reports_both_audiences(self):
        import json

        from backend.app.dashboard import email_routes as er

        crud = MagicMock()
        crud.EMAIL_AUDIENCES = ("all", "direct")
        crud.get_emailable_users = AsyncMock(side_effect=lambda db, **kw: [1] * (753 if kw["audience"] == "all" else 394))

        async def count(body):
            request = MagicMock(); request.json = AsyncMock(return_value=body)
            with patch.object(er, "_require_admin", AsyncMock(return_value=(_admin(), None))), patch.object(er, "crud", crud):
                return json.loads((await er.recipient_count(request, db=MagicMock())).body)

        assert await count({"mode": "all"}) == {"count": None, "counts": {"all": 753, "direct": 394}}   # nothing picked yet
        assert (await count({"mode": "all", "audience": "direct"}))["count"] == 394
        assert (await count({"mode": "groups", "group_ids": [], "audience": "all"}))["counts"] == {"all": 0, "direct": 0}


class TestBlogEmailRoute:
    async def _send(self, audience):
        from backend.app.dashboard import blog

        crud = MagicMock()
        crud.EMAIL_AUDIENCES = ("all", "direct")
        crud.get_blog_post_by_id = AsyncMock(return_value=MagicMock(id=16, is_published=True, title="T", content="c", slug="s"))
        crud.get_emailable_users = AsyncMock(return_value=[MagicMock(email="a@example.edu", username="a", full_name="A")])
        crud.create_email_log = AsyncMock(return_value=MagicMock(id=7))
        svc = MagicMock()
        svc.get_smtp_config = AsyncMock(return_value={"default_sender": "noreply@example.edu"})
        svc.is_smtp_configured = MagicMock(return_value=True)
        svc.get_base_url = AsyncMock(return_value="https://x")
        svc.load_blog_inline_images = AsyncMock(return_value=[])
        with patch.object(blog, "_require_admin", AsyncMock(return_value=(_admin(), None))), \
             patch.object(blog, "crud", crud), patch.object(blog, "email_service", svc), \
             patch.object(blog.asyncio, "create_task") as task:
            resp = await blog.admin_blog_send_email(MagicMock(), 16, audience=audience, db=MagicMock(commit=AsyncMock()))
        return resp, crud, task

    @pytest.mark.parametrize("audience", [None, "", "subscribers"])
    async def test_nothing_is_sent_without_an_explicit_audience(self, audience):
        resp, crud, task = await self._send(audience)
        assert resp.status_code == 302 and "error=Choose+an+audience" in resp.headers["location"]
        crud.get_emailable_users.assert_not_awaited(); task.assert_not_called()

    @pytest.mark.parametrize("audience", ["all", "direct"])
    async def test_the_chosen_audience_is_used_with_the_opt_out_and_logged(self, audience):
        resp, crud, task = await self._send(audience)
        assert "success=Sending+to+1+recipients" in resp.headers["location"]
        assert crud.get_emailable_users.call_args.kwargs == {"exclude_blog_optout": True, "audience": audience}
        assert crud.create_email_log.call_args.kwargs["audience"] == audience and task.call_count == 1


class TestFormsAndMigration:
    def test_neither_form_preselects_an_audience(self):
        for name in ("email.html", "blog_edit.html"):
            html = (_APP / "dashboard" / "templates" / "admin" / name).read_text()
            radios = [line for line in html.splitlines() if line.strip().startswith("<input") and 'name="audience"' in line]
            assert len(radios) == 2, name
            assert all("required" in r and "checked" not in r for r in radios), name
            assert {'value="all"', 'value="direct"'} <= {v for r in radios for v in ('value="all"', 'value="direct"') if v in r}

    def test_migration_088(self):
        path = _APP / "db" / "migrations" / "versions" / "20261004_000001_088_direct_login_and_email_audience.py"
        spec = importlib.util.spec_from_file_location("migration_088", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module.revision == "088" and module.down_revision == "087"
        src = path.read_text()
        # Every piece of evidence for direct use is in the backfill, and the no-apps case is handled.
        for clause in ("agreement_accepted_at IS NOT NULL", "password_hash IS NOT NULL", "created_at < :first_app",
                       "k.app_id IS NULL", "if first_app is None"):
            assert clause in src, clause

    def test_the_orm_has_the_columns_the_migration_adds(self):
        from backend.app.db.models import EmailLog, User

        assert User.__table__.c.last_direct_login_at.nullable and EmailLog.__table__.c.audience.nullable
