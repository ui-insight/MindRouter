############################################################
#
# mindrouter - functional tests for applying a quota grant
#
# Runs the REAL crud.review_quota_request against a real (in-memory
# SQLite) schema — the source-assertion tests in test_quota_budget.py
# prove the code is shaped right; these prove it actually works.
#
############################################################

"""End-to-end behaviour of the quota grant against an actual database."""

import pytest

pytest.importorskip("aiosqlite")

# Imported lazily inside the fixture: backend.app.db.__init__ builds the
# module-level engine from settings at import time, and doing that at module
# scope here would fire during collection for every test in the run.


@pytest.fixture
async def db():
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from backend.app.db.models import Base

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    tables = [Base.metadata.tables[t] for t in ("groups", "users", "quotas", "quota_requests")]
    async with engine.begin() as conn:
        await conn.run_sync(lambda s: Base.metadata.create_all(s, tables=tables))
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as session:
        yield session
    await engine.dispose()


@pytest.fixture
async def world(db):
    """A group with a 100k budget, a user in it with a quota row, one request."""
    from backend.app.db.models import Group, Quota, QuotaRequest, User

    g = Group(name="students", display_name="Students", token_budget=100_000, rpm_limit=30)
    db.add(g)
    await db.flush()
    u = User(username="alice", email="alice@example.edu", group_id=g.id)
    db.add(u)
    await db.flush()
    q = Quota(user_id=u.id, rpm_limit=30)
    db.add(q)
    await db.flush()
    req = QuotaRequest(
        user_id=u.id, requested_tokens=250_000, justification="thesis",
        request_type="quota_increase",
    )
    db.add(req)
    await db.flush()
    await db.commit()
    return g, u, q, req


async def test_explicit_grant_is_applied(db, world):
    from backend.app.db import crud
    from backend.app.db.models import QuotaRequestStatus

    _, _, q, req = world
    r = await crud.review_quota_request(
        db, req.id, reviewer_id=1, status=QuotaRequestStatus.APPROVED, granted_tokens=300_000
    )
    await db.commit()
    await db.refresh(q)
    assert r.status == QuotaRequestStatus.APPROVED
    assert q.token_budget_override == 300_000


async def test_blank_grant_defaults_to_the_requested_amount(db, world):
    from backend.app.db import crud
    from backend.app.db.models import QuotaRequestStatus

    _, _, q, req = world
    await crud.review_quota_request(db, req.id, 1, QuotaRequestStatus.APPROVED)
    await db.commit()
    await db.refresh(q)
    assert q.token_budget_override == 250_000


async def test_denial_applies_nothing_even_with_a_figure(db, world):
    from backend.app.db import crud
    from backend.app.db.models import QuotaRequestStatus

    _, _, q, req = world
    await crud.review_quota_request(db, req.id, 1, QuotaRequestStatus.DENIED, granted_tokens=999)
    await db.commit()
    await db.refresh(q)
    await db.refresh(req)
    assert req.status == QuotaRequestStatus.DENIED
    assert q.token_budget_override is None


async def test_rejected_amount_leaves_no_dirty_approval_behind(db, world):
    """The review finding on this PR.

    get_async_db() commits on normal request completion, and the dashboard
    path turns a ValueError into a redirect — a normal completion. If the
    status had been set before validation, the commit would persist an
    approval with no grant: the original bug, reintroduced.
    """
    from backend.app.db import crud
    from backend.app.db.models import QuotaRequestStatus

    _, _, q, req = world
    with pytest.raises(ValueError):
        await crud.review_quota_request(
            db, req.id, 1, QuotaRequestStatus.APPROVED, granted_tokens=-5
        )
    await db.commit()  # exactly what the dependency does after a redirect
    await db.refresh(req)
    await db.refresh(q)
    assert req.status == QuotaRequestStatus.PENDING
    assert req.reviewed_by is None
    assert q.token_budget_override is None


async def test_zero_grant_is_allowed_and_means_unlimited(db, world):
    from backend.app.core.quota_budget import UNLIMITED, effective_token_budget
    from backend.app.db import crud
    from backend.app.db.models import QuotaRequestStatus

    _, u, q, req = world
    await crud.review_quota_request(db, req.id, 1, QuotaRequestStatus.APPROVED, granted_tokens=0)
    await db.commit()
    await db.refresh(q)
    await db.refresh(u, attribute_names=["group"])
    assert q.token_budget_override == 0
    assert effective_token_budget(u, q) == UNLIMITED


async def test_enforcement_actually_sees_the_grant(db, world):
    """A user at 150k of a 100k group budget is unblocked by a 250k grant."""
    from backend.app.core.quota_budget import effective_token_budget, is_over_budget
    from backend.app.db import crud
    from backend.app.db.models import QuotaRequestStatus

    _, u, q, req = world
    await db.refresh(u, attribute_names=["group"])
    q.tokens_used = 150_000
    assert is_over_budget(u, q) is True                   # blocked on the group budget
    await crud.review_quota_request(db, req.id, 1, QuotaRequestStatus.APPROVED)  # 250k
    await db.commit()
    await db.refresh(q)
    assert effective_token_budget(u, q) == 250_000
    assert is_over_budget(u, q) is False                  # unblocked


async def test_user_with_no_quota_row_gets_one_and_keeps_the_grant(db, world):
    from backend.app.db import crud
    from backend.app.db.models import QuotaRequest, QuotaRequestStatus, User

    g, _, _, _ = world
    bob = User(username="bob", email="bob@example.edu", group_id=g.id)
    db.add(bob)
    await db.flush()
    req = QuotaRequest(user_id=bob.id, requested_tokens=50_000, justification="x",
                       request_type="quota_increase")
    db.add(req)
    await db.flush()
    await db.commit()
    assert await crud.get_user_quota(db, bob.id) is None

    await crud.review_quota_request(db, req.id, 1, QuotaRequestStatus.APPROVED)
    await db.commit()
    q = await crud.get_user_quota(db, bob.id)
    assert q is not None
    assert q.token_budget_override == 50_000
    assert q.rpm_limit == 30                              # inherited from the group


async def test_re_approval_is_idempotent(db, world):
    from backend.app.db import crud
    from backend.app.db.models import QuotaRequestStatus

    _, _, q, req = world
    for _ in range(3):
        await crud.review_quota_request(db, req.id, 1, QuotaRequestStatus.APPROVED)
        await db.commit()
    await db.refresh(q)
    assert q.token_budget_override == 250_000             # not 750_000
