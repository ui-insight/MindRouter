############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# session.py: Database session management and connection pooling
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Database session management."""

import asyncio
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.settings import get_settings

settings = get_settings()

# Synchronous engine for migrations and sync operations
engine = create_engine(
    settings.database_url,
    pool_size=settings.database_pool_size,
    max_overflow=settings.database_max_overflow,
    echo=settings.database_echo,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

# Async engine for application use
# Convert mysql+pymysql to mysql+aiomysql for async
async_database_url = settings.database_url.replace(
    "mysql+pymysql", "mysql+aiomysql"
).replace(
    "mariadb+pymysql", "mariadb+aiomysql"
)

async_engine = create_async_engine(
    async_database_url,
    pool_size=settings.database_pool_size,
    max_overflow=settings.database_max_overflow,
    echo=settings.database_echo,
    pool_pre_ping=True,
    pool_recycle=300,  # Recycle connections every 5 min
    pool_timeout=10,  # Don't block forever waiting for a connection
)

AsyncSessionLocal = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


def get_db() -> Generator[Session, None, None]:
    """Get synchronous database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """Get async database session for FastAPI dependency injection.

    Uses asyncio.shield() to protect session cleanup from CancelledError,
    which can corrupt connections and leak them from the pool.
    """
    session = AsyncSessionLocal()
    try:
        yield session
        await session.commit()
    except asyncio.CancelledError:
        # Shield rollback from further cancellation so the connection
        # is properly returned to the pool instead of being leaked.
        try:
            await asyncio.shield(session.rollback())
        except Exception:
            pass
        raise
    except Exception:
        try:
            await asyncio.shield(session.rollback())
        except Exception:
            pass
        raise
    finally:
        try:
            await asyncio.shield(session.close())
        except Exception:
            # Last resort: invalidate the connection so the pool discards it
            # rather than leaving a corrupted connection in the pool.
            try:
                await session.invalidate()
            except Exception:
                pass


@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """Context manager for synchronous database session."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@asynccontextmanager
async def get_async_db_context() -> AsyncGenerator[AsyncSession, None]:
    """Context manager for async database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def _finish_isolated_session(session: AsyncSession, rollback: bool) -> None:
    """Roll back (when the body failed) and close an isolated session.

    One coroutine, run as one task, so rollback and close happen strictly in
    sequence: even if the task waiting on it is cancelled mid-cleanup they can
    never overlap on the connection (two coroutines on one aiomysql connection
    is exactly the readexactly() failure isolated sessions exist to prevent).
    Every failure is swallowed — cleanup must never replace the exception the
    caller is classifying.
    """
    if rollback:
        try:
            await session.rollback()
        except Exception:
            pass
    try:
        await session.close()
    except Exception:
        # Last resort: invalidate so the pool discards the connection rather
        # than handing a corrupted one to the next checkout.
        try:
            await session.invalidate()
        except Exception:
            pass


@asynccontextmanager
async def isolated_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Short-lived, independent write session for work detached from a request.

    For writes that can outlive the request that scheduled them — e.g. the
    shielded post-[DONE] accounting write of a streaming response, which keeps
    running after a client disconnect while FastAPI tears down the
    request-scoped ``get_async_db`` session. Such writes must never share that
    session: concurrent use of its single aiomysql connection raises
    "readexactly() called while another coroutine is already waiting" (MySQL
    2013/2014) and loses the write.

    - Yields a fresh ``AsyncSessionLocal()`` session (its own pooled connection).
    - Does NOT commit: callers commit explicitly.
    - On any exception from the body, CancelledError included, rolls back under
      ``asyncio.shield``. A failing rollback is swallowed, so the ORIGINAL
      exception always propagates (callers classify ``e.orig.args[0]`` to
      decide on retries).
    - Always closes under ``asyncio.shield``, invalidating the connection if
      close fails.

    If the waiting task is cancelled during cleanup, the shielded cleanup still
    runs to completion in the background; that cancellation is re-raised only
    when it would not mask an exception from the body.

    ``get_async_db`` / ``get_async_db_context`` are unchanged; use those for
    request-scoped work.
    """
    session = AsyncSessionLocal()
    failed = False
    try:
        yield session
    except BaseException:
        failed = True
        raise
    finally:
        cleanup = asyncio.ensure_future(
            _finish_isolated_session(session, rollback=failed)
        )
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            if not failed:
                raise


# ------------------------------------------------------------------
# Archive database (lazy-init, only when archive_database_url is set)
# ------------------------------------------------------------------

_archive_async_engine = None
_ArchiveAsyncSessionLocal = None


def _init_archive_engine():
    """Lazily create the archive async engine and session factory."""
    global _archive_async_engine, _ArchiveAsyncSessionLocal
    if _archive_async_engine is not None:
        return

    archive_url = settings.archive_database_url
    if not archive_url:
        return

    async_archive_url = archive_url.replace(
        "mysql+pymysql", "mysql+aiomysql"
    ).replace(
        "mariadb+pymysql", "mariadb+aiomysql"
    )

    _archive_async_engine = create_async_engine(
        async_archive_url,
        pool_size=5,
        max_overflow=5,
        echo=settings.database_echo,
        pool_pre_ping=True,
        pool_recycle=300,
        pool_timeout=10,
    )

    _ArchiveAsyncSessionLocal = async_sessionmaker(
        _archive_async_engine,
        class_=AsyncSession,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
    )


def get_archive_engine():
    """Return the archive async engine (or None if not configured)."""
    _init_archive_engine()
    return _archive_async_engine


@asynccontextmanager
async def get_archive_db_context() -> AsyncGenerator[AsyncSession, None]:
    """Context manager for archive database session.

    Raises RuntimeError if archive DB is not configured.
    """
    _init_archive_engine()
    if _ArchiveAsyncSessionLocal is None:
        raise RuntimeError("Archive database is not configured (ARCHIVE_DATABASE_URL not set)")

    async with _ArchiveAsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def close_archive_engine():
    """Dispose the archive engine on shutdown."""
    global _archive_async_engine, _ArchiveAsyncSessionLocal
    if _archive_async_engine is not None:
        await _archive_async_engine.dispose()
        _archive_async_engine = None
        _ArchiveAsyncSessionLocal = None
