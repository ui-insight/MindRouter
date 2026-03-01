############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
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
