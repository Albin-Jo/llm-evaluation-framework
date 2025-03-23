# File: app/db/session.py

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from backend.app.core.config import settings

# Create an engine and session factory
engine = create_async_engine(settings.DB_URI, echo=settings.APP_DEBUG)
async_session_factory = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


@asynccontextmanager
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager for database sessions.

    Yields:
        AsyncSession: Database session
    """
    session = async_session_factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for database sessions.

    Yields:
        AsyncSession: Database session
    """
    async with db_session() as session:
        yield session