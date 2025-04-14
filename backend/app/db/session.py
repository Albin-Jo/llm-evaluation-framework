from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy import NullPool
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.app.core.config import settings

engine = create_async_engine(
    settings.DB_URI,
    echo=False,
    future=True,
    # Use connection pooling in production, disable in testing
    poolclass=NullPool if settings.APP_ENV == "testing" else None
)

# create session factory
async_session_factory = async_sessionmaker(

    engine,
    expire_on_commit=False,
    autoflush=False,
    class_=AsyncSession
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
