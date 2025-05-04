import logging
from typing import AsyncGenerator, Dict

import httpx
import pytest
import pytest_asyncio
from asgi_lifespan import LifespanManager
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool

from backend.app.core.config import settings
from backend.app.db.models.base import Base
from backend.app.db.session import get_db
from backend.app.main import app as main_app

# Configure logging for tests
logger = logging.getLogger("pytest")

# Use a test database URL
TEST_DATABASE_URL = settings.DB_URI.replace(
    "/" + settings.DB_NAME, "/test_" + settings.DB_NAME
)

engine = create_async_engine(TEST_DATABASE_URL, poolclass=NullPool, echo=False)
TestingSessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False)


# Setup and teardown for the database
@pytest_asyncio.fixture(scope="session")
async def setup_database():
    """Set up the test database once per session."""
    logger.info(f"Setting up test database: {TEST_DATABASE_URL}")
    try:
        # Create tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)

        yield
    except Exception as e:
        logger.error(f"Error setting up test database: {str(e)}")
        raise
    finally:
        # Clean up connections
        logger.info("Cleaning up database connections")
        await engine.dispose()


# Get a test database session
@pytest_asyncio.fixture
async def db(setup_database) -> AsyncGenerator[AsyncSession, None]:
    """Create a fresh database session for a test."""
    connection = None
    session = None

    try:
        async with TestingSessionLocal() as session:
            # Start transaction
            await session.begin()

            # Return session for the test to use
            yield session

            # Rollback transaction after test completes
            await session.rollback()
    except Exception as e:
        logger.error(f"Database session error: {str(e)}")
        if session:
            await session.rollback()
        raise
    finally:
        if session:
            await session.close()


# Override the dependency in FastAPI app
@pytest_asyncio.fixture
async def app(db) -> AsyncGenerator[FastAPI, None]:
    """Create a FastAPI test app with dependencies overridden."""

    async def override_get_db():
        yield db

    # Save original dependencies
    original_overrides = main_app.dependency_overrides.copy()

    # Apply overrides
    main_app.dependency_overrides[get_db] = override_get_db

    yield main_app

    # Restore original state
    main_app.dependency_overrides = original_overrides


# Create async test client
@pytest_asyncio.fixture
async def async_client(app) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Create an async HTTP client for testing."""
    try:
        async with LifespanManager(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                    timeout=10.0  # Add timeout to prevent hanging tests
            ) as client:
                yield client
    except Exception as e:
        logger.error(f"Error with async client: {str(e)}")
        raise


# Mock authentication fixture
@pytest.fixture
def mock_auth_headers() -> Dict[str, str]:
    """Generate mock authentication headers for testing protected endpoints."""
    return {
        "Authorization": "Bearer test_token"
    }


@pytest.fixture
def mock_storage_service(monkeypatch):
    """Mock the storage service for testing."""

    async def upload_file(file, path):
        logger.debug(f"Mock: Uploading file to {path}")
        return path

    async def file_exists(path):
        logger.debug(f"Mock: Checking if file exists: {path}")
        return True

    async def read_file(path):
        logger.debug(f"Mock: Reading file: {path}")
        if path.endswith(".csv"):
            return b"query,context,answer\nWhat is LLM?,Language model,Large Language Model"
        elif path.endswith(".json"):
            return b'[{"query": "What is LLM?", "answer": "Large Language Model"}]'
        return b"test content"

    async def read_file_stream(path):
        logger.debug(f"Mock: Streaming file: {path}")
        yield b"test content chunk"

    async def delete_file(path):
        logger.debug(f"Mock: Deleting file: {path}")
        return True

    class MockStorageService:
        pass

    from backend.app.services import storage
    monkeypatch.setattr(storage, "get_storage_service", lambda: MockStorageService())
    return MockStorageService()


# Add fixture to log test start/end for better debugging
@pytest.fixture(autouse=True)
def log_test_info(request):
    """Log info about each test as it runs."""
    test_name = request.node.name
    logger.info(f"Starting test: {test_name}")
    yield
    logger.info(f"Completed test: {test_name}")
