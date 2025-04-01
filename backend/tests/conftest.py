import asyncio
import os
from typing import AsyncGenerator, Dict, Generator
import uuid
from datetime import datetime, timedelta

import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from backend.app.core.config import settings
from backend.app.db.session import get_db
from backend.app.db.models.orm.base import Base
from backend.app.db.models.orm.models import User, DatasetType, UserRole
from backend.app.main import app as main_app
# from backend.app.services.auth import create_access_token

# Use a test database URL
TEST_DATABASE_URL = settings.DATABASE_URL.replace(
    "/" + settings.DATABASE_NAME, "/test_" + settings.DATABASE_NAME
)

# Create test engine with Echo for debugging
engine = create_async_engine(TEST_DATABASE_URL, poolclass=NullPool, echo=True)
TestingSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


# Setup and teardown for the database
@pytest_asyncio.fixture(scope="session")
async def setup_database():
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    yield

    # Drop tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


# Get a test database session
@pytest_asyncio.fixture
async def db(setup_database) -> AsyncGenerator[AsyncSession, None]:
    async with TestingSessionLocal() as session:
        # Start transaction
        async with session.begin():
            yield session
            # Roll back transaction after test
            await session.rollback()


# Override the dependency in FastAPI app
@pytest.fixture
def app(db) -> FastAPI:
    async def override_get_db():
        yield db

    main_app.dependency_overrides[get_db] = override_get_db
    return main_app


# Create test client
@pytest.fixture
def client(app) -> Generator[TestClient, None, None]:
    with TestClient(app) as c:
        yield c


# User fixtures
@pytest_asyncio.fixture
async def test_user(db) -> User:
    user = User(
        id=uuid.uuid4(),
        email="test@example.com",
        hashed_password="$2b$12$IKEQb00u5eHrkBkIG4tK8eW/4PN5EYLbtXGHILbHF.vLUYG3XnCXS",  # 'password123'
        is_active=True,
        role=UserRole.USER,
        full_name="Test User"
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


@pytest_asyncio.fixture
async def test_admin(db) -> User:
    admin = User(
        id=uuid.uuid4(),
        email="admin@example.com",
        hashed_password="$2b$12$IKEQb00u5eHrkBkIG4tK8eW/4PN5EYLbtXGHILbHF.vLUYG3XnCXS",  # 'password123'
        is_active=True,
        role=UserRole.ADMIN,
        full_name="Admin User"
    )
    db.add(admin)
    await db.commit()
    await db.refresh(admin)
    return admin


# Auth tokens
@pytest.fixture
def user_token(test_user) -> str:
    return create_access_token(
        data={"sub": str(test_user.id), "email": test_user.email},
        expires_delta=timedelta(minutes=30)
    )


@pytest.fixture
def admin_token(test_admin) -> str:
    return create_access_token(
        data={"sub": str(test_admin.id), "email": test_admin.email},
        expires_delta=timedelta(minutes=30)
    )


# Auth headers for requests
@pytest.fixture
def user_auth_headers(user_token) -> Dict[str, str]:
    return {"Authorization": f"Bearer {user_token}"}


@pytest.fixture
def admin_auth_headers(admin_token) -> Dict[str, str]:
    return {"Authorization": f"Bearer {admin_token}"}


# Mock file upload data
@pytest.fixture
def sample_csv_file() -> bytes:
    return b"query,context,answer\nWhat is LLM?,Language model,Large Language Model\nHow to evaluate LLM?,Use metrics,Use evaluation framework"


@pytest.fixture
def sample_dataset(db, test_user) -> Dict:
    return {
        "id": uuid.uuid4(),
        "name": "Test Dataset",
        "description": "Sample dataset for testing",
        "type": DatasetType.QUESTION_ANSWER,
        "file_path": "datasets/test_dataset.csv",
        "schema": {"type": "object", "properties": {}},
        "version": "1.0.0",
        "row_count": 2,
        "is_public": False,
        "owner_id": test_user.id,
        "domain": "test",
        "created_at": datetime.now(),
        "updated_at": datetime.now(),
    }


@pytest.fixture
def sample_prompt_template(db) -> Dict:
    return {
        "id": uuid.uuid4(),
        "name": "Test Template",
        "description": "A test prompt template",
        "content": "Here is a prompt with {variable}",
        "variables": ["variable"],
        "is_public": True,
        "created_at": datetime.now(),
        "updated_at": datetime.now(),
    }


@pytest.fixture
def sample_prompt(db, test_user, sample_prompt_template) -> Dict:
    return {
        "id": uuid.uuid4(),
        "name": "Test Prompt",
        "description": "A test prompt",
        "content": "Here is a prompt with {test_var}",
        "variables": ["test_var"],
        "is_public": False,
        "template_id": sample_prompt_template["id"],
        "owner_id": test_user.id,
        "created_at": datetime.now(),
        "updated_at": datetime.now(),
    }


# Mock storage service
@pytest.fixture
def mock_storage_service(monkeypatch):
    class MockStorageService:
        async def upload_file(self, file, path):
            return path

        async def file_exists(self, path):
            return True

        async def read_file(self, path):
            if path.endswith(".csv"):
                return b"query,context,answer\nWhat is LLM?,Language model,Large Language Model"
            return b"test content"

        async def read_file_stream(self, path):
            yield b"test content chunk"

        async def delete_file(self, path):
            return True

    from backend.app.services import storage
    monkeypatch.setattr(storage, "get_storage_service", lambda: MockStorageService())
    return MockStorageService()