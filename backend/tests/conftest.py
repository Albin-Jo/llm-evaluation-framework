# File: tests/conftest.py
import asyncio
from datetime import UTC

import pytest
import os
import uuid
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Set up test environment variables
os.environ["APP_ENV"] = "testing"
os.environ["DB_USER"] = "postgres"
os.environ["DB_PASSWORD"] = "postgres"
os.environ["DB_HOST"] = "localhost"
os.environ["DB_PORT"] = "5432"
os.environ["DB_NAME"] = "llm_evaluation_test"
os.environ["DB_URI"] = "sqlite+aiosqlite:///:memory:"


# Fixture for event loop - used by pytest-asyncio
@pytest.fixture(scope="session")
def event_loop_policy():
    """Create and return a custom event loop policy."""
    policy = asyncio.DefaultEventLoopPolicy()
    return policy


# Helper to run async functions
def run_async(coroutine):
    """Run an async function as a synchronous function."""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coroutine)


# Create database engine once for all tests
@pytest.fixture(scope="session")
def engine():
    """Create a SQLite in-memory database engine."""
    from app.models.orm.base import Base
    import app.models.orm.models  # Import all model classes

    # Create engine
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        future=True
    )

    # Set up
    async def setup():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    # Tear down
    async def teardown():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await engine.dispose()

    # Run setup
    run_async(setup())

    yield engine

    # Run teardown
    run_async(teardown())


@pytest.fixture
async def db_session(engine):
    """Create an async session for testing."""
    # Create session factory
    async_session_factory = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
    )

    # Create session
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@pytest.fixture
def db_session_sync():
    """Create a synchronous mock session for testing."""
    from unittest.mock import MagicMock, AsyncMock

    # Create a mock session with async methods
    session = MagicMock()
    session.commit = AsyncMock(return_value=None)
    session.rollback = AsyncMock(return_value=None)
    session.close = AsyncMock(return_value=None)
    session.execute = AsyncMock()
    session.refresh = AsyncMock()
    session.add = MagicMock()

    # Mock query results
    result_mock = MagicMock()
    result_mock.scalars = MagicMock(return_value=result_mock)
    result_mock.first = MagicMock(return_value=None)
    result_mock.all = MagicMock(return_value=[])

    # Set execute return value
    session.execute.return_value = result_mock

    return session


# Create fixtures for test entities
@pytest.fixture
def test_user_sync():
    """Create a synchronous test user."""
    from app.models.orm.models import User, UserRole
    from unittest.mock import MagicMock

    user = MagicMock(spec=User)
    user.id = uuid.uuid4()
    user.external_id = "test-user-id"
    user.email = "test@example.com"
    user.display_name = "Test User"
    user.role = UserRole.ADMIN
    user.is_active = True

    return user


@pytest.fixture
def test_microagent_sync():
    """Create a synchronous test microagent."""
    from app.models.orm.models import MicroAgent
    from unittest.mock import MagicMock

    microagent = MagicMock(spec=MicroAgent)
    microagent.id = uuid.uuid4()
    microagent.name = "Test MicroAgent"
    microagent.description = "A test micro-agent for evaluation"
    microagent.api_endpoint = "http://localhost:8000/test-agent"
    microagent.domain = "testing"
    microagent.is_active = True

    return microagent


@pytest.fixture
def test_dataset_sync():
    """Create a synchronous test dataset."""
    from app.models.orm.models import Dataset, DatasetType
    from unittest.mock import MagicMock

    dataset = MagicMock(spec=Dataset)
    dataset.id = uuid.uuid4()
    dataset.name = "Test Dataset"
    dataset.description = "A test dataset for evaluation"
    dataset.type = DatasetType.QUESTION_ANSWER
    dataset.file_path = "test_data/test_dataset.json"
    dataset.owner_id = uuid.uuid4()  # This would typically be test_user.id
    dataset.row_count = 2
    dataset.version = "1.0.0"
    dataset.is_public = True

    return dataset


@pytest.fixture
def test_prompt_sync():
    """Create a synchronous test prompt."""
    from app.models.orm.models import Prompt
    from unittest.mock import MagicMock

    prompt = MagicMock(spec=Prompt)
    prompt.id = uuid.uuid4()
    prompt.name = "Test Prompt"
    prompt.description = "A test prompt for evaluation"
    prompt.content = "Answer the following question: {query}\nContext: {context}"
    prompt.owner_id = uuid.uuid4()  # This would typically be test_user.id
    prompt.version = "1.0.0"
    prompt.is_public = True

    return prompt


# @pytest.fixture
# def test_evaluation_sync():
#     """Create a synchronous test evaluation."""
#     from app.models.orm.models import Evaluation, EvaluationMethod, EvaluationStatus
#     from unittest.mock import MagicMock
#
#     evaluation = MagicMock(spec=Evaluation)
#     evaluation.id = uuid.uuid4()
#     evaluation.name = "Test Evaluation"
#     evaluation.description = "A test evaluation"
#     evaluation.method = EvaluationMethod.RAGAS
#     evaluation.status = EvaluationStatus.PENDING
#     evaluation.created_by_id = uuid.uuid4()  # This would be test_user.id
#     evaluation.micro_agent_id = uuid.uuid4()  # This would be test_microagent.id
#     evaluation.dataset_id = uuid.uuid4()  # This would be test_dataset.id
#     evaluation.prompt_id = uuid.uuid4()  # This would be test_prompt.id
#     evaluation.config = {"metrics": ["faithfulness", "answer_relevancy", "context_relevancy"]}
#
#     return evaluation

@pytest.fixture
def test_evaluation_sync():
    """Synchronous test evaluation with properly initialized fields."""
    return create_mock_evaluation()


@pytest.fixture
def mock_httpx_client(monkeypatch):
    """Mock httpx client for API calls."""

    class MockResponse:
        def __init__(self, json_data, status_code=200):
            self.json_data = json_data
            self.status_code = status_code

        def json(self):
            return self.json_data

        def raise_for_status(self):
            if self.status_code >= 400:
                raise Exception(f"HTTP Error: {self.status_code}")

    class MockAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, url, json=None, headers=None, timeout=None):
            # Mock response based on the input
            if json and "query" in json:
                query = json.get("query", "")
                if query.lower() == "what is machine learning?":
                    return MockResponse({
                        "answer": "Machine learning is a branch of artificial intelligence that focuses on building systems that can learn from data.",
                        "processing_time_ms": 150
                    })
                elif query.lower() == "how does rag work?":
                    return MockResponse({
                        "answer": "RAG works by combining retrieval systems with generative models to produce more accurate and contextually relevant responses.",
                        "processing_time_ms": 200
                    })

            # Default response
            return MockResponse({
                "answer": "I'm a test response from the mock agent.",
                "processing_time_ms": 100
            })

    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient)

    return MockAsyncClient()


@pytest.fixture(scope="session")
def settings():
    """Provide app settings."""
    from app.core.config.settings import settings as app_settings
    # Ensure we're in testing mode
    app_settings.APP_ENV = "testing"
    app_settings.DB_URI = "sqlite+aiosqlite:///:memory:"
    return app_settings


# Add this to tests/conftest.py
def create_mock_evaluation(
        id=None,
        name="Test Evaluation",
        description="A test evaluation",
        method="ragas",
        status="pending",
        created_by_id=None,
        micro_agent_id=None,
        dataset_id=None,
        prompt_id=None
):
    """Create a properly initialized mock Evaluation object with all required fields."""
    from app.models.orm.models import Evaluation, EvaluationMethod, EvaluationStatus
    from unittest.mock import MagicMock
    from datetime import datetime
    import uuid

    # Use provided values or defaults
    eval_id = id or uuid.uuid4()
    created_by = created_by_id or uuid.uuid4()
    micro_agent = micro_agent_id or uuid.uuid4()
    dataset = dataset_id or uuid.uuid4()
    prompt = prompt_id or uuid.uuid4()
    now = datetime.now(UTC)

    # Create the mock with properly typed attributes
    mock_eval = MagicMock(spec=Evaluation)
    mock_eval.id = eval_id
    mock_eval.name = name
    mock_eval.description = description
    mock_eval.method = method
    mock_eval.status = status
    mock_eval.created_by_id = created_by
    mock_eval.micro_agent_id = micro_agent
    mock_eval.dataset_id = dataset
    mock_eval.prompt_id = prompt
    mock_eval.created_at = now
    mock_eval.updated_at = now
    mock_eval.config = {"metrics": ["faithfulness"]}  # Real dictionary
    mock_eval.experiment_id = "test-experiment-123"  # Real string
    mock_eval.metrics = ["faithfulness"]  # Real list
    mock_eval.start_time = now
    mock_eval.end_time = now

    return mock_eval