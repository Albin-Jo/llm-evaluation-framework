# File: tests/api/test_evaluations_api.py
from datetime import datetime, UTC

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession
from unittest.mock import patch, MagicMock, AsyncMock
from uuid import UUID

from app.api.endpoints.evaluations import router
from app.models.orm.models import User, Evaluation, EvaluationStatus
from app.services.auth import get_current_active_user
from app.db.session import get_db
from tests.conftest import create_mock_evaluation

app = FastAPI()
app.include_router(router)


# pytestmark = pytest.mark.skipif(
#     True,  # Change to False when ready to enable these tests
#     reason="Database tests are currently disabled"
# )

# Override dependencies
@pytest.fixture
def client(test_user_sync):
    # Override auth dependency
    async def override_get_current_active_user():
        return test_user_sync

    app.dependency_overrides[get_current_active_user] = override_get_current_active_user

    # Override DB dependency
    async def override_get_db():
        mock_session = AsyncMock(spec=AsyncSession)
        yield mock_session

    app.dependency_overrides[get_db] = override_get_db

    return TestClient(app)


def test_create_evaluation(client, test_user_sync, test_microagent_sync, test_dataset_sync, test_prompt_sync):
    """Test creating an evaluation via API."""
    # Mock service
    with patch('app.services.evaluation_service.EvaluationService.create_evaluation') as mock_create:
        # Setup mock return value with datetime fields
        now = datetime.now(UTC)
        mock_create.return_value = Evaluation(
            id=UUID('00000000-0000-0000-0000-000000000001'),
            name="Test API Evaluation",
            description="Testing API",
            method="ragas",
            status=EvaluationStatus.PENDING,
            created_by_id=test_user_sync.id,
            micro_agent_id=test_microagent_sync.id,
            dataset_id=test_dataset_sync.id,
            prompt_id=test_prompt_sync.id,
            created_at=now,
            updated_at=now
        )

        # Convert UUID objects to strings for JSON serialization
        # Call API
        response = client.post(
            "/",
            json={
                "name": "Test API Evaluation",
                "description": "Testing API",
                "method": "ragas",
                "micro_agent_id": str(test_microagent_sync.id),
                "dataset_id": str(test_dataset_sync.id),
                "prompt_id": str(test_prompt_sync.id),
                "config": {"metrics": ["faithfulness"]}
            }
        )

        # Check response
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Test API Evaluation"
        assert data["status"] == "pending"


# File: tests/api/test_evaluations_api.py (updated test functions)

def test_list_evaluations(client, test_evaluation_sync):
    """Test listing evaluations via API."""
    # Mock service
    with patch('app.services.evaluation_service.EvaluationService.list_evaluations') as mock_list:
        # Add datetime fields to the test_evaluation_sync
        now = datetime.now(UTC)
        test_evaluation_sync.created_at = now
        test_evaluation_sync.updated_at = now

        # Setup mock return value
        mock_list.return_value = [test_evaluation_sync]

        # Call API
        response = client.get("/")

        # Check response
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["id"] == str(test_evaluation_sync.id)


def test_get_evaluation(client, test_evaluation_sync):
    """Test getting an evaluation via API."""
    # Mock service
    with patch('app.services.evaluation_service.EvaluationService.get_evaluation') as mock_get:
        with patch('app.services.evaluation_service.EvaluationService.get_evaluation_results') as mock_results:
            # Add datetime fields to the test_evaluation_sync
            now = datetime.now(UTC)
            test_evaluation_sync.created_at = now
            test_evaluation_sync.updated_at = now

            # Setup mock return values
            mock_get.return_value = test_evaluation_sync
            mock_results.return_value = []

            # Convert UUID to string for URL path
            evaluation_id_str = str(test_evaluation_sync.id)

            # Call API
            response = client.get(f"/{evaluation_id_str}")

            # Check response
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == evaluation_id_str
            assert "results" in data


def test_update_evaluation(client, test_evaluation_sync):
    """Test updating an evaluation via API."""
    # Mock service
    with patch('app.services.evaluation_service.EvaluationService.get_evaluation') as mock_get:
        with patch('app.services.evaluation_service.EvaluationService.update_evaluation') as mock_update:
            # Setup mock return values
            mock_get.return_value = test_evaluation_sync

            # Create updated evaluation
            updated = create_mock_evaluation(
                id=test_evaluation_sync.id,
                name="Updated API Evaluation",
                description="Updated via API",
                method=test_evaluation_sync.method,
                status=test_evaluation_sync.status,
                created_by_id=test_evaluation_sync.created_by_id,
                micro_agent_id=test_evaluation_sync.micro_agent_id,
                dataset_id=test_evaluation_sync.dataset_id,
                prompt_id=test_evaluation_sync.prompt_id
            )

            mock_update.return_value = updated

            # Call API
            response = client.put(
                f"/{test_evaluation_sync.id}",
                json={
                    "name": "Updated API Evaluation",
                    "description": "Updated via API"
                }
            )

            # Check response
            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "Updated API Evaluation"
            assert data["description"] == "Updated via API"


def test_start_evaluation(client, test_evaluation_sync):
    """Test starting an evaluation via API."""
    # Mock service
    with patch('app.services.evaluation_service.EvaluationService.get_evaluation') as mock_get:
        with patch('app.services.evaluation_service.EvaluationService.queue_evaluation_job') as mock_queue:
            # Create running evaluation
            running_eval = create_mock_evaluation(
                id=test_evaluation_sync.id,
                name=test_evaluation_sync.name,
                description=test_evaluation_sync.description,
                method=test_evaluation_sync.method,
                status="running",
                created_by_id=test_evaluation_sync.created_by_id,
                micro_agent_id=test_evaluation_sync.micro_agent_id,
                dataset_id=test_evaluation_sync.dataset_id,
                prompt_id=test_evaluation_sync.prompt_id
            )

            # Set up side effects
            mock_get.side_effect = [test_evaluation_sync, running_eval]

            # Call API
            response = client.post(f"/{test_evaluation_sync.id}/start")

            # Check response
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "running"


def test_cancel_evaluation(client, test_evaluation_sync):
    """Test cancelling an evaluation via API."""
    # Mock service
    with patch('app.services.evaluation_service.EvaluationService.get_evaluation') as mock_get:
        with patch('app.services.evaluation_service.EvaluationService.update_evaluation') as mock_update:
            # Create running evaluation
            running_eval = create_mock_evaluation(
                id=test_evaluation_sync.id,
                name=test_evaluation_sync.name,
                description=test_evaluation_sync.description,
                method=test_evaluation_sync.method,
                status="running",
                created_by_id=test_evaluation_sync.created_by_id,
                micro_agent_id=test_evaluation_sync.micro_agent_id,
                dataset_id=test_evaluation_sync.dataset_id,
                prompt_id=test_evaluation_sync.prompt_id
            )

            mock_get.return_value = running_eval

            # Create cancelled evaluation
            cancelled_eval = create_mock_evaluation(
                id=test_evaluation_sync.id,
                name=test_evaluation_sync.name,
                description=test_evaluation_sync.description,
                method=test_evaluation_sync.method,
                status="cancelled",
                created_by_id=test_evaluation_sync.created_by_id,
                micro_agent_id=test_evaluation_sync.micro_agent_id,
                dataset_id=test_evaluation_sync.dataset_id,
                prompt_id=test_evaluation_sync.prompt_id
            )

            mock_update.return_value = cancelled_eval

            # Call API
            response = client.post(f"/{test_evaluation_sync.id}/cancel")

            # Check response
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "cancelled"


def test_get_evaluation_results(client, test_evaluation_sync):
    """Test getting evaluation results via API."""
    # Mock service
    with patch('app.services.evaluation_service.EvaluationService.get_evaluation') as mock_get:
        with patch('app.services.evaluation_service.EvaluationService.get_evaluation_results') as mock_results:
            with patch('app.services.evaluation_service.EvaluationService.get_metric_scores') as mock_scores:
                # Add datetime fields
                now = datetime.now(UTC)
                test_evaluation_sync.created_at = now
                test_evaluation_sync.updated_at = now

                # Setup mock return values
                mock_get.return_value = test_evaluation_sync

                # Create mock result with datetime fields
                from app.models.orm.models import EvaluationResult
                result = MagicMock(spec=EvaluationResult)
                result.id = UUID('00000000-0000-0000-0000-000000000002')
                result.evaluation_id = test_evaluation_sync.id
                result.overall_score = 0.85
                result.raw_results = {"faithfulness": 0.9, "answer_relevancy": 0.8}
                result.dataset_sample_id = "0"
                result.input_data = {"query": "Test query", "context": "Test context"}
                result.output_data = {"answer": "Test answer"}
                result.processing_time_ms = 120
                result.created_at = now
                result.updated_at = now

                result.to_dict = lambda: {
                    "id": str(result.id),
                    "evaluation_id": str(result.evaluation_id),
                    "overall_score": result.overall_score,
                    "raw_results": result.raw_results,
                    "dataset_sample_id": result.dataset_sample_id,
                    "input_data": result.input_data,
                    "output_data": result.output_data,
                    "processing_time_ms": result.processing_time_ms,
                    "created_at": result.created_at.isoformat(),
                    "updated_at": result.updated_at.isoformat()
                }

                mock_results.return_value = [result]

                # Create mock metric scores with datetime fields
                from app.models.orm.models import MetricScore
                metric = MagicMock(spec=MetricScore)
                metric.id = UUID('00000000-0000-0000-0000-000000000003')
                metric.result_id = result.id
                metric.name = "faithfulness"
                metric.value = 0.9
                metric.weight = 1.0
                metric.meta_info = {"description": "Test metric"}
                metric.created_at = now
                metric.updated_at = now

                metric.to_dict = lambda: {
                    "id": str(metric.id),
                    "result_id": str(metric.result_id),
                    "name": metric.name,
                    "value": metric.value,
                    "weight": metric.weight,
                    "meta_info": metric.meta_info,
                    "created_at": metric.created_at.isoformat(),
                    "updated_at": metric.updated_at.isoformat()
                }

                mock_scores.return_value = [metric]

                evaluation_id_str = str(test_evaluation_sync.id)

                # Call API
                response = client.get(f"/{evaluation_id_str}/results")

                # Check response
                assert response.status_code == 200
                data = response.json()
                assert len(data) == 1
                assert data[0]["evaluation_id"] == str(test_evaluation_sync.id)
                assert data[0]["overall_score"] == 0.85
                assert "metric_scores" in data[0]
                assert len(data[0]["metric_scores"]) == 1
                assert data[0]["metric_scores"][0]["name"] == "faithfulness"