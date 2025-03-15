# File: tests/services/test_evaluation_service.py
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from uuid import uuid4, UUID
from datetime import datetime, UTC

from app.models.orm.models import EvaluationStatus, EvaluationMethod
from app.schema.evaluation_schema import EvaluationCreate, EvaluationUpdate, EvaluationResultCreate
from app.services.evaluation_service import EvaluationService


@pytest.mark.asyncio
async def test_create_evaluation(
        db_session_sync, test_user_sync, test_microagent_sync, test_dataset_sync, test_prompt_sync
):
    """Test creating an evaluation."""
    # Initialize service
    service = EvaluationService(db_session_sync)

    # Mock repository methods with AsyncMock
    service.micro_agent_repo.get = AsyncMock(return_value=test_microagent_sync)
    service.dataset_repo.get = AsyncMock(return_value=test_dataset_sync)
    service.prompt_repo.get = AsyncMock(return_value=test_prompt_sync)

    # Create a mock evaluation to return
    eval_id = uuid4()
    mock_eval = MagicMock()
    mock_eval.id = eval_id
    mock_eval.name = "Test Service Evaluation"
    mock_eval.method = EvaluationMethod.RAGAS
    mock_eval.status = EvaluationStatus.PENDING
    mock_eval.created_by_id = test_user_sync.id
    mock_eval.micro_agent_id = test_microagent_sync.id
    mock_eval.dataset_id = test_dataset_sync.id
    mock_eval.prompt_id = test_prompt_sync.id

    service.evaluation_repo.create = AsyncMock(return_value=mock_eval)

    # Create evaluation data
    evaluation_data = EvaluationCreate(
        name="Test Service Evaluation",
        description="Testing evaluation service",
        method=EvaluationMethod.RAGAS,
        micro_agent_id=test_microagent_sync.id,
        dataset_id=test_dataset_sync.id,
        prompt_id=test_prompt_sync.id,
        config={"metrics": ["faithfulness", "answer_relevancy"]}
    )

    # Create evaluation
    evaluation = await service.create_evaluation(evaluation_data, test_user_sync)

    # Check evaluation
    assert evaluation is not None
    assert evaluation.name == "Test Service Evaluation"
    assert evaluation.method == EvaluationMethod.RAGAS
    assert evaluation.status == EvaluationStatus.PENDING
    assert evaluation.created_by_id == test_user_sync.id
    assert evaluation.micro_agent_id == test_microagent_sync.id
    assert evaluation.dataset_id == test_dataset_sync.id
    assert evaluation.prompt_id == test_prompt_sync.id


@pytest.mark.asyncio
async def test_get_evaluation(db_session_sync, test_evaluation_sync):
    """Test getting an evaluation."""
    # Initialize service
    service = EvaluationService(db_session_sync)

    # Mock repository method with AsyncMock
    service.evaluation_repo.get = AsyncMock(return_value=test_evaluation_sync)

    # Get evaluation
    evaluation = await service.get_evaluation(test_evaluation_sync.id)

    # Check evaluation
    assert evaluation is not None
    assert evaluation.id == test_evaluation_sync.id
    assert evaluation.name == test_evaluation_sync.name


@pytest.mark.asyncio
async def test_update_evaluation(db_session_sync, test_evaluation_sync):
    """Test updating an evaluation."""
    # Initialize service
    service = EvaluationService(db_session_sync)

    # Mock repository methods with AsyncMock
    service.evaluation_repo.get = AsyncMock(return_value=test_evaluation_sync)

    # Create updated evaluation mock
    updated_evaluation = MagicMock()
    updated_evaluation.name = "Updated Evaluation Name"
    updated_evaluation.description = "Updated description"
    updated_evaluation.status = EvaluationStatus.RUNNING

    service.evaluation_repo.update = AsyncMock(return_value=updated_evaluation)

    # Update data
    update_data = EvaluationUpdate(
        name="Updated Evaluation Name",
        description="Updated description",
        status=EvaluationStatus.RUNNING
    )

    # Update evaluation
    evaluation = await service.update_evaluation(test_evaluation_sync.id, update_data)

    # Check evaluation
    assert evaluation is not None
    assert evaluation.name == "Updated Evaluation Name"
    assert evaluation.description == "Updated description"
    assert evaluation.status == EvaluationStatus.RUNNING


@pytest.mark.asyncio
async def test_list_evaluations(db_session_sync, test_evaluation_sync):
    """Test listing evaluations."""
    # Initialize service
    service = EvaluationService(db_session_sync)

    # Mock repository method with AsyncMock
    service.evaluation_repo.get_multi = AsyncMock(return_value=[test_evaluation_sync])

    # List evaluations
    evaluations = await service.list_evaluations()

    # Check evaluations
    assert evaluations is not None
    assert len(evaluations) >= 1
    assert any(e.id == test_evaluation_sync.id for e in evaluations)


@pytest.mark.asyncio
async def test_start_evaluation(db_session_sync, test_evaluation_sync):
    """Test starting an evaluation."""
    # Initialize service
    service = EvaluationService(db_session_sync)

    # Mock repository methods with AsyncMock
    service.evaluation_repo.get = AsyncMock(return_value=test_evaluation_sync)

    # Create started evaluation mock
    started_evaluation = MagicMock()
    started_evaluation.status = EvaluationStatus.RUNNING
    started_evaluation.start_time = datetime.now(UTC)

    service.evaluation_repo.update = AsyncMock(return_value=started_evaluation)

    # Start evaluation
    evaluation = await service.start_evaluation(test_evaluation_sync.id)

    # Check evaluation
    assert evaluation is not None
    assert evaluation.status == EvaluationStatus.RUNNING
    assert evaluation.start_time is not None


@pytest.mark.asyncio
async def test_complete_evaluation(db_session_sync, test_evaluation_sync):
    """Test completing an evaluation."""
    # Initialize service
    service = EvaluationService(db_session_sync)

    # Mock repository methods with AsyncMock
    # First create a running evaluation
    running_evaluation = MagicMock()
    running_evaluation.status = EvaluationStatus.RUNNING
    service.evaluation_repo.get = AsyncMock(return_value=running_evaluation)

    # Create completed evaluation mock
    completed_evaluation = MagicMock()
    completed_evaluation.status = EvaluationStatus.COMPLETED
    completed_evaluation.end_time = datetime.now(UTC)

    service.evaluation_repo.update = AsyncMock(return_value=completed_evaluation)

    # Complete evaluation
    evaluation = await service.complete_evaluation(test_evaluation_sync.id, success=True)

    # Check evaluation
    assert evaluation is not None
    assert evaluation.status == EvaluationStatus.COMPLETED
    assert evaluation.end_time is not None


@pytest.mark.asyncio
async def test_create_evaluation_result(db_session_sync, test_evaluation_sync):
    """Test creating an evaluation result."""
    # Initialize service
    service = EvaluationService(db_session_sync)

    # Mock repository methods with AsyncMock
    result_mock = MagicMock()
    result_mock.id = uuid4()
    result_mock.evaluation_id = test_evaluation_sync.id

    service.result_repo.create = AsyncMock(return_value=result_mock)
    service.metric_repo.create = AsyncMock(return_value=MagicMock())

    # Create result data
    result_data = EvaluationResultCreate(
        evaluation_id=test_evaluation_sync.id,
        overall_score=0.85,
        raw_results={"faithfulness": 0.9, "answer_relevancy": 0.8},
        dataset_sample_id="0",
        input_data={"query": "Test query", "context": "Test context"},
        output_data={"answer": "Test answer"},
        processing_time_ms=120,
        metric_scores=[
            {"name": "faithfulness", "value": 0.9, "weight": 1.0},
            {"name": "answer_relevancy", "value": 0.8, "weight": 1.0}
        ]
    )

    # Create result
    result = await service.create_evaluation_result(result_data)

    # Check result
    assert result is not None
    assert result.evaluation_id == test_evaluation_sync.id


@pytest.mark.asyncio
async def test_get_evaluation_results(db_session_sync, test_evaluation_sync):
    """Test getting evaluation results."""
    # Initialize service
    service = EvaluationService(db_session_sync)

    # Mock repository method with AsyncMock
    result_mock = MagicMock()
    result_mock.evaluation_id = test_evaluation_sync.id

    service.result_repo.get_multi = AsyncMock(return_value=[result_mock])

    # Get results
    results = await service.get_evaluation_results(test_evaluation_sync.id)

    # Check results
    assert results is not None
    assert len(results) >= 1
    assert results[0].evaluation_id == test_evaluation_sync.id


@pytest.mark.asyncio
async def test_get_metric_scores(db_session_sync, test_evaluation_sync):
    """Test getting metric scores."""
    # Initialize service
    service = EvaluationService(db_session_sync)

    # Mock repository method with AsyncMock
    result_id = uuid4()
    score_mock1 = MagicMock()
    score_mock1.name = "faithfulness"
    score_mock1.result_id = result_id

    score_mock2 = MagicMock()
    score_mock2.name = "answer_relevancy"
    score_mock2.result_id = result_id

    service.metric_repo.get_multi = AsyncMock(return_value=[score_mock1, score_mock2])

    # Get metric scores
    scores = await service.get_metric_scores(result_id)

    # Check scores
    assert scores is not None
    assert len(scores) == 2
    assert scores[0].result_id == result_id
    assert any(s.name == "faithfulness" for s in scores)
    assert any(s.name == "answer_relevancy" for s in scores)