# File: tests/test_ragas_evaluation.py
import asyncio
import pytest
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.testclient import TestClient

from backend.app.db.models.orm.models import (
    User, UserRole, Dataset, Evaluation, EvaluationMethod,
    EvaluationStatus, MicroAgent, Prompt
)
from backend.app.services.evaluation_service import EvaluationService
from backend.app.utils.sample_dataset import SampleEvaluationBuilder
from backend.app.evaluation.methods.ragas import RagasEvaluationMethod


@pytest.fixture
async def test_user(db_session):
    """Create a test user."""
    from backend.app.db.repositories.base import BaseRepository

    user_repo = BaseRepository(User, db_session)
    test_user = await user_repo.create({
        "id": uuid.uuid4(),
        "external_id": f"test-user-{uuid.uuid4().hex[:8]}",
        "email": f"test-{uuid.uuid4().hex[:8]}@example.com",
        "display_name": "Test User",
        "role": UserRole.ADMIN,
        "is_active": True
    })

    return test_user


@pytest.fixture
async def sample_evaluation_setup(db_session, test_user):
    """Create a complete sample evaluation setup."""
    return await SampleEvaluationBuilder.create_sample_evaluation(
        db_session=db_session,
        user=test_user,
        method="ragas",
        num_samples=3,  # Small number for faster testing
        domain="general",
        include_contexts=True
    )


@pytest.mark.asyncio
async def test_ragas_method_initialization(db_session):
    """Test initialization of RAGAS evaluation method."""
    # Create the method
    ragas_method = RagasEvaluationMethod(db_session)

    # Check if it detects RAGAS availability correctly
    availability = ragas_method._check_ragas_available()
    print(f"RAGAS library available: {availability}")

    # Basic assertions
    assert hasattr(ragas_method, "ragas_available")
    assert ragas_method.method_name == "ragas"


@pytest.mark.asyncio
async def test_ragas_metrics_calculation(db_session):
    """Test calculation of RAGAS metrics."""
    # Create the method
    ragas_method = RagasEvaluationMethod(db_session)

    # Test data
    query = "What is the capital of France?"
    context = "France is a country in Western Europe. Its capital is Paris."
    answer = "The capital of France is Paris."
    ground_truth = "Paris is the capital of France."

    # Calculate metrics
    metrics = await ragas_method.calculate_metrics(
        input_data={"query": query, "context": context, "ground_truth": ground_truth},
        output_data={"answer": answer},
        config={"metrics": ["faithfulness", "answer_relevancy", "context_relevancy"]}
    )

    # Print metrics for inspection
    print("RAGAS metrics:", metrics)

    # Assertions
    assert isinstance(metrics, dict)
    assert len(metrics) > 0

    # Check metric values are between 0 and 1
    for name, value in metrics.items():
        assert 0 <= value <= 1, f"Metric {name} has value {value} which is not between 0 and 1"


@pytest.mark.asyncio
async def test_full_evaluation_flow(db_session, sample_evaluation_setup):
    """Test the full evaluation flow with RAGAS."""
    evaluation, dataset, prompt, microagent = sample_evaluation_setup

    # Create evaluation service
    evaluation_service = EvaluationService(db_session)

    # Run the evaluation
    summary = await SampleEvaluationBuilder.run_sample_evaluation(
        evaluation_service, evaluation.id
    )

    # Print summary for inspection
    print("Evaluation summary:", summary)

    # Refresh evaluation from database to get updated status
    updated_evaluation = await evaluation_service.get_evaluation(evaluation.id)

    # Get results
    results = await evaluation_service.get_evaluation_results(updated_evaluation.id)

    # Assertions
    assert updated_evaluation.status in [EvaluationStatus.COMPLETED, EvaluationStatus.RUNNING]
    assert len(results) > 0

    if updated_evaluation.status == EvaluationStatus.COMPLETED:
        # If completed, verify we have proper results
        for result in results:
            assert result.overall_score is not None

            # Get metric scores
            metric_scores = await evaluation_service.get_metric_scores(result.id)
            assert len(metric_scores) > 0

            # Check input and output data
            assert "query" in result.input_data
            assert "answer" in result.output_data


@pytest.mark.asyncio
async def test_fallback_metrics(db_session):
    """Test fallback metric calculations when RAGAS is not available."""
    # Create the method
    ragas_method = RagasEvaluationMethod(db_session)

    # Test data
    query = "What is the tallest mountain?"
    context = "Mount Everest is the tallest mountain in the world, standing at 8,849 meters."
    answer = "The tallest mountain is Mount Everest."

    # Calculate fallback metrics directly
    metrics = await ragas_method._calculate_fallback_metrics(
        query=query,
        context=context,
        answer=answer,
        ground_truth=None,
        enabled_metrics=["faithfulness", "answer_relevancy", "context_relevancy"]
    )

    # Print metrics for inspection
    print("Fallback metrics:", metrics)

    # Assertions
    assert "faithfulness" in metrics
    assert "answer_relevancy" in metrics
    assert "context_relevancy" in metrics

    # Check individual metric calculations
    faithfulness = ragas_method._calculate_faithfulness(answer, context)
    answer_relevancy = ragas_method._calculate_answer_relevancy(answer, query)
    context_relevancy = ragas_method._calculate_context_relevancy(context, query)

    assert faithfulness == metrics["faithfulness"]
    assert answer_relevancy == metrics["answer_relevancy"]
    assert context_relevancy == metrics["context_relevancy"]


@pytest.mark.asyncio
async def test_microagent_api_call(db_session):
    """Test the micro-agent API call functionality."""
    from backend.app.services.microagent_service import MicroAgentService

    # Create service
    service = MicroAgentService()

    # Test query
    response = await service.query_openai(
        prompt="You are a helpful assistant. Answer the question based on the provided context.",
        query="What is the capital of France?",
        context="France is a country in Western Europe. Its capital is Paris."
    )

    # Print response for inspection
    print("MicroAgent response:", response)

    # Assertions
    assert response.answer is not None
    assert "Paris" in response.answer
    assert response.processing_time_ms is not None


@pytest.mark.asyncio
async def test_evaluation_statistics(db_session, sample_evaluation_setup):
    """Test calculation of evaluation statistics."""
    evaluation, dataset, prompt, microagent = sample_evaluation_setup

    # Create evaluation service
    evaluation_service = EvaluationService(db_session)

    # Run the evaluation
    await SampleEvaluationBuilder.run_sample_evaluation(
        evaluation_service, evaluation.id
    )

    # Get statistics
    stats = await evaluation_service.get_evaluation_statistics(evaluation.id)

    # Print statistics for inspection
    print("Evaluation statistics:", stats)

    # Assertions
    assert "total_samples" in stats
    assert "avg_overall_score" in stats
    assert "metrics" in stats
    assert "processing_time" in stats

    # Check metrics
    assert len(stats["metrics"]) > 0
    for metric_name, metric_stats in stats["metrics"].items():
        assert "avg" in metric_stats
        assert "min" in metric_stats
        assert "max" in metric_stats