# File: tests/evaluation/test_deepeval_method.py
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from app.evaluation.methods.deepeval import DeepEvalEvaluationMethod
from app.schema.evaluation_schema import EvaluationResultCreate


@pytest.mark.asyncio
async def test_deepeval_run_evaluation(
        db_session_sync, test_evaluation_sync, mock_httpx_client
):
    """Test DeepEval evaluation method run_evaluation."""
    # Initialize the method
    method = DeepEvalEvaluationMethod(db_session_sync)

    # Add debug to see evaluation_id
    print(f"Test evaluation ID: {test_evaluation_sync.id}")

    # Mock the get_microagent, get_dataset, and get_prompt methods
    method.get_microagent = AsyncMock(return_value=test_evaluation_sync.micro_agent)
    method.get_dataset = AsyncMock(return_value=test_evaluation_sync.dataset)
    method.get_prompt = AsyncMock(return_value=test_evaluation_sync.prompt)

    # Mock the necessary methods with more explicit return values
    async def mock_load_dataset(*args, **kwargs):
        return [{"query": "test", "context": "test context", "ground_truth": "test answer"}]

    async def mock_call_api(*args, **kwargs):
        return {"answer": "test response", "processing_time_ms": 100}

    method.load_dataset = mock_load_dataset
    method._call_microagent_api = mock_call_api

    # Add a patch for the calculate_metrics method to ensure it returns valid metrics
    with patch.object(method, 'calculate_metrics') as mock_calculate_metrics:
        mock_calculate_metrics.return_value = [
            {
                "name": "bias",
                "value": 0.9,
                "weight": 1.0,
                "description": "Measures the absence of bias in the generated response"
            },
            {
                "name": "relevance",
                "value": 0.8,
                "weight": 1.0,
                "description": "Measures how relevant the response is to the query"
            }
        ]

        # Run the evaluation
        results = await method.run_evaluation(test_evaluation_sync)

        # Print debug information
        print(f"Results: {results}")

        # Check results
        assert len(results) > 0

        # If results are returned, check their structure
        if results:
            for result in results:
                assert isinstance(result, EvaluationResultCreate)
                assert result.evaluation_id == test_evaluation_sync.id
                assert result.overall_score is not None
                assert result.raw_results is not None
                assert len(result.metric_scores) > 0


@pytest.mark.asyncio
async def test_deepeval_calculate_metrics(db_session_sync):
    """Test DeepEval evaluation method calculate_metrics."""
    # Initialize the method
    method = DeepEvalEvaluationMethod(db_session_sync)

    # Test input
    input_data = {
        "query": "What is machine learning?",
        "context": "Machine learning is a branch of AI focused on building systems that learn from data.",
        "ground_truth": "Machine learning is a branch of AI focused on learning from data.",
        "references": ["Machine learning is a field of artificial intelligence."]
    }
    output_data = {
        "answer": "Machine learning is a field of AI that enables systems to learn from data."
    }
    config = {
        "metrics": ["bias", "relevance", "coherence", "groundedness", "fluency"]
    }

    # Calculate metrics
    metrics = await method.calculate_metrics(input_data, output_data, config)

    # Check metrics
    assert metrics is not None
    assert len(metrics) > 0
    for metric in metrics:
        assert "name" in metric
        assert "value" in metric
        assert "weight" in metric
        assert "description" in metric
        assert 0 <= metric["value"] <= 1


@pytest.mark.asyncio
async def test_deepeval_metric_calculations(db_session_sync):
    """Test DeepEval individual metric calculations."""
    # Initialize the method
    method = DeepEvalEvaluationMethod(db_session_sync)

    # Test bias
    answer = "Machine learning is definitely the best field in artificial intelligence."
    bias_score = await method._calculate_bias(answer)
    assert 0 <= bias_score <= 1

    # Test relevance
    answer = "Machine learning is a field of AI that enables systems to learn from data."
    query = "What is machine learning?"
    relevance_score = await method._calculate_relevance(answer, query)
    assert 0 <= relevance_score <= 1

    # Test coherence
    answer = "Machine learning is a field of AI. It enables systems to learn from data. This learning can be supervised or unsupervised."
    coherence_score = await method._calculate_coherence(answer)
    assert 0 <= coherence_score <= 1

    # Test groundedness
    answer = "Machine learning is a field of AI that enables systems to learn from data."
    context = "Machine learning is a branch of AI focused on building systems that learn from data."
    groundedness_score = await method._calculate_groundedness(answer, context)
    assert 0 <= groundedness_score <= 1

    # Test fluency
    answer = "Machine learning is a field of artificial intelligence that focuses on algorithms and models that can learn from and make predictions based on data."
    fluency_score = await method._calculate_fluency(answer)
    assert 0 <= fluency_score <= 1