# File: tests/evaluation/test_ragas_method.py
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from app.evaluation.methods.ragas import RagasEvaluationMethod
from app.schema.evaluation_schema import EvaluationResultCreate

# pytestmark = pytest.mark.skipif(
#     True,  # Change to False when ready to enable these tests
#     reason="Database tests are currently disabled"
# )
@pytest.mark.asyncio
async def test_ragas_run_evaluation(
        db_session_sync, test_evaluation_sync, mock_httpx_client
):
    """Test RAGAS evaluation method run_evaluation."""
    # Initialize the method
    method = RagasEvaluationMethod(db_session_sync)

    # Mock get_microagent, get_dataset, and get_prompt methods
    microagent_mock = MagicMock()
    microagent_mock.api_endpoint = "http://test-endpoint.com"
    method.get_microagent = AsyncMock(return_value=microagent_mock)

    dataset_mock = MagicMock()
    method.get_dataset = AsyncMock(return_value=dataset_mock)

    prompt_mock = MagicMock()
    prompt_mock.content = "Test prompt {query} {context}"
    method.get_prompt = AsyncMock(return_value=prompt_mock)

    # Mock the necessary methods
    async def mock_load_dataset(*args, **kwargs):
        return [{"query": "test", "context": "test context", "ground_truth": "test answer"}]

    async def mock_call_api(*args, **kwargs):
        return {"answer": "test response", "processing_time_ms": 100}

    # Mock calculate_metrics to return test metrics
    async def mock_calculate_metrics(*args, **kwargs):
        return {"faithfulness": 0.8, "answer_relevancy": 0.7, "context_relevancy": 0.9}

    method.load_dataset = mock_load_dataset
    method._call_microagent_api = mock_call_api
    method.calculate_metrics = mock_calculate_metrics
    method._format_prompt = MagicMock(return_value="Formatted prompt")

    # Run the evaluation
    results = await method.run_evaluation(test_evaluation_sync)

    # Check results
    assert len(results) > 0
    for result in results:
        assert isinstance(result, EvaluationResultCreate)
        assert result.evaluation_id == test_evaluation_sync.id
        assert result.overall_score is not None
        assert result.raw_results is not None
        assert "faithfulness" in result.raw_results
        assert "answer_relevancy" in result.raw_results
        assert "context_relevancy" in result.raw_results


@pytest.mark.asyncio
async def test_ragas_calculate_metrics(db_session_sync, mock_httpx_client):
    """Test RAGAS evaluation method calculate_metrics."""
    # Initialize the method
    method = RagasEvaluationMethod(db_session_sync)

    # Mock the specific calculation methods to return known values
    method._calculate_faithfulness = MagicMock(return_value=0.8)
    method._calculate_answer_relevancy = MagicMock(return_value=0.7)
    method._calculate_context_relevancy = MagicMock(return_value=0.9)
    method._calculate_correctness = MagicMock(return_value=0.85)

    # Test input
    input_data = {
        "query": "What is machine learning?",
        "context": "Machine learning is a branch of AI focused on building systems that learn from data.",
        "ground_truth": "Machine learning is a branch of AI focused on learning from data."
    }
    output_data = {
        "answer": "Machine learning is a field of AI that enables systems to learn from data."
    }
    config = {
        "metrics": ["faithfulness", "answer_relevancy", "context_relevancy", "correctness"]
    }

    # Calculate metrics
    metrics = await method.calculate_metrics(input_data, output_data, config)

    # Check metrics
    assert metrics is not None
    assert "faithfulness" in metrics
    assert "answer_relevancy" in metrics
    assert "context_relevancy" in metrics
    assert "correctness" in metrics
    assert all(0 <= metrics[key] <= 1 for key in metrics)

    # Verify the mock methods were called
    method._calculate_faithfulness.assert_called_once()
    method._calculate_answer_relevancy.assert_called_once()
    method._calculate_context_relevancy.assert_called_once()
    method._calculate_correctness.assert_called_once()


@pytest.mark.asyncio
async def test_ragas_fallback_methods(db_session_sync):
    """Test RAGAS evaluation method fallback methods."""
    # Initialize the method
    method = RagasEvaluationMethod(db_session_sync)

    # Test faithfulness
    faithfulness = method._calculate_faithfulness(
        "Machine learning is a field of AI that enables systems to learn from data.",
        "Machine learning is a branch of AI focused on building systems that learn from data."
    )
    assert 0 <= faithfulness <= 1

    # Test answer relevancy
    answer_relevancy = method._calculate_answer_relevancy(
        "Machine learning is a field of AI that enables systems to learn from data.",
        "What is machine learning?"
    )
    assert 0 <= answer_relevancy <= 1

    # Test context relevancy
    context_relevancy = method._calculate_context_relevancy(
        "Machine learning is a branch of AI focused on building systems that learn from data.",
        "What is machine learning?"
    )
    assert 0 <= context_relevancy <= 1

    # Test correctness
    correctness = method._calculate_correctness(
        "Machine learning is a field of AI that enables systems to learn from data.",
        "Machine learning is a branch of AI focused on learning from data."
    )
    assert 0 <= correctness <= 1


@pytest.mark.asyncio
async def test_ragas_run_evaluation_empty_dataset(db_session_sync, test_evaluation_sync):
    """Test RAGAS evaluation with empty dataset."""
    # Mock the load_dataset method to return empty list
    with patch.object(
            RagasEvaluationMethod, 'load_dataset', return_value=[]
    ):
        # Initialize the method
        method = RagasEvaluationMethod(db_session_sync)

        # Run the evaluation
        results = await method.run_evaluation(test_evaluation_sync)

        # Check results
        assert len(results) == 0


@pytest.mark.asyncio
async def test_ragas_run_evaluation_api_error(db_session_sync, test_evaluation_sync):
    """Test RAGAS evaluation with API error."""
    # Initialize the method
    method = RagasEvaluationMethod(db_session_sync)

    # Mock the get_microagent method
    microagent_mock = MagicMock()
    microagent_mock.api_endpoint = "http://test-endpoint.com"
    method.get_microagent = AsyncMock(return_value=microagent_mock)

    # Mock the get_dataset method
    dataset_mock = MagicMock()
    method.get_dataset = AsyncMock(return_value=dataset_mock)

    # Mock the get_prompt method
    prompt_mock = MagicMock()
    prompt_mock.content = "Test prompt {query} {context}"
    method.get_prompt = AsyncMock(return_value=prompt_mock)

    # Mock load_dataset to return test data
    method.load_dataset = AsyncMock(return_value=[
        {"query": "Test query", "context": "Test context", "ground_truth": "Test ground truth"}
    ])

    # Mock _format_prompt to return a simple string
    method._format_prompt = MagicMock(return_value="Formatted prompt")

    # Mock _call_microagent_api to raise an exception
    method._call_microagent_api = AsyncMock(side_effect=Exception("API Error"))

    # Run the evaluation
    results = await method.run_evaluation(test_evaluation_sync)

    # Check results
    assert len(results) == 1
    assert results[0].overall_score == 0.0
    assert "error" in results[0].raw_results
    assert "API Error" in str(results[0].raw_results["error"])
    # Check results
    assert len(results) == 1
    assert results[0].overall_score == 0.0
    assert "error" in results[0].raw_results
    assert "API Error" in str(results[0].raw_results["error"])