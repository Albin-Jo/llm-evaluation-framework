# File: tests/test_ragas_implementation.py
import asyncio
import logging
import os
import pytest
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the parent directory to sys.path to allow importing from backend
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.app.db.models.orm import EvaluationMethod
from backend.app.evaluation.factory import EvaluationMethodFactory
from backend.app.utils.sample_evaluation_runner import SampleEvaluationRunner


@pytest.fixture
def test_cases():
    """Fixture providing test cases for evaluation."""
    return [
        {
            "query": "What is the capital of France?",
            "context": "France is a country in Western Europe with over 65 million people. Its capital is Paris, which is known as the City of Light.",
            "ground_truth": "The capital of France is Paris.",
            "answer": "The capital of France is Paris."
        },
        {
            "query": "Who invented the light bulb?",
            "context": "Thomas Edison is credited with inventing the first commercially practical incandescent light bulb. However, some historians argue that there were several people who developed early versions, including Joseph Swan.",
            "ground_truth": "Thomas Edison is credited with inventing the first commercially practical light bulb.",
            "answer": "According to the context, Thomas Edison invented the light bulb, although some historians note that others like Joseph Swan developed early versions."
        },
        {
            "query": "What is machine learning?",
            "context": "Machine learning is a field of artificial intelligence that uses statistical techniques to give computer systems the ability to 'learn' from data, without being explicitly programmed.",
            "ground_truth": "Machine learning is a field of AI that enables systems to learn from data without explicit programming.",
            "answer": "Machine learning is a field of artificial intelligence that allows computer systems to learn from data without being explicitly programmed."
        },
        {
            "query": "What is the boiling point of water?",
            "context": "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at sea level, but this temperature decreases with increasing altitude.",
            "ground_truth": "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at sea level.",
            "answer": "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at sea level. The boiling point decreases as altitude increases."
        }
    ]


@pytest.mark.asyncio
async def test_ragas_fallback_metrics(test_cases):
    """Test the fallback metrics when RAGAS is not available."""
    from backend.app.evaluation.metrics.ragas_metrics import (
        calculate_faithfulness,
        calculate_response_relevancy,
        calculate_context_precision,
        calculate_context_recall,
        calculate_context_entity_recall,
        calculate_noise_sensitivity
    )

    # Force fallback mode for testing
    import backend.app.evaluation.metrics.ragas_metrics as ragas_metrics
    ragas_metrics.RAGAS_AVAILABLE = False

    test_case = test_cases[0]
    query = test_case["query"]
    context = test_case["context"]
    answer = test_case["answer"]
    ground_truth = test_case["ground_truth"]

    # Test each metric
    faithfulness = await calculate_faithfulness(answer, context)
    response_relevancy = await calculate_response_relevancy(answer, query)
    context_precision = await calculate_context_precision(context, query)
    context_recall = await calculate_context_recall(context, query, ground_truth)
    entity_recall = await calculate_context_entity_recall(context, ground_truth)
    noise_sensitivity = await calculate_noise_sensitivity(query, answer, context, ground_truth)

    # Log the results
    logger.info(f"Faithfulness: {faithfulness:.4f}")
    logger.info(f"Response Relevancy: {response_relevancy:.4f}")
    logger.info(f"Context Precision: {context_precision:.4f}")
    logger.info(f"Context Recall: {context_recall:.4f}")
    logger.info(f"Entity Recall: {entity_recall:.4f}")
    logger.info(f"Noise Sensitivity: {noise_sensitivity:.4f}")

    # Assert values are within expected ranges
    assert 0 <= faithfulness <= 1, f"Faithfulness score {faithfulness} is out of range [0,1]"
    assert 0 <= response_relevancy <= 1, f"Response Relevancy score {response_relevancy} is out of range [0,1]"
    assert 0 <= context_precision <= 1, f"Context Precision score {context_precision} is out of range [0,1]"
    assert 0 <= context_recall <= 1, f"Context Recall score {context_recall} is out of range [0,1]"
    assert 0 <= entity_recall <= 1, f"Entity Recall score {entity_recall} is out of range [0,1]"
    assert 0 <= noise_sensitivity <= 1, f"Noise Sensitivity score {noise_sensitivity} is out of range [0,1]"

    # For a perfect match test case, expect high scores
    if answer == ground_truth:
        assert faithfulness > 0.5, f"Expected high faithfulness score for perfect match, got {faithfulness}"
        assert response_relevancy > 0.5, f"Expected high relevancy score for perfect match, got {response_relevancy}"
        assert noise_sensitivity < 0.5, f"Expected low noise sensitivity for perfect match, got {noise_sensitivity}"


@pytest.mark.asyncio
async def test_evaluation_method_factory():
    """Test the evaluation method factory."""
    # Test creating a RAGAS evaluation method
    method_handler = EvaluationMethodFactory.create(EvaluationMethod.RAGAS, None)
    assert method_handler.method_name == "ragas", f"Expected 'ragas', got '{method_handler.method_name}'"

    # Test creating a CUSTOM evaluation method
    method_handler = EvaluationMethodFactory.create(EvaluationMethod.CUSTOM, None)
    assert method_handler.method_name == "custom", f"Expected 'custom', got '{method_handler.method_name}'"

    # Test invalid method
    with pytest.raises(ValueError):
        # Create fake evaluation method enum
        class FakeEvaluationMethod:
            INVALID = "invalid"

        EvaluationMethodFactory.create(FakeEvaluationMethod.INVALID, None)


@pytest.mark.asyncio
async def test_sample_evaluation_runner(test_cases):
    """Test the sample evaluation runner."""
    runner = SampleEvaluationRunner()

    # Run evaluation with test cases
    results = await runner.run_sample_evaluation(method="ragas", test_cases=test_cases)

    # Verify results structure
    assert "summary" in results, "Results should contain a summary"
    assert "results" in results, "Results should contain detailed results"

    summary = results["summary"]
    assert summary["method"] == "ragas", f"Expected method 'ragas', got '{summary['method']}'"
    assert summary["test_cases"] == len(
        test_cases), f"Expected {len(test_cases)} test cases, got {summary['test_cases']}"
    assert "overall_score" in summary, "Summary should contain an overall score"
    assert "metrics" in summary, "Summary should contain metrics"

    # Check all expected metrics are calculated
    expected_metrics = ["faithfulness", "response_relevancy", "context_precision"]
    for metric in expected_metrics:
        assert metric in summary["metrics"], f"Expected metric '{metric}' not found in results"

    # Verify detailed results
    detailed_results = results["results"]
    assert len(detailed_results) == len(
        test_cases), f"Expected {len(test_cases)} detailed results, got {len(detailed_results)}"

    for result in detailed_results:
        assert "overall_score" in result, "Result should contain an overall score"
        assert "input_data" in result, "Result should contain input data"
        assert "output_data" in result, "Result should contain output data"
        assert "metric_scores" in result, "Result should contain metric scores"

        # Check metric scores structure
        metric_scores = result["metric_scores"]
        assert len(metric_scores) > 0, "Metric scores should not be empty"

        for metric_score in metric_scores:
            assert "name" in metric_score, "Metric score should have a name"
            assert "value" in metric_score, "Metric score should have a value"
            assert 0 <= metric_score["value"] <= 1, f"Metric value {metric_score['value']} out of range [0,1]"


if __name__ == "__main__":
    # Run the tests directly if the script is executed
    asyncio.run(test_ragas_fallback_metrics([
        {
            "query": "What is the capital of France?",
            "context": "France is a country in Western Europe. Its capital is Paris.",
            "ground_truth": "The capital of France is Paris.",
            "answer": "The capital of France is Paris."
        }
    ]))
    print("Fallback metrics test completed successfully!")

    asyncio.run(test_evaluation_method_factory())
    print("Evaluation method factory test completed successfully!")

    asyncio.run(test_sample_evaluation_runner([
        {
            "query": "What is the capital of France?",
            "context": "France is a country in Western Europe. Its capital is Paris.",
            "ground_truth": "The capital of France is Paris.",
            "answer": "The capital of France is Paris."
        }
    ]))
    print("Sample evaluation runner test completed successfully!")
    print("All tests completed successfully!")