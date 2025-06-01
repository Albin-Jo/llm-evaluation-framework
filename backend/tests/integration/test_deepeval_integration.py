#!/usr/bin/env python3
"""
DeepEval debugging script to test your setup
Run this to verify DeepEval is working correctly
"""

import asyncio
import logging
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_available_models():
    """Test which models are available"""
    test_models = [
        "gpt-4", "gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-3.5-turbo"
    ]

    for model in test_models:
        try:
            # Try to initialize a metric with this model
            metric = AnswerRelevancyMetric(
                threshold=0.7,
                model=model,
                include_reason=True
            )
            logger.info(f"✓ Model '{model}' is available")
        except Exception as e:
            logger.error(f"✗ Model '{model}' failed: {str(e)}")


def test_deepeval_basic():
    """Test basic DeepEval functionality"""
    try:
        # Create a simple test case
        test_case = LLMTestCase(
            input="What is the capital of France?",
            actual_output="The capital of France is Paris.",
            expected_output="Paris",
            context=["France is a country in Western Europe.", "Paris is the largest city in France."]
        )

        # Create metrics
        metrics = [
            AnswerRelevancyMetric(
                threshold=0.7,
                model="gpt-4o-mini",
                include_reason=True
            )
        ]

        # Create dataset
        dataset = EvaluationDataset(test_cases=[test_case])

        # Run evaluation
        logger.info("Running DeepEval test...")
        result = evaluate(
            test_cases=dataset.test_cases,
            metrics=metrics
        )

        logger.info("✓ DeepEval basic test passed!")
        logger.info(f"Result: {result}")

    except Exception as e:
        logger.error(f"✗ DeepEval basic test failed: {str(e)}")


async def test_async_deepeval():
    """Test async DeepEval functionality"""
    try:
        loop = asyncio.get_event_loop()

        def run_deepeval():
            test_case = LLMTestCase(
                input="What is 2+2?",
                actual_output="2+2 equals 4.",
                expected_output="4"
            )

            metrics = [
                AnswerRelevancyMetric(
                    threshold=0.7,
                    model="gpt-4o-mini",
                    include_reason=True
                )
            ]

            dataset = EvaluationDataset(test_cases=[test_case])

            return evaluate(
                test_cases=dataset.test_cases,
                metrics=metrics
            )

        logger.info("Running async DeepEval test...")
        result = await loop.run_in_executor(None, run_deepeval)
        logger.info("✓ Async DeepEval test passed!")

    except Exception as e:
        logger.error(f"✗ Async DeepEval test failed: {str(e)}")


def main():
    """Run all tests"""
    logger.info("=== DeepEval Debugging Script ===\n")

    logger.info("1. Testing available models...")
    test_available_models()

    logger.info("\n2. Testing basic DeepEval functionality...")
    test_deepeval_basic()

    logger.info("\n3. Testing async DeepEval functionality...")
    asyncio.run(test_async_deepeval())

    logger.info("\n=== Testing complete ===")


if __name__ == "__main__":
    main()