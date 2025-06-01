import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional

from backend.app.db.models.orm import EvaluationMethod, EvaluationStatus
from backend.app.db.schema.evaluation_schema import EvaluationResultCreate
from backend.app.evaluation.factory import EvaluationMethodFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SampleEvaluationRunner:
    """Utility for running sample evaluations with the updated RAGAS implementation."""

    def __init__(self, db_session=None):
        """
        Initialize the sample evaluation runner.

        Args:
            db_session: Optional database session
        """
        self.db_session = db_session

    async def run_sample_evaluation(
            self,
            method: str = "ragas",
            test_cases_file: Optional[str] = None,
            test_cases: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Run a sample evaluation with provided test cases.

        Args:
            method: Evaluation method name
            test_cases_file: Path to test cases JSON file
            test_cases: List of test cases (alternative to file)

        Returns:
            Dict[str, Any]: Evaluation results
        """
        # Create a mock evaluation object
        evaluation = self._create_mock_evaluation(method)

        # Load test cases
        if test_cases is None:
            if test_cases_file:
                with open(test_cases_file, 'r') as f:
                    test_cases = json.load(f)
            else:
                # Use default test cases
                test_cases = self._get_default_test_cases()

        # Create evaluation method
        eval_method = getattr(EvaluationMethod, method.upper()) if hasattr(EvaluationMethod,
                                                                           method.upper()) else EvaluationMethod.RAGAS
        method_handler = EvaluationMethodFactory.create(eval_method, self.db_session)

        # Run evaluation
        logger.info(f"Running sample {method} evaluation with {len(test_cases)} test cases")
        start_time = datetime.now()

        results: List[EvaluationResultCreate] = []

        # Process each test case
        for i, test_case in enumerate(test_cases):
            logger.info(f"Processing test case {i + 1}/{len(test_cases)}: {test_case.get('query', '')[:50]}...")

            # Extract data from test case
            query = test_case.get("query", "")
            context = test_case.get("context", test_case.get("contexts", ""))
            ground_truth = test_case.get("ground_truth", test_case.get("expected_answer", ""))
            answer = test_case.get("answer", "")

            # If answer not provided, generate a mock answer
            if not answer:
                answer = f"This is a sample answer for the query: {query}"

            # Calculate metrics
            try:
                metrics = await method_handler.calculate_metrics(
                    input_data={
                        "query": query,
                        "context": context,
                        "ground_truth": ground_truth
                    },
                    output_data={"answer": answer},
                    config=evaluation.config or {}
                )

                # Calculate overall score
                overall_score = sum(metrics.values()) / len(metrics) if metrics else 0.0

                # Create metric scores for the result
                metric_scores = [
                    {
                        "name": name,
                        "value": value,
                        "weight": 1.0,
                        "meta_info": {"description": method_handler._get_metric_description(name)}
                    }
                    for name, value in metrics.items()
                ]

                # Create result
                result = {
                    "evaluation_id": evaluation.id,
                    "overall_score": overall_score,
                    "raw_results": metrics,
                    "dataset_sample_id": str(i),
                    "input_data": {
                        "query": query,
                        "context": context,
                        "ground_truth": ground_truth
                    },
                    "output_data": {"answer": answer},
                    "processing_time_ms": 0,  # Mock processing time
                    "metric_scores": metric_scores
                }

                results.append(result)

            except Exception as e:
                logger.error(f"Error processing test case {i + 1}: {e}")

        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()

        # Create summary
        summary = {
            "method": method,
            "test_cases": len(test_cases),
            "results": len(results),
            "processing_time_seconds": total_time,
            "overall_score": sum(r["overall_score"] for r in results) / len(results) if results else 0,
            "metrics": {}
        }

        # Calculate metric averages
        if results:
            all_metrics = {}
            for result in results:
                for metric in result.get("metric_scores", []):
                    name = metric["name"]
                    if name not in all_metrics:
                        all_metrics[name] = []
                    all_metrics[name].append(metric["value"])

            for metric_name, values in all_metrics.items():
                summary["metrics"][metric_name] = {
                    "average": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }

        return {
            "summary": summary,
            "results": results
        }

    def _create_mock_evaluation(self, method: str = "ragas") -> Any:
        """
        Create a mock evaluation object for testing.

        Args:
            method: Evaluation method name

        Returns:
            Any: Mock evaluation object
        """
        eval_method = getattr(EvaluationMethod, method.upper()) if hasattr(EvaluationMethod,
                                                                           method.upper()) else EvaluationMethod.RAGAS

        # Create a simple object with the necessary attributes
        class MockEvaluation:
            def __init__(self):
                self.id = uuid.uuid4()
                self.method = eval_method
                self.status = EvaluationStatus.RUNNING
                self.config = {
                    "metrics": [
                        "faithfulness",
                        "response_relevancy",
                        "context_precision"
                    ],
                    "batch_size": 2
                }

        return MockEvaluation()

    def _get_default_test_cases(self) -> List[Dict[str, Any]]:
        """
        Get default test cases for sample evaluation.

        Returns:
            List[Dict[str, Any]]: Default test cases
        """
        return [
            {
                "query": "What is the capital of France?",
                "context": "France is a country in Western Europe with over 65 million people. Its capital is Paris, which is known as the City of Light. Paris is famous for the Eiffel Tower and Louvre Museum.",
                "ground_truth": "The capital of France is Paris.",
                "answer": "The capital of France is Paris, which is known as the City of Light."
            },
            {
                "query": "Who wrote 'Pride and Prejudice'?",
                "context": "Pride and Prejudice is a romantic novel of manners written by Jane Austen in 1813. The novel follows the character development of Elizabeth Bennet, the dynamic protagonist of the book who learns about the repercussions of hasty judgments.",
                "ground_truth": "Jane Austen wrote 'Pride and Prejudice'.",
                "answer": "Jane Austen wrote 'Pride and Prejudice', which was published in 1813."
            },
            {
                "query": "What is the tallest mountain in the world?",
                "context": "Mount Everest is Earth's highest mountain above sea level, located in the Mahalangur Himal sub-range of the Himalayas. The China–Nepal border runs across its summit point. Its elevation of 29,031.7 feet was most recently established in 2020 by the Nepali and Chinese authorities.",
                "ground_truth": "Mount Everest is the tallest mountain in the world.",
                "answer": "The tallest mountain in the world is Mount Everest, which has an elevation of 29,031.7 feet."
            },
            {
                "query": "How does photosynthesis work?",
                "context": "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize nutrients from carbon dioxide and water. Photosynthesis in plants generally involves the green pigment chlorophyll and generates oxygen as a by-product.",
                "ground_truth": "Photosynthesis is a process where plants use sunlight, water, and carbon dioxide to create oxygen and energy in the form of sugar.",
                "answer": "Photosynthesis is the process where green plants use sunlight to convert carbon dioxide and water into nutrients. This process involves chlorophyll and produces oxygen as a by-product."
            },
            {
                "query": "When was the first iPhone released?",
                "context": "The history of iPhone began with a request from Apple Inc. CEO Steve Jobs to the company's engineers, asking them to investigate the use of touchscreen devices and tablet computers. The iPhone was eventually released in the United States on June 29, 2007, at the price of $499 for the 4 GB model and $599 for the 8 GB model.",
                "ground_truth": "The first iPhone was released on June 29, 2007.",
                "answer": "The first iPhone was released in the United States on June 29, 2007. It was priced at $499 for the 4 GB model and $599 for the 8 GB model."
            }
        ]


async def run_sample():
    """Run a sample evaluation for testing."""
    runner = SampleEvaluationRunner()
    results = await runner.run_sample_evaluation(method="ragas")

    # Print summary
    print("\n===== EVALUATION SUMMARY =====")
    summary = results["summary"]
    print(f"Method: {summary['method']}")
    print(f"Test cases: {summary['test_cases']}")
    print(f"Overall score: {summary['overall_score']:.4f}")
    print(f"Processing time: {summary['processing_time_seconds']:.2f} seconds")

    print("\n===== METRIC AVERAGES =====")
    for metric_name, stats in summary["metrics"].items():
        print(f"{metric_name}: {stats['average']:.4f} (min: {stats['min']:.4f}, max: {stats['max']:.4f})")

    # Print detailed results
    print("\n===== DETAILED RESULTS =====")
    for i, result in enumerate(results["results"]):
        print(f"\nTest Case {i + 1}:")
        print(f"Query: {result['input_data']['query']}")
        print(f"Answer: {result['output_data']['answer'][:100]}...")
        print(f"Overall Score: {result['overall_score']:.4f}")
        print("Metrics:")
        for metric in result["metric_scores"]:
            print(f"  - {metric['name']}: {metric['value']:.4f}")


if __name__ == "__main__":
    asyncio.run(run_sample())
