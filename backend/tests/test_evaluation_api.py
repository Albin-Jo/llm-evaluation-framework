# File: scripts/test_evaluation_api.py
import argparse
import asyncio
import httpx
import json
import logging
import uuid
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EvaluationAPITester:
    """Helper class for testing the evaluation API endpoints."""

    def __init__(self, base_url: str, api_key: str = None):
        """
        Initialize the API tester.

        Args:
            base_url: Base URL for the API
            api_key: Optional API key for authentication
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.http_client = httpx.AsyncClient(
            timeout=30.0,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}" if api_key else None
            }
        )

    async def create_evaluation(self, evaluation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new evaluation.

        Args:
            evaluation_data: Evaluation data

        Returns:
            Dict[str, Any]: Created evaluation
        """
        endpoint = f"{self.base_url}/api/v1/evaluations/"
        response = await self.http_client.post(endpoint, json=evaluation_data)
        response.raise_for_status()
        return response.json()

    async def get_evaluation(self, evaluation_id: str) -> Dict[str, Any]:
        """
        Get an evaluation by ID.

        Args:
            evaluation_id: Evaluation ID

        Returns:
            Dict[str, Any]: Evaluation details
        """
        endpoint = f"{self.base_url}/api/v1/evaluations/{evaluation_id}"
        response = await self.http_client.get(endpoint)
        response.raise_for_status()
        return response.json()

    async def list_evaluations(self, status: str = None) -> List[Dict[str, Any]]:
        """
        List evaluations with optional filtering.

        Args:
            status: Optional status filter

        Returns:
            List[Dict[str, Any]]: List of evaluations
        """
        endpoint = f"{self.base_url}/api/v1/evaluations/"
        params = {}
        if status:
            params["status"] = status

        response = await self.http_client.get(endpoint, params=params)
        response.raise_for_status()
        return response.json()

    async def start_evaluation(self, evaluation_id: str) -> Dict[str, Any]:
        """
        Start an evaluation.

        Args:
            evaluation_id: Evaluation ID

        Returns:
            Dict[str, Any]: Updated evaluation
        """
        endpoint = f"{self.base_url}/api/v1/evaluations/{evaluation_id}/start"
        response = await self.http_client.post(endpoint)
        response.raise_for_status()
        return response.json()

    async def get_evaluation_results(self, evaluation_id: str) -> List[Dict[str, Any]]:
        """
        Get evaluation results.

        Args:
            evaluation_id: Evaluation ID

        Returns:
            List[Dict[str, Any]]: Evaluation results
        """
        endpoint = f"{self.base_url}/api/v1/evaluations/{evaluation_id}/results"
        response = await self.http_client.get(endpoint)
        response.raise_for_status()
        return response.json()

    async def test_evaluation(self, evaluation_id: str, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test an evaluation with sample data.

        Args:
            evaluation_id: Evaluation ID
            test_data: Test data

        Returns:
            Dict[str, Any]: Test results
        """
        endpoint = f"{self.base_url}/api/v1/evaluations/{evaluation_id}/test"
        response = await self.http_client.post(endpoint, json=test_data)
        response.raise_for_status()
        return response.json()

    async def close(self):
        """Close the HTTP client."""
        await self.http_client.aclose()


async def run_sample_test(args):
    """Run a sample API test."""
    api_tester = EvaluationAPITester(args.base_url, args.api_key)

    try:
        # Step 1: List existing evaluations
        logger.info("Listing existing evaluations...")
        evaluations = await api_tester.list_evaluations()
        logger.info(f"Found {len(evaluations)} existing evaluations")

        # Step 2: Create a new evaluation if not using existing
        if args.evaluation_id:
            evaluation_id = args.evaluation_id
            logger.info(f"Using existing evaluation with ID: {evaluation_id}")
        else:
            # We need to have agent_id, dataset_id, and prompt_id to create a new evaluation
            # For simplicity in this script, we'll use the first evaluation as a template
            if evaluations:
                template = evaluations[0]
                logger.info(f"Using template from existing evaluation: {template['id']}")

                # Create a new evaluation based on the template
                new_eval_data = {
                    "name": f"RAGAS Test Evaluation {uuid.uuid4().hex[:8]}",
                    "description": "Test evaluation created by API script",
                    "method": "ragas",
                    "micro_agent_id": template["micro_agent_id"],
                    "dataset_id": template["dataset_id"],
                    "prompt_id": template["prompt_id"],
                    "config": {
                        "metrics": [
                            "faithfulness",
                            "response_relevancy",
                            "context_precision",
                            "context_recall",
                            "context_entity_recall",
                            "noise_sensitivity"
                        ],
                        "batch_size": 2
                    }
                }

                # Create the evaluation
                logger.info("Creating new evaluation...")
                created = await api_tester.create_evaluation(new_eval_data)
                evaluation_id = created["id"]
                logger.info(f"Created new evaluation with ID: {evaluation_id}")
            else:
                logger.error("No existing evaluations found and no evaluation ID provided")
                logger.error("Cannot create new evaluation without template")
                return

        # Step 3: Test the evaluation with sample data
        logger.info("Testing the evaluation with sample data...")
        test_data = {
            "query": "What is the capital of France?",
            "context": "France is a country in Western Europe with over 65 million people. Its capital is Paris, which is known as the City of Light.",
            "answer": "The capital of France is Paris, which is known as the City of Light.",
            "ground_truth": "The capital of France is Paris."
        }

        test_result = await api_tester.test_evaluation(evaluation_id, test_data)
        logger.info("Test completed successfully")
        logger.info(f"Overall score: {test_result['overall_score']:.4f}")

        # Print metrics
        logger.info("Metrics:")
        for metric_name, score in test_result["metrics"].items():
            logger.info(f"  - {metric_name}: {score:.4f}")

        # Step 4: Start the evaluation if requested
        if args.start:
            logger.info(f"Starting evaluation {evaluation_id}...")
            started = await api_tester.start_evaluation(evaluation_id)
            logger.info(f"Evaluation started with status: {started['status']}")

            # Optionally wait for results
            if args.wait:
                logger.info("Waiting for evaluation to complete...")

                # Poll for completion
                complete = False
                retries = 0
                max_retries = 30  # Wait up to 5 minutes

                while not complete and retries < max_retries:
                    await asyncio.sleep(10)  # Wait 10 seconds between checks

                    # Get evaluation status
                    eval_status = await api_tester.get_evaluation(evaluation_id)
                    logger.info(f"Current status: {eval_status['status']}")

                    if eval_status["status"] in ["completed", "failed", "cancelled"]:
                        complete = True
                    else:
                        retries += 1

                if complete:
                    # Get results
                    results = await api_tester.get_evaluation_results(evaluation_id)
                    logger.info(f"Evaluation completed with {len(results)} results")

                    # Print summary
                    if results:
                        total_score = sum(result["overall_score"] for result in results) / len(results)
                        logger.info(f"Average overall score: {total_score:.4f}")

                        # Collect metric scores
                        metric_scores = {}
                        for result in results:
                            for metric in result.get("metric_scores", []):
                                name = metric["name"]
                                value = metric["value"]
                                if name not in metric_scores:
                                    metric_scores[name] = []
                                metric_scores[name].append(value)

                        # Print metric averages
                        logger.info("Metric averages:")
                        for metric_name, values in metric_scores.items():
                            avg = sum(values) / len(values)
                            logger.info(f"  - {metric_name}: {avg:.4f}")
                else:
                    logger.warning("Evaluation did not complete within wait time")

    finally:
        # Close the HTTP client
        await api_tester.close()


def main():
    parser = argparse.ArgumentParser(description="Test the evaluation API endpoints")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Base URL for the API")
    parser.add_argument("--api-key", help="API key for authentication")
    parser.add_argument("--evaluation-id", help="Use existing evaluation ID instead of creating new")
    parser.add_argument("--start", action="store_true", help="Start the evaluation after testing")
    parser.add_argument("--wait", action="store_true", help="Wait for evaluation to complete")

    args = parser.parse_args()

    # Run the test
    asyncio.run(run_sample_test(args))


if __name__ == "__main__":
    main()