# File: ragas_evaluation.py
"""
Script to test RAGAS (Retrieval Augmented Generation Assessment) evaluation
using the official RAGAS package and Azure OpenAI API.
"""

import argparse
import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any

import pandas as pd
import requests
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
# Import RAGAS with updated imports
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithoutReference,
    LLMContextRecall,
    ContextEntityRecall,
    NoiseSensitivity
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ragas_evaluation.log.json"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Constants
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://azcorpstgapi.qatarairways.com.qa")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "98a26ff989784c8fa8212d80e704c829")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "qr-oai-4om")
API_VERSION = os.getenv("API_VERSION", "2024-06-01")


@dataclass
class RAGTestCase:
    """Represents a single RAG test case"""
    question: str
    contexts: List[str]  # Using 'contexts' to match RAGAS naming
    answer: str = ""  # Will be populated during execution
    ground_truth: str = ""  # Optional ground truth answer
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the test case to a dictionary"""
        return {
            "question": self.question,
            "contexts": self.contexts,
            "answer": self.answer,
            "ground_truth": self.ground_truth,
            "metadata": self.metadata
        }


class AzureOpenAIClient:
    """Client for interacting with Azure OpenAI API directly (for generation)"""

    def __init__(
            self,
            endpoint: str = AZURE_OPENAI_ENDPOINT,
            api_key: str = AZURE_OPENAI_API_KEY,
            deployment: str = AZURE_OPENAI_DEPLOYMENT,
            api_version: str = API_VERSION
    ):
        """
        Initialize the Azure OpenAI client.

        Args:
            endpoint: Azure OpenAI endpoint URL
            api_key: Azure OpenAI API key
            deployment: Model deployment name
            api_version: API version to use
        """
        self.endpoint = endpoint
        self.api_key = api_key
        self.deployment = deployment
        self.api_version = api_version

        if not self.api_key:
            logger.warning("API key not provided. Set AZURE_OPENAI_API_KEY environment variable.")

        logger.info(f"Initialized Azure OpenAI client with deployment: {self.deployment}")

    def get_completion(
            self,
            messages: List[Dict[str, str]],
            max_tokens: int = 500,
            temperature: float = 0.0
    ) -> Dict[str, Any]:
        """
        Get a completion from the Azure OpenAI API.

        Args:
            messages: List of message dictionaries with role and content
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)

        Returns:
            Response from the API as a dictionary

        Raises:
            Exception: If the API request fails
        """
        url = f"{self.endpoint}/openai/deployments/{self.deployment}/chat/completions"
        params = {"api-version": self.api_version}
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        try:
            response = requests.post(
                url,
                params=params,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response body: {e.response.text}")
            raise


class RAGASEvaluator:
    """Class to evaluate RAG systems using RAGAS metrics"""

    def __init__(self, api_key: str, endpoint: str, deployment: str, api_version: str):
        """
        Initialize the RAGAS evaluator with Azure OpenAI API details.

        Args:
            api_key: Azure OpenAI API key
            endpoint: Azure OpenAI endpoint
            deployment: Model deployment name
            api_version: API version
        """
        self.api_key = api_key
        self.endpoint = endpoint
        self.deployment = deployment
        self.api_version = api_version

        # Initialize Azure client for generation
        self.azure_client = AzureOpenAIClient(
            endpoint=endpoint,
            api_key=api_key,
            deployment=deployment,
            api_version=api_version
        )

        # Initialize RAGAS evaluator with Azure OpenAI
        self.llm = AzureChatOpenAI(
            openai_api_key=api_key,
            azure_endpoint=endpoint,
            azure_deployment=deployment,
            api_version=api_version,
            temperature=0.0
        )

        # Wrap in the RAGAS LLM interface
        self.ragas_llm = LangchainLLMWrapper(self.llm)

        # Initialize scorers with updated metric names
        self.scorers = {
            "faithfulness": Faithfulness(llm=self.ragas_llm),
            "response_relevancy": ResponseRelevancy(llm=self.ragas_llm),
            "context_precision": LLMContextPrecisionWithoutReference(llm=self.ragas_llm),
            "context_recall": LLMContextRecall(llm=self.ragas_llm),
            "context_entity_recall": ContextEntityRecall(llm=self.ragas_llm),
            "noise_sensitivity": NoiseSensitivity(llm=self.ragas_llm)
        }

        self.results = []

    def run_rag_system(self, test_case: RAGTestCase) -> str:
        """
        Run the RAG system to generate an answer for a test case.

        Args:
            test_case: The test case to run

        Returns:
            Generated answer
        """
        combined_context = "\n\n".join(test_case.contexts)

        messages = [
            {"role": "system", "content": (
                "You are a helpful assistant. Use the provided context to answer the user's question. "
                "If the context doesn't contain the information needed to answer the question, say 'I don't have enough information to answer this question.'"
            )},
            {"role": "user", "content": f"Context:\n{combined_context}\n\nQuestion: {test_case.question}"}
        ]

        response = self.azure_client.get_completion(messages, max_tokens=500)
        try:
            return response["choices"][0]["message"]["content"].strip()
        except KeyError as e:
            logger.error(f"Error parsing RAG response: {e}")
            return "Error generating response"

    async def evaluate_async(self, test_cases: List[RAGTestCase]) -> Dict[str, Any]:
        """
        Evaluate test cases using RAGAS metrics with async API.

        Args:
            test_cases: List of test cases

        Returns:
            Dictionary of evaluation results
        """
        # Generate answers if not already provided
        for i, test_case in enumerate(test_cases):
            if not test_case.answer:
                logger.info(f"Generating answer for test case {i + 1}/{len(test_cases)}")
                logger.info(f"Question: {test_case.question}")
                test_case.answer = self.run_rag_system(test_case)
                logger.info(f"Generated answer: {test_case.answer[:100]}...")

        # Prepare results
        processed_results = []

        # Evaluate each test case
        for i, test_case in enumerate(test_cases):
            logger.info(f"Evaluating test case {i + 1}/{len(test_cases)}")

            # Create a SingleTurnSample
            sample = SingleTurnSample(
                user_input=test_case.question,
                response=test_case.answer,
                retrieved_contexts=test_case.contexts,
                reference=test_case.ground_truth if test_case.ground_truth else None
            )

            # Calculate metrics
            metrics_results = {}
            for metric_name, scorer in self.scorers.items():
                logger.info(f"  Calculating {metric_name}...")
                try:
                    # Use the async scoring method
                    score = await scorer.single_turn_ascore(sample)
                    metrics_results[metric_name] = float(score)
                    logger.info(f"  {metric_name} score: {score:.4f}")
                except Exception as e:
                    logger.error(f"  Error calculating {metric_name}: {e}")
                    metrics_results[metric_name] = None

            # Calculate overall score (average of available metrics)
            available_scores = [v for v in metrics_results.values() if v is not None]
            if available_scores:
                metrics_results["overall_score"] = sum(available_scores) / len(available_scores)
            else:
                metrics_results["overall_score"] = None

            # Create result entry
            result = {
                "question": test_case.question,
                "answer": test_case.answer,
                "contexts": test_case.contexts,
                "ground_truth": test_case.ground_truth,
                "metrics": metrics_results,
                "metadata": test_case.metadata,
                "timestamp": datetime.now().isoformat()
            }

            processed_results.append(result)

        self.results = processed_results
        return processed_results

    def save_results(self, output_file: str = "ragas_results.json") -> None:
        """
        Save evaluation results to a file.

        Args:
            output_file: Path to the output file
        """
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"Saved results to {output_file}")

        # Also save as CSV for easier analysis
        csv_file = output_file.replace('.json', '.csv')

        # Flatten the results for CSV
        flattened_results = []
        for result in self.results:
            flat_result = {
                "question": result["question"],
                "answer": result["answer"],
                # Join contexts with a separator for CSV
                "contexts": " | ".join(result["contexts"]),
                "ground_truth": result["ground_truth"],
                "timestamp": result["timestamp"]
            }

            # Add metrics as top-level columns
            for metric_name, score in result["metrics"].items():
                flat_result[metric_name] = score

            # Add metadata as top-level columns
            for meta_key, meta_value in result.get("metadata", {}).items():
                flat_result[f"metadata_{meta_key}"] = meta_value

            flattened_results.append(flat_result)

        df = pd.DataFrame(flattened_results)
        df.to_csv(csv_file, index=False)
        logger.info(f"Saved results to {csv_file}")

    def print_summary(self) -> None:
        """Print a summary of the evaluation results"""
        if not self.results:
            logger.warning("No results to summarize")
            return

        # Collect metric names
        metric_names = set()
        for result in self.results:
            for metric_name in result["metrics"]:
                metric_names.add(metric_name)

        # Calculate averages
        summary = {
            "total_test_cases": len(self.results)
        }

        for metric_name in metric_names:
            # Get all non-None values for this metric
            values = [
                result["metrics"][metric_name]
                for result in self.results
                if metric_name in result["metrics"] and result["metrics"][metric_name] is not None
            ]

            if values:
                summary[f"avg_{metric_name}"] = sum(values) / len(values)
            else:
                summary[f"avg_{metric_name}"] = None

        logger.info("=== RAGAS Evaluation Summary ===")
        for key, value in summary.items():
            logger.info(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")


def load_test_cases(file_path: str) -> List[RAGTestCase]:
    """
    Load test cases from a JSON file.

    Args:
        file_path: Path to the JSON file containing test cases

    Returns:
        List of RAGTestCase objects
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        test_cases = []
        for item in data:
            # Map 'context' to 'contexts' if needed
            contexts = item.get("contexts", item.get("context", []))
            if isinstance(contexts, str):
                contexts = [contexts]

            # Map 'expected_answer' to 'ground_truth' if needed
            ground_truth = item.get("ground_truth", item.get("expected_answer", ""))

            test_cases.append(RAGTestCase(
                question=item["question"],
                contexts=contexts,
                ground_truth=ground_truth,
                metadata=item.get("metadata", {})
            ))

        return test_cases
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        logger.error(f"Error loading test cases: {e}")
        return []


async def run_evaluation(evaluator, test_cases, output_file):
    """Run the evaluation asynchronously"""
    await evaluator.evaluate_async(test_cases)
    evaluator.save_results(output_file)
    evaluator.print_summary()


def main():
    """Main function to run the RAGAS evaluation"""
    parser = argparse.ArgumentParser(description="RAGAS Evaluation")
    parser.add_argument(
        "--test-cases",
        default="test_cases.json",
        help="Path to JSON file containing test cases"
    )
    parser.add_argument(
        "--output",
        default="ragas_results.json",
        help="Path to output file for results"
    )
    args = parser.parse_args()

    # Check if API key is set
    if not AZURE_OPENAI_API_KEY:
        logger.error("API key not set. Please set AZURE_OPENAI_API_KEY environment variable.")
        return

    # Create evaluator
    evaluator = RAGASEvaluator(
        api_key=AZURE_OPENAI_API_KEY,
        endpoint=AZURE_OPENAI_ENDPOINT,
        deployment=AZURE_OPENAI_DEPLOYMENT,
        api_version=API_VERSION
    )

    # Load test cases
    test_cases = load_test_cases(args.test_cases)
    if not test_cases:
        logger.error(f"No test cases found in {args.test_cases}")
        return

    logger.info(f"Loaded {len(test_cases)} test cases")

    # Run evaluation asynchronously
    asyncio.run(run_evaluation(evaluator, test_cases, args.output))


if __name__ == "__main__":
    main()
