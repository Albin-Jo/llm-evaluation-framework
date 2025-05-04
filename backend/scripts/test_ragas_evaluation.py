"""
Script to test RAGAS (Retrieval Augmented Generation Assessment) evaluation
using Azure OpenAI API.
"""

import os
import json
import logging
import requests
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import argparse
from dataclasses import dataclass
import pandas as pd
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("Pixi tasks.txt"),
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
    context: List[str]
    expected_answer: str
    metadata: Optional[Dict[str, Any]] = None


class AzureOpenAIClient:
    """Client for interacting with Azure OpenAI API"""

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

    def __init__(self, client: AzureOpenAIClient):
        """
        Initialize the RAGAS evaluator.

        Args:
            client: Azure OpenAI client for API calls
        """
        self.client = client
        self.results = []

    def evaluate_faithfulness(self, answer: str, context: List[str]) -> float:
        """
        Evaluate if the generated answer is faithful to the provided context.

        Args:
            answer: Generated answer
            context: List of context passages used to generate the answer

        Returns:
            Faithfulness score between 0.0 and 1.0
        """
        combined_context = "\n\n".join(context)

        messages = [
            {"role": "system", "content": (
                "You are an evaluation assistant. Your task is to determine if the provided answer "
                "is faithful to the given context. The answer should only contain information that "
                "can be derived from the context. Score from 0.0 (completely unfaithful) to 1.0 "
                "(completely faithful). Respond with just the score."
            )},
            {"role": "user",
             "content": f"Context:\n{combined_context}\n\nAnswer:\n{answer}\n\nFaithfulness score (0.0 to 1.0):"}
        ]

        response = self.client.get_completion(messages)
        try:
            content = response["choices"][0]["message"]["content"].strip()
            # Extract number from content if it contains other text
            import re
            score_match = re.search(r"(\d+\.\d+|\d+)", content)
            if score_match:
                score = float(score_match.group(1))
                return min(max(score, 0.0), 1.0)  # Ensure score is between 0 and 1
            else:
                logger.warning(f"Could not extract score from: {content}")
                return 0.0
        except (KeyError, ValueError) as e:
            logger.error(f"Error parsing faithfulness score: {e}")
            return 0.0

    def evaluate_answer_relevancy(self, question: str, answer: str) -> float:
        """
        Evaluate how relevant the answer is to the question.

        Args:
            question: User question
            answer: Generated answer

        Returns:
            Relevancy score between 0.0 and 1.0
        """
        messages = [
            {"role": "system", "content": (
                "You are an evaluation assistant. Your task is to determine how relevant the provided "
                "answer is to the question. Score from 0.0 (completely irrelevant) to 1.0 (perfectly "
                "relevant). Respond with just the score."
            )},
            {"role": "user", "content": f"Question:\n{question}\n\nAnswer:\n{answer}\n\nRelevancy score (0.0 to 1.0):"}
        ]

        response = self.client.get_completion(messages)
        try:
            content = response["choices"][0]["message"]["content"].strip()
            # Extract number from content
            import re
            score_match = re.search(r"(\d+\.\d+|\d+)", content)
            if score_match:
                score = float(score_match.group(1))
                return min(max(score, 0.0), 1.0)  # Ensure score is between 0 and 1
            else:
                logger.warning(f"Could not extract score from: {content}")
                return 0.0
        except (KeyError, ValueError) as e:
            logger.error(f"Error parsing relevancy score: {e}")
            return 0.0

    def evaluate_context_precision(self, question: str, context: List[str]) -> float:
        """
        Evaluate how precise and relevant the retrieved context is for answering the question.

        Args:
            question: User question
            context: List of context passages retrieved

        Returns:
            Context precision score between 0.0 and 1.0
        """
        combined_context = "\n\n".join(context)

        messages = [
            {"role": "system", "content": (
                "You are an evaluation assistant. Your task is to determine how precise and relevant "
                "the retrieved context is for answering the given question. Score from 0.0 (completely "
                "irrelevant context) to 1.0 (perfectly relevant context). Respond with just the score."
            )},
            {"role": "user",
             "content": f"Question:\n{question}\n\nRetrieved context:\n{combined_context}\n\nContext precision score (0.0 to 1.0):"}
        ]

        response = self.client.get_completion(messages)
        try:
            content = response["choices"][0]["message"]["content"].strip()
            # Extract number from content
            import re
            score_match = re.search(r"(\d+\.\d+|\d+)", content)
            if score_match:
                score = float(score_match.group(1))
                return min(max(score, 0.0), 1.0)  # Ensure score is between 0 and 1
            else:
                logger.warning(f"Could not extract score from: {content}")
                return 0.0
        except (KeyError, ValueError) as e:
            logger.error(f"Error parsing context precision score: {e}")
            return 0.0

    def evaluate_test_case(self, test_case: RAGTestCase, actual_answer: str) -> Dict[str, Any]:
        """
        Evaluate a single RAG test case.

        Args:
            test_case: The test case to evaluate
            actual_answer: The actual answer generated by the system

        Returns:
            Dictionary containing evaluation metrics
        """
        faithfulness = self.evaluate_faithfulness(actual_answer, test_case.context)
        relevancy = self.evaluate_answer_relevancy(test_case.question, actual_answer)
        context_precision = self.evaluate_context_precision(test_case.question, test_case.context)

        # Calculate an overall score (simple average)
        overall_score = (faithfulness + relevancy + context_precision) / 3

        result = {
            "question": test_case.question,
            "context": test_case.context,
            "expected_answer": test_case.expected_answer,
            "actual_answer": actual_answer,
            "faithfulness": faithfulness,
            "answer_relevancy": relevancy,
            "context_precision": context_precision,
            "overall_score": overall_score,
            "metadata": test_case.metadata,
            "timestamp": datetime.now().isoformat()
        }

        self.results.append(result)
        return result

    def run_rag_system(self, test_case: RAGTestCase) -> str:
        """
        Run the RAG system to generate an answer for a test case.

        Args:
            test_case: The test case to run

        Returns:
            Generated answer
        """
        combined_context = "\n\n".join(test_case.context)

        messages = [
            {"role": "system", "content": (
                "You are a helpful assistant. Use the provided context to answer the user's question. "
                "If the context doesn't contain the information needed to answer the question, say 'I don't have enough information to answer this question.'"
            )},
            {"role": "user", "content": f"Context:\n{combined_context}\n\nQuestion: {test_case.question}"}
        ]

        response = self.client.get_completion(messages, max_tokens=500)
        try:
            return response["choices"][0]["message"]["content"].strip()
        except KeyError as e:
            logger.error(f"Error parsing RAG response: {e}")
            return "Error generating response"

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
        df = pd.DataFrame(self.results)

        # Flatten context for CSV
        df['context'] = df['context'].apply(lambda x: ' | '.join(x))

        # Drop metadata column for CSV as it might be complex
        if 'metadata' in df.columns:
            df = df.drop('metadata', axis=1)

        df.to_csv(csv_file, index=False)
        logger.info(f"Saved results to {csv_file}")

    def print_summary(self) -> None:
        """Print a summary of the evaluation results"""
        if not self.results:
            logger.warning("No results to summarize")
            return

        df = pd.DataFrame(self.results)

        summary = {
            "total_test_cases": len(df),
            "avg_faithfulness": df["faithfulness"].mean(),
            "avg_answer_relevancy": df["answer_relevancy"].mean(),
            "avg_context_precision": df["context_precision"].mean(),
            "avg_overall_score": df["overall_score"].mean()
        }

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
            test_cases.append(RAGTestCase(
                question=item["question"],
                context=item["context"],
                expected_answer=item["expected_answer"],
                metadata=item.get("metadata")
            ))

        return test_cases
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        logger.error(f"Error loading test cases: {e}")
        return []


def main():
    """Main function to run the RAGAS evaluation"""
    parser = argparse.ArgumentParser(description="RAGAS Evaluation Test Script")
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

    # Create client and evaluator
    client = AzureOpenAIClient()
    evaluator = RAGASEvaluator(client)

    # Load test cases
    test_cases = load_test_cases(args.test_cases)
    if not test_cases:
        logger.error(f"No test cases found in {args.test_cases}")
        return

    logger.info(f"Loaded {len(test_cases)} test cases")

    # Run evaluation for each test case
    for i, test_case in enumerate(test_cases):
        logger.info(f"Evaluating test case {i + 1}/{len(test_cases)}")

        # Run RAG system to get actual answer
        actual_answer = evaluator.run_rag_system(test_case)

        # Evaluate test case
        result = evaluator.evaluate_test_case(test_case, actual_answer)

        logger.info(f"Test case {i + 1} - Overall score: {result['overall_score']:.4f}")

    # Save results and print summary
    evaluator.save_results(args.output)
    evaluator.print_summary()


if __name__ == "__main__":
    main()