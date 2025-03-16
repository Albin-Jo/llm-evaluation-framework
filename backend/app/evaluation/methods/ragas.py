# File: app/evaluation/methods/ragas.py
import json
import logging
from typing import Any, Dict, List, Optional
import httpx
import pandas as pd

from app.core.config.settings import settings
from app.evaluation.methods.base import BaseEvaluationMethod
from app.models.orm.models import Evaluation
from app.schema.evaluation_schema import EvaluationResultCreate, MetricScoreCreate

# Configure logging
logger = logging.getLogger(__name__)


class RagasEvaluationMethod(BaseEvaluationMethod):
    """Evaluation method using RAGAS library."""

    # async def run_evaluation(self, evaluation: Evaluation) -> List[EvaluationResultCreate]:
    #     """
    #     Run evaluation using RAGAS.
    #
    #     Args:
    #         evaluation: Evaluation model
    #
    #     Returns:
    #         List[EvaluationResultCreate]: List of evaluation results
    #     """
    #     # Get related entities
    #     microagent = await self.get_microagent(evaluation.micro_agent_id)
    #     dataset = await self.get_dataset(evaluation.dataset_id)
    #     prompt = await self.get_prompt(evaluation.prompt_id)
    #
    #     if not microagent or not dataset or not prompt:
    #         logger.error(f"Missing required entities for evaluation {evaluation.id}")
    #         return []
    #
    #     # Load dataset
    #     dataset_items = await self.load_dataset(dataset)
    #
    #     # Prepare data for batch processing with RAGAS
    #     queries = []
    #     contexts = []
    #     responses = []
    #     ground_truths = []
    #     item_indices = []
    #
    #     # First pass: get all the responses from the microagent
    #     for item_index, item in enumerate(dataset_items):
    #         try:
    #             # Process dataset item
    #             query = item.get("query", "")
    #             context = item.get("context", "")
    #             ground_truth = item.get("ground_truth", "")
    #
    #             if not query or not context:
    #                 logger.warning(f"Skipping dataset item {item_index}: missing query or context")
    #                 continue
    #
    #             # Format prompt with dataset item
    #             formatted_prompt = self._format_prompt(prompt.content, item)
    #
    #             # Call the micro-agent API
    #             response = await self._call_microagent_api(
    #                 microagent.api_endpoint,
    #                 {
    #                     "prompt": formatted_prompt,
    #                     "query": query,
    #                     "context": context
    #                 }
    #             )
    #
    #             # Extract LLM answer from response
    #             answer = response.get("answer", "")
    #             if not answer:
    #                 logger.warning(f"Skipping dataset item {item_index}: empty answer from microagent")
    #                 continue
    #
    #             # Add to collections for batch processing
    #             queries.append(query)
    #             contexts.append(context)
    #             responses.append(answer)
    #             ground_truths.append(ground_truth if ground_truth else None)
    #             item_indices.append(item_index)
    #
    #         except Exception as e:
    #             logger.exception(f"Error processing dataset item {item_index}: {e}")
    #
    #     # Check if we have any valid items to evaluate
    #     if not queries:
    #         logger.error("No valid items to evaluate")
    #         return []
    #
    #     # Second pass: run RAGAS evaluation on all items
    #     try:
    #         # Run RAGAS evaluation
    #         metrics_results = await self._run_ragas_evaluation(
    #             queries,
    #             contexts,
    #             responses,
    #             ground_truths if all(ground_truths) else None,
    #             evaluation.config or {}
    #         )
    #
    #         # Third pass: create evaluation results
    #         results = []
    #         for i, idx in enumerate(item_indices):
    #             item = dataset_items[idx]
    #             formatted_prompt = self._format_prompt(prompt.content, item)
    #
    #             # Get metrics for this item
    #             item_metrics = {k: v[i] for k, v in metrics_results.items()}
    #
    #             # Calculate overall score (average of all metrics)
    #             overall_score = sum(item_metrics.values()) / len(item_metrics) if item_metrics else 0.0
    #
    #             # Create metric scores
    #             metric_scores = [
    #                 MetricScoreCreate(
    #                     name=name,
    #                     value=value,
    #                     weight=1.0,
    #                     metadata={"description": self._get_metric_description(name)}
    #                 )
    #                 for name, value in item_metrics.items()
    #             ]
    #
    #             # Create evaluation result
    #             result = EvaluationResultCreate(
    #                 evaluation_id=evaluation.id,
    #                 overall_score=overall_score,
    #                 raw_results=item_metrics,
    #                 dataset_sample_id=str(idx),
    #                 input_data={
    #                     "query": queries[i],
    #                     "context": contexts[i],
    #                     "ground_truth": ground_truths[i] if ground_truths[i] else "",
    #                     "prompt": formatted_prompt
    #                 },
    #                 output_data={"answer": responses[i]},
    #                 processing_time_ms=None,  # We don't have this for batch processing
    #                 metric_scores=metric_scores
    #             )
    #
    #             results.append(result)
    #
    #         return results
    #
    #     except Exception as e:
    #         logger.exception(f"Error running RAGAS evaluation: {e}")
    #
    #         # Create a failed result for each item
    #         results = []
    #         for idx in item_indices:
    #             item = dataset_items[idx]
    #             results.append(
    #                 EvaluationResultCreate(
    #                     evaluation_id=evaluation.id,
    #                     overall_score=0.0,
    #                     raw_results={"error": str(e)},
    #                     dataset_sample_id=str(idx),
    #                     input_data=item,
    #                     output_data={"error": str(e)},
    #                     metric_scores=[]
    #                 )
    #             )
    #
    #         return results

    async def run_evaluation(self, evaluation: Evaluation) -> List[EvaluationResultCreate]:
        """
        Run evaluation using RAGAS.

        Args:
            evaluation: Evaluation model

        Returns:
            List[EvaluationResultCreate]: List of evaluation results
        """
        # Get related entities
        microagent = await self.get_microagent(evaluation.micro_agent_id)
        dataset = await self.get_dataset(evaluation.dataset_id)
        prompt = await self.get_prompt(evaluation.prompt_id)

        if not microagent or not dataset or not prompt:
            logger.error(f"Missing required entities for evaluation {evaluation.id}")
            return []

        # Load dataset
        dataset_items = await self.load_dataset(dataset)

        results = []

        for item_index, item in enumerate(dataset_items):
            try:
                # Process dataset item
                query = item.get("query", "")
                context = item.get("context", "")
                ground_truth = item.get("ground_truth", "")

                # Format prompt with dataset item
                formatted_prompt = self._format_prompt(prompt.content, item)

                # Call the micro-agent API
                response = await self._call_microagent_api(
                    microagent.api_endpoint,
                    {
                        "prompt": formatted_prompt,
                        "query": query,
                        "context": context
                    }
                )

                # Extract LLM answer from response
                answer = response.get("answer", "")

                # Calculate metrics
                metrics = await self.calculate_metrics(
                    input_data={
                        "query": query,
                        "context": context,
                        "ground_truth": ground_truth
                    },
                    output_data={"answer": answer},
                    config=evaluation.config or {}
                )

                # Calculate overall score (average of all metrics)
                overall_score = sum(metrics.values()) / len(metrics) if metrics else 0.0

                # Create metric scores
                metric_scores = [
                    MetricScoreCreate(
                        name=name,
                        value=value,
                        weight=1.0,
                        metadata={"description": self._get_metric_description(name)}
                    )
                    for name, value in metrics.items()
                ]

                # Create evaluation result
                result = EvaluationResultCreate(
                    evaluation_id=evaluation.id,
                    overall_score=overall_score,
                    raw_results=metrics,
                    dataset_sample_id=str(item_index),
                    input_data={
                        "query": query,
                        "context": context,
                        "ground_truth": ground_truth,
                        "prompt": formatted_prompt
                    },
                    output_data={"answer": answer},
                    processing_time_ms=response.get("processing_time_ms"),
                    metric_scores=metric_scores
                )

                results.append(result)

            except Exception as e:
                logger.exception(f"Error processing dataset item {item_index}: {e}")

                # Create failed evaluation result
                result = EvaluationResultCreate(
                    evaluation_id=evaluation.id,
                    overall_score=0.0,
                    raw_results={"error": str(e)},
                    dataset_sample_id=str(item_index),
                    input_data=item,
                    output_data={"error": str(e)},
                    metric_scores=[]
                )

                results.append(result)

        return results

    async def _run_ragas_evaluation(
            self,
            queries: List[str],
            contexts: List[str],
            responses: List[str],
            ground_truths: Optional[List[str]] = None,
            config: Dict[str, Any] = None
    ) -> Dict[str, List[float]]:
        """
        Run evaluation using RAGAS library.

        Args:
            queries: List of queries
            contexts: List of contexts
            responses: List of LLM answers
            ground_truths: Optional list of ground truth answers
            config: Evaluation configuration

        Returns:
            Dict[str, List[float]]: Dictionary mapping metric names to lists of values
        """
        try:
            # Import RAGAS components
            from ragas.metrics import (
                faithfulness, answer_relevancy, context_relevancy,
                context_precision, context_recall
            )
            from ragas.metrics.critique import harmfulness
            from datasets import Dataset

            # Create dataset for RAGAS
            data = {
                "question": queries,
                "contexts": [[ctx] for ctx in contexts],  # RAGAS expects a list of contexts for each item
                "answer": responses,
            }

            if ground_truths and all(ground_truths):
                data["ground_truth"] = ground_truths

            # Convert to HuggingFace dataset
            ds = Dataset.from_dict(data)

            # Get enabled metrics from config
            enabled_metrics = config.get("metrics", ["faithfulness", "answer_relevancy", "context_relevancy"])

            # Initialize metrics
            metrics = []
            if "faithfulness" in enabled_metrics:
                metrics.append(faithfulness)
            if "answer_relevancy" in enabled_metrics:
                metrics.append(answer_relevancy)
            if "context_relevancy" in enabled_metrics:
                metrics.append(context_relevancy)
            if "context_precision" in enabled_metrics and ground_truths and all(ground_truths):
                metrics.append(context_precision)
            if "context_recall" in enabled_metrics and ground_truths and all(ground_truths):
                metrics.append(context_recall)
            if "harmfulness" in enabled_metrics:
                metrics.append(harmfulness)

            if not metrics:
                logger.warning("No valid RAGAS metrics selected")
                return {}

            # Run evaluation
            from ragas import evaluate
            results = evaluate(ds, metrics)

            # Convert results to the expected format
            metrics_dict = {}
            for metric in metrics:
                metric_name = metric.__name__.lower()
                metrics_dict[metric_name] = results[metric_name].tolist()

            return metrics_dict

        except ImportError:
            logger.exception("RAGAS library not installed. Using fallback implementation.")
            return self._fallback_ragas_evaluation(queries, contexts, responses, ground_truths)

    async def _fallback_ragas_evaluation(
            self,
            queries: List[str],
            contexts: List[str],
            responses: List[str],
            ground_truths: Optional[List[str]] = None
    ) -> Dict[str, List[float]]:
        """
        Fallback implementation for RAGAS evaluation when the library is not available.

        Args:
            queries: List of queries
            contexts: List of contexts
            responses: List of LLM answers
            ground_truths: Optional list of ground truth answers

        Returns:
            Dict[str, List[float]]: Dictionary mapping metric names to lists of values
        """
        metrics = {}

        # Calculate faithfulness for each item
        faithfulness_scores = []
        for response, context in zip(responses, contexts):
            faithfulness_scores.append(self._calculate_faithfulness(response, context))
        metrics["faithfulness"] = faithfulness_scores

        # Calculate answer_relevancy for each item
        answer_relevancy_scores = []
        for response, query in zip(responses, queries):
            answer_relevancy_scores.append(self._calculate_answer_relevancy(response, query))
        metrics["answer_relevancy"] = answer_relevancy_scores

        # Calculate context_relevancy for each item
        context_relevancy_scores = []
        for context, query in zip(contexts, queries):
            context_relevancy_scores.append(self._calculate_context_relevancy(context, query))
        metrics["context_relevancy"] = context_relevancy_scores

        # Calculate correctness if ground truths are available
        if ground_truths and all(ground_truths):
            correctness_scores = []
            for response, ground_truth in zip(responses, ground_truths):
                correctness_scores.append(self._calculate_correctness(response, ground_truth))
            metrics["correctness"] = correctness_scores

        return metrics

    # async def calculate_metrics(
    #         self, input_data: Dict[str, Any], output_data: Dict[str, Any], config: Dict[str, Any]
    # ) -> Dict[str, float]:
    #     """
    #     Calculate RAGAS metrics for a single evaluation item.
    #
    #     Args:
    #         input_data: Input data for the evaluation
    #         output_data: Output data from the LLM
    #         config: Evaluation configuration
    #
    #     Returns:
    #         Dict[str, float]: Dictionary mapping metric names to values
    #     """
    #     # Extract inputs and outputs
    #     query = input_data.get("query", "")
    #     context = input_data.get("context", "")
    #     ground_truth = input_data.get("ground_truth", "")
    #     answer = output_data.get("answer", "")
    #
    #     # Run single-item evaluation
    #     queries = [query]
    #     contexts = [context]
    #     responses = [answer]
    #     ground_truths = [ground_truth] if ground_truth else None
    #
    #     try:
    #         # Use the RAGAS evaluation
    #         metrics_results = await self._run_ragas_evaluation(
    #             queries, contexts, responses, ground_truths, config
    #         )
    #
    #         # Convert list results to single values
    #         return {k: v[0] for k, v in metrics_results.items()}
    #
    #     except Exception as e:
    #         logger.exception(f"Error calculating RAGAS metrics: {e}")
    #         return {}

    async def calculate_metrics(
            self, input_data: Dict[str, Any], output_data: Dict[str, Any], config: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculate RAGAS metrics for a single evaluation item.

        Args:
            input_data: Input data for the evaluation
            output_data: Output data from the LLM
            config: Evaluation configuration

        Returns:
            Dict[str, float]: Dictionary mapping metric names to values
        """
        try:
            # Extract inputs and outputs
            query = input_data.get("query", "")
            context = input_data.get("context", "")
            ground_truth = input_data.get("ground_truth", "")
            answer = output_data.get("answer", "")

            if not query or not context or not answer:
                logger.warning("Missing required data for RAGAS evaluation")
                return {}

            # Get enabled metrics from config or use defaults
            enabled_metrics = config.get("metrics", ["faithfulness", "answer_relevancy", "context_relevancy"])

            # Initialize metrics dictionary
            metrics = {}

            # Try to use RAGAS library if available
            try:
                from ragas.metrics import (
                    faithfulness, answer_relevancy, context_relevancy,
                    context_precision, context_recall
                )
                from datasets import Dataset

                # Create dataset for RAGAS
                data = {
                    "question": [query],
                    "contexts": [[context]],  # RAGAS expects a list of contexts for each item
                    "answer": [answer],
                }

                if ground_truth:
                    data["ground_truth"] = [ground_truth]

                # Convert to HuggingFace dataset
                ds = Dataset.from_dict(data)

                # Initialize metrics
                ragas_metrics = []
                if "faithfulness" in enabled_metrics:
                    ragas_metrics.append(faithfulness)
                if "answer_relevancy" in enabled_metrics:
                    ragas_metrics.append(answer_relevancy)
                if "context_relevancy" in enabled_metrics:
                    ragas_metrics.append(context_relevancy)
                if "context_precision" in enabled_metrics and ground_truth:
                    ragas_metrics.append(context_precision)
                if "context_recall" in enabled_metrics and ground_truth:
                    ragas_metrics.append(context_recall)

                if ragas_metrics:
                    # Run evaluation
                    from ragas import evaluate
                    results = evaluate(ds, ragas_metrics)

                    # Extract results
                    for metric in ragas_metrics:
                        metric_name = metric.__name__.lower()
                        metrics[metric_name] = results[metric_name][0]  # Get first (only) result

                    return metrics

            except (ImportError, Exception) as e:
                logger.warning(f"Error using RAGAS library, falling back to simple implementations: {e}")
                # Fall back to simple implementations

            # Calculate fallback metrics
            if "faithfulness" in enabled_metrics:
                metrics["faithfulness"] = self._calculate_faithfulness(answer, context)

            if "answer_relevancy" in enabled_metrics:
                metrics["answer_relevancy"] = self._calculate_answer_relevancy(answer, query)

            if "context_relevancy" in enabled_metrics:
                metrics["context_relevancy"] = self._calculate_context_relevancy(context, query)

            # Correctness metric needs ground truth
            if ground_truth and "correctness" in enabled_metrics:
                metrics["correctness"] = self._calculate_correctness(answer, ground_truth)

            return metrics

        except Exception as e:
            logger.exception(f"Error calculating RAGAS metrics: {e}")
            return {}

    def _calculate_faithfulness(self, answer: str, context: str) -> float:
        """
        Calculate faithfulness score.

        Args:
            answer: LLM answer
            context: Input context

        Returns:
            float: Faithfulness score (0-1)
        """
        # Simple implementation - check if answer words are in context
        if not answer or not context:
            return 0.0

        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())

        if not answer_words:
            return 0.0

        overlap = answer_words.intersection(context_words)
        return len(overlap) / len(answer_words)

    def _calculate_answer_relevancy(self, answer: str, query: str) -> float:
        """
        Calculate answer relevancy score.

        Args:
            answer: LLM answer
            query: User query

        Returns:
            float: Answer relevancy score (0-1)
        """
        # Simple implementation - check keyword overlap
        if not answer or not query:
            return 0.0

        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())

        if not query_words:
            return 0.0

        overlap = query_words.intersection(answer_words)
        return len(overlap) / len(query_words)

    def _calculate_context_relevancy(self, context: str, query: str) -> float:
        """
        Calculate context relevancy score.

        Args:
            context: Input context
            query: User query

        Returns:
            float: Context relevancy score (0-1)
        """
        # Simple implementation - check keyword overlap
        if not context or not query:
            return 0.0

        query_words = set(query.lower().split())
        context_words = set(context.lower().split())

        if not query_words:
            return 0.0

        overlap = query_words.intersection(context_words)
        return len(overlap) / len(query_words)

    def _calculate_correctness(self, answer: str, ground_truth: str) -> float:
        """
        Calculate correctness score.

        Args:
            answer: LLM answer
            ground_truth: Expected answer

        Returns:
            float: Correctness score (0-1)
        """
        # Simple implementation - check token overlap
        if not answer or not ground_truth:
            return 0.0

        answer_tokens = answer.lower().split()
        ground_truth_tokens = ground_truth.lower().split()

        if not ground_truth_tokens:
            return 0.0

        # Calculate token overlap (very simplistic)
        common_tokens = sum(1 for token in answer_tokens if token in ground_truth_tokens)
        return common_tokens / len(ground_truth_tokens)

    def _format_prompt(self, prompt_template: str, item: Dict[str, Any]) -> str:
        """
        Format prompt template with dataset item.

        Args:
            prompt_template: Prompt template string
            item: Dataset item

        Returns:
            str: Formatted prompt
        """
        formatted_prompt = prompt_template

        # Replace placeholders in the template
        for key, value in item.items():
            placeholder = f"{{{key}}}"
            if placeholder in formatted_prompt:
                formatted_prompt = formatted_prompt.replace(placeholder, str(value))

        return formatted_prompt

    async def _call_microagent_api(
            self, api_endpoint: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Call the micro-agent API.

        Args:
            api_endpoint: API endpoint URL
            payload: Request payload

        Returns:
            Dict[str, Any]: API response

        Raises:
            Exception: If API call fails
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    api_endpoint,
                    json=payload,
                    headers={"Authorization": f"Bearer {settings.OPENAI_API_KEY}"},
                    timeout=60.0
                )

                response.raise_for_status()
                return response.json()

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error calling micro-agent API: {e}")
            raise Exception(f"Micro-agent API returned error: {e.response.status_code}")

        except httpx.RequestError as e:
            logger.error(f"Request error calling micro-agent API: {e}")
            raise Exception(f"Error connecting to micro-agent API: {str(e)}")

        except Exception as e:
            logger.error(f"Unexpected error calling micro-agent API: {e}")
            raise Exception(f"Error calling micro-agent API: {str(e)}")

    def _get_metric_description(self, metric_name: str) -> str:
        """
        Get description for a metric.

        Args:
            metric_name: Metric name

        Returns:
            str: Metric description
        """
        descriptions = {
            "faithfulness": "Measures how well the answer sticks to the information in the context without hallucinating.",
            "answer_relevancy": "Measures how relevant the answer is to the query asked.",
            "context_relevancy": "Measures how relevant the context is to the query asked.",
            "context_precision": "Measures how precise the retrieved context is compared to the ground truth context.",
            "context_recall": "Measures how well the retrieved context covers the ground truth context.",
            "harmfulness": "Measures whether the answer contains harmful or inappropriate content.",
            "correctness": "Measures how well the answer matches the ground truth."
        }

        return descriptions.get(metric_name, "")