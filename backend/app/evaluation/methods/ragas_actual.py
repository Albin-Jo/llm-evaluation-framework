# File: app/evaluation/methods/ragas_actual.py
import asyncio
import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from app.evaluation.methods.base import BaseEvaluationMethod
from app.models.orm.models import Evaluation
from app.schema.evaluation_schema import EvaluationResultCreate, MetricScoreCreate

# Configure logging
logger = logging.getLogger(__name__)


class ActualRagasEvaluationMethod(BaseEvaluationMethod):
    """Evaluation method using the actual RAGAS library."""

    method_name = "ragas_actual"

    async def run_evaluation(self, evaluation: Evaluation) -> List[EvaluationResultCreate]:
        """
        Run evaluation using actual RAGAS.

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

        logger.info(f"Starting RAGAS evaluation {evaluation.id} with {len(dataset_items)} items")

        # Process in batches for better performance
        results = []
        batch_size = 5  # Small batch size for RAGAS due to potential memory usage

        for batch_start in range(0, len(dataset_items), batch_size):
            batch_end = min(batch_start + batch_size, len(dataset_items))
            batch = dataset_items[batch_start:batch_end]

            # Process items one by one to avoid overwhelming the RAGAS library
            for item_index, item in enumerate(batch):
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
                    processing_time_ms = response.get("processing_time_ms", 0)

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
                        dataset_sample_id=str(batch_start + item_index),
                        input_data={
                            "query": query,
                            "context": context,
                            "ground_truth": ground_truth,
                            "prompt": formatted_prompt
                        },
                        output_data={"answer": answer},
                        processing_time_ms=processing_time_ms,
                        metric_scores=metric_scores
                    )

                    results.append(result)
                    logger.info(f"Processed item {batch_start + item_index} for evaluation {evaluation.id}")

                except Exception as e:
                    logger.exception(f"Error processing dataset item {batch_start + item_index}: {e}")

                    # Create failed evaluation result
                    result = EvaluationResultCreate(
                        evaluation_id=evaluation.id,
                        overall_score=0.0,
                        raw_results={"error": str(e)},
                        dataset_sample_id=str(batch_start + item_index),
                        input_data=item,
                        output_data={"error": str(e)},
                        metric_scores=[]
                    )

                    results.append(result)

        logger.info(f"Completed RAGAS evaluation {evaluation.id} with {len(results)} results")
        return results

    async def calculate_metrics(
            self, input_data: Dict[str, Any], output_data: Dict[str, Any], config: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculate RAGAS metrics using the actual RAGAS library.

        Args:
            input_data: Input data for the evaluation
            output_data: Output data from the LLM
            config: Evaluation configuration

        Returns:
            Dict[str, float]: Dictionary mapping metric names to values
        """
        try:
            # Import RAGAS here to avoid loading it unless needed
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_relevancy,
                context_precision,
                context_recall
            )

            # Extract inputs and outputs
            query = input_data.get("query", "")
            context = input_data.get("context", "")
            ground_truth = input_data.get("ground_truth", "")
            answer = output_data.get("answer", "")

            # Prepare data for RAGAS
            # RAGAS expects a specific format
            contexts = [context] if isinstance(context, str) else context

            # Create dataset entry
            dataset_entry = {
                "question": query,
                "contexts": contexts,
                "answer": answer,
                "ground_truth": ground_truth if ground_truth else None
            }

            # Create single-element dataset
            dataset = [dataset_entry]

            # Run metrics in thread pool to avoid blocking
            metrics = {}

            # Calculate faithfulness score
            if config.get("metrics", {}).get("faithfulness", True):
                try:
                    faithfulness_score = await asyncio.to_thread(
                        faithfulness.from_instances, dataset
                    )
                    metrics["faithfulness"] = faithfulness_score[0] if faithfulness_score else 0.0
                except Exception as e:
                    logger.error(f"Error calculating faithfulness: {e}")
                    metrics["faithfulness"] = 0.0

            # Calculate answer relevancy
            if config.get("metrics", {}).get("answer_relevancy", True):
                try:
                    answer_rel_score = await asyncio.to_thread(
                        answer_relevancy.from_instances, dataset
                    )
                    metrics["answer_relevancy"] = answer_rel_score[0] if answer_rel_score else 0.0
                except Exception as e:
                    logger.error(f"Error calculating answer_relevancy: {e}")
                    metrics["answer_relevancy"] = 0.0

            # Calculate context relevancy
            if config.get("metrics", {}).get("context_relevancy", True):
                try:
                    context_rel_score = await asyncio.to_thread(
                        context_relevancy.from_instances, dataset
                    )
                    metrics["context_relevancy"] = context_rel_score[0] if context_rel_score else 0.0
                except Exception as e:
                    logger.error(f"Error calculating context_relevancy: {e}")
                    metrics["context_relevancy"] = 0.0

            # Calculate context precision
            if config.get("metrics", {}).get("context_precision", False):
                try:
                    context_prec_score = await asyncio.to_thread(
                        context_precision.from_instances, dataset
                    )
                    metrics["context_precision"] = context_prec_score[0] if context_prec_score else 0.0
                except Exception as e:
                    logger.error(f"Error calculating context_precision: {e}")
                    metrics["context_precision"] = 0.0

            # Calculate context recall (if ground truth is available)
            if ground_truth and config.get("metrics", {}).get("context_recall", False):
                try:
                    context_recall_score = await asyncio.to_thread(
                        context_recall.from_instances, dataset
                    )
                    metrics["context_recall"] = context_recall_score[0] if context_recall_score else 0.0
                except Exception as e:
                    logger.error(f"Error calculating context_recall: {e}")
                    metrics["context_recall"] = 0.0

            return metrics

        except Exception as e:
            logger.exception(f"Error calculating RAGAS metrics: {e}")
            # Fall back to the simulated implementation
            return await self._calculate_simulated_metrics(input_data, output_data, config)

    async def _calculate_simulated_metrics(
            self, input_data: Dict[str, Any], output_data: Dict[str, Any], config: Dict[str, Any]
    ) -> Dict[str, float]:
        """Fallback to simulated metrics if RAGAS fails."""
        logger.warning("Falling back to simulated RAGAS metrics")

        # Extract inputs and outputs
        query = input_data.get("query", "")
        context = input_data.get("context", "")
        ground_truth = input_data.get("ground_truth", "")
        answer = output_data.get("answer", "")

        # Calculate simulated metrics
        metrics = {}

        # Faithfulness: Measure how well the answer sticks to the context
        if not answer or not context:
            metrics["faithfulness"] = 0.0
        else:
            answer_words = set(answer.lower().split())
            context_words = set(context.lower().split())
            if not answer_words:
                metrics["faithfulness"] = 0.0
            else:
                overlap = answer_words.intersection(context_words)
                metrics["faithfulness"] = len(overlap) / len(answer_words)

        # Answer Relevancy: Measure how relevant the answer is to the query
        if not answer or not query:
            metrics["answer_relevancy"] = 0.0
        else:
            query_words = set(query.lower().split())
            answer_words = set(answer.lower().split())
            if not query_words:
                metrics["answer_relevancy"] = 0.0
            else:
                overlap = query_words.intersection(answer_words)
                metrics["answer_relevancy"] = len(overlap) / len(query_words)

        # Context Relevancy: Measure how relevant the context is to the query
        if not context or not query:
            metrics["context_relevancy"] = 0.0
        else:
            query_words = set(query.lower().split())
            context_words = set(context.lower().split())
            if not query_words:
                metrics["context_relevancy"] = 0.0
            else:
                overlap = query_words.intersection(context_words)
                metrics["context_relevancy"] = len(overlap) / len(query_words)

        # Correctness: Compare answer with ground truth (if available)
        if ground_truth and config.get("metrics", {}).get("correctness", False):
            if not answer or not ground_truth:
                metrics["correctness"] = 0.0
            else:
                answer_tokens = answer.lower().split()
                ground_truth_tokens = ground_truth.lower().split()
                if not ground_truth_tokens:
                    metrics["correctness"] = 0.0
                else:
                    # Calculate token overlap (very simplistic)
                    common_tokens = sum(1 for token in answer_tokens if token in ground_truth_tokens)
                    metrics["correctness"] = common_tokens / len(ground_truth_tokens)

        return metrics

    def _format_prompt(self, prompt_template: str, item: Dict[str, Any]) -> str:
        """
        Format prompt template with dataset item.
        """
        formatted_prompt = prompt_template

        # Replace placeholders in the template
        for key, value in item.items():
            placeholder = f"{{{key}}}"
            if placeholder in formatted_prompt:
                formatted_prompt = formatted_prompt.replace(placeholder, str(value))

        return formatted_prompt

    async def _call_microagent_api(self, api_endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call the micro-agent API with retry logic.
        """
        import httpx
        from app.core.config.settings import settings

        max_retries = 3
        retry_delay = 1.0  # seconds

        for attempt in range(max_retries):
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
                if e.response.status_code == 429:  # Too many requests
                    if attempt < max_retries - 1:
                        # Get retry-after header if available
                        retry_after = e.response.headers.get("retry-after")
                        wait_time = float(retry_after) if retry_after else retry_delay * (2 ** attempt)
                        logger.warning(f"Rate limited. Retrying after {wait_time} seconds...")
                        await asyncio.sleep(wait_time)
                        continue
                logger.error(f"HTTP error calling micro-agent API: {e}")
                raise Exception(f"Micro-agent API returned error: {e.response.status_code}")
            except Exception as e:
                logger.error(f"Error calling micro-agent API: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2 ** attempt))
                    continue
                raise

    def _get_metric_description(self, metric_name: str) -> str:
        """
        Get description for a metric.
        """
        descriptions = {
            "faithfulness": "Measures how well the answer sticks to the information in the context.",
            "answer_relevancy": "Measures how relevant the answer is to the query.",
            "context_relevancy": "Measures how relevant the context is to the query.",
            "correctness": "Measures how well the answer matches the ground truth.",
            "context_precision": "Measures the precision of the context in answering the query.",
            "context_recall": "Measures the recall of relevant information from the context."
        }

        return descriptions.get(metric_name, f"Measures the {metric_name} of the response.")