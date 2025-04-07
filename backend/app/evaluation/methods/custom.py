# File: backend/app/evaluation/methods/custom.py
import asyncio
import logging
import time
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import settings
from backend.app.db.models.orm.models import Evaluation, EvaluationStatus
from backend.app.db.schema.evaluation_schema import EvaluationResultCreate, MetricScoreCreate
from backend.app.evaluation.methods.base import BaseEvaluationMethod
from backend.app.evaluation.metrics.registry import MetricsRegistry

# Configure logging
logger = logging.getLogger(__name__)


class CustomEvaluationMethod(BaseEvaluationMethod):
    """Custom evaluation method using registered metrics."""

    method_name = "custom"

    def __init__(self, db_session: AsyncSession):
        """Initialize the evaluation method."""
        super().__init__(db_session)

    async def run_evaluation(self, evaluation: Evaluation) -> List[EvaluationResultCreate]:
        """
        Run evaluation using custom metrics.

        Args:
            evaluation: Evaluation model

        Returns:
            List[EvaluationResultCreate]: List of evaluation results
        """
        # Update the evaluation to running status if needed
        if evaluation.status == EvaluationStatus.PENDING:
            await self._update_evaluation_status(
                evaluation.id,
                EvaluationStatus.RUNNING,
                {"start_time": time.time()}
            )

        logger.info(f"Starting custom evaluation {evaluation.id}")

        # Process using batch processing approach
        try:
            results = await self.batch_process(evaluation)

            # Update evaluation to completed status
            await self._update_evaluation_status(
                evaluation.id,
                EvaluationStatus.COMPLETED,
                {"end_time": time.time()}
            )

            logger.info(f"Completed custom evaluation {evaluation.id} with {len(results)} results")
            return results

        except Exception as e:
            # Update evaluation to failed status
            await self._update_evaluation_status(
                evaluation.id,
                EvaluationStatus.FAILED,
                {"end_time": time.time()}
            )

            logger.exception(f"Failed custom evaluation {evaluation.id}: {str(e)}")
            raise

    async def _update_evaluation_status(
            self, evaluation_id: UUID, status: EvaluationStatus, additional_data: Dict[str, Any] = None
    ) -> None:
        """
        Update evaluation status in the database.

        Args:
            evaluation_id: Evaluation ID
            status: New status
            additional_data: Additional data to update
        """
        from backend.app.db.repositories.base import BaseRepository
        from backend.app.db.models.orm.models import Evaluation

        evaluation_repo = BaseRepository(Evaluation, self.db_session)

        update_data = {"status": status}
        if additional_data:
            update_data.update(additional_data)

        await evaluation_repo.update(evaluation_id, update_data)

    async def calculate_metrics(
            self, input_data: Dict[str, Any], output_data: Dict[str, Any], config: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculate custom metrics.

        Args:
            input_data: Input data for the evaluation
            output_data: Output data from the LLM
            config: Evaluation configuration

        Returns:
            Dict[str, float]: Dictionary mapping metric names to values
        """
        # Extract inputs and outputs
        query = input_data.get("query", "")
        context = input_data.get("context", "")
        ground_truth = input_data.get("ground_truth", "")
        answer = output_data.get("answer", "")

        if not query or not context or not answer:
            logger.warning("Missing required data for custom evaluation")
            return {}

        # Get enabled metrics from config or use defaults
        enabled_metrics = config.get("metrics", ["faithfulness", "response_relevancy", "context_precision"])

        # Apply weights from config if provided
        weights = config.get("weights", {})

        # Calculate metrics
        metrics_results = {}
        registry = MetricsRegistry()

        for metric_name in enabled_metrics:
            metric_info = registry.get(metric_name)

            if not metric_info:
                logger.warning(f"Metric '{metric_name}' not found in registry")
                continue

            metric_func = metric_info["func"]

            try:
                # Call the metric function with appropriate parameters
                if metric_name == "faithfulness":
                    score = await metric_func(answer, context)
                elif metric_name == "response_relevancy":
                    score = await metric_func(answer, query, context)
                elif metric_name == "context_precision":
                    score = await metric_func(context, query)
                elif metric_name == "context_recall":
                    # Requires ground truth
                    if not ground_truth:
                        logger.warning(f"Skipping {metric_name} - ground truth not provided")
                        continue
                    score = await metric_func(context, query, ground_truth)
                elif metric_name == "context_entity_recall":
                    # Requires ground truth
                    if not ground_truth:
                        logger.warning(f"Skipping {metric_name} - ground truth not provided")
                        continue
                    score = await metric_func(context, ground_truth)
                elif metric_name == "noise_sensitivity":
                    # Requires ground truth
                    if not ground_truth:
                        logger.warning(f"Skipping {metric_name} - ground truth not provided")
                        continue
                    score = await metric_func(query, answer, context, ground_truth)
                else:
                    # Default case - try to call with all parameters
                    score = await metric_func(query=query, answer=answer, context=context, ground_truth=ground_truth)

                # Apply weight if specified
                weight = weights.get(metric_name, 1.0)
                metrics_results[metric_name] = score * weight
                logger.info(f"Calculated {metric_name} score: {score:.4f} (weight: {weight})")

            except Exception as e:
                logger.error(f"Error calculating {metric_name}: {e}")

        return metrics_results

    def _format_prompt(self, prompt_template: str, item: Dict[str, Any]) -> str:
        """Format prompt template with dataset item."""
        formatted_prompt = prompt_template

        # Replace placeholders in the template
        for key, value in item.items():
            placeholder = f"{{{key}}}"
            if placeholder in formatted_prompt:
                formatted_prompt = formatted_prompt.replace(placeholder, str(value))

        return formatted_prompt

    def _get_metric_description(self, metric_name: str) -> str:
        """Get description for a metric."""
        registry = MetricsRegistry()
        metric_info = registry.get(metric_name)

        if metric_info and "description" in metric_info:
            return metric_info["description"]

        # Fallback descriptions
        descriptions = {
            "faithfulness": "Measures how well the answer sticks to the information in the context without hallucinating.",
            "response_relevancy": "Measures how relevant the answer is to the query asked.",
            "context_precision": "Measures how precisely the retrieved context matches what's needed to answer the query.",
            "context_recall": "Measures how well the retrieved context covers all the information needed to answer the query.",
            "context_entity_recall": "Measures how well the retrieved context captures the entities mentioned in the reference answer.",
            "noise_sensitivity": "Measures the model's tendency to be misled by irrelevant information in the context (lower is better)."
        }

        return descriptions.get(metric_name, f"Measures the {metric_name} of the response.")