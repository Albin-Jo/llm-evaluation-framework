import datetime
import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models.orm import Evaluation, EvaluationStatus, Dataset
from backend.app.db.schema.evaluation_schema import EvaluationResultCreate
from backend.app.evaluation.methods.base import BaseEvaluationMethod
from backend.app.evaluation.metrics.ragas_metrics import (
    DATASET_TYPE_METRICS, METRIC_REQUIREMENTS, RAGAS_AVAILABLE
)

# Configure logging
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.ERROR)


class RagasEvaluationMethod(BaseEvaluationMethod):
    """Evaluation method using RAGAS library for RAG evaluation."""

    method_name = "ragas"

    def __init__(self, db_session: AsyncSession):
        """Initialize the evaluation method."""
        super().__init__(db_session)
        self.ragas_available = RAGAS_AVAILABLE
        logger.info(f"Initializing RAGAS evaluation method. RAGAS available: {self.ragas_available}")

    async def run_evaluation(self,
                             evaluation: Evaluation,
                             jwt_token: Optional[str] = None) -> List[EvaluationResultCreate]:
        """
        Run evaluation using RAGAS.

        Args:
            evaluation: Evaluation model
            jwt_token: Optional JWT token to use for authentication with MCP agents

        Returns:
            List[EvaluationResultCreate]: List of evaluation results
        """
        logger.info(f"Starting RAGAS evaluation {evaluation.id}")

        # Process using batch processing approach
        try:
            # Get dataset
            dataset = await self.get_dataset(evaluation.dataset_id)
            if not dataset:
                raise ValueError(f"Dataset with ID {evaluation.dataset_id} not found")

            # Load dataset to get total count
            dataset_items = await self.load_dataset(dataset)
            total_items = len(dataset_items)

            # Update dataset row count if needed
            if not dataset.row_count or dataset.row_count != total_items:
                from backend.app.db.repositories.base import BaseRepository
                dataset_repo = BaseRepository(Dataset, self.db_session)
                await dataset_repo.update(dataset.id, {"row_count": total_items})
                logger.info(f"Updated dataset {dataset.id} row_count to {total_items}")

            # Validate metrics based on dataset type
            if not evaluation.metrics:
                # Default metrics based on dataset type
                evaluation.metrics = DATASET_TYPE_METRICS.get(dataset.type, ["faithfulness", "context_precision"])
                logger.info(f"Using default metrics for {dataset.type} dataset: {evaluation.metrics}")
            else:
                # Check that selected metrics are appropriate for this dataset type
                valid_metrics = DATASET_TYPE_METRICS.get(dataset.type, [])
                invalid_metrics = [m for m in evaluation.metrics if m not in valid_metrics]
                if invalid_metrics:
                    logger.warning(
                        f"Metrics {invalid_metrics} are not typically appropriate for {dataset.type} datasets. "
                        f"Valid metrics are: {valid_metrics}"
                    )

            # Pass the JWT token to batch_process for MCP agents
            results = await self.batch_process(evaluation, jwt_token=jwt_token)

            logger.info(f"Completed RAGAS evaluation {evaluation.id} with {len(results)} results")
            return results

        except Exception as e:
            # Update evaluation to failed status
            await self._update_evaluation_status(
                evaluation.id,
                EvaluationStatus.FAILED,
                {"end_time": datetime.datetime.now()}
            )

            logger.exception(f"Failed RAGAS evaluation {evaluation.id}: {str(e)}")
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
        from backend.app.db.models.orm import Evaluation

        evaluation_repo = BaseRepository(Evaluation, self.db_session)

        update_data = {"status": status}
        if additional_data:
            update_data.update(additional_data)

        await evaluation_repo.update(evaluation_id, update_data)
        logger.info(f"Updated evaluation {evaluation_id} status to {status}")

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
        # Extract inputs and outputs
        query = input_data.get("query", "")
        context = input_data.get("context", "")
        ground_truth = input_data.get("ground_truth", "")
        answer = output_data.get("answer", "")

        logger.info(f"Calculating metrics:")
        logger.info(f"Query: {query[:100]}..." if len(query) > 100 else f"Query: {query}")
        logger.info(f"Context length: {len(context)} chars")
        logger.info(f"Ground truth: {ground_truth[:100]}..." if ground_truth and len(
            ground_truth) > 100 else f"Ground truth: {ground_truth}")
        logger.info(f"Answer: {answer[:100]}..." if len(answer) > 100 else f"Answer: {answer}")

        # Get enabled metrics from config or use defaults
        # Update this to use more metrics from the DATASET_TYPE_METRICS mapping
        # Instead of using a hard-coded list of metrics
        dataset_type = config.get("dataset_type", "custom")
        available_metrics = DATASET_TYPE_METRICS.get(dataset_type, [
            "faithfulness",
            "response_relevancy",
            "context_precision",
            "answer_correctness",
            "answer_relevancy",
            "factual_correctness"
        ])

        # Use configured metrics or fall back to all available ones
        enabled_metrics = config.get("metrics", available_metrics)

        # Initialize metrics dictionary
        metrics = {}

        # Calculate each enabled metric
        for metric_name in enabled_metrics:
            if metric_name not in METRIC_REQUIREMENTS:
                logger.warning(f"Unknown metric: {metric_name}")
                continue

            metric_info = METRIC_REQUIREMENTS[metric_name]
            required_fields = metric_info["required_fields"]
            calculation_func = metric_info["calculation_func"]

            # Check if we have all required fields
            missing_fields = []
            field_map = {
                "query": query,
                "context": context,
                "ground_truth": ground_truth,
                "answer": answer
            }

            for field in required_fields:
                if not field_map.get(field):
                    missing_fields.append(field)

            if missing_fields:
                logger.warning(f"Cannot calculate {metric_name} - missing required fields: {missing_fields}")
                continue

            try:
                # Call the appropriate calculation function with the required parameters
                params = {field: field_map[field] for field in required_fields}
                score = await calculation_func(**params)
                metrics[metric_name] = score
                logger.info(f"Calculated {metric_name} score: {score:.4f}")
            except Exception as e:
                logger.error(f"Error calculating {metric_name}: {e}", exc_info=True)

        if not metrics:
            logger.warning("No metrics were successfully calculated")

        return metrics

    def _get_metric_description(self, metric_name: str) -> str:
        """Get description for a metric."""
        if metric_name in METRIC_REQUIREMENTS:
            return METRIC_REQUIREMENTS[metric_name]["description"]

        return f"Measures the {metric_name.replace('_', ' ')} of the response."
