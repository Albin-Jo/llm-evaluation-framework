# File: backend/app/services/evaluation_service.py
import asyncio
import datetime
import hashlib
import json
import logging
import traceback
from functools import lru_cache
from typing import Any, Dict, List, Optional, Type, Union, Tuple
from uuid import UUID

from fastapi import BackgroundTasks, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.repositories.base import BaseRepository
from backend.app.evaluation.factory import EvaluationMethodFactory
from backend.app.evaluation.methods.base import BaseEvaluationMethod
from backend.app.db.models.orm.models import (
    Dataset, Evaluation, EvaluationMethod, EvaluationResult,
    EvaluationStatus, MetricScore, MicroAgent, Prompt, User
)
from backend.app.db.schema.evaluation_schema import (
    EvaluationCreate, EvaluationResultCreate, EvaluationUpdate,
    MetricScoreCreate
)

# Configure logging
logger = logging.getLogger(__name__)


class EvaluationService:
    """Service for handling evaluation operations."""

    def __init__(self, db_session: AsyncSession):
        """
        Initialize the evaluation service.

        Args:
            db_session: Database session
        """
        self.db_session = db_session
        self.evaluation_repo = BaseRepository(Evaluation, db_session)
        self.result_repo = BaseRepository(EvaluationResult, db_session)
        self.metric_repo = BaseRepository(MetricScore, db_session)
        self.micro_agent_repo = BaseRepository(MicroAgent, db_session)
        self.dataset_repo = BaseRepository(Dataset, db_session)
        self.prompt_repo = BaseRepository(Prompt, db_session)

    # File: backend/app/services/evaluation_service.py
    async def create_evaluation(
            self, evaluation_data: EvaluationCreate, user: User
    ) -> Evaluation:
        """
        Create a new evaluation.

        Args:
            evaluation_data: Evaluation data
            user: Current user

        Returns:
            Evaluation: Created evaluation

        Raises:
            HTTPException: If referenced entities don't exist
        """
        # Verify that referenced entities exist
        micro_agent = await self.micro_agent_repo.get(evaluation_data.micro_agent_id)
        if not micro_agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"MicroAgent with ID {evaluation_data.micro_agent_id} not found"
            )

        dataset = await self.dataset_repo.get(evaluation_data.dataset_id)
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset with ID {evaluation_data.dataset_id} not found"
            )

        prompt = await self.prompt_repo.get(evaluation_data.prompt_id)
        if not prompt:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Prompt with ID {evaluation_data.prompt_id} not found"
            )

        # Create evaluation
        evaluation_dict = evaluation_data.model_dump()
        evaluation_dict["created_by_id"] = user.id

        try:
            evaluation = await self.evaluation_repo.create(evaluation_dict)
            logger.info(f"Created evaluation {evaluation.id} by user {user.id}")
            return evaluation
        except Exception as e:
            logger.error(f"Failed to create evaluation: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create evaluation: {str(e)}"
            )

    async def get_evaluation(self, evaluation_id: UUID) -> Optional[Evaluation]:
        """
        Get evaluation by ID.

        Args:
            evaluation_id: Evaluation ID

        Returns:
            Optional[Evaluation]: Evaluation if found, None otherwise
        """
        try:
            return await self.evaluation_repo.get(evaluation_id)
        except Exception as e:
            logger.error(f"Error retrieving evaluation {evaluation_id}: {str(e)}")
            return None

    async def update_evaluation(
            self, evaluation_id: UUID, evaluation_data: EvaluationUpdate
    ) -> Optional[Evaluation]:
        """
        Update evaluation by ID.

        Args:
            evaluation_id: Evaluation ID
            evaluation_data: Evaluation update data

        Returns:
            Optional[Evaluation]: Updated evaluation if found, None otherwise

        Raises:
            HTTPException: If update fails
        """
        # Filter out None values
        update_data = {
            k: v for k, v in evaluation_data.model_dump().items() if v is not None
        }

        if not update_data:
            # No update needed
            return await self.evaluation_repo.get(evaluation_id)

        try:
            evaluation = await self.evaluation_repo.update(evaluation_id, update_data)
            if evaluation:
                logger.info(f"Updated evaluation {evaluation_id}: {update_data}")
            else:
                logger.warning(f"Failed to update evaluation {evaluation_id}: not found")

            return evaluation
        except Exception as e:
            logger.error(f"Error updating evaluation {evaluation_id}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to update evaluation: {str(e)}"
            )

    async def list_evaluations(
            self, skip: int = 0, limit: int = 100, filters: Dict[str, Any] = None
    ) -> List[Evaluation]:
        """
        List evaluations with pagination and optional filtering.

        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            filters: Optional filters

        Returns:
            List[Evaluation]: List of evaluations
        """
        try:
            return await self.evaluation_repo.get_multi(skip=skip, limit=limit, filters=filters)
        except Exception as e:
            logger.error(f"Error listing evaluations: {str(e)}")
            return []

    async def delete_evaluation(self, evaluation_id: UUID) -> bool:
        """
        Delete evaluation by ID.

        Args:
            evaluation_id: Evaluation ID

        Returns:
            bool: True if deleted, False if not found
        """
        try:
            # First, check if evaluation exists
            evaluation = await self.evaluation_repo.get(evaluation_id)
            if not evaluation:
                logger.warning(f"Attempted to delete non-existent evaluation {evaluation_id}")
                return False

            # Delete related results first
            results = await self.result_repo.get_multi(filters={"evaluation_id": evaluation_id})

            for result in results:
                # Delete related metric scores
                await self.metric_repo.delete_multi(filters={"result_id": result.id})
                # Delete result
                await self.result_repo.delete(result.id)

            # Finally delete the evaluation
            success = await self.evaluation_repo.delete(evaluation_id)
            if success:
                logger.info(f"Deleted evaluation {evaluation_id} and related data")
            return success

        except Exception as e:
            logger.error(f"Error deleting evaluation {evaluation_id}: {str(e)}")
            return False

    async def start_evaluation(self, evaluation_id: UUID) -> Optional[Evaluation]:
        """
        Start an evaluation.

        Args:
            evaluation_id: Evaluation ID

        Returns:
            Optional[Evaluation]: Updated evaluation if found, None otherwise

        Raises:
            HTTPException: If evaluation cannot be started
        """
        try:
            evaluation = await self.evaluation_repo.get(evaluation_id)
            if not evaluation:
                logger.warning(f"Attempted to start non-existent evaluation {evaluation_id}")
                return None

            if evaluation.status != EvaluationStatus.PENDING:
                msg = f"Evaluation is already in {evaluation.status} status"
                logger.warning(f"Cannot start evaluation {evaluation_id}: {msg}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=msg
                )

            # Update status and start time
            now = datetime.datetime.now()
            update_data = {
                "status": EvaluationStatus.RUNNING,
                "start_time": now
            }

            updated = await self.evaluation_repo.update(evaluation_id, update_data)
            logger.info(f"Started evaluation {evaluation_id} at {now}")
            return updated

        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            logger.error(f"Error starting evaluation {evaluation_id}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to start evaluation: {str(e)}"
            )

    async def complete_evaluation(
            self, evaluation_id: UUID, success: bool = True
    ) -> Optional[Evaluation]:
        """
        Mark an evaluation as completed or failed.

        Args:
            evaluation_id: Evaluation ID
            success: Whether the evaluation was successful

        Returns:
            Optional[Evaluation]: Updated evaluation if found, None otherwise

        Raises:
            HTTPException: If evaluation cannot be completed
        """
        try:
            evaluation = await self.evaluation_repo.get(evaluation_id)
            if not evaluation:
                logger.warning(f"Attempted to complete non-existent evaluation {evaluation_id}")
                return None

            if evaluation.status not in [EvaluationStatus.RUNNING, EvaluationStatus.PENDING]:
                msg = f"Evaluation is not in RUNNING or PENDING status (current: {evaluation.status})"
                logger.warning(f"Cannot complete evaluation {evaluation_id}: {msg}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=msg
                )

            # Update status and end time
            now = datetime.datetime.now()
            new_status = EvaluationStatus.COMPLETED if success else EvaluationStatus.FAILED
            update_data = {
                "status": new_status,
                "end_time": now
            }

            updated = await self.evaluation_repo.update(evaluation_id, update_data)
            logger.info(f"Completed evaluation {evaluation_id} with status {new_status} at {now}")
            return updated

        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            logger.error(f"Error completing evaluation {evaluation_id}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to complete evaluation: {str(e)}"
            )

    async def create_evaluation_result(
            self, result_data: EvaluationResultCreate
    ) -> EvaluationResult:
        """
        Create a new evaluation result.

        Args:
            result_data: Evaluation result data

        Returns:
            EvaluationResult: Created evaluation result

        Raises:
            HTTPException: If result creation fails
        """
        try:
            # Create evaluation result
            result_dict = result_data.model_dump(exclude={"metric_scores"})
            result = await self.result_repo.create(result_dict)
            logger.debug(f"Created evaluation result {result.id} for evaluation {result_data.evaluation_id}")

            # Create metric scores if provided
            metric_scores = []
            if result_data.metric_scores:
                for metric_data in result_data.metric_scores:
                    metric_dict = metric_data.model_dump()
                    metric_dict["result_id"] = result.id
                    metric = await self.metric_repo.create(metric_dict)
                    metric_scores.append(metric)

                logger.debug(f"Created {len(metric_scores)} metric scores for result {result.id}")

            return result

        except Exception as e:
            logger.error(f"Error creating evaluation result: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create evaluation result: {str(e)}"
            )

    async def get_evaluation_results(
            self, evaluation_id: UUID
    ) -> List[EvaluationResult]:
        """
        Get all results for an evaluation.

        Args:
            evaluation_id: Evaluation ID

        Returns:
            List[EvaluationResult]: List of evaluation results
        """
        try:
            results = await self.result_repo.get_multi(filters={"evaluation_id": evaluation_id})
            logger.debug(f"Retrieved {len(results)} results for evaluation {evaluation_id}")
            return results
        except Exception as e:
            logger.error(f"Error retrieving results for evaluation {evaluation_id}: {str(e)}")
            return []

    @lru_cache(maxsize=128)
    async def get_cached_evaluation_result(
            self, evaluation_id: UUID, dataset_sample_id: str
    ) -> Optional[EvaluationResult]:
        """
        Get cached evaluation result.

        Args:
            evaluation_id: Evaluation ID
            dataset_sample_id: Dataset sample ID

        Returns:
            Optional[EvaluationResult]: Evaluation result if found, None otherwise
        """
        try:
            results = await self.result_repo.get_multi(
                filters={
                    "evaluation_id": evaluation_id,
                    "dataset_sample_id": dataset_sample_id
                }
            )
            return results[0] if results else None
        except Exception as e:
            logger.error(
                f"Error retrieving cached result for evaluation {evaluation_id}, sample {dataset_sample_id}: {str(e)}")
            return None

    def _compute_cache_key(self, input_data: Dict[str, Any], method: EvaluationMethod) -> str:
        """
        Compute a cache key for evaluation inputs.

        Args:
            input_data: Input data
            method: Evaluation method

        Returns:
            str: Cache key
        """
        serialized = json.dumps(input_data, sort_keys=True)
        return f"{method}:{hashlib.md5(serialized.encode()).hexdigest()}"

    async def get_metric_scores(
            self, result_id: UUID
    ) -> List[MetricScore]:
        """
        Get all metric scores for an evaluation result.

        Args:
            result_id: Evaluation result ID

        Returns:
            List[MetricScore]: List of metric scores
        """
        try:
            metrics = await self.metric_repo.get_multi(filters={"result_id": result_id})
            logger.debug(f"Retrieved {len(metrics)} metric scores for result {result_id}")
            return metrics
        except Exception as e:
            logger.error(f"Error retrieving metric scores for result {result_id}: {str(e)}")
            return []

    async def queue_evaluation_job(self, evaluation_id: UUID) -> None:
        """
        Queue an evaluation job for background processing.

        This can either send a task to Celery or run directly in a background task.

        Args:
            evaluation_id: Evaluation ID

        Raises:
            HTTPException: If queueing fails
        """
        try:
            # Get the evaluation to check if it can be queued
            evaluation = await self.evaluation_repo.get(evaluation_id)
            if not evaluation:
                raise ValueError(f"Evaluation {evaluation_id} not found")

            if evaluation.status != EvaluationStatus.PENDING:
                raise ValueError(f"Evaluation is already in {evaluation.status} status")

            # Update to running status
            await self.start_evaluation(evaluation_id)

            # In a real implementation, this would send a task to Celery
            from backend.app.workers.tasks import run_evaluation_task

            # Check if we're running in a test environment
            import os
            if os.getenv("APP_ENV") == "testing":
                # For testing, run directly
                asyncio.create_task(self._run_evaluation_directly(evaluation_id))
            else:
                # For production, use Celery
                run_evaluation_task.delay(str(evaluation_id))

            logger.info(f"Queued evaluation job for evaluation {evaluation_id}")

        except ValueError as e:
            logger.warning(f"Cannot queue evaluation job: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        except Exception as e:
            logger.error(f"Error queueing evaluation job: {str(e)}")
            # Set evaluation to failed status
            try:
                await self.complete_evaluation(evaluation_id, success=False)
            except Exception:
                pass

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to queue evaluation job: {str(e)}"
            )

    async def _run_evaluation_directly(self, evaluation_id: UUID) -> None:
        """
        Run evaluation directly (for testing or when Celery is not available).

        Args:
            evaluation_id: Evaluation ID
        """
        try:
            # Get the evaluation
            evaluation = await self.evaluation_repo.get(evaluation_id)
            if not evaluation:
                logger.error(f"Evaluation {evaluation_id} not found")
                return

            # Get the evaluation method handler
            method_handler = await self.get_evaluation_method_handler(evaluation.method)

            # Run the evaluation
            results = await method_handler.run_evaluation(evaluation)

            # Process results
            for result_data in results:
                await self.create_evaluation_result(result_data)

            # Mark as completed
            await self.complete_evaluation(evaluation_id, success=True)

            logger.info(f"Completed evaluation {evaluation_id} with {len(results)} results")

        except Exception as e:
            logger.exception(f"Error running evaluation {evaluation_id} directly: {str(e)}")

            # Mark as failed
            try:
                await self.complete_evaluation(evaluation_id, success=False)
            except Exception:
                pass

    async def get_evaluation_method_handler(
            self, method: EvaluationMethod
    ) -> BaseEvaluationMethod:
        """
        Get the appropriate evaluation method handler based on the method.

        Args:
            method: Evaluation method

        Returns:
            BaseEvaluationMethod: Evaluation method handler instance

        Raises:
            HTTPException: If method is not supported
        """
        try:
            handler = EvaluationMethodFactory.create(method, self.db_session)
            return handler
        except ValueError as e:
            logger.error(f"Unsupported evaluation method: {method}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        except Exception as e:
            logger.error(f"Error creating evaluation method handler: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error initializing evaluation method: {str(e)}"
            )

    async def get_evaluation_statistics(self, evaluation_id: UUID) -> Dict[str, Any]:
        """
        Get statistics for an evaluation.

        Args:
            evaluation_id: Evaluation ID

        Returns:
            Dict[str, Any]: Evaluation statistics
        """
        try:
            # Get the evaluation
            evaluation = await self.evaluation_repo.get(evaluation_id)
            if not evaluation:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Evaluation with ID {evaluation_id} not found"
                )

            # Get results
            results = await self.get_evaluation_results(evaluation_id)

            # Initialize statistics
            stats = {
                "total_samples": len(results),
                "avg_overall_score": 0.0,
                "metrics": {},
                "processing_time": {
                    "min": None,
                    "max": None,
                    "avg": None
                },
                "error_rate": 0.0
            }

            if not results:
                return stats

            # Calculate statistics
            total_overall_score = 0.0
            processing_times = []
            error_count = 0
            metric_scores = {}

            for result in results:
                # Calculate overall score
                if result.overall_score is not None:
                    total_overall_score += result.overall_score
                else:
                    error_count += 1

                # Track processing time
                if result.processing_time_ms is not None:
                    processing_times.append(result.processing_time_ms)

                # Collect metric scores
                result_metrics = await self.get_metric_scores(result.id)
                for metric in result_metrics:
                    if metric.name not in metric_scores:
                        metric_scores[metric.name] = []
                    metric_scores[metric.name].append(metric.value)

            # Calculate averages
            if len(results) > error_count:
                stats["avg_overall_score"] = total_overall_score / (len(results) - error_count)

            if processing_times:
                stats["processing_time"]["min"] = min(processing_times)
                stats["processing_time"]["max"] = max(processing_times)
                stats["processing_time"]["avg"] = sum(processing_times) / len(processing_times)

            stats["error_rate"] = error_count / len(results) if results else 0.0

            # Calculate metric statistics
            for metric_name, values in metric_scores.items():
                if values:
                    stats["metrics"][metric_name] = {
                        "avg": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                        "count": len(values)
                    }

            return stats

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting evaluation statistics: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error getting evaluation statistics: {str(e)}"
            )

        dataset = await self.dataset_repo.get(evaluation_data.dataset_id)
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset with ID {evaluation_data.dataset_id} not found"
            )

        prompt = await self.prompt_repo.get(evaluation_data.prompt_id)
        if not prompt:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Prompt with ID {evaluation_data.prompt_id} not found"
            )

        # Create evaluation
        evaluation_dict = evaluation_data.model_dump()
        evaluation_dict["created_by_id"] = user.id

        try:
            evaluation = await self.evaluation_repo.create(evaluation_dict)
            logger.info(f"Created evaluation {evaluation.id} by user {user.id}")
            return evaluation
        except Exception as e:
            logger.error(f"Failed to create evaluation: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create evaluation: {str(e)}"
            )