# backend/app/services/evaluation_service.py
import datetime
import logging
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import asc, desc, select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.app.db.models.orm import (
    Dataset, Evaluation, EvaluationResult,
    EvaluationStatus, MetricScore, Agent, Prompt, EvaluationMethod
)
from backend.app.db.repositories.base import BaseRepository
from backend.app.db.schema.evaluation_schema import (
    EvaluationCreate, EvaluationResultCreate, EvaluationUpdate
)
from backend.app.evaluation.methods.base import BaseEvaluationMethod
from backend.app.evaluation.metrics.ragas_metrics import DATASET_TYPE_METRICS

# Configure logging
logger = logging.getLogger(__name__)


async def _run_evaluation_as_separate_task(evaluation_id_str: str) -> None:
    """
    Run evaluation in a completely separate task with its own database session.

    This prevents session conflicts when running async tasks.

    Args:
        evaluation_id_str: Evaluation ID as string
    """
    from backend.app.db.session import db_session
    from uuid import UUID

    evaluation_id = UUID(evaluation_id_str)

    # Create a new database session specifically for this task
    async with db_session() as session:
        # Create a new service instance with this session
        service = EvaluationService(session)

        try:
            # Get the evaluation
            evaluation = await service.get_evaluation(evaluation_id)
            if not evaluation:
                logger.error(f"Evaluation {evaluation_id} not found")
                return

            # Check if already running
            if evaluation.status not in [EvaluationStatus.PENDING, EvaluationStatus.RUNNING]:
                logger.warning(f"Evaluation {evaluation_id} is in {evaluation.status} state, not running")
                return

            # Make sure it's running
            if evaluation.status == EvaluationStatus.PENDING:
                await service.start_evaluation(evaluation_id)

            # Get the evaluation method handler
            method_handler = await service.get_evaluation_method_handler(evaluation.method)

            # Run the evaluation
            results = await method_handler.run_evaluation(evaluation)

            # Process results
            for result_data in results:
                await service.create_evaluation_result(result_data)

            # IMPORTANT: Check current status before marking as completed
            # This prevents trying to complete an already completed evaluation
            current_status = (await service.get_evaluation(evaluation_id)).status
            if current_status in [EvaluationStatus.RUNNING, EvaluationStatus.PENDING]:
                await service.complete_evaluation(evaluation_id, success=True)
                logger.info(f"Completed evaluation {evaluation_id} with {len(results)} results")
            else:
                logger.info(f"Evaluation {evaluation_id} already in {current_status} state, skipping completion")

        except Exception as e:
            logger.exception(f"Error running evaluation {evaluation_id} in separate task: {str(e)}")

            # Mark as failed only if still in RUNNING or PENDING state
            try:
                current_status = (await service.get_evaluation(evaluation_id)).status
                if current_status in [EvaluationStatus.RUNNING, EvaluationStatus.PENDING]:
                    await service.complete_evaluation(evaluation_id, success=False)
                else:
                    logger.info(f"Evaluation {evaluation_id} already in {current_status} state, not marking as failed")
            except Exception as complete_error:
                logger.error(f"Failed to mark evaluation as failed: {str(complete_error)}")


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
        self.agent_repo = BaseRepository(Agent, db_session)
        self.dataset_repo = BaseRepository(Dataset, db_session)
        self.prompt_repo = BaseRepository(Prompt, db_session)

    async def create_evaluation(
            self, evaluation_data: EvaluationCreate,
    ) -> Evaluation:
        """
        Create a new evaluation.

        Args:
            evaluation_data: Evaluation data

        Returns:
            Evaluation: Created evaluation

        Raises:
            HTTPException: If referenced entities don't exist or validation fails
        """
        # Verify that referenced entities exist
        agent = await self.agent_repo.get(evaluation_data.agent_id)
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent with ID {evaluation_data.agent_id} not found"
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

        # Validate metrics based on dataset type
        await self._validate_metrics_for_dataset(evaluation_data.metrics, dataset)

        # Use a transaction to ensure atomicity
        async with self.db_session.begin():
            # Create evaluation
            evaluation_dict = evaluation_data.model_dump()

            try:
                evaluation = await self.evaluation_repo.create(evaluation_dict)
                logger.info(f"Created evaluation {evaluation.id}")
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

    async def get_evaluation_with_relationships(self, evaluation_id: UUID) -> Optional[Tuple[Evaluation, List[Dict]]]:
        """
        Get evaluation with all its relationships in one query.

        Args:
            evaluation_id: Evaluation ID

        Returns:
            Tuple[Evaluation, List[Dict]]: Evaluation and its processed results
        """
        try:
            # Create a query that loads evaluation with results and metric scores in one go
            query = (
                select(Evaluation)
                .options(
                    selectinload(Evaluation.results)
                    .selectinload(EvaluationResult.metric_scores)
                )
                .where(Evaluation.id == evaluation_id)
            )

            result = await self.db_session.execute(query)
            evaluation = result.scalars().first()

            if not evaluation:
                return None, []

            # Process results
            processed_results = []
            for result in evaluation.results:
                result_dict = {
                    "id": result.id,
                    "evaluation_id": result.evaluation_id,
                    "overall_score": result.overall_score,
                    "raw_results": result.raw_results,
                    "dataset_sample_id": result.dataset_sample_id,
                    "input_data": result.input_data,
                    "output_data": result.output_data,
                    "processing_time_ms": result.processing_time_ms,
                    "created_at": result.created_at,
                    "updated_at": result.updated_at,
                    "metric_scores": [
                        {
                            "id": score.id,
                            "name": score.name,
                            "value": score.value,
                            "weight": score.weight,
                            "meta_info": score.meta_info,
                            "result_id": score.result_id,
                            "created_at": score.created_at,
                            "updated_at": score.updated_at
                        }
                        for score in result.metric_scores
                    ]
                }
                processed_results.append(result_dict)

            return evaluation, processed_results
        except Exception as e:
            logger.error(f"Error retrieving evaluation with relationships {evaluation_id}: {str(e)}")
            return None, []

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

        # Use a transaction to ensure atomicity
        async with self.db_session.begin():
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

    async def count_evaluations(self, filters: Dict[str, Any] = None) -> int:
        """
        Count evaluations with optional filtering.

        Args:
            filters: Optional filters

        Returns:
            int: Count of matching evaluations
        """
        try:
            # Create a direct count query with SQLAlchemy
            query = select(func.count()).select_from(Evaluation)

            # Apply filters directly
            if filters:
                for field, value in filters.items():
                    if hasattr(Evaluation, field):
                        # Handle special case for string fields with LIKE operation
                        if isinstance(value, str) and field not in ["status", "method"]:
                            query = query.where(getattr(Evaluation, field).ilike(f"%{value}%"))
                        else:
                            query = query.where(getattr(Evaluation, field) == value)

            # Execute query directly with session
            result = await self.db_session.execute(query)
            count = result.scalar_one_or_none() or 0

            return count
        except Exception as e:
            logger.error(f"Error counting evaluations: {str(e)}")
            return 0

    async def list_evaluations(
            self,
            skip: int = 0,
            limit: int = 100,
            filters: Dict[str, Any] = None,
            sort_options: Dict[str, str] = None
    ) -> List[Evaluation]:
        """
        List evaluations with pagination and optional filtering.

        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            filters: Optional filters
            sort_options: Sorting options (field and direction)

        Returns:
            List[Evaluation]: List of evaluations
        """
        try:
            # Apply sorting if provided
            sort_expr = None
            if sort_options and "sort_by" in sort_options:
                sort_by = sort_options["sort_by"]
                sort_dir = sort_options.get("sort_dir", "desc").lower()

                # Validate that sort_by is a valid field on the model
                if not hasattr(Evaluation, sort_by):
                    logger.warning(f"Invalid sort field: {sort_by}")
                    # Default to created_at if invalid field
                    sort_by = "created_at"

                # Create sort expression
                if sort_dir == "asc":
                    sort_expr = asc(getattr(Evaluation, sort_by))
                else:
                    sort_expr = desc(getattr(Evaluation, sort_by))

            # Add relationships to load eagerly
            load_relationships = ["agent", "dataset", "prompt"]

            return await self.evaluation_repo.get_multi(
                skip=skip,
                limit=limit,
                filters=filters,
                sort=sort_expr,
                load_relationships=load_relationships
            )
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
        # Use a transaction to ensure atomicity
        async with self.db_session.begin():
            try:
                # First, check if evaluation exists
                evaluation = await self.evaluation_repo.get(evaluation_id)
                if not evaluation:
                    logger.warning(f"Attempted to delete non-existent evaluation {evaluation_id}")
                    return False

                # Get all results for this evaluation
                results = await self.result_repo.get_multi(filters={"evaluation_id": evaluation_id})

                # Delete metric scores for each result
                for result in results:
                    deleted_metrics = await self.metric_repo.delete_multi(filters={"result_id": result.id})
                    logger.debug(f"Deleted {deleted_metrics} metric scores for result {result.id}")

                # Delete all results for this evaluation
                deleted_results = await self.result_repo.delete_multi(filters={"evaluation_id": evaluation_id})
                logger.debug(f"Deleted {deleted_results} results for evaluation {evaluation_id}")

                # Delete the evaluation
                success = await self.evaluation_repo.delete(evaluation_id)
                if success:
                    logger.info(f"Deleted evaluation {evaluation_id} and related data")
                return success

            except Exception as e:
                logger.error(f"Error deleting evaluation {evaluation_id}: {str(e)}")
                return False

    async def _validate_metrics_for_dataset(self, metrics: Optional[List[str]], dataset: Dataset) -> None:
        """
        Validate that the selected metrics are appropriate for the dataset type.

        Args:
            metrics: List of metrics to validate
            dataset: Dataset to validate against

        Raises:
            HTTPException: If metrics are invalid for the dataset type
        """
        if not metrics:
            return  # No metrics specified, will use defaults

        # Get allowed metrics for this dataset type
        allowed_metrics = DATASET_TYPE_METRICS.get(dataset.type, [])

        if not allowed_metrics:
            logger.warning(f"No metrics defined for dataset type {dataset.type}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No metrics are defined for dataset type {dataset.type}"
            )

        # Check if any specified metrics are not allowed
        invalid_metrics = [m for m in metrics if m not in allowed_metrics]

        if invalid_metrics:
            logger.warning(f"Invalid metrics for dataset type {dataset.type}: {invalid_metrics}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Metrics {invalid_metrics} are not valid for {dataset.type} datasets. "
                       f"Valid metrics are: {allowed_metrics}"
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
        # Use a transaction to ensure atomicity
        async with self.db_session.begin():
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
            self, evaluation_id: UUID, skip: int = 0, limit: int = 100
    ) -> List[EvaluationResult]:
        """
        Get results for an evaluation with pagination.

        Args:
            evaluation_id: Evaluation ID
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List[EvaluationResult]: List of evaluation results
        """
        try:
            results = await self.result_repo.get_multi(
                skip=skip,
                limit=limit,
                filters={"evaluation_id": evaluation_id},
                load_relationships=["metric_scores"]
            )
            logger.debug(f"Retrieved {len(results)} results for evaluation {evaluation_id}")
            return results
        except Exception as e:
            logger.error(f"Error retrieving results for evaluation {evaluation_id}: {str(e)}")
            return []

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
        # Use a transaction to ensure atomicity
        async with self.db_session.begin():
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

    async def queue_evaluation_job(self, evaluation_id: UUID) -> None:
        """
        Queue an evaluation job for background processing.

        This will send a task to Celery or run directly in a background task.

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

            # In production, use Celery
            from backend.app.workers.tasks import run_evaluation_task

            # Queue the task asynchronously so the API can return immediately
            try:
                # Use Celery in production
                run_evaluation_task.delay(str(evaluation_id))
                logger.info(f"Queued evaluation job {evaluation_id} to Celery")
            except Exception as e:
                # Fallback to direct execution if Celery is not available
                logger.warning(f"Failed to queue to Celery: {e}. Running as separate task.")

                # IMPORTANT: Don't use the current database session for the task
                # Create a separate, detached task instead
                # This prevents session conflicts
                import asyncio
                asyncio.create_task(_run_evaluation_as_separate_task(str(evaluation_id)))
                logger.info(f"Started evaluation {evaluation_id} as separate task")

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
            except Exception as complete_error:
                logger.error(f"Failed to mark evaluation as failed: {str(complete_error)}")

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to queue evaluation job: {str(e)}"
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
        # Use a transaction to ensure atomicity
        async with self.db_session.begin():
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

    async def get_evaluation_progress(self, evaluation_id: UUID) -> Dict[str, Any]:
        """
        Get the progress of an evaluation.

        Args:
            evaluation_id: Evaluation ID

        Returns:
            Dict[str, Any]: Evaluation progress information
        """
        try:
            # Get evaluation
            evaluation = await self.evaluation_repo.get(evaluation_id)
            if not evaluation:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Evaluation with ID {evaluation_id} not found"
                )

            # Get dataset to calculate total items
            dataset = await self.dataset_repo.get(evaluation.dataset_id)
            total_items = dataset.row_count if dataset and dataset.row_count else 0

            # Get completed results count (instead of loading full objects)
            query = select(func.count()).select_from(EvaluationResult).where(
                EvaluationResult.evaluation_id == evaluation_id
            )
            result = await self.db_session.execute(query)
            completed_items = result.scalar_one_or_none() or 0

            # Calculate progress percentage
            progress_pct = (completed_items / total_items * 100) if total_items > 0 else 0

            # Calculate running time safely handling timezone differences
            running_time_seconds = 0
            if evaluation.start_time:
                if evaluation.end_time:
                    # Both start and end time exist
                    # Convert to UTC if they have tzinfo, or assume they're in the same timezone
                    start = evaluation.start_time.replace(
                        tzinfo=None) if evaluation.start_time.tzinfo else evaluation.start_time
                    end = evaluation.end_time.replace(
                        tzinfo=None) if evaluation.end_time.tzinfo else evaluation.end_time
                    running_time_seconds = (end - start).total_seconds()
                else:
                    # Only start time exists, use current time for comparison
                    # Convert start_time to naive if it has tzinfo
                    start = evaluation.start_time.replace(
                        tzinfo=None) if evaluation.start_time.tzinfo else evaluation.start_time
                    now = datetime.datetime.now()
                    running_time_seconds = (now - start).total_seconds()

            # Get estimated time to completion
            estimated_time_remaining = None
            if evaluation.status == EvaluationStatus.RUNNING and completed_items > 0 and total_items > completed_items:
                if running_time_seconds > 0:
                    time_per_item = running_time_seconds / completed_items
                    estimated_time_remaining = time_per_item * (total_items - completed_items)

            # Return progress information
            return {
                "status": evaluation.status,
                "total_items": total_items,
                "completed_items": completed_items,
                "progress_percentage": round(progress_pct, 2),
                "start_time": evaluation.start_time,
                "end_time": evaluation.end_time,
                "running_time_seconds": round(running_time_seconds, 2),
                "estimated_time_remaining_seconds": round(estimated_time_remaining,
                                                          2) if estimated_time_remaining else None
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting evaluation progress: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error getting evaluation progress: {str(e)}"
            )

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
        from backend.app.evaluation.factory import EvaluationMethodFactory

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