# File: app/services/evaluation_service.py
from datetime import datetime
import hashlib
import json
from functools import lru_cache
from typing import Any, Dict, List, Optional, Type
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repositories.base import BaseRepository
from app.evaluation.factory import EvaluationMethodFactory
from app.evaluation.methods.base import BaseEvaluationMethod
from app.models.orm.models import (
    Dataset, Evaluation, EvaluationMethod, EvaluationResult,
    EvaluationStatus, MetricScore, MicroAgent, Prompt, User
)
from app.schema.evaluation_schema import (
    EvaluationCreate, EvaluationResultCreate, EvaluationUpdate,
    MetricScoreCreate
)


class EvaluationService:
    """Service for handling evaluation operations."""

    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session
        self.evaluation_repo = BaseRepository(Evaluation, db_session)
        self.result_repo = BaseRepository(EvaluationResult, db_session)
        self.metric_repo = BaseRepository(MetricScore, db_session)
        self.micro_agent_repo = BaseRepository(MicroAgent, db_session)
        self.dataset_repo = BaseRepository(Dataset, db_session)
        self.prompt_repo = BaseRepository(Prompt, db_session)

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

        evaluation = await self.evaluation_repo.create(evaluation_dict)

        # Queue the evaluation job (This would normally trigger a Celery task)
        # await self.queue_evaluation_job(evaluation.id)

        return evaluation

    async def get_evaluation(self, evaluation_id: UUID) -> Optional[Evaluation]:
        """
        Get evaluation by ID.

        Args:
            evaluation_id: Evaluation ID

        Returns:
            Optional[Evaluation]: Evaluation if found, None otherwise
        """
        return await self.evaluation_repo.get(evaluation_id)

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
        """
        # Filter out None values
        update_data = {
            k: v for k, v in evaluation_data.model_dump().items() if v is not None
        }

        if not update_data:
            # No update needed
            return await self.evaluation_repo.get(evaluation_id)

        return await self.evaluation_repo.update(evaluation_id, update_data)

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
        return await self.evaluation_repo.get_multi(skip=skip, limit=limit, filters=filters)

    async def delete_evaluation(self, evaluation_id: UUID) -> bool:
        """
        Delete evaluation by ID.

        Args:
            evaluation_id: Evaluation ID

        Returns:
            bool: True if deleted, False if not found
        """
        return await self.evaluation_repo.delete(evaluation_id)

    async def start_evaluation(self, evaluation_id: UUID) -> Optional[Evaluation]:
        """
        Start an evaluation.

        Args:
            evaluation_id: Evaluation ID

        Returns:
            Optional[Evaluation]: Updated evaluation if found, None otherwise
        """
        evaluation = await self.evaluation_repo.get(evaluation_id)
        if not evaluation:
            return None

        if evaluation.status != EvaluationStatus.PENDING:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Evaluation is already in {evaluation.status} status"
            )

        # Update status and start time
        update_data = {
            "status": EvaluationStatus.RUNNING,
            "start_time": datetime.now()
        }

        return await self.evaluation_repo.update(evaluation_id, update_data)

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
        """
        evaluation = await self.evaluation_repo.get(evaluation_id)
        if not evaluation:
            return None

        if evaluation.status != EvaluationStatus.RUNNING:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Evaluation is not in RUNNING status"
            )

        # Update status and end time
        update_data = {
            "status": EvaluationStatus.COMPLETED if success else EvaluationStatus.FAILED,
            "end_time": datetime.now()
        }

        return await self.evaluation_repo.update(evaluation_id, update_data)

    async def create_evaluation_result(
            self, result_data: EvaluationResultCreate
    ) -> EvaluationResult:
        """
        Create a new evaluation result.

        Args:
            result_data: Evaluation result data

        Returns:
            EvaluationResult: Created evaluation result
        """
        # Create evaluation result
        result_dict = result_data.model_dump(exclude={"metric_scores"})
        result = await self.result_repo.create(result_dict)

        # Create metric scores if provided
        if result_data.metric_scores:
            for metric_data in result_data.metric_scores:
                metric_dict = metric_data.model_dump()
                metric_dict["result_id"] = result.id
                await self.metric_repo.create(metric_dict)

        return result

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
        return await self.result_repo.get_multi(filters={"evaluation_id": evaluation_id})

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
        results = await self.result_repo.get_multi(
            filters={
                "evaluation_id": evaluation_id,
                "dataset_sample_id": dataset_sample_id
            }
        )
        return results[0] if results else None

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
        return await self.metric_repo.get_multi(filters={"result_id": result_id})

    async def queue_evaluation_job(self, evaluation_id: UUID) -> None:
        """
        Queue an evaluation job for background processing.

        This would typically send a task to Celery.

        Args:
            evaluation_id: Evaluation ID
        """
        # In a real implementation, this would send a task to Celery
        from app.workers.tasks import run_evaluation_task
        run_evaluation_task.delay(str(evaluation_id))

        # Start the evaluation (in this example - would normally be done by the worker)
        await self.start_evaluation(evaluation_id)

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
            return EvaluationMethodFactory.create(method, self.db_session)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )