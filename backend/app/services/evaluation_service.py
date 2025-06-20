import datetime
import logging
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import asc, desc
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.exceptions import NotFoundException, AuthorizationException
from backend.app.db.models.orm import (
    Dataset, Evaluation, EvaluationResult,
    EvaluationStatus, MetricScore, Agent, Prompt, EvaluationMethod, IntegrationType
)
from backend.app.db.repositories.base import BaseRepository
from backend.app.db.repositories.evaluation_repository import EvaluationRepository
from backend.app.db.schema.evaluation_schema import (
    EvaluationCreate, EvaluationResultCreate, EvaluationUpdate
)
from backend.app.evaluation.methods.base import BaseEvaluationMethod
from backend.app.evaluation.utils.progress import get_evaluation_result_count
from backend.app.services.impersonation_service import ImpersonationService
from backend.app.utils.credential_utils import encrypt_credentials

# Configure logging
logger = logging.getLogger(__name__)


async def _run_evaluation_as_separate_task(evaluation_id_str: str, jwt_token: Optional[str] = None) -> None:
    """Run evaluation with impersonation support."""
    from backend.app.db.session import db_session
    from uuid import UUID

    evaluation_id = UUID(evaluation_id_str)

    async with db_session() as session:
        service = EvaluationService(session)

        try:
            evaluation = await service.get_evaluation(evaluation_id)
            if not evaluation:
                logger.error(f"Evaluation {evaluation_id} not found")
                return

            # Check if already running
            if evaluation.status not in [EvaluationStatus.PENDING, EvaluationStatus.RUNNING]:
                logger.warning(f"Evaluation {evaluation_id} is in {evaluation.status} state, not running")
                return

            # Start evaluation
            if evaluation.status == EvaluationStatus.PENDING:
                now = datetime.datetime.now()
                update_data = {
                    "status": EvaluationStatus.RUNNING,
                    "start_time": now
                }
                await service.evaluation_repo.update(evaluation_id, update_data)
                logger.info(f"Started evaluation {evaluation_id} at {now}")

            # Get evaluation method handler
            method_handler = await service.get_evaluation_method_handler(evaluation.method)

            # Get impersonated token if available
            impersonated_token = await _get_impersonated_token(evaluation)

            # Use impersonated token for MCP agents, fallback to provided JWT token
            evaluation_token = impersonated_token or jwt_token

            # Run evaluation with the appropriate token
            results = await method_handler.run_evaluation(evaluation, jwt_token=evaluation_token)

            # Process results
            for result_data in results:
                try:
                    await service.create_evaluation_result(result_data)
                except Exception as result_error:
                    logger.error(f"Error creating result for evaluation {evaluation_id}: {str(result_error)}")

            # Mark as completed
            current_status = (await service.get_evaluation(evaluation_id)).status
            if current_status in [EvaluationStatus.RUNNING, EvaluationStatus.PENDING]:
                now = datetime.datetime.now()
                update_data = {
                    "status": EvaluationStatus.COMPLETED,
                    "end_time": now
                }
                await service.evaluation_repo.update(evaluation_id, update_data)
                logger.info(f"Completed evaluation {evaluation_id} with {len(results)} results")

        except Exception as e:
            logger.exception(f"Error running evaluation {evaluation_id}: {str(e)}")

            # Mark as failed
            try:
                current_status = (await service.get_evaluation(evaluation_id)).status
                if current_status in [EvaluationStatus.RUNNING, EvaluationStatus.PENDING]:
                    now = datetime.datetime.now()
                    update_data = {
                        "status": EvaluationStatus.FAILED,
                        "end_time": now
                    }
                    await service.evaluation_repo.update(evaluation_id, update_data)
                    logger.info(f"Marked evaluation {evaluation_id} as failed at {now}")
            except Exception as complete_error:
                logger.error(f"Failed to mark evaluation as failed: {str(complete_error)}")


async def _get_impersonated_token(evaluation: Evaluation) -> dict[str, Any] | None:
    """Get decrypted impersonated token for evaluation."""
    if not evaluation.impersonated_token:
        return None

    try:
        from backend.app.utils.credential_utils import decrypt_credentials
        return decrypt_credentials(evaluation.impersonated_token)
    except Exception as e:
        logger.error(f"Failed to decrypt impersonated token: {str(e)}")
        return None


class EvaluationService:
    """Service for handling evaluation operations with user access control."""

    def __init__(self, db_session: AsyncSession):
        """
        Initialize the evaluation service.

        Args:
            db_session: Database session
        """
        self.db_session = db_session
        # Use our enhanced repository for evaluations
        self.evaluation_repo = EvaluationRepository(db_session)
        self.result_repo = BaseRepository(EvaluationResult, db_session)
        self.metric_repo = BaseRepository(MetricScore, db_session)
        self.agent_repo = BaseRepository(Agent, db_session)
        self.dataset_repo = BaseRepository(Dataset, db_session)
        self.prompt_repo = BaseRepository(Prompt, db_session)
        self.impersonate = ImpersonationService()

    async def create_evaluation(
            self, evaluation_data: EvaluationCreate, jwt_token: Optional[str] = None
    ) -> Evaluation:
        """
        Create a new evaluation with user attribution.

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

        # Validate metrics based on dataset type AND evaluation method
        if evaluation_data.metrics:
            await self._validate_metrics_for_dataset(
                evaluation_data.metrics,
                dataset,
                evaluation_data.method
            )

        # Ensure a user ID is provided for attribution
        if not evaluation_data.created_by_id:
            logger.warning("No user ID provided for evaluation creation")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User ID is required for evaluation creation"
            )

        impersonation_data = {}
        if evaluation_data.impersonated_employee_id:
            # Verify agent is MCP type
            agent = await self.agent_repo.get(evaluation_data.agent_id)
            if not agent:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Agent with ID {evaluation_data.agent_id} not found"
                )

            if agent.integration_type != IntegrationType.MCP:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Impersonation is only supported for MCP agents"
                )

            # Get impersonation token
            try:
                impersonation_result = await self.impersonate.impersonate_user(
                    employee_id=evaluation_data.impersonated_employee_id, auth_token=jwt_token
                )

                # Store impersonation data
                impersonation_data = {
                    "impersonated_user_id": evaluation_data.impersonated_employee_id,
                    "impersonated_user_info": impersonation_result.get("user_info"),
                    "impersonated_token": encrypt_credentials(impersonation_result["token"])
                }

                logger.info(
                    f"Set up impersonation for evaluation with employee {evaluation_data.impersonated_employee_id}")

            except Exception as e:
                logger.error(f"Failed to set up impersonation: {str(e)}")
                raise

        # Create evaluation with impersonation data
        evaluation_dict = evaluation_data.model_dump(exclude={"impersonated_employee_id"})
        evaluation_dict.update(impersonation_data)

        try:
            evaluation = await self.evaluation_repo.create(evaluation_dict)
            logger.info(f"Created evaluation {evaluation.id} for user {evaluation_data.created_by_id}")
            return evaluation
        except Exception as e:
            logger.error(f"Failed to create evaluation: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create evaluation: {str(e)}"
            )

    async def complete_evaluation(
            self, evaluation_id: UUID, success: bool = True, user_id: Optional[UUID] = None
    ) -> Optional[Evaluation]:
        """
        Mark an evaluation as completed or failed with optional user verification.

        Args:
            evaluation_id: Evaluation ID
            success: Whether the evaluation was successful
            user_id: Optional user ID for ownership verification

        Returns:
            Optional[Evaluation]: Updated evaluation if found, None otherwise

        Raises:
            HTTPException: If evaluation cannot be completed
            AuthorizationException: If user doesn't have permission
        """
        # REMOVED TRANSACTION to avoid conflict with FastAPI dependency
        try:
            evaluation = await self.evaluation_repo.get(evaluation_id)
            if not evaluation:
                logger.warning(f"Attempted to complete non-existent evaluation {evaluation_id}")
                return None

            # Check ownership if user_id is provided
            if user_id and evaluation.created_by_id and evaluation.created_by_id != user_id:
                logger.warning(f"Unauthorized completion attempt on evaluation {evaluation_id} by user {user_id}")
                raise AuthorizationException(
                    detail="You don't have permission to complete this evaluation"
                )

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

        except (HTTPException, AuthorizationException):
            # Re-raise these exceptions
            raise
        except Exception as e:
            logger.error(f"Error completing evaluation {evaluation_id}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to complete evaluation: {str(e)}"
            )

    # File: backend/app/services/evaluation_service.py

    # Update the get_evaluation_progress method to rely more on database counts
    # when the cache is empty or unreliable

    async def get_evaluation_progress(self, evaluation_id: UUID, user_id: Optional[UUID] = None) -> Dict[str, Any]:
        """
        Get the progress of an evaluation with optional user verification.
        Uses database for tracking progress.
        """
        try:
            # Get evaluation with all necessary data
            evaluation = await self.evaluation_repo.get(evaluation_id)
            if not evaluation:
                raise NotFoundException(
                    resource="Evaluation",
                    resource_id=str(evaluation_id),
                    detail=f"Evaluation with ID {evaluation_id} not found"
                )

            # Check ownership if user_id is provided
            if user_id and evaluation.created_by_id and evaluation.created_by_id != user_id:
                logger.warning(f"Unauthorized progress check on evaluation {evaluation_id} by user {user_id}")
                raise AuthorizationException(
                    detail="You don't have permission to view this evaluation's progress"
                )

            # Get dataset to calculate total items
            dataset = await self.dataset_repo.get(evaluation.dataset_id)
            total_items = dataset.row_count if dataset and dataset.row_count else 0

            # Get current progress from the evaluation object
            processed_items = evaluation.processed_items or 0

            # Also check the actual results count in case processed_items is stale
            result_count = await get_evaluation_result_count(self.db_session, evaluation_id)

            # Use the max value between the two sources
            completed_items = max(processed_items, result_count)

            # Calculate progress percentage
            progress_pct = (completed_items / total_items * 100) if total_items > 0 else 0

            # Calculate running time safely handling timezone differences
            running_time_seconds = 0
            if evaluation.start_time:
                if evaluation.end_time:
                    # Both start and end time exist
                    start = evaluation.start_time.replace(
                        tzinfo=None) if evaluation.start_time.tzinfo else evaluation.start_time
                    end = evaluation.end_time.replace(
                        tzinfo=None) if evaluation.end_time.tzinfo else evaluation.end_time
                    running_time_seconds = (end - start).total_seconds()
                else:
                    # Only start time exists, use current time for comparison
                    start = evaluation.start_time.replace(
                        tzinfo=None) if evaluation.start_time.tzinfo else evaluation.start_time
                    now = datetime.datetime.now()
                    running_time_seconds = (now - start).total_seconds()

            # Get estimated time to completion
            estimated_time_remaining = None
            if evaluation.status == EvaluationStatus.RUNNING and 0 < completed_items < total_items:
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
                                                          2) if estimated_time_remaining else None,
                "last_updated": datetime.datetime.now().isoformat()
            }

        except (NotFoundException, AuthorizationException):
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
        if evaluation_data.metrics:
            await self._validate_metrics_for_dataset(evaluation_data.metrics, dataset)

        # Ensure a user ID is provided for attribution
        if not evaluation_data.created_by_id:
            logger.warning("No user ID provided for evaluation creation")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User ID is required for evaluation creation"
            )

        # Create evaluation - REMOVED TRANSACTION to avoid conflict with FastAPI dependency
        evaluation_dict = evaluation_data.model_dump()

        try:
            evaluation = await self.evaluation_repo.create(evaluation_dict)
            logger.info(f"Created evaluation {evaluation.id} for user {evaluation_data.created_by_id}")
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

    async def get_user_evaluation(self, evaluation_id: UUID, user_id: UUID) -> Optional[Evaluation]:
        """
        Get evaluation by ID with user ownership check.

        Args:
            evaluation_id: Evaluation ID
            user_id: User ID for ownership verification

        Returns:
            Optional[Evaluation]: Evaluation if found and owned by user, None otherwise
        """
        try:
            return await self.evaluation_repo.get_user_evaluation(evaluation_id, user_id)
        except Exception as e:
            logger.error(f"Error retrieving user evaluation {evaluation_id}: {str(e)}")
            return None

    async def get_evaluation_with_relationships(
            self,
            evaluation_id: UUID,
            user_id: Optional[UUID] = None
    ) -> Tuple[Optional[Evaluation], List[Dict]]:
        """
        Get evaluation with all its relationships in one query.
        Optionally filter by user ID to ensure ownership.

        Args:
            evaluation_id: Evaluation ID
            user_id: Optional user ID for ownership check

        Returns:
            Tuple[Evaluation, List[Dict]]: Evaluation and its processed results
        """
        try:
            # Use the repository method that loads relationships
            evaluation = await self.evaluation_repo.get_evaluation_with_details(
                evaluation_id,
                user_id=user_id
            )

            if not evaluation:
                logger.warning(f"Evaluation {evaluation_id} not found")
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
            self, evaluation_id: UUID, evaluation_data: EvaluationUpdate, user_id: Optional[UUID] = None
    ) -> Optional[Evaluation]:
        """
        Update evaluation by ID with optional user ownership check.

        Args:
            evaluation_id: Evaluation ID
            evaluation_data: Evaluation update data
            user_id: Optional user ID for ownership verification

        Returns:
            Optional[Evaluation]: Updated evaluation if found, None otherwise

        Raises:
            HTTPException: If update fails or user doesn't have permission
        """
        # Filter out None values
        update_data = {
            k: v for k, v in evaluation_data.model_dump().items() if v is not None
        }

        if not update_data:
            # No update needed
            return await self.evaluation_repo.get(evaluation_id)

        # Check ownership if user_id is provided
        if user_id:
            evaluation = await self.evaluation_repo.get_user_evaluation(evaluation_id, user_id)
            if not evaluation:
                logger.warning(f"Unauthorized update attempt on evaluation {evaluation_id} by user {user_id}")
                raise AuthorizationException(
                    detail="You don't have permission to update this evaluation"
                )

        # REMOVED TRANSACTION to avoid conflict with FastAPI dependency
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
            # Use repository method directly
            return await self.evaluation_repo.count(filters)
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

            # Debug log.json the query parameters
            logger.debug(f"Listing evaluations with filters={filters}, skip={skip}, limit={limit}, sort={sort_expr}")

            # Execute the query through the repository
            evaluations = await self.evaluation_repo.get_multi(
                skip=skip,
                limit=limit,
                filters=filters,
                sort=sort_expr,
                load_relationships=load_relationships
            )

            # Debug log.json the result count
            logger.debug(f"Query returned {len(evaluations)} evaluations")

            return evaluations
        except Exception as e:
            logger.error(f"Error listing evaluations: {str(e)}", exc_info=True)
            return []

    async def list_user_evaluations(
            self,
            user_id: UUID,
            skip: int = 0,
            limit: int = 100,
            status: Optional[EvaluationStatus] = None,
            sort_options: Dict[str, str] = None
    ) -> List[Evaluation]:
        """
        List evaluations for a specific user with optional status filter.

        Args:
            user_id: User ID to filter by
            skip: Number of records to skip
            limit: Maximum number of records to return
            status: Optional status filter
            sort_options: Sorting options

        Returns:
            List[Evaluation]: List of user's evaluations
        """
        try:
            filters = {"created_by_id": user_id}
            if status:
                filters["status"] = status

            return await self.list_evaluations(
                skip=skip,
                limit=limit,
                filters=filters,
                sort_options=sort_options
            )

        except Exception as e:
            logger.error(f"Error listing user evaluations: {str(e)}", exc_info=True)
            return []

    async def delete_evaluation(self, evaluation_id: UUID, user_id: Optional[UUID] = None) -> bool:
        """
        Delete evaluation by ID with optional user ownership check.

        Args:
            evaluation_id: Evaluation ID
            user_id: Optional user ID for ownership verification

        Returns:
            bool: True if deleted, False if not found

        Raises:
            AuthorizationException: If user doesn't have permission
        """
        # REMOVED TRANSACTION to avoid conflict with FastAPI dependency
        try:
            # First, check if evaluation exists and user has permission
            evaluation = await self.evaluation_repo.get(evaluation_id)
            if not evaluation:
                logger.warning(f"Attempted to delete non-existent evaluation {evaluation_id}")
                return False

            # Check ownership if user_id is provided
            if user_id and evaluation.created_by_id and evaluation.created_by_id != user_id:
                logger.warning(f"Unauthorized deletion attempt on evaluation {evaluation_id} by user {user_id}")
                raise AuthorizationException(
                    detail="You don't have permission to delete this evaluation"
                )

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

        except AuthorizationException:
            # Re-raise authorization exceptions
            raise
        except Exception as e:
            logger.error(f"Error deleting evaluation {evaluation_id}: {str(e)}")
            return False

    @staticmethod
    async def _validate_metrics_for_dataset(
            metrics: Optional[List[str]],
            dataset: Dataset,
            evaluation_method: EvaluationMethod
    ) -> None:
        """
        Validate that the selected metrics are appropriate for the dataset type and evaluation method.

        Args:
            metrics: List of metrics to validate
            dataset: Dataset to validate against
            evaluation_method: Evaluation method (RAGAS or DeepEval)

        Raises:
            HTTPException: If metrics are invalid for the dataset type and method combination
        """
        if not metrics:
            return  # No metrics specified, will use defaults

        # Get allowed metrics based on evaluation method
        dataset_type = dataset.type

        if evaluation_method == EvaluationMethod.RAGAS:
            from backend.app.evaluation.metrics.ragas_metrics import DATASET_TYPE_METRICS
            allowed_metrics = DATASET_TYPE_METRICS.get(dataset_type, [])
            method_name = "RAGAS"

        elif evaluation_method == EvaluationMethod.DEEPEVAL:
            from backend.app.evaluation.metrics.deepeval_metrics import get_supported_metrics_for_dataset_type
            allowed_metrics = get_supported_metrics_for_dataset_type(dataset_type)
            method_name = "DeepEval"

        elif evaluation_method == EvaluationMethod.CUSTOM:
            # For custom methods, allow any metrics (no validation)
            logger.info(f"Custom evaluation method - skipping metric validation")
            return

        elif evaluation_method == EvaluationMethod.MANUAL:
            # For manual methods, allow any metrics (no validation)
            logger.info(f"Manual evaluation method - skipping metric validation")
            return

        else:
            logger.warning(f"Unknown evaluation method: {evaluation_method}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown evaluation method: {evaluation_method}"
            )

        if not dataset_type:
            logger.warning(f"Dataset {dataset.id} has no type specified")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Dataset has no type specified"
            )

        if not allowed_metrics:
            logger.warning(f"No {method_name} metrics defined for dataset type {dataset_type}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No {method_name} metrics are defined for dataset type {dataset_type}"
            )

        # Check if any specified metrics are not allowed
        invalid_metrics = [m for m in metrics if m not in allowed_metrics]

        if invalid_metrics:
            logger.warning(f"Invalid {method_name} metrics for dataset type {dataset_type}: {invalid_metrics}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Metrics {invalid_metrics} are not valid for {dataset_type} datasets using {method_name}. "
                       f"Valid {method_name} metrics are: {allowed_metrics}"
            )

        logger.info(f"Validated {len(metrics)} {method_name} metrics for dataset type {dataset_type}: {metrics}")

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
            logger.info(f"Created evaluation result {result.id} for evaluation {result_data.evaluation_id}")

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
            self, evaluation_id: UUID, skip: int = 0, limit: int = 100, user_id: Optional[UUID] = None
    ) -> List[EvaluationResult]:
        """
        Get results for an evaluation with pagination and optional user verification.

        Args:
            evaluation_id: Evaluation ID
            skip: Number of records to skip
            limit: Maximum number of records to return
            user_id: Optional user ID for ownership verification

        Returns:
            List[EvaluationResult]: List of evaluation results

        Raises:
            AuthorizationException: If user doesn't have permission
        """
        try:
            # Check ownership if user_id is provided
            if user_id:
                evaluation = await self.evaluation_repo.get(evaluation_id)
                if evaluation and evaluation.created_by_id and evaluation.created_by_id != user_id:
                    logger.warning(
                        f"Unauthorized access attempt to results for evaluation {evaluation_id} by user {user_id}")
                    raise AuthorizationException(
                        detail="You don't have permission to access these evaluation results"
                    )

            results = await self.result_repo.get_multi(
                skip=skip,
                limit=limit,
                filters={"evaluation_id": evaluation_id},
                load_relationships=["metric_scores"]
            )
            logger.debug(f"Retrieved {len(results)} results for evaluation {evaluation_id}")
            return results
        except AuthorizationException:
            # Re-raise authorization exceptions
            raise
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

    async def start_evaluation(self, evaluation_id: UUID, user_id: Optional[UUID] = None) -> Optional[Evaluation]:
        """
        Start an evaluation with optional user verification.

        Args:
            evaluation_id: Evaluation ID
            user_id: Optional user ID for ownership verification

        Returns:
            Optional[Evaluation]: Updated evaluation if found, None otherwise

        Raises:
            HTTPException: If evaluation cannot be started
            AuthorizationException: If user doesn't have permission
        """
        # REMOVED TRANSACTION to avoid conflict with FastAPI dependency
        try:
            evaluation = await self.evaluation_repo.get(evaluation_id)
            if not evaluation:
                logger.warning(f"Attempted to start non-existent evaluation {evaluation_id}")
                return None

            # Check ownership if user_id is provided
            if user_id and evaluation.created_by_id and evaluation.created_by_id != user_id:
                logger.warning(f"Unauthorized start attempt on evaluation {evaluation_id} by user {user_id}")
                raise AuthorizationException(
                    detail="You don't have permission to start this evaluation"
                )

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

        except (HTTPException, AuthorizationException):
            # Re-raise these exceptions
            raise
        except Exception as e:
            logger.error(f"Error starting evaluation {evaluation_id}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to start evaluation: {str(e)}"
            )

    async def queue_evaluation_job(self, evaluation_id: UUID, user_id: Optional[UUID] = None) -> None:
        """
        Queue an evaluation job for background processing with optional user verification.

        This will send a task to Celery or run directly in a background task.

        Args:
            evaluation_id: Evaluation ID
            user_id: Optional user ID for ownership verification

        Raises:
            HTTPException: If queueing fails
            AuthorizationException: If user doesn't have permission
        """
        try:
            # Get the evaluation to check if it can be queued
            evaluation = await self.evaluation_repo.get(evaluation_id)
            if not evaluation:
                logger.error(f"Evaluation {evaluation_id} not found")
                raise ValueError(f"Evaluation {evaluation_id} not found")

            # Check ownership if user_id is provided
            if user_id and evaluation.created_by_id and evaluation.created_by_id != user_id:
                logger.warning(f"Unauthorized queue attempt on evaluation {evaluation_id} by user {user_id}")
                raise AuthorizationException(
                    detail="You don't have permission to queue this evaluation"
                )

            if evaluation.status != EvaluationStatus.PENDING:
                logger.warning(f"Evaluation {evaluation_id} is in {evaluation.status} status, not PENDING")
                raise ValueError(f"Evaluation is already in {evaluation.status} status")

            # Update to running status - don't use a nested transaction here
            # Instead of calling start_evaluation (which might start another transaction)
            # update the status directly
            now = datetime.datetime.now()
            update_data = {
                "status": EvaluationStatus.RUNNING,
                "start_time": now
            }

            # Direct update without transaction
            await self.evaluation_repo.update(evaluation_id, update_data)
            logger.info(f"Started evaluation {evaluation_id} at {now}")

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

                # Create a separate, detached task instead to prevent session conflicts
                import asyncio
                asyncio.create_task(_run_evaluation_as_separate_task(str(evaluation_id)))
                logger.info(f"Started evaluation {evaluation_id} as separate task")

        except ValueError as e:
            logger.warning(f"Cannot queue evaluation job: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        except AuthorizationException:
            # Re-raise authorization exceptions
            raise
        except Exception as e:
            logger.error(f"Error queueing evaluation job: {str(e)}", exc_info=True)
            # Set evaluation to failed status
            try:
                # Update status directly instead of calling complete_evaluation
                # to avoid another transaction
                failed_update = {
                    "status": EvaluationStatus.FAILED,
                    "end_time": datetime.datetime.now()
                }
                await self.evaluation_repo.update(evaluation_id, failed_update)
                logger.info(f"Marked evaluation {evaluation_id} as failed")
            except Exception as complete_error:
                logger.error(f"Failed to mark evaluation as failed: {str(complete_error)}")

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to queue evaluation job: {str(e)}"
            )


_progress_cache = {}
