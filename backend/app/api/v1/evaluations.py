import datetime
import logging
from typing import Dict, List, Optional, Union, Any, Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Body, Query, Path, BackgroundTasks
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.dependencies.auth import get_required_current_user, get_jwt_token
from backend.app.api.dependencies.rate_limiter import rate_limit
from backend.app.api.middleware.jwt_validator import UserContext
from backend.app.core.exceptions import NotFoundException, ValidationException, InvalidStateException
from backend.app.db.models.orm import EvaluationStatus, EvaluationMethod, EvaluationResult, DatasetType, Evaluation
from backend.app.db.schema.evaluation_schema import (
    EvaluationCreate, EvaluationDetailResponse,
    EvaluationResponse, EvaluationUpdate
)
from backend.app.db.session import get_db
from backend.app.evaluation.metrics.ragas_metrics import DATASET_TYPE_METRICS
from backend.app.services.dataset_service import DatasetService
from backend.app.services.evaluation_service import EvaluationService, _run_evaluation_as_separate_task

# Configure logger
logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/", response_model=EvaluationResponse)
async def create_evaluation(
        evaluation_data: EvaluationCreate,
        db: AsyncSession = Depends(get_db),
        current_user: UserContext = Depends(get_required_current_user),
        _: None = Depends(rate_limit(max_requests=10, period_seconds=60))
):
    """
    Create a new evaluation.

    This endpoint creates a new evaluation configuration with the specified parameters.

    - **evaluation_data**: Required evaluation configuration data

    Returns the created evaluation object with an ID that can be used for future operations.
    """
    logger.info(f"Creating new evaluation with dataset_id={evaluation_data.dataset_id}, "
                f"agent_id={evaluation_data.agent_id}")

    # Add the user ID to the evaluation data
    if not evaluation_data.created_by_id and current_user.db_user:
        evaluation_data.created_by_id = current_user.db_user.id

    evaluation_service = EvaluationService(db)
    try:
        evaluation = await evaluation_service.create_evaluation(evaluation_data)
        logger.info(evaluation)
        logger.info(f"Successfully created evaluation id={evaluation.id}")
        return evaluation
    except Exception as e:
        logger.error(f"Failed to create evaluation: {str(e)}", exc_info=True)
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create evaluation: {str(e)}"
        )


@router.get("/", response_model=Dict[str, Any])
async def list_evaluations(
        skip: Annotated[int, Query(ge=0, description="Number of records to skip")] = 0,
        limit: Annotated[int, Query(ge=1, le=100, description="Maximum number of records to return")] = 10,
        status: Annotated[Optional[EvaluationStatus], Query(description="Filter by evaluation status")] = None,
        agent_id: Annotated[Optional[UUID], Query(description="Filter by agent ID")] = None,
        dataset_id: Annotated[Optional[UUID], Query(description="Filter by dataset ID")] = None,
        name: Annotated[
            Optional[str], Query(description="Filter by evaluation name (case-insensitive, partial match)")] = None,
        method: Annotated[Optional[EvaluationMethod], Query(description="Filter by evaluation method")] = None,
        sort_by: Annotated[Optional[str], Query(description="Field to sort by")] = "created_at",
        sort_dir: Annotated[Optional[str], Query(description="Sort direction (asc or desc)")] = "desc",
        db: AsyncSession = Depends(get_db),
        current_user: UserContext = Depends(get_required_current_user),
        _: None = Depends(rate_limit(max_requests=50, period_seconds=60))
):
    """
    List evaluations with optional filtering, sorting and pagination.

    This endpoint returns both the evaluations array and pagination information.

    - **skip**: Number of records to skip (for pagination)
    - **limit**: Maximum number of records to return
    - **status**: Optional filter by evaluation status
    - **agent_id**: Optional filter by agent ID
    - **dataset_id**: Optional filter by dataset ID
    - **name**: Optional filter by evaluation name (case-insensitive, supports partial matching)
    - **method**: Optional filter by evaluation method
    - **sort_by**: Field to sort results by (default: created_at)
    - **sort_dir**: Sort direction, either "asc" or "desc" (default: desc)

    Returns a dictionary containing the list of evaluations and pagination information.
    """
    filters = {}

    # Add filters if provided
    if status:
        filters["status"] = status
    if agent_id:
        filters["agent_id"] = agent_id
    if dataset_id:
        filters["dataset_id"] = dataset_id
    if name:
        filters["name"] = name
    if method:
        filters["method"] = method

    # Add user ID to filter only user's evaluations
    if current_user.db_user:
        filters["created_by_id"] = current_user.db_user.id

    # Validate sort_by parameter
    valid_sort_fields = ["created_at", "updated_at", "name", "status", "method", "start_time", "end_time"]
    if sort_by not in valid_sort_fields:
        logger.warning(f"Invalid sort field: {sort_by}, defaulting to created_at")
        sort_by = "created_at"

    # Validate sort_dir parameter
    if sort_dir.lower() not in ["asc", "desc"]:
        logger.warning(f"Invalid sort direction: {sort_dir}, defaulting to desc")
        sort_dir = "desc"

    # Add sorting instructions
    sort_options = {
        "sort_by": sort_by,
        "sort_dir": sort_dir.lower()
    }

    logger.info(f"Listing evaluations with filters={filters}, skip={skip}, limit={limit}, sort={sort_options}")

    evaluation_service = EvaluationService(db)
    try:
        # Get total count first for pagination
        total_count = await evaluation_service.count_evaluations(filters)

        # Then get the actual page of results
        evaluations = await evaluation_service.list_evaluations(skip, limit, filters, sort_options)

        logger.debug(f"Retrieved {len(evaluations)} evaluations from total of {total_count}")

        # Convert SQLAlchemy model instances to dictionaries
        evaluation_dicts = []
        for evaluation in evaluations:
            # Use the to_dict method from ModelMixin
            eval_dict = evaluation.to_dict()

            # Add any relationships that need to be included
            if hasattr(evaluation, 'agent') and evaluation.agent:
                eval_dict['agent'] = evaluation.agent.to_dict() if evaluation.agent else None

            if hasattr(evaluation, 'dataset') and evaluation.dataset:
                eval_dict['dataset'] = evaluation.dataset.to_dict() if evaluation.dataset else None

            if hasattr(evaluation, 'prompt') and evaluation.prompt:
                eval_dict['prompt'] = evaluation.prompt.to_dict() if evaluation.prompt else None

            evaluation_dicts.append(eval_dict)

        # Return both results and total count
        return {
            "items": evaluation_dicts,
            "total": total_count,
            "page": (skip // limit) + 1,
            "page_size": limit,
            "has_next": (skip + limit) < total_count,
            "has_previous": skip > 0
        }
    except Exception as e:
        logger.error(f"Error listing evaluations: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing evaluations: {str(e)}"
        )


@router.post("/search", response_model=Dict[str, Any])
async def search_evaluations(
        query: Optional[str] = Body(None, description="Search query for name or description"),
        filters: Optional[Dict[str, Any]] = Body(None, description="Additional filters"),
        skip: int = Body(0, ge=0, description="Number of records to skip"),
        limit: int = Body(100, ge=1, le=1000, description="Maximum number of records to return"),
        sort_by: str = Body("created_at", description="Field to sort by"),
        sort_dir: str = Body("desc", description="Sort direction (asc or desc)"),
        db: AsyncSession = Depends(get_db),
        current_user: UserContext = Depends(get_required_current_user)
):
    """
    Advanced search for evaluations across multiple fields.

    Supports text search across name and description fields,
    as well as additional filters for exact matches.

    Args:
        query: Search query text for name and description
        filters: Additional filters (exact match)
        skip: Number of records to skip
        limit: Maximum number of records to return
        sort_by: Field to sort by
        sort_dir: Sort direction
        db: Database session
        current_user: The authenticated user

    Returns:
        Dict containing search results and pagination info
    """
    logger.info(f"Advanced search for evaluations with query: '{query}' and filters: {filters}")

    # Initialize filters if not provided
    if filters is None:
        filters = {}

    # Add user ID to filter only user's evaluations
    if current_user.db_user:
        filters["created_by_id"] = current_user.db_user.id

    # Create repository directly since we need the search methods
    from backend.app.db.repositories.evaluation_repository import EvaluationRepository
    eval_repo = EvaluationRepository(db)

    # Get total count
    total_count = await eval_repo.count_search_evaluations(query, filters)

    # Get evaluations
    evaluations = await eval_repo.search_evaluations(
        query_text=query,
        filters=filters,
        skip=skip,
        limit=limit,
        sort_by=sort_by,
        sort_dir=sort_dir
    )

    # Convert to dictionaries
    evaluation_dicts = []
    for evaluation in evaluations:
        eval_dict = evaluation.to_dict()

        # Add relationships
        if hasattr(evaluation, 'agent') and evaluation.agent:
            eval_dict['agent'] = evaluation.agent.to_dict()
        if hasattr(evaluation, 'dataset') and evaluation.dataset:
            eval_dict['dataset'] = evaluation.dataset.to_dict()
        if hasattr(evaluation, 'prompt') and evaluation.prompt:
            eval_dict['prompt'] = evaluation.prompt.to_dict()

        evaluation_dicts.append(eval_dict)

    return {
        "items": evaluation_dicts,
        "total": total_count,
        "page": (skip // limit) + 1,
        "page_size": limit,
        "has_next": (skip + limit) < total_count,
        "has_previous": skip > 0
    }


@router.get("/{evaluation_id}", response_model=EvaluationDetailResponse)
async def get_evaluation(
        evaluation_id: Annotated[UUID, Path(description="The ID of the evaluation to retrieve")],
        db: AsyncSession = Depends(get_db),
        current_user: UserContext = Depends(get_required_current_user)
):
    """
    Get evaluation by ID with all related details.

    This endpoint retrieves comprehensive information about an evaluation, including:
    - Basic evaluation metadata
    - Configuration details
    - All results with their metrics
    - Pass/fail status for each result

    - **evaluation_id**: The unique identifier of the evaluation

    Returns the complete evaluation object with nested results and metrics.
    """
    try:
        # Create evaluation service
        evaluation_service = EvaluationService(db)

        # Get the evaluation with all relationships in one query
        evaluation, result_responses = await evaluation_service.get_evaluation_with_relationships(evaluation_id)

        if not evaluation:
            raise NotFoundException(resource="Evaluation", resource_id=str(evaluation_id))

        # Check if the user has permission to access this evaluation
        if evaluation.created_by_id and evaluation.created_by_id != current_user.db_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to access this evaluation"
            )

        # Create response dictionary with all needed fields
        response_dict = {
            "id": evaluation.id,
            "name": evaluation.name,
            "description": evaluation.description,
            "method": evaluation.method,
            "status": evaluation.status,
            "config": evaluation.config,
            "metrics": evaluation.metrics,
            "experiment_id": evaluation.experiment_id,
            "start_time": evaluation.start_time,
            "end_time": evaluation.end_time,
            "agent_id": evaluation.agent_id,
            "dataset_id": evaluation.dataset_id,
            "prompt_id": evaluation.prompt_id,
            "created_at": evaluation.created_at,
            "updated_at": evaluation.updated_at,
            "pass_threshold": evaluation.pass_threshold,
            "results": result_responses
        }

        # Calculate summary statistics if there are results
        if result_responses:
            # Calculate overall pass rate
            pass_count = sum(1 for r in result_responses if r.get("passed", False))
            total_count = len(result_responses)
            pass_rate = (pass_count / total_count) * 100 if total_count > 0 else 0

            # Calculate metric averages
            metric_averages = {}
            for result in result_responses:
                for metric in result.get("metric_scores", []):
                    metric_name = metric.get("name")
                    metric_value = metric.get("value")

                    if metric_name and metric_value is not None:
                        if metric_name not in metric_averages:
                            metric_averages[metric_name] = []
                        metric_averages[metric_name].append(metric_value)

            # Calculate the averages
            metric_summary = {}
            for metric_name, values in metric_averages.items():
                if values:
                    metric_summary[metric_name] = {
                        "average": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                        "count": len(values)
                    }

            # Add summary to response
            response_dict["summary"] = {
                "overall_score": sum(r.get("overall_score", 0) for r in result_responses) / len(
                    result_responses) if result_responses else 0,
                "pass_rate": pass_rate,
                "pass_count": pass_count,
                "total_count": total_count,
                "pass_threshold": evaluation.pass_threshold,
                "metrics": metric_summary
            }

        # Return the response data directly - FastAPI will handle conversion
        return response_dict

    except NotFoundException:
        raise
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error retrieving evaluation details: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving evaluation details: {str(e)}"
        )


@router.put("/{evaluation_id}", response_model=EvaluationResponse)
async def update_evaluation(
        evaluation_id: Annotated[UUID, Path(description="The ID of the evaluation to update")],
        evaluation_data: EvaluationUpdate,
        db: AsyncSession = Depends(get_db),
        current_user: UserContext = Depends(get_required_current_user)
):
    """
    Update evaluation by ID.

    This endpoint allows updating various properties of an existing evaluation.
    Note that some properties cannot be changed once an evaluation has started.

    - **evaluation_id**: The unique identifier of the evaluation to update
    - **evaluation_data**: The evaluation properties to update

    Returns the updated evaluation object.
    """
    logger.info(f"Updating evaluation id={evaluation_id}")

    evaluation_service = EvaluationService(db)

    try:
        # Get the evaluation to check permissions
        evaluation = await evaluation_service.get_evaluation(evaluation_id)
        if not evaluation:
            raise NotFoundException(resource="Evaluation", resource_id=str(evaluation_id))

        # Check if the user has permission to test this evaluation
        if evaluation.created_by_id and evaluation.created_by_id != current_user.db_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to test this evaluation"
            )

        # Check if the user has permission to update this evaluation
        if evaluation.created_by_id and evaluation.created_by_id != current_user.db_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to update this evaluation"
            )

        # Check if we're trying to update fields that shouldn't be changed after certain states
        if evaluation.status not in [EvaluationStatus.PENDING, EvaluationStatus.FAILED]:
            protected_fields = ["method", "metrics", "config"]
            update_dict = evaluation_data.model_dump(exclude_unset=True)

            if any(field in update_dict for field in protected_fields):
                logger.warning(
                    f"Attempted to update protected fields of evaluation {evaluation_id} in {evaluation.status} state")
                raise InvalidStateException(
                    resource="Evaluation",
                    current_state=evaluation.status.value,
                    operation="update protected fields"
                )

        # Update the evaluation
        logger.debug(f"Updating evaluation id={evaluation_id} with data: {evaluation_data}")
        updated_evaluation = await evaluation_service.update_evaluation(
            evaluation_id, evaluation_data
        )

        if not updated_evaluation:
            logger.error(f"Failed to update evaluation id={evaluation_id}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update evaluation"
            )

        logger.info(f"Successfully updated evaluation id={evaluation_id}")
        return updated_evaluation
    except (NotFoundException, InvalidStateException):
        raise
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating evaluation id={evaluation_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating evaluation: {str(e)}"
        )


@router.delete("/{evaluation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_evaluation(
        evaluation_id: Annotated[UUID, Path(description="The ID of the evaluation to delete")],
        db: AsyncSession = Depends(get_db),
        current_user: UserContext = Depends(get_required_current_user)
):
    """
    Delete evaluation by ID.

    This endpoint completely removes an evaluation and all its associated data,
    including results and metric scores. This operation cannot be undone.

    - **evaluation_id**: The unique identifier of the evaluation to delete

    Returns no content on success (HTTP 204).
    """
    logger.info(f"Deleting evaluation id={evaluation_id}")

    evaluation_service = EvaluationService(db)

    try:
        # Get the evaluation to check permissions
        evaluation = await evaluation_service.get_evaluation(evaluation_id)
        if not evaluation:
            raise NotFoundException(resource="Evaluation", resource_id=str(evaluation_id))

        # Check if the user has permission to delete this evaluation
        if evaluation.created_by_id and evaluation.created_by_id != current_user.db_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to delete this evaluation"
            )

        # Check if evaluation is running before deleting
        if evaluation.status == EvaluationStatus.RUNNING:
            logger.warning(f"Attempted to delete running evaluation {evaluation_id}")
            raise InvalidStateException(
                resource="Evaluation",
                current_state=evaluation.status.value,
                operation="delete",
                detail="Cannot delete an evaluation while it's running. Cancel it first."
            )

        # Delete the evaluation
        logger.debug(f"Attempting to delete evaluation id={evaluation_id}")
        success = await evaluation_service.delete_evaluation(evaluation_id)
        if not success:
            logger.error(f"Failed to delete evaluation id={evaluation_id}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete evaluation"
            )

        logger.info(f"Successfully deleted evaluation id={evaluation_id}")
    except (NotFoundException, InvalidStateException):
        raise
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting evaluation id={evaluation_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting evaluation: {str(e)}"
        )


@router.post("/{evaluation_id}/start", response_model=EvaluationResponse)
async def start_evaluation(
        evaluation_id: Annotated[UUID, Path(description="The ID of the evaluation to start")],
        background_tasks: BackgroundTasks,
        db: AsyncSession = Depends(get_db),
        current_user: UserContext = Depends(get_required_current_user),
        jwt_token: Optional[str] = Depends(get_jwt_token),  # Get JWT token from request
        _: None = Depends(rate_limit(max_requests=5, period_seconds=60))
):
    logger.info(f"Starting evaluation id={evaluation_id}")

    evaluation_service = EvaluationService(db)

    try:
        # Get the evaluation to check permissions
        evaluation = await evaluation_service.get_evaluation(evaluation_id)
        if not evaluation:
            raise NotFoundException(
                resource="Evaluation",
                resource_id=str(evaluation_id),
                detail=f"Evaluation with ID {evaluation_id} not found. Please check the ID and try again."
            )

        # Check if the user has permission to start this evaluation
        if evaluation.created_by_id and evaluation.created_by_id != current_user.db_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to start this evaluation"
            )

        # Check if evaluation can be started
        if evaluation.status != EvaluationStatus.PENDING:
            status_messages = {
                EvaluationStatus.RUNNING: "already running",
                EvaluationStatus.COMPLETED: "already completed",
                EvaluationStatus.FAILED: "failed previously",
                EvaluationStatus.CANCELLED: "cancelled"
            }
            message = status_messages.get(evaluation.status, f"in {evaluation.status} status")

            raise InvalidStateException(
                resource="Evaluation",
                current_state=evaluation.status.value,
                operation="start",
                detail=f"Cannot start evaluation because it is {message}. Create a new evaluation or retry if failed."
            )

        # Update status directly rather than calling start_evaluation
        # which might create a nested transaction
        now = datetime.datetime.now()
        update_data = {
            "status": EvaluationStatus.RUNNING,
            "start_time": now
        }

        # Direct update with the repository
        await evaluation_service.evaluation_repo.update(evaluation_id, update_data)
        logger.info(f"Started evaluation {evaluation_id} at {now}")

        # Queue the evaluation job
        # This is another place where we'd call a separate function, which
        # might be trying to start its own transaction
        from backend.app.workers.tasks import run_evaluation_task

        try:
            # Queue in Celery when available - pass JWT token
            run_evaluation_task.delay(str(evaluation_id), jwt_token)
            logger.info(f"Queued evaluation job {evaluation_id} to Celery with JWT token")
        except Exception as e:
            # Fallback to separate task if Celery not available
            logger.warning(f"Failed to queue to Celery: {e}. Running as separate task.")
            background_tasks.add_task(
                _run_evaluation_as_separate_task,
                str(evaluation_id),
                jwt_token  # Pass the JWT token to the background task
            )
            logger.info(f"Added evaluation {evaluation_id} as background task with JWT token")

        # Get updated evaluation
        updated_evaluation = await evaluation_service.get_evaluation(evaluation_id)
        return updated_evaluation

    except (NotFoundException, InvalidStateException):
        raise
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting evaluation id={evaluation_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error starting evaluation: {str(e)}"
        )


@router.get("/{evaluation_id}/progress", response_model=Dict)
async def get_evaluation_progress(
        evaluation_id: Annotated[UUID, Path(description="The ID of the evaluation to check progress")],
        db: AsyncSession = Depends(get_db),
        current_user: UserContext = Depends(get_required_current_user)
):
    """
    Get the progress of an evaluation.

    This endpoint returns detailed information about the evaluation progress,
    including the current status, number of processed items, percentage complete,
    and estimated time remaining.

    - **evaluation_id**: The unique identifier of the evaluation

    Returns a dictionary with detailed progress information.
    """
    logger.info(f"Getting progress for evaluation id={evaluation_id}")

    evaluation_service = EvaluationService(db)

    try:
        # First get the evaluation to check permissions
        evaluation = await evaluation_service.get_evaluation(evaluation_id)
        if not evaluation:
            raise NotFoundException(
                resource="Evaluation",
                resource_id=str(evaluation_id)
            )

        # Check if the user has permission to view this evaluation's progress
        if evaluation.created_by_id and evaluation.created_by_id != current_user.db_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to view this evaluation's progress"
            )

        progress = await evaluation_service.get_evaluation_progress(evaluation_id)
        return progress
    except NotFoundException:
        raise
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting evaluation progress: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting evaluation progress: {str(e)}"
        )


@router.post("/{evaluation_id}/cancel", response_model=EvaluationResponse)
async def cancel_evaluation(
        evaluation_id: Annotated[UUID, Path(description="The ID of the evaluation to cancel")],
        db: AsyncSession = Depends(get_db),
        current_user: UserContext = Depends(get_required_current_user)
):
    """
    Cancel a running evaluation.

    This endpoint allows you to stop an evaluation that is currently running.
    The evaluation's status will be changed to CANCELLED.

    - **evaluation_id**: The unique identifier of the evaluation to cancel

    Returns the updated evaluation object with CANCELLED status.
    """
    logger.info(f"Cancelling evaluation id={evaluation_id}")

    evaluation_service = EvaluationService(db)

    try:
        # Get the evaluation to check permissions
        evaluation = await evaluation_service.get_evaluation(evaluation_id)
        if not evaluation:
            raise NotFoundException(resource="Evaluation", resource_id=str(evaluation_id))

        # Check if the user has permission to cancel this evaluation
        if evaluation.created_by_id and evaluation.created_by_id != current_user.db_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to cancel this evaluation"
            )

        # Check if the evaluation is running
        if evaluation.status != EvaluationStatus.RUNNING:
            raise InvalidStateException(
                resource="Evaluation",
                current_state=evaluation.status.value,
                operation="cancel",
                detail=f"Evaluation is not in RUNNING status (current status: {evaluation.status})"
            )

        # Cancel the evaluation
        logger.debug(f"Updating evaluation id={evaluation_id} status to CANCELLED")
        update_data = EvaluationUpdate(status=EvaluationStatus.CANCELLED)
        updated_evaluation = await evaluation_service.update_evaluation(
            evaluation_id, update_data
        )

        logger.info(f"Successfully cancelled evaluation id={evaluation_id}")
        return updated_evaluation
    except (NotFoundException, InvalidStateException):
        raise
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling evaluation id={evaluation_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error cancelling evaluation: {str(e)}"
        )


@router.get("/{evaluation_id}/results", response_model=Dict[str, Any])
async def get_evaluation_results(
        evaluation_id: Annotated[UUID, Path(description="The ID of the evaluation to get results for")],
        skip: Annotated[int, Query(ge=0, description="Number of records to skip")] = 0,
        limit: Annotated[int, Query(ge=1, le=100, description="Maximum number of records to return")] = 100,
        db: AsyncSession = Depends(get_db),
        current_user: UserContext = Depends(get_required_current_user)
):
    """
    Get results for an evaluation with pagination.

    This endpoint returns the detailed results for an evaluation, including all metric scores
    and pass/fail status for each result.

    - **evaluation_id**: The unique identifier of the evaluation
    - **skip**: Number of records to skip (for pagination)
    - **limit**: Maximum number of records to return

    Returns a dictionary containing the list of results and the total count.
    """
    logger.info(f"Requesting results for evaluation id={evaluation_id}")

    evaluation_service = EvaluationService(db)

    try:
        # Get the evaluation to check permissions
        evaluation = await evaluation_service.get_evaluation(evaluation_id)
        if not evaluation:
            raise NotFoundException(resource="Evaluation", resource_id=str(evaluation_id))

        # Check if the user has permission to view this evaluation's results
        if evaluation.created_by_id and evaluation.created_by_id != current_user.db_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to view this evaluation's results"
            )

        # Get evaluation results
        logger.debug(f"Fetching results for evaluation id={evaluation_id}")
        results = await evaluation_service.get_evaluation_results(evaluation_id, skip, limit)

        # Get total count for pagination
        total_query = select(func.count()).select_from(EvaluationResult).where(
            EvaluationResult.evaluation_id == evaluation_id
        )
        total_result = await db.execute(total_query)
        total_count = total_result.scalar_one_or_none() or 0

        # Process results for response
        processed_results = []
        for result in results:
            result_dict = result.to_dict()

            # Get metric scores
            logger.debug(f"Fetching metric scores for result id={result.id}")
            metric_scores = await evaluation_service.get_metric_scores(result.id)
            result_dict["metric_scores"] = [score.to_dict() for score in metric_scores]

            # Include pass/fail status
            if hasattr(result, 'passed'):
                result_dict["passed"] = result.passed
                result_dict["pass_threshold"] = result.pass_threshold or evaluation.pass_threshold or 0.7

            processed_results.append(result_dict)

        # Calculate summary statistics
        pass_count = sum(1 for r in processed_results if r.get("passed", False))
        pass_rate = (pass_count / len(processed_results)) * 100 if processed_results else 0

        logger.info(f"Successfully retrieved {len(processed_results)} results for evaluation id={evaluation_id}")
        return {
            "items": processed_results,
            "total": total_count,
            "page": (skip // limit) + 1,
            "page_size": limit,
            "has_next": (skip + limit) < total_count,
            "has_previous": skip > 0,
            "summary": {
                "pass_rate": pass_rate,
                "pass_count": pass_count,
                "total_evaluated": len(processed_results),
                "pass_threshold": evaluation.pass_threshold or 0.7
            }
        }
    except NotFoundException:
        raise
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving results for evaluation id={evaluation_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving results: {str(e)}"
        )


@router.get("/metrics/{dataset_type}", response_model=Dict[str, Union[str, List[str]]])
async def get_supported_metrics(
        dataset_type: Annotated[str, Path(description="The dataset type to get supported metrics for")]
):
    """
    Get supported metrics for a specific dataset type.

    This endpoint returns the list of metrics that can be calculated for a given dataset type.

    - **dataset_type**: The type of dataset (e.g., user_query, context, question_answer, etc.)

    Returns a dictionary with the dataset type and list of supported metrics.
    """
    try:
        if dataset_type not in DATASET_TYPE_METRICS:
            raise ValidationException(
                detail=f"Invalid dataset type: {dataset_type}. Valid types are: {list(DATASET_TYPE_METRICS.keys())}"
            )

        return {
            "dataset_type": dataset_type,
            "supported_metrics": DATASET_TYPE_METRICS[dataset_type]
        }
    except ValidationException:
        raise
    except Exception as e:
        logger.error(f"Error getting supported metrics: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting supported metrics: {str(e)}"
        )


@router.post("/{evaluation_id}/test", response_model=Dict)
async def test_evaluation(
        evaluation_id: Annotated[UUID, Path(description="The ID of the evaluation to test")],
        test_data: Dict = Body(..., example={
            "query": "What is the capital of France?",
            "context": "France is a country in Western Europe. Its capital is Paris.",
            "answer": "The capital of France is Paris.",
            "ground_truth": "Paris"
        }),
        db: AsyncSession = Depends(get_db),
        _: None = Depends(rate_limit(max_requests=20, period_seconds=60))
):
    """
    Test an evaluation with sample data without creating results.

    This is useful for validating configurations and testing metrics
    before running a full evaluation.

    The request body should contain test data in the format:
    ```
    {
        "query": "Sample query",
        "context": "Sample context",
        "answer": "Sample answer",
        "ground_truth": "Optional ground truth"
    }
    ```

    - **evaluation_id**: The unique identifier of the evaluation to test
    - **test_data**: Sample data for testing the metrics

    Returns the calculated metrics and overall score for the test data.
    """
    logger.info(f"Testing evaluation id={evaluation_id}")

    evaluation_service = EvaluationService(db)

    try:
        # Get the evaluation to check permissions
        evaluation = await evaluation_service.get_evaluation(evaluation_id)
        if not evaluation:
            raise NotFoundException(resource="Evaluation", resource_id=str(evaluation_id))

        # Validate minimum required test data
        required_fields = ["query", "context", "answer"]
        missing_fields = [field for field in required_fields if field not in test_data]

        if missing_fields:
            logger.warning(f"Missing required fields {missing_fields} in test data for evaluation id={evaluation_id}")
            raise ValidationException(
                detail=f"Missing required fields: {', '.join(missing_fields)}"
            )

        # Get evaluation method
        logger.debug(f"Getting evaluation method handler for method={evaluation.method}")
        method_handler = await evaluation_service.get_evaluation_method_handler(evaluation.method)

        # Calculate metrics
        logger.debug(f"Calculating metrics for test data on evaluation id={evaluation_id}")
        metrics = await method_handler.calculate_metrics(
            input_data={
                "query": test_data.get("query", ""),
                "context": test_data.get("context", ""),
                "ground_truth": test_data.get("ground_truth", "")
            },
            output_data={"answer": test_data.get("answer", "")},
            config=evaluation.config or {}
        )

        # Calculate overall score
        overall_score = sum(metrics.values()) / len(metrics) if metrics else 0.0

        logger.info(f"Successfully tested evaluation id={evaluation_id} with overall score={overall_score:.2f}")
        return {
            "evaluation_id": str(evaluation_id),
            "overall_score": overall_score,
            "metrics": metrics,
            "config": evaluation.config
        }
    except (NotFoundException, ValidationException):
        raise
    except Exception as e:
        logger.error(f"Error testing evaluation id={evaluation_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error testing evaluation: {str(e)}"
        )


from backend.app.evaluation.metrics.deepeval_metrics import (
    DEEPEVAL_DATASET_TYPE_METRICS, get_supported_metrics_for_dataset_type,
    get_recommended_metrics
)
from backend.app.evaluation.adapters.dataset_adapter import DatasetAdapter


@router.post("/{evaluation_id}/validate-deepeval")
async def validate_dataset_for_deepeval(
        evaluation_id: Annotated[UUID, Path(description="The ID of the evaluation to validate")],
        validation_request: Dict[str, Any] = Body(..., example={
            "metrics": ["answer_relevancy", "faithfulness", "hallucination"]
        }),
        db: AsyncSession = Depends(get_db),
        current_user: UserContext = Depends(get_required_current_user)
):
    """
    Validate that dataset is compatible with selected DeepEval metrics.

    This endpoint checks if the dataset associated with an evaluation
    contains the necessary fields for the selected DeepEval metrics.

    Args:
        evaluation_id: The ID of the evaluation to validate
        validation_request: Dictionary containing metrics to validate

    Returns:
        Dict containing validation results and recommendations
        :param validation_request:
        :param evaluation_id:
        :param current_user:
        :param db:
    """
    try:
        evaluation_service = EvaluationService(db)

        # Get evaluation and check permissions
        evaluation = await evaluation_service.get_evaluation(evaluation_id)
        if not evaluation:
            raise NotFoundException(resource="Evaluation", resource_id=str(evaluation_id))

        if evaluation.created_by_id and evaluation.created_by_id != current_user.db_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to access this evaluation"
            )

        # Get dataset
        dataset_service = DatasetService(db)
        dataset = await dataset_service.get_dataset(evaluation.dataset_id)

        # Load dataset content for analysis
        from backend.app.services.storage import get_storage_service
        storage_service = get_storage_service()
        file_content = await storage_service.read_file(dataset.file_path)

        # Parse dataset content
        import json
        if dataset.file_path.endswith('.json'):
            dataset_content = json.loads(file_content)
        else:  # CSV
            import pandas as pd
            import io
            df = pd.read_csv(io.StringIO(file_content))
            dataset_content = df.to_dict('records')

        # Validate using dataset adapter
        dataset_adapter = DatasetAdapter()
        validation_results = dataset_adapter.validate_dataset_for_deepeval(
            dataset_content,
            validation_request.get("metrics", [])
        )

        # Add dataset-specific recommendations
        supported_metrics = get_supported_metrics_for_dataset_type(dataset.type)
        recommended_metrics = get_recommended_metrics(dataset.type)

        return {
            "evaluation_id": str(evaluation_id),
            "dataset_id": str(dataset.id),
            "dataset_type": dataset.type.value,
            "validation": validation_results,
            "supported_metrics": supported_metrics,
            "recommended_metrics": recommended_metrics,
            "deepeval_compatibility": {
                "available_metrics": list(DEEPEVAL_DATASET_TYPE_METRICS.get(dataset.type, [])),
                "metric_categories": {
                    "relevance": ["answer_relevancy"],
                    "groundedness": ["faithfulness"],
                    "safety": ["hallucination", "toxicity", "bias"],
                    "quality": ["g_eval_coherence", "g_eval_correctness"]
                }
            }
        }

    except Exception as e:
        logger.error(f"Error validating dataset for DeepEval: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error validating dataset: {str(e)}"
        )


@router.get("/datasets/{dataset_id}/deepeval-preview")
async def preview_dataset_for_deepeval(
        dataset_id: Annotated[UUID, Path(description="The ID of the dataset to preview")],
        limit: Annotated[int, Query(ge=1, le=20, description="Number of items to preview")] = 5,
        db: AsyncSession = Depends(get_db),
        current_user: UserContext = Depends(get_required_current_user)
):
    """
    Preview how your dataset will be converted for DeepEval.

    This endpoint shows how the first few items of your dataset
    will be converted to DeepEval TestCase format.

    Args:
        dataset_id: The ID of the dataset to preview
        limit: Maximum number of items to preview (1-20)

    Returns:
        Dict containing preview of dataset conversion
        :param current_user:
        :param limit:
        :param dataset_id:
        :param db:
    """
    try:
        dataset_service = DatasetService(db)

        # Get dataset with access control
        dataset = await dataset_service.get_accessible_dataset(dataset_id, current_user.db_user.id)

        # Load dataset content
        from backend.app.services.storage import get_storage_service
        storage_service = get_storage_service()
        file_content = await storage_service.read_file(dataset.file_path)

        # Parse dataset content
        import json
        if dataset.file_path.endswith('.json'):
            dataset_content = json.loads(file_content)
        else:  # CSV
            import pandas as pd
            import io
            df = pd.read_csv(io.StringIO(file_content))
            dataset_content = df.to_dict('records')

        # Limit items for preview
        preview_content = dataset_content[:limit]

        # Convert using dataset adapter
        dataset_adapter = DatasetAdapter()
        deepeval_dataset = await dataset_adapter.convert_to_deepeval_dataset(dataset, preview_content)

        # Create preview response
        preview_cases = []
        for i, test_case in enumerate(deepeval_dataset.test_cases):
            preview_cases.append({
                "index": i,
                "input": test_case.input,
                "expected_output": test_case.expected_output,
                "context": test_case.context,
                "has_context": bool(test_case.context),
                "has_expected_output": bool(test_case.expected_output),
                "additional_metadata": getattr(test_case, 'additional_metadata', {})
            })

        return {
            "dataset_id": str(dataset_id),
            "dataset_type": dataset.type.value,
            "total_items_in_dataset": len(dataset_content),
            "preview_items": len(preview_cases),
            "preview": preview_cases,
            "deepeval_compatibility": {
                "supported_metrics": get_supported_metrics_for_dataset_type(dataset.type),
                "recommended_metrics": get_recommended_metrics(dataset.type),
                "conversion_summary": {
                    "items_with_input": sum(1 for case in preview_cases if case["input"]),
                    "items_with_expected_output": sum(1 for case in preview_cases if case["has_expected_output"]),
                    "items_with_context": sum(1 for case in preview_cases if case["has_context"])
                }
            }
        }

    except Exception as e:
        logger.error(f"Error previewing dataset for DeepEval: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error previewing dataset: {str(e)}"
        )


@router.post("/prompts/{prompt_id}/test-with-deepeval")
async def test_prompt_with_deepeval_metrics(
        prompt_id: Annotated[UUID, Path(description="The ID of the prompt to test")],
        test_request: Dict[str, Any] = Body(..., example={
            "agent_id": "123e4567-e89b-12d3-a456-426614174000",
            "sample_input": "What is the capital of France?",
            "expected_output": "Paris",
            "context": ["France is a country in Western Europe.", "Paris is the largest city in France."],
            "metrics": [
                {"name": "answer_relevancy", "threshold": 0.7},
                {"name": "faithfulness", "threshold": 0.8}
            ]
        }),
        db: AsyncSession = Depends(get_db),
        current_user: UserContext = Depends(get_required_current_user),
        _: None = Depends(rate_limit(max_requests=10, period_seconds=60))
):
    """
    Test a prompt template with DeepEval metrics on sample data.

    This endpoint allows you to test how a prompt performs with DeepEval
    metrics before running a full evaluation.

    Args:
        prompt_id: The ID of the prompt to test
        test_request: Test configuration with sample data and metrics

    Returns:
        Dict containing test results and metric scores
    """
    try:
        # Get prompt
        from backend.app.db.repositories.base import BaseRepository
        from backend.app.db.models.orm import Prompt
        prompt_repo = BaseRepository(Prompt, db)
        prompt = await prompt_repo.get(prompt_id)

        if not prompt:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Prompt with ID {prompt_id} not found"
            )

        # Get agent
        from backend.app.db.repositories.agent_repository import AgentRepository
        agent_repo = AgentRepository(db)
        agent = await agent_repo.get_with_decrypted_credentials(test_request["agent_id"])

        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent with ID {test_request['agent_id']} not found"
            )

        # Create test case
        from deepeval.test_case import LLMTestCase
        test_case = LLMTestCase(
            input=test_request["sample_input"],
            expected_output=test_request.get("expected_output"),
            context=test_request.get("context", [])
        )

        # Generate response using prompt
        from backend.app.services.agent_clients.factory import AgentClientFactory
        from backend.app.evaluation.adapters.prompt_adapter import PromptAdapter

        agent_client = await AgentClientFactory.create_client(agent)
        prompt_adapter = PromptAdapter()

        actual_output = await prompt_adapter.apply_prompt_to_agent_client(
            agent_client, prompt, test_request["sample_input"], test_request.get("context", [])
        )

        test_case.actual_output = actual_output

        # Initialize DeepEval metrics
        from backend.app.evaluation.methods.deepeval import DeepEvalMethod
        deepeval_method = DeepEvalMethod(db)

        # Run metrics
        metric_configs = test_request.get("metrics", [{"name": "answer_relevancy", "threshold": 0.7}])
        metrics = deepeval_method._initialize_deepeval_metrics(
            [m["name"] for m in metric_configs],
            {"deepeval_config": {m["name"]: m for m in metric_configs}}
        )

        # Evaluate single test case
        from deepeval.dataset import EvaluationDataset
        dataset = EvaluationDataset(test_cases=[test_case])
        await deepeval_method._run_deepeval_async(dataset, metrics)

        # Extract results
        results = {}
        for metric in metrics:
            metric_name = metric.__name__ if hasattr(metric, '__name__') else str(metric)
            results[metric_name] = {
                "score": getattr(metric, 'score', 0.0),
                "success": getattr(metric, 'success', False),
                "reason": getattr(metric, 'reason', 'No reason provided'),
                "threshold": getattr(metric, 'threshold', 0.7)
            }

        return {
            "prompt_id": str(prompt_id),
            "agent_id": test_request["agent_id"],
            "test_input": test_request["sample_input"],
            "actual_output": actual_output,
            "expected_output": test_request.get("expected_output"),
            "context": test_request.get("context", []),
            "metric_results": results,
            "overall_summary": {
                "total_metrics": len(results),
                "passed_metrics": sum(1 for r in results.values() if r["success"]),
                "average_score": sum(r["score"] for r in results.values()) / len(results) if results else 0
            }
        }

    except Exception as e:
        logger.error(f"Error testing prompt with DeepEval: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error testing prompt: {str(e)}"
        )


@router.get("/{evaluation_id}/deepeval-insights")
async def get_deepeval_insights(
        evaluation_id: Annotated[UUID, Path(description="The ID of the evaluation to get insights for")],
        db: AsyncSession = Depends(get_db),
        current_user: UserContext = Depends(get_required_current_user)
):
    """
    Get DeepEval-specific insights and reasoning for an evaluation.

    This endpoint provides detailed insights that are specific to DeepEval
    evaluations, including metric reasoning and failure analysis.

    Args:
        evaluation_id: The ID of the evaluation

    Returns:
        Dict containing DeepEval-specific insights
        :param evaluation_id:
        :param current_user:
        :param db:
    """
    try:
        evaluation_service = EvaluationService(db)

        # Get evaluation and check permissions
        evaluation = await evaluation_service.get_evaluation(evaluation_id)
        if not evaluation:
            raise NotFoundException(resource="Evaluation", resource_id=str(evaluation_id))

        if evaluation.created_by_id and evaluation.created_by_id != current_user.db_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to access this evaluation"
            )

        if evaluation.method != EvaluationMethod.DEEPEVAL:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="This endpoint is only for DeepEval evaluations"
            )

        # Get detailed results with reasoning
        results = await evaluation_service.get_evaluation_results(evaluation_id)

        # Analyze results for insights
        insights = {
            "evaluation_id": str(evaluation_id),
            "method": "deepeval",
            "total_test_cases": len(results),
            "metric_insights": {},
            "failure_analysis": {},
            "recommendations": []
        }

        # Analyze metric performance
        metric_stats = {}
        failure_reasons = {}

        for result in results:
            for metric_score in result.metric_scores:
                metric_name = metric_score.name

                if metric_name not in metric_stats:
                    metric_stats[metric_name] = {
                        "scores": [],
                        "successes": 0,
                        "failures": 0,
                        "reasons": []
                    }

                metric_stats[metric_name]["scores"].append(metric_score.value)

                if metric_score.meta_info and metric_score.meta_info.get("success"):
                    metric_stats[metric_name]["successes"] += 1
                else:
                    metric_stats[metric_name]["failures"] += 1

                if metric_score.meta_info and metric_score.meta_info.get("reason"):
                    metric_stats[metric_name]["reasons"].append(metric_score.meta_info["reason"])

        # Generate insights for each metric
        for metric_name, stats in metric_stats.items():
            total_cases = len(stats["scores"])
            avg_score = sum(stats["scores"]) / total_cases if total_cases > 0 else 0
            success_rate = (stats["successes"] / total_cases) * 100 if total_cases > 0 else 0

            insights["metric_insights"][metric_name] = {
                "average_score": round(avg_score, 3),
                "success_rate": round(success_rate, 1),
                "total_cases": total_cases,
                "min_score": min(stats["scores"]) if stats["scores"] else 0,
                "max_score": max(stats["scores"]) if stats["scores"] else 0,
                "common_failure_reasons": _analyze_failure_reasons(stats["reasons"])
            }

        # Generate recommendations
        recommendations = []

        for metric_name, insight in insights["metric_insights"].items():
            if insight["success_rate"] < 70:
                if metric_name == "faithfulness":
                    recommendations.append(
                        f"Low faithfulness score ({insight['success_rate']:.1f}%). "
                        "Consider improving context quality or training the model to stick to provided information."
                    )
                elif metric_name == "answer_relevancy":
                    recommendations.append(
                        f"Low answer relevancy ({insight['success_rate']:.1f}%). "
                        "Review prompt templates to ensure they guide the model to address questions directly."
                    )
                elif metric_name == "hallucination":
                    recommendations.append(
                        f"High hallucination rate. "
                        "Consider adding stronger instructions to only use provided context."
                    )

        insights["recommendations"] = recommendations

        return insights

    except Exception as e:
        logger.error(f"Error getting DeepEval insights: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting insights: {str(e)}"
        )


def _analyze_failure_reasons(reasons: List[str]) -> List[Dict[str, Any]]:
    """Analyze failure reasons to find common patterns."""
    if not reasons:
        return []

    # Count reason patterns
    reason_counts = {}
    for reason in reasons:
        if reason and len(reason) > 10:  # Filter out very short reasons
            # Simple pattern matching for common issues
            if "context" in reason.lower():
                key = "context_issues"
            elif "relevant" in reason.lower():
                key = "relevance_issues"
            elif "accurate" in reason.lower() or "incorrect" in reason.lower():
                key = "accuracy_issues"
            elif "incomplete" in reason.lower():
                key = "completeness_issues"
            else:
                key = "other_issues"

            reason_counts[key] = reason_counts.get(key, 0) + 1

    # Sort by frequency and return top patterns
    sorted_patterns = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)

    return [
        {"pattern": pattern, "count": count, "percentage": round((count / len(reasons)) * 100, 1)}
        for pattern, count in sorted_patterns[:5]  # Top 5 patterns
    ]


@router.get("/metrics/deepeval/{dataset_type}")
async def get_deepeval_metrics_for_dataset_type(
        dataset_type: Annotated[DatasetType, Path(description="The dataset type to get metrics for")]
):
    """
    Get supported DeepEval metrics for a specific dataset type.

    Args:
        dataset_type: The type of dataset

    Returns:
        Dict containing supported DeepEval metrics and their configurations
    """
    try:
        from backend.app.evaluation.metrics.deepeval_metrics import (
            get_supported_metrics_for_dataset_type, get_recommended_metrics,
            get_metric_requirements, get_default_config, get_metric_categories
        )

        supported_metrics = get_supported_metrics_for_dataset_type(dataset_type)
        recommended_metrics = get_recommended_metrics(dataset_type)

        # Get detailed info for each metric
        metric_details = {}
        for metric_name in supported_metrics:
            requirements = get_metric_requirements(metric_name)
            default_config = get_default_config(metric_name)

            metric_details[metric_name] = {
                "description": requirements.get("description", ""),
                "required_fields": requirements.get("required_fields", []),
                "optional_fields": requirements.get("optional_fields", []),
                "threshold_range": requirements.get("threshold_range", (0.0, 1.0)),
                "higher_is_better": requirements.get("higher_is_better", True),
                "category": requirements.get("category", "quality"),
                "default_config": default_config,
                "recommended": metric_name in recommended_metrics
            }

        return {
            "dataset_type": dataset_type.value,
            "supported_metrics": supported_metrics,
            "recommended_metrics": recommended_metrics,
            "metric_categories": get_metric_categories(),
            "metric_details": metric_details
        }

    except Exception as e:
        logger.error(f"Error getting DeepEval metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting metrics: {str(e)}"
        )


# Add these improved endpoints to backend/app/api/v1/evaluations.py

@router.post("/{evaluation_id}/run-deepeval")
async def run_deepeval_evaluation(
        evaluation_id: Annotated[UUID, Path(description="The ID of the evaluation to run")],
        config: Optional[Dict[str, Any]] = Body(None, example={
            "metrics": ["answer_relevancy", "faithfulness", "contextual_precision"],
            "batch_size": 5,
            "include_reasoning": True,
            "thresholds": {
                "answer_relevancy": 0.7,
                "faithfulness": 0.8
            }
        }),
        db: AsyncSession = Depends(get_db),
        current_user: UserContext = Depends(get_required_current_user),
        jwt_token: Optional[str] = Depends(get_jwt_token),
        _: None = Depends(rate_limit(max_requests=3, period_seconds=300))
        # More restrictive for DeepEval
        , background_tasks=None):
    """
    Run a DeepEval evaluation with enhanced configuration options.

    This endpoint provides more control over DeepEval execution including:
    - Custom metric selection and thresholds
    - Batch size configuration for performance tuning
    - Real-time progress tracking
    - Enhanced error handling

    Args:
        evaluation_id: The ID of the evaluation to run
        config: DeepEval-specific configuration options

    Returns:
        Dict containing evaluation status and job information
        :param db:
    """
    logger.info(f"Starting DeepEval evaluation {evaluation_id}")

    evaluation_service = EvaluationService(db)

    try:
        # Get the evaluation
        evaluation = await evaluation_service.get_evaluation(evaluation_id)
        if not evaluation:
            raise NotFoundException(
                resource="Evaluation",
                resource_id=str(evaluation_id)
            )

        # Check permissions
        if evaluation.created_by_id and evaluation.created_by_id != current_user.db_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to run this evaluation"
            )

        # Validate evaluation method
        if evaluation.method != EvaluationMethod.DEEPEVAL:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Evaluation method is {evaluation.method}, not DEEPEVAL"
            )

        # Check if evaluation can be started
        if evaluation.status != EvaluationStatus.PENDING:
            raise InvalidStateException(
                resource="Evaluation",
                current_state=evaluation.status.value,
                operation="start",
                detail=f"Cannot start evaluation in {evaluation.status} status"
            )

        # Merge configuration
        deepeval_config = evaluation.config or {}
        if config:
            deepeval_config.update(config)

        # Update evaluation with DeepEval config
        await evaluation_service.update_evaluation(
            evaluation_id,
            EvaluationUpdate(config=deepeval_config)
        )

        # Start the evaluation
        now = datetime.datetime.now()
        await evaluation_service.evaluation_repo.update(evaluation_id, {
            "status": EvaluationStatus.RUNNING,
            "start_time": now
        })

        # Queue the evaluation with enhanced DeepEval support
        try:
            from backend.app.workers.tasks import run_deepeval_evaluation_task

            # Use DeepEval-specific Celery task
            task = run_deepeval_evaluation_task.delay(
                str(evaluation_id),
                jwt_token,
                deepeval_config
            )
            logger.info(f"Queued DeepEval evaluation {evaluation_id} to Celery")

            return {
                "evaluation_id": str(evaluation_id),
                "status": "running",
                "message": "DeepEval evaluation started successfully",
                "task_id": task.id,
                "config": deepeval_config,
                "estimated_time_minutes": _estimate_deepeval_time(evaluation, deepeval_config)
            }

        except Exception as e:
            logger.warning(f"Failed to queue to Celery: {e}. Using background task.")

            # Fallback to background task
            background_tasks.add_task(
                _run_deepeval_as_background_task,
                str(evaluation_id),
                jwt_token,
                deepeval_config
            )

            return {
                "evaluation_id": str(evaluation_id),
                "status": "running",
                "message": "DeepEval evaluation started as background task",
                "config": deepeval_config
            }

    except (NotFoundException, InvalidStateException):
        raise
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting DeepEval evaluation {evaluation_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error starting DeepEval evaluation: {str(e)}"
        )


@router.post("/{evaluation_id}/test-deepeval-metrics")
async def test_deepeval_metrics(
        evaluation_id: Annotated[UUID, Path(description="The ID of the evaluation to test")],
        test_request: Dict[str, Any] = Body(..., example={
            "sample_data": [
                {
                    "input": "What is machine learning?",
                    "context": [
                        "Machine learning is a subset of AI that enables computers to learn without explicit programming."],
                    "expected_output": "Machine learning is a method of data analysis that automates analytical model building."
                }
            ],
            "metrics": ["answer_relevancy", "faithfulness"],
            "thresholds": {
                "answer_relevancy": 0.7,
                "faithfulness": 0.8
            }
        }),
        db: AsyncSession = Depends(get_db),
        current_user: UserContext = Depends(get_required_current_user),
        jwt_token: Optional[str] = Depends(get_jwt_token),
        _: None = Depends(rate_limit(max_requests=10, period_seconds=60))
):
    """
    Test DeepEval metrics on sample data before running full evaluation.

    This endpoint allows you to:
    - Test specific metrics with sample data
    - Validate agent responses
    - Fine-tune thresholds
    - Preview evaluation behavior

    Args:
        evaluation_id: The ID of the evaluation to test
        test_request: Test configuration with sample data and metrics

    Returns:
        Dict containing test results and recommendations
    """
    logger.info(f"Testing DeepEval metrics for evaluation {evaluation_id}")

    evaluation_service = EvaluationService(db)

    try:
        # Get evaluation and verify access
        evaluation = await evaluation_service.get_evaluation(evaluation_id)
        if not evaluation:
            raise NotFoundException(resource="Evaluation", resource_id=str(evaluation_id))

        if evaluation.created_by_id and evaluation.created_by_id != current_user.db_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to test this evaluation"
            )

        # Get related entities
        from backend.app.db.repositories.agent_repository import AgentRepository
        from backend.app.services.agent_clients.factory import AgentClientFactory

        agent_repo = AgentRepository(db)
        agent = await agent_repo.get_with_decrypted_credentials(evaluation.agent_id)
        prompt = await evaluation_service.get_prompt(evaluation.prompt_id)

        if not agent or not prompt:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing required agent or prompt for testing"
            )

        # Create agent client
        from backend.app.db.models.orm import IntegrationType
        if agent.integration_type == IntegrationType.MCP and jwt_token:
            agent_client = await AgentClientFactory.create_client(agent, jwt_token)
        else:
            agent_client = await AgentClientFactory.create_client(agent)

        # Process sample data
        sample_data = test_request.get("sample_data", [])
        metrics = test_request.get("metrics", ["answer_relevancy"])
        thresholds = test_request.get("thresholds", {})

        if not sample_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No sample data provided for testing"
            )

        # Import DeepEval components
        from backend.app.evaluation.methods.deepeval import DeepEvalMethod, CustomDeepEvalLLM
        from deepeval.test_case import LLMTestCase
        from deepeval.dataset import EvaluationDataset

        # Create test cases
        test_cases = []
        for i, item in enumerate(sample_data):
            # Generate agent response
            from backend.app.evaluation.adapters.prompt_adapter import PromptAdapter
            prompt_adapter = PromptAdapter()

            try:
                actual_output = await prompt_adapter.apply_prompt_to_agent_client(
                    agent_client, prompt, item.get("input", ""), item.get("context", [])
                )
            except Exception as e:
                logger.error(f"Error generating response for test item {i}: {e}")
                actual_output = f"Error: {str(e)}"

            test_case = LLMTestCase(
                input=item.get("input", ""),
                actual_output=actual_output,
                expected_output=item.get("expected_output", ""),
                context=item.get("context", [])
            )
            test_cases.append(test_case)

        # Initialize DeepEval metrics
        deepeval_method = DeepEvalMethod(db)
        custom_llm = CustomDeepEvalLLM(agent_client)

        # Create config for metrics
        config = {
            "deepeval_config": {
                metric: {"threshold": thresholds.get(metric, 0.7)}
                for metric in metrics
            }
        }

        deepeval_metrics = deepeval_method._initialize_deepeval_metrics(
            metrics, config, custom_llm
        )

        # Run evaluation
        test_dataset = EvaluationDataset(test_cases=test_cases)
        await deepeval_method._run_deepeval_async(test_dataset, deepeval_metrics)

        # Process results
        test_results = []
        metric_summaries = {}

        for i, test_case in enumerate(test_cases):
            case_result = {
                "test_case_index": i,
                "input": test_case.input,
                "actual_output": test_case.actual_output,
                "expected_output": test_case.expected_output,
                "context": test_case.context,
                "metrics": []
            }

            # Extract metric results
            for attr_name, attr_value in test_case.__dict__.items():
                if attr_name.endswith('_metric') and hasattr(attr_value, 'score'):
                    metric = attr_value
                    metric_name = attr_name.replace('_metric', '')

                    metric_result = {
                        "name": metric_name,
                        "score": getattr(metric, 'score', 0),
                        "success": getattr(metric, 'success', False),
                        "threshold": getattr(metric, 'threshold', 0.7),
                        "reason": getattr(metric, 'reason', '')
                    }

                    case_result["metrics"].append(metric_result)

                    # Track for summary
                    if metric_name not in metric_summaries:
                        metric_summaries[metric_name] = {"scores": [], "successes": 0, "total": 0}

                    metric_summaries[metric_name]["scores"].append(metric_result["score"])
                    metric_summaries[metric_name]["total"] += 1
                    if metric_result["success"]:
                        metric_summaries[metric_name]["successes"] += 1

            test_results.append(case_result)

        # Calculate summary statistics
        summary_stats = {}
        for metric_name, data in metric_summaries.items():
            if data["total"] > 0:
                summary_stats[metric_name] = {
                    "average_score": sum(data["scores"]) / len(data["scores"]),
                    "success_rate": (data["successes"] / data["total"]) * 100,
                    "total_tests": data["total"],
                    "min_score": min(data["scores"]),
                    "max_score": max(data["scores"])
                }

        # Generate recommendations
        recommendations = []
        for metric_name, stats in summary_stats.items():
            if stats["success_rate"] < 50:
                recommendations.append(
                    f"Consider lowering the threshold for {metric_name} (current success rate: {stats['success_rate']:.1f}%)"
                )
            elif stats["success_rate"] > 90:
                recommendations.append(
                    f"Excellent performance on {metric_name} (success rate: {stats['success_rate']:.1f}%)"
                )

        return {
            "evaluation_id": str(evaluation_id),
            "test_summary": {
                "total_test_cases": len(test_results),
                "metrics_tested": list(summary_stats.keys()),
                "overall_success": all(
                    stats["success_rate"] > 50 for stats in summary_stats.values()
                )
            },
            "metric_performance": summary_stats,
            "detailed_results": test_results,
            "recommendations": recommendations,
            "next_steps": [
                "Review metric performance and adjust thresholds if needed",
                "Run full evaluation if test results are satisfactory",
                "Consider adding more metrics based on your use case"
            ]
        }

    except Exception as e:
        logger.error(f"Error testing DeepEval metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error testing metrics: {str(e)}"
        )


def _estimate_deepeval_time(evaluation: Evaluation, config: Dict[str, Any]) -> int:
    """Estimate DeepEval evaluation time in minutes."""
    try:
        # Get dataset to estimate size
        from backend.app.db.repositories.base import BaseRepository
        from backend.app.db.models.orm import Dataset

        dataset_repo = BaseRepository(Dataset, evaluation.db_session)
        dataset = dataset_repo.get(evaluation.dataset_id)

        if not dataset or not dataset.row_count:
            return 10  # Default estimate

        # Estimate based on dataset size and metrics
        items = dataset.row_count
        metrics_count = len(config.get("metrics", ["answer_relevancy"]))
        batch_size = config.get("batch_size", 5)

        # Rough estimate: 30 seconds per item per metric, plus batch overhead
        time_per_item = 30 * metrics_count  # seconds
        total_time = (items * time_per_item) / batch_size

        # Convert to minutes and add buffer
        return max(5, int(total_time / 60 * 1.2))  # 20% buffer

    except Exception:
        return 15  # Conservative default


async def _run_deepeval_as_background_task(
        evaluation_id: str,
        jwt_token: Optional[str],
        config: Dict[str, Any]
) -> None:
    """Run DeepEval evaluation as a background task."""
    try:
        from backend.app.db.session import db_session
        from backend.app.services.evaluation_service import EvaluationService
        from uuid import UUID

        evaluation_uuid = UUID(evaluation_id)

        async with db_session() as session:
            service = EvaluationService(session)

            # Get evaluation method handler
            evaluation = await service.get_evaluation(evaluation_uuid)
            method_handler = await service.get_evaluation_method_handler(evaluation.method)

            # Run the evaluation
            results = await method_handler.run_evaluation(evaluation, jwt_token=jwt_token)

            # Save results
            for result_data in results:
                await service.create_evaluation_result(result_data)

            # Mark as completed
            await service.complete_evaluation(evaluation_uuid, success=True)

            logger.info(f"DeepEval background task completed for evaluation {evaluation_id}")

    except Exception as e:
        logger.error(f"DeepEval background task failed for evaluation {evaluation_id}: {e}")
        # Mark as failed if possible
        try:
            async with db_session() as session:
                service = EvaluationService(session)
                await service.complete_evaluation(UUID(evaluation_id), success=False)
        except Exception:
            pass
