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
from backend.app.db.models.orm import EvaluationStatus, EvaluationMethod, EvaluationResult
from backend.app.db.schema.evaluation_schema import (
    EvaluationCreate, EvaluationDetailResponse,
    EvaluationResponse, EvaluationUpdate
)
from backend.app.db.session import get_db
from backend.app.evaluation.metrics.ragas_metrics import DATASET_TYPE_METRICS
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
