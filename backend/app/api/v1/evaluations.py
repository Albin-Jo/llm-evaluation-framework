# backend/app/api/v1/evaluations.py
import logging
from typing import Dict, List, Optional, Union, Any, Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Body, Query, Path, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models.orm import EvaluationStatus, EvaluationMethod
from backend.app.db.schema.evaluation_schema import (
    EvaluationCreate, EvaluationDetailResponse,
    EvaluationResponse, EvaluationUpdate
)
from backend.app.db.session import get_db
from backend.app.services.evaluation_service import EvaluationService
from backend.app.evaluation.metrics.ragas_metrics import DATASET_TYPE_METRICS
from backend.app.api.dependencies.rate_limiter import rate_limit

# Configure logger
logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/", response_model=EvaluationResponse)
async def create_evaluation(
        evaluation_data: EvaluationCreate,
        db: AsyncSession = Depends(get_db),
        _: None = Depends(rate_limit(max_requests=10, period_seconds=60))
):
    """
    Create a new evaluation.

    This endpoint creates a new evaluation configuration with the specified parameters.
    After creation, you'll need to call the `/start` endpoint to begin the evaluation process.

    - **evaluation_data**: Required evaluation configuration data

    Returns the created evaluation object with an ID that can be used for future operations.
    """
    logger.info(f"Creating new evaluation with dataset_id={evaluation_data.dataset_id}, "
                f"agent_id={evaluation_data.agent_id}")

    evaluation_service = EvaluationService(db)
    try:
        evaluation = await evaluation_service.create_evaluation(evaluation_data)
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
        _: None = Depends(rate_limit(max_requests=50, period_seconds=60))
):
    """
    List evaluations with optional filtering, sorting and pagination.

    This endpoint returns both the evaluations array and a total count for pagination purposes.

    - **skip**: Number of records to skip (for pagination)
    - **limit**: Maximum number of records to return
    - **status**: Optional filter by evaluation status
    - **agent_id**: Optional filter by agent ID
    - **dataset_id**: Optional filter by dataset ID
    - **name**: Optional filter by evaluation name (case-insensitive, supports partial matching)
    - **method**: Optional filter by evaluation method
    - **sort_by**: Field to sort results by (default: created_at)
    - **sort_dir**: Sort direction, either "asc" or "desc" (default: desc)

    Returns a dictionary containing the list of evaluations and the total count.
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

        # Return both results and total count
        return {
            "items": evaluations,
            "total": total_count
        }
    except Exception as e:
        logger.error(f"Error listing evaluations: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing evaluations: {str(e)}"
        )


@router.get("/{evaluation_id}", response_model=EvaluationDetailResponse)
async def get_evaluation(
        evaluation_id: Annotated[UUID, Path(description="The ID of the evaluation to retrieve")],
        db: AsyncSession = Depends(get_db),
):
    """
    Get evaluation by ID with all related details.

    This endpoint retrieves comprehensive information about an evaluation, including:
    - Basic evaluation metadata
    - Configuration details
    - All results with their metrics

    - **evaluation_id**: The unique identifier of the evaluation

    Returns the complete evaluation object with nested results and metrics.
    """
    try:
        # Create evaluation service
        evaluation_service = EvaluationService(db)

        # Get the evaluation with all relationships in one query
        evaluation, result_responses = await evaluation_service.get_evaluation_with_relationships(evaluation_id)

        if not evaluation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Evaluation with ID {evaluation_id} not found"
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
            "results": result_responses
        }

        # Return the response data directly - FastAPI will handle conversion
        return response_dict

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
        db: AsyncSession = Depends(get_db)
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
            logger.warning(f"Evaluation id={evaluation_id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Evaluation with ID {evaluation_id} not found"
            )

        # Check if we're trying to update fields that shouldn't be changed after certain states
        if evaluation.status not in [EvaluationStatus.PENDING, EvaluationStatus.FAILED]:
            protected_fields = ["method", "metrics", "config"]
            update_dict = evaluation_data.model_dump(exclude_unset=True)

            if any(field in update_dict for field in protected_fields):
                logger.warning(
                    f"Attempted to update protected fields of evaluation {evaluation_id} in {evaluation.status} state")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Cannot update method, metrics, or config for an evaluation in {evaluation.status} status"
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
    except HTTPException:
        # Re-raise HTTP exceptions
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
        db: AsyncSession = Depends(get_db)
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
            logger.warning(f"Evaluation id={evaluation_id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Evaluation with ID {evaluation_id} not found"
            )

        # Check if evaluation is running before deleting
        if evaluation.status == EvaluationStatus.RUNNING:
            logger.warning(f"Attempted to delete running evaluation {evaluation_id}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
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
    except HTTPException:
        # Re-raise HTTP exceptions
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
        _: None = Depends(rate_limit(max_requests=5, period_seconds=60))
):
    """
    Start an evaluation.

    This endpoint initiates the evaluation process, which will run asynchronously.
    The status will change from PENDING to RUNNING, and eventually to COMPLETED or FAILED.
    You can check the progress using the /progress endpoint.

    - **evaluation_id**: The unique identifier of the evaluation to start

    Returns the updated evaluation object with RUNNING status.
    """
    logger.info(f"Starting evaluation id={evaluation_id}")

    evaluation_service = EvaluationService(db)

    try:
        # Get the evaluation to check permissions
        evaluation = await evaluation_service.get_evaluation(evaluation_id)
        if not evaluation:
            logger.warning(f"Evaluation id={evaluation_id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Evaluation with ID {evaluation_id} not found. Please check the ID and try again."
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
            logger.warning(f"Cannot start evaluation id={evaluation_id} because it is {message}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot start evaluation because it is {message}. Create a new evaluation or retry if failed."
            )

        # Queue the evaluation job
        logger.debug(f"Queuing evaluation job for id={evaluation_id}")
        try:
            await evaluation_service.queue_evaluation_job(evaluation_id)
        except Exception as e:
            logger.error(f"Failed to queue evaluation job for id={evaluation_id}: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to queue evaluation job: {str(e)}. Please check the server logs or try again later."
            )

        # Get updated evaluation
        updated_evaluation = await evaluation_service.get_evaluation(evaluation_id)
        logger.info(f"Successfully started evaluation id={evaluation_id}")
        return updated_evaluation
    except HTTPException:
        # Re-raise HTTP exceptions
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
        db: AsyncSession = Depends(get_db)
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
        progress = await evaluation_service.get_evaluation_progress(evaluation_id)
        return progress
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
        db: AsyncSession = Depends(get_db)
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
            logger.warning(f"Evaluation id={evaluation_id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Evaluation with ID {evaluation_id} not found"
            )

        # Check if the evaluation is running
        if evaluation.status != EvaluationStatus.RUNNING:
            logger.warning(f"Cannot cancel evaluation id={evaluation_id} because it is not running")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
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
    except HTTPException:
        # Re-raise HTTP exceptions
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
        db: AsyncSession = Depends(get_db)
):
    """
    Get results for an evaluation with pagination.

    This endpoint returns the detailed results for an evaluation, including all metric scores.
    Results are paginated for better performance with large evaluations.

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
            logger.warning(f"Evaluation id={evaluation_id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Evaluation with ID {evaluation_id} not found"
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

            processed_results.append(result_dict)

        logger.info(f"Successfully retrieved {len(processed_results)} results for evaluation id={evaluation_id}")
        return {
            "items": processed_results,
            "total": total_count
        }
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error retrieving results for evaluation id={evaluation_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving results: {str(e)}"
        )


@router.get("/metrics/{dataset_type}", response_model=Dict[str, Union[str, List[str]]])
async def get_supported_metrics(
        dataset_type: Annotated[str, Path(description="The dataset type to get supported metrics for")],
        db: AsyncSession = Depends(get_db)
):
    """
    Get supported metrics for a specific dataset type.

    This endpoint returns the list of metrics that can be calculated for a given dataset type.

    - **dataset_type**: The type of dataset (e.g., user_query, context, question_answer, etc.)

    Returns a dictionary with the dataset type and list of supported metrics.
    """
    try:
        if dataset_type not in DATASET_TYPE_METRICS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid dataset type: {dataset_type}. Valid types are: {list(DATASET_TYPE_METRICS.keys())}"
            )

        return {
            "dataset_type": dataset_type,
            "supported_metrics": DATASET_TYPE_METRICS[dataset_type]
        }
    except HTTPException:
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
            logger.warning(f"Evaluation id={evaluation_id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Evaluation with ID {evaluation_id} not found"
            )

        # Validate minimum required test data
        required_fields = ["query", "context", "answer"]
        missing_fields = [field for field in required_fields if field not in test_data]

        if missing_fields:
            logger.warning(f"Missing required fields {missing_fields} in test data for evaluation id={evaluation_id}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
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
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error testing evaluation id={evaluation_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error testing evaluation: {str(e)}"
        )