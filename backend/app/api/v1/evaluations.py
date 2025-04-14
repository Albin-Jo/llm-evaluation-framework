# backend/app/api/v1/evaluations.py
import logging
from typing import Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Body
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models.orm import EvaluationStatus
from backend.app.db.schema.evaluation_schema import (
    EvaluationCreate, EvaluationDetailResponse,
    EvaluationResponse, EvaluationUpdate
)
from backend.app.db.session import get_db
from backend.app.services.evaluation_service import EvaluationService
from backend.app.evaluation.metrics.ragas_metrics import DATASET_TYPE_METRICS

# Configure logger
logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/", response_model=EvaluationResponse)
async def create_evaluation(
        evaluation_data: EvaluationCreate,
        db: AsyncSession = Depends(get_db)
):
    """
    Create a new evaluation.
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
        raise


@router.get("/", response_model=List[EvaluationResponse])
async def list_evaluations(
        skip: int = 0,
        limit: int = 100,
        status: Optional[EvaluationStatus] = None,
        agent_id: Optional[UUID] = None,
        dataset_id: Optional[UUID] = None,
        db: AsyncSession = Depends(get_db)
):
    """
    List evaluations with optional filtering.
    """
    filters = {}

    # Add filters if provided
    if status:
        filters["status"] = status
    if agent_id:
        filters["agent_id"] = agent_id
    if dataset_id:
        filters["dataset_id"] = dataset_id

    logger.info(f"Listing evaluations with filters={filters}, skip={skip}, limit={limit}")

    evaluation_service = EvaluationService(db)
    try:
        evaluations = await evaluation_service.list_evaluations(skip, limit, filters)
        logger.debug(f"Retrieved {len(evaluations)} evaluations")
        return evaluations
    except Exception as e:
        logger.error(f"Error listing evaluations: {str(e)}", exc_info=True)
        raise


@router.get("/{evaluation_id}", response_model=EvaluationDetailResponse)
async def get_evaluation(
        evaluation_id: UUID,
        db: AsyncSession = Depends(get_db),
):
    """
    Get evaluation by ID with all related details.
    """
    try:
        # Create evaluation service
        evaluation_service = EvaluationService(db)

        # Get the evaluation without triggering lazy loading
        evaluation = await evaluation_service.get_evaluation(evaluation_id)

        if not evaluation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Evaluation with ID {evaluation_id} not found"
            )

        # Get results separately to avoid lazy loading issues
        results = await evaluation_service.get_evaluation_results(evaluation_id)

        # Process results to add their metric scores
        result_responses = []
        for result in results:
            # Get metric scores for this result
            metric_scores = await evaluation_service.get_metric_scores(result.id)

            # Create a dictionary for result data
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
                    for score in metric_scores
                ]
            }

            result_responses.append(result_dict)

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
        evaluation_id: UUID,
        evaluation_data: EvaluationUpdate,
        db: AsyncSession = Depends(get_db)
):
    """
    Update evaluation by ID.
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
        raise


@router.delete("/{evaluation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_evaluation(
        evaluation_id: UUID,
        db: AsyncSession = Depends(get_db)
):
    """
    Delete evaluation by ID.
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
        raise


@router.post("/{evaluation_id}/start", response_model=EvaluationResponse)
async def start_evaluation(
        evaluation_id: UUID,
        db: AsyncSession = Depends(get_db)
):
    """
    Start an evaluation.

    This endpoint initiates the evaluation process, which will run asynchronously.
    The status will change from PENDING to RUNNING, and eventually to COMPLETED or FAILED.
    You can check the progress using the /progress endpoint.
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
        raise


@router.get("/{evaluation_id}/progress", response_model=Dict)
async def get_evaluation_progress(
        evaluation_id: UUID,
        db: AsyncSession = Depends(get_db)
):
    """
    Get the progress of an evaluation.

    This endpoint returns detailed information about the evaluation progress,
    including the current status, number of processed items, and percentage complete.
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
        evaluation_id: UUID,
        db: AsyncSession = Depends(get_db)
):
    """
    Cancel a running evaluation.
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
                detail=f"Evaluation is not in RUNNING status"
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
        raise


@router.get("/{evaluation_id}/results", response_model=List[Dict])
async def get_evaluation_results(
        evaluation_id: UUID,
        db: AsyncSession = Depends(get_db)
):
    """
    Get results for an evaluation.
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
        results = await evaluation_service.get_evaluation_results(evaluation_id)

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
        return processed_results
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error retrieving results for evaluation id={evaluation_id}: {str(e)}", exc_info=True)
        raise


@router.get("/metrics/{dataset_type}", response_model=Dict[str, List[str]])
async def get_supported_metrics(
        dataset_type: str,
        db: AsyncSession = Depends(get_db)
):
    """
    Get supported metrics for a specific dataset type.

    This endpoint returns the list of metrics that can be calculated for a given dataset type.
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
        evaluation_id: UUID,
        test_data: Dict = Body(...),
        db: AsyncSession = Depends(get_db)
):
    """
    Test an evaluation with sample data without creating results.

    This is useful for validating configurations and testing metrics
    before running a full evaluation.

    The request body should contain test data in the format:
    {
        "query": "Sample query",
        "context": "Sample context",
        "answer": "Sample answer",
        "ground_truth": "Optional ground truth"
    }
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
        for field in required_fields:
            if field not in test_data:
                logger.warning(f"Missing required field '{field}' in test data for evaluation id={evaluation_id}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Missing required field: {field}"
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