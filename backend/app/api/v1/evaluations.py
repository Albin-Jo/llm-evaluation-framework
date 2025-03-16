# File: app/api/v1/evaluations.py
from typing import Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.orm.models import Evaluation, EvaluationStatus, User
from app.schema.evaluation_schema import (
    EvaluationCreate, EvaluationDetailResponse,
    EvaluationResponse, EvaluationUpdate
)
from app.services.auth import get_current_active_user
from app.services.evaluation_service import EvaluationService

router = APIRouter()


@router.post("/", response_model=EvaluationResponse)
async def create_evaluation(
        evaluation_data: EvaluationCreate,
        current_user: User = Depends(get_current_active_user),
        db: AsyncSession = Depends(get_db)
):
    """
    Create a new evaluation.
    """
    evaluation_service = EvaluationService(db)
    evaluation = await evaluation_service.create_evaluation(evaluation_data, current_user)
    return evaluation


@router.get("/", response_model=List[EvaluationResponse])
async def list_evaluations(
        skip: int = 0,
        limit: int = 100,
        status: Optional[EvaluationStatus] = None,
        micro_agent_id: Optional[UUID] = None,
        dataset_id: Optional[UUID] = None,
        current_user: User = Depends(get_current_active_user),
        db: AsyncSession = Depends(get_db)
):
    """
    List evaluations with optional filtering.
    """
    filters = {}

    # Add filters if provided
    if status:
        filters["status"] = status
    if micro_agent_id:
        filters["micro_agent_id"] = micro_agent_id
    if dataset_id:
        filters["dataset_id"] = dataset_id

    # Always filter by current user unless user is admin
    if current_user.role.value != "admin":
        filters["created_by_id"] = current_user.id

    evaluation_service = EvaluationService(db)
    evaluations = await evaluation_service.list_evaluations(skip, limit, filters)
    return evaluations


@router.get("/{evaluation_id}", response_model=EvaluationDetailResponse)
async def get_evaluation(
        evaluation_id: UUID,
        current_user: User = Depends(get_current_active_user),
        db: AsyncSession = Depends(get_db)
):
    """
    Get evaluation by ID, including results.
    """
    evaluation_service = EvaluationService(db)

    # Get the evaluation
    evaluation = await evaluation_service.get_evaluation(evaluation_id)
    if not evaluation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evaluation with ID {evaluation_id} not found"
        )

    # Check if user has permission to view this evaluation
    if current_user.role.value != "admin" and evaluation.created_by_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )

    # Get evaluation results
    results = await evaluation_service.get_evaluation_results(evaluation_id)

    # Create response with results
    response = EvaluationDetailResponse.model_validate(evaluation)
    response.results = results

    return response


@router.put("/{evaluation_id}", response_model=EvaluationResponse)
async def update_evaluation(
        evaluation_id: UUID,
        evaluation_data: EvaluationUpdate,
        current_user: User = Depends(get_current_active_user),
        db: AsyncSession = Depends(get_db)
):
    """
    Update evaluation by ID.
    """
    evaluation_service = EvaluationService(db)

    # Get the evaluation to check permissions
    evaluation = await evaluation_service.get_evaluation(evaluation_id)
    if not evaluation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evaluation with ID {evaluation_id} not found"
        )

    # Check if user has permission to update this evaluation
    if current_user.role.value != "admin" and evaluation.created_by_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )

    # Update the evaluation
    updated_evaluation = await evaluation_service.update_evaluation(
        evaluation_id, evaluation_data
    )

    if not updated_evaluation:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update evaluation"
        )

    return updated_evaluation


@router.delete("/{evaluation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_evaluation(
        evaluation_id: UUID,
        current_user: User = Depends(get_current_active_user),
        db: AsyncSession = Depends(get_db)
):
    """
    Delete evaluation by ID.
    """
    evaluation_service = EvaluationService(db)

    # Get the evaluation to check permissions
    evaluation = await evaluation_service.get_evaluation(evaluation_id)
    if not evaluation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evaluation with ID {evaluation_id} not found"
        )

    # Check if user has permission to delete this evaluation
    if current_user.role.value != "admin" and evaluation.created_by_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )

    # Delete the evaluation
    success = await evaluation_service.delete_evaluation(evaluation_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete evaluation"
        )


# File: app/api/v1/evaluations.py
# Enhance error handling in start_evaluation function
@router.post("/{evaluation_id}/start", response_model=EvaluationResponse)
async def start_evaluation(
        evaluation_id: UUID,
        current_user: User = Depends(get_current_active_user),
        db: AsyncSession = Depends(get_db)
):
    """
    Start an evaluation.

    This endpoint initiates the evaluation process, which will run asynchronously.
    The status will change from PENDING to RUNNING, and eventually to COMPLETED or FAILED.
    """
    evaluation_service = EvaluationService(db)

    # Get the evaluation to check permissions
    evaluation = await evaluation_service.get_evaluation(evaluation_id)
    if not evaluation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evaluation with ID {evaluation_id} not found. Please check the ID and try again."
        )

    # Check if user has permission to start this evaluation
    if current_user.role.value != "admin" and evaluation.created_by_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to start this evaluation. Only the creator or an admin can start it."
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
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot start evaluation because it is {message}. Create a new evaluation or retry if failed."
        )

    # Queue the evaluation job
    try:
        await evaluation_service.queue_evaluation_job(evaluation_id)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to queue evaluation job: {str(e)}. Please check the server logs or try again later."
        )

    # Get updated evaluation
    updated_evaluation = await evaluation_service.get_evaluation(evaluation_id)
    return updated_evaluation


@router.post("/{evaluation_id}/cancel", response_model=EvaluationResponse)
async def cancel_evaluation(
        evaluation_id: UUID,
        current_user: User = Depends(get_current_active_user),
        db: AsyncSession = Depends(get_db)
):
    """
    Cancel a running evaluation.
    """
    evaluation_service = EvaluationService(db)

    # Get the evaluation to check permissions
    evaluation = await evaluation_service.get_evaluation(evaluation_id)
    if not evaluation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evaluation with ID {evaluation_id} not found"
        )

    # Check if user has permission to cancel this evaluation
    if current_user.role.value != "admin" and evaluation.created_by_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )

    # Check if the evaluation is running
    if evaluation.status != EvaluationStatus.RUNNING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Evaluation is not in RUNNING status"
        )

    # Cancel the evaluation
    update_data = EvaluationUpdate(status=EvaluationStatus.CANCELLED)
    updated_evaluation = await evaluation_service.update_evaluation(
        evaluation_id, update_data
    )

    return updated_evaluation


@router.get("/{evaluation_id}/results", response_model=List[Dict])
async def get_evaluation_results(
        evaluation_id: UUID,
        current_user: User = Depends(get_current_active_user),
        db: AsyncSession = Depends(get_db)
):
    """
    Get results for an evaluation.
    """
    evaluation_service = EvaluationService(db)

    # Get the evaluation to check permissions
    evaluation = await evaluation_service.get_evaluation(evaluation_id)
    if not evaluation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evaluation with ID {evaluation_id} not found"
        )

    # Check if user has permission to view this evaluation
    if current_user.role.value != "admin" and evaluation.created_by_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )

    # Get evaluation results
    results = await evaluation_service.get_evaluation_results(evaluation_id)

    # Process results for response
    processed_results = []
    for result in results:
        result_dict = result.to_dict()

        # Get metric scores
        metric_scores = await evaluation_service.get_metric_scores(result.id)
        result_dict["metric_scores"] = [score.to_dict() for score in metric_scores]

        processed_results.append(result_dict)

    return processed_results