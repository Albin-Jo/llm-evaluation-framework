# File: app/api/v1/comparisons.py
from typing import Dict, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.orm.models import Evaluation, EvaluationResult, MetricScore, User, evaluation_comparison
from app.schema.evaluation_schema import (
    EvaluationComparisonCreate, EvaluationComparisonResponse
)
from app.services.auth import get_current_active_user

router = APIRouter()


@router.post("/", response_model=EvaluationComparisonResponse)
async def create_comparison(
        comparison_data: EvaluationComparisonCreate,
        current_user: User = Depends(get_current_active_user),
        db: AsyncSession = Depends(get_db)
):
    """
    Create a comparison between two evaluations.
    """
    # Check if evaluations exist
    query = select(Evaluation).where(
        Evaluation.id.in_([comparison_data.evaluation_a_id, comparison_data.evaluation_b_id])
    )
    result = await db.execute(query)
    evaluations = result.scalars().all()

    if len(evaluations) != 2:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="One or both evaluations not found"
        )

    # Check if user has permission to view these evaluations
    for evaluation in evaluations:
        if (
                evaluation.created_by_id != current_user.id
                and current_user.role.value != "admin"
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Not enough permissions to access evaluation {evaluation.id}"
            )

    # Check if evaluations are completed
    for evaluation in evaluations:
        if evaluation.status != "completed":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Evaluation {evaluation.id} is not completed"
            )

    # Generate comparison results
    comparison_results = await generate_comparison_results(
        db, comparison_data.evaluation_a_id, comparison_data.evaluation_b_id
    )

    # Create comparison record
    stmt = evaluation_comparison.insert().values(
        name=comparison_data.name,
        description=comparison_data.description,
        created_by_id=current_user.id,
        evaluation_a_id=comparison_data.evaluation_a_id,
        evaluation_b_id=comparison_data.evaluation_b_id,
        comparison_results=comparison_results
    ).returning(evaluation_comparison)

    result = await db.execute(stmt)
    db_comparison = result.one()
    await db.commit()

    # Return comparison
    return db_comparison


@router.get("/{comparison_id}", response_model=EvaluationComparisonResponse)
async def get_comparison(
        comparison_id: UUID,
        current_user: User = Depends(get_current_active_user),
        db: AsyncSession = Depends(get_db)
):
    """
    Get comparison by ID.
    """
    query = select(evaluation_comparison).where(evaluation_comparison.c.id == comparison_id)
    result = await db.execute(query)
    comparison = result.one_or_none()

    if not comparison:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Comparison with ID {comparison_id} not found"
        )

    # Check if user has permission to view this comparison
    if comparison.created_by_id != current_user.id and current_user.role.value != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )

    return comparison


@router.get("/", response_model=List[EvaluationComparisonResponse])
async def list_comparisons(
        skip: int = 0,
        limit: int = 100,
        current_user: User = Depends(get_current_active_user),
        db: AsyncSession = Depends(get_db)
):
    """
    List comparisons created by the current user or visible to them.
    """
    if current_user.role.value == "admin":
        # Admins can see all comparisons
        query = select(evaluation_comparison).offset(skip).limit(limit)
    else:
        # Regular users can only see their own comparisons
        query = select(evaluation_comparison).where(
            evaluation_comparison.c.created_by_id == current_user.id
        ).offset(skip).limit(limit)

    result = await db.execute(query)
    comparisons = result.all()

    return comparisons


@router.delete("/{comparison_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_comparison(
        comparison_id: UUID,
        current_user: User = Depends(get_current_active_user),
        db: AsyncSession = Depends(get_db)
):
    """
    Delete comparison by ID.
    """
    query = select(evaluation_comparison).where(evaluation_comparison.c.id == comparison_id)
    result = await db.execute(query)
    comparison = result.one_or_none()

    if not comparison:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Comparison with ID {comparison_id} not found"
        )

    # Check if user has permission to delete this comparison
    if comparison.created_by_id != current_user.id and current_user.role.value != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )

    # Delete the comparison
    stmt = evaluation_comparison.delete().where(evaluation_comparison.c.id == comparison_id)
    await db.execute(stmt)
    await db.commit()


async def generate_comparison_results(
        db: AsyncSession, evaluation_a_id: UUID, evaluation_b_id: UUID
) -> Dict:
    """
    Generate comparison results between two evaluations.

    Args:
        db: Database session
        evaluation_a_id: First evaluation ID
        evaluation_b_id: Second evaluation ID

    Returns:
        Dict: Comparison results
    """
    # Get evaluations with results
    query_a = select(Evaluation).where(Evaluation.id == evaluation_a_id)
    query_b = select(Evaluation).where(Evaluation.id == evaluation_b_id)

    result_a = await db.execute(query_a)
    result_b = await db.execute(query_b)

    evaluation_a = result_a.scalar_one_or_none()
    evaluation_b = result_b.scalar_one_or_none()

    if not evaluation_a or not evaluation_b:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="One or both evaluations not found"
        )

    # Get evaluation results
    query_results_a = select(EvaluationResult).where(
        EvaluationResult.evaluation_id == evaluation_a_id
    )
    query_results_b = select(EvaluationResult).where(
        EvaluationResult.evaluation_id == evaluation_b_id
    )

    result_results_a = await db.execute(query_results_a)
    result_results_b = await db.execute(query_results_b)

    results_a = result_results_a.scalars().all()
    results_b = result_results_b.scalars().all()

    # Get metric scores for all results
    result_ids_a = [result.id for result in results_a]
    result_ids_b = [result.id for result in results_b]

    query_metrics_a = select(MetricScore).where(MetricScore.result_id.in_(result_ids_a))
    query_metrics_b = select(MetricScore).where(MetricScore.result_id.in_(result_ids_b))

    result_metrics_a = await db.execute(query_metrics_a)
    result_metrics_b = await db.execute(query_metrics_b)

    metrics_a = result_metrics_a.scalars().all()
    metrics_b = result_metrics_b.scalars().all()

    # Organize metrics by name
    metrics_by_name_a = {}
    metrics_by_name_b = {}

    for metric in metrics_a:
        if metric.name not in metrics_by_name_a:
            metrics_by_name_a[metric.name] = []
        metrics_by_name_a[metric.name].append(metric.value)

    for metric in metrics_b:
        if metric.name not in metrics_by_name_b:
            metrics_by_name_b[metric.name] = []
        metrics_by_name_b[metric.name].append(metric.value)

    # Calculate average metrics
    avg_metrics_a = {
        name: sum(values) / len(values) for name, values in metrics_by_name_a.items()
    }
    avg_metrics_b = {
        name: sum(values) / len(values) for name, values in metrics_by_name_b.items()
    }

    # Calculate overall scores
    overall_score_a = sum(result.overall_score or 0 for result in results_a) / len(results_a) if results_a else 0
    overall_score_b = sum(result.overall_score or 0 for result in results_b) / len(results_b) if results_b else 0

    # Calculate differences
    metric_differences = {}

    all_metric_names = set(avg_metrics_a.keys()) | set(avg_metrics_b.keys())

    for name in all_metric_names:
        value_a = avg_metrics_a.get(name, 0)
        value_b = avg_metrics_b.get(name, 0)

        metric_differences[name] = {
            "evaluation_a": value_a,
            "evaluation_b": value_b,
            "difference": value_b - value_a,
            "percent_change": ((value_b - value_a) / value_a * 100) if value_a else float('inf')
        }

    # Determine winner for each metric
    for name, diff in metric_differences.items():
        if diff["difference"] > 0:
            diff["winner"] = "evaluation_b"
        elif diff["difference"] < 0:
            diff["winner"] = "evaluation_a"
        else:
            diff["winner"] = "tie"

    # Count wins
    wins_a = sum(1 for diff in metric_differences.values() if diff["winner"] == "evaluation_a")
    wins_b = sum(1 for diff in metric_differences.values() if diff["winner"] == "evaluation_b")
    ties = sum(1 for diff in metric_differences.values() if diff["winner"] == "tie")

    # Determine overall winner
    if overall_score_a > overall_score_b:
        overall_winner = "evaluation_a"
    elif overall_score_b > overall_score_a:
        overall_winner = "evaluation_b"
    else:
        overall_winner = "tie"

    # Build comparison result
    comparison_result = {
        "evaluation_a": {
            "id": str(evaluation_a.id),
            "name": evaluation_a.name,
            "overall_score": overall_score_a,
            "metrics": avg_metrics_a
        },
        "evaluation_b": {
            "id": str(evaluation_b.id),
            "name": evaluation_b.name,
            "overall_score": overall_score_b,
            "metrics": avg_metrics_b
        },
        "metric_differences": metric_differences,
        "wins": {
            "evaluation_a": wins_a,
            "evaluation_b": wins_b,
            "ties": ties
        },
        "overall_winner": overall_winner,
        "overall_score_difference": overall_score_b - overall_score_a,
        "overall_score_percent_change": (
                    (overall_score_b - overall_score_a) / overall_score_a * 100) if overall_score_a else float('inf')
    }

    return comparison_result