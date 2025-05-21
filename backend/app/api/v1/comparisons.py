import logging
from typing import Dict, List, Optional, Any, Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Body, Query, Path
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.dependencies.auth import get_required_current_user
from backend.app.api.dependencies.rate_limiter import rate_limit
from backend.app.api.middleware.jwt_validator import UserContext
from backend.app.core.exceptions import NotFoundException
from backend.app.db.schema.comparison_schema import (
    ComparisonCreate, ComparisonUpdate, ComparisonResponse, ComparisonDetailResponse,
    MetricDifferenceResponse
)
from backend.app.db.session import get_db
from backend.app.services.comparison_service import ComparisonService
from backend.app.utils.response_utils import create_paginated_response

# Configure logger
logger = logging.getLogger(__name__)

comparisons_router = APIRouter()


@comparisons_router.post("/", response_model=ComparisonResponse)
async def create_comparison(
        comparison_data: ComparisonCreate,
        db: AsyncSession = Depends(get_db),
        current_user: UserContext = Depends(get_required_current_user),
        _: None = Depends(rate_limit(max_requests=10, period_seconds=60))
):
    """
    Create a new evaluation comparison.

    This endpoint creates a new comparison between two evaluations with the specified parameters.

    - **comparison_data**: Required comparison configuration data including metric configs

    Returns the created comparison object with an ID that can be used for future operations.
    """
    logger.info(f"Creating new comparison between evaluation_a_id={comparison_data.evaluation_a_id}, "
                f"evaluation_b_id={comparison_data.evaluation_b_id}")

    # Add the user ID to the comparison data
    if not comparison_data.created_by_id and current_user.db_user:
        comparison_data.created_by_id = current_user.db_user.id

    comparison_service = ComparisonService(db)
    try:
        comparison = await comparison_service.create_comparison(comparison_data)
        logger.info(f"Successfully created comparison id={comparison.id}")
        return comparison
    except Exception as e:
        logger.error(f"Failed to create comparison: {str(e)}", exc_info=True)
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create comparison: {str(e)}"
        )


@comparisons_router.get("/", response_model=Dict[str, Any])
async def list_comparisons(
        skip: Annotated[int, Query(ge=0, description="Number of records to skip")] = 0,
        limit: Annotated[int, Query(ge=1, le=100, description="Maximum number of records to return")] = 10,
        name: Annotated[
            Optional[str], Query(description="Filter by comparison name (case-insensitive, partial match)")] = None,
        sort_by: Annotated[Optional[str], Query(description="Field to sort by")] = "created_at",
        sort_dir: Annotated[Optional[str], Query(description="Sort direction (asc or desc)")] = "desc",
        db: AsyncSession = Depends(get_db),
        current_user: UserContext = Depends(get_required_current_user),
        _: None = Depends(rate_limit(max_requests=50, period_seconds=60))
):
    """
    List comparisons with optional filtering, sorting and pagination.

    This endpoint returns both the comparisons array and pagination information.

    - **skip**: Number of records to skip (for pagination)
    - **limit**: Maximum number of records to return
    - **name**: Optional filter by comparison name (case-insensitive, supports partial matching)
    - **sort_by**: Field to sort results by (default: created_at)
    - **sort_dir**: Sort direction, either "asc" or "desc" (default: desc)

    Returns a dictionary containing the list of comparisons and pagination information.
    """
    filters = {}

    # Add filters if provided
    if name:
        filters["name"] = name

    # Add user ID to filter only user's comparisons
    if current_user.db_user:
        filters["created_by_id"] = current_user.db_user.id

    # Validate sort_by parameter
    valid_sort_fields = ["created_at", "updated_at", "name", "status"]
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

    logger.info(f"Listing comparisons with filters={filters}, skip={skip}, limit={limit}, sort={sort_options}")

    comparison_service = ComparisonService(db)
    try:
        # Get total count first for pagination
        total_count = await comparison_service.comparison_repo.count_search_comparisons(
            query_text=name,
            filters=filters,
            user_id=current_user.db_user.id if current_user.db_user else None
        )

        # Then get the actual page of results
        comparisons = await comparison_service.comparison_repo.search_comparisons(
            query_text=name,
            filters=filters,
            skip=skip,
            limit=limit,
            sort_by=sort_by,
            sort_dir=sort_dir,
            user_id=current_user.db_user.id if current_user.db_user else None
        )

        logger.debug(f"Retrieved {len(comparisons)} comparisons from total of {total_count}")

        # Convert to response format
        comparison_items = [ComparisonResponse.from_orm(comp) for comp in comparisons]

        # Return both results and total count using the utility function
        return create_paginated_response(comparison_items, total_count, skip, limit)

    except Exception as e:
        logger.error(f"Error listing comparisons: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing comparisons: {str(e)}"
        )


@comparisons_router.post("/search", response_model=Dict[str, Any])
async def search_comparisons(
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
    Advanced search for comparisons across multiple fields.

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
    logger.info(f"Advanced search for comparisons with query: '{query}' and filters: {filters}")

    # Initialize filters if not provided
    if filters is None:
        filters = {}

    # Add user ID to filter only user's comparisons
    if current_user.db_user:
        filters["created_by_id"] = current_user.db_user.id

    comparison_service = ComparisonService(db)

    # Get total count
    total_count = await comparison_service.comparison_repo.count_search_comparisons(
        query_text=query,
        filters=filters,
        user_id=current_user.db_user.id if current_user.db_user else None
    )

    # Get comparisons
    comparisons = await comparison_service.comparison_repo.search_comparisons(
        query_text=query,
        filters=filters,
        skip=skip,
        limit=limit,
        sort_by=sort_by,
        sort_dir=sort_dir,
        user_id=current_user.db_user.id if current_user.db_user else None
    )

    # Convert to response format
    comparison_items = [ComparisonResponse.from_orm(comp) for comp in comparisons]

    return create_paginated_response(comparison_items, total_count, skip, limit)


@comparisons_router.get("/{comparison_id}", response_model=ComparisonDetailResponse)
async def get_comparison(
        comparison_id: Annotated[UUID, Path(description="The ID of the comparison to retrieve")],
        db: AsyncSession = Depends(get_db),
        current_user: UserContext = Depends(get_required_current_user)
):
    """
    Get comparison by ID with all related details.

    This endpoint retrieves comprehensive information about a comparison, including:
    - Basic comparison metadata
    - Configuration details
    - Results with metric differences
    - Summary statistics
    - Natural language insights

    - **comparison_id**: The unique identifier of the comparison

    Returns the complete comparison object with all details.
    """
    try:
        # Create comparison service
        comparison_service = ComparisonService(db)

        # Get the comparison with relationships
        comparison = await comparison_service.get_comparison(comparison_id)

        if not comparison:
            raise NotFoundException(resource="Comparison", resource_id=str(comparison_id))

        # Check if the user has permission to access this comparison
        if comparison.created_by_id and comparison.created_by_id != current_user.db_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to access this comparison"
            )

        # Get compatibility warnings if available
        compatibility_warnings = []
        if comparison.comparison_results and "compatibility_warnings" in comparison.comparison_results:
            compatibility_warnings = comparison.comparison_results["compatibility_warnings"]

        # Build the response
        response = ComparisonDetailResponse(
            id=comparison.id,
            name=comparison.name,
            description=comparison.description,
            evaluation_a_id=comparison.evaluation_a_id,
            evaluation_b_id=comparison.evaluation_b_id,
            config=comparison.config,
            metric_configs=comparison.metric_configs,
            comparison_results=comparison.comparison_results,
            summary=comparison.summary,
            status=comparison.status,
            error=comparison.error,
            created_at=comparison.created_at,
            updated_at=comparison.updated_at,
            created_by_id=comparison.created_by_id,
            narrative_insights=comparison.narrative_insights,
            evaluation_a=comparison.evaluation_a.to_dict() if comparison.evaluation_a else None,
            evaluation_b=comparison.evaluation_b.to_dict() if comparison.evaluation_b else None,
            metric_differences=[],
            result_differences={},
            summary_stats={},
            compatibility_warnings=compatibility_warnings
        )

        # Add metric differences if available
        if comparison.comparison_results and "metric_comparison" in comparison.comparison_results:
            metric_diffs = []
            for metric_name, data in comparison.comparison_results["metric_comparison"].items():
                if "comparison" in data and data["comparison"].get("absolute_difference") is not None:
                    metric_diff = MetricDifferenceResponse(
                        metric_name=metric_name,
                        evaluation_a_value=data["evaluation_a"]["average"],
                        evaluation_b_value=data["evaluation_b"]["average"],
                        absolute_difference=data["comparison"]["absolute_difference"],
                        percentage_change=data["comparison"]["percentage_change"],
                        is_improvement=data["comparison"]["is_improvement"],
                        p_value=data["comparison"].get("p_value"),
                        is_significant=data["comparison"].get("is_significant"),
                        weight=data["comparison"].get("weight", 1.0)
                    )
                    metric_diffs.append(metric_diff)

            # Sort by absolute difference
            response.metric_differences = sorted(
                metric_diffs,
                key=lambda x: abs(x.absolute_difference),
                reverse=True
            )

        # Add sample comparison data if available
        if comparison.comparison_results and "sample_comparison" in comparison.comparison_results:
            response.result_differences = comparison.comparison_results["sample_comparison"].get("matched_results", {})
            response.summary_stats = comparison.comparison_results["sample_comparison"].get("stats", {})

        return response

    except NotFoundException:
        raise
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error retrieving comparison details: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving comparison details: {str(e)}"
        )


@comparisons_router.put("/{comparison_id}", response_model=ComparisonResponse)
async def update_comparison(
        comparison_id: Annotated[UUID, Path(description="The ID of the comparison to update")],
        comparison_data: ComparisonUpdate,
        db: AsyncSession = Depends(get_db),
        current_user: UserContext = Depends(get_required_current_user)
):
    """
    Update comparison by ID.

    This endpoint allows updating various properties of an existing comparison.

    - **comparison_id**: The unique identifier of the comparison to update
    - **comparison_data**: The comparison properties to update

    Returns the updated comparison object.
    """
    logger.info(f"Updating comparison id={comparison_id}")

    comparison_service = ComparisonService(db)

    try:
        # Update the comparison with user verification
        updated_comparison = await comparison_service.update_comparison(
            comparison_id,
            comparison_data,
            user_id=current_user.db_user.id if current_user.db_user else None
        )

        if not updated_comparison:
            raise NotFoundException(resource="Comparison", resource_id=str(comparison_id))

        logger.info(f"Successfully updated comparison id={comparison_id}")
        return updated_comparison

    except NotFoundException:
        raise
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating comparison id={comparison_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating comparison: {str(e)}"
        )


@comparisons_router.delete("/{comparison_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_comparison(
        comparison_id: Annotated[UUID, Path(description="The ID of the comparison to delete")],
        db: AsyncSession = Depends(get_db),
        current_user: UserContext = Depends(get_required_current_user)
):
    """
    Delete comparison by ID.

    This endpoint completely removes a comparison and all its associated data.
    This operation cannot be undone.

    - **comparison_id**: The unique identifier of the comparison to delete

    Returns no content on success (HTTP 204).
    """
    logger.info(f"Deleting comparison id={comparison_id}")

    comparison_service = ComparisonService(db)

    try:
        # Delete with user verification
        success = await comparison_service.delete_comparison(
            comparison_id,
            user_id=current_user.db_user.id if current_user.db_user else None
        )

        if not success:
            raise NotFoundException(resource="Comparison", resource_id=str(comparison_id))

        logger.info(f"Successfully deleted comparison id={comparison_id}")
    except NotFoundException:
        raise
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting comparison id={comparison_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting comparison: {str(e)}"
        )


@comparisons_router.post("/{comparison_id}/run", response_model=ComparisonResponse)
async def run_comparison_calculation(
        comparison_id: Annotated[UUID, Path(description="The ID of the comparison to run")],
        db: AsyncSession = Depends(get_db),
        current_user: UserContext = Depends(get_required_current_user),
        _: None = Depends(rate_limit(max_requests=5, period_seconds=60))
):
    """
    Run comparison calculation.

    This endpoint runs or re-runs the comparison calculation, analyzing the differences
    between the two evaluations and storing the results.

    - **comparison_id**: The unique identifier of the comparison to run

    Returns the updated comparison object with calculation results.
    """
    logger.info(f"Running comparison calculation for id={comparison_id}")

    comparison_service = ComparisonService(db)

    try:
        # Run calculation with user verification
        updated_comparison = await comparison_service.run_comparison_calculation(
            comparison_id,
            user_id=current_user.db_user.id if current_user.db_user else None
        )

        logger.info(f"Successfully ran comparison calculation for id={comparison_id}")
        return updated_comparison

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running comparison calculation id={comparison_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error running comparison calculation: {str(e)}"
        )


@comparisons_router.get("/{comparison_id}/metrics", response_model=List[MetricDifferenceResponse])
async def get_comparison_metrics(
        comparison_id: Annotated[UUID, Path(description="The ID of the comparison to retrieve metrics for")],
        db: AsyncSession = Depends(get_db),
        current_user: UserContext = Depends(get_required_current_user)
):
    """
    Get detailed metrics breakdown for a comparison.

    This endpoint retrieves the detailed metric differences between the two evaluations,
    including absolute and percentage changes, statistical significance, and weights.

    - **comparison_id**: The unique identifier of the comparison

    Returns a list of metric differences.
    """
    logger.info(f"Getting metrics for comparison id={comparison_id}")

    comparison_service = ComparisonService(db)

    try:
        # Get metrics with user verification
        metrics = await comparison_service.get_comparison_metrics(
            comparison_id,
            user_id=current_user.db_user.id if current_user.db_user else None
        )

        return metrics

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting comparison metrics id={comparison_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting comparison metrics: {str(e)}"
        )


@comparisons_router.get("/{comparison_id}/report", response_model=Dict[str, Any])
async def get_comparison_report(
        comparison_id: Annotated[UUID, Path(description="The ID of the comparison to generate a report for")],
        format: Annotated[str, Query(description="Report format (json, html, pdf)")] = "json",
        db: AsyncSession = Depends(get_db),
        current_user: UserContext = Depends(get_required_current_user)
):
    """
    Generate a downloadable report for a comparison.

    This endpoint generates a detailed report of the comparison in the specified format.

    - **comparison_id**: The unique identifier of the comparison
    - **format**: Report format (json, html, pdf)

    Returns the comparison report in the requested format.
    """
    logger.info(f"Generating report for comparison id={comparison_id} in {format} format")

    comparison_service = ComparisonService(db)

    try:
        # Generate report with user verification
        report = await comparison_service.generate_comparison_report(
            comparison_id,
            format=format,
            user_id=current_user.db_user.id if current_user.db_user else None
        )

        return report

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating comparison report id={comparison_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating comparison report: {str(e)}"
        )


@comparisons_router.get("/{comparison_id}/visualizations/{visualization_type}", response_model=Dict[str, Any])
async def get_comparison_visualizations(
        comparison_id: Annotated[UUID, Path(description="The ID of the comparison to visualize")],
        visualization_type: Annotated[str, Path(description="Visualization type (radar, bar, line, significance)")],
        db: AsyncSession = Depends(get_db),
        current_user: UserContext = Depends(get_required_current_user)
):
    """
    Get visualization data for charts.

    This endpoint generates data for different types of visualizations to display comparison results.

    - **comparison_id**: The unique identifier of the comparison
    - **visualization_type**: Type of visualization (radar, bar, line, significance)

    Returns data for the requested visualization type.
    """
    logger.info(f"Getting {visualization_type} visualization for comparison id={comparison_id}")

    # Validate visualization type
    valid_types = ["radar", "bar", "line", "significance"]
    if visualization_type not in valid_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid visualization type. Supported types: {', '.join(valid_types)}"
        )

    comparison_service = ComparisonService(db)

    try:
        # Get visualization data with user verification
        visualization_data = await comparison_service.get_comparison_visualizations(
            comparison_id,
            visualization_type=visualization_type,
            user_id=current_user.db_user.id if current_user.db_user else None
        )

        return visualization_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting visualization for comparison id={comparison_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting visualization data: {str(e)}"
        )
