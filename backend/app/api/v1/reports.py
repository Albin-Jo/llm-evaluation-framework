import logging
from typing import Dict, Optional, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Query, Body, Response
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models.orm import ReportStatus, ReportFormat
from backend.app.db.repositories.report_repository import ReportRepository
from backend.app.db.schema.report_schema import (
    ReportCreate, ReportResponse, ReportUpdate, ReportDetailResponse, SendReportRequest
)
from backend.app.db.session import get_db
from backend.app.services.report_service import ReportService
from backend.app.utils.response_utils import create_paginated_response
from backend.app.api.dependencies.auth import get_required_current_user
from backend.app.api.middleware.jwt_validator import UserContext
from backend.app.api.dependencies.rate_limiter import rate_limit

# Set up logging
logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/", response_model=ReportResponse, status_code=status.HTTP_201_CREATED)
async def create_report(
        report_data: ReportCreate,
        db: AsyncSession = Depends(get_db),
        current_user: UserContext = Depends(get_required_current_user)
):
    """
    Create a new Report.

    Args:
        report_data: The report data to create
        db: Database session
        current_user: Current authenticated user

    Returns:
        The created report

    Raises:
        HTTPException: If validation fails
    """
    logger.info(f"Creating new report for evaluation: {report_data.evaluation_id}")

    # Add the current user to the report data
    if not report_data.created_by_id and current_user.db_user:
        report_data.created_by_id = current_user.db_user.id

    report_service = ReportService(db)

    try:
        report = await report_service.create_report(report_data)
        logger.info(f"Report created successfully: {report.id}")
        return report
    except Exception as e:
        logger.error(f"Failed to create report: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create report: {str(e)}"
            )


@router.get("/", response_model=Dict[str, Any])
async def list_reports(
        skip: int = 0,
        limit: int = 100,
        evaluation_id: Optional[UUID] = Query(None, description="Filter by evaluation ID"),
        status: Optional[ReportStatus] = Query(None, description="Filter by report status"),
        is_public: Optional[bool] = Query(None, description="Filter by public status"),
        name: Optional[str] = Query(None, description="Filter by name (partial match)"),
        db: AsyncSession = Depends(get_db),
        current_user: UserContext = Depends(get_required_current_user)
):
    """
    List Reports with optional filtering and pagination.

    Args:
        skip: Number of records to skip
        limit: Maximum number of records to return
        evaluation_id: Optional filter by evaluation ID
        status: Optional filter by report status
        is_public: Optional filter by public status
        name: Optional filter by name (partial match)
        db: Database session
        current_user: Current authenticated user

    Returns:
        Dict containing list of reports and pagination info
    """
    logger.debug(
        f"Listing reports with filters: evaluation_id={evaluation_id}, status={status}, is_public={is_public}, name={name}")

    report_service = ReportService(db)

    # Get total count
    filters = {}
    if evaluation_id:
        filters["evaluation_id"] = evaluation_id
    if status:
        filters["status"] = status
    if is_public is not None:
        filters["is_public"] = is_public
    if name:
        filters["name"] = name

    # Add user filter to show only user's reports
    if current_user.db_user:
        filters["created_by_id"] = current_user.db_user.id

    total_count = await report_service.report_repo.count(filters)

    reports = await report_service.list_reports(
        skip=skip,
        limit=limit,
        evaluation_id=evaluation_id,
        status=status,
        is_public=is_public,
        name=name,
        user_id=current_user.db_user.id if current_user.db_user else None
    )
    reports_schema_list = [ReportResponse.from_orm(report) for report in reports]
    return create_paginated_response(reports_schema_list, total_count, skip, limit)


@router.get("/{report_id}", response_model=ReportDetailResponse)
async def get_report(
        report_id: UUID,
        db: AsyncSession = Depends(get_db),
        current_user: UserContext = Depends(get_required_current_user)
):
    """
    Get Report by ID.

    Args:
        report_id: The ID of the report to retrieve
        db: Database session
        current_user: Current authenticated user

    Returns:
        The requested report

    Raises:
        HTTPException: If report not found or user doesn't have access
    """
    logger.debug(f"Getting report with ID: {report_id}")

    report_service = ReportService(db)

    try:
        # Get report with user verification
        report = await report_service.get_user_report(
            report_id=report_id,
            user_id=current_user.db_user.id if current_user.db_user else None
        )

        # Add evaluation summary if available
        # In a real implementation, fetch additional data as needed
        if report.content and isinstance(report.content, dict):
            evaluation_summary = {}

            # Extract executive summary from content if available
            for section in report.content.get("sections", []):
                if section.get("title") == "Executive Summary":
                    evaluation_summary = section.get("content", {})
                    break

            # Create response with evaluation summary
            response_dict = report.to_dict()
            response_dict["evaluation_summary"] = evaluation_summary

            return response_dict

        return report
    except Exception as e:
        logger.error(f"Failed to get report: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get report: {str(e)}"
            )


@router.put("/{report_id}", response_model=ReportResponse)
async def update_report(
        report_id: UUID,
        report_data: ReportUpdate,
        db: AsyncSession = Depends(get_db),
        current_user: UserContext = Depends(get_required_current_user)
):
    """
    Update Report by ID.

    Args:
        report_id: The ID of the report to update
        report_data: The updated report data
        db: Database session
        current_user: Current authenticated user

    Returns:
        The updated report

    Raises:
        HTTPException: If report not found, user doesn't have access, or validation fails
    """
    logger.info(f"Updating report with ID: {report_id}")

    report_service = ReportService(db)

    try:
        # Update with user verification
        updated_report = await report_service.update_report(
            report_id=report_id,
            report_data=report_data,
            user_id=current_user.db_user.id if current_user.db_user else None
        )
        logger.info(f"Report updated successfully: {report_id}")
        return updated_report
    except Exception as e:
        logger.error(f"Failed to update report: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to update report: {str(e)}"
            )


@router.delete("/{report_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_report(
        report_id: UUID,
        db: AsyncSession = Depends(get_db),
        current_user: UserContext = Depends(get_required_current_user)
):
    """
    Delete Report by ID.

    Args:
        report_id: The ID of the report to delete
        db: Database session
        current_user: Current authenticated user

    Raises:
        HTTPException: If report not found, user doesn't have access, or deletion fails
    """
    logger.info(f"Deleting report with ID: {report_id}")

    report_service = ReportService(db)

    try:
        # Delete with user verification
        await report_service.delete_report(
            report_id=report_id,
            user_id=current_user.db_user.id if current_user.db_user else None
        )
        logger.info(f"Report deleted successfully: {report_id}")
    except Exception as e:
        logger.error(f"Failed to delete report: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete report: {str(e)}"
            )


@router.post("/{report_id}/generate", response_model=ReportResponse)
async def generate_report(
        report_id: UUID,
        force_regenerate: bool = Body(False, embed=True),
        db: AsyncSession = Depends(get_db),
        current_user: UserContext = Depends(get_required_current_user),
        _: None = Depends(rate_limit(max_requests=10, period_seconds=60))
):
    """
    Generate a report file.

    Args:
        report_id: The ID of the report to generate
        force_regenerate: Whether to force regeneration even if already generated
        db: Database session
        current_user: Current authenticated user

    Returns:
        The updated report

    Raises:
        HTTPException: If report not found, user doesn't have access, or generation fails
    """
    logger.info(f"Generating report file for report ID: {report_id}")

    report_service = ReportService(db)

    try:
        # Generate with user verification
        report = await report_service.generate_report(
            report_id=report_id,
            force_regenerate=force_regenerate,
            user_id=current_user.db_user.id if current_user.db_user else None
        )
        logger.info(f"Report file generated successfully: {report_id}")
        return report
    except Exception as e:
        logger.error(f"Failed to generate report: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate report: {str(e)}"
            )


@router.post("/{report_id}/send", status_code=status.HTTP_200_OK)
async def send_report(
        report_id: UUID,
        send_data: SendReportRequest,
        db: AsyncSession = Depends(get_db),
        current_user: UserContext = Depends(get_required_current_user),
        _: None = Depends(rate_limit(max_requests=5, period_seconds=60))
):
    """
    Send a report via email.

    Args:
        report_id: The ID of the report to send
        send_data: Email sending configuration
        db: Database session
        current_user: Current authenticated user

    Returns:
        Success message

    Raises:
        HTTPException: If report not found, user doesn't have access, or sending fails
    """
    logger.info(f"Sending report {report_id} via email to {len(send_data.recipients)} recipients")

    report_service = ReportService(db)

    try:
        # Send with user verification
        success = await report_service.send_report_email(
            report_id=report_id,
            recipients=send_data.recipients,
            subject=send_data.subject,
            message=send_data.message,
            include_pdf=send_data.include_pdf,
            user_id=current_user.db_user.id if current_user.db_user else None
        )

        if success:
            logger.info(f"Report sent successfully: {report_id}")
            return {"message": "Report sent successfully", "recipients_count": len(send_data.recipients)}
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to send report"
            )
    except Exception as e:
        logger.error(f"Failed to send report: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to send report: {str(e)}"
            )


@router.get("/{report_id}/download")
async def download_report(
        report_id: UUID,
        db: AsyncSession = Depends(get_db),
        current_user: UserContext = Depends(get_required_current_user)
):
    """
    Download a report file.

    Args:
        report_id: The ID of the report to download
        db: Database session
        current_user: Current authenticated user

    Returns:
        The report file for download

    Raises:
        HTTPException: If report not found, user doesn't have access, or file not generated
    """
    logger.info(f"Downloading report with ID: {report_id}")

    report_service = ReportService(db)

    try:
        # Get report with user verification
        report = await report_service.get_user_report(
            report_id=report_id,
            user_id=current_user.db_user.id if current_user.db_user else None
        )

        if report.status != ReportStatus.GENERATED or not report.file_path:
            logger.warning(f"Report {report_id} file is not generated yet")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Report file is not generated yet. Please generate the report first."
            )

        # Get file object
        file = await report_service.get_report_file(report_id)

        # Determine content type based on format
        content_type = "application/pdf"
        if report.format == ReportFormat.HTML:
            content_type = "text/html"
        elif report.format == ReportFormat.JSON:
            content_type = "application/json"

        # Get filename from file path
        filename = report.file_path.split("/")[-1]

        # Create streaming response
        return StreamingResponse(
            file,
            media_type=content_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        logger.error(f"Failed to download report: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to download report: {str(e)}"
            )


@router.get("/{report_id}/preview")
async def preview_report(
        report_id: UUID,
        db: AsyncSession = Depends(get_db),
        current_user: UserContext = Depends(get_required_current_user)
):
    """
    Preview a report in HTML format.

    Args:
        report_id: The ID of the report to preview
        db: Database session
        current_user: Current authenticated user

    Returns:
        HTML preview of the report

    Raises:
        HTTPException: If report not found, user doesn't have access, or preview generation fails
    """
    logger.info(f"Previewing report with ID: {report_id}")

    report_service = ReportService(db)

    try:
        # Get report with user verification
        report = await report_service.get_user_report(
            report_id=report_id,
            user_id=current_user.db_user.id if current_user.db_user else None
        )

        if not report.content:
            # Generate content if not available
            report = await report_service.generate_report(
                report_id=report_id,
                user_id=current_user.db_user.id if current_user.db_user else None
            )

        # Generate HTML preview
        if not report.content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Report content not available"
            )

        html_content = report_service.generate_html_report(report.content)

        # Return HTML response
        return Response(content=html_content, media_type="text/html")
    except Exception as e:
        logger.error(f"Failed to preview report: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to preview report: {str(e)}"
            )


@router.get("/status/counts", response_model=Dict[str, int])
async def get_report_status_counts(
        db: AsyncSession = Depends(get_db),
        current_user: UserContext = Depends(get_required_current_user)
):
    """
    Get counts of reports by status for the current user.

    Args:
        db: Database session
        current_user: Current authenticated user

    Returns:
        Dictionary mapping status to count
    """
    logger.debug("Getting report status counts")

    report_service = ReportService(db)

    try:
        # Count with user filter
        counts = await report_service.count_reports_by_status(
            user_id=current_user.db_user.id if current_user.db_user else None
        )
        return counts
    except Exception as e:
        logger.error(f"Failed to get report status counts: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get report status counts: {str(e)}"
        )