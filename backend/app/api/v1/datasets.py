from typing import Optional, Dict, Any
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.dependencies.auth import get_required_current_user
from backend.app.api.middleware.jwt_validator import UserContext
from backend.app.db.models.orm import DatasetType
from backend.app.db.schema.dataset_schema import (
    DatasetResponse, DatasetUpdate, DatasetSchemaResponse, MetricsResponse
)
from backend.app.db.session import get_db
from backend.app.db.validators.dataset_validator import (
    get_dataset_schema, validate_dataset_schema
)
from backend.app.evaluation.metrics.ragas_metrics import DATASET_TYPE_METRICS
from backend.app.services.dataset_service import DatasetService
from backend.app.utils.response_utils import create_paginated_response

router = APIRouter()


@router.post("/", response_model=DatasetResponse)
async def create_dataset(
        name: str = Form(...),
        description: Optional[str] = Form(None),
        type: DatasetType = Form(...),
        file: UploadFile = File(...),
        is_public: bool = Form(False),
        db: AsyncSession = Depends(get_db),
        current_user: UserContext = Depends(get_required_current_user)
):
    """
    Create a new dataset with file upload.

    Args:
        name: Dataset name
        description: Dataset description
        type: Dataset type (USER_QUERY, CONTEXT, QUESTION_ANSWER, CONVERSATION, CUSTOM)
        file: Uploaded file (CSV, JSON, or plain text)
        is_public: Whether the dataset is public
        db: Database session
        current_user: Current authenticated user

    Returns:
        DatasetResponse: Created dataset

    Raises:
        HTTPException: If file is too large, has invalid format, or other errors occur
    """
    dataset_service = DatasetService(db)

    try:
        # Validate the file content against the schema
        file_contents = await file.read()
        if isinstance(file_contents, bytes):
            file_contents = file_contents.decode('utf-8')

        # Reset file pointer for the dataset service to use
        await file.seek(0)

        # Validate dataset against schema
        try:
            validation_result = validate_dataset_schema(file_contents, type)
            # Add validation info to dataset creation
            extra_meta = {
                "validation_result": validation_result,
                "supported_metrics": DATASET_TYPE_METRICS.get(type, []),
                "row_count": validation_result.get("count", 0)
            }
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid dataset format: {str(e)}"
            )

        dataset = await dataset_service.create_dataset(
            name=name,
            description=description,
            dataset_type=type,
            file=file,
            is_public=is_public,
            extra_metadata=extra_meta,
            user_id=current_user.db_user.id  # Pass user ID for ownership tracking
        )
        return dataset
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating dataset: {str(e)}"
        )


@router.get("/", response_model=Dict[str, Any])
async def list_datasets(
        skip: int = 0,
        limit: int = 100,
        type: Optional[DatasetType] = None,
        is_public: Optional[bool] = None,
        db: AsyncSession = Depends(get_db),
        current_user: UserContext = Depends(get_required_current_user)
):
    """
    List datasets with optional filtering and pagination.
    Returns both user's own datasets and public datasets.

    Args:
        skip: Number of records to skip (for pagination)
        limit: Maximum number of records to return
        type: Filter by dataset type
        is_public: Filter by public/private status
        db: Database session
        current_user: Current authenticated user

    Returns:
        Dict containing list of datasets and pagination info
    """
    dataset_service = DatasetService(db)

    # Get total count first
    filters = {}
    if type:
        filters["type"] = type
    if is_public is not None:
        filters["is_public"] = is_public

    total_count = await dataset_service.count_accessible_datasets(
        user_id=current_user.db_user.id,
        filters=filters
    )

    datasets = await dataset_service.list_accessible_datasets(
        user_id=current_user.db_user.id,
        skip=skip,
        limit=limit,
        dataset_type=type,
        is_public=is_public
    )
    datasets_schema_list = [DatasetResponse.from_orm(dataset) for dataset in datasets]
    return create_paginated_response(datasets_schema_list, total_count, skip, limit)


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
        dataset_id: UUID,
        db: AsyncSession = Depends(get_db),
        current_user: UserContext = Depends(get_required_current_user)
):
    """
    Get dataset by ID.
    User can access their own datasets or public datasets.

    Args:
        dataset_id: Dataset ID
        db: Database session
        current_user: Current authenticated user

    Returns:
        DatasetResponse: Dataset details

    Raises:
        HTTPException: If dataset not found or user doesn't have access
    """
    dataset_service = DatasetService(db)

    try:
        dataset = await dataset_service.get_accessible_dataset(
            dataset_id=dataset_id,
            user_id=current_user.db_user.id
        )
        return dataset
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching dataset: {str(e)}"
        )


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(
        dataset_id: UUID,
        db: AsyncSession = Depends(get_db),
        current_user: UserContext = Depends(get_required_current_user)
):
    """
    Delete dataset by ID.
    User can only delete their own datasets.

    Args:
        dataset_id: Dataset ID
        db: Database session
        current_user: Current authenticated user

    Raises:
        HTTPException: If dataset not found or user doesn't have permission
    """
    dataset_service = DatasetService(db)

    try:
        await dataset_service.delete_user_dataset(
            dataset_id=dataset_id,
            user_id=current_user.db_user.id
        )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting dataset: {str(e)}"
        )


@router.put("/{dataset_id}", response_model=DatasetResponse)
async def update_dataset(
        dataset_id: UUID,
        dataset_data: DatasetUpdate,
        db: AsyncSession = Depends(get_db),
        current_user: UserContext = Depends(get_required_current_user)
):
    """
    Update dataset by ID.
    User can only update their own datasets.

    Args:
        dataset_id: Dataset ID
        dataset_data: Dataset update data
        db: Database session
        current_user: Current authenticated user

    Returns:
        DatasetResponse: Updated dataset

    Raises:
        HTTPException: If dataset not found or user doesn't have permission
    """
    dataset_service = DatasetService(db)

    try:
        updated_dataset = await dataset_service.update_user_dataset(
            dataset_id=dataset_id,
            dataset_data=dataset_data,
            user_id=current_user.db_user.id
        )
        return updated_dataset
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating dataset: {str(e)}"
        )


@router.get("/schema/{dataset_type}", response_model=DatasetSchemaResponse)
async def get_dataset_type_schema(
        dataset_type: DatasetType,
):
    """
    Get the schema for a specific dataset type.

    This endpoint provides information about required and optional fields,
    as well as which evaluation metrics are supported for this dataset type.

    Args:
        dataset_type: Type of dataset to get schema for

    Returns:
        DatasetSchemaResponse: Schema information including supported metrics
    """
    try:
        schema = get_dataset_schema(dataset_type)
        supported_metrics = DATASET_TYPE_METRICS.get(dataset_type, [])

        return {
            "dataset_type": dataset_type,
            "schema": schema,
            "supported_metrics": supported_metrics
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting dataset schema: {str(e)}"
        )


@router.get("/metrics/{dataset_type}", response_model=MetricsResponse)
async def get_supported_metrics(
        dataset_type: DatasetType
):
    """
    Get supported metrics for a specific dataset type.

    Args:
        dataset_type: Type of dataset to get metrics for

    Returns:
        Dict: Dictionary with dataset type and list of supported metrics
    """
    try:
        print(f"dataset_type: {dataset_type}")
        if dataset_type not in DATASET_TYPE_METRICS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid dataset type: {dataset_type}"
            )

        return {
            "dataset_type": dataset_type,
            "supported_metrics": DATASET_TYPE_METRICS[dataset_type]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting supported metrics: {str(e)}"
        )


@router.get("/{dataset_id}/content", response_model=Dict[str, Any])
async def get_dataset_content(
        dataset_id: UUID,
        limit_rows: Optional[int] = 50,
        db: AsyncSession = Depends(get_db),
        current_user: UserContext = Depends(get_required_current_user)
):
    """
    Get the content of a dataset file with optional row limiting for preview.

    Args:
        dataset_id: Dataset ID
        limit_rows: Maximum number of rows to return (for preview)
        db: Database session
        current_user: Current authenticated user

    Returns:
        Dict with content and metadata

    Raises:
        HTTPException: If dataset not found or user doesn't have access
    """
    dataset_service = DatasetService(db)

    try:
        # Get dataset first to check permissions
        dataset = await dataset_service.get_accessible_dataset(
            dataset_id=dataset_id,
            user_id=current_user.db_user.id
        )

        # Get content preview
        content_preview = await dataset_service.get_dataset_content_preview(
            dataset=dataset,
            limit_rows=limit_rows
        )

        return content_preview
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching dataset content: {str(e)}"
        )
