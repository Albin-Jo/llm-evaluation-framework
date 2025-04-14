from typing import Optional, List, Dict
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models.orm import DatasetType
from backend.app.db.schema.dataset_schema import (
    DatasetResponse, DatasetUpdate, DatasetSchemaResponse
)
from backend.app.db.session import get_db
from backend.app.db.validators.dataset_validator import (
    get_dataset_schema, validate_dataset_schema
)
from backend.app.evaluation.metrics.ragas_metrics import DATASET_TYPE_METRICS
from backend.app.services.dataset_service import DatasetService

router = APIRouter()


@router.post("/", response_model=DatasetResponse)
async def create_dataset(
        name: str = Form(...),
        description: Optional[str] = Form(None),
        type: DatasetType = Form(...),
        file: UploadFile = File(...),
        is_public: bool = Form(False),
        db: AsyncSession = Depends(get_db)
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
            extra_metadata=extra_meta
        )
        return dataset
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating dataset: {str(e)}"
        )


@router.get("/", response_model=List[DatasetResponse])
async def list_datasets(
        skip: int = 0,
        limit: int = 100,
        type: Optional[DatasetType] = None,
        is_public: Optional[bool] = None,
        db: AsyncSession = Depends(get_db)
):
    """
    List datasets with optional filtering.

    Args:
        skip: Number of records to skip (for pagination)
        limit: Maximum number of records to return
        type: Filter by dataset type
        is_public: Filter by public/private status
        db: Database session

    Returns:
        List[DatasetResponse]: List of datasets
    """
    dataset_service = DatasetService(db)

    datasets = await dataset_service.list_datasets(
        skip=skip,
        limit=limit,
        dataset_type=type,
        is_public=is_public
    )

    return datasets


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
        dataset_id: UUID,
        db: AsyncSession = Depends(get_db)
):
    """
    Get dataset by ID.

    Args:
        dataset_id: Dataset ID
        db: Database session

    Returns:
        DatasetResponse: Dataset details

    Raises:
        HTTPException: If dataset not found or user doesn't have access
    """
    dataset_service = DatasetService(db)

    try:
        dataset = await dataset_service.get_dataset(dataset_id)
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
        db: AsyncSession = Depends(get_db)
):
    """
    Delete dataset by ID.

    Args:
        dataset_id: Dataset ID
        db: Database session

    Raises:
        HTTPException: If dataset not found or user doesn't have permission
    """
    dataset_service = DatasetService(db)

    try:
        await dataset_service.delete_dataset(dataset_id)
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
        db: AsyncSession = Depends(get_db)
):
    """
    Update dataset by ID.

    Args:
        dataset_id: Dataset ID
        dataset_data: Dataset update data
        db: Database session

    Returns:
        DatasetResponse: Updated dataset

    Raises:
        HTTPException: If dataset not found or user doesn't have permission
    """
    dataset_service = DatasetService(db)

    try:
        updated_dataset = await dataset_service.update_dataset(
            dataset_id, dataset_data
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
        dataset_type: DatasetType
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


@router.get("/metrics/{dataset_type}", response_model=Dict[str, List[str]])
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