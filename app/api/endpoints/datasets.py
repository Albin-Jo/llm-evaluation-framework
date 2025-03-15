# File: app/api/endpoints/datasets.py
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Response, UploadFile, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.orm.models import DatasetType, User
from app.schema.dataset_schema import (
    DatasetResponse, DatasetUpdate,
    DatasetValidationResult
)
from app.services.auth import get_current_active_user
from app.services.dataset_service import DatasetService
from app.services.storage import get_storage_service

router = APIRouter()


@router.post("/", response_model=DatasetResponse)
async def create_dataset(
        name: str = Form(...),
        description: Optional[str] = Form(None),
        type: DatasetType = Form(...),
        file: UploadFile = File(...),
        is_public: bool = Form(False),
        current_user: User = Depends(get_current_active_user),
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
        current_user: Current authenticated user
        db: Database session

    Returns:
        DatasetResponse: Created dataset

    Raises:
        HTTPException: If file is too large, has invalid format, or other errors occur
    """
    dataset_service = DatasetService(db)

    try:
        dataset = await dataset_service.create_dataset(
            name=name,
            description=description,
            dataset_type=type,
            file=file,
            is_public=is_public,
            user=current_user
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
        current_user: User = Depends(get_current_active_user),
        db: AsyncSession = Depends(get_db)
):
    """
    List datasets with optional filtering.

    Args:
        skip: Number of records to skip (for pagination)
        limit: Maximum number of records to return
        type: Filter by dataset type
        is_public: Filter by public/private status
        current_user: Current authenticated user
        db: Database session

    Returns:
        List[DatasetResponse]: List of datasets
    """
    dataset_service = DatasetService(db)

    datasets = await dataset_service.list_datasets(
        user=current_user,
        skip=skip,
        limit=limit,
        dataset_type=type,
        is_public=is_public
    )

    return datasets


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
        dataset_id: UUID,
        current_user: User = Depends(get_current_active_user),
        db: AsyncSession = Depends(get_db)
):
    """
    Get dataset by ID.

    Args:
        dataset_id: Dataset ID
        current_user: Current authenticated user
        db: Database session

    Returns:
        DatasetResponse: Dataset details

    Raises:
        HTTPException: If dataset not found or user doesn't have access
    """
    dataset_service = DatasetService(db)

    try:
        dataset = await dataset_service.get_dataset(dataset_id, current_user)
        return dataset
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching dataset: {str(e)}"
        )


@router.put("/{dataset_id}", response_model=DatasetResponse)
async def update_dataset(
        dataset_id: UUID,
        dataset_data: DatasetUpdate,
        current_user: User = Depends(get_current_active_user),
        db: AsyncSession = Depends(get_db)
):
    """
    Update dataset by ID.

    Args:
        dataset_id: Dataset ID
        dataset_data: Dataset update data
        current_user: Current authenticated user
        db: Database session

    Returns:
        DatasetResponse: Updated dataset

    Raises:
        HTTPException: If dataset not found or user doesn't have permission
    """
    dataset_service = DatasetService(db)

    try:
        updated_dataset = await dataset_service.update_dataset(
            dataset_id, dataset_data, current_user
        )
        return updated_dataset
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating dataset: {str(e)}"
        )


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(
        dataset_id: UUID,
        current_user: User = Depends(get_current_active_user),
        db: AsyncSession = Depends(get_db)
):
    """
    Delete dataset by ID.

    Args:
        dataset_id: Dataset ID
        current_user: Current authenticated user
        db: Database session

    Raises:
        HTTPException: If dataset not found or user doesn't have permission
    """
    dataset_service = DatasetService(db)

    try:
        await dataset_service.delete_dataset(dataset_id, current_user)
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting dataset: {str(e)}"
        )


@router.get("/{dataset_id}/preview", response_model=List[Dict[str, Any]])
async def preview_dataset(
        dataset_id: UUID,
        limit: int = 10,
        current_user: User = Depends(get_current_active_user),
        db: AsyncSession = Depends(get_db)
):
    """
    Preview dataset content.

    Args:
        dataset_id: Dataset ID
        limit: Number of records to return
        current_user: Current authenticated user
        db: Database session

    Returns:
        List[Dict[str, Any]]: Dataset preview

    Raises:
        HTTPException: If dataset not found or user doesn't have access
    """
    dataset_service = DatasetService(db)

    try:
        preview = await dataset_service.preview_dataset(
            dataset_id, current_user, limit
        )
        return preview
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error previewing dataset: {str(e)}"
        )


@router.post("/{dataset_id}/validate", response_model=DatasetValidationResult)
async def validate_dataset(
        dataset_id: UUID,
        current_user: User = Depends(get_current_active_user),
        db: AsyncSession = Depends(get_db)
):
    """
    Validate a dataset against its schema.

    Args:
        dataset_id: Dataset ID
        current_user: Current authenticated user
        db: Database session

    Returns:
        DatasetValidationResult: Validation results

    Raises:
        HTTPException: If dataset not found or user doesn't have access
    """
    dataset_service = DatasetService(db)

    try:
        validation_result = await dataset_service.validate_dataset(
            dataset_id, current_user
        )
        return validation_result
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error validating dataset: {str(e)}"
        )


@router.get("/{dataset_id}/statistics", response_model=Dict[str, Any])
async def get_dataset_statistics(
        dataset_id: UUID,
        current_user: User = Depends(get_current_active_user),
        db: AsyncSession = Depends(get_db)
):
    """
    Get statistical information about a dataset.

    Args:
        dataset_id: Dataset ID
        current_user: Current authenticated user
        db: Database session

    Returns:
        Dict[str, Any]: Dataset statistics

    Raises:
        HTTPException: If dataset not found or user doesn't have access
    """
    dataset_service = DatasetService(db)

    try:
        statistics = await dataset_service.get_dataset_statistics(
            dataset_id, current_user
        )
        return statistics
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error calculating dataset statistics: {str(e)}"
        )


@router.get("/{dataset_id}/export")
async def export_dataset(
        dataset_id: UUID,
        format: str = "json",
        current_user: User = Depends(get_current_active_user),
        db: AsyncSession = Depends(get_db)
):
    """
    Export a dataset to various formats.

    Args:
        dataset_id: Dataset ID
        format: Export format (json, csv)
        current_user: Current authenticated user
        db: Database session

    Returns:
        Response: Exported dataset file

    Raises:
        HTTPException: If dataset not found or user doesn't have access
    """
    dataset_service = DatasetService(db)

    try:
        filename, content_type, content = await dataset_service.export_dataset(
            dataset_id, current_user, format
        )

        return Response(
            content=content,
            media_type=content_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error exporting dataset: {str(e)}"
        )


@router.post("/{dataset_id}/versions", response_model=DatasetResponse)
async def create_dataset_version(
        dataset_id: UUID,
        file: UploadFile = File(...),
        version_notes: Optional[str] = Form(None),
        current_user: User = Depends(get_current_active_user),
        db: AsyncSession = Depends(get_db)
):
    """
    Create a new version of an existing dataset.

    Args:
        dataset_id: Dataset ID
        file: Uploaded file
        version_notes: Notes about this version
        current_user: Current authenticated user
        db: Database session

    Returns:
        DatasetResponse: Updated dataset

    Raises:
        HTTPException: If dataset not found or user doesn't have permission
    """
    dataset_service = DatasetService(db)

    try:
        updated_dataset = await dataset_service.create_dataset_version(
            dataset_id, file, version_notes, current_user
        )
        return updated_dataset
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating dataset version: {str(e)}"
        )


@router.get("/search", response_model=List[DatasetResponse])
async def search_datasets(
        query: str,
        types: Optional[List[DatasetType]] = Query(None),
        include_content: bool = False,
        skip: int = 0,
        limit: int = 20,
        current_user: User = Depends(get_current_active_user),
        db: AsyncSession = Depends(get_db)
):
    """
    Search for datasets by name, description, or content.

    Args:
        query: Search term
        types: Filter by dataset types
        include_content: Whether to search in dataset content
        skip: Number of records to skip (for pagination)
        limit: Maximum number of records to return
        current_user: Current authenticated user
        db: Database session

    Returns:
        List[DatasetResponse]: Matching datasets
    """
    dataset_service = DatasetService(db)

    try:
        datasets = await dataset_service.search_datasets(
            query=query,
            user=current_user,
            types=types,
            include_content=include_content,
            skip=skip,
            limit=limit
        )
        return datasets
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error searching datasets: {str(e)}"
        )


# @router.get("/files/{path:path}")
# async def get_dataset_file(
#         path: str,
#         validated_path: str = Depends(validate_file_access_token),
#         expires: int,
#         db: AsyncSession = Depends(get_db)
# ):
#     """
#     Protected endpoint to access dataset files through a secure token.
#
#     Args:
#         path: File path
#         token: Security token
#         expires: Expiration timestamp
#         db: Database session
#
#     Returns:
#         Response: The requested file if token is valid
#
#     Raises:
#         HTTPException: If token is invalid or expired
#     """
#     # Check if token has expired
#     current_time = int(time.time())
#     if current_time > expires:
#         raise HTTPException(
#             status_code=status.HTTP_403_FORBIDDEN,
#             detail="Access token has expired"
#         )
#
#     # Validate token
#     expected_token = hashlib.sha256(f"{path}:{expires}:{settings.APP_SECRET_KEY}".encode()).hexdigest()
#     if token != expected_token:
#         raise HTTPException(
#             status_code=status.HTTP_403_FORBIDDEN,
#             detail="Invalid access token"
#         )
#
#     # Get storage service
#     storage_service = get_storage_service()
#
#     # Check if file exists
#     if not await storage_service.file_exists(path):
#         raise HTTPException(
#             status_code=status.HTTP_404_NOT_FOUND,
#             detail="File not found"
#         )
#
#     # For local storage, we can use FileResponse
#     from app.services.storage import LocalStorageService
#     if isinstance(storage_service, LocalStorageService):
#         full_path = storage_service._get_full_path(path)
#         return FileResponse(full_path)
#     else:
#         # For other storage types, stream the file
#         try:
#             # Try to determine content type
#             content_type = "application/octet-stream"
#             if path.endswith(".json"):
#                 content_type = "application/json"
#             elif path.endswith(".csv"):
#                 content_type = "text/csv"
#             elif path.endswith(".txt"):
#                 content_type = "text/plain"
#
#             # Stream the file
#             async def file_iterator():
#                 async for chunk in storage_service.read_file_stream(path):
#                     yield chunk
#
#             return StreamingResponse(
#                 file_iterator(),
#                 media_type=content_type,
#                 headers={"Content-Disposition": f"attachment; filename={path.split('/')[-1]}"}
#             )
#         except Exception as e:
#             raise HTTPException(
#                 status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#                 detail=f"Error retrieving file: {str(e)}"
#             )


@router.get("/{dataset_id}/download")
async def download_dataset(
        dataset_id: UUID,
        current_user: User = Depends(get_current_active_user),
        db: AsyncSession = Depends(get_db)
):
    """
    Download a dataset file directly.

    Args:
        dataset_id: Dataset ID
        current_user: Current authenticated user
        db: Database session

    Returns:
        StreamingResponse: Dataset file stream

    Raises:
        HTTPException: If dataset not found or user doesn't have access
    """
    dataset_service = DatasetService(db)

    try:
        # Check if user has access to the dataset
        dataset = await dataset_service.get_dataset(dataset_id, current_user)

        # Get storage service
        storage_service = get_storage_service()

        # Check if file exists
        if not await storage_service.file_exists(dataset.file_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Dataset file not found"
            )

        # Determine content type based on file extension
        content_type = "application/octet-stream"
        if dataset.file_path.endswith(".json"):
            content_type = "application/json"
        elif dataset.file_path.endswith(".csv"):
            content_type = "text/csv"
        elif dataset.file_path.endswith(".txt"):
            content_type = "text/plain"

        # Stream the file
        async def file_iterator():
            async for chunk in dataset_service.get_dataset_content_stream(dataset_id, current_user):
                yield chunk

        filename = f"{dataset.name}{dataset.file_path[dataset.file_path.rfind('.'):]}"

        return StreamingResponse(
            file_iterator(),
            media_type=content_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error downloading dataset: {str(e)}"
        )
