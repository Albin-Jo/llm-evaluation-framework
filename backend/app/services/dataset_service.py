import json
import logging
from datetime import datetime
from json import JSONDecodeError
from typing import Optional, Dict, Any, Tuple, List
from uuid import UUID

from fastapi import HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import settings
from backend.app.db.models.base import Base
from backend.app.db.models.orm import Dataset, DatasetType
from backend.app.db.repositories.dataset_repository import DatasetRepository
from backend.app.db.schema.dataset_schema import (
    DatasetCreate, DatasetUpdate
)
from backend.app.db.validators.dataset_validator import (
    validate_dataset_schema
)
from backend.app.evaluation.metrics.ragas_metrics import DATASET_TYPE_METRICS
from backend.app.services.storage import get_storage_service

# Configure logging
logger = logging.getLogger(__name__)


async def _get_file_size(file: UploadFile) -> int:
    """
    Get the size of an uploaded file.

    Args:
        file: Uploaded file

    Returns:
        int: File size in bytes
    """
    file_size = 0
    chunk_size = 1024 * 1024  # 1MB chunks

    # Calculate file size without loading entire file
    await file.seek(0)
    chunk = await file.read(chunk_size)
    while chunk:
        file_size += len(chunk)
        chunk = await file.read(chunk_size)

    # Reset file position
    await file.seek(0)

    return file_size


async def analyze_file(
        file: UploadFile, dataset_type: DatasetType) -> Tuple[Dict[str, Any], int, Dict[str, Any]]:
    """
    Analyze file and get metadata using streaming.

    Args:
        file: Uploaded file
        dataset_type: Dataset type

    Returns:
        tuple: (metadata, row_count, schema)
    """
    # Reset file position
    await file.seek(0)

    # Basic file metadata
    metadata = {
        "filename": file.filename,
        "content_type": file.content_type,
    }

    try:
        # Different processing based on file type
        if file.content_type == "application/json" or dataset_type.value.endswith("json"):
            # For JSON, we still need to read the whole file to parse it
            content = await file.read()
            metadata["size"] = len(content)

            # Parse JSON
            data = json.loads(content.decode("utf-8"))

            # Validate against schema for dataset type
            try:
                validation_result = validate_dataset_schema(content.decode("utf-8"), dataset_type)
                metadata["validation_result"] = validation_result
                metadata["supported_metrics"] = DATASET_TYPE_METRICS.get(dataset_type, [])
                schema = validation_result.get("schema", {})
            except ValueError as e:
                # Log validation error but continue with basic schema inference
                logger.warning(f"Dataset validation failed: {str(e)}")
                metadata["validation_error"] = str(e)

                # Infer basic schema as fallback
                if isinstance(data, list):
                    schema = {"properties": {}} if not data else {
                        "properties": {
                            k: {"type": type(v).__name__} for k, v in data[0].items()
                        }
                    }
                else:
                    schema = {"properties": {
                        k: {"type": type(v).__name__} for k, v in data.items()
                    }}

            # Determine row count
            if isinstance(data, list):
                row_count = len(data)
            else:
                row_count = 1

        elif file.content_type == "text/csv" or dataset_type.value.endswith("csv"):
            # For CSV, process line by line to count rows and infer schema
            await file.seek(0)

            # Read header first
            header_line = await file.readline()
            header = header_line.decode("utf-8").strip().split(",")

            # Read first data row to infer types
            first_row_line = await file.readline()
            if first_row_line:
                first_row = first_row_line.decode("utf-8").strip().split(",")
            else:
                first_row = []

            # Infer schema from header and first row
            schema = {"properties": {}}
            if header and first_row:
                for i, col in enumerate(header):
                    if i < len(first_row):
                        # Try to infer type
                        val = first_row[i]
                        try:
                            int(val)
                            schema["properties"][col] = {"type": "integer"}
                        except ValueError:
                            try:
                                float(val)
                                schema["properties"][col] = {"type": "number"}
                            except ValueError:
                                schema["properties"][col] = {"type": "string"}
                    else:
                        schema["properties"][col] = {"type": "string"}

            # Count rows (we already read 2 lines)
            row_count = 1  # Start with 1 for the data row we've already read

            # Read and count remaining lines
            chunk_size = 8192  # Read ~8KB at a time
            remaining_content = b""

            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break

                remaining_content += chunk
                lines = remaining_content.split(b"\n")

                # The last line might be incomplete, so keep it for the next iteration
                remaining_content = lines.pop() if lines else b""

                row_count += len(lines)

            # Don't forget the last line if it's not empty
            if remaining_content:
                row_count += 1

            # Get file size by reading the whole file
            await file.seek(0)
            content = await file.read()
            metadata["size"] = len(content)

            # For CSV, add metrics that would be supported based on dataset type
            metadata["supported_metrics"] = DATASET_TYPE_METRICS.get(dataset_type, [])

        else:
            # For other formats, read the whole file to get size
            await file.seek(0)
            content = await file.read()
            metadata["size"] = len(content)

            row_count = 1  # Default for non-structured data
            schema = {"type": "string"}

        # Reset file position for potential reuse
        await file.seek(0)

        return metadata, row_count, schema

    except Exception as e:
        # If analysis fails, return basic info
        logger.exception(f"Error analyzing file: {str(e)}")

        # Get file size if possible by reading the file
        try:
            await file.seek(0)
            content = await file.read()
            metadata["size"] = len(content)
            await file.seek(0)  # Reset position
        except:
            metadata["size"] = 0

        return {
            **metadata,
            "error": str(e)
        }, 0, {}


class DatasetService:
    """Service for dataset operations."""

    def __init__(self, db_session: AsyncSession):
        """
        Initialize the dataset service.

        Args:
            db_session: Database session
        """
        self.db_session = db_session
        self.dataset_repo = DatasetRepository(db_session)
        self.storage_service = get_storage_service()

    async def create_dataset(
            self, name: str, description: Optional[str], dataset_type: DatasetType,
            file: UploadFile, is_public: bool, extra_metadata: Optional[Dict[str, Any]] = None
    ) -> Dataset:
        """
        Create a new dataset.

        Args:
            name: Dataset name
            description: Dataset description
            dataset_type: Dataset type
            file: Uploaded file
            is_public: Whether the dataset is public
            extra_metadata: Additional metadata to include

        Returns:
            Dataset: Created dataset

        Raises:
            HTTPException: If file validation fails
        """
        # Validate file size
        file_size = await _get_file_size(file)
        if file_size > settings.MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size: {settings.MAX_UPLOAD_SIZE / (1024 * 1024)}MB"
            )

        # Validate file content type
        content_type = file.content_type
        valid_types = {
            DatasetType.USER_QUERY: ["text/csv", "application/json"],
            DatasetType.CONTEXT: ["text/csv", "application/json"],
            DatasetType.QUESTION_ANSWER: ["text/csv", "application/json"],
            DatasetType.CONVERSATION: ["application/json"],
            DatasetType.CUSTOM: ["text/csv", "application/json"]
        }

        if content_type not in valid_types.get(dataset_type, []):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported file type for dataset "
                       f"type {dataset_type}. Supported types: {valid_types.get(dataset_type)}"
            )

        try:
            environment = settings.APP_ENV
            dir_path = f"{environment}/uploads/datasets/{dataset_type.value}"

            # Upload file
            file_path = await self.storage_service.upload_file(file, dir_path)

            # Analyze file and get metadata
            meta_info, row_count, schema = await analyze_file(file, dataset_type)

            # Add extra metadata if provided
            if extra_metadata:
                meta_info.update(extra_metadata)

            # Add supported metrics if not already present
            if "supported_metrics" not in meta_info:
                meta_info["supported_metrics"] = DATASET_TYPE_METRICS.get(dataset_type, [])

            # Create dataset
            dataset_data = DatasetCreate(
                name=name,
                description=description,
                type=dataset_type,
                file_path=file_path,
                schema_definition=schema,
                meta_info=meta_info,
                row_count=row_count,
                is_public=is_public
            )

            # Create dataset in DB
            dataset_dict = dataset_data.model_dump()
            # dataset_dict["owner_id"] = user.id

            dataset = await self.dataset_repo.create(dataset_dict)
            return dataset

        except JSONDecodeError:
            # Clean up uploaded file if it exists
            if 'file_path' in locals():
                await self.storage_service.delete_file(file_path)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid JSON format in the uploaded file"
            )
        except Exception as e:
            # Clean up uploaded file if it exists
            if 'file_path' in locals():
                await self.storage_service.delete_file(file_path)
            logger.exception(f"Error creating dataset: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error creating dataset: {str(e)}"
            )

    async def list_datasets(
            self, skip: int = 0, limit: int = 100,
            dataset_type: Optional[DatasetType] = None,
            is_public: Optional[bool] = None
    ) -> List[Dataset]:
        """
        List datasets accessible to the user.

        Args:
            skip: Number of records to skip
            limit: Number of records to return
            dataset_type: Filter by dataset type
            is_public: Filter by public/private status

        Returns:
            List[Dataset]: List of datasets
        """
        filters = {}

        # Add filters if provided
        if dataset_type:
            filters["type"] = dataset_type

        if is_public is not None:
            filters["is_public"] = is_public

        return await self.dataset_repo.get_accessible_datasets(
            skip=skip,
            limit=limit,
            filters=filters
        )

    async def get_dataset(self, dataset_id: UUID) -> Dataset:
        """
        Get a dataset by ID.

        Args:
            dataset_id: Dataset ID

        Returns:
            Dataset: Retrieved dataset

        Raises:
            HTTPException: If dataset not found or user doesn't have access
        """
        dataset = await self.dataset_repo.get(dataset_id)

        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset with ID {dataset_id} not found"
            )

        return dataset

    async def delete_dataset(self, dataset_id: UUID) -> bool:
        """
        Delete a dataset.

        Args:
            dataset_id: Dataset ID

        Returns:
            bool: True if deleted successfully

        Raises:
            HTTPException: If dataset not found or user doesn't have permission
        """
        dataset = await self.dataset_repo.get(dataset_id)

        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset with ID {dataset_id} not found"
            )

        # Delete the dataset file
        await self.storage_service.delete_file(dataset.file_path)

        # Delete the dataset from the database
        success = await self.dataset_repo.delete(dataset_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete dataset"
            )

        return True

    async def update_dataset(self, dataset_id: UUID, dataset_data: DatasetUpdate) -> Base | None | Any:
        """
        Update a dataset.

        Args:
            dataset_id: Dataset ID
            dataset_data: Dataset update data

        Returns:
            Dataset: Updated dataset

        Raises:
            HTTPException: If dataset not found or user doesn't have permission
        """
        dataset = await self.dataset_repo.get(dataset_id)

        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset with ID {dataset_id} not found"
            )

        # Update the dataset
        update_data = {
            k: v for k, v in dataset_data.model_dump().items() if v is not None
        }

        if not update_data:
            return dataset

        updated_dataset = await self.dataset_repo.update(dataset_id, update_data)
        return updated_dataset

    async def create_dataset_version(
            self, dataset_id: UUID, file: UploadFile, version_notes: Optional[str],
    ) -> Dataset:
        """
        Create a new version of an existing dataset.

        Args:
            dataset_id: Dataset ID
            file: Uploaded file
            version_notes: Version notes

        Returns:
            Dataset: Updated dataset

        Raises:
            HTTPException: If dataset not found or user doesn't have access
        """
        original_dataset = await self.get_dataset(dataset_id)

        # Increment version
        current_version = original_dataset.version
        try:
            # Parse version (assuming semantic versioning)
            major, minor, patch = map(int, current_version.split('.'))
            new_version = f"{major}.{minor}.{patch + 1}"
        except ValueError:
            # If not semantic versioning, just append .1
            new_version = f"{current_version}.1"

        try:
            # Upload new file using organized directory structure
            # /{environment}/uploads/datasets/{dataset_type}/versions/{dataset_id}/{version}/
            environment = settings.APP_ENV
            version_path = f"{environment}/uploads/datasets/{original_dataset.type.value}/versions/{dataset_id}/{new_version}"

            file_path = await self.storage_service.upload_file(file, version_path)

            # Analyze file and get metadata
            meta_info, row_count, schema = await analyze_file(file, original_dataset.type)

            # Add version metadata
            if not meta_info:
                meta_info = {}

            meta_info["version_history"] = meta_info.get("version_history", [])
            meta_info["version_history"].append({
                "version": current_version,
                "timestamp": datetime.now().isoformat(),
                "notes": version_notes
            })

            # Create new dataset with existing metadata but new file
            dataset_data = DatasetUpdate(
                file_path=file_path,
                schema_definition=schema,
                meta_info=meta_info,
                row_count=row_count,
                version=new_version
            )

            # Update the dataset
            updated_dataset = await self.dataset_repo.update(dataset_id, dataset_data.model_dump(exclude_none=True))

            return updated_dataset

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error creating dataset version: {str(e)}"
            )

    async def generate_file_path(self, name: str, filename: str, dataset_type: DatasetType) -> str:
        """
        Generate a unique file path for a dataset file.

        Args:
            name: Dataset name
            filename: Original filename
            dataset_type: Dataset type

        Returns:
            str: Generated file path
        """
        # Generate a unique ID for the file
        import uuid
        file_id = str(uuid.uuid4())

        # Get file extension
        ext = filename.split('.')[-1] if '.' in filename else 'json'

        # Sanitize dataset name for use in path
        import re
        sanitized_name = re.sub(r'[^\w\-]', '_', name).lower()

        # Create path in format: {environment}/uploads/datasets/{type}/{sanitized_name}_{file_id}.{ext}
        environment = settings.APP_ENV
        return f"{environment}/uploads/datasets/{dataset_type.value}/{sanitized_name}_{file_id}.{ext}"
