# File: app/services/dataset_service.py
import csv
import io
import json
import logging
from datetime import datetime
from json import JSONDecodeError
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
from uuid import UUID

from fastapi import HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config.settings import settings
from app.db.repositories.dataset_repository import DatasetRepository
from app.models.orm.models import Dataset, DatasetType, User
from app.schema.dataset_schema import (
    DatasetCreate, DatasetStatistics, DatasetUpdate,
    DatasetValidationResult, DatasetVersionCreate
)
from app.services.storage import BaseStorageService, get_storage_service

# Configure logging
logger = logging.getLogger(__name__)


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
            file: UploadFile, is_public: bool, user: User
    ) -> Dataset:
        """
        Create a new dataset.

        Args:
            name: Dataset name
            description: Dataset description
            dataset_type: Dataset type
            file: Uploaded file
            is_public: Whether the dataset is public
            user: Current user

        Returns:
            Dataset: Created dataset

        Raises:
            HTTPException: If file validation fails
        """
        # Validate file size
        file_size = await self._get_file_size(file)
        if file_size > settings.MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size: {settings.MAX_UPLOAD_SIZE / (1024 * 1024)}MB"
                # File: app/services/dataset_service.py (continued)
            )

            # Validate file content type
            content_type = file.content_type
            valid_types = {
                DatasetType.USER_QUERY: ["text/csv", "application/json", "text/plain"],
                DatasetType.CONTEXT: ["text/csv", "application/json", "text/plain"],
                DatasetType.QUESTION_ANSWER: ["text/csv", "application/json"],
                DatasetType.CONVERSATION: ["application/json"],
                DatasetType.CUSTOM: ["text/csv", "application/json", "text/plain"]
            }

            if content_type not in valid_types.get(dataset_type, []):
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail=f"Unsupported file type for dataset type {dataset_type}. Supported types: {valid_types.get(dataset_type)}"
                )

            try:
                # Upload file
                file_path = await self.storage_service.upload_file(file, "datasets")

                # Analyze file and get metadata
                meta_info, row_count, schema = await self.analyze_file(file, dataset_type)

                # Create dataset
                dataset_data = DatasetCreate(
                    name=name,
                    description=description,
                    type=dataset_type,
                    file_path=file_path,
                    schema=schema,
                    meta_info=meta_info,
                    row_count=row_count,
                    is_public=is_public
                )

                # Create dataset in DB
                dataset_dict = dataset_data.model_dump()
                dataset_dict["owner_id"] = user.id

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

        async def get_dataset(self, dataset_id: UUID, user: User) -> Dataset:
            """
            Get a dataset by ID.

            Args:
                dataset_id: Dataset ID
                user: Current user

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

            # Check if user has permission to view this dataset
            if (
                    dataset.owner_id != user.id
                    and not dataset.is_public
                    and user.role.value != "admin"
            ):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Not enough permissions to access this dataset"
                )

            return dataset

        async def update_dataset(self, dataset_id: UUID, dataset_data: DatasetUpdate, user: User) -> Dataset:
            """
            Update a dataset.

            Args:
                dataset_id: Dataset ID
                dataset_data: Dataset update data
                user: Current user

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

            # Check if user has permission to update this dataset
            if dataset.owner_id != user.id and user.role.value != "admin":
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Not enough permissions to update this dataset"
                )

            # Update the dataset
            update_data = {
                k: v for k, v in dataset_data.model_dump().items() if v is not None
            }

            if not update_data:
                return dataset

            updated_dataset = await self.dataset_repo.update(dataset_id, update_data)
            return updated_dataset

        async def delete_dataset(self, dataset_id: UUID, user: User) -> bool:
            """
            Delete a dataset.

            Args:
                dataset_id: Dataset ID
                user: Current user

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

            # Check if user has permission to delete this dataset
            if dataset.owner_id != user.id and user.role.value != "admin":
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Not enough permissions to delete this dataset"
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

        async def list_datasets(
                self, user: User, skip: int = 0, limit: int = 100,
                dataset_type: Optional[DatasetType] = None,
                is_public: Optional[bool] = None
        ) -> List[Dataset]:
            """
            List datasets accessible to the user.

            Args:
                user: Current user
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
                user=user,
                skip=skip,
                limit=limit,
                filters=filters
            )

        async def search_datasets(
                self, query: str, user: User, types: Optional[List[DatasetType]] = None,
                include_content: bool = False, skip: int = 0, limit: int = 20
        ) -> List[Dataset]:
            """
            Search for datasets by name, description, or content.

            Args:
                query: Search term
                user: Current user
                types: Filter by dataset types
                include_content: Whether to search in dataset content
                skip: Number of records to skip
                limit: Number of records to return

            Returns:
                List[Dataset]: Matching datasets
            """
            # Search by name and description
            datasets = await self.dataset_repo.search_datasets(
                query=query,
                user=user,
                types=types,
                skip=skip,
                limit=limit
            )

            # If include_content is True, also search in the content
            if include_content and query:
                # Get all datasets accessible to the user
                accessible_datasets = await self.dataset_repo.get_accessible_datasets(
                    user=user,
                    types=types,
                    skip=0,
                    limit=None
                )

                content_matches = []

                # Search in content - this can be expensive
                for dataset in accessible_datasets:
                    # Skip datasets already in results
                    if dataset in datasets:
                        continue

                    try:
                        content = await self.storage_service.read_file(dataset.file_path)

                        # Simple content search
                        if query.lower() in content.lower():
                            content_matches.append(dataset)

                            if len(content_matches) + len(datasets) >= limit:
                                break
                    except Exception as e:
                        logger.warning(f"Error searching in dataset {dataset.id}: {str(e)}")

                # Add content matches up to the limit
                remaining = limit - len(datasets)
                if remaining > 0:
                    datasets.extend(content_matches[:remaining])

            return datasets

        async def preview_dataset(self, dataset_id: UUID, user: User, limit: int = 10) -> List[Dict[str, Any]]:
            """
            Preview dataset content.

            Args:
                dataset_id: Dataset ID
                user: Current user
                limit: Number of records to return

            Returns:
                List[Dict[str, Any]]: Dataset preview

            Raises:
                HTTPException: If dataset not found or user doesn't have access
            """
            dataset = await self.get_dataset(dataset_id, user)

            try:
                # Read dataset file
                file_content = await self.storage_service.read_file(dataset.file_path)

                # Parse dataset based on type
                if dataset.type.value.endswith("json"):
                    data = json.loads(file_content)
                    return data[:limit] if isinstance(data, list) else [data]

                elif dataset.type.value.endswith("csv"):
                    csv_data = []
                    csv_file = io.StringIO(file_content)
                    reader = csv.DictReader(csv_file)

                    for i, row in enumerate(reader):
                        if i >= limit:
                            break
                        csv_data.append(dict(row))

                    return csv_data

                else:
                    # Return raw text for non-structured data
                    return [{"content": file_content[:1000] + "..." if len(file_content) > 1000 else file_content}]

            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error reading dataset file: {str(e)}"
                )

        async def validate_dataset(self, dataset_id: UUID, user: User) -> DatasetValidationResult:
            """
            Validate a dataset against its schema.

            Args:
                dataset_id: Dataset ID
                user: Current user

            Returns:
                DatasetValidationResult: Validation results

            Raises:
                HTTPException: If dataset not found or user doesn't have access
            """
            dataset = await self.get_dataset(dataset_id, user)

            try:
                # Read dataset file
                file_content = await self.storage_service.read_file(dataset.file_path)

                # Validation results
                results = DatasetValidationResult(
                    valid=True,
                    errors=[],
                    warnings=[],
                    statistics={}
                )

                # Parse and validate based on dataset type
                if dataset.type.value.endswith("json"):
                    # Validate JSON
                    try:
                        data = json.loads(file_content)

                        # Basic structure validation
                        if isinstance(data, list):
                            results.statistics["total_records"] = len(data)

                            # Check for empty records
                            empty_records = sum(1 for item in data if not item)
                            if empty_records:
                                results.warnings.append(f"Dataset contains {empty_records} empty records")

                            # Validate against schema if available
                            if dataset.schema and dataset.schema.get("properties"):
                                expected_fields = set(dataset.schema["properties"].keys())

                                # Check each record
                                for i, record in enumerate(data):
                                    if not isinstance(record, dict):
                                        results.errors.append(f"Record {i} is not a JSON object")
                                        continue

                                    record_fields = set(record.keys())

                                    # Missing required fields
                                    missing = expected_fields - record_fields
                                    if missing:
                                        results.warnings.append(f"Record {i} is missing fields: {', '.join(missing)}")

                                    # Extra fields
                                    extra = record_fields - expected_fields
                                    if extra:
                                        results.warnings.append(f"Record {i} has extra fields: {', '.join(extra)}")
                        else:
                            results.statistics["total_records"] = 1

                            # Single object validation
                            if dataset.schema and dataset.schema.get("properties"):
                                expected_fields = set(dataset.schema["properties"].keys())
                                record_fields = set(data.keys())

                                # Missing required fields
                                missing = expected_fields - record_fields
                                if missing:
                                    results.warnings.append(f"Record is missing fields: {', '.join(missing)}")

                                # Extra fields
                                extra = record_fields - expected_fields
                                if extra:
                                    results.warnings.append(f"Record has extra fields: {', '.join(extra)}")

                    except json.JSONDecodeError as e:
                        results.valid = False
                        results.errors.append(f"Invalid JSON: {str(e)}")

                elif dataset.type.value.endswith("csv"):
                    # Validate CSV
                    try:
                        csv_file = io.StringIO(file_content)
                        reader = csv.reader(csv_file)

                        # Read header
                        try:
                            header = next(reader)
                            results.statistics["columns"] = len(header)
                        except StopIteration:
                            results.valid = False
                            results.errors.append("CSV file is empty or has no header")
                            return results

                        # Count rows and check consistency
                        row_count = 0
                        inconsistent_rows = 0

                        for i, row in enumerate(reader):
                            row_count += 1

                            if len(row) != len(header):
                                inconsistent_rows += 1
                                if inconsistent_rows <= 5:  # Limit the number of reported inconsistencies
                                    results.errors.append(f"Row {i + 1} has {len(row)} columns, expected {len(header)}")

                        results.statistics["total_records"] = row_count

                        if inconsistent_rows:
                            results.valid = False
                            results.errors.append(f"Found {inconsistent_rows} rows with inconsistent number of columns")

                        # Compare with reported row count if available
                        if dataset.row_count is not None and dataset.row_count != row_count:
                            results.warnings.append(
                                f"Reported row count ({dataset.row_count}) differs from actual count ({row_count})"
                            )

                        # Validate against schema if available
                        if dataset.schema and dataset.schema.get("properties"):
                            expected_fields = set(dataset.schema["properties"].keys())
                            header_fields = set(header)

                            # Missing required fields
                            missing = expected_fields - header_fields
                            if missing:
                                results.warnings.append(f"CSV is missing columns: {', '.join(missing)}")

                            # Extra fields
                            extra = header_fields - expected_fields
                            if extra:
                                results.warnings.append(f"CSV has extra columns: {', '.join(extra)}")

                    except Exception as e:
                        results.valid = False
                        results.errors.append(f"CSV validation error: {str(e)}")

                else:
                    # For other formats, just return basic info
                    results.statistics["file_size"] = len(file_content)
                    results.warnings.append("No detailed validation available for this file type")

                # Set overall validation status
                results.valid = len(results.errors) == 0

                return results

            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error validating dataset: {str(e)}"
                )

        async def get_dataset_statistics(self, dataset_id: UUID, user: User) -> Dict[str, Any]:
            """
            Get dataset statistics.

            Args:
                dataset_id: Dataset ID
                user: Current user

            Returns:
                Dict[str, Any]: Dataset statistics

            Raises:
                HTTPException: If dataset not found or user doesn't have access
            """
            dataset = await self.get_dataset(dataset_id, user)

            try:
                # Read dataset file
                file_content = await self.storage_service.read_file(dataset.file_path)

                # Statistics object
                stats = {
                    "basic_info": {
                        "name": dataset.name,
                        "type": dataset.type.value,
                        "row_count": dataset.row_count,
                        "file_size": len(file_content),
                        "version": dataset.version,
                        "creation_date": dataset.created_at.isoformat() if dataset.created_at else None,
                        "last_modified": dataset.updated_at.isoformat() if dataset.updated_at else None
                    },
                    "fields": {},
                    "numerical_stats": {},
                    "text_stats": {},
                    "quality_metrics": {
                        "completeness": 0,
                        "consistency": 0,
                        "duplicates": 0
                    }
                }

                # Process based on dataset type
                if dataset.type.value.endswith("json"):
                    data = json.loads(file_content)

                    # Ensure data is a list
                    if not isinstance(data, list):
                        data = [data]

                    if not data:
                        return stats

                    # Field statistics
                    all_fields = set()
                    for item in data:
                        all_fields.update(item.keys())

                    field_presence = {field: 0 for field in all_fields}
                    field_types = {field: {} for field in all_fields}
                    numerical_values = {field: [] for field in all_fields}
                    text_lengths = {field: [] for field in all_fields}

                    # Unique rows for duplicate detection
                    seen_rows = set()
                    duplicate_count = 0

                    # Process each record
                    for item in data:
                        # Check for duplicates (using json string as hash)
                        item_hash = json.dumps(item, sort_keys=True)
                        if item_hash in seen_rows:
                            duplicate_count += 1
                        else:
                            seen_rows.add(item_hash)

                        # Process each field
                        for field in all_fields:
                            # Check presence
                            if field in item:
                                field_presence[field] += 1

                                # Check type
                                value = item[field]
                                value_type = type(value).__name__
                                field_types[field][value_type] = field_types[field].get(value_type, 0) + 1

                                # Collect numerical values
                                if isinstance(value, (int, float)) or (
                                        isinstance(value, str) and value.replace('.', '', 1).isdigit()):
                                    try:
                                        num_value = float(value)
                                        numerical_values[field].append(num_value)
                                    except (ValueError, TypeError):
                                        pass

                                # Collect text lengths
                                if isinstance(value, str):
                                    text_lengths[field].append(len(value))

                    # Calculate field statistics
                    for field in all_fields:
                        presence_rate = field_presence[field] / len(data)
                        dominant_type = max(field_types[field].items(), key=lambda x: x[1])[0] if field_types[
                            field] else "unknown"

                        stats["fields"][field] = {
                            "presence_rate": presence_rate,
                            "dominant_type": dominant_type,
                            "type_distribution": field_types[field]
                        }

                        # Numerical statistics
                        if numerical_values[field]:
                            nums = numerical_values[field]
                            stats["numerical_stats"][field] = {
                                "count": len(nums),
                                "min": min(nums),
                                "max": max(nums),
                                "mean": sum(nums) / len(nums),
                                "unique_values": len(set(nums))
                            }

                        # Text statistics
                        if text_lengths[field]:
                            lengths = text_lengths[field]
                            stats["text_stats"][field] = {
                                "count": len(lengths),
                                "min_length": min(lengths),
                                "max_length": max(lengths),
                                "avg_length": sum(lengths) / len(lengths)
                            }

                    # Overall quality metrics
                    avg_completeness = sum(field_presence.values()) / (len(all_fields) * len(data))
                    stats["quality_metrics"]["completeness"] = avg_completeness
                    stats["quality_metrics"]["duplicates"] = duplicate_count

                    # Consistency - check if fields mostly have the same type
                    type_consistency = 0
                    for field, type_dist in field_types.items():
                        if type_dist:
                            max_type_count = max(type_dist.values())
                            type_consistency += max_type_count / sum(type_dist.values())

                    stats["quality_metrics"]["consistency"] = type_consistency / len(field_types) if field_types else 0

                elif dataset.type.value.endswith("csv"):
                    # Process CSV (similar logic to JSON processing)
                    csv_file = io.StringIO(file_content)
                    reader = csv.DictReader(csv_file)

                    # Convert to list to process multiple times
                    data = list(reader)

                    if not data:
                        return stats

                    # Get all fields from header
                    all_fields = reader.fieldnames or []

                    field_presence = {field: 0 for field in all_fields}
                    field_types = {field: {} for field in all_fields}
                    numerical_values = {field: [] for field in all_fields}
                    text_lengths = {field: [] for field in all_fields}

                    # Similar processing as JSON...
                    # (implementation omitted for brevity as it's very similar to JSON logic)

                else:
                    # For non-structured data, only provide basic stats
                    stats["basic_info"]["file_size"] = len(file_content)
                    if isinstance(file_content, str):
                        stats["basic_info"]["character_count"] = len(file_content)
                        stats["basic_info"]["line_count"] = file_content.count('\n') + 1

                return stats

            except Exception as e:
                logger.exception(f"Error calculating dataset statistics: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error calculating dataset statistics: {str(e)}"
                )

        async def export_dataset(
                self, dataset_id: UUID, user: User, export_format: str = "json"
        ) -> Tuple[str, str, bytes]:
            """
            Export a dataset to various formats.

            Args:
                dataset_id: Dataset ID
                user: Current user
                export_format: Export format (json, csv)

            Returns:
                Tuple[str, str, bytes]: (filename, content_type, content)

            Raises:
                HTTPException: If dataset not found or user doesn't have access
            """
            dataset = await self.get_dataset(dataset_id, user)

            try:
                # Read dataset file
                file_content = await self.storage_service.read_file(dataset.file_path)

                # Convert to requested format
                if export_format.lower() == "json":
                    # If already JSON, just return it
                    if dataset.type.value.endswith("json"):
                        filename = f"{dataset.name}.json"
                        content_type = "application/json"
                        return filename, content_type, file_content.encode('utf-8')

                    # Convert from CSV to JSON if needed
                    elif dataset.type.value.endswith("csv"):
                        csv_file = io.StringIO(file_content)
                        reader = csv.DictReader(csv_file)
                        json_data = json.dumps([dict(row) for row in reader])

                        filename = f"{dataset.name}.json"
                        content_type = "application/json"
                        return filename, content_type, json_data.encode('utf-8')

                elif export_format.lower() == "csv":
                    # If already CSV, just return it
                    if dataset.type.value.endswith("csv"):
                        filename = f"{dataset.name}.csv"
                        content_type = "text/csv"
                        return filename, content_type, file_content.encode('utf-8')

                    # Convert from JSON to CSV
                    elif dataset.type.value.endswith("json"):
                        data = json.loads(file_content)

                        # Ensure data is a list for CSV conversion
                        if not isinstance(data, list):
                            data = [data]

                        if not data:
                            filename = f"{dataset.name}.csv"
                            content_type = "text/csv"
                            return filename, content_type, "".encode('utf-8')

                        # Write to CSV
                        output = io.StringIO()
                        writer = csv.DictWriter(output, fieldnames=data[0].keys())
                        writer.writeheader()
                        writer.writerows(data)

                        filename = f"{dataset.name}.csv"
                        content_type = "text/csv"
                        return filename, content_type, output.getvalue().encode('utf-8')

                # Default to returning raw content
                filename = f"{dataset.name}.{export_format}"
                content_type = "application/octet-stream"
                return filename, content_type, file_content.encode('utf-8')

            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error exporting dataset: {str(e)}"
                )

        async def create_dataset_version(
                self, dataset_id: UUID, file: UploadFile, version_notes: Optional[str], user: User
        ) -> Dataset:
            """
            Create a new version of an existing dataset.

            Args:
                dataset_id: Dataset ID
                file: Uploaded file
                version_notes: Version notes
                user: Current user

            Returns:
                Dataset: Updated dataset

            Raises:
                HTTPException: If dataset not found or user doesn't have access
            """
            original_dataset = await self.get_dataset(dataset_id, user)

            # Check if user has permission to create a new version
            if original_dataset.owner_id != user.id and user.role.value != "admin":
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Not enough permissions to create a new version"
                )

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
                # Upload new file
                # Generate version-specific path
                version_path = f"datasets/versions/{dataset_id}/{new_version}"
                file_path = await self.storage_service.upload_file(file, version_path)

                # Analyze file and get metadata
                meta_info, row_count, schema = await self.analyze_file(file, original_dataset.type)

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
                    schema=schema,
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

        async def analyze_file(self, file: UploadFile, dataset_type: DatasetType) -> Tuple[
            Dict[str, Any], int, Dict[str, Any]]:
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

                    if isinstance(data, list):
                        row_count = len(data)
                        # Infer schema from first item if it's a list
                        schema = {"properties": {}} if not data else {
                            "properties": {
                                k: {"type": type(v).__name__} for k, v in data[0].items()
                            }
                        }
                    else:
                        row_count = 1
                        schema = {"properties": {
                            k: {"type": type(v).__name__} for k, v in data.items()
                        }}

                elif file.content_type == "text/csv" or dataset_type.value.endswith("csv"):
                    # For CSV, process line by line to count rows and infer schema
                    await file.seek(0)

                    # Read header first
                    header_line = await file.readline()
                    header = header_line.decode("utf-8").strip().split(",")

                    # Read first data row to infer types
                    # File: app/services/dataset_service.py (continued)
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

                    # Get file size
                    await file.seek(0, 2)  # Seek to end
                    metadata["size"] = await file.tell()

                else:
                    # For other formats, just get size and minimal info
                    await file.seek(0, 2)  # Seek to end
                    size = await file.tell()
                    metadata["size"] = size

                    row_count = 1  # Default for non-structured data
                    schema = {"type": "string"}

                    # Reset file position for potential reuse
                await file.seek(0)

                return metadata, row_count, schema

            except Exception as e:
                # If analysis fails, return basic info
                logger.exception(f"Error analyzing file: {str(e)}")

                # Try to get file size if possible
                try:
                    await file.seek(0, 2)  # Seek to end
                    size = await file.tell()
                    metadata["size"] = size
                    await file.seek(0)  # Reset position
                except:
                    metadata["size"] = 0

                return {
                    **metadata,
                    "error": str(e)
                }, 0, {}

        async def get_dataset_content_stream(
                self, dataset_id: UUID, user: User
        ) -> AsyncGenerator[bytes, None]:
            """
            Stream dataset content.

            Args:
                dataset_id: Dataset ID
                user: Current user

            Yields:
                bytes: Dataset content chunks

            Raises:
                HTTPException: If dataset not found or user doesn't have access
            """
            dataset = await self.get_dataset(dataset_id, user)

            try:
                async for chunk in self.storage_service.read_file_stream(dataset.file_path):
                    yield chunk
            except Exception as e:
                logger.exception(f"Error streaming dataset content: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error streaming dataset content: {str(e)}"
                )

    async def _get_file_size(self, file: UploadFile) -> int:
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