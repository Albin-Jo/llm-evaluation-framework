# File: app/schema/dataset_schema.py
from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import UUID

from pydantic import BaseModel, Field

from backend.app.db.models.orm.models import DatasetType


class DatasetBase(BaseModel):
    """Base schema for Dataset data."""
    name: str
    description: Optional[str] = None
    type: DatasetType
    schema: Optional[Dict] = None  # Changed from schema_definition to match DB model
    meta_info: Optional[Dict] = None  # Renamed from metadata to avoid conflict
    version: str = "1.0.0"
    row_count: Optional[int] = None
    is_public: bool = False


class DatasetCreate(DatasetBase):
    """Schema for creating a new Dataset."""
    # This will be filled in by the service layer
    file_path: Optional[str] = None


class DatasetUpdate(BaseModel):
    """Schema for updating a Dataset."""
    name: Optional[str] = None
    description: Optional[str] = None
    type: Optional[DatasetType] = None
    schema: Optional[Dict] = None  # Changed from schema_definition
    meta_info: Optional[Dict] = None
    version: Optional[str] = None
    row_count: Optional[int] = None
    is_public: Optional[bool] = None


class DatasetInDB(DatasetBase):
    """Schema for Dataset data from database."""
    id: UUID
    file_path: str
    owner_id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DatasetResponse(DatasetInDB):
    """Schema for Dataset response."""
    pass


class DatasetUpload(BaseModel):
    """Schema for Dataset file upload."""
    name: str
    description: Optional[str] = None
    type: DatasetType
    file: bytes  # This will contain the file data


class DatasetStatistics(BaseModel):
    """Schema for Dataset statistics."""
    basic_info: Dict[str, Any]
    fields: Dict[str, Dict[str, Any]]
    numerical_stats: Optional[Dict[str, Dict[str, Any]]] = None
    text_stats: Optional[Dict[str, Dict[str, Any]]] = None
    quality_metrics: Dict[str, float]


class DatasetValidationResult(BaseModel):
    """Schema for Dataset validation results."""
    valid: bool
    errors: List[str] = []
    warnings: List[str] = []
    statistics: Dict[str, Any] = {}


class DatasetVersionCreate(BaseModel):
    """Schema for creating a new version of a dataset."""
    version_notes: Optional[str] = None