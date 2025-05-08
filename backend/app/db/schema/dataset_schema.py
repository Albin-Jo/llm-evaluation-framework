from datetime import datetime
from typing import Dict, Optional, List
from uuid import UUID

from pydantic import BaseModel, Field

from backend.app.db.models.orm import DatasetType


class DatasetBase(BaseModel):
    """Base schema for Dataset data."""
    name: str
    description: Optional[str] = None
    type: DatasetType
    schema_definition: Optional[Dict] = None  # Changed from schema_definition to match DB model
    meta_info: Optional[Dict] = None  # Renamed from metadata to avoid conflict
    version: str = "1.0.0"
    row_count: Optional[int] = None
    is_public: bool = False
    # Added created_by_id for user ownership
    created_by_id: Optional[UUID] = None


class DatasetCreate(DatasetBase):
    """Schema for creating a new Dataset."""
    # This will be filled in by the service layer
    file_path: Optional[str] = None


class DatasetInDB(DatasetBase):
    """Schema for Dataset data from database."""
    id: UUID
    file_path: str
    created_at: datetime
    updated_at: datetime
    # Keep created_by_id from base class

    model_config = {"from_attributes": True}


class DatasetResponse(DatasetInDB):
    """Schema for Dataset response."""
    pass


class DatasetUpdate(BaseModel):
    """Schema for updating a Dataset."""
    name: Optional[str] = None
    description: Optional[str] = None
    type: Optional[DatasetType] = None
    schema_definition: Optional[Dict] = None
    meta_info: Optional[Dict] = None
    version: Optional[str] = None
    row_count: Optional[int] = None
    is_public: Optional[bool] = None
    # Don't allow updating created_by_id after creation


class DatasetSchemaInfo(BaseModel):
    """Schema information for a dataset type."""
    required_fields: List[str] = Field(..., description="Fields that must be present in dataset items")
    optional_fields: List[str] = Field(default_factory=list, description="Fields that may be present in dataset items")
    field_descriptions: Dict[str, str] = Field(default_factory=dict, description="Descriptions of each field")
    description: str = Field("", description="Description of the dataset schema")


class DatasetSchemaResponse(BaseModel):
    """Response for dataset schema information."""
    dataset_type: DatasetType = Field(..., description="Type of the dataset")
    schema: DatasetSchemaInfo = Field(..., description="Schema information for the dataset type")
    supported_metrics: List[str] = Field(default_factory=list,
                                         description="Metrics that can be calculated with this dataset type")