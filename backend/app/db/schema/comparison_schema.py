from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class MetricConfig(BaseModel):
    """Configuration for a metric in a comparison."""
    higher_is_better: bool = Field(True, description="Whether higher values are better for this metric")
    weight: float = Field(1.0, description="Weight of this metric in overall scores")
    description: Optional[str] = Field(None, description="Description of this metric")


class ComparisonBase(BaseModel):
    """Base schema for Comparison data."""
    name: str = Field(..., min_length=1, max_length=255, description="Name of the comparison")
    description: Optional[str] = Field(None, description="Description of the comparison")
    evaluation_a_id: UUID = Field(..., description="ID of the first evaluation to compare")
    evaluation_b_id: UUID = Field(..., description="ID of the second evaluation to compare")
    config: Optional[Dict] = Field(None, description="Configuration options for the comparison")
    created_by_id: Optional[UUID] = Field(None, description="ID of the user who created this comparison")
    # Add metric configurations and weights
    metric_configs: Optional[Dict[str, MetricConfig]] = Field(None, description="Configuration for each metric")


class ComparisonCreate(ComparisonBase):
    """Schema for creating a new Comparison."""
    pass


class ComparisonUpdate(BaseModel):
    """Schema for updating a Comparison."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    config: Optional[Dict] = None
    metric_configs: Optional[Dict[str, MetricConfig]] = None


class ComparisonInDB(ComparisonBase):
    """Schema for Comparison data from database."""
    id: UUID
    comparison_results: Optional[Dict] = None
    summary: Optional[Dict] = None
    status: str
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ComparisonResponse(ComparisonInDB):
    """Schema for Comparison response."""
    narrative_insights: Optional[str] = None


class MetricDifferenceResponse(BaseModel):
    """Schema for individual metric differences."""
    metric_name: str
    evaluation_a_value: float
    evaluation_b_value: float
    absolute_difference: float
    percentage_change: float
    is_improvement: bool
    p_value: Optional[float] = None
    is_significant: Optional[bool] = None
    weight: float = 1.0


class ComparisonDetailResponse(ComparisonResponse):
    """Schema for detailed Comparison response with full details."""
    evaluation_a: Dict
    evaluation_b: Dict
    metric_differences: List[MetricDifferenceResponse] = []
    result_differences: Dict = {}
    summary_stats: Dict = {}
    compatibility_warnings: List[str] = []