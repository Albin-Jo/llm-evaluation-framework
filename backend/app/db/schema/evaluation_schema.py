# File: app/schema/evaluation_schema.py
from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict

from backend.app.db.models.orm.models import EvaluationMethod, EvaluationStatus


class MetricScoreBase(BaseModel):
    """Base schema for MetricScore data."""
    name: str
    value: float
    weight: float = 1.0
    metadata: Optional[Dict] = None


class MetricScoreCreate(MetricScoreBase):
    """Schema for creating a new MetricScore."""
    pass


class MetricScoreInDB(MetricScoreBase):
    """Schema for MetricScore data from database."""
    id: UUID
    result_id: UUID
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class MetricScoreResponse(MetricScoreInDB):
    """Schema for MetricScore response."""
    pass


class EvaluationResultBase(BaseModel):
    """Base schema for EvaluationResult data."""
    overall_score: Optional[float] = None
    raw_results: Optional[Dict] = None
    dataset_sample_id: Optional[str] = None
    input_data: Optional[Dict] = None
    output_data: Optional[Dict] = None
    processing_time_ms: Optional[int] = None


class EvaluationResultCreate(EvaluationResultBase):
    """Schema for creating a new EvaluationResult."""
    evaluation_id: UUID
    metric_scores: Optional[List[MetricScoreCreate]] = None


class EvaluationResultInDB(EvaluationResultBase):
    """Schema for EvaluationResult data from database."""
    id: UUID
    evaluation_id: UUID
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class EvaluationResultResponse(EvaluationResultInDB):
    """Schema for EvaluationResult response."""
    metric_scores: List[MetricScoreResponse] = []


class EvaluationBase(BaseModel):
    """Base schema for Evaluation data."""
    name: str
    description: Optional[str] = None
    method: EvaluationMethod
    status: EvaluationStatus = EvaluationStatus.PENDING
    config: Optional[Dict] = None
    metrics: Optional[List[str]] = None
    experiment_id: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class EvaluationCreate(EvaluationBase):
    """Schema for creating a new Evaluation."""
    micro_agent_id: UUID
    dataset_id: UUID
    prompt_id: UUID


class EvaluationUpdate(BaseModel):
    """Schema for updating an Evaluation."""
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[EvaluationStatus] = None
    config: Optional[Dict] = None
    metrics: Optional[List[str]] = None
    experiment_id: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class EvaluationInDB(EvaluationBase):
    """Schema for Evaluation data from database."""
    id: UUID
    created_by_id: UUID
    micro_agent_id: UUID
    dataset_id: UUID
    prompt_id: UUID
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class EvaluationResponse(EvaluationInDB):
    """Schema for Evaluation response."""
    pass


class EvaluationDetailResponse(EvaluationResponse):
    """Schema for detailed Evaluation response with results."""
    results: List[EvaluationResultResponse] = []


class EvaluationComparisonBase(BaseModel):
    """Base schema for Evaluation comparison data."""
    name: str
    description: Optional[str] = None
    evaluation_a_id: UUID
    evaluation_b_id: UUID
    comparison_results: Optional[Dict] = None


class EvaluationComparisonCreate(EvaluationComparisonBase):
    """Schema for creating a new Evaluation comparison."""
    pass


class EvaluationComparisonInDB(EvaluationComparisonBase):
    """Schema for Evaluation comparison data from database."""
    id: UUID
    created_by_id: UUID
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class EvaluationComparisonResponse(EvaluationComparisonInDB):
    """Schema for Evaluation comparison response."""
    pass