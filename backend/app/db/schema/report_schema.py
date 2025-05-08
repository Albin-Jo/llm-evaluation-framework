# File: backend/app/db/schema/report_schema.py
from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field, ConfigDict

from backend.app.db.models.orm import ReportStatus, ReportFormat


class ReportBase(BaseModel):
    """Base schema for Report data."""
    name: str = Field(..., min_length=1, max_length=255, description="Name of the report")
    description: Optional[str] = Field(None, description="Description of the report")
    format: ReportFormat = Field(default=ReportFormat.PDF, description="Format of the report")
    config: Optional[Dict[str, Any]] = Field(None, description="Configuration options for the report")
    # Added created_by_id for user ownership
    created_by_id: Optional[UUID] = Field(None, description="ID of the user who created this report")


class ReportCreate(ReportBase):
    """Schema for creating a new Report."""
    evaluation_id: UUID = Field(..., description="ID of the evaluation this report is based on")
    include_executive_summary: bool = Field(default=True, description="Include executive summary in the report")
    include_evaluation_details: bool = Field(default=True, description="Include evaluation details in the report")
    include_metrics_overview: bool = Field(default=True, description="Include metrics overview in the report")
    include_detailed_results: bool = Field(default=True, description="Include detailed results in the report")
    include_agent_responses: bool = Field(default=True, description="Include agent responses in the report")
    max_examples: Optional[int] = Field(None, description="Maximum number of examples to include in detailed results")


class ReportUpdate(ReportBase):
    """Schema for updating a Report."""
    status: Optional[ReportStatus] = None


class ReportInDB(ReportBase):
    """Schema for Report data from database."""
    id: UUID
    evaluation_id: UUID
    status: ReportStatus
    file_path: Optional[str] = None
    content: Optional[Dict[str, Any]] = None
    last_sent_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    # Keep created_by_id from base class

    model_config = ConfigDict(from_attributes=True)


class ReportResponse(ReportInDB):
    """Schema for Report response."""
    pass


class ReportDetailResponse(ReportResponse):
    """Schema for detailed Report response with evaluation summary."""
    evaluation_summary: Optional[Dict[str, Any]] = Field(None, description="Summary of the evaluation")


class EmailRecipient(BaseModel):
    """Schema for email recipient."""
    email: EmailStr = Field(..., description="Email address of the recipient")
    name: Optional[str] = Field(None, description="Name of the recipient")


class SendReportRequest(BaseModel):
    """Schema for sending a report via email."""
    recipients: List[EmailRecipient] = Field(..., description="List of recipients")
    subject: Optional[str] = Field(None, description="Email subject")
    message: Optional[str] = Field(None, description="Email message")
    include_pdf: bool = Field(default=True, description="Whether to include the PDF attachment")