from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict, field_validator


class AgentBase(BaseModel):
    """Base schema for Agent data."""
    name: str = Field(..., min_length=1, max_length=255, description="Name of the agent")
    description: Optional[str] = Field(None, description="Description of the agent's purpose and capabilities")
    api_endpoint: str = Field(..., description="API endpoint URL for the agent")
    domain: str = Field(..., min_length=1, max_length=100, description="Domain/category the agent specializes in")
    config: Optional[Dict] = Field(None, description="Configuration options for the agent")
    is_active: bool = Field(True, description="Whether the agent is currently active")
    model_type: Optional[str] = Field(None, description="Type of model used by the agent")
    version: Optional[str] = Field("1.0.0", pattern=r"^\d+\.\d+\.\d+$", description="Version of the agent")
    tags: Optional[List[str]] = Field(None, description="Tags for categorizing the agent")

    @field_validator('api_endpoint')
    def validate_api_endpoint(cls, v):
        """Validate that the API endpoint is a valid URL."""
        # Simple validation, could be extended to use HttpUrl type for stricter validation
        if not v.startswith(('http://', 'https://')):
            raise ValueError('API endpoint must be a valid HTTP or HTTPS URL')
        return v


class AgentCreate(AgentBase):
    """Schema for creating a new Agent."""
    pass


class AgentUpdate(BaseModel):
    """Schema for updating an Agent."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    api_endpoint: Optional[str] = None
    domain: Optional[str] = Field(None, min_length=1, max_length=100)
    config: Optional[Dict] = None
    is_active: Optional[bool] = None
    model_type: Optional[str] = None
    version: Optional[str] = Field(None, pattern=r"^\d+\.\d+\.\d+$")
    tags: Optional[List[str]] = None

    @field_validator('api_endpoint')
    def validate_api_endpoint(cls, v):
        """Validate that the API endpoint is a valid URL if provided."""
        if v is not None and not v.startswith(('http://', 'https://')):
            raise ValueError('API endpoint must be a valid HTTP or HTTPS URL')
        return v


class AgentInDB(AgentBase):
    """Schema for Agent data from database."""
    id: UUID
    created_at: datetime
    updated_at: datetime
    # created_by_id: Optional[UUID] = None

    model_config = ConfigDict(from_attributes=True)


class AgentResponse(AgentInDB):
    """Schema for Agent response."""
    pass


class AgentTest(BaseModel):
    """Schema for testing an agent."""
    input: Dict = Field(..., description="Input data to send to the agent for testing")


class AgentTestResponse(BaseModel):
    """Schema for the response of an agent test."""
    result: Dict = Field(..., description="Result from the agent")
    processing_time_ms: Optional[int] = Field(None, description="Processing time in milliseconds")
    status: str = Field(..., description="Status of the test (success/error)")
