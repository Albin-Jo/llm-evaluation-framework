from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict, field_validator

from backend.app.db.models.orm import IntegrationType, AuthType


class AgentBase(BaseModel):
    """Base schema for Agent data."""
    name: str = Field(..., min_length=1, max_length=255, description="Name of the agent")
    description: Optional[str] = Field(None, description="Description of the agent's purpose and capabilities")
    api_endpoint: str = Field(..., description="API endpoint URL for the agent")
    domain: str = Field(..., min_length=1, max_length=100, description="Domain/category the agent specializes in")
    config: Optional[Dict] = Field(None, description="Configuration options for the agent")
    is_active: bool = Field(True, description="Whether the agent is currently active")
    tags: Optional[List[str]] = Field(None, description="Tags for categorizing the agent")

    # Integration fields
    integration_type: Optional[IntegrationType] = Field(
        IntegrationType.AZURE_OPENAI,
        description="Type of integration for this agent"
    )
    auth_type: Optional[AuthType] = Field(
        AuthType.API_KEY,
        description="Authentication method for this agent"
    )

    @field_validator('api_endpoint')
    @classmethod
    def validate_api_endpoint(cls, v):
        """Validate that the API endpoint is a valid URL."""
        if not v.startswith(('http://', 'https://')):
            raise ValueError('API endpoint must be a valid HTTP or HTTPS URL')
        return v


class AgentCreate(AgentBase):
    """Schema for creating a new Agent."""
    auth_credentials: Optional[Dict[str, Any]] = Field(
        None,
        description="Credentials for authentication (stored securely)"
    )

    @field_validator('auth_credentials')
    @classmethod
    def validate_credentials(cls, v, info):
        """Validate that credentials match auth_type."""
        if not v:
            return v

        # Get auth_type from the data being validated
        auth_type = getattr(info.data, 'auth_type', None) if hasattr(info, 'data') else None

        if not auth_type:
            return v

        if auth_type == AuthType.API_KEY and 'api_key' not in v:
            raise ValueError("API key auth type requires 'api_key' in credentials")

        if auth_type == AuthType.BEARER_TOKEN and 'token' not in v:
            raise ValueError("Bearer token auth type requires 'token' in credentials")

        return v


class AgentUpdate(BaseModel):
    """Schema for updating an Agent."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    api_endpoint: Optional[str] = None
    domain: Optional[str] = Field(None, min_length=1, max_length=100)
    config: Optional[Dict] = None
    is_active: Optional[bool] = None
    tags: Optional[List[str]] = None

    # Integration fields
    integration_type: Optional[IntegrationType] = None
    auth_type: Optional[AuthType] = None
    auth_credentials: Optional[Dict[str, Any]] = None

    @field_validator('api_endpoint')
    @classmethod
    def validate_api_endpoint(cls, v):
        """Validate that the API endpoint is a valid URL if provided."""
        if v is not None and not v.startswith(('http://', 'https://')):
            raise ValueError('API endpoint must be a valid HTTP or HTTPS URL')
        return v

    @field_validator('auth_credentials')
    @classmethod
    def validate_credentials(cls, v, info):
        """Validate that credentials match auth_type."""
        if not v:
            return v

        # Get auth_type from the data being validated
        auth_type = getattr(info.data, 'auth_type', None) if hasattr(info, 'data') else None

        if not auth_type:
            return v

        if auth_type == AuthType.API_KEY and 'api_key' not in v:
            raise ValueError("API key auth type requires 'api_key' in credentials")

        if auth_type == AuthType.BEARER_TOKEN and 'token' not in v:
            raise ValueError("Bearer token auth type requires 'token' in credentials")

        return v


class AgentInDB(BaseModel):
    """Schema for Agent data from database."""
    id: UUID
    name: str
    description: Optional[str] = None
    api_endpoint: str
    domain: str
    config: Optional[Dict] = None
    is_active: bool = True
    tags: Optional[List[str]] = None
    integration_type: Optional[IntegrationType] = IntegrationType.AZURE_OPENAI
    auth_type: Optional[AuthType] = AuthType.API_KEY

    # Credentials in DB are encrypted string, not a dictionary
    auth_credentials: Optional[Union[Dict[str, Any], str]] = None

    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class AgentResponse(AgentInDB):
    """Schema for Agent response."""
    # Optional fields for MCP agents only
    tools: Optional[List[Dict]] = Field(None, description="Available tools (MCP agents only)")
    capabilities: Optional[List[str]] = Field(None, description="Agent capabilities (MCP agents only)")


class AgentTest(BaseModel):
    """Schema for testing an agent."""
    input: Dict = Field(..., description="Input data to send to the agent for testing")


class AgentTestResponse(BaseModel):
    """Schema for the response of an agent test."""
    result: Dict = Field(..., description="Result from the agent")
    processing_time_ms: Optional[int] = Field(None, description="Processing time in milliseconds")
    status: str = Field(..., description="Status of the test (success/error)")