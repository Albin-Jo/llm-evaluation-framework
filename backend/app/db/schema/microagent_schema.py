# File: app/schema/microagent_schema.py
from datetime import datetime
from typing import Dict, Optional
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict


class MicroAgentBase(BaseModel):
    """Base schema for MicroAgent data."""
    name: str
    description: Optional[str] = None
    api_endpoint: str
    domain: str
    config: Optional[Dict] = None
    is_active: bool = True


class MicroAgentCreate(MicroAgentBase):
    """Schema for creating a new MicroAgent."""
    pass


class MicroAgentUpdate(BaseModel):
    """Schema for updating a MicroAgent."""
    name: Optional[str] = None
    description: Optional[str] = None
    api_endpoint: Optional[str] = None
    domain: Optional[str] = None
    config: Optional[Dict] = None
    is_active: Optional[bool] = None


class MicroAgentInDB(MicroAgentBase):
    """Schema for MicroAgent data from database."""
    id: UUID
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class MicroAgentResponse(MicroAgentInDB):
    """Schema for MicroAgent response."""
    pass