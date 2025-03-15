# File: app/schema/prompt_schema.py
from datetime import datetime
from typing import Dict, Optional
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict


class PromptTemplateBase(BaseModel):
    """Base schema for PromptTemplate data."""
    name: str
    description: Optional[str] = None
    template: str
    variables: Optional[Dict] = None
    is_public: bool = False
    version: str = "1.0.0"


class PromptTemplateCreate(PromptTemplateBase):
    """Schema for creating a new PromptTemplate."""
    pass


class PromptTemplateUpdate(BaseModel):
    """Schema for updating a PromptTemplate."""
    name: Optional[str] = None
    description: Optional[str] = None
    template: Optional[str] = None
    variables: Optional[Dict] = None
    is_public: Optional[bool] = None
    version: Optional[str] = None


class PromptTemplateInDB(PromptTemplateBase):
    """Schema for PromptTemplate data from database."""
    id: UUID
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class PromptTemplateResponse(PromptTemplateInDB):
    """Schema for PromptTemplate response."""
    pass


class PromptBase(BaseModel):
    """Base schema for Prompt data."""
    name: str
    description: Optional[str] = None
    content: str
    parameters: Optional[Dict] = None
    version: str = "1.0.0"
    is_public: bool = False


class PromptCreate(PromptBase):
    """Schema for creating a new Prompt."""
    template_id: Optional[UUID] = None


class PromptUpdate(BaseModel):
    """Schema for updating a Prompt."""
    name: Optional[str] = None
    description: Optional[str] = None
    content: Optional[str] = None
    parameters: Optional[Dict] = None
    version: Optional[str] = None
    is_public: Optional[bool] = None
    template_id: Optional[UUID] = None


class PromptInDB(PromptBase):
    """Schema for Prompt data from database."""
    id: UUID
    owner_id: UUID
    template_id: Optional[UUID] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class PromptResponse(PromptInDB):
    """Schema for Prompt response."""
    pass