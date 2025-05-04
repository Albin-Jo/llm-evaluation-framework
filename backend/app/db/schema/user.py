from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, EmailStr, ConfigDict

from backend.app.db.models.orm import UserRole


class UserBase(BaseModel):
    """Base schema for User data."""
    email: EmailStr
    display_name: str
    role: UserRole = UserRole.VIEWER
    is_active: bool = True


class UserCreate(UserBase):
    """Schema for creatinguser."""
    external_id: str


class UserUpdate(BaseModel):
    """Schema for updating a user."""
    email: Optional[EmailStr] = None
    display_name: Optional[str] = None
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None


class UserInDB(UserBase):
    """Schema for User data from database."""
    id: UUID
    external_id: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class UserResponse(UserInDB):
    """Schema for User response."""
    pass


class UserWithToken(UserResponse):
    """Schema for User with authentication token."""
    access_token: str
    token_type: str = "bearer"
