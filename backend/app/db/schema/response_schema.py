from typing import Generic, TypeVar, List, Optional

from pydantic import BaseModel, Field

T = TypeVar('T')


class PaginatedResponse(BaseModel, Generic[T]):
    """Standard response format for paginated list operations."""
    items: List[T] = Field(..., description="List of items")
    total: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of items per page")
    has_next: bool = Field(..., description="Whether there is a next page")
    has_previous: bool = Field(..., description="Whether there is a previous page")


class ErrorResponse(BaseModel):
    """Standard error response format."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[dict] = Field(None, description="Additional error details")
    code: Optional[str] = Field(None, description="Error code for client handling")
    request_id: Optional[str] = Field(None, description="Request tracking ID")
