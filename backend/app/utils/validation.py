import re
from typing import Optional

from pydantic import BaseModel, validator, Field


class SanitizedString(str):
    """Custom string type that sanitizes input."""

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not isinstance(v, str):
            raise TypeError('string required')
        # Remove control characters and sanitize
        v = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', v)
        # Trim whitespace
        v = v.strip()
        return cls(v)


class ValidatedQueryParams(BaseModel):
    """Base model for validated query parameters."""
    skip: int = Field(0, ge=0, le=10000, description="Number of records to skip")
    limit: int = Field(100, ge=1, le=1000, description="Maximum number of records to return")
    sort_by: Optional[str] = Field(None, regex=r'^[a-zA-Z_][a-zA-Z0-9_]*$', description="Field to sort by")
    sort_dir: Optional[str] = Field("desc", regex=r'^(asc|desc)$', description="Sort direction")

    @validator('sort_by')
    def validate_sort_field(cls, v, values):
        """Validate sort field against allowed fields."""
        if v is None:
            return v

        # This should be customized per endpoint
        allowed_fields = ['created_at', 'updated_at', 'name', 'status']
        if v not in allowed_fields:
            raise ValueError(f"Invalid sort field. Allowed: {', '.join(allowed_fields)}")
        return v
