from typing import Any, Dict, Optional
from fastapi import HTTPException, status


class BaseAPIException(HTTPException):
    """Base exception for all API exceptions."""

    def __init__(
            self,
            status_code: int,
            detail: str,
            error_code: Optional[str] = None,
            headers: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(status_code=status_code, detail=detail, headers=headers)
        self.error_code = error_code or f"ERR_{status_code}"


class NotFoundException(BaseAPIException):
    """Exception raised when a requested resource is not found."""

    def __init__(
            self,
            resource: str,
            resource_id: Optional[str] = None,
            detail: Optional[str] = None,
    ):
        if detail is None:
            if resource_id:
                detail = f"{resource} with ID {resource_id} not found"
            else:
                detail = f"{resource} not found"

        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail,
            error_code="RESOURCE_NOT_FOUND"
        )


class DuplicateResourceException(BaseAPIException):
    """Exception raised when attempting to create a duplicate resource."""

    def __init__(
            self,
            resource: str,
            field: str,
            value: str,
            detail: Optional[str] = None,
    ):
        if detail is None:
            detail = f"{resource} with {field} '{value}' already exists"

        super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            detail=detail,
            error_code="DUPLICATE_RESOURCE"
        )


class ValidationException(BaseAPIException):
    """Exception raised for validation errors."""

    def __init__(
            self,
            detail: str,
            field: Optional[str] = None,
    ):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
            error_code="VALIDATION_ERROR"
        )
        self.field = field


class AuthenticationException(BaseAPIException):
    """Exception raised for authentication failures."""

    def __init__(self, detail: str = "Authentication required"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            error_code="AUTHENTICATION_REQUIRED",
            headers={"WWW-Authenticate": "Bearer"},
        )


class AuthorizationException(BaseAPIException):
    """Exception raised for authorization failures."""

    def __init__(self, detail: str = "Insufficient permissions"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
            error_code="INSUFFICIENT_PERMISSIONS"
        )


class ResourceInUseException(BaseAPIException):
    """Exception raised when attempting to delete a resource that is in use."""

    def __init__(
            self,
            resource: str,
            detail: Optional[str] = None,
    ):
        if detail is None:
            detail = f"Cannot delete {resource} because it is currently in use"

        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
            error_code="RESOURCE_IN_USE"
        )


class ServiceUnavailableException(BaseAPIException):
    """Exception raised when an external service is unavailable."""

    def __init__(
            self,
            service: str,
            detail: Optional[str] = None,
    ):
        if detail is None:
            detail = f"{service} is temporarily unavailable"

        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
            error_code="SERVICE_UNAVAILABLE"
        )


class RateLimitExceededException(BaseAPIException):
    """Exception raised when rate limit is exceeded."""

    def __init__(
            self,
            detail: str = "Rate limit exceeded. Please try again later.",
            retry_after: Optional[int] = None,
    ):
        headers = {}
        if retry_after:
            headers["Retry-After"] = str(retry_after)

        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=detail,
            error_code="RATE_LIMIT_EXCEEDED",
            headers=headers
        )


class InvalidStateException(BaseAPIException):
    """Exception raised when an operation is attempted in an invalid state."""

    def __init__(
            self,
            resource: str,
            current_state: str,
            operation: str,
            detail: Optional[str] = None,
    ):
        if detail is None:
            detail = f"Cannot {operation} {resource} in {current_state} state"

        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
            error_code="INVALID_STATE"
        )


class DatabaseException(BaseAPIException):
    """Exception raised for database-related errors."""

    def __init__(
            self,
            detail: str = "A database error occurred",
            original_error: Optional[Exception] = None,
    ):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
            error_code="DATABASE_ERROR"
        )
        self.original_error = original_error


class FileOperationException(BaseAPIException):
    """Exception raised for file operation errors."""

    def __init__(
            self,
            operation: str,
            filename: str,
            detail: Optional[str] = None,
    ):
        if detail is None:
            detail = f"Failed to {operation} file: {filename}"

        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
            error_code="FILE_OPERATION_ERROR"
        )


class ExternalServiceException(BaseAPIException):
    """Exception raised when an external service call fails."""

    def __init__(
            self,
            service: str,
            detail: Optional[str] = None,
            status_code: int = status.HTTP_502_BAD_GATEWAY,
    ):
        if detail is None:
            detail = f"Error communicating with {service}"

        super().__init__(
            status_code=status_code,
            detail=detail,
            error_code="EXTERNAL_SERVICE_ERROR"
        )