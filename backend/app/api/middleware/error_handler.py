import logging
import traceback
from datetime import datetime
from uuid import uuid4

from fastapi import Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from backend.app.db.schema.response_schema import ErrorResponse, ErrorDetail

logger = logging.getLogger(__name__)


async def error_handler_middleware(request: Request, call_next):
    """
    Global error handler middleware for consistent error responses.
    """
    # Generate request ID
    request_id = str(uuid4())
    request.state.request_id = request_id

    try:
        response = await call_next(request)
        return response
    except Exception as exc:
        return await handle_exception(exc, request_id)


async def handle_exception(exc: Exception, request_id: str) -> JSONResponse:
    """
    Handle different types of exceptions and return standardized error responses.
    """
    timestamp = datetime.utcnow().isoformat()

    if isinstance(exc, HTTPException):
        # FastAPI HTTPException
        error_response = ErrorResponse(
            error="HTTPException",
            message=exc.detail,
            code=f"HTTP_{exc.status_code}",
            request_id=request_id,
            timestamp=timestamp
        )

        logger.warning(
            f"HTTP Exception: {exc.status_code} - {exc.detail}",
            extra={
                "request_id": request_id,
                "status_code": exc.status_code,
                "error_detail": exc.detail
            }
        )

        return JSONResponse(
            status_code=exc.status_code,
            content=error_response.model_dump()
        )

    elif isinstance(exc, RequestValidationError):
        # Pydantic validation error
        error_details = []
        for error in exc.errors():
            field = ".".join(str(loc) for loc in error["loc"])
            error_details.append(ErrorDetail(
                field=field,
                message=error["msg"],
                code=error["type"]
            ))

        error_response = ErrorResponse(
            error="ValidationError",
            message="Request validation failed",
            details=error_details,
            code="VALIDATION_ERROR",
            request_id=request_id,
            timestamp=timestamp
        )

        logger.warning(
            "Validation error",
            extra={
                "request_id": request_id,
                "validation_errors": exc.errors()
            }
        )

        return JSONResponse(
            status_code=422,
            content=error_response.model_dump()
        )

    elif isinstance(exc, SQLAlchemyError):
        # Database errors
        error_response = ErrorResponse(
            error="DatabaseError",
            message="A database error occurred",
            code="DB_ERROR",
            request_id=request_id,
            timestamp=timestamp
        )

        logger.error(
            f"Database error: {str(exc)}",
            extra={
                "request_id": request_id,
                "error_type": type(exc).__name__,
                "error_detail": str(exc)
            },
            exc_info=True
        )

        return JSONResponse(
            status_code=500,
            content=error_response.model_dump()
        )

    else:
        # Unhandled exceptions
        error_response = ErrorResponse(
            error="InternalServerError",
            message="An unexpected error occurred",
            code="INTERNAL_ERROR",
            request_id=request_id,
            timestamp=timestamp
        )

        logger.error(
            f"Unhandled exception: {str(exc)}",
            extra={
                "request_id": request_id,
                "error_type": type(exc).__name__,
                "error_detail": str(exc),
                "traceback": traceback.format_exc()
            }
        )

        return JSONResponse(
            status_code=500,
            content=error_response.model_dump()
        )
