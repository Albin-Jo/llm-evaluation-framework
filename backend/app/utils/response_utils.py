from typing import List, Any

from backend.app.db.schema.response_schema import PaginatedResponse


def create_paginated_response(
        items: List[Any],
        total: int,
        skip: int,
        limit: int
) -> PaginatedResponse:
    """Create a standardized paginated response."""
    page = (skip // limit) + 1
    has_next = (skip + limit) < total
    has_previous = skip > 0

    return PaginatedResponse(
        items=items,
        total=total,
        page=page,
        page_size=limit,
        has_next=has_next,
        has_previous=has_previous
    )
